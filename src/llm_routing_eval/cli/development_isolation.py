"""Fail-closed development/evaluation overlap checks for the offline kNN CLI.

This is an input-boundary policy, not a router or embedding algorithm. It
catches normalized copies and high-confidence near copies; it does not claim
to identify all semantic paraphrases or replace a versioned dataset split.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher

from llm_routing_eval.application import BenchmarkInputError
from llm_routing_eval.cli.configuration import OfflineSettings
from llm_routing_eval.core import BenchmarkCase

# Bounds keep approximate comparison deterministic on untrusted demo inputs.
_MAX_NORMALIZED_CHARS = 4096
_MIN_NEAR_COPY_CHARS = 32
_MIN_NEAR_COPY_TOKENS = 4
_NEAR_COPY_RATIO = 0.94


def _turkish_casefold(value: str) -> str:
    """Apply stable Unicode compatibility and Turkish-aware I/i folding."""
    normalized = unicodedata.normalize("NFC", unicodedata.normalize("NFKC", value))
    folded = normalized.replace("I", "ı").replace("İ", "i").casefold()
    # Casefolded/decomposed dotted-i also appears as "i" plus a combining dot.
    return unicodedata.normalize("NFKC", folded.replace("i\u0307", "i"))


def _canonical_id(value: str) -> str:
    return " ".join(_turkish_casefold(value).split())


def _canonical_text(value: str) -> str:
    """Normalize casing, Unicode, punctuation and whitespace for comparison."""
    folded = _turkish_casefold(value)
    normalized = " ".join("".join(c if c.isalnum() else " " for c in folded).split())
    if not normalized:
        raise BenchmarkInputError("KNN_ISOLATION_EMPTY_NORMALIZED_TEXT")
    if len(normalized) > _MAX_NORMALIZED_CHARS:
        raise BenchmarkInputError("KNN_ISOLATION_INPUT_TOO_LONG")
    return normalized


def _content_fingerprint(normalized: str) -> bytes:
    """Ephemeral comparison key. Neither texts nor digests are reported."""
    return hashlib.sha256(normalized.encode("utf-8")).digest()


def _is_high_confidence_near_copy(left: str, right: str) -> bool:
    """Conservatively detect light character edits, not semantic similarity."""
    if min(len(left), len(right)) < _MIN_NEAR_COPY_CHARS:
        return False
    left_tokens = left.split()
    right_tokens = right.split()
    if min(len(left_tokens), len(right_tokens)) < _MIN_NEAR_COPY_TOKENS:
        return False
    if len(set(left_tokens) & set(right_tokens)) < 3:
        return False

    matcher = SequenceMatcher(None, left, right, autojunk=False)
    return (
        matcher.real_quick_ratio() >= _NEAR_COPY_RATIO
        and matcher.quick_ratio() >= _NEAR_COPY_RATIO
        and matcher.ratio() >= _NEAR_COPY_RATIO
    )


def enforce_development_isolation(
    settings: OfflineSettings,
    cases: Sequence[BenchmarkCase],
) -> None:
    """Prevent development kNN examples from overlapping active test cases.

    Fail before index construction or output writes. Only stable error codes
    reach the CLI; input text is never echoed or embedded in an error.
    """
    case_ids = {_canonical_id(case.case_id) for case in cases}
    case_texts = tuple(_canonical_text(case.input_text) for case in cases)
    case_fingerprints = {_content_fingerprint(text) for text in case_texts}

    for example in settings.routing.knn.examples:
        if _canonical_id(example.example_id) in case_ids:
            raise BenchmarkInputError("KNN_DEVELOPMENT_TEST_LEAKAGE")
        development_text = _canonical_text(example.input_text)
        if _content_fingerprint(development_text) in case_fingerprints:
            raise BenchmarkInputError("KNN_DEVELOPMENT_TEST_LEAKAGE")
        if any(
            _is_high_confidence_near_copy(development_text, case_text)
            for case_text in case_texts
        ):
            raise BenchmarkInputError("KNN_DEVELOPMENT_TEST_LEAKAGE")
