"""CLI composition for development-only routing calibration (offline).

Reads the Member 03 hand-off JSON, applies a calibrated kNN/Cascade policy to
``OfflineSettings`` and produces a text-free evidence report. The calibrated
settings are validated by the same strict loader, F4 isolation guard and router
factory used by ``run`` before anything is written.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llm_routing_eval.application import BenchmarkInputError
from llm_routing_eval.cli.configuration import (
    CascadeSettings,
    DevelopmentExampleSettings,
    KnnSettings,
    OfflineSettings,
)
from llm_routing_eval.cli.development_isolation import enforce_development_isolation
from llm_routing_eval.core import BenchmarkCase
from llm_routing_eval.routing.calibration import (
    MAX_CALIBRATION_EXAMPLES,
    CalibrationExample,
    CascadeCalibration,
    KnnCalibration,
    KnnGridPoint,
)

_MAX_INPUT_BYTES = 16_777_216


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class CalibrationRecord(_Frozen):
    example_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_./-]+$")
    input_text: str = Field(min_length=1, max_length=4096, repr=False)
    scenario_type: str = Field(min_length=1, max_length=100)
    tier: Literal["small", "large"]
    split: Literal["development"] = "development"
    # Mirrors DevelopmentExampleSettings: the offline demo only indexes synthetic data.
    synthetic: Literal[True]


class CalibrationDataset(_Frozen):
    """Member 03 hand-off format. Any non-development split is rejected."""

    schema_version: Literal["1.0.0"]
    dataset_version: str = Field(min_length=1, max_length=100)
    split: Literal["development"]
    examples: tuple[CalibrationRecord, ...] = Field(
        min_length=2, max_length=MAX_CALIBRATION_EXAMPLES
    )

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if len({ex.example_id for ex in self.examples}) != len(self.examples):
            raise ValueError("DUPLICATE_EXAMPLE_ID")
        return self

    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return "sha256:" + hashlib.sha256(("calibration-v1\n" + canonical).encode()).hexdigest()

    def calibration_examples(self) -> tuple[CalibrationExample, ...]:
        return tuple(
            CalibrationExample(
                example_id=ex.example_id,
                input_text=ex.input_text,
                scenario_type=ex.scenario_type,
                tier=ex.tier,
            )
            for ex in self.examples
        )

    def development_settings(self) -> tuple[DevelopmentExampleSettings, ...]:
        """Build the exact labeled examples used by a calibrated kNN index."""
        return tuple(
            DevelopmentExampleSettings(
                example_id=ex.example_id,
                input_text=ex.input_text,
                tier=ex.tier,
                split="development",
                synthetic=True,
            )
            for ex in self.examples
        )


def enforce_calibration_isolation(
    settings: OfflineSettings,
    dataset: CalibrationDataset,
    cases: Sequence[BenchmarkCase],
) -> None:
    """Fail closed before embedding, scoring or producing an output artifact.

    Validate overlap using the exact development examples that will later be
    written to the calibrated configuration. This check is required even if
    no viable kNN hyperparameter can be selected.
    """
    knn = settings.routing.knn.model_copy(update={"examples": dataset.development_settings()})
    routing = settings.routing.model_copy(update={"knn": knn})
    candidate = settings.model_copy(update={"routing": routing})
    enforce_development_isolation(candidate, cases)


def read_calibration_dataset(path: Path) -> CalibrationDataset:
    try:
        if not path.is_file() or path.stat().st_size > _MAX_INPUT_BYTES:
            raise BenchmarkInputError("CALIBRATION_DATA_MISSING_OR_TOO_LARGE")
        return CalibrationDataset.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        if isinstance(exc, BenchmarkInputError):
            raise
        raise BenchmarkInputError("CALIBRATION_DATA_INVALID") from exc


def apply_calibration(
    settings: OfflineSettings,
    dataset: CalibrationDataset,
    knn: KnnCalibration,
    cascade: CascadeCalibration,
) -> OfflineSettings:
    """Return new validated settings; an infeasible kNN calibration is never applied."""
    if knn.selected is None:
        raise BenchmarkInputError("KNN_CALIBRATION_INFEASIBLE")
    knn_settings = KnnSettings(
        k=knn.selected.k,
        minimum_similarity=knn.selected.minimum_similarity,
        embedding_dimensions=settings.routing.knn.embedding_dimensions,
        examples=dataset.development_settings(),
    )
    cascade_settings = CascadeSettings(
        escalate_scenarios=cascade.escalate_scenarios,
        escalate_on_failure=settings.routing.cascade.escalate_on_failure,
    )
    routing = settings.routing.model_copy(update={"knn": knn_settings, "cascade": cascade_settings})
    # Round-trip through the strict model so the result is exactly what read_settings accepts.
    return OfflineSettings.model_validate(
        settings.model_copy(update={"routing": routing}).model_dump(mode="json")
    )


def _point(p: KnnGridPoint) -> dict[str, object]:
    return {
        "k": p.k,
        "minimum_similarity": p.minimum_similarity,
        "accuracy": round(p.accuracy, 6),
        "under_routing_rate": round(p.under_routing_rate, 6),
        "correct": p.correct,
        "under_routed": p.under_routed,
        "over_routed": p.over_routed,
        "small_selected": p.small_selected,
        "total": p.total,
    }


def calibration_report(
    dataset: CalibrationDataset, knn: KnnCalibration, cascade: CascadeCalibration
) -> dict[str, object]:
    """Reviewable evidence with IDs/counts only; never example text."""
    return {
        "schema_version": "1.0.0",
        "method": "leave-one-out-development-only",
        "dataset_version": dataset.dataset_version,
        "dataset_fingerprint": dataset.fingerprint(),
        "example_count": len(dataset.examples),
        "knn": {
            "feasible": knn.feasible,
            "max_under_routing_rate": knn.max_under_routing_rate,
            "selected": _point(knn.selected) if knn.selected is not None else None,
            "grid": [_point(p) for p in knn.grid],
        },
        "cascade": {
            "max_small_risk": cascade.max_small_risk,
            "min_scenario_examples": cascade.min_scenario_examples,
            "escalate_scenarios": list(cascade.escalate_scenarios),
            "scenarios": [
                {
                    "scenario_type": s.scenario_type,
                    "examples": s.examples,
                    "large_labels": s.large_labels,
                    "escalate": s.escalate,
                    "reason": s.reason,
                }
                for s in cascade.scenarios
            ],
        },
        "limitations": [
            "Development-only labels; not a held-out quality result.",
            "Labels encode the smallest sufficient tier as judged by the data owner.",
            "Lexical hashing embedding; not validated semantic retrieval.",
        ],
    }


def dump_settings(settings: OfflineSettings) -> str:
    header = (
        "# Generated by `llm-routing-eval calibrate-routing` from development-only data.\n"
        "# Review before use; this file NEVER enables real LLM providers.\n"
    )
    body = yaml.safe_dump(
        settings.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=100
    )
    return header + body


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
