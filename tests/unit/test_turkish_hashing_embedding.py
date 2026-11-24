"""Turkish lexical hashing regressions for review #12 F6 (fully offline)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from llm_routing_eval.cli.configuration import read_settings, routing_policy_digest
from llm_routing_eval.core import BenchmarkCase, RoutingContext, StrategyId
from llm_routing_eval.routing import DevelopmentExample, HashingEmbedding, KnnRouter, ModelPair
from llm_routing_eval.routing.embeddings import (
    HASHING_EMBEDDING_VERSION,
    _normalize_lexical_text,
)

CONFIG = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("İŞLEM ÜCRETİ", "işlem ücreti"),
        ("IŞIK HIZI", "ışık hızı"),
        ("İSTANBUL", "istanbul"),
        ("I\u0307STANBUL", "istanbul"),
        ("Istanbul", "ıstanbul"),
        ("HESAP", "hesap"),
        ("KREDİ KARTI", "kredi kartı"),
    ],
)
def test_turkish_orthographic_case_variants_have_identical_vectors(
    first: str, second: str
) -> None:
    embedder = HashingEmbedding(dimensions=256)

    first_vector, second_vector = asyncio.run(embedder.embed((first, second)))

    assert first_vector == second_vector
    assert sum(component**2 for component in first_vector) == pytest.approx(1)


def test_turkish_dotless_i_is_not_conflated_with_dotted_i() -> None:
    assert _normalize_lexical_text("I") == "ı"
    assert _normalize_lexical_text("İ") == "i"
    assert _normalize_lexical_text("I") != _normalize_lexical_text("İ")


def test_turkish_hashes_are_stable_across_separate_calls() -> None:
    embedder = HashingEmbedding(dimensions=64)
    phrase = "İstanbul'da IŞIK ve işlem ücretleri"

    first = asyncio.run(embedder.embed((phrase,)))
    repeated = asyncio.run(embedder.embed((phrase,)))

    assert first == repeated


def test_knn_routes_equivalent_turkish_case_variants_consistently() -> None:
    models = ModelPair(small="offline-small", large="offline-large")
    embedder = HashingEmbedding(dimensions=256)
    development = DevelopmentExample(
        example_id="dev-case-001",
        input_text="IŞIK HIZI VE BİLGİ İŞLEM",
        tier="small",
    )
    router = asyncio.run(
        KnnRouter.prepare(models, embedder, (development,), minimum_similarity=0.99)
    )
    evaluation = BenchmarkCase(
        case_id="distinct-synthetic-test-001",
        input_text="ışık hızı ve bilgi işlem",
        domain="turkish_banking",
        scenario_type="account_general",
        metadata={"synthetic": True},
    )
    context = RoutingContext(current_stage=0, remaining_model_calls=2)

    decision = asyncio.run(router.decide(evaluation, context))

    assert decision.selected_model == "offline-small"
    assert decision.reason_code == "KNN_SMALL"


def test_knn_fingerprint_records_updated_embedding_version() -> None:
    settings = read_settings(CONFIG)
    current = routing_policy_digest(settings, StrategyId.KNN)
    legacy_payload = {
        "fingerprint_schema": "offline-routing-policy-v1",
        "strategy": StrategyId.KNN.value,
        "models": settings.routing.models.model_dump(mode="json"),
        "selected_policy": {
            "knn": settings.routing.knn.model_dump(mode="json"),
            "embedding_implementation": "sha256-lexical-feature-hash-v1",
        },
    }
    legacy_canonical = json.dumps(
        legacy_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    old_digest = "sha256:" + hashlib.sha256(legacy_canonical).hexdigest()

    assert HASHING_EMBEDDING_VERSION == "sha256-lexical-feature-hash-tr-casefold-v2"
    assert current != old_digest
    assert current == routing_policy_digest(settings, StrategyId.KNN)


@pytest.mark.parametrize(
    "strategy",
    [StrategyId.ALWAYS_LARGE, StrategyId.RULE_BASED, StrategyId.CASCADE],
)
def test_other_strategy_fingerprints_remain_stable(strategy: StrategyId) -> None:
    settings = read_settings(CONFIG)
    assert routing_policy_digest(settings, strategy) == routing_policy_digest(
        read_settings(CONFIG), strategy
    )
