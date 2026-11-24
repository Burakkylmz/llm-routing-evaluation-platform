"""Member 02 F8: deterministic RouterPort and EmbeddingPort boundary regressions.

Tests use synthetic, in-memory inputs. No provider, network, secrets or golden data.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    RoutingAction,
    RoutingContext,
    RoutingFeedback,
)
from llm_routing_eval.routing import (
    AlwaysLargeRouter,
    CascadeRouter,
    DevelopmentExample,
    HashingEmbedding,
    KnnRouter,
    ModelPair,
    RuleBasedRouter,
    ScenarioRule,
)

MODELS = ModelPair(small="offline-small", large="offline-large")
EXAMPLE = DevelopmentExample(example_id="dev-001", input_text="synthetic development", tier="small")


def _case(scenario: str = "account_general") -> BenchmarkCase:
    return BenchmarkCase(
        case_id="heldout-001",
        input_text="different synthetic evaluation text",
        domain="turkish_banking",
        scenario_type=scenario,
        metadata={"synthetic": True},
    )


def _context(
    *,
    stage: int = 0,
    attempt_status: CallStatus | None = None,
) -> RoutingContext:
    attempts = (
        (RoutingFeedback(attempt_id="attempt-001", model=MODELS.small, status=attempt_status),)
        if attempt_status is not None
        else ()
    )
    return RoutingContext(
        current_stage=stage,
        remaining_model_calls=2,
        prior_attempts=attempts,
    )


class ScriptedEmbedding:
    """Minimal EmbeddingPort fake with predetermined index/query responses."""

    def __init__(self, *responses: tuple[tuple[float, ...], ...]) -> None:
        self._responses = iter(responses)

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        assert texts
        return next(self._responses)


@pytest.mark.parametrize("k", [0, 2, True])
def test_knn_rejects_invalid_neighbour_counts(k: int) -> None:
    with pytest.raises(ValueError, match="Invalid k"):
        asyncio.run(
            KnnRouter.prepare(
                MODELS,
                ScriptedEmbedding(),
                (EXAMPLE,),
                k=k,
            )
        )


@pytest.mark.parametrize(
    "threshold",
    [-1.01, 1.01, float("nan"), float("inf"), float("-inf")],
)
def test_knn_rejects_invalid_similarity_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="Similarity threshold"):
        asyncio.run(
            KnnRouter.prepare(
                MODELS,
                ScriptedEmbedding(),
                (EXAMPLE,),
                minimum_similarity=threshold,
            )
        )


@pytest.mark.parametrize("example_count", [0, 2])
def test_knn_rejects_missing_and_duplicate_development_examples(example_count: int) -> None:
    examples = () if example_count == 0 else (EXAMPLE, EXAMPLE)
    with pytest.raises(ValueError, match="unique IDs"):
        asyncio.run(KnnRouter.prepare(MODELS, ScriptedEmbedding(), examples))


@pytest.mark.parametrize("count", [0, 2])
def test_knn_rejects_incorrect_index_embedding_counts(count: int) -> None:
    indexed = ((1.0, 0.0),) * count
    with pytest.raises(ValueError, match="incorrect count"):
        asyncio.run(KnnRouter.prepare(MODELS, ScriptedEmbedding(indexed), (EXAMPLE,)))


@pytest.mark.parametrize(
    ("vector", "reason"),
    [
        ((), "Missing or inconsistent"),
        ((0.0, 0.0), "all-zero"),
        ((math.inf, 1.0), "non-finite"),
        ((math.nan, 1.0), "non-finite"),
    ],
)
def test_knn_rejects_unsafe_development_vectors(vector: tuple[float, ...], reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        asyncio.run(
            KnnRouter.prepare(
                MODELS,
                ScriptedEmbedding((vector,)),
                (EXAMPLE,),
            )
        )


def test_knn_rejects_inconsistent_index_vector_dimensions() -> None:
    second = DevelopmentExample(
        example_id="dev-002", input_text="different development", tier="large"
    )
    with pytest.raises(ValueError, match="inconsistent embedding dimension"):
        asyncio.run(
            KnnRouter.prepare(
                MODELS,
                ScriptedEmbedding(((1.0, 0.0), (1.0, 0.0, 0.0))),
                (EXAMPLE, second),
            )
        )


@pytest.mark.parametrize("count", [0, 2])
def test_knn_rejects_incorrect_query_embedding_count(count: int) -> None:
    embedder = ScriptedEmbedding(((1.0, 0.0),), ((1.0, 0.0),) * count)
    router = asyncio.run(KnnRouter.prepare(MODELS, embedder, (EXAMPLE,)))
    with pytest.raises(ValueError, match="incorrect query count"):
        asyncio.run(router.decide(_case(), _context()))


@pytest.mark.parametrize(
    ("vector", "reason"),
    [
        ((), "Missing or inconsistent"),
        ((0.0, 0.0), "all-zero"),
        ((math.nan, 1.0), "non-finite"),
        ((math.inf, 1.0), "non-finite"),
        ((1.0, 0.0, 0.0), "inconsistent embedding dimension"),
    ],
)
def test_knn_rejects_unsafe_query_vectors(vector: tuple[float, ...], reason: str) -> None:
    embedder = ScriptedEmbedding(((1.0, 0.0),), (vector,))
    router = asyncio.run(KnnRouter.prepare(MODELS, embedder, (EXAMPLE,)))
    with pytest.raises(ValueError, match=reason):
        asyncio.run(router.decide(_case(), _context()))


def test_knn_equal_similarity_neighbour_order_uses_stable_ids() -> None:
    small = DevelopmentExample(example_id="z-small", input_text="a", tier="small")
    large = DevelopmentExample(example_id="a-large", input_text="b", tier="large")
    embedder = ScriptedEmbedding(
        ((1.0, 0.0), (1.0, 0.0)),
        ((1.0, 0.0),),
    )
    router = asyncio.run(KnnRouter.prepare(MODELS, embedder, (small, large), k=1))
    selected = asyncio.run(router.decide(_case(), _context()))
    assert selected.selected_model == MODELS.large
    assert selected.reason_code == "KNN_LARGE"


def test_knn_does_not_reembed_after_a_completed_attempt() -> None:
    embedder = ScriptedEmbedding(((1.0, 0.0),))
    router = asyncio.run(KnnRouter.prepare(MODELS, embedder, (EXAMPLE,)))
    decision = asyncio.run(
        router.decide(_case(), _context(stage=1, attempt_status=CallStatus.SUCCESS))
    )
    assert decision.action is RoutingAction.FINISH
    assert decision.reason_code == "BASELINE_DONE"


@pytest.mark.parametrize("dimensions", [15, 4097])
def test_hashing_embedding_rejects_out_of_range_dimensions(dimensions: int) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        HashingEmbedding(dimensions=dimensions)


@pytest.mark.parametrize("dimensions", [16, 4096])
def test_hashing_embedding_accepts_documented_dimension_limits(dimensions: int) -> None:
    embedder = HashingEmbedding(dimensions=dimensions)
    (vector,) = asyncio.run(embedder.embed(("synthetic text",)))
    assert len(vector) == dimensions
    assert sum(x * x for x in vector) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("stage", "attempt_status", "reason"),
    [
        (0, CallStatus.SUCCESS, "CASCADE_DONE"),
        (1, None, "CASCADE_DONE"),
        (2, CallStatus.SUCCESS, "CASCADE_DONE"),
    ],
)
def test_cascade_finishes_in_inconsistent_or_terminal_contexts(
    stage: int, attempt_status: CallStatus | None, reason: str
) -> None:
    router = CascadeRouter(MODELS)
    decision = asyncio.run(
        router.decide(_case(), _context(stage=stage, attempt_status=attempt_status))
    )
    assert decision.action is RoutingAction.FINISH
    assert decision.reason_code == reason


def test_rule_based_stable_first_match_after_overlapping_rules() -> None:
    router = RuleBasedRouter(
        MODELS,
        (
            ScenarioRule(code="FIRST", scenario_types=("account_general",), tier="small"),
            ScenarioRule(code="SECOND", scenario_types=("account_general",), tier="large"),
        ),
    )
    decision = asyncio.run(router.decide(_case(), _context()))
    assert decision.selected_model == MODELS.small
    assert decision.reason_code == "RULE_FIRST"


def test_always_large_stops_after_previous_provider_failure() -> None:
    router = AlwaysLargeRouter(MODELS)
    decision = asyncio.run(
        router.decide(_case(), _context(stage=1, attempt_status=CallStatus.FAILED))
    )
    assert decision.action is RoutingAction.FINISH
    assert decision.selected_model is None
    assert decision.reason_code == "BASELINE_DONE"
