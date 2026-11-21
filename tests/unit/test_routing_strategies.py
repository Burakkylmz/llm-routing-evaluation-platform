"""Member 02 routing tests: deterministic, synthetic and network-free."""

import asyncio

import pytest

from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    RoutingAction,
    RoutingContext,
    RoutingFeedback,
    StrategyId,
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


def case(scenario: str = "account_general", text: str = "Banka hesabı açma") -> BenchmarkCase:
    return BenchmarkCase(
        case_id="synthetic-dev-01",
        input_text=text,
        domain="turkish_banking",
        scenario_type=scenario,
        metadata={"synthetic": True},
    )


MODELS = ModelPair(small="test-small", large="test-large")


def context(stage: int = 0, status: CallStatus | None = None) -> RoutingContext:
    feedback = (
        (RoutingFeedback(attempt_id="attempt1", model="test-small", status=status),)
        if status is not None
        else ()
    )
    return RoutingContext(
        current_stage=stage, remaining_model_calls=2 - stage, prior_attempts=feedback
    )


def test_model_pair_validates_distinct_identifiers() -> None:
    with pytest.raises(ValueError, match="distinct"):
        ModelPair(small="same", large="same")
    with pytest.raises(ValueError, match="nonempty"):
        ModelPair(small="", large="large")


def test_always_large_single_selection_and_finish() -> None:
    router = AlwaysLargeRouter(MODELS)
    initial = asyncio.run(router.decide(case(), context()))
    finished = asyncio.run(router.decide(case(), context(1, CallStatus.SUCCESS)))
    assert initial.strategy is StrategyId.ALWAYS_LARGE
    assert initial.action is RoutingAction.INVOKE_MODEL
    assert initial.selected_model == "test-large"
    assert initial.reason_code == "ALWAYS_LARGE"
    assert finished.action is RoutingAction.FINISH
    assert finished.selected_model is None


def test_rule_priority_default_and_safe_reasons() -> None:
    router = RuleBasedRouter(
        MODELS,
        [
            ScenarioRule(code="FIRST", scenario_types=("account_general",), tier="small"),
            ScenarioRule(code="SECOND", scenario_types=("account_general",), tier="large"),
        ],
    )
    first = asyncio.run(router.decide(case(), context()))
    fallback = asyncio.run(router.decide(case("unknown"), context()))
    finished = asyncio.run(router.decide(case(), context(1, CallStatus.SUCCESS)))
    assert first.selected_model == "test-small"
    assert first.reason_code == "RULE_FIRST"
    assert fallback.selected_model == "test-large"
    assert fallback.reason_code == "RULE_DEFAULT"
    assert finished.action is RoutingAction.FINISH


def test_rule_configuration_rejects_unsafe_and_duplicate_identifiers() -> None:
    with pytest.raises(ValueError, match="safe uppercase"):
        ScenarioRule(code="prompt from user", scenario_types=("x",), tier="small")
    rule = ScenarioRule(code="SIMPLE", scenario_types=("x",), tier="small")
    with pytest.raises(ValueError, match="Duplicate"):
        RuleBasedRouter(MODELS, [rule, rule])


def test_hashing_embedding_is_stable_and_normalized() -> None:
    embedder = HashingEmbedding(dimensions=64)
    first = asyncio.run(embedder.embed(("Banka HESABI",)))
    repeated = asyncio.run(embedder.embed(("banka hesabi",)))
    assert first == repeated
    assert len(first[0]) == 64
    assert sum(component**2 for component in first[0]) == pytest.approx(1)
    with pytest.raises(ValueError, match="lexical tokens"):
        asyncio.run(embedder.embed((" ",)))


def test_knn_uses_only_development_index_and_identical_query() -> None:
    example = DevelopmentExample(example_id="ex1", input_text="kredi kartı", tier="small")
    router = asyncio.run(
        KnnRouter.prepare(MODELS, HashingEmbedding(), [example], minimum_similarity=0.9)
    )
    decision = asyncio.run(router.decide(case(text="kredi kartı"), context()))
    assert decision.selected_model == "test-small"
    assert decision.reason_code == "KNN_SMALL"
    with pytest.raises(ValueError, match="Held-out"):
        DevelopmentExample(
            example_id="bad",
            input_text="held out",
            tier="large",
            split="golden",  # type: ignore[arg-type]
        )


def test_knn_low_similarity_falls_back_to_large() -> None:
    class FixedEmbedder:
        async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            return tuple((1.0, 0.0) if text == "development" else (0.0, 1.0) for text in texts)

    router = asyncio.run(
        KnnRouter.prepare(
            MODELS,
            FixedEmbedder(),
            [DevelopmentExample(example_id="dev", input_text="development", tier="small")],
            minimum_similarity=0.5,
        )
    )
    result = asyncio.run(router.decide(case(text="unseen"), context()))
    assert result.selected_model == "test-large"
    assert result.reason_code == "KNN_LOW_SIMILARITY"


def test_knn_ties_prefer_large_for_quality_safety() -> None:
    class ConstantEmbedder:
        async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            return ((1.0, 0.0),) * len(texts)

    router = asyncio.run(
        KnnRouter.prepare(
            MODELS,
            ConstantEmbedder(),
            [
                DevelopmentExample(example_id="b", input_text="B", tier="small"),
                DevelopmentExample(example_id="a", input_text="A", tier="large"),
            ],
            k=2,
        )
    )
    result = asyncio.run(router.decide(case(), context()))
    assert result.selected_model == "test-large"
    assert result.reason_code == "KNN_LARGE"


def test_knn_rejects_invalid_vectors_duplicate_examples_and_k() -> None:
    class BrokenEmbedder:
        async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            return ((float("nan"), 1.0),) * len(texts)

    dev = DevelopmentExample(example_id="ex", input_text="x", tier="large")
    with pytest.raises(ValueError, match="unique"):
        asyncio.run(KnnRouter.prepare(MODELS, HashingEmbedding(), [dev, dev]))
    with pytest.raises(ValueError, match="Invalid k"):
        asyncio.run(KnnRouter.prepare(MODELS, HashingEmbedding(), [dev], k=2))
    with pytest.raises(ValueError, match="non-finite"):
        asyncio.run(KnnRouter.prepare(MODELS, BrokenEmbedder(), [dev]))


def test_cascade_accepts_small_unless_scenario_requires_escalation() -> None:
    router = CascadeRouter(MODELS, escalate_scenarios=frozenset({"loan_complex"}))
    start = asyncio.run(router.decide(case("loan_complex"), context()))
    escalate = asyncio.run(router.decide(case("loan_complex"), context(1, CallStatus.SUCCESS)))
    accept = asyncio.run(router.decide(case(), context(1, CallStatus.SUCCESS)))
    finish = asyncio.run(router.decide(case(), context(2, CallStatus.SUCCESS)))
    assert start.selected_model == "test-small"
    assert escalate.selected_model == "test-large"
    assert escalate.reason_code == "CASCADE_SCENARIO"
    assert accept.action is RoutingAction.FINISH
    assert finish.action is RoutingAction.FINISH


def test_cascade_failure_escalation_is_configurable() -> None:
    enable = CascadeRouter(MODELS, escalate_on_failure=True)
    disable = CascadeRouter(MODELS, escalate_on_failure=False)
    failover = asyncio.run(enable.decide(case(), context(1, CallStatus.FAILED)))
    blocked = asyncio.run(disable.decide(case(), context(1, CallStatus.TIMEOUT)))
    assert failover.selected_model == "test-large"
    assert failover.reason_code == "CASCADE_FAILOVER"
    assert blocked.action is RoutingAction.FINISH
