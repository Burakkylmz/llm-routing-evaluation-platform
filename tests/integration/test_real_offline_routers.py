"""Real Member 02 routers through Member 01 orchestration, fake provider and privacy."""

import asyncio

import pytest

from llm_routing_eval.application import BenchmarkOrchestrator, RunOptions
from llm_routing_eval.core import BenchmarkCase, ExecutionStatus, StrategyId, TraceStepKind
from llm_routing_eval.demo.fakes import (
    AllowlistedDemoPrivacy,
    DemoFakeProvider,
    DemoFixtureEvaluator,
    InMemoryDemoRepository,
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

MODELS = ModelPair(small="local-small", large="local-large")
PROMPT = "Banka hesabı açmak için genel süreç nedir?"


def synthetic_case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="m02-offline-case",
        input_text=PROMPT,
        domain="turkish_banking",
        scenario_type="account_general",
        metadata={"synthetic": True},
    )


@pytest.mark.parametrize(
    ("strategy", "expected_models"),
    [
        (StrategyId.ALWAYS_LARGE, ["local-large"]),
        (StrategyId.RULE_BASED, ["local-small"]),
        (StrategyId.KNN, ["local-small"]),
        (StrategyId.CASCADE, ["local-small", "local-large"]),
    ],
)
def test_all_four_strategies_share_orchestrator_contract(
    strategy: StrategyId, expected_models: list[str]
) -> None:
    async def execute() -> tuple[ExecutionStatus, list[str], bool]:
        if strategy is StrategyId.ALWAYS_LARGE:
            router = AlwaysLargeRouter(MODELS)
        elif strategy is StrategyId.RULE_BASED:
            router = RuleBasedRouter(
                MODELS,
                [ScenarioRule(code="ACCOUNT", scenario_types=("account_general",), tier="small")],
            )
        elif strategy is StrategyId.KNN:
            router = await KnnRouter.prepare(
                MODELS,
                HashingEmbedding(),
                [DevelopmentExample(example_id="dev-1", input_text=PROMPT, tier="small")],
            )
        else:
            router = CascadeRouter(MODELS, escalate_scenarios=frozenset({"account_general"}))
        engine = BenchmarkOrchestrator(
            router=router,
            provider=DemoFakeProvider(),
            evaluator=DemoFixtureEvaluator(),
            privacy=AllowlistedDemoPrivacy(),
            repository=InMemoryDemoRepository(),
        )
        run = await engine.run(
            [synthetic_case()],
            RunOptions(
                strategy=strategy,
                provider="fake",
                allowed_models=("local-small", "local-large"),
                max_model_calls_per_case=2,
            ),
        )
        trace = run.traces[0]
        attempts = [
            step.model for step in trace.steps if step.kind is TraceStepKind.PROVIDER_ATTEMPT
        ]
        return (
            trace.status,
            [model for model in attempts if model is not None],
            (run.evaluations[0].quality_scores.get("demo_fixture_match") == 1.0),
        )

    status, models, fixture_match = asyncio.run(execute())
    assert status is ExecutionStatus.COMPLETED
    assert models == expected_models
    assert fixture_match
