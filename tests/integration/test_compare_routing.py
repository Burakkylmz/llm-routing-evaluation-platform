"""Four-strategy offline comparison: runner invariants and compare-routing CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from llm_routing_eval.application import (
    ALL_STRATEGIES,
    BenchmarkInputError,
    BenchmarkOrchestrator,
    RunOptions,
    StrategyComparisonRunner,
)
from llm_routing_eval.cli.app import app
from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
    StrategyId,
)
from llm_routing_eval.demo.fakes import (
    DEMO_PROMPTS,
    AllowlistedDemoPrivacy,
    DemoFakeProvider,
    DemoFixtureEvaluator,
    InMemoryDemoRepository,
)
from llm_routing_eval.routing import (
    AlwaysLargeRouter,
    CascadeRouter,
    ModelPair,
    RuleBasedRouter,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "settings.yaml"
DATASET = ROOT / "examples" / "offline_demo_cases.json"
SMALL, LARGE = "m-small", "m-large"
PAIR = ModelPair(small=SMALL, large=LARGE)
runner = CliRunner()


def cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            case_id=f"c{i}",
            input_text=text,
            domain="turkish_banking",
            scenario_type="account_general",
            metadata={"synthetic": True},
        )
        for i, text in enumerate(sorted(DEMO_PROMPTS))
    ]


class LargeModelFails:
    async def generate(self, request: ModelRequest) -> ModelResponse:
        if request.model == LARGE:
            return ModelResponse(
                provider=request.provider,
                model=request.model,
                attempt_id=request.attempt_id,
                status=CallStatus.FAILED,
                metric_origin=MetricOrigin.SIMULATED,
                error=ErrorInfo(category=ErrorCategory.PROVIDER, code="SYNTHETIC_LARGE_DOWN"),
            )
        return await DemoFakeProvider().generate(request)


def make_runner(
    *,
    provider: object | None = None,
    options: dict[StrategyId, RunOptions] | None = None,
    built: list[StrategyId] | None = None,
) -> StrategyComparisonRunner:
    routers = {
        StrategyId.ALWAYS_LARGE: AlwaysLargeRouter(PAIR),
        StrategyId.RULE_BASED: RuleBasedRouter(PAIR, ()),
        # A kNN stand-in with the right strategy ID keeps this test embedding-free.
        StrategyId.KNN: _KnnStandIn(),
        StrategyId.CASCADE: CascadeRouter(PAIR, escalate_scenarios=frozenset({"account_general"})),
    }

    async def orchestrator_for(strategy: StrategyId) -> BenchmarkOrchestrator:
        if built is not None:
            built.append(strategy)
        return BenchmarkOrchestrator(
            router=routers[strategy],  # type: ignore[arg-type]
            provider=provider or DemoFakeProvider(),  # type: ignore[arg-type]
            evaluator=DemoFixtureEvaluator(),
            privacy=AllowlistedDemoPrivacy(),
            repository=InMemoryDemoRepository(),
        )

    def options_for(strategy: StrategyId) -> RunOptions:
        if options is not None and strategy in options:
            return options[strategy]
        return RunOptions(strategy=strategy, allowed_models=(SMALL, LARGE))

    return StrategyComparisonRunner(
        orchestrator_factory=orchestrator_for, options_factory=options_for
    )


class _KnnStandIn:
    async def decide(self, case: BenchmarkCase, context: Any) -> Any:
        decision = await AlwaysLargeRouter(PAIR).decide(case, context)
        return decision.model_copy(update={"strategy": StrategyId.KNN})


def test_all_four_strategies_run_on_the_same_workload() -> None:
    report = asyncio.run(make_runner().compare(cases(), dataset_version="v1"))
    assert tuple(row.strategy for row in report.strategies) == ALL_STRATEGIES
    assert {row.case_count for row in report.strategies} == {2}
    assert len({row.run_id for row in report.strategies}) == 4
    cascade = report.strategies[3]
    assert cascade.provider_attempts == 4 and cascade.attempts_per_case == 2.0
    assert cascade.calls_by_model == {LARGE: 2, SMALL: 2}
    assert report.dataset_content_sha256.startswith("sha256:")
    serialized = report.model_dump_json()
    assert all(prompt not in serialized for prompt in DEMO_PROMPTS)


def test_failed_escalations_reduce_strict_success_only_for_cascade() -> None:
    report = asyncio.run(
        make_runner(provider=LargeModelFails()).compare(cases(), dataset_version="v1")
    )
    rows = {row.strategy: row for row in report.strategies}
    cascade = rows[StrategyId.CASCADE]
    assert (cascade.successful_cases, cascade.degraded_cases) == (2, 2)
    assert cascade.strict_successful_cases == 0
    serialized = json.loads(report.model_dump_json())
    cascade_json = next(row for row in serialized["strategies"] if row["strategy"] == "cascade")
    assert cascade_json["strict_successful_cases"] == 0
    assert rows[StrategyId.ALWAYS_LARGE].successful_cases == 0  # large model is down
    assert rows[StrategyId.ALWAYS_LARGE].degraded_cases == 0


def test_all_options_are_validated_before_any_strategy_runs() -> None:
    built: list[StrategyId] = []

    def bad_cascade_options() -> dict[StrategyId, RunOptions]:
        # A valid object for the wrong strategy must be rejected before running.
        return {StrategyId.CASCADE: RunOptions(strategy=StrategyId.KNN, allowed_models=(LARGE,))}

    with pytest.raises(BenchmarkInputError, match="COMPARISON_OPTIONS_STRATEGY_MISMATCH"):
        asyncio.run(
            make_runner(options=bad_cascade_options(), built=built).compare(
                cases(), dataset_version="v1"
            )
        )
    live = {
        StrategyId.ALWAYS_LARGE: RunOptions(
            strategy=StrategyId.ALWAYS_LARGE, provider="groq", allowed_models=(LARGE,)
        )
    }
    with pytest.raises(BenchmarkInputError, match="COMPARISON_OFFLINE_ONLY"):
        asyncio.run(make_runner(options=live, built=built).compare(cases(), dataset_version="v1"))
    assert built == []


def test_workload_must_be_synthetic_and_unique() -> None:
    real = [c.model_copy(update={"metadata": {}}) for c in cases()]
    with pytest.raises(BenchmarkInputError, match="COMPARISON_REQUIRES_SYNTHETIC_CASES"):
        asyncio.run(make_runner().compare(real, dataset_version="v1"))
    duplicate = [cases()[0], cases()[0]]
    with pytest.raises(BenchmarkInputError, match="COMPARISON_INVALID_WORKLOAD"):
        asyncio.run(make_runner().compare(duplicate, dataset_version="v1"))
    with pytest.raises(BenchmarkInputError, match="COMPARISON_INVALID_WORKLOAD"):
        asyncio.run(make_runner().compare([], dataset_version="v1"))


# ---------------------------------------------------------------------- CLI


def _cli(tmp_path: Path, config: Path = CONFIG) -> tuple[Any, Path]:
    output = tmp_path / "comparison.json"
    result = runner.invoke(
        app,
        [
            "compare-routing",
            "--config",
            str(config),
            "--dataset",
            str(DATASET),
            "--output",
            str(output),
        ],
    )
    return result, output


def _config_with(tmp_path: Path, mutate: Any) -> Path:
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    mutate(settings)
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return path


def test_cli_writes_four_rows_with_per_strategy_policy_digests(tmp_path: Path) -> None:
    result, output = _cli(tmp_path)
    assert result.exit_code == 0, result.output
    assert "STRICT" in result.output and "COST: UNKNOWN" in result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    rows = report["strategies"]
    assert [row["strategy"] for row in rows] == [s.value for s in ALL_STRATEGIES]
    digests = [row["routing_policy_digest"] for row in rows]
    assert all(d and d.startswith("sha256:") for d in digests) and len(set(digests)) == 4
    assert report["execution_mode"] == "OFFLINE_SIMULATION"
    assert all(row["cost_amount"] is None for row in rows)
    assert all(
        row["strict_successful_cases"] == row["successful_cases"] - row["degraded_cases"]
        for row in rows
    )


def test_cli_rejects_cascade_budget_that_starves_escalation(tmp_path: Path) -> None:
    """The F3 guard applies inside the comparison exactly as in a standalone run."""
    config = _config_with(tmp_path, lambda s: s["execution"].update(max_retries=1))
    result, output = _cli(tmp_path, config)
    assert result.exit_code == 2
    assert "CASCADE_CALL_BUDGET_TOO_SMALL_FOR_RETRIES" in result.output
    assert not output.exists()


def test_cli_rejects_knn_examples_overlapping_evaluation_cases(tmp_path: Path) -> None:
    """The F4 guard always applies because kNN is part of every comparison."""
    demo = json.loads(DATASET.read_text(encoding="utf-8"))["cases"][0]["input_text"]
    config = _config_with(
        tmp_path,
        lambda s: s["routing"]["knn"]["examples"][0].update(input_text=demo),
    )
    result, output = _cli(tmp_path, config)
    assert result.exit_code == 2
    assert "KNN_DEVELOPMENT_TEST_LEAKAGE" in result.output
    assert not output.exists()


def test_cli_missing_config_fails_closed(tmp_path: Path) -> None:
    result, output = _cli(tmp_path, tmp_path / "missing.yaml")
    assert result.exit_code == 2 and "CONFIG_NOT_FOUND" in result.output
    assert not output.exists()
