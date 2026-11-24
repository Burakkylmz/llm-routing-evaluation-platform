"""Compare all four routing strategies on one immutable, synthetic offline workload.

Application service: depends only on the orchestrator's contracts and injected
factories. Router/provider/evaluator composition stays in the CLI.

Each strategy receives its own *validated* ``RunOptions`` from an options
factory (never ``model_copy``), so strategy-specific rules such as the Cascade
call-budget guard (review #12 F3) and per-strategy policy digests (F7) apply
exactly as in a standalone ``run``. All option sets are validated before any
strategy executes.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    EvaluationStatus,
    ExecutionStatus,
    MetricOrigin,
    StrategyId,
    TraceStep,
    TraceStepKind,
)

from .orchestrator import (
    BenchmarkInputError,
    BenchmarkOrchestrator,
    RunOptions,
    RunResult,
    is_degraded_trace,
)

ALL_STRATEGIES: tuple[StrategyId, ...] = (
    StrategyId.ALWAYS_LARGE,
    StrategyId.RULE_BASED,
    StrategyId.KNN,
    StrategyId.CASCADE,
)


class ComparisonRow(BaseModel):
    """Secret-free per-strategy accounting; no raw cases or model responses."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    strategy: StrategyId
    run_id: str
    routing_policy_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    case_count: int = Field(ge=1)
    successful_cases: int = Field(ge=0)
    # Cascade cases that kept a lower-tier answer after a failed escalation (F2).
    degraded_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    blocked_cases: int = Field(ge=0)
    evaluated_cases: int = Field(ge=0)
    provider_attempts: int = Field(ge=0)
    successful_attempts: int = Field(ge=0)
    attempts_per_case: float = Field(ge=0, allow_inf_nan=False)
    calls_by_model: dict[str, int]
    local_case_latency_mean_ms: float | None = Field(default=None, ge=0)
    latency_case_samples: int = Field(ge=0)
    token_usage_samples: int = Field(ge=0)
    input_tokens_total: int | None = Field(default=None, ge=0)
    output_tokens_total: int | None = Field(default=None, ge=0)
    input_tokens_per_attempt: float | None = Field(default=None, ge=0)
    output_tokens_per_attempt: float | None = Field(default=None, ge=0)
    token_origin: Literal["SIMULATED", "OBSERVED", "MIXED", "UNKNOWN"]
    fixture_match_mean: float | None = Field(default=None, ge=0, le=1)
    fixture_score_samples: int = Field(ge=0)
    cost_amount: None = None  # no verified pricing: unknown, not zero

    @computed_field(return_type=int)
    @property
    def strict_successful_cases(self) -> int:
        """Comparable with baselines: excludes retained lower-tier answers."""
        return self.successful_cases - self.degraded_cases

    @model_validator(mode="after")
    def check_coverage(self) -> Self:
        if (
            self.successful_cases + self.failed_cases + self.blocked_cases != self.case_count
            or self.degraded_cases > self.successful_cases
            or self.successful_attempts > self.provider_attempts
            or self.evaluated_cases > self.case_count
            or self.token_usage_samples > self.provider_attempts
            or sum(self.calls_by_model.values()) != self.provider_attempts
        ):
            raise ValueError("COMPARISON_COUNTS_INCONSISTENT")
        return self


class ComparisonReport(BaseModel):
    """One four-strategy snapshot; no persisted prompts or evaluation references."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    execution_mode: Literal["OFFLINE_SIMULATION"] = "OFFLINE_SIMULATION"
    dataset_version: str = Field(min_length=1)
    dataset_content_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    created_at: datetime
    strategies: tuple[ComparisonRow, ...] = Field(min_length=4, max_length=4)
    limitations: tuple[str, ...] = (
        "Synthetic offline prompts and fake model outputs; no real financial quality evaluated.",
        "Fixture-match is a fake-provider contract test, not answer accuracy.",
        "Token counts are simulated, not billed usage; cost is unknown, not zero.",
        "Local end-to-end timings are NOT provider/API latency benchmarks.",
        "A handful of synthetic cases cannot establish statistical significance or rankings.",
    )

    @model_validator(mode="after")
    def require_all_strategies(self) -> Self:
        if tuple(row.strategy for row in self.strategies) != ALL_STRATEGIES:
            raise ValueError("COMPARISON_STRATEGY_ORDER_INVALID")
        if len({row.run_id for row in self.strategies}) != len(self.strategies):
            raise ValueError("COMPARISON_RUN_IDS_REUSED")
        return self


def _digest_cases(cases: tuple[BenchmarkCase, ...]) -> str:
    """Content-address the workload without writing original inputs to reports."""
    payload = [case.model_dump(mode="json") for case in cases]
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def _token_values(
    attempts: list[TraceStep], field: Literal["input_tokens", "output_tokens"]
) -> tuple[int | None, float | None]:
    """Incomplete usage means no misleading total or per-attempt average."""
    if not attempts:
        return None, None
    samples = [getattr(step.usage, field) if step.usage is not None else None for step in attempts]
    if any(value is None for value in samples):
        return None, None
    numeric = [value for value in samples if value is not None]
    return sum(numeric), sum(numeric) / len(attempts)


def _row(
    result: RunResult,
    strategy: StrategyId,
    options: RunOptions,
    expected_cases: tuple[str, ...],
) -> ComparisonRow:
    traces = result.traces
    evaluations = result.evaluations
    if (
        tuple(trace.case_id for trace in traces) != expected_cases
        or tuple(item.case_id for item in evaluations) != expected_cases
        or any(trace.strategy != strategy for trace in traces)
        or strategy not in result.report.strategy_metrics
    ):
        raise BenchmarkInputError("COMPARISON_WORKLOAD_MISMATCH")

    attempts = [
        step
        for trace in traces
        for step in trace.steps
        if step.kind is TraceStepKind.PROVIDER_ATTEMPT
    ]
    models = Counter(step.model for step in attempts if step.model is not None)
    origins = {step.metric_origin for step in attempts if step.metric_origin is not None}
    origin: Literal["SIMULATED", "OBSERVED", "MIXED", "UNKNOWN"]
    if not origins:
        origin = "UNKNOWN"
    elif origins == {MetricOrigin.SIMULATED}:
        origin = "SIMULATED"
    elif origins == {MetricOrigin.OBSERVED}:
        origin = "OBSERVED"
    else:
        origin = "MIXED"

    case_latencies = [t.total_latency_ms for t in traces if t.total_latency_ms is not None]
    input_total, input_average = _token_values(attempts, "input_tokens")
    output_total, output_average = _token_values(attempts, "output_tokens")
    fixture_scores = [
        score
        for evaluation in evaluations
        if evaluation.status is EvaluationStatus.COMPLETED
        and evaluation.evaluation_method == "offline_fixture_only"
        and (score := evaluation.quality_scores.get("demo_fixture_match")) is not None
    ]
    return ComparisonRow(
        strategy=strategy,
        run_id=result.report.run_id,
        routing_policy_digest=options.routing_policy_digest,
        case_count=len(traces),
        successful_cases=sum(t.status is ExecutionStatus.COMPLETED for t in traces),
        degraded_cases=sum(is_degraded_trace(t) for t in traces),
        failed_cases=sum(
            t.status in (ExecutionStatus.FAILED, ExecutionStatus.PARTIAL) for t in traces
        ),
        blocked_cases=sum(t.status is ExecutionStatus.BLOCKED for t in traces),
        evaluated_cases=sum(e.status is EvaluationStatus.COMPLETED for e in evaluations),
        provider_attempts=len(attempts),
        successful_attempts=sum(s.call_status is CallStatus.SUCCESS for s in attempts),
        attempts_per_case=len(attempts) / len(traces),
        calls_by_model=dict(sorted(models.items())),
        local_case_latency_mean_ms=_mean(case_latencies),
        latency_case_samples=len(case_latencies),
        token_usage_samples=sum(step.usage is not None for step in attempts),
        input_tokens_total=input_total,
        output_tokens_total=output_total,
        input_tokens_per_attempt=input_average,
        output_tokens_per_attempt=output_average,
        token_origin=origin,
        fixture_match_mean=_mean(fixture_scores),
        fixture_score_samples=len(fixture_scores),
    )


class StrategyComparisonRunner:
    """Run one independent orchestrator per strategy against the same workload."""

    def __init__(
        self,
        *,
        orchestrator_factory: Callable[[StrategyId], Awaitable[BenchmarkOrchestrator]],
        options_factory: Callable[[StrategyId], RunOptions],
    ) -> None:
        self._orchestrator_factory = orchestrator_factory
        self._options_factory = options_factory

    async def compare(
        self, cases: Sequence[BenchmarkCase], *, dataset_version: str
    ) -> ComparisonReport:
        workload = tuple(cases)
        if not workload or len({case.case_id for case in workload}) != len(workload):
            raise BenchmarkInputError("COMPARISON_INVALID_WORKLOAD")
        if any(case.metadata.get("synthetic") is not True for case in workload):
            raise BenchmarkInputError("COMPARISON_REQUIRES_SYNTHETIC_CASES")
        # Validate every strategy's options before running anything.
        per_strategy: list[RunOptions] = []
        for strategy in ALL_STRATEGIES:
            options = self._options_factory(strategy)
            if not isinstance(options, RunOptions) or options.strategy is not strategy:
                raise BenchmarkInputError("COMPARISON_OPTIONS_STRATEGY_MISMATCH")
            if options.provider != "fake":
                raise BenchmarkInputError("COMPARISON_OFFLINE_ONLY")
            per_strategy.append(options)
        expected = tuple(case.case_id for case in workload)
        rows: list[ComparisonRow] = []
        for strategy, options in zip(ALL_STRATEGIES, per_strategy, strict=True):
            orchestrator = await self._orchestrator_factory(strategy)
            result = await orchestrator.run(workload, options)
            rows.append(_row(result, strategy, options, expected))
        return ComparisonReport(
            dataset_version=dataset_version,
            dataset_content_sha256=_digest_cases(workload),
            created_at=datetime.now(UTC),
            strategies=tuple(rows),
        )
