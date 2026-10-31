"""Offline-only fake adapters for the Member 01 integration smoke test.

These are intentionally NOT banking scorers, real routers, PII detectors or providers.
Replace with Member 02–04 adapters before using any real traffic.
"""

from __future__ import annotations

from llm_routing_eval.core import (
    BenchmarkCase,
    BenchmarkReport,
    CallStatus,
    EvaluationResult,
    EvaluationStatus,
    ExecutionTrace,
    GoldenReference,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
    RoutingAction,
    RoutingContext,
    RoutingDecision,
    SanitizationStatus,
    SanitizedText,
    StrategyId,
    TokenUsage,
)

DEMO_PROMPTS = frozenset({
    "Banka kartı ile kredi kartı arasındaki fark nedir?",
    "Banka hesabı açmak için genel süreç nedir?",
})
DEMO_RESPONSE = "OFFLINE_FAKE_RESPONSE_NOT_FINANCIAL_ADVICE"


class DemoAlwaysLargeRouter:
    """Test double, not Member 02's production Always-Large implementation."""

    def __init__(self, model: str) -> None:
        self._model = model

    async def decide(self, case: BenchmarkCase, context: RoutingContext) -> RoutingDecision:
        _ = case
        return RoutingDecision(
            strategy=StrategyId.ALWAYS_LARGE,
            action=(RoutingAction.INVOKE_MODEL if not context.prior_attempts
                    else RoutingAction.FINISH),
            stage_index=context.current_stage,
            reason_code="OFFLINE_BASELINE_TEST_DOUBLE",
            decision_time_ms=0.0,
            selected_model=self._model if not context.prior_attempts else None,
        )


class DemoFakeProvider:
    async def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            provider=request.provider, model=request.model,
            attempt_id=request.attempt_id, status=CallStatus.SUCCESS,
            metric_origin=MetricOrigin.SIMULATED,
            content=DEMO_RESPONSE, usage=TokenUsage(input_tokens=16, output_tokens=8),
            latency_ms=5.0,
        )


class DemoFixtureEvaluator:
    """Checks test-double behavior; not financial correctness or a golden rubric."""

    async def evaluate(
        self,
        case: BenchmarkCase,
        response: ModelResponse | None,
        reference: GoldenReference | None,
        *,
        run_id: str,
        trace_id: str,
    ) -> EvaluationResult:
        _ = reference
        is_success = response is not None and response.status is CallStatus.SUCCESS
        return EvaluationResult(
            run_id=run_id, case_id=case.case_id, trace_id=trace_id,
            quality_scores={"demo_fixture_match": float(response.content == DEMO_RESPONSE)}
            if is_success and response is not None else {},
            critical_errors=(), evaluation_method="offline_fixture_only",
            evaluator_version="demo-v1",
            status=EvaluationStatus.COMPLETED if is_success else EvaluationStatus.NOT_EVALUATED,
        )


class AllowlistedDemoPrivacy:
    """Allows ONLY bundled synthetic sentences; not a production sanitizer."""

    def sanitize(self, text: str, *, purpose: str) -> SanitizedText:
        allowed = purpose in ("routing", "provider") and text in DEMO_PROMPTS
        return SanitizedText(
            text=text if allowed else None,
            status=SanitizationStatus.ALLOWED if allowed else SanitizationStatus.BLOCKED,
            policy_version="offline-allowlist-v1",
            reason_code="KNOWN_SYNTHETIC_FIXTURE" if allowed else "NOT_DEMO_FIXTURE",
        )


class InMemoryDemoRepository:
    """No external IO; CLI exports only redacted trace/evaluation/report contracts."""

    def __init__(self) -> None:
        self.traces: list[ExecutionTrace] = []
        self.evaluations: list[EvaluationResult] = []
        self.reports: list[BenchmarkReport] = []

    async def save_trace(self, trace: ExecutionTrace) -> None:
        self.traces.append(trace)

    async def save_evaluation(self, result: EvaluationResult) -> None:
        self.evaluations.append(result)

    async def save_report(self, report: BenchmarkReport) -> None:
        self.reports.append(report)
