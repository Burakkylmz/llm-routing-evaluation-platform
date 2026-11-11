"""Member 01 application-service tests. No network, provider keys, or real financial data."""

import asyncio
from collections.abc import Callable

import pytest

from llm_routing_eval.application import (
    BenchmarkInputError,
    BenchmarkOrchestrator,
    BenchmarkPersistenceError,
    RunOptions,
)
from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    EvaluationResult,
    EvaluationStatus,
    ExecutionStatus,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
    RoutingAction,
    RoutingContext,
    RoutingDecision,
    SanitizationStatus,
    SanitizedText,
    StrategyId,
    TraceStepKind,
)
from llm_routing_eval.demo.fakes import (
    DEMO_PROMPTS,
    AllowlistedDemoPrivacy,
    DemoAlwaysLargeRouter,
    DemoFakeProvider,
    DemoFixtureEvaluator,
    InMemoryDemoRepository,
)


def fixture_case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="demo-001",
        input_text=sorted(DEMO_PROMPTS)[0],
        domain="turkish_banking",
        scenario_type="synthetic_test",
        metadata={"synthetic": True, "golden_label": "MUST_NOT_REACH_ROUTER"},
    )


def setup(
    *,
    router: object | None = None,
    provider: object | None = None,
    evaluator: object | None = None,
    privacy: object | None = None,
    repo: InMemoryDemoRepository | None = None,
    sleep: Callable[..., object] | None = None,
) -> tuple[BenchmarkOrchestrator, InMemoryDemoRepository]:
    repository = repo or InMemoryDemoRepository()
    options: dict[str, object] = {
        "router": router or DemoAlwaysLargeRouter("large"),
        "provider": provider or DemoFakeProvider(),
        "evaluator": evaluator or DemoFixtureEvaluator(),
        "privacy": privacy or AllowlistedDemoPrivacy(),
        "repository": repository,
    }
    if sleep:
        options["sleep"] = sleep
    return BenchmarkOrchestrator(**options), repository  # type: ignore[arg-type]


def opts(**kwargs: object) -> RunOptions:
    return RunOptions(allowed_models=("large", "small"), **kwargs)


def test_offline_success_id_relationships_and_simulated_cost() -> None:
    engine, repo = setup()
    result = asyncio.run(engine.run([fixture_case()], opts()))
    assert len(repo.traces) == len(repo.evaluations) == len(repo.reports) == 1
    trace, evaluation = result.traces[0], result.evaluations[0]
    assert result.report.run_id == trace.run_id == evaluation.run_id
    assert trace.trace_id == evaluation.trace_id
    assert trace.status is ExecutionStatus.COMPLETED
    assert [item.kind for item in trace.steps] == [
        TraceStepKind.PRIVACY_CHECK,
        TraceStepKind.ROUTING_DECISION,
        TraceStepKind.PROVIDER_ATTEMPT,
        TraceStepKind.ROUTING_DECISION,
        TraceStepKind.EVALUATION,
    ]
    metrics = result.report.strategy_metrics[StrategyId.ALWAYS_LARGE]
    assert metrics.case_count == metrics.evaluated_cases == 1
    assert metrics.cost_amount is None and metrics.cost_samples == 0
    assert metrics.latency_p95_ms is None
    assert trace.steps[2].metric_origin is MetricOrigin.SIMULATED
    assert "MUST_NOT_REACH_ROUTER" not in trace.model_dump_json()
    assert fixture_case().input_text not in trace.model_dump_json()


def test_repeated_runs_are_equivalent_ignoring_volatile_fields() -> None:
    engine, _ = setup()
    left = asyncio.run(engine.run([fixture_case()], opts()))
    right = asyncio.run(engine.run([fixture_case()], opts()))
    assert left.report.run_id != right.report.run_id
    assert left.report.configuration_fingerprint == right.report.configuration_fingerprint
    assert left.report.strategy_metrics[StrategyId.ALWAYS_LARGE].quality_scores == (
        right.report.strategy_metrics[StrategyId.ALWAYS_LARGE].quality_scores
    )
    assert left.evaluations[0].quality_scores == right.evaluations[0].quality_scores
    assert left.traces[0].steps[2].usage == right.traces[0].steps[2].usage


def test_privacy_failure_blocks_all_downstream_calls() -> None:
    class Deny:
        def sanitize(self, text: str, *, purpose: str) -> SanitizedText:
            return SanitizedText(
                text=None,
                policy_version="test",
                reason_code="DENIED",
                status=SanitizationStatus.BLOCKED,
            )

    class MustNeverCall:
        async def decide(self, *args: object) -> None:
            raise AssertionError("router invoked despite privacy denial")

        async def generate(self, *args: object) -> None:
            raise AssertionError("provider invoked despite privacy denial")

        async def evaluate(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("judge invoked despite privacy denial")

    deny = MustNeverCall()
    engine, _ = setup(router=deny, provider=deny, evaluator=deny, privacy=Deny())
    result = asyncio.run(engine.run([fixture_case()], opts()))
    trace = result.traces[0]
    assert trace.status is ExecutionStatus.BLOCKED
    assert all(step.kind is not TraceStepKind.PROVIDER_ATTEMPT for step in trace.steps)
    assert result.evaluations[0].status is EvaluationStatus.NOT_EVALUATED
    assert result.report.strategy_metrics[StrategyId.ALWAYS_LARGE].blocked_cases == 1


def test_provider_exception_becomes_safe_failure_without_judging() -> None:
    class FailingProvider:
        async def generate(self, request: ModelRequest) -> ModelResponse:
            raise RuntimeError("secret=PRIVATE_KEY_VALUE")

    engine, _ = setup(provider=FailingProvider())
    result = asyncio.run(engine.run([fixture_case()], opts()))
    trace = result.traces[0]
    assert trace.status is ExecutionStatus.FAILED
    assert trace.steps[2].call_status is CallStatus.FAILED
    assert result.evaluations[0].status is EvaluationStatus.NOT_EVALUATED
    assert "PRIVATE_KEY_VALUE" not in trace.model_dump_json()


def test_provider_timeout_retries_with_distinct_attempts() -> None:
    calls = []
    delays = []

    class Intermittent:
        async def generate(self, request: ModelRequest) -> ModelResponse:
            calls.append(request.attempt_id)
            if len(calls) == 1:
                return ModelResponse(
                    provider=request.provider,
                    model=request.model,
                    attempt_id=request.attempt_id,
                    status=CallStatus.TIMEOUT,
                    metric_origin=MetricOrigin.SIMULATED,
                    error=ErrorInfo(
                        category=ErrorCategory.PROVIDER_TIMEOUT, code="TIMEOUT", retryable=True
                    ),
                )
            return await DemoFakeProvider().generate(request)

    async def no_sleep(seconds: float) -> None:
        delays.append(seconds)

    engine, _ = setup(provider=Intermittent(), sleep=no_sleep)
    result = asyncio.run(engine.run([fixture_case()], opts(max_retries=1)))
    attempts = [s for s in result.traces[0].steps if s.kind is TraceStepKind.PROVIDER_ATTEMPT]
    assert [a.call_status for a in attempts] == [CallStatus.TIMEOUT, CallStatus.SUCCESS]
    assert len(calls) == len(set(calls)) == 2
    assert len(delays) == 1


def test_cascade_multiple_decisions_and_model_calls() -> None:
    class TwoStageRouter:
        async def decide(self, case: BenchmarkCase, context: RoutingContext) -> RoutingDecision:
            assert case.metadata == {}
            model = "small" if context.current_stage == 0 else "large"
            return RoutingDecision(
                strategy=StrategyId.CASCADE,
                action=RoutingAction.INVOKE_MODEL,
                selected_model=model,
                stage_index=context.current_stage,
                decision_time_ms=0.1,
                reason_code="STAGE_ROUTE",
            )

    engine, _ = setup(router=TwoStageRouter())
    result = asyncio.run(
        engine.run(
            [fixture_case()],
            opts(
                strategy=StrategyId.CASCADE,
                max_model_calls_per_case=2,
            ),
        )
    )
    trace = result.traces[0]
    assert trace.status is ExecutionStatus.COMPLETED
    assert [step.model for step in trace.steps if step.kind is TraceStepKind.PROVIDER_ATTEMPT] == [
        "small",
        "large",
    ]
    assert len([step for step in trace.steps if step.decision is not None]) == 2


def test_unapproved_model_is_not_called() -> None:
    class RogueRouter:
        async def decide(self, case: BenchmarkCase, context: RoutingContext) -> RoutingDecision:
            return RoutingDecision(
                strategy=StrategyId.ALWAYS_LARGE,
                action=RoutingAction.INVOKE_MODEL,
                selected_model="unapproved",
                stage_index=context.current_stage,
                decision_time_ms=0,
                reason_code="INVALID_MODEL",
            )

    engine, _ = setup(router=RogueRouter())
    result = asyncio.run(engine.run([fixture_case()], opts()))
    assert result.traces[0].status is ExecutionStatus.FAILED
    assert all(step.kind is not TraceStepKind.PROVIDER_ATTEMPT for step in result.traces[0].steps)


def test_evaluator_exception_is_partial_and_no_fabricated_score() -> None:
    class FailingEvaluator:
        async def evaluate(self, *args: object, **kwargs: object) -> EvaluationResult:
            raise RuntimeError("private sample text")

    engine, _ = setup(evaluator=FailingEvaluator())
    result = asyncio.run(engine.run([fixture_case()], opts()))
    assert result.traces[0].status is ExecutionStatus.PARTIAL
    assert result.evaluations[0].status is EvaluationStatus.FAILED
    assert result.evaluations[0].quality_scores == {}
    assert "private sample text" not in result.traces[0].model_dump_json()


def test_invalid_input_rejected_before_execution() -> None:
    engine, _ = setup()
    with pytest.raises(BenchmarkInputError):
        asyncio.run(engine.run([], opts()))
    case = fixture_case()
    with pytest.raises(BenchmarkInputError):
        asyncio.run(engine.run([case, case], opts()))


def test_persistence_failure_cannot_be_reported_successfully() -> None:
    class BrokenRepository(InMemoryDemoRepository):
        async def save_trace(self, trace: object) -> None:
            raise RuntimeError("secret TOKEN")

    repo = BrokenRepository()
    engine, _ = setup(repo=repo)
    with pytest.raises(BenchmarkPersistenceError, match="PERSISTENCE_FAILED"):
        asyncio.run(engine.run([fixture_case()], opts()))
    assert repo.reports == []


def test_injected_router_reason_is_not_written_to_trace() -> None:
    class UnsafeRouter:
        async def decide(self, case: BenchmarkCase, context: RoutingContext) -> RoutingDecision:
            return RoutingDecision(
                strategy=StrategyId.ALWAYS_LARGE,
                action=(
                    RoutingAction.INVOKE_MODEL
                    if not context.prior_attempts
                    else RoutingAction.FINISH
                ),
                selected_model="large" if not context.prior_attempts else None,
                stage_index=context.current_stage,
                decision_time_ms=0,
                reason_code="secret=SHOULD_BE_REDACTED",
                safe_reason="BANK_ACCOUNT_123456",
            )

    engine, _ = setup(router=UnsafeRouter())
    result = asyncio.run(engine.run([fixture_case()], opts()))
    text = result.traces[0].model_dump_json()
    assert "SHOULD_BE_REDACTED" not in text
    assert "BANK_ACCOUNT_123456" not in text
    assert "ROUTER_REASON_REDACTED" in text


def test_evaluator_untrusted_notes_are_not_persisted() -> None:
    class UnsafeNotes:
        async def evaluate(
            self,
            case: BenchmarkCase,
            response: ModelResponse | None,
            reference: object,
            *,
            run_id: str,
            trace_id: str,
        ) -> EvaluationResult:
            return EvaluationResult(
                case_id=case.case_id,
                run_id=run_id,
                trace_id=trace_id,
                quality_scores={"fixture_match": 1.0},
                critical_errors=(),
                evaluation_method="offline_fixture_only",
                evaluator_version="test-v1",
                status=EvaluationStatus.COMPLETED,
                safe_notes=("account=SECRET_12345",),
                review_status="SECRET_REVIEWER",
            )

    engine, _ = setup(evaluator=UnsafeNotes())
    result = asyncio.run(engine.run([fixture_case()], opts()))
    payload = result.evaluations[0].model_dump_json()
    assert "SECRET" not in payload
    assert result.evaluations[0].safe_notes == ()


def test_provider_privacy_rejection_prevents_model_call() -> None:
    class OnlyRoutePrivacy:
        def sanitize(self, text: str, *, purpose: str) -> SanitizedText:
            permitted = purpose == "routing"
            return SanitizedText(
                text=text if permitted else None,
                policy_version="deny-provider-v1",
                reason_code="TEST",
                status=(SanitizationStatus.ALLOWED if permitted else SanitizationStatus.BLOCKED),
            )

    class MustNeverCall:
        async def generate(self, request: ModelRequest) -> None:
            raise AssertionError("Privacy should have prevented provider call")

    engine, _ = setup(privacy=OnlyRoutePrivacy(), provider=MustNeverCall())
    result = asyncio.run(engine.run([fixture_case()], opts()))
    assert result.traces[0].status is ExecutionStatus.BLOCKED
    assert result.evaluations[0].status is EvaluationStatus.NOT_EVALUATED


def test_evaluator_receives_only_golden_reference_not_router() -> None:
    from llm_routing_eval.core import GoldenReference, VerificationStatus

    class ReferenceCheckingEvaluator(DemoFixtureEvaluator):
        async def evaluate(
            self,
            case: BenchmarkCase,
            response: ModelResponse | None,
            reference: GoldenReference | None,
            *,
            run_id: str,
            trace_id: str,
        ) -> EvaluationResult:
            assert reference is not None
            assert reference.expected_facts == ("PRIVATE_GOLDEN_ONLY",)
            assert case.metadata == {}
            return await super().evaluate(
                case,
                response,
                reference,
                run_id=run_id,
                trace_id=trace_id,
            )

    class MetadataCheckingRouter(DemoAlwaysLargeRouter):
        async def decide(self, case: BenchmarkCase, context: RoutingContext) -> RoutingDecision:
            assert "PRIVATE_GOLDEN_ONLY" not in case.model_dump_json()
            return await super().decide(case, context)

    ref = GoldenReference(
        case_id="demo-001",
        dataset_version="synthetic-dev-v1",
        expected_facts=("PRIVATE_GOLDEN_ONLY",),
        rubric_version="test",
        verification_status=VerificationStatus.UNREVIEWED,
    )
    engine, _ = setup(
        router=MetadataCheckingRouter("large"), evaluator=ReferenceCheckingEvaluator()
    )
    result = asyncio.run(engine.run([fixture_case()], opts(), references={"demo-001": ref}))
    assert result.traces[0].status is ExecutionStatus.COMPLETED
    assert "PRIVATE_GOLDEN_ONLY" not in result.traces[0].model_dump_json()


def test_provider_error_message_is_redacted_from_trace_root() -> None:
    class UnsafeErrorProvider:
        async def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                provider=request.provider,
                model=request.model,
                attempt_id=request.attempt_id,
                status=CallStatus.FAILED,
                metric_origin=MetricOrigin.SIMULATED,
                error=ErrorInfo(
                    category=ErrorCategory.PROVIDER,
                    code="account=1234567",
                    safe_message="my very secret passcode",
                ),
            )

    engine, _ = setup(provider=UnsafeErrorProvider())
    result = asyncio.run(engine.run([fixture_case()], opts()))
    trace_text = result.traces[0].model_dump_json()
    assert "1234567" not in trace_text
    assert "secret passcode" not in trace_text
    assert "PROVIDER_ERROR_REDACTED" in trace_text
