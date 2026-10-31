"""Contract invariants: versioning, routing, telemetry, trace, and references."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from llm_routing_eval.core import (
    BenchmarkCase,
    BenchmarkReport,
    CallStatus,
    CostOrigin,
    ErrorCategory,
    ErrorInfo,
    EvaluationResult,
    EvaluationStatus,
    ExecutionStatus,
    ExecutionTrace,
    GoldenReference,
    MetricOrigin,
    ModelResponse,
    RoutingAction,
    RoutingDecision,
    SanitizationStatus,
    SanitizedText,
    StrategyId,
    StrategyMetrics,
    TokenUsage,
    TraceStep,
    TraceStepKind,
    VerificationStatus,
)


def case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="case-001",
        input_text="Kredi kartı yıllık ücreti nedir?",
        domain="turkish_banking",
        scenario_type="card_fee",
        metadata={"synthetic": True, "segments": [1, "card"]},
    )


def invoke(stage: int = 0) -> RoutingDecision:
    return RoutingDecision(
        strategy=StrategyId.ALWAYS_LARGE,
        action=RoutingAction.INVOKE_MODEL,
        reason_code="BASELINE_LARGE",
        stage_index=stage,
        decision_time_ms=0.7,
        selected_model="openai/gpt-oss-120b",
    )


def test_seven_top_level_schemas_are_json_serializable_and_versioned() -> None:
    started = datetime(2026, 10, 31, tzinfo=UTC)
    decision = invoke()
    success = ModelResponse(
        provider="fake",
        model="openai/gpt-oss-120b",
        attempt_id="a-1",
        status=CallStatus.SUCCESS,
        metric_origin=MetricOrigin.SIMULATED,
        content="Bankaya göre değişir.",
        usage=TokenUsage(input_tokens=12, output_tokens=5),
        latency_ms=1.2,
    )
    trace = ExecutionTrace(
        trace_id="trace-1",
        run_id="run-1",
        case_id=case().case_id,
        strategy=StrategyId.ALWAYS_LARGE,
        steps=(
            TraceStep(sequence=0, kind=TraceStepKind.ROUTING_DECISION, decision=decision),
            TraceStep(
                sequence=1,
                kind=TraceStepKind.PROVIDER_ATTEMPT,
                attempt_id=success.attempt_id,
                call_status=success.status,
                model=success.model,
                provider=success.provider,
                usage=success.usage,
                metric_origin=success.metric_origin,
            ),
        ),
        status=ExecutionStatus.COMPLETED,
        started_at=started,
        ended_at=started + timedelta(seconds=1),
    )
    result = EvaluationResult(
        run_id="run-1",
        case_id="case-001",
        trace_id="trace-1",
        quality_scores={"completeness": 1.0, "safety": None},
        critical_errors=(),
        evaluation_method="deterministic-v1",
        evaluator_version="0.1.0",
        status=EvaluationStatus.COMPLETED,
    )
    report = BenchmarkReport(
        run_id="run-1",
        dataset_version="dev-fixtures-v1",
        configuration_fingerprint="sha256:abc",
        strategy_metrics={
            StrategyId.ALWAYS_LARGE: StrategyMetrics(
                case_count=1,
                successful_cases=1,
                failed_cases=0,
                blocked_cases=0,
                evaluated_cases=1,
                critical_error_cases=0,
                latency_samples=1,
                latency_p50_ms=10.0,
            )
        },
        generated_at=started,
    )
    reference = GoldenReference(
        case_id="case-001",
        dataset_version="dev-fixtures-v1",
        expected_facts=("Fee may vary.",),
        rubric_version="rubric-v1",
        verification_status=VerificationStatus.UNREVIEWED,
    )
    top_level_records = (case(), reference, decision, success, result, trace, report)
    for record in top_level_records:
        assert record.schema_version == "1.0.0"
        assert type(record).model_validate_json(record.model_dump_json()) == record
        assert type(record).model_json_schema()

    assert "expected_facts" not in case().model_dump()
    assert "Bankaya" not in trace.model_dump_json()
    assert '"cost_amount":null' in report.model_dump_json()


def test_frozen_records_cannot_be_reassigned_or_given_extra_fields() -> None:
    with pytest.raises(ValidationError):
        case().case_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        BenchmarkCase(
            case_id="c", input_text="x", domain="bank", scenario_type="fee", golden="leak"
        )
    with pytest.raises(ValidationError):
        BenchmarkCase(
            case_id="c", input_text="x", domain="bank", scenario_type="fee",
            schema_version="9.0.0",
        )
    with pytest.raises(ValidationError):
        BenchmarkCase(
            case_id="c", input_text="x", domain="b", scenario_type="f",
            metadata={"x": object()},
        )


@pytest.mark.parametrize(
    ("action", "selected_model"),
    [(RoutingAction.INVOKE_MODEL, None), (RoutingAction.FINISH, "model")],
)
def test_routing_model_action_consistency(
    action: RoutingAction, selected_model: str | None
) -> None:
    with pytest.raises(ValidationError):
        RoutingDecision(
            strategy=StrategyId.CASCADE,
            action=action,
            selected_model=selected_model,
            reason_code="stage-gate",
            stage_index=1,
            decision_time_ms=0.0,
        )


def test_finish_decision_has_no_selected_model() -> None:
    decision = RoutingDecision(
        strategy=StrategyId.CASCADE,
        action=RoutingAction.FINISH,
        reason_code="sufficient",
        stage_index=1,
        decision_time_ms=0.0,
    )
    assert decision.selected_model is None


def test_missing_telemetry_stays_unknown_not_zero() -> None:
    response = ModelResponse(
        provider="fake",
        model="large",
        attempt_id="a",
        status=CallStatus.SUCCESS,
        metric_origin=MetricOrigin.SIMULATED,
        content="ok",
    )
    assert response.usage is None
    assert response.latency_ms is None
    assert TokenUsage().input_tokens is None
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=3, cached_tokens=4)
    with pytest.raises(ValidationError):
        TokenUsage(output_tokens=-1)


def test_failed_provider_call_requires_error_and_no_content() -> None:
    with pytest.raises(ValidationError):
        ModelResponse(
            provider="fake", model="m", attempt_id="a", status=CallStatus.TIMEOUT,
            metric_origin=MetricOrigin.SIMULATED,
        )
    with pytest.raises(ValidationError):
        ModelResponse(
            provider="fake", model="m", attempt_id="a", status=CallStatus.SUCCESS,
            metric_origin=MetricOrigin.SIMULATED, content="", error=ErrorInfo(
                category=ErrorCategory.PROVIDER, code="failed"
            ),
        )
    failed = ModelResponse(
        provider="fake",
        model="m",
        attempt_id="a",
        status=CallStatus.TIMEOUT,
        metric_origin=MetricOrigin.SIMULATED,
        error=ErrorInfo(category=ErrorCategory.PROVIDER_TIMEOUT, code="timeout", retryable=True),
    )
    assert failed.content is None


def test_trace_supports_multiple_attempts_and_rejects_duplicate_order() -> None:
    started = datetime(2026, 10, 31, tzinfo=UTC)
    error = ErrorInfo(category=ErrorCategory.PROVIDER_TIMEOUT, code="timeout", retryable=True)
    steps = (
        TraceStep(sequence=0, kind=TraceStepKind.ROUTING_DECISION, decision=invoke()),
        TraceStep(
            sequence=1, kind=TraceStepKind.PROVIDER_ATTEMPT, attempt_id="a1",
            call_status=CallStatus.TIMEOUT, error=error,
        ),
        TraceStep(
            sequence=2, kind=TraceStepKind.PROVIDER_ATTEMPT, attempt_id="a2",
            call_status=CallStatus.SUCCESS, metric_origin=MetricOrigin.SIMULATED,
        ),
    )
    result = ExecutionTrace(
        trace_id="t", run_id="r", case_id="c", strategy=StrategyId.ALWAYS_LARGE,
        steps=steps, status=ExecutionStatus.COMPLETED, started_at=started,
    )
    assert len(result.steps) == 3
    with pytest.raises(ValidationError):
        ExecutionTrace(
            trace_id="t", run_id="r", case_id="c", strategy=StrategyId.ALWAYS_LARGE,
            steps=(steps[1], steps[1]), status=ExecutionStatus.FAILED,
            error=error, started_at=started,
        )
    with pytest.raises(ValidationError):
        ExecutionTrace(
            trace_id="t", run_id="r", case_id="c", strategy=StrategyId.ALWAYS_LARGE,
            steps=(), status=ExecutionStatus.COMPLETED, started_at=started,
        )


def test_reference_requires_review_evidence_before_expert_claim() -> None:
    with pytest.raises(ValidationError):
        GoldenReference(
            case_id="c", dataset_version="v1", expected_facts=("x",),
            rubric_version="r1", verification_status=VerificationStatus.EXPERT_VERIFIED,
        )
    with pytest.raises(ValidationError):
        GoldenReference(
            case_id="c", dataset_version="v1", expected_facts=("x",),
            rubric_version="r1", verification_status=VerificationStatus.UNREVIEWED,
            reviewed_at=datetime(2026, 10, 31),
        )


def test_failed_evaluation_does_not_claim_scores() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult(
            run_id="r", case_id="c", trace_id="t", quality_scores={"quality": 0.9},
            critical_errors=(), evaluation_method="deterministic", evaluator_version="v1",
            status=EvaluationStatus.FAILED,
        )
    with pytest.raises(ValidationError):
        EvaluationResult(
            run_id="r", case_id="c", trace_id="t", quality_scores={"quality": 1.1},
            critical_errors=(), evaluation_method="deterministic", evaluator_version="v1",
            status=EvaluationStatus.COMPLETED,
        )


def test_cost_decimal_serializes_as_string_and_missing_cost_is_none() -> None:
    metrics = StrategyMetrics(
        case_count=1, successful_cases=1, failed_cases=0, blocked_cases=0,
        evaluated_cases=1, critical_error_cases=0, cost_amount=Decimal("0.0031"),
        cost_currency="USD", cost_origin=CostOrigin.SIMULATED, cost_samples=1,
    )
    assert metrics.model_dump(mode="json")["cost_amount"] == "0.0031"
    with pytest.raises(ValidationError):
        StrategyMetrics(
            case_count=1, successful_cases=2, failed_cases=0, blocked_cases=0,
            evaluated_cases=0, critical_error_cases=0,
        )


def test_privacy_status_fails_closed() -> None:
    with pytest.raises(ValidationError):
        SanitizedText(
            text="unsafe", status=SanitizationStatus.BLOCKED,
            policy_version="v1", reason_code="PII_DETECTED",
        )
    assert SanitizedText(
        text=None, status=SanitizationStatus.BLOCKED,
        policy_version="v1", reason_code="PII_DETECTED",
    ).text is None


def test_trace_mismatched_strategy_and_attempt_ids_are_rejected() -> None:
    started = datetime(2026, 10, 31, tzinfo=UTC)
    decision = TraceStep(sequence=0, kind=TraceStepKind.ROUTING_DECISION, decision=invoke())
    attempt = TraceStep(
        sequence=1, kind=TraceStepKind.PROVIDER_ATTEMPT,
        attempt_id="a1", call_status=CallStatus.SUCCESS,
    )
    with pytest.raises(ValidationError):
        ExecutionTrace(
            trace_id="t", run_id="r", case_id="c", strategy=StrategyId.CASCADE,
            steps=(decision, attempt), status=ExecutionStatus.COMPLETED, started_at=started,
        )
    with pytest.raises(ValidationError):
        ExecutionTrace(
            trace_id="t", run_id="r", case_id="c", strategy=StrategyId.ALWAYS_LARGE,
            steps=(decision, attempt, attempt.model_copy(update={"sequence": 2})),
            status=ExecutionStatus.COMPLETED, started_at=started,
        )


def test_blocked_trace_requires_error_without_successful_provider() -> None:
    trace = ExecutionTrace(
        trace_id="t", run_id="r", case_id="c", strategy=StrategyId.ALWAYS_LARGE,
        steps=(
            TraceStep(
                sequence=0, kind=TraceStepKind.PRIVACY_CHECK,
                safe_metadata={"blocked": True},
            ),
        ),
        status=ExecutionStatus.BLOCKED,
        error=ErrorInfo(category=ErrorCategory.PRIVACY, code="PII_POLICY_BLOCK"),
        started_at=datetime(2026, 10, 31, tzinfo=UTC),
    )
    assert trace.status is ExecutionStatus.BLOCKED
