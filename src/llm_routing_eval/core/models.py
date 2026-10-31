"""Versioned Pydantic contracts and internal transfer objects (no provider dependencies).

These models define *structure*, not an irreversible-anonymization guarantee.
Only sanitized strings are permitted at persistence and outbound boundaries.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from .enums import (
    CallStatus,
    CostOrigin,
    ErrorCategory,
    EvaluationStatus,
    ExecutionStatus,
    MetricOrigin,
    RoutingAction,
    SanitizationStatus,
    StrategyId,
    TraceStepKind,
    VerificationStatus,
)

SCHEMA_VERSION = "1.0.0"
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeMs = Annotated[float, Field(ge=0, allow_inf_nan=False)]
NonNegativeCount = Annotated[int, Field(ge=0)]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonNegativeMoney = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class ImmutableModel(BaseModel):
    """Shared frozen shape; forbids undeclared fields."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, hide_input_in_errors=True
    )


class VersionedRecord(ImmutableModel):
    """All persisted top-level records start at schema v1.0.0."""

    schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^1\.0\.0$")


class Evidence(ImmutableModel):
    source_id: NonEmptyStr
    as_of: date
    provenance: NonEmptyStr
    verification_note: str | None = None


class TokenUsage(ImmutableModel):
    """Counts are nullable; reasoning/cached may overlap other totals."""

    input_tokens: NonNegativeCount | None = None
    output_tokens: NonNegativeCount | None = None
    cached_tokens: NonNegativeCount | None = None
    reasoning_tokens: NonNegativeCount | None = None

    @model_validator(mode="after")
    def cached_cannot_exceed_input(self) -> Self:
        if (
            self.cached_tokens is not None
            and self.input_tokens is not None
            and self.cached_tokens > self.input_tokens
        ):
            raise ValueError("cached_tokens cannot exceed input_tokens")
        return self


class ErrorInfo(ImmutableModel):
    category: ErrorCategory
    code: NonEmptyStr
    retryable: bool = False
    safe_message: str | None = Field(default=None, repr=False)


class BenchmarkCase(VersionedRecord):
    case_id: NonEmptyStr
    input_text: NonEmptyStr = Field(repr=False)
    domain: NonEmptyStr
    scenario_type: NonEmptyStr
    metadata: dict[str, JsonValue] = Field(default_factory=dict, repr=False)
    locale: NonEmptyStr = "tr-TR"


class GoldenReference(VersionedRecord):
    """Only evaluator-facing; must never be given to a router/provider."""

    case_id: NonEmptyStr
    dataset_version: NonEmptyStr
    expected_facts: tuple[NonEmptyStr, ...] = Field(repr=False)
    rubric_version: NonEmptyStr
    verification_status: VerificationStatus
    evidence: tuple[Evidence, ...] = ()
    reviewed_at: datetime | None = None
    reviewer_role: NonEmptyStr | None = None

    @field_validator("reviewed_at")
    @classmethod
    def normalize_review_timestamp(cls, value: datetime | None) -> datetime | None:
        return _utc(value) if value is not None else None

    @model_validator(mode="after")
    def validate_review_status(self) -> Self:
        if self.verification_status is VerificationStatus.EXPERT_VERIFIED and (
            self.reviewed_at is None or self.reviewer_role is None
        ):
            raise ValueError("EXPERT_VERIFIED requires reviewed_at and reviewer_role")
        return self


class RoutingDecision(VersionedRecord):
    strategy: StrategyId
    action: RoutingAction
    reason_code: NonEmptyStr
    stage_index: NonNegativeCount
    decision_time_ms: NonNegativeMs
    selected_model: NonEmptyStr | None = None
    safe_reason: str | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def model_selection_matches_action(self) -> Self:
        if self.action is RoutingAction.INVOKE_MODEL and self.selected_model is None:
            raise ValueError("INVOKE_MODEL requires selected_model")
        if self.action is RoutingAction.FINISH and self.selected_model is not None:
            raise ValueError("FINISH must not select a model")
        return self


class ModelResponse(VersionedRecord):
    """Ephemeral response; content must be sanitized before any persistence."""

    provider: NonEmptyStr
    model: NonEmptyStr
    attempt_id: NonEmptyStr
    status: CallStatus
    metric_origin: MetricOrigin
    content: str | None = Field(default=None, repr=False)
    usage: TokenUsage | None = None
    latency_ms: NonNegativeMs | None = None
    error: ErrorInfo | None = None
    provider_request_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.status is CallStatus.SUCCESS:
            if self.content is None or not self.content.strip():
                raise ValueError("Successful calls require nonempty content")
            if self.error is not None:
                raise ValueError("Successful calls cannot include error")
        elif self.content is not None or self.error is None:
            raise ValueError("Failed calls require error and must not include content")
        return self


class EvaluationResult(VersionedRecord):
    run_id: NonEmptyStr
    case_id: NonEmptyStr
    trace_id: NonEmptyStr
    quality_scores: dict[NonEmptyStr, Score | None]
    critical_errors: tuple[NonEmptyStr, ...]
    evaluation_method: NonEmptyStr
    evaluator_version: NonEmptyStr
    status: EvaluationStatus
    review_status: str | None = None
    safe_notes: tuple[str, ...] = ()
    judge_usage: TokenUsage | None = None

    @model_validator(mode="after")
    def failed_evaluation_is_not_scored(self) -> Self:
        if self.status in (EvaluationStatus.FAILED, EvaluationStatus.NOT_EVALUATED) and any(
            value is not None for value in self.quality_scores.values()
        ):
            raise ValueError("Failed/unavailable evaluations must not have valid scores")
        return self


class TraceStep(ImmutableModel):
    """Safe audit metadata, deliberately not a raw ModelResponse or prompt."""

    sequence: NonNegativeCount
    kind: TraceStepKind
    decision: RoutingDecision | None = None
    attempt_id: NonEmptyStr | None = None
    provider: NonEmptyStr | None = None
    model: NonEmptyStr | None = None
    call_status: CallStatus | None = None
    metric_origin: MetricOrigin | None = None
    usage: TokenUsage | None = None
    elapsed_ms: NonNegativeMs | None = None
    error: ErrorInfo | None = None
    safe_metadata: dict[str, JsonValue] = Field(default_factory=dict, repr=False)

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if self.kind is TraceStepKind.ROUTING_DECISION and self.decision is None:
            raise ValueError("ROUTING_DECISION event requires decision")
        if self.kind is TraceStepKind.PROVIDER_ATTEMPT and (
            self.attempt_id is None or self.call_status is None
        ):
            raise ValueError("PROVIDER_ATTEMPT requires attempt_id and call_status")
        return self


class ExecutionTrace(VersionedRecord):
    trace_id: NonEmptyStr
    run_id: NonEmptyStr
    case_id: NonEmptyStr
    strategy: StrategyId
    steps: tuple[TraceStep, ...]
    status: ExecutionStatus
    started_at: datetime
    ended_at: datetime | None = None
    total_latency_ms: NonNegativeMs | None = None
    error: ErrorInfo | None = None

    @field_validator("started_at", "ended_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime | None) -> datetime | None:
        return _utc(value) if value is not None else None

    @model_validator(mode="after")
    def validate_trace(self) -> Self:
        sequence = [step.sequence for step in self.steps]
        if sequence != sorted(set(sequence)):
            raise ValueError("Trace steps must have strictly increasing sequence numbers")
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        if self.status is ExecutionStatus.COMPLETED and not any(
            step.kind is TraceStepKind.PROVIDER_ATTEMPT
            and step.call_status is CallStatus.SUCCESS
            for step in self.steps
        ):
            raise ValueError("COMPLETED trace requires a successful provider attempt")
        decision_strategies = [
            step.decision.strategy for step in self.steps if step.decision is not None
        ]
        if any(strategy != self.strategy for strategy in decision_strategies):
            raise ValueError("Trace decisions must match trace strategy")
        attempt_ids = [
            step.attempt_id for step in self.steps
            if step.kind is TraceStepKind.PROVIDER_ATTEMPT
        ]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("Provider attempts require unique attempt_id")
        if self.status in (ExecutionStatus.FAILED, ExecutionStatus.BLOCKED) and self.error is None:
            raise ValueError("FAILED/BLOCKED trace requires an error")
        return self


class StrategyMetrics(ImmutableModel):
    """Unknown latency/quality/cost remains None with explicit observation coverage."""

    case_count: NonNegativeCount
    successful_cases: NonNegativeCount
    failed_cases: NonNegativeCount
    blocked_cases: NonNegativeCount
    evaluated_cases: NonNegativeCount
    critical_error_cases: NonNegativeCount
    quality_scores: dict[NonEmptyStr, Score | None] = Field(default_factory=dict)
    latency_p50_ms: NonNegativeMs | None = None
    latency_p95_ms: NonNegativeMs | None = None
    latency_samples: NonNegativeCount = 0
    cost_amount: NonNegativeMoney | None = None
    cost_currency: NonEmptyStr | None = None
    cost_origin: CostOrigin | None = None
    cost_samples: NonNegativeCount = 0

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        if self.successful_cases + self.failed_cases + self.blocked_cases > self.case_count:
            raise ValueError("Outcome counts exceed case_count")
        if self.evaluated_cases > self.case_count or self.critical_error_cases > self.case_count:
            raise ValueError("Evaluation counts exceed case_count")
        if self.latency_samples > self.case_count:
            raise ValueError("latency_samples exceed case_count")
        if self.cost_amount is not None and (
            self.cost_currency is None or self.cost_origin is None
        ):
            raise ValueError("Cost requires currency and cost origin")
        return self


class BenchmarkReport(VersionedRecord):
    run_id: NonEmptyStr
    dataset_version: NonEmptyStr
    configuration_fingerprint: NonEmptyStr
    strategy_metrics: dict[StrategyId, StrategyMetrics]
    generated_at: datetime
    pricing_version: NonEmptyStr | None = None
    run_status: NonEmptyStr = "COMPLETED"
    limitations: tuple[str, ...] = ()
    provenance: dict[str, str] = Field(default_factory=dict)

    @field_validator("generated_at")
    @classmethod
    def normalize_generation_timestamp(cls, value: datetime) -> datetime:
        return _utc(value)


class RoutingFeedback(ImmutableModel):
    """Prior model-call outcome made available to Cascade; no golden fields."""

    attempt_id: NonEmptyStr
    model: NonEmptyStr
    status: CallStatus
    safe_response_excerpt: str | None = Field(default=None, repr=False)
    error_code: NonEmptyStr | None = None


class RoutingContext(ImmutableModel):
    current_stage: NonNegativeCount
    remaining_model_calls: NonNegativeCount
    prior_decisions: tuple[RoutingDecision, ...] = ()
    prior_attempts: tuple[RoutingFeedback, ...] = ()


class ModelRequest(ImmutableModel):
    """An already sanitized prompt; credentials are intentionally not part of this DTO."""

    run_id: NonEmptyStr
    case_id: NonEmptyStr
    trace_id: NonEmptyStr
    attempt_id: NonEmptyStr
    provider: NonEmptyStr
    model: NonEmptyStr
    prompt: NonEmptyStr = Field(repr=False)
    timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] | None = None
    max_output_tokens: Annotated[int, Field(gt=0)] | None = None
    generation_parameters: dict[str, JsonValue] = Field(default_factory=dict, repr=False)


class SanitizedText(ImmutableModel):
    text: str | None = Field(repr=False)
    policy_version: NonEmptyStr
    status: SanitizationStatus
    reason_code: NonEmptyStr

    @model_validator(mode="after")
    def enforce_fail_closed(self) -> Self:
        if self.status is SanitizationStatus.BLOCKED and self.text is not None:
            raise ValueError("BLOCKED sanitization must not return text")
        if self.status is SanitizationStatus.ALLOWED and (
            self.text is None or not self.text.strip()
        ):
            raise ValueError("ALLOWED sanitization requires nonempty text")
        return self


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware")
    return value.astimezone(UTC)
