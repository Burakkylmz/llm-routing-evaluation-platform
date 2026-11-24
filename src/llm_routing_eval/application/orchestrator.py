"""Application service: execute one controlled, auditable routing benchmark run.

The service depends only on core contracts; no provider/storage/domain imports.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import statistics
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_routing_eval.core import (
    BenchmarkCase,
    BenchmarkReport,
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorPort,
    ExecutionStatus,
    ExecutionTrace,
    GoldenReference,
    LLMProviderPort,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
    PrivacyPort,
    ResultRepositoryPort,
    RouterPort,
    RoutingAction,
    RoutingContext,
    RoutingDecision,
    RoutingFeedback,
    SanitizationStatus,
    StrategyId,
    StrategyMetrics,
    TraceStep,
    TraceStepKind,
)


# F9: positive minimum retry delay, bounded exponential growth and equal jitter.
RETRY_BACKOFF_POLICY_VERSION = "bounded-equal-jitter-v1"
_RETRY_BASE_SECONDS = 0.25
_RETRY_CAP_SECONDS = 2.0


def _retry_backoff_seconds(retry_index: int, jitter: float) -> float:
    """Delay before the next retry; jitter is a sampled uniform [0, 1) value.

    Unlike full jitter, equal jitter never reaches zero; worst-case bursts
    from concurrent retrying clients are reduced without extending the
    configured retry or provider-call budget.
    """
    ceiling = min(_RETRY_CAP_SECONDS, _RETRY_BASE_SECONDS * (2**retry_index))
    return ceiling * (0.5 + 0.5 * jitter)


class RunOptions(BaseModel):
    """Validated, secret-free configuration for a single sequential benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    strategy: StrategyId = StrategyId.ALWAYS_LARGE
    provider: str = "fake"
    allowed_models: tuple[str, ...] = ("openai/gpt-oss-120b",)
    dataset_version: str = "synthetic-dev-v1"
    max_model_calls_per_case: int = Field(default=2, ge=1, le=20)
    max_retries: int = Field(default=0, ge=0, le=3)
    timeout_seconds: float = Field(default=20, gt=0, le=120, allow_inf_nan=False)
    temperature: float | None = Field(default=0, ge=0, le=2)
    max_output_tokens: int | None = Field(default=128, gt=0)
    prompt_version: str = "offline-demo-v1"
    # Digest of the effective strategy policy, supplied by the composition root.
    # Keep configuration details (notably kNN training text) out of persisted reports.
    routing_policy_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_models(self) -> Self:
        if (
            not self.provider.strip()
            or not self.allowed_models
            or any(not model.strip() for model in self.allowed_models)
        ):
            raise ValueError("Provider and allowed models must be defined")
        if len(set(self.allowed_models)) != len(self.allowed_models):
            raise ValueError("Allowed models must be unique")
        # Review #12 F3: retries of the small stage must never consume the
        # escalation stage's budget; the large model gets the same retry
        # allowance as in the Always-Large baseline.
        if self.strategy is StrategyId.CASCADE and self.max_model_calls_per_case < 2 * (
            self.max_retries + 1
        ):
            raise ValueError("CASCADE_CALL_BUDGET_TOO_SMALL_FOR_RETRIES")
        return self

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        if self.max_retries > 0:
            # Backoff changes observable latency; fingerprint the selected
            # policy without modifying the public RunOptions schema.
            payload["retry_backoff_policy_version"] = RETRY_BACKOFF_POLICY_VERSION
        return (
            "sha256:"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )


class BenchmarkInputError(ValueError):
    """Input or configuration is invalid; message is intentionally non-sensitive."""


class BenchmarkPersistenceError(RuntimeError):
    """A result could not be durably saved; no success report should be issued."""


class RunResult(BaseModel):
    """Caller-visible summary, without prompt or answer bodies."""

    model_config = ConfigDict(frozen=True)
    report: BenchmarkReport
    traces: tuple[ExecutionTrace, ...]
    evaluations: tuple[EvaluationResult, ...]


class BenchmarkOrchestrator:
    """Coordinate privacy, routing, retries, model calls, evaluation and persistence."""

    def __init__(
        self,
        *,
        router: RouterPort,
        provider: LLMProviderPort,
        evaluator: EvaluatorPort,
        privacy: PrivacyPort,
        repository: ResultRepositoryPort,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._router = router
        self._provider = provider
        self._evaluator = evaluator
        self._privacy = privacy
        self._repository = repository
        self._sleep = sleep

    async def run(
        self,
        cases: Sequence[BenchmarkCase],
        options: RunOptions,
        *,
        references: Mapping[str, GoldenReference] | None = None,
    ) -> RunResult:
        if not cases or len({case.case_id for case in cases}) != len(cases):
            raise BenchmarkInputError("Cases must be nonempty and have unique identifiers")
        if references and any(key != ref.case_id for key, ref in references.items()):
            raise BenchmarkInputError("Reference identifiers do not match their keys")

        run_id = uuid4().hex
        traces: list[ExecutionTrace] = []
        evaluations: list[EvaluationResult] = []
        for case in cases:
            trace, evaluation = await self._execute_case(
                case, options, run_id, (references or {}).get(case.case_id)
            )
            # A failed save makes the entire run visibly unsuccessful.
            await self._persist(self._repository.save_trace(trace))
            await self._persist(self._repository.save_evaluation(evaluation))
            traces.append(trace)
            evaluations.append(evaluation)

        metrics = _summarize(traces, evaluations)
        report = BenchmarkReport(
            run_id=run_id,
            dataset_version=options.dataset_version,
            configuration_fingerprint=options.fingerprint(),
            strategy_metrics={options.strategy: metrics},
            generated_at=datetime.now(UTC),
            run_status=(
                "COMPLETED"
                if all(t.status is ExecutionStatus.COMPLETED for t in traces)
                else "PARTIAL"
            ),
            limitations=(
                "Offline fake outputs are not evidence of financial answer quality.",
                "Fake-model tokens and inference latency are simulated, not observed.",
                "End-to-end latency measures local application execution, not real API latency.",
                "No verified pricing; model cost is unknown, not zero.",
                "Small sample sizes are not sufficient for production latency conclusions.",
            ),
            provenance={
                "execution_mode": "offline_demo" if options.provider == "fake" else "configured",
                "prompt_version": options.prompt_version,
                "privacy_boundary": "pre-routing and pre-provider",
                "retry_backoff_policy": (
                    RETRY_BACKOFF_POLICY_VERSION if options.max_retries else "no_retries"
                ),
                "dataset_identifiers_sha256": hashlib.sha256(
                    json.dumps(
                        [case.case_id for case in cases],
                        sort_keys=True,
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest(),
            },
        )
        await self._persist(self._repository.save_report(report))
        return RunResult(report=report, traces=tuple(traces), evaluations=tuple(evaluations))

    async def _persist(self, operation: Awaitable[None]) -> None:
        try:
            await operation
        except Exception as exc:
            raise BenchmarkPersistenceError("PERSISTENCE_FAILED") from exc

    async def _execute_case(
        self,
        case: BenchmarkCase,
        options: RunOptions,
        run_id: str,
        reference: GoldenReference | None,
    ) -> tuple[ExecutionTrace, EvaluationResult]:
        trace_id = uuid4().hex
        started = datetime.now(UTC)
        tick = perf_counter()
        steps: list[TraceStep] = []
        decisions: list[RoutingDecision] = []
        feedback: list[RoutingFeedback] = []
        response: ModelResponse | None = None
        # Cascade may try a higher-tier model after a successful lower-tier answer.
        # Keep that answer available if the escalation ultimately fails.
        last_successful_response: ModelResponse | None = None
        model_calls_used = 0
        last_error: ErrorInfo | None = None
        status = ExecutionStatus.FAILED

        def add_step(kind: TraceStepKind, **fields: Any) -> None:
            steps.append(TraceStep(sequence=len(steps), kind=kind, **fields))

        try:
            permitted = self._privacy.sanitize(case.input_text, purpose="routing")
        except Exception:
            # Privacy adapter exceptions fail closed; never forward caller content.
            permitted = None
        try:
            if (
                permitted is None
                or permitted.status is SanitizationStatus.BLOCKED
                or permitted.text is None
            ):
                status = ExecutionStatus.BLOCKED
                last_error = _error(ErrorCategory.PRIVACY, "PRIVACY_BLOCKED")
                add_step(TraceStepKind.PRIVACY_CHECK, error=last_error)
            else:
                # Original caller-provided metadata may contain labels/PII; never route it.
                safe_case = case.model_copy(update={"input_text": permitted.text, "metadata": {}})
                add_step(TraceStepKind.PRIVACY_CHECK, safe_metadata={"status": "ALLOWED"})
                for stage in range(options.max_model_calls_per_case):
                    # The cap applies to all provider attempts, including retries.
                    remaining_calls = options.max_model_calls_per_case - model_calls_used
                    if remaining_calls <= 0:
                        if options.strategy is StrategyId.CASCADE and feedback:
                            # Defense in depth for budgets that bypass RunOptions
                            # validation: never skip an escalation silently.
                            try:
                                wanted = await asyncio.wait_for(
                                    self._router.decide(
                                        safe_case,
                                        RoutingContext(
                                            current_stage=stage,
                                            remaining_model_calls=0,
                                            prior_decisions=tuple(decisions),
                                            prior_attempts=tuple(feedback),
                                        ),
                                    ),
                                    timeout=options.timeout_seconds,
                                )
                            except Exception:
                                wanted = None
                            if wanted is not None and wanted.action is RoutingAction.INVOKE_MODEL:
                                add_step(
                                    TraceStepKind.ERROR,
                                    error=_error(
                                        ErrorCategory.ROUTING, ESCALATION_SKIPPED_CALL_BUDGET
                                    ),
                                )
                        break
                    context = RoutingContext(
                        current_stage=stage,
                        remaining_model_calls=remaining_calls,
                        prior_decisions=tuple(decisions),
                        prior_attempts=tuple(feedback),
                    )
                    try:
                        decision_started = perf_counter()
                        decision = await asyncio.wait_for(
                            self._router.decide(safe_case, context),
                            timeout=options.timeout_seconds,
                        )
                        if decision.strategy != options.strategy or decision.stage_index != stage:
                            raise BenchmarkInputError("Router returned inconsistent strategy/stage")
                        if (
                            decision.action is RoutingAction.INVOKE_MODEL
                            and decision.selected_model not in options.allowed_models
                        ):
                            raise BenchmarkInputError("Router selected an unapproved model")
                    except Exception:
                        last_error = _error(ErrorCategory.ROUTING, "ROUTING_FAILED")
                        add_step(TraceStepKind.ERROR, error=last_error)
                        break
                    # Never persist unvalidated router-generated explanatory prose.
                    safe_decision = decision.model_copy(
                        update={
                            "decision_time_ms": (perf_counter() - decision_started) * 1000,
                            "safe_reason": None,
                            "reason_code": _safe_code(
                                decision.reason_code, "ROUTER_REASON_REDACTED"
                            ),
                        }
                    )
                    decisions.append(safe_decision)
                    add_step(TraceStepKind.ROUTING_DECISION, decision=safe_decision)
                    if decision.action is RoutingAction.FINISH:
                        break
                    # The second gate covers transformed/recomposed provider prompts.
                    try:
                        sanitized_prompt = self._privacy.sanitize(
                            safe_case.input_text, purpose="provider"
                        )
                    except Exception:
                        sanitized_prompt = None
                    if (
                        sanitized_prompt is None
                        or sanitized_prompt.status is not SanitizationStatus.ALLOWED
                        or sanitized_prompt.text is None
                    ):
                        status = ExecutionStatus.BLOCKED
                        last_error = _error(ErrorCategory.PRIVACY, "PROVIDER_PRIVACY_BLOCKED")
                        add_step(TraceStepKind.PRIVACY_CHECK, error=last_error)
                        break
                    assert decision.selected_model is not None
                    response, attempts_used = await self._invoke_with_retries(
                        options,
                        run_id,
                        trace_id,
                        case.case_id,
                        decision.selected_model,
                        sanitized_prompt.text,
                        add_step,
                        remaining_call_budget=remaining_calls,
                    )
                    model_calls_used += attempts_used
                    feedback.append(
                        RoutingFeedback(
                            attempt_id=response.attempt_id,
                            model=response.model,
                            status=response.status,
                            error_code=_safe_code(response.error.code, "PROVIDER_ERROR_REDACTED")
                            if response.error
                            else None,
                        )
                    )
                    if response.status is CallStatus.SUCCESS:
                        last_successful_response = response
                        # A non-Cascade strategy must make exactly one model invocation.
                        if options.strategy is not StrategyId.CASCADE:
                            break
                    else:
                        last_error = (
                            _error(
                                response.error.category,
                                _safe_code(response.error.code, "PROVIDER_ERROR_REDACTED"),
                                response.error.retryable,
                            )
                            if response.error
                            else _error(ErrorCategory.PROVIDER, "PROVIDER_FAILURE")
                        )
                        # No implicit fallback for baselines; Cascade may explicitly escalate.
                        if options.strategy is not StrategyId.CASCADE:
                            break
                else:
                    # Reaching a configured bound is normal; no unbounded routing loop.
                    pass
        except Exception:
            # Fail safely but distinguish software faults from policy denials.
            status = ExecutionStatus.FAILED
            last_error = _error(ErrorCategory.UNKNOWN, "EXECUTION_FAILED")
            add_step(TraceStepKind.ERROR, error=last_error)

        if last_successful_response is not None:
            if (
                status is not ExecutionStatus.BLOCKED
                and response is not None
                and response is not last_successful_response
            ):
                # Review #12 F2 (decision D-M02-F2): keep the lower-tier answer the
                # user would realistically receive, but make the failed escalation
                # visible and count the case as degraded, not as a clean success.
                add_step(
                    TraceStepKind.ERROR,
                    error=_error(ErrorCategory.PROVIDER, ESCALATION_FAILED_RETAINED_LOWER_TIER),
                )
            response = last_successful_response
        succeeded = response is not None and response.status is CallStatus.SUCCESS
        if status is not ExecutionStatus.BLOCKED:
            status = ExecutionStatus.COMPLETED if succeeded else ExecutionStatus.FAILED
        if not succeeded and last_error is None:
            last_error = _error(ErrorCategory.ROUTING, "NO_SUCCESSFUL_PROVIDER_CALL")

        if status is ExecutionStatus.BLOCKED or not succeeded:
            # A blocked prompt must not reach a potentially remote judge.
            evaluation = EvaluationResult(
                run_id=run_id,
                case_id=case.case_id,
                trace_id=trace_id,
                quality_scores={},
                critical_errors=(),
                evaluation_method="not_run",
                evaluator_version="not_run",
                status=EvaluationStatus.NOT_EVALUATED,
            )
        else:
            try:
                evaluation = await asyncio.wait_for(
                    self._evaluator.evaluate(
                        safe_case,
                        response,
                        reference,
                        run_id=run_id,
                        trace_id=trace_id,
                    ),
                    timeout=options.timeout_seconds,
                )
                if (
                    evaluation.run_id != run_id
                    or evaluation.trace_id != trace_id
                    or evaluation.case_id != case.case_id
                ):
                    raise BenchmarkInputError("Evaluator returned mismatched identifiers")
                if (
                    any(not _is_safe_code(key) for key in evaluation.quality_scores)
                    or any(not _is_safe_code(key) for key in evaluation.critical_errors)
                    or not _is_safe_code(evaluation.evaluation_method)
                    or not _is_safe_code(evaluation.evaluator_version)
                ):
                    raise BenchmarkInputError("Evaluator emitted unsafe codes")
                # Until Member 04 supplies a vetted sanitizer, do not persist prose notes.
                evaluation = evaluation.model_copy(
                    update={
                        "safe_notes": (),
                        "review_status": None,
                    }
                )
            except Exception:
                evaluation = EvaluationResult(
                    run_id=run_id,
                    case_id=case.case_id,
                    trace_id=trace_id,
                    quality_scores={},
                    critical_errors=(),
                    evaluation_method="unavailable",
                    evaluator_version="unknown",
                    status=EvaluationStatus.FAILED,
                )
                add_step(
                    TraceStepKind.ERROR, error=_error(ErrorCategory.EVALUATION, "EVALUATION_FAILED")
                )
                if status is ExecutionStatus.COMPLETED:
                    status = ExecutionStatus.PARTIAL

        add_step(TraceStepKind.EVALUATION, safe_metadata={"status": evaluation.status.value})
        trace = ExecutionTrace(
            trace_id=trace_id,
            run_id=run_id,
            case_id=case.case_id,
            strategy=options.strategy,
            steps=tuple(steps),
            status=status,
            started_at=started,
            ended_at=datetime.now(UTC),
            total_latency_ms=(perf_counter() - tick) * 1000,
            error=(
                last_error if status in (ExecutionStatus.FAILED, ExecutionStatus.BLOCKED) else None
            ),
        )
        return trace, evaluation

    async def _invoke_with_retries(
        self,
        options: RunOptions,
        run_id: str,
        trace_id: str,
        case_id: str,
        model: str,
        prompt: str,
        add_step: Callable[..., None],
        *,
        remaining_call_budget: int,
    ) -> tuple[ModelResponse, int]:
        if remaining_call_budget <= 0:
            raise BenchmarkInputError("MODEL_CALL_BUDGET_EXHAUSTED")
        result: ModelResponse | None = None
        attempt_limit = min(options.max_retries + 1, remaining_call_budget)
        for index in range(attempt_limit):
            attempt_id = uuid4().hex
            request = ModelRequest(
                run_id=run_id,
                trace_id=trace_id,
                case_id=case_id,
                attempt_id=attempt_id,
                provider=options.provider,
                model=model,
                prompt=prompt,
                timeout_seconds=options.timeout_seconds,
                temperature=options.temperature,
                max_output_tokens=options.max_output_tokens,
            )
            started = perf_counter()
            try:
                result = await asyncio.wait_for(
                    self._provider.generate(request), timeout=options.timeout_seconds
                )
                if (
                    result.attempt_id != attempt_id
                    or result.model != model
                    or result.provider != options.provider
                ):
                    raise BenchmarkInputError("Provider returned mismatched identifiers")
            except TimeoutError:
                result = _failure(
                    options.provider,
                    model,
                    attempt_id,
                    ErrorCategory.PROVIDER_TIMEOUT,
                    "TIMEOUT",
                    retryable=True,
                )
            except Exception:
                result = _failure(
                    options.provider, model, attempt_id, ErrorCategory.PROVIDER, "PROVIDER_FAILURE"
                )
            add_step(
                TraceStepKind.PROVIDER_ATTEMPT,
                attempt_id=attempt_id,
                model=model,
                provider=options.provider,
                call_status=result.status,
                metric_origin=result.metric_origin,
                usage=result.usage,
                elapsed_ms=(perf_counter() - started) * 1000,
                error=_error(
                    result.error.category,
                    _safe_code(result.error.code, "PROVIDER_ERROR_REDACTED"),
                    result.error.retryable,
                )
                if result.error is not None
                else None,
            )
            if result.status is CallStatus.SUCCESS or not (
                result.error is not None and result.error.retryable and index + 1 < attempt_limit
            ):
                return result, index + 1
            # Strictly positive exponential equal-jitter wait prevents retry
            # bursts; Retry-After handling requires a separate provider policy.
            await self._sleep(_retry_backoff_seconds(index, random.random()))
        assert result is not None
        return result, attempt_limit


# Trace error codes for Cascade outcomes that would otherwise be silent (review #12).
ESCALATION_FAILED_RETAINED_LOWER_TIER = "ESCALATION_FAILED_RETAINED_LOWER_TIER"
ESCALATION_SKIPPED_CALL_BUDGET = "ESCALATION_SKIPPED_CALL_BUDGET"


def is_degraded_trace(trace: ExecutionTrace) -> bool:
    """COMPLETED only because a lower-tier answer was kept after a failed escalation."""
    return trace.status is ExecutionStatus.COMPLETED and any(
        step.kind is TraceStepKind.ERROR
        and step.error is not None
        and step.error.code == ESCALATION_FAILED_RETAINED_LOWER_TIER
        for step in trace.steps
    )


_SAFE_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_./-]{1,80}$")


def _is_safe_code(value: str) -> bool:
    return bool(_SAFE_CODE_PATTERN.fullmatch(value))


def _safe_code(value: str, fallback: str) -> str:
    return value if _is_safe_code(value) else fallback


def _error(category: ErrorCategory, code: str, retryable: bool = False) -> ErrorInfo:
    return ErrorInfo(category=category, code=code, retryable=retryable)


def _failure(
    provider: str,
    model: str,
    attempt_id: str,
    category: ErrorCategory,
    code: str,
    *,
    retryable: bool = False,
) -> ModelResponse:
    status = CallStatus.TIMEOUT if category is ErrorCategory.PROVIDER_TIMEOUT else CallStatus.FAILED
    return ModelResponse(
        provider=provider,
        model=model,
        attempt_id=attempt_id,
        status=status,
        metric_origin=MetricOrigin.SIMULATED,
        error=_error(category, code, retryable),
    )


def _summarize(
    traces: Sequence[ExecutionTrace], evaluations: Sequence[EvaluationResult]
) -> StrategyMetrics:
    completed = sum(t.status is ExecutionStatus.COMPLETED for t in traces)
    degraded = sum(is_degraded_trace(t) for t in traces)
    blocked = sum(t.status is ExecutionStatus.BLOCKED for t in traces)
    failed = len(traces) - completed - blocked
    evaluated = [e for e in evaluations if e.status is EvaluationStatus.COMPLETED]
    critical = sum(bool(e.critical_errors) for e in evaluated)
    latencies = [t.total_latency_ms for t in traces if t.total_latency_ms is not None]
    # Only available numerical scores are aggregated; missing metrics remain unknown.
    metric_names = {name for item in evaluated for name in item.quality_scores}
    aggregated_scores: dict[str, float | None] = {}
    for name in sorted(metric_names):
        samples = [
            score for item in evaluated if (score := item.quality_scores.get(name)) is not None
        ]
        aggregated_scores[name] = statistics.mean(samples) if samples else None
    # Do not extrapolate P95 from fewer than 20 data points.
    return StrategyMetrics(
        case_count=len(traces),
        successful_cases=completed,
        degraded_cases=degraded,
        failed_cases=failed,
        blocked_cases=blocked,
        evaluated_cases=len(evaluated),
        critical_error_cases=critical,
        quality_scores=aggregated_scores,
        latency_p50_ms=statistics.median(latencies) if latencies else None,
        latency_p95_ms=(
            sorted(latencies)[int((len(latencies) - 1) * 0.95)] if len(latencies) >= 20 else None
        ),
        latency_samples=len(latencies),
        # Cost remains unknown until priced observations are wired into the report.
    )
