"""F9: retry waits have a positive floor and respect existing call budgets.

No outbound HTTP, credentials or customer data: provider and clock are fakes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from llm_routing_eval.application import BenchmarkOrchestrator, RunOptions
from llm_routing_eval.application.orchestrator import (
    RETRY_BACKOFF_POLICY_VERSION,
    _retry_backoff_seconds,
)
from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    ExecutionStatus,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
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

MODEL = "large"


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="f9-synthetic-001",
        input_text=sorted(DEMO_PROMPTS)[0],
        domain="turkish_banking",
        scenario_type="synthetic_test",
        metadata={"synthetic": True},
    )


class RetrySequence:
    def __init__(self, *, fail_count: int, retryable: bool = True) -> None:
        self.fail_count = fail_count
        self.retryable = retryable
        self.attempts: list[str] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.attempts.append(request.attempt_id)
        if len(self.attempts) > self.fail_count:
            return await DemoFakeProvider().generate(request)
        return ModelResponse(
            provider=request.provider,
            model=request.model,
            attempt_id=request.attempt_id,
            status=CallStatus.RATE_LIMITED,
            metric_origin=MetricOrigin.SIMULATED,
            error=ErrorInfo(
                category=ErrorCategory.RATE_LIMIT,
                code="SYNTHETIC_RATE_LIMIT",
                retryable=self.retryable,
            ),
        )


@pytest.mark.parametrize(
    ("retry_index", "minimum_seconds", "maximum_seconds"),
    [
        (0, 0.125, 0.25),
        (1, 0.25, 0.5),
        (2, 0.5, 1.0),
        (3, 1.0, 2.0),
        (4, 1.0, 2.0),
    ],
)
def test_equal_jitter_has_strict_positive_floor_and_cap(
    retry_index: int, minimum_seconds: float, maximum_seconds: float
) -> None:
    assert _retry_backoff_seconds(retry_index, 0.0) == pytest.approx(minimum_seconds)
    assert _retry_backoff_seconds(retry_index, 1.0) == pytest.approx(maximum_seconds)
    assert minimum_seconds <= _retry_backoff_seconds(retry_index, 0.5) <= maximum_seconds


def test_zero_jitter_still_waits_between_retryable_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llm_routing_eval.application.orchestrator.random.random", lambda: 0.0)
    observed_delays: list[float] = []
    provider = RetrySequence(fail_count=3)

    async def capture_sleep(seconds: float) -> None:
        observed_delays.append(seconds)

    engine = BenchmarkOrchestrator(
        router=DemoAlwaysLargeRouter(MODEL),
        provider=provider,
        evaluator=DemoFixtureEvaluator(),
        privacy=AllowlistedDemoPrivacy(),
        repository=InMemoryDemoRepository(),
        sleep=capture_sleep,
    )
    opts = RunOptions(
        allowed_models=(MODEL,),
        max_model_calls_per_case=4,
        max_retries=3,
    )
    result = asyncio.run(engine.run((_case(),), opts))

    attempts = [
        item
        for item in result.traces[0].steps
        if item.kind is TraceStepKind.PROVIDER_ATTEMPT
    ]
    assert observed_delays == pytest.approx([0.125, 0.25, 0.5])
    assert len(attempts) == len(provider.attempts) == len(set(provider.attempts)) == 4
    assert [attempt.call_status for attempt in attempts] == [
        CallStatus.RATE_LIMITED,
        CallStatus.RATE_LIMITED,
        CallStatus.RATE_LIMITED,
        CallStatus.SUCCESS,
    ]
    assert result.traces[0].status is ExecutionStatus.COMPLETED
    assert result.report.provenance["retry_backoff_policy"] == RETRY_BACKOFF_POLICY_VERSION


@pytest.mark.parametrize(
    ("fail_count", "retryable", "expected_calls"),
    [(0, True, 1), (1, False, 1), (5, True, 2)],
)
def test_no_extra_sleep_when_success_not_retryable_or_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    fail_count: int,
    retryable: bool,
    expected_calls: int,
) -> None:
    monkeypatch.setattr("llm_routing_eval.application.orchestrator.random.random", lambda: 0.0)
    delays: list[float] = []
    provider = RetrySequence(fail_count=fail_count, retryable=retryable)

    async def capture_sleep(seconds: float) -> None:
        delays.append(seconds)

    engine = BenchmarkOrchestrator(
        router=DemoAlwaysLargeRouter(MODEL),
        provider=provider,
        evaluator=DemoFixtureEvaluator(),
        privacy=AllowlistedDemoPrivacy(),
        repository=InMemoryDemoRepository(),
        sleep=capture_sleep,
    )
    result = asyncio.run(
        engine.run(
            (_case(),),
            RunOptions(allowed_models=(MODEL,), max_retries=3, max_model_calls_per_case=2),
        )
    )
    assert len(provider.attempts) == expected_calls
    assert len(delays) == max(0, expected_calls - 1)
    assert all(wait >= 0.125 for wait in delays)
    assert len(
        [step for step in result.traces[0].steps if step.kind is TraceStepKind.PROVIDER_ATTEMPT]
    ) == expected_calls


def test_cancelled_backoff_does_not_generate_an_extra_provider_attempt() -> None:
    provider = RetrySequence(fail_count=4)

    async def cancel_sleep(seconds: float) -> None:
        assert seconds > 0
        raise asyncio.CancelledError

    engine = BenchmarkOrchestrator(
        router=DemoAlwaysLargeRouter(MODEL),
        provider=provider,
        evaluator=DemoFixtureEvaluator(),
        privacy=AllowlistedDemoPrivacy(),
        repository=InMemoryDemoRepository(),
        sleep=cancel_sleep,
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            engine.run(
                (_case(),),
                RunOptions(
                    allowed_models=(MODEL,), max_retries=1, max_model_calls_per_case=2
                ),
            )
        )
    assert len(provider.attempts) == 1


def test_retry_policy_version_is_part_of_fingerprint_only_when_retries_enabled() -> None:
    no_retry = RunOptions(allowed_models=(MODEL,), max_retries=0)
    with_retry = RunOptions(allowed_models=(MODEL,), max_retries=1)
    def legacy_fingerprint(options: RunOptions) -> str:
        return "sha256:" + hashlib.sha256(
            json.dumps(
                options.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    assert no_retry.fingerprint() == legacy_fingerprint(no_retry)
    assert with_retry.fingerprint() != legacy_fingerprint(with_retry)
