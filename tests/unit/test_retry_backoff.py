"""Review #12 F9: retry backoff must never be (almost) zero."""

from __future__ import annotations

import asyncio
import math

import pytest

from llm_routing_eval.application.orchestrator import (
    RETRY_BACKOFF_MAX_SECONDS,
    BenchmarkOrchestrator,
    RunOptions,
    retry_backoff_seconds,
)
from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
)
from llm_routing_eval.demo.fakes import (
    DEMO_PROMPTS,
    AllowlistedDemoPrivacy,
    DemoAlwaysLargeRouter,
    DemoFixtureEvaluator,
    InMemoryDemoRepository,
)


@pytest.mark.parametrize(
    ("index", "low", "high"),
    [(0, 0.25, 0.5), (1, 0.5, 1.0), (2, 1.0, 2.0), (3, 1.0, 2.0), (50, 1.0, 2.0)],
)
def test_backoff_window_per_retry(index: int, low: float, high: float) -> None:
    assert retry_backoff_seconds(index, 0.0) == pytest.approx(low)
    assert retry_backoff_seconds(index, 1.0) == pytest.approx(high)
    assert low <= retry_backoff_seconds(index, 0.5) <= high


def test_backoff_has_a_floor_even_with_zero_jitter() -> None:
    # The old formula min(2.0, 2**i * random()) returned 0.0 for random() == 0.
    assert min(retry_backoff_seconds(i, 0.0) for i in range(10)) >= 0.25


def test_backoff_is_monotonic_and_capped() -> None:
    floors = [retry_backoff_seconds(i, 0.0) for i in range(10)]
    assert floors == sorted(floors)
    assert all(
        retry_backoff_seconds(i, j) <= RETRY_BACKOFF_MAX_SECONDS
        for i in range(10)
        for j in (0.0, 0.3, 1.0)
    )
    assert math.isfinite(retry_backoff_seconds(10_000, 1.0))


@pytest.mark.parametrize(("index", "jitter"), [(-1, 0.5), (True, 0.5), (0, -0.1), (0, 1.1)])
def test_backoff_rejects_invalid_inputs(index: int, jitter: float) -> None:
    with pytest.raises(ValueError):
        retry_backoff_seconds(index, jitter)


def test_orchestrator_waits_at_least_the_floor_between_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_routing_eval.application.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator.random, "random", lambda: 0.0)  # worst case
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    class AlwaysRetryable:
        async def generate(self, request: ModelRequest) -> ModelResponse:
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

    engine = BenchmarkOrchestrator(
        router=DemoAlwaysLargeRouter("large"),
        provider=AlwaysRetryable(),
        evaluator=DemoFixtureEvaluator(),
        privacy=AllowlistedDemoPrivacy(),
        repository=InMemoryDemoRepository(),
        sleep=record,
    )
    case = BenchmarkCase(
        case_id="c",
        input_text=sorted(DEMO_PROMPTS)[0],
        domain="turkish_banking",
        scenario_type="synthetic_test",
    )
    asyncio.run(
        engine.run(
            [case],
            RunOptions(allowed_models=("large",), max_model_calls_per_case=4, max_retries=3),
        )
    )
    assert waits == pytest.approx([0.25, 0.5, 1.0])  # 3 retries, never ~0
