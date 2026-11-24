"""HTTP 429 Retry-After: safe parsing, provider propagation and bounded orchestration.

All offline: HTTPX MockTransport and in-memory fakes; no network, no real keys.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic import ValidationError

from llm_routing_eval.adapters.providers.groq import GroqConfig, GroqHttpResult, GroqProvider
from llm_routing_eval.adapters.providers.httpx_transport import (
    HttpxGroqTransport,
    HttpxNvidiaTransport,
    managed_httpx_client,
)
from llm_routing_eval.adapters.providers.nvidia_build import (
    NvidiaBuildConfig,
    NvidiaBuildProvider,
)
from llm_routing_eval.adapters.providers.retry_after import checked_hint, parse_retry_after
from llm_routing_eval.application import BenchmarkOrchestrator, RunOptions
from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
    TraceStepKind,
)
from llm_routing_eval.demo.fakes import (
    DEMO_PROMPTS,
    AllowlistedDemoPrivacy,
    DemoAlwaysLargeRouter,
    DemoFixtureEvaluator,
    InMemoryDemoRepository,
)


class AllowAllForTests:
    """Test-only OutboundAuthorization. Never use outside offline tests."""

    def authorize(self, request: object) -> bool:
        return True


LARGE = "openai/gpt-oss-120b"
SECRET = "IN_MEMORY_FAKE_CREDENTIAL"


# ----------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0.0),
        ("7", 7.0),
        (" 2.5 ", 2.5),
        ("1.125", 1.125),
        ("999999", 86_400.0),  # saturated, never dropped: still exceeds any budget
    ],
)
def test_numeric_retry_after_is_parsed(raw: str, expected: float) -> None:
    assert parse_retry_after({"Retry-After": raw}) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "-1", "1e3", "nan", "inf", "01", "1.2345", "Wed, 21 Oct 2026 07:28:00 GMT", "5s"],
)
def test_invalid_or_date_retry_after_is_ignored(raw: str) -> None:
    assert parse_retry_after({"Retry-After": raw}) is None


def test_lowercase_header_and_missing_headers() -> None:
    assert parse_retry_after({"retry-after": "3"}) == 3.0
    assert parse_retry_after({}) is None
    assert parse_retry_after(None) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 90_000.0, True])
def test_checked_hint_rejects_unsafe_values(value: float) -> None:
    assert checked_hint(value) is None


def test_error_info_hint_only_for_retryable_rate_limits() -> None:
    ErrorInfo(category=ErrorCategory.RATE_LIMIT, code="RL", retryable=True, retry_after_seconds=1.0)
    for category, retryable in (
        (ErrorCategory.PROVIDER, True),
        (ErrorCategory.RATE_LIMIT, False),
    ):
        with pytest.raises(ValidationError, match="RETRY_HINT_REQUIRES_RETRYABLE_RATE_LIMIT"):
            ErrorInfo(category=category, code="X", retryable=retryable, retry_after_seconds=1.0)
    with pytest.raises(ValidationError):
        ErrorInfo(
            category=ErrorCategory.RATE_LIMIT,
            code="RL",
            retryable=True,
            retry_after_seconds=float("inf"),
        )


# ----------------------------------------------------------- providers


def _request(provider: str) -> ModelRequest:
    return ModelRequest(
        run_id="r",
        case_id="c",
        trace_id="t",
        attempt_id="a",
        provider=provider,
        model=LARGE,
        prompt=sorted(DEMO_PROMPTS)[0],
        timeout_seconds=1.0,
    )


@pytest.mark.parametrize(
    ("provider_id", "code"),
    [("groq", "GROQ_RATE_LIMITED"), ("nvidia_build", "NVIDIA_RATE_LIMITED")],
)
@pytest.mark.parametrize(("header", "hint"), [("4", 4.0), ("garbage", None), (None, None)])
def test_providers_propagate_only_safe_hints(
    monkeypatch: pytest.MonkeyPatch,
    provider_id: str,
    code: str,
    header: str | None,
    hint: float | None,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setenv("NVIDIA_API_KEY", SECRET)

    async def handler(_: httpx.Request) -> httpx.Response:
        headers = {"Retry-After": header} if header is not None else {}
        return httpx.Response(429, headers=headers, text="PRIVATE RATE LIMIT BODY")

    async def run() -> ModelResponse:
        async with managed_httpx_client(mock_transport=httpx.MockTransport(handler)) as client:
            provider: GroqProvider | NvidiaBuildProvider
            if provider_id == "groq":
                provider = GroqProvider(
                    GroqConfig(
                        allowed_models=frozenset({LARGE}), live_enabled=True, privacy_approved=True
                    ),
                    transport=HttpxGroqTransport(client),
                    outbound_authorization=AllowAllForTests(),
                )
            else:
                provider = NvidiaBuildProvider(
                    NvidiaBuildConfig(
                        allowed_models=frozenset({LARGE}), live_enabled=True, privacy_approved=True
                    ),
                    transport=HttpxNvidiaTransport(client),
                    outbound_authorization=AllowAllForTests(),
                )
            return await provider.generate(_request(provider_id))

    response = asyncio.run(run())
    assert response.status is CallStatus.RATE_LIMITED
    assert response.error is not None
    assert (response.error.code, response.error.retryable) == (code, True)
    assert response.error.retry_after_seconds == hint
    assert "PRIVATE" not in repr(response)


def test_provider_drops_unsafe_hint_from_injected_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", SECRET)

    class BadHintTransport:
        async def post_completion(self, payload, *, api_key, timeout_seconds):  # type: ignore[no-untyped-def]
            return GroqHttpResult(status_code=429, body=b"", retry_after_seconds=float("nan"))

    provider = GroqProvider(
        GroqConfig(allowed_models=frozenset({LARGE}), live_enabled=True, privacy_approved=True),
        transport=BadHintTransport(),
        outbound_authorization=AllowAllForTests(),
    )
    response = asyncio.run(provider.generate(_request("groq")))
    assert response.error is not None and response.error.retry_after_seconds is None


# ------------------------------------------------------------ orchestrator


class RateLimitedThenOk:
    """First call: 429 with a hint; then success."""

    def __init__(self, hint: float | None, failures: int = 1) -> None:
        self.hint = hint
        self.failures = failures
        self.calls = 0

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self.calls <= self.failures:
            return ModelResponse(
                provider=request.provider,
                model=request.model,
                attempt_id=request.attempt_id,
                status=CallStatus.RATE_LIMITED,
                metric_origin=MetricOrigin.SIMULATED,
                error=ErrorInfo(
                    category=ErrorCategory.RATE_LIMIT,
                    code="RL",
                    retryable=True,
                    retry_after_seconds=self.hint,
                ),
            )
        return ModelResponse(
            provider=request.provider,
            model=request.model,
            attempt_id=request.attempt_id,
            status=CallStatus.SUCCESS,
            metric_origin=MetricOrigin.SIMULATED,
            content="ok",
        )


def _run(provider: RateLimitedThenOk, **options: object) -> tuple[list[float], list[str]]:
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    engine = BenchmarkOrchestrator(
        router=DemoAlwaysLargeRouter(LARGE),
        provider=provider,
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
    result = asyncio.run(
        engine.run([case], RunOptions(allowed_models=(LARGE,), max_retries=1, **options))
    )
    statuses = [
        str(step.call_status)
        for step in result.traces[0].steps
        if step.kind is TraceStepKind.PROVIDER_ATTEMPT
    ]
    return waits, statuses


def test_orchestrator_waits_at_least_the_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_routing_eval.application.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator.random, "random", lambda: 1.0)  # max backoff 0.5 s
    waits, statuses = _run(RateLimitedThenOk(hint=3.0))
    assert waits == [3.0]
    assert statuses == ["RATE_LIMITED", "SUCCESS"]


def test_backoff_wins_when_hint_is_shorter(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_routing_eval.application.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator.random, "random", lambda: 1.0)
    waits, _ = _run(RateLimitedThenOk(hint=0.0))
    assert waits == [0.5]


def test_hint_beyond_budget_stops_retrying_instead_of_retrying_early() -> None:
    provider = RateLimitedThenOk(hint=30.0)
    waits, statuses = _run(provider, max_retry_wait_seconds=10.0)
    assert waits == []
    assert statuses == ["RATE_LIMITED"]
    assert provider.calls == 1


def test_missing_hint_uses_floored_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_routing_eval.application.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator.random, "random", lambda: 0.0)
    waits, statuses = _run(RateLimitedThenOk(hint=None))
    assert waits == [0.25]
    assert statuses == ["RATE_LIMITED", "SUCCESS"]


def test_retry_wait_budget_is_validated() -> None:
    for bad in (-1.0, 61.0, float("nan")):
        with pytest.raises(ValidationError):
            RunOptions(max_retry_wait_seconds=bad)
