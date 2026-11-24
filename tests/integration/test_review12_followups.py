"""Regression tests for review #12 follow-ups F1–F3 (issues #17, #16, #15).

All offline: HTTPX MockTransport and in-memory fakes; no network, no real keys.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from llm_routing_eval.adapters.providers.groq import GroqConfig, GroqProvider
from llm_routing_eval.adapters.providers.httpx_transport import (
    HttpxGroqTransport,
    HttpxNvidiaTransport,
    managed_httpx_client,
)
from llm_routing_eval.adapters.providers.nvidia_build import (
    NvidiaBuildConfig,
    NvidiaBuildProvider,
)
from llm_routing_eval.application import BenchmarkOrchestrator, RunOptions
from llm_routing_eval.cli.app import app
from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    ExecutionStatus,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
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
from llm_routing_eval.routing import CascadeRouter, ModelPair

SMALL = "openai/gpt-oss-20b"
LARGE = "openai/gpt-oss-120b"
SECRET = "IN_MEMORY_FAKE_CREDENTIAL"


def demo_case(scenario: str = "account_general") -> BenchmarkCase:
    return BenchmarkCase(
        case_id="review12-case",
        input_text=sorted(DEMO_PROMPTS)[0],
        domain="turkish_banking",
        scenario_type=scenario,
    )


def request(provider: str, model: str = LARGE) -> ModelRequest:
    return ModelRequest(
        run_id="r",
        case_id="c",
        trace_id="t",
        attempt_id="a",
        provider=provider,
        model=model,
        prompt=sorted(DEMO_PROMPTS)[0],
        timeout_seconds=1.0,
        max_output_tokens=16,
    )


async def no_sleep(_: float) -> None:
    return None


def engine(router: object, provider: object) -> BenchmarkOrchestrator:
    return BenchmarkOrchestrator(
        router=router,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
        evaluator=DemoFixtureEvaluator(),
        privacy=AllowlistedDemoPrivacy(),
        repository=InMemoryDemoRepository(),
        sleep=no_sleep,
    )


# --------------------------------------------------------------------- F1 (#17)


def test_f1_providers_have_no_implicit_thread_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setenv("NVIDIA_API_KEY", SECRET)
    groq = GroqProvider(
        GroqConfig(allowed_models=frozenset({LARGE}), live_enabled=True, privacy_approved=True)
    )
    nvidia = NvidiaBuildProvider(
        NvidiaBuildConfig(
            allowed_models=frozenset({LARGE}), live_enabled=True, privacy_approved=True
        )
    )
    for provider, provider_id, code in (
        (groq, "groq", "GROQ_TRANSPORT_NOT_CONFIGURED"),
        (nvidia, "nvidia_build", "NVIDIA_TRANSPORT_NOT_CONFIGURED"),
    ):
        response = asyncio.run(provider.generate(request(provider_id)))
        assert response.status is CallStatus.FAILED
        assert response.error is not None and response.error.code == code
        assert not response.error.retryable


def test_f1_timeout_leaves_no_call_in_flight_and_retries_never_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #17 repro: transport blocks 0.6 s, timeout 0.2 s, max_retries=1."""
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    in_flight = 0
    max_concurrent = 0
    started = 0

    async def slow(_: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_concurrent, started
        started += 1
        in_flight += 1
        max_concurrent = max(max_concurrent, in_flight)
        try:
            await asyncio.sleep(0.6)
            return httpx.Response(500)
        finally:
            in_flight -= 1

    async def run() -> tuple[int, list[CallStatus | None]]:
        async with managed_httpx_client(mock_transport=httpx.MockTransport(slow)) as client:
            provider = GroqProvider(
                GroqConfig(
                    allowed_models=frozenset({LARGE}), live_enabled=True, privacy_approved=True
                ),
                transport=HttpxGroqTransport(client),
            )
            result = await engine(DemoAlwaysLargeRouter(LARGE), provider).run(
                [demo_case()],
                RunOptions(
                    provider="groq",
                    allowed_models=(LARGE,),
                    max_retries=1,
                    timeout_seconds=0.2,
                ),
            )
            leaked = in_flight  # checked as soon as run() returns, pool still open
        statuses = [
            step.call_status
            for step in result.traces[0].steps
            if step.kind is TraceStepKind.PROVIDER_ATTEMPT
        ]
        return leaked, statuses

    leaked, statuses = asyncio.run(run())
    assert leaked == 0
    assert max_concurrent == 1
    assert started == 2
    assert statuses == [CallStatus.TIMEOUT, CallStatus.TIMEOUT]


@pytest.mark.parametrize("provider_id", ["groq", "nvidia_build"])
def test_f1_httpx_transports_keep_stdlib_safety_properties(
    monkeypatch: pytest.MonkeyPatch, provider_id: str
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setenv("NVIDIA_API_KEY", SECRET)
    seen: list[httpx.Request] = []
    body = {
        "id": "req-1",
        "model": LARGE,
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }

    async def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if len(seen) == 1:
            return httpx.Response(200, json=body)
        if len(seen) == 2:
            return httpx.Response(302, headers={"Location": "https://attacker.invalid/"})
        return httpx.Response(500, text="PRIVATE ERROR BODY")

    async def run() -> list[ModelResponse]:
        async with managed_httpx_client(mock_transport=httpx.MockTransport(handler)) as client:
            provider: GroqProvider | NvidiaBuildProvider
            if provider_id == "groq":
                provider = GroqProvider(
                    GroqConfig(
                        allowed_models=frozenset({LARGE}), live_enabled=True, privacy_approved=True
                    ),
                    transport=HttpxGroqTransport(client),
                )
            else:
                provider = NvidiaBuildProvider(
                    NvidiaBuildConfig(
                        allowed_models=frozenset({LARGE}), live_enabled=True, privacy_approved=True
                    ),
                    transport=HttpxNvidiaTransport(client),
                )
            return [await provider.generate(request(provider_id)) for _ in range(3)]

    ok, redirect, server_error = asyncio.run(run())
    assert ok.status is CallStatus.SUCCESS and ok.content == "ok"
    assert redirect.status is CallStatus.FAILED  # redirect not followed
    assert server_error.error is not None and server_error.error.retryable
    assert "PRIVATE" not in repr(server_error)
    assert len(seen) == 3  # no request to the redirect target
    expected_host = "api.groq.com" if provider_id == "groq" else "integrate.api.nvidia.com"
    assert {r.url.host for r in seen} == {expected_host}
    assert seen[0].headers["Authorization"] == "Bearer " + SECRET
    assert json.loads(seen[0].content)["model"] == LARGE


def test_f1_httpx_timeout_and_connect_errors_are_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", SECRET)

    async def timeout(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret detail", request=req)

    async def broken(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret detail", request=req)

    async def run(handler: object) -> ModelResponse:
        async with managed_httpx_client(
            mock_transport=httpx.MockTransport(handler)  # type: ignore[arg-type]
        ) as client:
            provider = GroqProvider(
                GroqConfig(
                    allowed_models=frozenset({LARGE}), live_enabled=True, privacy_approved=True
                ),
                transport=HttpxGroqTransport(client),
            )
            return await provider.generate(request("groq"))

    timed_out = asyncio.run(run(timeout))
    failed = asyncio.run(run(broken))
    assert timed_out.status is CallStatus.TIMEOUT
    assert failed.error is not None and failed.error.code == "GROQ_TRANSPORT_FAILURE"
    assert "secret" not in repr(timed_out) + repr(failed)
    with pytest.raises(ValueError, match="HTTPX_POOL_CONNECTIONS_INVALID"):
        asyncio.run(run_pool_invalid())


async def run_pool_invalid() -> None:
    async with managed_httpx_client(max_connections=0):
        pass


# --------------------------------------------------------------------- F2 (#16)


class LargeModelFails:
    def __init__(self) -> None:
        self.models: list[str] = []

    async def generate(self, req: ModelRequest) -> ModelResponse:
        self.models.append(req.model)
        if req.model == LARGE:
            return ModelResponse(
                provider=req.provider,
                model=req.model,
                attempt_id=req.attempt_id,
                status=CallStatus.FAILED,
                metric_origin=MetricOrigin.SIMULATED,
                error=ErrorInfo(category=ErrorCategory.PROVIDER, code="SYNTHETIC_LARGE_DOWN"),
            )
        return await DemoFakeProvider().generate(req)


def test_f2_failed_escalation_is_visible_and_counted_as_degraded() -> None:
    """Issue #16 repro: escalate account_general, small OK, large fails."""
    provider = LargeModelFails()
    router = CascadeRouter(
        ModelPair(small=SMALL, large=LARGE), escalate_scenarios=frozenset({"account_general"})
    )
    result = asyncio.run(
        engine(router, provider).run(
            [demo_case()], RunOptions(strategy=StrategyId.CASCADE, allowed_models=(SMALL, LARGE))
        )
    )
    trace = result.traces[0]
    assert provider.models == [SMALL, LARGE]
    assert trace.status is ExecutionStatus.COMPLETED
    codes = [s.error.code for s in trace.steps if s.kind is TraceStepKind.ERROR and s.error]
    assert codes == ["ESCALATION_FAILED_RETAINED_LOWER_TIER"]
    metrics = result.report.strategy_metrics[StrategyId.CASCADE]
    assert (metrics.successful_cases, metrics.degraded_cases) == (1, 1)

    base = asyncio.run(
        engine(DemoAlwaysLargeRouter(LARGE), LargeModelFails()).run(
            [demo_case()], RunOptions(allowed_models=(SMALL, LARGE))
        )
    )
    base_metrics = base.report.strategy_metrics[StrategyId.ALWAYS_LARGE]
    assert base.traces[0].status is ExecutionStatus.FAILED
    # Strict success is what is comparable with the baseline: 0 vs 0.
    assert metrics.successful_cases - metrics.degraded_cases == base_metrics.successful_cases


def test_f2_accepted_small_answer_is_not_degraded() -> None:
    router = CascadeRouter(ModelPair(small=SMALL, large=LARGE))
    result = asyncio.run(
        engine(router, LargeModelFails()).run(
            [demo_case("card_general")],
            RunOptions(strategy=StrategyId.CASCADE, allowed_models=(SMALL, LARGE)),
        )
    )
    metrics = result.report.strategy_metrics[StrategyId.CASCADE]
    assert (metrics.successful_cases, metrics.degraded_cases) == (1, 0)
    assert not any(s.kind is TraceStepKind.ERROR for s in result.traces[0].steps)


# --------------------------------------------------------------------- F3 (#15)


@pytest.mark.parametrize(("calls", "retries"), [(2, 1), (3, 1), (5, 2), (7, 3)])
def test_f3_cascade_budget_that_starves_escalation_is_rejected(calls: int, retries: int) -> None:
    with pytest.raises(ValidationError, match="CASCADE_CALL_BUDGET_TOO_SMALL_FOR_RETRIES"):
        RunOptions(
            strategy=StrategyId.CASCADE,
            allowed_models=(SMALL, LARGE),
            max_model_calls_per_case=calls,
            max_retries=retries,
        )


@pytest.mark.parametrize(("calls", "retries"), [(2, 0), (4, 1), (6, 2), (8, 3)])
def test_f3_minimum_cascade_budget_is_accepted(calls: int, retries: int) -> None:
    RunOptions(
        strategy=StrategyId.CASCADE,
        allowed_models=(SMALL, LARGE),
        max_model_calls_per_case=calls,
        max_retries=retries,
    )


def test_f3_rule_does_not_affect_single_call_strategies() -> None:
    for strategy in (StrategyId.ALWAYS_LARGE, StrategyId.RULE_BASED, StrategyId.KNN):
        RunOptions(strategy=strategy, max_model_calls_per_case=2, max_retries=1)


def test_f3_issue_repro_now_escalates_after_small_timeouts() -> None:
    """Issue #15 repro with the minimum valid budget: small times out, large is tried."""

    class SmallTimesOut:
        def __init__(self) -> None:
            self.models: list[str] = []

        async def generate(self, req: ModelRequest) -> ModelResponse:
            self.models.append(req.model)
            if req.model == SMALL:
                return ModelResponse(
                    provider=req.provider,
                    model=req.model,
                    attempt_id=req.attempt_id,
                    status=CallStatus.TIMEOUT,
                    metric_origin=MetricOrigin.SIMULATED,
                    error=ErrorInfo(
                        category=ErrorCategory.PROVIDER_TIMEOUT, code="TIMEOUT", retryable=True
                    ),
                )
            return await DemoFakeProvider().generate(req)

    provider = SmallTimesOut()
    result = asyncio.run(
        engine(CascadeRouter(ModelPair(small=SMALL, large=LARGE)), provider).run(
            [demo_case()],
            RunOptions(
                strategy=StrategyId.CASCADE,
                allowed_models=(SMALL, LARGE),
                max_model_calls_per_case=4,
                max_retries=1,
            ),
        )
    )
    assert provider.models == [SMALL, SMALL, LARGE]
    assert result.traces[0].status is ExecutionStatus.COMPLETED


@pytest.mark.parametrize(("retries", "exit_ok"), [(0, True), (1, False)])
def test_f3_cli_rejects_cascade_budget_that_starves_escalation(
    tmp_path: Path, retries: int, exit_ok: bool
) -> None:
    root = Path(__file__).resolve().parents[2]
    settings = yaml.safe_load((root / "config" / "settings.yaml").read_text(encoding="utf-8"))
    settings["execution"]["max_retries"] = retries  # max_model_calls_per_case stays 2
    config = tmp_path / "settings.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    response = CliRunner().invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--dataset",
            str(root / "examples" / "offline_demo_cases.json"),
            "--strategy",
            "cascade",
            "--output",
            str(tmp_path / "out.jsonl"),
        ],
    )
    if exit_ok:
        assert response.exit_code == 0, response.output
        assert "STRICT_SUCCESS: 2, DEGRADED: 0" in response.output
    else:
        assert response.exit_code != 0
        assert "CASCADE_CALL_BUDGET_TOO_SMALL_FOR_RETRIES" in response.output


@pytest.mark.parametrize("provider_id", ["groq", "nvidia_build"])
def test_f1_compressed_response_is_rejected_before_decompression(
    monkeypatch: pytest.MonkeyPatch, provider_id: str
) -> None:
    """Codex review on #18: a gzip bomb must not be expanded past the body cap."""
    import gzip

    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setenv("NVIDIA_API_KEY", SECRET)
    bomb = gzip.compress(b"0" * 60_000_000)
    assert len(bomb) < 100_000

    class CountingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.yielded = 0

        async def __aiter__(self):  # type: ignore[no-untyped-def]
            for start in range(0, len(bomb), 4096):
                chunk = bomb[start : start + 4096]
                self.yielded += len(chunk)
                yield chunk

    stream = CountingStream()

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=stream)

    async def run() -> ModelResponse:
        async with managed_httpx_client(mock_transport=httpx.MockTransport(handler)) as client:
            provider: GroqProvider | NvidiaBuildProvider
            if provider_id == "groq":
                provider = GroqProvider(
                    GroqConfig(
                        allowed_models=frozenset({LARGE}), live_enabled=True, privacy_approved=True
                    ),
                    transport=HttpxGroqTransport(client),
                )
            else:
                provider = NvidiaBuildProvider(
                    NvidiaBuildConfig(
                        allowed_models=frozenset({LARGE}), live_enabled=True, privacy_approved=True
                    ),
                    transport=HttpxNvidiaTransport(client),
                )
            return await provider.generate(request(provider_id))

    response = asyncio.run(run())
    assert response.status is not CallStatus.SUCCESS
    assert stream.yielded == 0  # rejected from headers; nothing read or decompressed


@pytest.mark.parametrize("encoding", ["", "identity"])
def test_f1_uncompressed_response_still_accepted(
    monkeypatch: pytest.MonkeyPatch, encoding: str
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    body = {
        "model": LARGE,
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    }

    async def handler(_: httpx.Request) -> httpx.Response:
        headers = {"Content-Encoding": encoding} if encoding else {}
        return httpx.Response(200, headers=headers, content=json.dumps(body).encode())

    async def run() -> ModelResponse:
        async with managed_httpx_client(mock_transport=httpx.MockTransport(handler)) as client:
            provider = GroqProvider(
                GroqConfig(
                    allowed_models=frozenset({LARGE}), live_enabled=True, privacy_approved=True
                ),
                transport=HttpxGroqTransport(client),
            )
            return await provider.generate(request("groq"))

    assert asyncio.run(run()).status is CallStatus.SUCCESS
