"""Per-request OutboundAuthorization: deny by default, fail closed, asked per attempt.

All offline: in-memory transports and fakes; no network, no real keys.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from llm_routing_eval.adapters.providers.authorization import DenyAllOutbound, is_authorized
from llm_routing_eval.adapters.providers.groq import GroqConfig, GroqHttpResult, GroqProvider
from llm_routing_eval.adapters.providers.nvidia_build import (
    NvidiaBuildConfig,
    NvidiaBuildProvider,
    NvidiaHttpResult,
)
from llm_routing_eval.application import BenchmarkOrchestrator, RunOptions
from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    ErrorCategory,
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

MODEL = "openai/gpt-oss-120b"
SECRET = "IN_MEMORY_FAKE_CREDENTIAL"
SYNTHETIC = sorted(DEMO_PROMPTS)[0]
OK_BODY = (
    b'{"model": "openai/gpt-oss-120b", "choices": '
    b'[{"message": {"content": "ok"}, "finish_reason": "stop"}]}'
)


class RecordingTransport:
    """Satisfies both provider transport ports; records whether it was used."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.calls = 0

    async def post_completion(
        self, payload: Mapping[str, object], *, api_key: str, timeout_seconds: float
    ) -> GroqHttpResult:
        self.calls += 1
        body = OK_BODY if self.status == 200 else b""
        return GroqHttpResult(status_code=self.status, body=body)

    async def post_chat(
        self, payload: Mapping[str, object], *, api_key: str, timeout_seconds: float
    ) -> NvidiaHttpResult:
        self.calls += 1
        body = OK_BODY if self.status == 200 else b""
        return NvidiaHttpResult(status_code=self.status, body=body)


class SyntheticPromptOnly:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def authorize(self, request: ModelRequest) -> bool:
        self.seen.append(request.attempt_id)
        return request.prompt == SYNTHETIC


def _provider(
    provider_id: str,
    transport: RecordingTransport,
    authorization: object | None,
    *,
    live_enabled: bool = True,
    privacy_approved: bool = True,
) -> GroqProvider | NvidiaBuildProvider:
    kwargs: dict[str, object] = {"transport": transport}
    if authorization is not None:
        kwargs["outbound_authorization"] = authorization
    if provider_id == "groq":
        return GroqProvider(
            GroqConfig(
                allowed_models=frozenset({MODEL}),
                live_enabled=live_enabled,
                privacy_approved=privacy_approved,
            ),
            **kwargs,  # type: ignore[arg-type]
        )
    return NvidiaBuildProvider(
        NvidiaBuildConfig(
            allowed_models=frozenset({MODEL}),
            live_enabled=live_enabled,
            privacy_approved=privacy_approved,
        ),
        **kwargs,  # type: ignore[arg-type]
    )


def _request(provider_id: str, prompt: str = SYNTHETIC, model: str = MODEL) -> ModelRequest:
    return ModelRequest(
        run_id="r",
        case_id="c",
        trace_id="t",
        attempt_id="a",
        provider=provider_id,
        model=model,
        prompt=prompt,
        timeout_seconds=1.0,
    )


PROVIDERS = [("groq", "GROQ"), ("nvidia_build", "NVIDIA")]


@pytest.fixture(autouse=True)
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setenv("NVIDIA_API_KEY", SECRET)


@pytest.mark.parametrize(("provider_id", "prefix"), PROVIDERS)
def test_default_denies_even_with_all_flags_key_and_transport(
    provider_id: str, prefix: str
) -> None:
    transport = RecordingTransport()
    response = asyncio.run(_provider(provider_id, transport, None).generate(_request(provider_id)))
    assert response.status is CallStatus.FAILED
    assert response.error is not None
    assert response.error.category is ErrorCategory.PRIVACY
    assert response.error.code == f"{prefix}_OUTBOUND_NOT_AUTHORIZED"
    assert not response.error.retryable
    assert transport.calls == 0


@pytest.mark.parametrize(("provider_id", "prefix"), PROVIDERS)
def test_authorizer_decides_per_request(provider_id: str, prefix: str) -> None:
    transport = RecordingTransport()
    authorizer = SyntheticPromptOnly()
    provider = _provider(provider_id, transport, authorizer)
    allowed = asyncio.run(provider.generate(_request(provider_id)))
    denied = asyncio.run(provider.generate(_request(provider_id, prompt="müşteri verisi")))
    assert allowed.status is CallStatus.SUCCESS
    assert denied.error is not None and denied.error.code == f"{prefix}_OUTBOUND_NOT_AUTHORIZED"
    assert transport.calls == 1
    assert "müşteri" not in repr(denied)


class Raises:
    def authorize(self, request: ModelRequest) -> bool:
        raise RuntimeError("authorizer bug with " + request.prompt)


class Truthy:
    def __init__(self, value: object) -> None:
        self.value = value

    def authorize(self, request: ModelRequest) -> object:
        return self.value


@pytest.mark.parametrize(("provider_id", "prefix"), PROVIDERS)
@pytest.mark.parametrize("authorizer", [Raises(), Truthy(1), Truthy("yes"), Truthy(None)])
def test_authorizer_errors_and_non_true_values_fail_closed(
    provider_id: str, prefix: str, authorizer: object
) -> None:
    transport = RecordingTransport()
    response = asyncio.run(
        _provider(provider_id, transport, authorizer).generate(_request(provider_id))
    )
    assert response.error is not None
    assert response.error.code == f"{prefix}_OUTBOUND_NOT_AUTHORIZED"
    assert SYNTHETIC not in repr(response)
    assert transport.calls == 0


@pytest.mark.parametrize(("provider_id", "prefix"), PROVIDERS)
def test_static_gates_run_before_the_authorizer(provider_id: str, prefix: str) -> None:
    authorizer = SyntheticPromptOnly()
    closed = _provider(provider_id, RecordingTransport(), authorizer, privacy_approved=False)
    asyncio.run(closed.generate(_request(provider_id)))
    open_provider = _provider(provider_id, RecordingTransport(), authorizer)
    asyncio.run(open_provider.generate(_request(provider_id, model="not/allowlisted")))
    assert authorizer.seen == []  # never asked for requests that fail static gates


def test_privacy_approved_alone_is_not_sufficient() -> None:
    assert not is_authorized(DenyAllOutbound(), _request("groq"))


def test_orchestrator_retries_ask_the_authorizer_for_every_attempt() -> None:
    authorizer = SyntheticPromptOnly()
    transport = RecordingTransport(status=503)  # retryable server error
    provider = _provider("groq", transport, authorizer)

    async def no_sleep(_: float) -> None:
        return None

    engine = BenchmarkOrchestrator(
        router=DemoAlwaysLargeRouter(MODEL),
        provider=provider,
        evaluator=DemoFixtureEvaluator(),
        privacy=AllowlistedDemoPrivacy(),
        repository=InMemoryDemoRepository(),
        sleep=no_sleep,
    )
    case = BenchmarkCase(
        case_id="c", input_text=SYNTHETIC, domain="turkish_banking", scenario_type="synthetic"
    )
    result = asyncio.run(
        engine.run(
            [case],
            RunOptions(
                provider="groq",
                allowed_models=(MODEL,),
                max_model_calls_per_case=3,
                max_retries=2,
            ),
        )
    )
    attempts = [s for s in result.traces[0].steps if s.kind is TraceStepKind.PROVIDER_ATTEMPT]
    assert len(attempts) == 3
    assert len(authorizer.seen) == 3 and len(set(authorizer.seen)) == 3  # one per attempt_id
    assert transport.calls == 3


def test_provider_responses_are_model_responses() -> None:
    # Guards the test double: both ports return the shared Core contract.
    response = asyncio.run(
        _provider("groq", RecordingTransport(), SyntheticPromptOnly()).generate(_request("groq"))
    )
    assert isinstance(response, ModelResponse)
