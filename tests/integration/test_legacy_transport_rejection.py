"""Offline regressions: provider transport DI must reject non-cancellable built-ins."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

import pytest

from llm_routing_eval.adapters.providers.groq import (
    GroqConfig,
    GroqHttpResult,
    GroqHttpTransport,
    GroqProvider,
)
from llm_routing_eval.adapters.providers.nvidia_build import (
    NvidiaBuildConfig,
    NvidiaHttpResult,
    NvidiaHttpTransport,
    NvidiaBuildProvider,
)
from llm_routing_eval.core import CallStatus, ModelRequest

MODEL = "openai/gpt-oss-120b"
FAKE_KEY = "offline-test-only-secret"


def _request(provider: str) -> ModelRequest:
    return ModelRequest(
        run_id="transport-regression",
        case_id="synthetic",
        trace_id="trace",
        attempt_id="attempt",
        provider=provider,
        model=MODEL,
        prompt="A synthetic offline test prompt.",
        timeout_seconds=0.2,
        max_output_tokens=16,
    )


def _response_body() -> bytes:
    return json.dumps(
        {
            "id": "safe-offline-id",
            "choices": [
                {"message": {"content": "Synthetic output"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
    ).encode("utf-8")


class ForbiddenLegacyGroq(GroqHttpTransport):
    async def post_completion(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> GroqHttpResult:
        raise AssertionError("Legacy thread transport must never execute")


class ForbiddenLegacyNvidia(NvidiaHttpTransport):
    async def post_chat(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> NvidiaHttpResult:
        raise AssertionError("Legacy thread transport must never execute")


class AsyncGroqFixtureTransport:
    async def post_completion(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> GroqHttpResult:
        assert api_key == FAKE_KEY
        assert payload["model"] == MODEL
        return GroqHttpResult(status_code=200, body=_response_body())


class AsyncNvidiaFixtureTransport:
    async def post_chat(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> NvidiaHttpResult:
        assert api_key == FAKE_KEY
        assert payload["model"] == MODEL
        return NvidiaHttpResult(status_code=200, body=_response_body())


@pytest.mark.parametrize(
    ("provider_name", "error_code"),
    [
        ("groq", "GROQ_UNSAFE_THREAD_TRANSPORT"),
        ("nvidia_build", "NVIDIA_UNSAFE_THREAD_TRANSPORT"),
    ],
)
def test_explicit_legacy_transport_fails_closed_before_network(
    monkeypatch: pytest.MonkeyPatch, provider_name: str, error_code: str
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)

    provider: GroqProvider | NvidiaBuildProvider
    if provider_name == "groq":
        provider = GroqProvider(
            GroqConfig(
                allowed_models=frozenset({MODEL}),
                live_enabled=True,
                privacy_approved=True,
            ),
            transport=ForbiddenLegacyGroq(),
        )
    else:
        provider = NvidiaBuildProvider(
            NvidiaBuildConfig(
                allowed_models=frozenset({MODEL}),
                live_enabled=True,
                privacy_approved=True,
            ),
            transport=ForbiddenLegacyNvidia(),
        )

    result = asyncio.run(provider.generate(_request(provider_name)))

    assert result.status is CallStatus.FAILED
    assert result.error is not None
    assert result.error.code == error_code
    assert result.error.retryable is False
    assert FAKE_KEY not in repr(result)


@pytest.mark.parametrize("provider_name", ["groq", "nvidia_build"])
def test_cancellable_transport_port_injection_still_works_offline(
    monkeypatch: pytest.MonkeyPatch, provider_name: str
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)

    provider: GroqProvider | NvidiaBuildProvider
    if provider_name == "groq":
        provider = GroqProvider(
            GroqConfig(
                allowed_models=frozenset({MODEL}),
                live_enabled=True,
                privacy_approved=True,
            ),
            transport=AsyncGroqFixtureTransport(),
        )
    else:
        provider = NvidiaBuildProvider(
            NvidiaBuildConfig(
                allowed_models=frozenset({MODEL}),
                live_enabled=True,
                privacy_approved=True,
            ),
            transport=AsyncNvidiaFixtureTransport(),
        )

    result = asyncio.run(provider.generate(_request(provider_name)))

    assert result.status is CallStatus.SUCCESS
    assert result.content == "Synthetic output"
    assert FAKE_KEY not in repr(result)
