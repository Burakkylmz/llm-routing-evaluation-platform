"""Opt-in live provider checks (``pytest -m live``). SKIPPED by default; never run in CI.

Each test sends one fixed synthetic prompt through the production code path:
``GroqProvider`` / ``NvidiaBuildProvider`` + the cancellable HTTPX transport +
an ``OutboundAuthorization`` that allows only that prompt. A pass proves wire
compatibility and account/model access only, not quality, latency SLOs or cost.

Environment:
    LLM_ROUTING_LIVE_TESTS=1          required opt-in
    GROQ_API_KEY                      for the Groq test
    GROQ_LIVE_TEST_MODEL              optional, default openai/gpt-oss-20b
    NVIDIA_API_KEY                    for the NVIDIA test
    NVIDIA_LIVE_TEST_MODEL            required for NVIDIA (no default: unverified)
    LIVE_TEST_MAX_OUTPUT_TOKENS       optional, default 256
"""

from __future__ import annotations

import asyncio
import os

import pytest
from conftest import SYNTHETIC_PROMPT, FixedSyntheticPromptOnly, require_live

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
from llm_routing_eval.core import CallStatus, MetricOrigin, ModelRequest, ModelResponse

pytestmark = pytest.mark.live


def _request(provider: str, model: str) -> ModelRequest:
    return ModelRequest(
        run_id="live-smoke",
        case_id="synthetic-live-smoke",
        trace_id="live-smoke",
        attempt_id="live-smoke-1",
        provider=provider,
        model=model,
        prompt=SYNTHETIC_PROMPT,
        timeout_seconds=30.0,
        temperature=0.0,
        max_output_tokens=int(os.getenv("LIVE_TEST_MAX_OUTPUT_TOKENS", "256")),
    )


def _assert_live_success(response: ModelResponse, secret_env: str) -> None:
    assert response.status is CallStatus.SUCCESS, response.error
    assert response.metric_origin is MetricOrigin.OBSERVED
    assert response.content is not None and response.content.strip()
    assert response.latency_ms is not None and response.latency_ms > 0
    assert os.environ[secret_env] not in repr(response)


def test_groq_live_chat_completion() -> None:
    require_live("GROQ_API_KEY")
    model = os.getenv("GROQ_LIVE_TEST_MODEL", "openai/gpt-oss-20b")
    authorizer = FixedSyntheticPromptOnly()

    async def run() -> ModelResponse:
        async with managed_httpx_client(max_connections=1) as client:
            provider = GroqProvider(
                GroqConfig(
                    allowed_models=frozenset({model}), live_enabled=True, privacy_approved=True
                ),
                transport=HttpxGroqTransport(client),
                outbound_authorization=authorizer,
            )
            return await provider.generate(_request("groq", model))

    response = asyncio.run(run())
    _assert_live_success(response, "GROQ_API_KEY")
    assert authorizer.authorized == 1
    # Usage is provider-reported; record what was observed without asserting cost.
    assert response.usage is None or (response.usage.input_tokens or 0) >= 0


def test_nvidia_build_live_chat_completion() -> None:
    # No default model: NVIDIA Build availability must be confirmed by the operator.
    require_live("NVIDIA_API_KEY", "NVIDIA_LIVE_TEST_MODEL")
    model = os.environ["NVIDIA_LIVE_TEST_MODEL"]
    authorizer = FixedSyntheticPromptOnly()

    async def run() -> ModelResponse:
        async with managed_httpx_client(max_connections=1) as client:
            provider = NvidiaBuildProvider(
                NvidiaBuildConfig(
                    allowed_models=frozenset({model}), live_enabled=True, privacy_approved=True
                ),
                transport=HttpxNvidiaTransport(client),
                outbound_authorization=authorizer,
            )
            return await provider.generate(_request("nvidia_build", model))

    response = asyncio.run(run())
    _assert_live_success(response, "NVIDIA_API_KEY")
    assert authorizer.authorized == 1
