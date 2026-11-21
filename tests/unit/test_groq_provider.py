"""Groq adapter contract/transport tests. No real network calls or credentials."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest

from llm_routing_eval.adapters.providers.groq import (
    GROQ_CHAT_ENDPOINT,
    GroqConfig,
    GroqHttpResult,
    GroqHttpTransport,
    GroqProvider,
    GroqTransportError,
    GroqTransportTimeout,
    _RejectRedirects,
)
from llm_routing_eval.core import (
    CallStatus,
    ErrorCategory,
    MetricOrigin,
    ModelRequest,
)

MODEL = "openai/gpt-oss-120b"


def _request(**overrides: Any) -> ModelRequest:
    values: dict[str, Any] = {
        "run_id": "synthetic-run",
        "case_id": "synthetic-case",
        "trace_id": "synthetic-trace",
        "attempt_id": "synthetic-attempt",
        "provider": "groq",
        "model": MODEL,
        "prompt": "Synthetic general banking question",
        "timeout_seconds": 4.0,
        "temperature": 0.0,
        "max_output_tokens": 64,
    }
    values.update(overrides)
    return ModelRequest(**values)


def _completion() -> dict[str, Any]:
    return {
        "id": "chatcmpl_safe123",
        "model": MODEL,
        "choices": [{"finish_reason": "stop", "message": {"content": "Synthetic answer"}}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 8},
    }


class FakeTransport:
    def __init__(
        self,
        *,
        status: int = 200,
        body: dict[str, Any] | bytes | None = None,
        error: Exception | None = None,
    ) -> None:
        self.status = status
        self.body = body if body is not None else _completion()
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def post_completion(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> GroqHttpResult:
        self.calls.append(
            {"payload": dict(payload), "api_key_present": bool(api_key), "timeout": timeout_seconds}
        )
        if self.error is not None:
            raise self.error
        data = self.body if isinstance(self.body, bytes) else json.dumps(self.body).encode("utf-8")
        return GroqHttpResult(status_code=self.status, body=data)


def _provider(
    transport: FakeTransport, *, live_enabled: bool = True, privacy_approved: bool = True
) -> GroqProvider:
    return GroqProvider(
        GroqConfig(
            allowed_models=frozenset({MODEL}),
            live_enabled=live_enabled,
            privacy_approved=privacy_approved,
        ),
        transport=transport,
    )


def _run(provider: GroqProvider, model_request: ModelRequest | None = None) -> Any:
    return asyncio.run(provider.generate(model_request or _request()))


@pytest.mark.parametrize(
    "live,privacy",
    [(False, False), (True, False), (False, True)],
)
def test_live_and_privacy_gates_prevent_network(
    monkeypatch: pytest.MonkeyPatch, live: bool, privacy: bool
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "never-send-this")
    fake = FakeTransport()
    result = _run(_provider(fake, live_enabled=live, privacy_approved=privacy))
    assert result.status is CallStatus.FAILED
    assert result.error is not None and result.error.code == "GROQ_LIVE_GATE_CLOSED"
    assert fake.calls == []


def test_missing_api_key_fails_before_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    fake = FakeTransport()
    result = _run(_provider(fake))
    assert result.error is not None and result.error.code == "GROQ_API_KEY_MISSING"
    assert fake.calls == []


@pytest.mark.parametrize(
    "model_request",
    [
        _request(provider="nvidia_build"),
        _request(model="unapproved/model"),
        _request(generation_parameters={"top_k": 12}),
        _request(generation_parameters={"top_p": 0}),
        _request(generation_parameters={"top_p": True}),
        _request(generation_parameters={"top_p": float("nan")}),
    ],
)
def test_rejects_unsupported_requests_without_network(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fixture-key")
    fake = FakeTransport()
    response = _run(_provider(fake), model_request)
    assert response.status is CallStatus.FAILED
    assert response.error is not None and response.error.category is ErrorCategory.VALIDATION
    assert fake.calls == []


def test_success_and_observed_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-secret-key")
    fake = FakeTransport()
    result = _run(_provider(fake), _request(generation_parameters={"top_p": 0.9}))
    assert result.status is CallStatus.SUCCESS
    assert result.content == "Synthetic answer"
    assert result.metric_origin is MetricOrigin.OBSERVED
    assert result.usage is not None
    assert result.usage.input_tokens == 15
    assert result.usage.output_tokens == 8
    assert result.provider_request_id == "chatcmpl_safe123"
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert fake.calls[0]["payload"] == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Synthetic general banking question"}],
        "stream": False,
        "temperature": 0.0,
        "max_completion_tokens": 64,
        "top_p": 0.9,
    }
    assert "test-secret-key" not in repr(result)


@pytest.mark.parametrize(
    "status,expected_status,expected_code,retry",
    [
        (429, CallStatus.RATE_LIMITED, "GROQ_RATE_LIMITED", True),
        (408, CallStatus.TIMEOUT, "GROQ_HTTP_TIMEOUT", True),
        (504, CallStatus.TIMEOUT, "GROQ_HTTP_TIMEOUT", True),
        (401, CallStatus.FAILED, "GROQ_AUTH_FAILURE", False),
        (403, CallStatus.FAILED, "GROQ_AUTH_FAILURE", False),
        (404, CallStatus.FAILED, "GROQ_MODEL_NOT_AVAILABLE", False),
        (400, CallStatus.FAILED, "GROQ_REQUEST_REJECTED", False),
        (422, CallStatus.FAILED, "GROQ_REQUEST_REJECTED", False),
        (500, CallStatus.FAILED, "GROQ_HTTP_ERROR", True),
        (302, CallStatus.FAILED, "GROQ_HTTP_ERROR", False),
    ],
)
def test_safe_http_failure_normalization(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    expected_status: CallStatus,
    expected_code: str,
    retry: bool,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    reply = _run(_provider(FakeTransport(status=status, body=b"key and prompt leaked")))
    assert reply.status is expected_status
    assert reply.error is not None
    assert reply.error.code == expected_code
    assert reply.error.retryable is retry
    assert reply.content is None
    assert "leaked" not in repr(reply)


@pytest.mark.parametrize(
    "transport_error,expected_status,code",
    [
        (GroqTransportTimeout(), CallStatus.TIMEOUT, "GROQ_TIMEOUT"),
        (GroqTransportError("secret"), CallStatus.FAILED, "GROQ_TRANSPORT_FAILURE"),
        (RuntimeError("raw sensitive data"), CallStatus.FAILED, "GROQ_TRANSPORT_FAILURE"),
    ],
)
def test_transport_error_data_is_never_exposed(
    monkeypatch: pytest.MonkeyPatch,
    transport_error: Exception,
    expected_status: CallStatus,
    code: str,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    reply = _run(_provider(FakeTransport(error=transport_error)))
    assert reply.status is expected_status
    assert reply.error is not None and reply.error.code == code
    assert "sensitive" not in repr(reply)


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b'{"choices":[]}',
        b'{"choices":[{"finish_reason":"stop","message":{"content":""}}]}',
        b'{"choices":[{"finish_reason":"stop","message":{"content":null}}]}',
    ],
)
def test_malformed_response_is_explicit_failure(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    reply = _run(_provider(FakeTransport(body=body)))
    assert reply.status is CallStatus.FAILED
    assert reply.error is not None and reply.error.code == "GROQ_MALFORMED_RESPONSE"


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
def test_incomplete_output_never_passes_as_complete(
    monkeypatch: pytest.MonkeyPatch, finish: str | None
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    body = _completion()
    body["choices"][0]["finish_reason"] = finish
    reply = _run(_provider(FakeTransport(body=body)))
    assert reply.status is CallStatus.FAILED
    assert reply.error is not None and reply.error.code == "GROQ_INCOMPLETE_RESPONSE"


def test_provider_model_mismatch_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    body = _completion()
    body["model"] = "other/model"
    reply = _run(_provider(FakeTransport(body=body)))
    assert reply.error is not None and reply.error.code == "GROQ_MODEL_MISMATCH"


def test_unknown_token_usage_is_not_fabricated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    body = _completion()
    body["usage"] = {"prompt_tokens": -1, "completion_tokens": "five"}
    reply = _run(_provider(FakeTransport(body=body)))
    assert reply.status is CallStatus.SUCCESS
    assert reply.usage is None


def test_usage_details_and_untrusted_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    body = _completion()
    body["id"] = "unsafe id <prompt>"
    body["usage"] = {
        "prompt_tokens": 17,
        "completion_tokens": 8,
        "prompt_tokens_details": {"cached_tokens": 3},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    reply = _run(_provider(FakeTransport(body=body)))
    assert reply.provider_request_id is None
    assert reply.usage is not None
    assert reply.usage.cached_tokens == 3
    assert reply.usage.reasoning_tokens == 2


def test_invalid_allowlist_cannot_be_configured() -> None:
    with pytest.raises(ValueError, match="GROQ_INVALID_MODEL_CONFIGURATION"):
        GroqConfig(allowed_models=frozenset())
    with pytest.raises(ValueError, match="GROQ_INVALID_MODEL_CONFIGURATION"):
        GroqConfig(allowed_models=frozenset({"invalid model id"}))


def test_https_transport_pins_endpoint_and_disables_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_routing_eval.adapters.providers.groq as module

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self, size: int) -> bytes:
            assert size == 2_000_001
            return b'{"choices":[]}'

    class Opener:
        def open(self, request: Any, *, timeout: float) -> Response:
            assert request.full_url == GROQ_CHAT_ENDPOINT
            assert request.get_header("Authorization") == "Bearer local-test-key"
            assert request.get_header("Content-type") == "application/json"
            assert timeout == 4.0
            return Response()

    def opener(handler: Any) -> Opener:
        assert isinstance(handler, _RejectRedirects)
        return Opener()

    monkeypatch.setattr(module, "build_opener", opener)
    response = GroqHttpTransport()._post({"model": MODEL}, api_key="local-test-key", timeout_seconds=4)
    assert response.status_code == 200
    assert response.body == b'{"choices":[]}'


def test_redirect_handler_rejects_token_forwarding() -> None:
    from urllib.request import Request

    handler = _RejectRedirects()
    request = Request(GROQ_CHAT_ENDPOINT, method="POST")
    assert (
        handler.redirect_request(request, None, 307, "redirect", {}, "https://evil.invalid/")
        is None
    )


def test_http_transport_ignores_provider_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    from urllib.error import HTTPError

    import llm_routing_eval.adapters.providers.groq as module

    class Opener:
        def open(self, request: Any, *, timeout: float) -> Any:
            _ = request, timeout
            raise HTTPError(GROQ_CHAT_ENDPOINT, 429, "secret", {}, None)

    monkeypatch.setattr(module, "build_opener", lambda handler: Opener())
    response = GroqHttpTransport()._post({}, api_key="key", timeout_seconds=2)
    assert response.status_code == 429
    assert response.body == b""


def test_http_transport_rejects_oversized_body(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_routing_eval.adapters.providers.groq as module

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self, size: int) -> bytes:
            return b"*" * size

    class Opener:
        def open(self, request: Any, *, timeout: float) -> Response:
            _ = request, timeout
            return Response()

    monkeypatch.setattr(module, "build_opener", lambda handler: Opener())
    with pytest.raises(GroqTransportError, match="GROQ_RESPONSE_TOO_LARGE"):
        GroqHttpTransport()._post({}, api_key="key", timeout_seconds=2)
