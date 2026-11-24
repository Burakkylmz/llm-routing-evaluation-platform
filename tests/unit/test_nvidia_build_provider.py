"""Network-free contract tests for the NVIDIA Build provider adapter."""

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest

from llm_routing_eval.adapters.providers.nvidia_build import (
    NvidiaBuildConfig,
    NvidiaBuildProvider,
    NvidiaHttpResult,
    NvidiaTransportError,
    NvidiaTransportTimeout,
)
from llm_routing_eval.core import CallStatus, ErrorCategory, MetricOrigin, ModelRequest


class AllowAllForTests:
    """Test-only OutboundAuthorization. Never use outside offline tests."""

    def authorize(self, request: object) -> bool:
        return True


MODEL = "vendor/offline-test-model"


class FakeTransport:
    def __init__(
        self,
        *,
        status: int = 200,
        response: dict[str, Any] | bytes | None = None,
        error: Exception | None = None,
    ) -> None:
        self.status = status
        self.response = response if response is not None else _completion()
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def post_chat(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> NvidiaHttpResult:
        self.calls.append(
            {"payload": dict(payload), "key_present": bool(api_key), "timeout": timeout_seconds}
        )
        if self.error is not None:
            raise self.error
        body = (
            self.response
            if isinstance(self.response, bytes)
            else json.dumps(self.response).encode("utf-8")
        )
        return NvidiaHttpResult(status_code=self.status, body=body)


def _completion() -> dict[str, Any]:
    return {
        "id": "chatcmpl_safe123",
        "choices": [{"finish_reason": "stop", "message": {"content": "Synthetic answer"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5},
    }


def _request(**changes: Any) -> ModelRequest:
    base: dict[str, Any] = {
        "run_id": "run-1",
        "case_id": "case-1",
        "trace_id": "trace-1",
        "attempt_id": "attempt-1",
        "provider": "nvidia_build",
        "model": MODEL,
        "prompt": "A synthetic Turkish banking question",
        "timeout_seconds": 4.0,
        "temperature": 0.0,
        "max_output_tokens": 64,
    }
    base.update(changes)
    return ModelRequest(**base)


def _provider(
    transport: FakeTransport, *, live: bool = True, privacy: bool = True
) -> NvidiaBuildProvider:
    return NvidiaBuildProvider(
        NvidiaBuildConfig(
            allowed_models=frozenset({MODEL}),
            live_enabled=live,
            privacy_approved=privacy,
        ),
        transport=transport,
        outbound_authorization=AllowAllForTests(),
    )


def _run(provider: NvidiaBuildProvider, request: ModelRequest | None = None) -> Any:
    return asyncio.run(provider.generate(request or _request()))


@pytest.mark.parametrize("live,privacy", [(False, False), (False, True), (True, False)])
def test_default_and_unapproved_live_paths_fail_closed(
    monkeypatch: pytest.MonkeyPatch, live: bool, privacy: bool
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "should-never-be-sent")
    transport = FakeTransport()
    result = _run(_provider(transport, live=live, privacy=privacy))
    assert result.status is CallStatus.FAILED
    assert result.error is not None and result.error.code == "NVIDIA_LIVE_GATE_CLOSED"
    assert result.content is None
    assert transport.calls == []


def test_missing_api_key_fails_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    transport = FakeTransport()
    result = _run(_provider(transport))
    assert result.error is not None and result.error.code == "NVIDIA_API_KEY_MISSING"
    assert not transport.calls


@pytest.mark.parametrize(
    "model_request",
    [
        _request(provider="groq"),
        _request(model="unapproved/model"),
        _request(generation_parameters={"unexpected": "injected"}),
        _request(generation_parameters={"top_p": 0.0}),
        _request(generation_parameters={"top_p": True}),
    ],
)
def test_invalid_model_or_parameters_never_sent(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-key")
    transport = FakeTransport()
    result = _run(_provider(transport), model_request)
    assert result.status is CallStatus.FAILED
    assert result.error is not None and result.error.category is ErrorCategory.VALIDATION
    assert not transport.calls


def test_success_uses_explicit_model_and_observed_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-key-secret")
    transport = FakeTransport()
    result = _run(_provider(transport), _request(generation_parameters={"top_p": 0.9}))
    assert result.status is CallStatus.SUCCESS
    assert result.content == "Synthetic answer"
    assert result.metric_origin is MetricOrigin.OBSERVED
    assert result.usage is not None
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 5
    assert result.provider_request_id == "chatcmpl_safe123"
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert transport.calls[0]["payload"] == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "A synthetic Turkish banking question"}],
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 64,
        "top_p": 0.9,
    }
    assert transport.calls[0]["key_present"] is True
    assert "fixture-key-secret" not in repr(result)


@pytest.mark.parametrize(
    "status,expected_status,code,retryable",
    [
        (429, CallStatus.RATE_LIMITED, "NVIDIA_RATE_LIMITED", True),
        (408, CallStatus.TIMEOUT, "NVIDIA_HTTP_TIMEOUT", True),
        (504, CallStatus.TIMEOUT, "NVIDIA_HTTP_TIMEOUT", True),
        (401, CallStatus.FAILED, "NVIDIA_AUTH_FAILED", False),
        (403, CallStatus.FAILED, "NVIDIA_AUTH_FAILED", False),
        (404, CallStatus.FAILED, "NVIDIA_MODEL_UNAVAILABLE", False),
        (422, CallStatus.FAILED, "NVIDIA_REQUEST_REJECTED", False),
        (500, CallStatus.FAILED, "NVIDIA_HTTP_ERROR", True),
        (302, CallStatus.FAILED, "NVIDIA_HTTP_ERROR", False),
    ],
)
def test_http_error_mapping_is_safe(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    expected_status: CallStatus,
    code: str,
    retryable: bool,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-key")
    result = _run(_provider(FakeTransport(status=status, response=b"secret echoed prompt")))
    assert result.status is expected_status
    assert result.error is not None
    assert result.error.code == code
    assert result.error.retryable is retryable
    assert "secret echoed" not in repr(result)
    assert result.content is None


@pytest.mark.parametrize(
    "error,status,code",
    [
        (NvidiaTransportTimeout(), CallStatus.TIMEOUT, "NVIDIA_TIMEOUT"),
        (NvidiaTransportError("contains key"), CallStatus.FAILED, "NVIDIA_TRANSPORT_ERROR"),
        (RuntimeError("sensitive prompt"), CallStatus.FAILED, "NVIDIA_TRANSPORT_ERROR"),
    ],
)
def test_transport_failures_never_surface_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status: CallStatus,
    code: str,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-key")
    result = _run(_provider(FakeTransport(error=error)))
    assert result.status is status
    assert result.error is not None and result.error.code == code
    assert "sensitive prompt" not in repr(result)
    assert "contains key" not in repr(result)


@pytest.mark.parametrize(
    "raw",
    [
        b"not-json",
        b'{"choices":[]}',
        b'{"choices":[{"finish_reason":"stop","message":{"content":""}}]}',
        b'{"choices":[{"finish_reason":"stop","message":{"content":null}}]}',
    ],
)
def test_malformed_completion_is_explicit_failure(
    monkeypatch: pytest.MonkeyPatch, raw: bytes
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-key")
    result = _run(_provider(FakeTransport(response=raw)))
    assert result.status is CallStatus.FAILED
    assert result.error is not None and result.error.code == "NVIDIA_MALFORMED_COMPLETION"


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
def test_incomplete_completion_is_not_scored_as_success(
    monkeypatch: pytest.MonkeyPatch, finish: str | None
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-key")
    data = _completion()
    data["choices"][0]["finish_reason"] = finish
    result = _run(_provider(FakeTransport(response=data)))
    assert result.status is CallStatus.FAILED
    assert result.error is not None and result.error.code == "NVIDIA_INCOMPLETE_COMPLETION"


def test_missing_or_invalid_token_usage_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-key")
    data = _completion()
    data["usage"] = {"prompt_tokens": -10, "completion_tokens": "invalid"}
    result = _run(_provider(FakeTransport(response=data)))
    assert result.status is CallStatus.SUCCESS
    assert result.usage is None


def test_valid_details_and_untrusted_id_are_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-key")
    data = _completion()
    data["id"] = "secret <user data>"
    data["usage"] = {
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 2},
        "completion_tokens_details": {"reasoning_tokens": 3},
    }
    result = _run(_provider(FakeTransport(response=data)))
    assert result.provider_request_id is None
    assert result.usage is not None
    assert result.usage.cached_tokens == 2
    assert result.usage.reasoning_tokens == 3


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="NVIDIA_MODEL_CONFIGURATION_INVALID"):
        NvidiaBuildConfig(allowed_models=frozenset())
    with pytest.raises(ValueError, match="NVIDIA_MODEL_CONFIGURATION_INVALID"):
        NvidiaBuildConfig(allowed_models=frozenset({"bad model!"}))


def test_https_transport_uses_fixed_endpoint_and_disables_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_routing_eval.adapters.providers.nvidia_build as module

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self, max_bytes: int) -> bytes:
            assert max_bytes == 2_000_001
            return b'{"choices":[]}'

    class Opener:
        def open(self, request: Any, *, timeout: float) -> Response:
            assert request.full_url == module.NVIDIA_CHAT_ENDPOINT
            assert request.get_header("Authorization") == "Bearer fixture-key"
            assert request.get_header("Content-type") == "application/json"
            assert timeout == 4.0
            return Response()

    def build(handler: Any) -> Opener:
        assert isinstance(handler, module._RejectRedirects)
        return Opener()

    monkeypatch.setattr(module, "build_opener", build)
    outcome = module.NvidiaHttpTransport()._request(
        {"model": MODEL}, api_key="fixture-key", timeout_seconds=4.0
    )
    assert outcome.status_code == 200
    assert outcome.body == b'{"choices":[]}'


def test_https_transport_does_not_follow_redirects() -> None:
    from urllib.request import Request

    import llm_routing_eval.adapters.providers.nvidia_build as module

    handler = module._RejectRedirects()
    request = Request(module.NVIDIA_CHAT_ENDPOINT, method="POST")
    assert (
        handler.redirect_request(
            request, None, 302, "redirect", {}, "https://untrusted.example/redirect"
        )
        is None
    )


def test_https_transport_does_not_read_http_error_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from urllib.error import HTTPError

    import llm_routing_eval.adapters.providers.nvidia_build as module

    class Opener:
        def open(self, _request: Any, *, timeout: float) -> Any:
            _ = timeout
            raise HTTPError(module.NVIDIA_CHAT_ENDPOINT, 429, "possibly-sensitive", {}, None)

    monkeypatch.setattr(module, "build_opener", lambda handler: Opener())
    result = module.NvidiaHttpTransport()._request(
        {"model": MODEL}, api_key="fixture-key", timeout_seconds=4.0
    )
    assert result.status_code == 429
    assert result.body == b""


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("secret network message", NvidiaTransportError),
        (TimeoutError("secret"), NvidiaTransportTimeout),
    ],
)
def test_https_transport_converts_network_errors_to_safe_types(
    monkeypatch: pytest.MonkeyPatch, reason: object, expected: type[Exception]
) -> None:
    from urllib.error import URLError

    import llm_routing_eval.adapters.providers.nvidia_build as module

    class Opener:
        def open(self, _request: Any, *, timeout: float) -> Any:
            _ = timeout
            raise URLError(reason)

    monkeypatch.setattr(module, "build_opener", lambda handler: Opener())
    with pytest.raises(expected) as exc:
        module.NvidiaHttpTransport()._request(
            {"model": MODEL}, api_key="fixture-key", timeout_seconds=4.0
        )
    assert "secret" not in str(exc.value)


def test_https_transport_rejects_oversized_response(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_routing_eval.adapters.providers.nvidia_build as module

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self, max_bytes: int) -> bytes:
            return b"x" * max_bytes

    class Opener:
        def open(self, _request: Any, *, timeout: float) -> Response:
            _ = timeout
            return Response()

    monkeypatch.setattr(module, "build_opener", lambda handler: Opener())
    with pytest.raises(NvidiaTransportError, match="NVIDIA_RESPONSE_TOO_LARGE"):
        module.NvidiaHttpTransport()._request(
            {"model": MODEL}, api_key="fixture-key", timeout_seconds=4.0
        )
