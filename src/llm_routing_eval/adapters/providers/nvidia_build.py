"""NVIDIA Build hosted NIM chat adapter, isolated behind LLMProviderPort.

No live traffic without both an explicit runtime opt-in and an approved
upstream privacy boundary. This module never logs credentials or model text.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)

from llm_routing_eval.core import (
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)

NVIDIA_CHAT_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
_MAX_RESPONSE_BYTES = 2_000_000
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@dataclass(frozen=True, slots=True)
class NvidiaBuildConfig:
    """Explicitly approved model IDs; no unverified provider defaults."""

    allowed_models: frozenset[str]
    live_enabled: bool = False
    privacy_approved: bool = False

    def __post_init__(self) -> None:
        if not self.allowed_models or any(
            not isinstance(model, str) or _MODEL_PATTERN.fullmatch(model) is None
            for model in self.allowed_models
        ):
            raise ValueError("NVIDIA_MODEL_CONFIGURATION_INVALID")


@dataclass(frozen=True, slots=True)
class NvidiaHttpResult:
    status_code: int
    body: bytes


class NvidiaTransportTimeout(Exception):
    """Transport timed out. Original exception details are never surfaced."""


class NvidiaTransportError(Exception):
    """A sanitized transport error with no request/credential details."""


class NvidiaTransportPort(Protocol):
    async def post_chat(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> NvidiaHttpResult: ...


class _RejectRedirects(HTTPRedirectHandler):
    """Never forward the bearer token to an HTTP redirect target."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class NvidiaHttpTransport:
    """Stdlib HTTPS transport; async wrapper keeps the application nonblocking.

    Does not follow redirects, restricts response size, and never reads error
    response bodies. The orchestrator, not the transport, controls retries.
    """

    async def post_chat(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> NvidiaHttpResult:
        return await asyncio.to_thread(
            self._request,
            payload,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )

    def _request(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> NvidiaHttpResult:
        request = Request(
            NVIDIA_CHAT_ENDPOINT,
            data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        opener = build_opener(_RejectRedirects())
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                body = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(body) > _MAX_RESPONSE_BYTES:
                    raise NvidiaTransportError("NVIDIA_RESPONSE_TOO_LARGE")
                return NvidiaHttpResult(status_code=response.status, body=body)
        except HTTPError as exc:
            # Do not read or expose provider error bodies (may echo prompts).
            code = exc.code
            exc.close()
            return NvidiaHttpResult(status_code=code, body=b"")
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise NvidiaTransportTimeout from None
            raise NvidiaTransportError from None
        except TimeoutError:
            raise NvidiaTransportTimeout from None
        except OSError:
            raise NvidiaTransportError from None


class NvidiaBuildProvider:
    """Provider adapter. Composition must explicitly establish privacy approval."""

    def __init__(
        self,
        config: NvidiaBuildConfig,
        *,
        transport: NvidiaTransportPort | None = None,
    ) -> None:
        self._config = config
        self._transport = transport if transport is not None else NvidiaHttpTransport()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        started = perf_counter()

        def failed(
            category: ErrorCategory,
            code: str,
            status: CallStatus = CallStatus.FAILED,
            *,
            retryable: bool = False,
        ) -> ModelResponse:
            return ModelResponse(
                provider="nvidia_build",
                model=request.model,
                attempt_id=request.attempt_id,
                status=status,
                metric_origin=MetricOrigin.OBSERVED,
                latency_ms=(perf_counter() - started) * 1000,
                error=ErrorInfo(category=category, code=code, retryable=retryable),
            )

        if not self._config.live_enabled or not self._config.privacy_approved:
            return failed(ErrorCategory.PRIVACY, "NVIDIA_LIVE_GATE_CLOSED")
        if request.provider != "nvidia_build" or request.model not in self._config.allowed_models:
            return failed(ErrorCategory.VALIDATION, "NVIDIA_UNAPPROVED_MODEL_OR_PROVIDER")
        if set(request.generation_parameters) - {"top_p"}:
            return failed(ErrorCategory.VALIDATION, "NVIDIA_UNSUPPORTED_PARAMETERS")
        top_p = request.generation_parameters.get("top_p")
        if top_p is not None and (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not 0 < top_p <= 1
        ):
            return failed(ErrorCategory.VALIDATION, "NVIDIA_INVALID_TOP_P")

        key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if not key:
            return failed(ErrorCategory.PROVIDER, "NVIDIA_API_KEY_MISSING")

        payload: dict[str, object] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if top_p is not None:
            payload["top_p"] = top_p

        try:
            result = await self._transport.post_chat(
                payload, api_key=key, timeout_seconds=request.timeout_seconds
            )
        except NvidiaTransportTimeout:
            return failed(
                ErrorCategory.PROVIDER_TIMEOUT,
                "NVIDIA_TIMEOUT",
                CallStatus.TIMEOUT,
                retryable=True,
            )
        except Exception:
            # Fail closed: never expose exception repr, URL, prompt or key.
            return failed(ErrorCategory.PROVIDER, "NVIDIA_TRANSPORT_ERROR", retryable=True)

        if result.status_code != 200:
            if result.status_code == 429:
                return failed(
                    ErrorCategory.RATE_LIMIT,
                    "NVIDIA_RATE_LIMITED",
                    CallStatus.RATE_LIMITED,
                    retryable=True,
                )
            if result.status_code in (408, 504):
                return failed(
                    ErrorCategory.PROVIDER_TIMEOUT,
                    "NVIDIA_HTTP_TIMEOUT",
                    CallStatus.TIMEOUT,
                    retryable=True,
                )
            if result.status_code in (401, 403):
                return failed(ErrorCategory.PROVIDER, "NVIDIA_AUTH_FAILED")
            if result.status_code == 404:
                return failed(ErrorCategory.PROVIDER, "NVIDIA_MODEL_UNAVAILABLE")
            if result.status_code in (400, 422):
                return failed(ErrorCategory.VALIDATION, "NVIDIA_REQUEST_REJECTED")
            return failed(
                ErrorCategory.PROVIDER,
                "NVIDIA_HTTP_ERROR",
                retryable=500 <= result.status_code < 600,
            )

        try:
            data = json.loads(result.body)
            if not isinstance(data, dict):
                raise ValueError("Invalid completion root")
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("No choices")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise ValueError("Invalid choice")
            message = choice.get("message")
            if not isinstance(message, dict):
                raise ValueError("Invalid message")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("No textual completion")
            if choice.get("finish_reason") != "stop":
                return failed(ErrorCategory.PROVIDER, "NVIDIA_INCOMPLETE_COMPLETION")
            usage = _usage(data.get("usage"))
            raw_id = data.get("id")
            request_id = (
                raw_id
                if isinstance(raw_id, str) and _REQUEST_ID_PATTERN.fullmatch(raw_id)
                else None
            )
            return ModelResponse(
                provider="nvidia_build",
                model=request.model,
                attempt_id=request.attempt_id,
                status=CallStatus.SUCCESS,
                metric_origin=MetricOrigin.OBSERVED,
                content=content,
                usage=usage,
                latency_ms=(perf_counter() - started) * 1000,
                provider_request_id=request_id,
            )
        except (ValueError, TypeError, UnicodeDecodeError):
            return failed(ErrorCategory.PROVIDER, "NVIDIA_MALFORMED_COMPLETION")


def _usage(raw: object) -> TokenUsage | None:
    """Unknown/invalid usage stays unknown, never silently becomes zero."""
    if not isinstance(raw, dict):
        return None

    def count(value: object) -> int | None:
        return value if type(value) is int and value >= 0 else None

    prompt = count(raw.get("prompt_tokens"))
    completion = count(raw.get("completion_tokens"))
    prompt_details = raw.get("prompt_tokens_details")
    completion_details = raw.get("completion_tokens_details")
    cached = (
        count(prompt_details.get("cached_tokens"))
        if isinstance(prompt_details, dict)
        else None
    )
    reasoning = (
        count(completion_details.get("reasoning_tokens"))
        if isinstance(completion_details, dict)
        else None
    )
    if all(value is None for value in (prompt, completion, cached, reasoning)):
        return None
    if prompt is not None and cached is not None and cached > prompt:
        cached = None
    return TokenUsage(
        input_tokens=prompt,
        output_tokens=completion,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )
