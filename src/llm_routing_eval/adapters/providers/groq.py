"""Groq Chat Completions adapter behind the stable Core LLMProviderPort.

Live execution is disabled by default and requires an explicit external
privacy-boundary sign-off. This module never logs prompts or credentials.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from llm_routing_eval.core import (
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    MetricOrigin,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)

from .authorization import DenyAllOutbound, OutboundAuthorization, is_authorized
from .retry_after import checked_hint

GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_MAX_BODY_BYTES = 2_000_000
_MODEL_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_./-]{0,127}$")
_REQUEST_ID = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


@dataclass(frozen=True, slots=True)
class GroqConfig:
    """Provider runtime safety gates and an explicitly reviewed model allowlist."""

    allowed_models: frozenset[str]
    live_enabled: bool = False
    privacy_approved: bool = False

    def __post_init__(self) -> None:
        if not self.allowed_models or any(
            not isinstance(model, str) or _MODEL_ID.fullmatch(model) is None
            for model in self.allowed_models
        ):
            raise ValueError("GROQ_INVALID_MODEL_CONFIGURATION")


@dataclass(frozen=True, slots=True)
class GroqHttpResult:
    status_code: int
    body: bytes
    # Parsed numeric Retry-After seconds for HTTP 429 only; never the raw header.
    retry_after_seconds: float | None = None


class GroqTransportTimeout(Exception):
    """Sanitized timeout sentinel; never carries raw transport information."""


class GroqTransportError(Exception):
    """Sanitized network failure sentinel; never carries raw request information."""


class GroqTransportPort(Protocol):
    async def post_completion(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> GroqHttpResult: ...


class _RejectRedirects(HTTPRedirectHandler):
    """Never forward bearer credentials to a redirect destination."""

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


class GroqHttpTransport:
    """Bounded, no-redirect HTTPS transport; orchestrator owns retry decisions.

    WARNING (review #12 F1): runs in ``asyncio.to_thread``. A cancelled or
    timed-out attempt does NOT stop the thread, so retries can overlap it. It is
    never a default; prefer ``HttpxGroqTransport`` for any live run.
    """

    async def post_completion(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> GroqHttpResult:
        return await asyncio.to_thread(
            self._post, payload, api_key=api_key, timeout_seconds=timeout_seconds
        )

    def _post(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> GroqHttpResult:
        request = Request(
            GROQ_CHAT_ENDPOINT,
            data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        opener = build_opener(_RejectRedirects())
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                body = response.read(_MAX_BODY_BYTES + 1)
                if len(body) > _MAX_BODY_BYTES:
                    raise GroqTransportError("GROQ_RESPONSE_TOO_LARGE")
                return GroqHttpResult(status_code=response.status, body=body)
        except HTTPError as exc:
            # Never read provider-generated errors that could echo private prompt text.
            status = exc.code
            exc.close()
            return GroqHttpResult(status_code=status, body=b"")
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise GroqTransportTimeout from None
            raise GroqTransportError from None
        except TimeoutError:
            raise GroqTransportTimeout from None
        except OSError:
            raise GroqTransportError from None


class GroqProvider:
    """Single-attempt adapter. Caller supplies *already sanitized* ModelRequest.

    Neither a secret-bearing client nor a privacy sanitizer is constructed here;
    the composition root must independently verify outbound privacy enforcement.
    """

    def __init__(
        self,
        config: GroqConfig,
        *,
        transport: GroqTransportPort | None = None,
        outbound_authorization: OutboundAuthorization | None = None,
    ) -> None:
        self._config = config
        # Per-request privacy capability check; deny-all unless a reviewed
        # authorizer is injected (privacy_approved alone is not sufficient).
        self._outbound = (
            outbound_authorization if outbound_authorization is not None else DenyAllOutbound()
        )
        # Review #12 F1: no implicit default. GroqHttpTransport runs urllib in a
        # worker thread that a timeout cannot cancel; live use must inject the
        # cancellable HttpxGroqTransport (see httpx_transport.py).
        self._transport = transport

    async def generate(self, request: ModelRequest) -> ModelResponse:
        start = perf_counter()
        safe_model = (
            request.model if request.model in self._config.allowed_models else "unapproved-model"
        )

        def failure(
            category: ErrorCategory,
            code: str,
            status: CallStatus = CallStatus.FAILED,
            *,
            retryable: bool = False,
            retry_after_seconds: float | None = None,
        ) -> ModelResponse:
            return ModelResponse(
                provider="groq",
                model=safe_model,
                attempt_id=request.attempt_id,
                status=status,
                metric_origin=MetricOrigin.OBSERVED,
                latency_ms=(perf_counter() - start) * 1000,
                error=ErrorInfo(
                    category=category,
                    code=code,
                    retryable=retryable,
                    retry_after_seconds=retry_after_seconds,
                ),
            )

        if not self._config.live_enabled or not self._config.privacy_approved:
            return failure(ErrorCategory.PRIVACY, "GROQ_LIVE_GATE_CLOSED")
        if request.provider != "groq" or request.model not in self._config.allowed_models:
            return failure(ErrorCategory.VALIDATION, "GROQ_UNAPPROVED_MODEL_OR_PROVIDER")
        if set(request.generation_parameters) - {"top_p"}:
            return failure(ErrorCategory.VALIDATION, "GROQ_UNSUPPORTED_GENERATION_PARAMETERS")
        top_p = request.generation_parameters.get("top_p")
        if top_p is not None:
            if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
                return failure(ErrorCategory.VALIDATION, "GROQ_INVALID_TOP_P")
            if not math.isfinite(top_p) or not 0 < top_p <= 1:
                return failure(ErrorCategory.VALIDATION, "GROQ_INVALID_TOP_P")
        # Asked per request, before any credential is read or byte is sent.
        if not is_authorized(self._outbound, request):
            return failure(ErrorCategory.PRIVACY, "GROQ_OUTBOUND_NOT_AUTHORIZED")
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            return failure(ErrorCategory.PROVIDER, "GROQ_API_KEY_MISSING")
        if self._transport is None:
            return failure(ErrorCategory.PROVIDER, "GROQ_TRANSPORT_NOT_CONFIGURED")
        # Non-cancellable to_thread requests can outlive a timed-out attempt and
        # overlap retries. Reject built-in legacy transports even if explicitly
        # injected; reviewed async transports and offline test doubles remain valid.
        if isinstance(self._transport, GroqHttpTransport):
            return failure(ErrorCategory.PROVIDER, "GROQ_UNSAFE_THREAD_TRANSPORT")

        payload: dict[str, object] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_completion_tokens"] = request.max_output_tokens
        if top_p is not None:
            payload["top_p"] = top_p

        try:
            response = await self._transport.post_completion(
                payload, api_key=key, timeout_seconds=request.timeout_seconds
            )
        except GroqTransportTimeout:
            return failure(
                ErrorCategory.PROVIDER_TIMEOUT, "GROQ_TIMEOUT", CallStatus.TIMEOUT, retryable=True
            )
        except Exception:
            # Never expose raw exception text, which may contain key or prompt.
            return failure(ErrorCategory.PROVIDER, "GROQ_TRANSPORT_FAILURE", retryable=True)

        if response.status_code != 200:
            status = response.status_code
            if status == 429:
                return failure(
                    ErrorCategory.RATE_LIMIT,
                    "GROQ_RATE_LIMITED",
                    CallStatus.RATE_LIMITED,
                    retryable=True,
                    retry_after_seconds=checked_hint(response.retry_after_seconds),
                )
            if status in (408, 504):
                return failure(
                    ErrorCategory.PROVIDER_TIMEOUT,
                    "GROQ_HTTP_TIMEOUT",
                    CallStatus.TIMEOUT,
                    retryable=True,
                )
            if status in (401, 403):
                return failure(ErrorCategory.PROVIDER, "GROQ_AUTH_FAILURE")
            if status == 404:
                return failure(ErrorCategory.PROVIDER, "GROQ_MODEL_NOT_AVAILABLE")
            if status in (400, 422):
                return failure(ErrorCategory.VALIDATION, "GROQ_REQUEST_REJECTED")
            return failure(
                ErrorCategory.PROVIDER,
                "GROQ_HTTP_ERROR",
                retryable=500 <= status < 600,
            )

        try:
            parsed = json.loads(response.body)
            if not isinstance(parsed, dict):
                raise ValueError("Invalid root")
            choices = parsed.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("Missing choice")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict):
                raise ValueError("Missing message")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Missing text")
            if choice.get("finish_reason") != "stop":
                return failure(ErrorCategory.PROVIDER, "GROQ_INCOMPLETE_RESPONSE")
            model_returned = parsed.get("model")
            if model_returned is not None and model_returned != request.model:
                return failure(ErrorCategory.PROVIDER, "GROQ_MODEL_MISMATCH")
            request_id = parsed.get("id")
            safe_id = (
                request_id
                if isinstance(request_id, str) and _REQUEST_ID.fullmatch(request_id)
                else None
            )
            return ModelResponse(
                provider="groq",
                model=request.model,
                attempt_id=request.attempt_id,
                status=CallStatus.SUCCESS,
                metric_origin=MetricOrigin.OBSERVED,
                content=content,
                latency_ms=(perf_counter() - start) * 1000,
                usage=_tokens(parsed.get("usage")),
                provider_request_id=safe_id,
            )
        except (ValueError, TypeError, UnicodeDecodeError):
            return failure(ErrorCategory.PROVIDER, "GROQ_MALFORMED_RESPONSE")


def _tokens(raw: object) -> TokenUsage | None:
    if not isinstance(raw, dict):
        return None

    def count(value: object) -> int | None:
        return value if type(value) is int and value >= 0 else None

    prompt = count(raw.get("prompt_tokens"))
    completion = count(raw.get("completion_tokens"))
    detail_in = raw.get("prompt_tokens_details")
    detail_out = raw.get("completion_tokens_details")
    cached = count(detail_in.get("cached_tokens")) if isinstance(detail_in, dict) else None
    reasoning = count(detail_out.get("reasoning_tokens")) if isinstance(detail_out, dict) else None
    if prompt is not None and cached is not None and cached > prompt:
        cached = None
    if all(value is None for value in (prompt, completion, cached, reasoning)):
        return None
    return TokenUsage(
        input_tokens=prompt,
        output_tokens=completion,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )
