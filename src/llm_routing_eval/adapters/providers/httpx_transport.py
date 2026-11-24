"""Cancellable native-async HTTPS transports for Groq and NVIDIA Build (review #12 F1).

The stdlib transports run urllib in ``asyncio.to_thread``. When the orchestrator's
``asyncio.wait_for`` times out, the awaiting task is cancelled but the worker
thread keeps the HTTP request running (urllib's timeout applies per socket
operation, not per request), so a retry overlaps the abandoned call.

These transports use one shared, externally owned ``httpx.AsyncClient``.
Cancelling the awaiting coroutine closes the in-flight stream before the
cancellation completes, so no request outlives its attempt. Behaviour otherwise
matches the stdlib transports: fixed HTTPS endpoint, no redirects, no proxy
environment variables, error bodies never read, bounded success bodies and
sanitized exceptions only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx

from .groq import (
    _MAX_BODY_BYTES as _GROQ_MAX_BYTES,
)
from .groq import (
    GROQ_CHAT_ENDPOINT,
    GroqHttpResult,
    GroqTransportError,
    GroqTransportTimeout,
)
from .nvidia_build import (
    _MAX_RESPONSE_BYTES as _NVIDIA_MAX_BYTES,
)
from .nvidia_build import (
    NVIDIA_CHAT_ENDPOINT,
    NvidiaHttpResult,
    NvidiaTransportError,
    NvidiaTransportTimeout,
)

_CHUNK = 65_536


async def _post(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: Mapping[str, object],
    *,
    api_key: str,
    timeout_seconds: float,
    max_bytes: int,
) -> tuple[int, bytes] | None:
    """Return (status, body), or None if the success body exceeds ``max_bytes``.

    Raises ``httpx.TimeoutException`` / ``httpx.TransportError`` for the caller
    to map onto sanitized provider exceptions.
    """
    async with client.stream(
        "POST",
        endpoint,
        json=dict(payload),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
        timeout=timeout_seconds,
        follow_redirects=False,
    ) as response:
        if response.status_code != 200:
            # Never read provider error bodies; they may echo prompt text.
            return response.status_code, b""
        body = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=_CHUNK):
            if len(body) + len(chunk) > max_bytes:
                return None
            body.extend(chunk)
        return 200, bytes(body)


class HttpxGroqTransport:
    """``GroqTransportPort`` over an injected, externally managed AsyncClient."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def post_completion(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> GroqHttpResult:
        try:
            result = await _post(
                self._client,
                GROQ_CHAT_ENDPOINT,
                payload,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_bytes=_GROQ_MAX_BYTES,
            )
        except httpx.TimeoutException:
            raise GroqTransportTimeout from None
        except httpx.HTTPError:
            raise GroqTransportError from None
        if result is None:
            raise GroqTransportError("GROQ_RESPONSE_TOO_LARGE")
        return GroqHttpResult(status_code=result[0], body=result[1])


class HttpxNvidiaTransport:
    """``NvidiaTransportPort`` over an injected, externally managed AsyncClient."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def post_chat(
        self,
        payload: Mapping[str, object],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> NvidiaHttpResult:
        try:
            result = await _post(
                self._client,
                NVIDIA_CHAT_ENDPOINT,
                payload,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_bytes=_NVIDIA_MAX_BYTES,
            )
        except httpx.TimeoutException:
            raise NvidiaTransportTimeout from None
        except httpx.HTTPError:
            raise NvidiaTransportError from None
        if result is None:
            raise NvidiaTransportError("NVIDIA_RESPONSE_TOO_LARGE")
        return NvidiaHttpResult(status_code=result[0], body=result[1])


@asynccontextmanager
async def managed_httpx_client(
    *,
    max_connections: int = 4,
    mock_transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    """One TLS-verified pool for a whole run; inject ``mock_transport`` only in tests."""
    if not 1 <= max_connections <= 32:
        raise ValueError("HTTPX_POOL_CONNECTIONS_INVALID")
    async with httpx.AsyncClient(
        verify=True,
        trust_env=False,
        follow_redirects=False,
        limits=httpx.Limits(
            max_connections=max_connections, max_keepalive_connections=max_connections
        ),
        transport=mock_transport,
    ) as client:
        yield client
