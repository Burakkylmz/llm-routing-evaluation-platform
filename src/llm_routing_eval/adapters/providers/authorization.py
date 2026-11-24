"""Per-request outbound authorization for live provider calls (deny by default).

``GroqConfig.privacy_approved`` / ``NvidiaBuildConfig.privacy_approved`` are a
one-time composition sign-off: once set, every prompt could leave the process.
An ``OutboundAuthorization`` adds a narrowly scoped capability check that is
asked for **each** request, after the provider's own validation and before any
credential is read or any byte is sent.

Only an independently reviewed privacy adapter (Member 04) may implement a
permissive authorizer. Until then the default is ``DenyAllOutbound``, so live
calls fail closed even when keys, ``live_enabled`` and ``privacy_approved`` are
all present. Authorizer exceptions are treated as a denial.
"""

from __future__ import annotations

from typing import Protocol

from llm_routing_eval.core import ModelRequest


class OutboundAuthorization(Protocol):
    """Decide whether this already-sanitized request may be sent to the provider.

    Implementations must not log or persist the prompt, and must not mutate it.
    """

    def authorize(self, request: ModelRequest) -> bool: ...


class DenyAllOutbound:
    """Safe default until a reviewed privacy authorizer is composed in."""

    def authorize(self, request: ModelRequest) -> bool:
        _ = request
        return False


def is_authorized(authorization: OutboundAuthorization, request: ModelRequest) -> bool:
    """Fail closed: only a literal ``True`` permits; exceptions deny."""
    try:
        return authorization.authorize(request) is True
    except Exception:
        return False
