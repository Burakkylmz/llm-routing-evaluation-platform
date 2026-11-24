"""Gating for opt-in live provider tests. Every live test is SKIPPED unless explicitly enabled.

Enable only on an operator machine after the account owner has confirmed model
access and the privacy owner has approved a synthetic-only pilot:

    LLM_ROUTING_LIVE_TESTS=1 GROQ_API_KEY=... uv run pytest -m live tests/live -q

Live tests refuse to run in CI (``CI`` set or ``BENCHMARK_MODE=offline``), even
if the opt-in flag is present.
"""

from __future__ import annotations

import os

import pytest

from llm_routing_eval.core import ModelRequest

# The ONLY prompt any live test may send. Synthetic, non-customer, no PII.
SYNTHETIC_PROMPT = "Reply with the single word: OK"


def require_live(*required_env: str) -> None:
    """Skip unless live tests are explicitly enabled outside CI and env is present."""
    if os.getenv("LLM_ROUTING_LIVE_TESTS") != "1":
        pytest.skip("live tests disabled: set LLM_ROUTING_LIVE_TESTS=1 to opt in")
    if os.getenv("CI") or os.getenv("BENCHMARK_MODE", "").lower() == "offline":
        pytest.skip("live tests never run in CI or offline benchmark mode")
    missing = [name for name in required_env if not os.getenv(name, "").strip()]
    if missing:
        pytest.skip("missing live-test environment: " + ",".join(missing))


class FixedSyntheticPromptOnly:
    """OutboundAuthorization that permits exactly one hard-coded synthetic prompt."""

    def __init__(self) -> None:
        self.authorized = 0

    def authorize(self, request: ModelRequest) -> bool:
        if request.prompt != SYNTHETIC_PROMPT:
            return False
        self.authorized += 1
        return True
