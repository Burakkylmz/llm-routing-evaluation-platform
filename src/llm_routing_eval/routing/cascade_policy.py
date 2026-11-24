"""Pluggable, deterministic escalation decision policy for offline Cascade routing.

Policies see only the already-sanitized BenchmarkCase and prior call feedback.
They have no access to golden labels, evaluator results or provider transports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from llm_routing_eval.core import BenchmarkCase, CallStatus, RoutingFeedback

_SAFE_CASCADE_REASON = re.compile(r"^[A-Z][A-Z0-9_]{0,79}$")


@dataclass(frozen=True, slots=True)
class CascadeEscalationDecision:
    """Validated, PII-safe routing outcome from an escalation policy."""

    escalate: bool
    reason_code: str

    def __post_init__(self) -> None:
        if type(self.escalate) is not bool or not isinstance(self.reason_code, str):
            raise ValueError("INVALID_CASCADE_POLICY_DECISION")
        if _SAFE_CASCADE_REASON.fullmatch(self.reason_code) is None:
            raise ValueError("INVALID_CASCADE_POLICY_REASON")


class CascadeEscalationPolicy(Protocol):
    """Extension point: choose whether stage one should escalate."""

    def evaluate(
        self, case: BenchmarkCase, feedback: RoutingFeedback
    ) -> CascadeEscalationDecision: ...


@dataclass(frozen=True, slots=True)
class ConfiguredCascadeEscalationPolicy:
    """Preserve the current failover and scenario policy, including reason codes."""

    escalate_scenarios: frozenset[str] = frozenset()
    escalate_on_failure: bool = True

    def __post_init__(self) -> None:
        if any(not name.strip() for name in self.escalate_scenarios):
            raise ValueError("Escalation scenarios must be nonempty")

    def evaluate(self, case: BenchmarkCase, feedback: RoutingFeedback) -> CascadeEscalationDecision:
        if feedback.status is CallStatus.SUCCESS:
            escalate = case.scenario_type in self.escalate_scenarios
            return CascadeEscalationDecision(
                escalate=escalate,
                reason_code="CASCADE_SCENARIO" if escalate else "CASCADE_ACCEPT_SMALL",
            )
        escalate = self.escalate_on_failure
        return CascadeEscalationDecision(
            escalate=escalate,
            reason_code="CASCADE_FAILOVER" if escalate else "CASCADE_STOP_FAILURE",
        )
