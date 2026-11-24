"""Member 02 F5: Cascade policy extension must preserve all legacy decisions.

Synthetic, fully offline and independent of providers, golden labels or I/O.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    RoutingAction,
    RoutingContext,
    RoutingFeedback,
    StrategyId,
)
from llm_routing_eval.routing import (
    CascadeEscalationDecision,
    CascadeRouter,
    ConfiguredCascadeEscalationPolicy,
    ModelPair,
)

MODELS = ModelPair(small="offline-small", large="offline-large")


def _case(scenario_type: str = "account_general") -> BenchmarkCase:
    return BenchmarkCase(
        case_id="synthetic-f5-001",
        input_text="Synthetic account question for offline routing.",
        domain="turkish_banking",
        scenario_type=scenario_type,
        metadata={"synthetic": True},
    )


def _context(
    *,
    stage: int = 1,
    status: CallStatus | None = CallStatus.SUCCESS,
) -> RoutingContext:
    attempts = (
        (
            RoutingFeedback(
                attempt_id="synthetic-attempt-1",
                model=MODELS.small,
                status=status,
            ),
        )
        if status is not None
        else ()
    )
    return RoutingContext(
        current_stage=stage,
        remaining_model_calls=2,
        prior_attempts=attempts,
    )


@pytest.mark.parametrize(
    ("scenario", "status", "scenarios", "on_failure", "expected_model", "reason"),
    [
        (
            "loan_complex",
            CallStatus.SUCCESS,
            frozenset({"loan_complex"}),
            True,
            "offline-large",
            "CASCADE_SCENARIO",
        ),
        (
            "account_general",
            CallStatus.SUCCESS,
            frozenset({"loan_complex"}),
            True,
            None,
            "CASCADE_ACCEPT_SMALL",
        ),
        (
            "account_general",
            CallStatus.FAILED,
            frozenset(),
            True,
            "offline-large",
            "CASCADE_FAILOVER",
        ),
        (
            "account_general",
            CallStatus.FAILED,
            frozenset(),
            False,
            None,
            "CASCADE_STOP_FAILURE",
        ),
        (
            "account_general",
            CallStatus.TIMEOUT,
            frozenset(),
            True,
            "offline-large",
            "CASCADE_FAILOVER",
        ),
        (
            "account_general",
            CallStatus.TIMEOUT,
            frozenset(),
            False,
            None,
            "CASCADE_STOP_FAILURE",
        ),
    ],
)
def test_default_policy_preserves_model_choice_and_reason_code(
    scenario: str,
    status: CallStatus,
    scenarios: frozenset[str],
    on_failure: bool,
    expected_model: str | None,
    reason: str,
) -> None:
    router = CascadeRouter(
        MODELS, escalate_scenarios=scenarios, escalate_on_failure=on_failure
    )
    decision = asyncio.run(router.decide(_case(scenario), _context(status=status)))

    assert decision.strategy is StrategyId.CASCADE
    assert decision.stage_index == 1
    assert decision.selected_model == expected_model
    assert decision.reason_code == reason
    assert decision.action is (
        RoutingAction.INVOKE_MODEL if expected_model else RoutingAction.FINISH
    )


def test_default_policy_stays_frozen_and_validates_scenario_names() -> None:
    with pytest.raises(ValueError, match="Escalation scenarios must be nonempty"):
        ConfiguredCascadeEscalationPolicy(escalate_scenarios=frozenset({"  "}))
    with pytest.raises(ValueError, match="Escalation scenarios must be nonempty"):
        CascadeRouter(MODELS, escalate_scenarios=frozenset({"  "}))


@dataclass
class CustomPolicy:
    """An external extension that needs no changes to CascadeRouter.decide."""

    calls: int = 0

    def evaluate(
        self, case: BenchmarkCase, feedback: RoutingFeedback
    ) -> CascadeEscalationDecision:
        self.calls += 1
        return CascadeEscalationDecision(
            escalate=(
                case.scenario_type == "synthetic_special"
                and feedback.status is CallStatus.SUCCESS
            ),
            reason_code=(
                "CUSTOM_ESCALATE"
                if case.scenario_type == "synthetic_special"
                else "CUSTOM_ACCEPT"
            ),
        )


def test_custom_policy_is_injected_without_changing_router_contract() -> None:
    policy = CustomPolicy()
    router = CascadeRouter(MODELS, escalation_policy=policy)
    initial = asyncio.run(router.decide(_case(), _context(stage=0, status=None)))
    assert initial.selected_model == MODELS.small
    assert initial.reason_code == "CASCADE_SMALL_FIRST"
    assert policy.calls == 0

    selected = asyncio.run(
        router.decide(_case("synthetic_special"), _context(status=CallStatus.SUCCESS))
    )
    assert selected.selected_model == MODELS.large
    assert selected.reason_code == "CUSTOM_ESCALATE"
    assert policy.calls == 1

    finished = asyncio.run(router.decide(_case(), _context(stage=2)))
    assert finished.action is RoutingAction.FINISH
    assert finished.reason_code == "CASCADE_DONE"
    assert policy.calls == 1


def test_custom_policy_configuration_conflicts_are_rejected() -> None:
    policy = CustomPolicy()
    with pytest.raises(ValueError, match="CUSTOM_CASCADE_POLICY_CONFIGURATION_CONFLICT"):
        CascadeRouter(
            MODELS,
            escalation_policy=policy,
            escalate_scenarios=frozenset({"account_general"}),
        )
    with pytest.raises(ValueError, match="CUSTOM_CASCADE_POLICY_CONFIGURATION_CONFLICT"):
        CascadeRouter(MODELS, escalation_policy=policy, escalate_on_failure=False)


@pytest.mark.parametrize(
    ("escalate", "reason"),
    [(1, "CUSTOM_ROUTE"), (True, "secret=account"), (False, ""), (True, "bad reason")],
)
def test_custom_policy_decision_rejects_unsafe_or_invalid_data(
    escalate: object, reason: str
) -> None:
    with pytest.raises(ValueError, match="INVALID_CASCADE_POLICY"):
        CascadeEscalationDecision(escalate=escalate, reason_code=reason)  # type: ignore[arg-type]


def test_custom_policy_must_return_typed_decision() -> None:
    class InvalidPolicy:
        def evaluate(
            self, case: BenchmarkCase, feedback: RoutingFeedback
        ) -> CascadeEscalationDecision:
            return "unsafe free text"  # type: ignore[return-value]

    router = CascadeRouter(MODELS, escalation_policy=InvalidPolicy())
    with pytest.raises(ValueError, match="INVALID_CASCADE_POLICY_DECISION"):
        asyncio.run(router.decide(_case(), _context()))


def test_existing_stage_boundaries_skip_policy_execution() -> None:
    class ShouldNotRun:
        def evaluate(
            self, case: BenchmarkCase, feedback: RoutingFeedback
        ) -> CascadeEscalationDecision:
            raise AssertionError("Policy must not run outside stage one")

    router = CascadeRouter(MODELS, escalation_policy=ShouldNotRun())
    first = asyncio.run(router.decide(_case(), _context(stage=0, status=None)))
    done = asyncio.run(router.decide(_case(), _context(stage=2)))
    malformed = asyncio.run(router.decide(_case(), _context(stage=1, status=None)))

    assert first.reason_code == "CASCADE_SMALL_FIRST"
    assert done.reason_code == "CASCADE_DONE"
    assert malformed.reason_code == "CASCADE_DONE"
