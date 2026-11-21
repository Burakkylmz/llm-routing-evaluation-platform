"""Regression tests: report fingerprints identify the *effective* routing policy."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from llm_routing_eval.application import RunOptions
from llm_routing_eval.cli.configuration import (
    OfflineSettings,
    demo_options,
    read_settings,
    routing_policy_digest,
)
from llm_routing_eval.core import StrategyId

CONFIG = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def _settings() -> OfflineSettings:
    return read_settings(CONFIG)


def _mutated_settings(change: Any) -> OfflineSettings:
    raw = deepcopy(_settings().model_dump(mode="python"))
    change(raw)
    return OfflineSettings.model_validate(raw)


def _digest(settings: OfflineSettings, strategy: StrategyId) -> str:
    return demo_options(settings, strategy_override=strategy).fingerprint()


@pytest.mark.parametrize(
    "strategy",
    list(StrategyId),
)
def test_each_strategy_has_a_stable_non_disclosing_digest(strategy: StrategyId) -> None:
    original = _settings()
    options = demo_options(original, strategy_override=strategy)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", options.routing_policy_digest or "")
    assert options.routing_policy_digest == routing_policy_digest(original, strategy)
    assert _digest(original, strategy) == _digest(read_settings(CONFIG), strategy)
    assert "hesap" not in str(options.model_dump(mode="json")).lower()
    assert "example_id" not in str(options.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("strategy", "change"),
    [
        (StrategyId.RULE_BASED, lambda x: x["routing"]["rule_based"].update(default_tier="small")),
        (
            StrategyId.RULE_BASED,
            lambda x: x["routing"]["rule_based"].update(
                rules=tuple(reversed(x["routing"]["rule_based"]["rules"]))
            ),
        ),
        (StrategyId.KNN, lambda x: x["routing"]["knn"].update(k=2)),
        (StrategyId.KNN, lambda x: x["routing"]["knn"].update(minimum_similarity=0.8)),
        (StrategyId.KNN, lambda x: x["routing"]["knn"].update(embedding_dimensions=128)),
        (
            StrategyId.KNN,
            lambda x: x["routing"]["knn"]["examples"][0].update(
                input_text="different private synthetic training text"
            ),
        ),
        (
            StrategyId.KNN,
            lambda x: x["routing"]["knn"]["examples"][0].update(tier="large"),
        ),
        (
            StrategyId.CASCADE,
            lambda x: x["routing"]["cascade"].update(escalate_scenarios=[]),
        ),
        (
            StrategyId.CASCADE,
            lambda x: x["routing"]["cascade"].update(escalate_on_failure=False),
        ),
    ],
)
def test_effective_policy_change_updates_report_fingerprint(
    strategy: StrategyId, change: Any
) -> None:
    baseline = _digest(_settings(), strategy)
    modified = _mutated_settings(change)
    assert baseline != _digest(modified, strategy)


@pytest.mark.parametrize("strategy", list(StrategyId))
def test_changing_model_pair_updates_fingerprint(strategy: StrategyId) -> None:
    modified = _mutated_settings(lambda x: x["routing"]["models"].update(small="new-small"))
    assert _digest(_settings(), strategy) != _digest(modified, strategy)


@pytest.mark.parametrize(
    "strategy",
    [
        StrategyId.ALWAYS_LARGE,
        StrategyId.KNN,
        StrategyId.CASCADE,
    ],
)
def test_irrelevant_rule_policy_changes_do_not_change_fingerprint(strategy: StrategyId) -> None:
    modified = _mutated_settings(lambda x: x["routing"]["rule_based"].update(default_tier="small"))
    assert _digest(_settings(), strategy) == _digest(modified, strategy)


def test_cli_override_hashes_selected_strategy_not_unselected_yaml_policy() -> None:
    original = _settings()
    modified = _mutated_settings(lambda x: x["benchmark"].update(strategy="rule_based"))
    assert _digest(original, StrategyId.CASCADE) == _digest(modified, StrategyId.CASCADE)


def test_cascade_scenario_order_is_semantically_ignored() -> None:
    modified = _mutated_settings(
        lambda x: x["routing"]["cascade"].update(
            escalate_scenarios=["fees_general", "account_general"]
        )
    )
    reordered = _mutated_settings(
        lambda x: x["routing"]["cascade"].update(
            escalate_scenarios=["account_general", "fees_general"]
        )
    )
    assert _digest(modified, StrategyId.CASCADE) == _digest(reordered, StrategyId.CASCADE)


def test_fingerprint_is_not_just_a_hash_of_the_selected_policy() -> None:
    initial = demo_options(_settings(), strategy_override=StrategyId.CASCADE)
    changed_calls = initial.model_copy(update={"max_model_calls_per_case": 3})
    assert initial.fingerprint() != changed_calls.fingerprint()


def test_invalid_digest_is_rejected_by_runoptions() -> None:
    with pytest.raises(ValidationError):
        RunOptions(routing_policy_digest="raw development example text")


def test_legacy_programmatic_runs_remain_backwards_compatible() -> None:
    options = RunOptions()
    assert options.routing_policy_digest is None
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", options.fingerprint())
