"""F7: routing fingerprints should reflect effective policies, not incidental ordering."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from llm_routing_eval.cli.configuration import (
    OfflineSettings,
    read_settings,
    routing_policy_digest,
)
from llm_routing_eval.core import StrategyId

CONFIG = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def _settings() -> OfflineSettings:
    return read_settings(CONFIG)


def _changed(change: Callable[[dict[str, Any]], None]) -> OfflineSettings:
    raw = copy.deepcopy(_settings().model_dump(mode="python"))
    change(raw)
    return OfflineSettings.model_validate(raw)


def _digest(settings: OfflineSettings, strategy: StrategyId) -> str:
    return routing_policy_digest(settings, strategy)


def test_rule_scenario_membership_order_is_not_a_policy_change() -> None:
    def extend_and_reorder(raw: dict[str, Any]) -> None:
        raw["routing"]["rule_based"]["rules"][0]["scenario_types"] = (
            "account_general",
            "fees_general",
            "account_general",
        )

    baseline = _changed(extend_and_reorder)

    def different_order(raw: dict[str, Any]) -> None:
        raw["routing"]["rule_based"]["rules"][0]["scenario_types"] = (
            "fees_general",
            "account_general",
        )

    equivalent = _changed(different_order)
    assert _digest(baseline, StrategyId.RULE_BASED) == _digest(equivalent, StrategyId.RULE_BASED)


def test_rule_priority_order_remains_material_to_fingerprint() -> None:
    reordered = _changed(
        lambda raw: raw["routing"]["rule_based"].update(
            rules=tuple(reversed(raw["routing"]["rule_based"]["rules"]))
        )
    )
    assert _digest(_settings(), StrategyId.RULE_BASED) != _digest(reordered, StrategyId.RULE_BASED)


def test_rule_scenario_membership_change_remains_material() -> None:
    changed = _changed(
        lambda raw: raw["routing"]["rule_based"]["rules"][0].update(
            scenario_types=("fees_general",)
        )
    )
    assert _digest(_settings(), StrategyId.RULE_BASED) != _digest(changed, StrategyId.RULE_BASED)


def test_knn_development_list_order_is_not_a_policy_change() -> None:
    reversed_examples = _changed(
        lambda raw: raw["routing"]["knn"].update(
            examples=tuple(reversed(raw["routing"]["knn"]["examples"]))
        )
    )
    assert _digest(_settings(), StrategyId.KNN) == _digest(reversed_examples, StrategyId.KNN)


@pytest.mark.parametrize("field", ["example_id", "input_text", "tier"])
def test_knn_example_identity_or_content_change_remains_material(field: str) -> None:
    new_values = {
        "example_id": "renamed-training-example",
        "input_text": "Different synthetic calibration phrase",
        "tier": "large",
    }
    changed = _changed(
        lambda raw: raw["routing"]["knn"]["examples"][0].update({field: new_values[field]})
    )
    assert _digest(_settings(), StrategyId.KNN) != _digest(changed, StrategyId.KNN)


def test_knn_similarity_threshold_change_remains_material() -> None:
    changed = _changed(lambda raw: raw["routing"]["knn"].update(minimum_similarity=0.73))
    assert _digest(_settings(), StrategyId.KNN) != _digest(changed, StrategyId.KNN)


@pytest.mark.parametrize("strategy", list(StrategyId))
def test_selected_policy_canonicalization_still_keeps_model_identity(
    strategy: StrategyId,
) -> None:
    changed = _changed(lambda raw: raw["routing"]["models"].update(small="other-offline-small"))
    assert _digest(_settings(), strategy) != _digest(changed, strategy)


@pytest.mark.parametrize("strategy", list(StrategyId))
def test_irrelevant_knn_list_order_does_not_affect_other_strategies(
    strategy: StrategyId,
) -> None:
    reordered = _changed(
        lambda raw: raw["routing"]["knn"].update(
            examples=tuple(reversed(raw["routing"]["knn"]["examples"]))
        )
    )
    assert _digest(_settings(), strategy) == _digest(reordered, strategy)
