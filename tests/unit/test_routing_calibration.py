"""Development-only kNN/Cascade calibration (pure routing layer)."""

from __future__ import annotations

import asyncio
import random

import pytest

from llm_routing_eval.routing import calibration as calibration_module
from llm_routing_eval.routing.calibration import (
    MAX_CALIBRATION_EXAMPLES,
    CalibrationExample,
    calibrate_cascade,
    calibrate_knn,
)
from llm_routing_eval.routing.strategies import knn_vote


class TableEmbedding:
    """Hand-built vectors so leave-one-out outcomes are exactly predictable."""

    def __init__(self, table: dict[str, tuple[float, ...]]) -> None:
        self._table = table

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._table[text] for text in texts)


def ex(i: int, text: str, tier: str, scenario: str = "s") -> CalibrationExample:
    return CalibrationExample(
        example_id=f"e{i}",
        input_text=text,
        scenario_type=scenario,
        tier=tier,  # type: ignore[arg-type]
    )


def test_leave_one_out_selects_zero_under_routing_and_best_accuracy() -> None:
    examples = (
        ex(1, "a1", "small"),
        ex(2, "a2", "small"),
        ex(3, "b1", "large"),
        ex(4, "b2", "large"),
        ex(5, "c", "large"),  # isolated large example: only safe via low-similarity fallback
    )
    table = {
        "a1": (1.0, 0.0, 0.0),
        "a2": (0.95, 0.05, 0.0),
        "b1": (0.0, 1.0, 0.0),
        "b2": (0.05, 0.95, 0.0),
        "c": (0.6, 0.0, 0.8),
    }
    result = asyncio.run(
        calibrate_knn(
            examples,
            TableEmbedding(table),
            k_values=(1,),
            thresholds=(0.0, 0.9),
            max_under_routing_rate=0.0,
        )
    )
    by_threshold = {p.minimum_similarity: p for p in result.grid}
    assert by_threshold[0.0].under_routed == 1  # "c" nearest neighbour is small
    assert by_threshold[0.9].under_routed == 0 and by_threshold[0.9].correct == 5
    assert result.feasible and result.selected == by_threshold[0.9]


def test_infeasible_ceiling_selects_nothing() -> None:
    examples = (ex(1, "a", "small"), ex(2, "b", "large"))
    table = {"a": (1.0, 0.0), "b": (1.0, 0.01)}
    result = asyncio.run(
        calibrate_knn(
            examples,
            TableEmbedding(table),
            k_values=(1,),
            thresholds=(0.0,),
            max_under_routing_rate=0.0,
        )
    )
    assert not result.feasible and result.selected is None


def test_grid_matches_naive_per_point_knn_vote() -> None:
    rng = random.Random(7)
    examples = tuple(ex(i, f"t{i}", rng.choice(["small", "large"])) for i in range(25))
    table = {f"t{i}": tuple(rng.uniform(-1, 1) for _ in range(6)) for i in range(25)}
    result = asyncio.run(
        calibrate_knn(
            examples,
            TableEmbedding(table),
            k_values=(1, 3, 5),
            thresholds=(0.0, 0.3, 0.6),
            max_under_routing_rate=1.0,
        )
    )
    refs = [(table[e.input_text], e.example_id, e.tier) for e in examples]
    for point in result.grid:
        correct = under = 0
        for i, example in enumerate(examples):
            predicted, _ = knn_vote(
                refs[i][0],
                refs[:i] + refs[i + 1 :],
                k=point.k,
                minimum_similarity=point.minimum_similarity,
            )
            correct += predicted == example.tier
            under += predicted == "small" and example.tier == "large"
        assert (point.correct, point.under_routed) == (correct, under)


def test_neighbour_ranking_is_computed_once_per_example(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = calibration_module.rank_neighbours

    def counting(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(calibration_module, "rank_neighbours", counting)
    examples = tuple(ex(i, f"t{i}", "small" if i % 2 else "large") for i in range(10))
    table = {f"t{i}": (1.0, float(i)) for i in range(10)}
    result = asyncio.run(calibrate_knn(examples, TableEmbedding(table)))
    assert len(result.grid) > 1
    assert calls == len(examples)


def test_thresholds_outside_settings_range_are_dropped() -> None:
    examples = (ex(1, "a", "small"), ex(2, "b", "large"))
    table = {"a": (1.0, 0.0), "b": (0.0, 1.0)}
    result = asyncio.run(
        calibrate_knn(examples, TableEmbedding(table), k_values=(1,), thresholds=(-0.5, 0.5, 1.5))
    )
    assert {p.minimum_similarity for p in result.grid} == {0.5}
    with pytest.raises(ValueError, match="CALIBRATION_GRID_EMPTY"):
        asyncio.run(calibrate_knn(examples, TableEmbedding(table), k_values=(2, 3)))
    with pytest.raises(ValueError, match="UNDER_ROUTING_CEILING_INVALID"):
        asyncio.run(
            calibrate_knn(examples, TableEmbedding(table), max_under_routing_rate=float("nan"))
        )


def test_cascade_escalates_risky_and_under_evidenced_scenarios() -> None:
    examples = (
        *(ex(i, f"t{i}", "small", "easy") for i in range(5)),
        ex(10, "h1", "small", "mixed"),
        ex(11, "h2", "small", "mixed"),
        ex(12, "h3", "large", "mixed"),
        ex(20, "r1", "small", "rare"),
    )
    result = calibrate_cascade(examples, max_small_risk=0.2, min_scenario_examples=3)
    assert {s.scenario_type: s.reason for s in result.scenarios} == {
        "easy": "ACCEPT_SMALL",
        "mixed": "LARGE_SHARE_ABOVE_CEILING",
        "rare": "INSUFFICIENT_EVIDENCE",
    }
    assert result.escalate_scenarios == ("mixed", "rare")
    with pytest.raises(ValueError, match="SMALL_RISK_CEILING_INVALID"):
        calibrate_cascade(examples, max_small_risk=1.5)
    with pytest.raises(ValueError, match="MIN_SCENARIO_EXAMPLES_INVALID"):
        calibrate_cascade(examples, min_scenario_examples=0)


def test_cascade_preserves_unseen_escalations_and_updates_observed_policy() -> None:
    examples = (
        ex(1, "safe1", "small", "observed_safe"),
        ex(2, "safe2", "small", "observed_safe"),
        ex(3, "safe3", "small", "observed_safe"),
    )
    result = calibrate_cascade(
        examples,
        known_scenarios=("unseen_guard", "observed_safe", "unseen_guard"),
    )
    evidence = {item.scenario_type: item for item in result.scenarios}
    assert result.escalate_scenarios == ("unseen_guard",)
    assert (evidence["unseen_guard"].examples, evidence["unseen_guard"].large_labels) == (0, 0)
    assert evidence["unseen_guard"].reason == "INSUFFICIENT_EVIDENCE"
    assert evidence["observed_safe"].reason == "ACCEPT_SMALL"
    assert evidence["observed_safe"].escalate is False

    with pytest.raises(ValueError, match="CALIBRATION_SCENARIO_EMPTY"):
        calibrate_cascade(examples, known_scenarios=("",))


def test_example_and_dataset_validation() -> None:
    with pytest.raises(ValueError, match="HELD_OUT_DATA_FORBIDDEN_IN_CALIBRATION"):
        CalibrationExample("x", "t", "s", "small", split="test")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="CALIBRATION_TIER_INVALID"):
        CalibrationExample("x", "t", "s", "medium")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="CALIBRATION_EXAMPLE_EMPTY"):
        CalibrationExample("x", "  ", "s", "small")
    with pytest.raises(ValueError, match="CALIBRATION_DUPLICATE_EXAMPLE_ID"):
        calibrate_cascade((ex(1, "a", "small"), ex(1, "b", "large")))
    with pytest.raises(ValueError, match="CALIBRATION_EXAMPLE_COUNT_INVALID"):
        calibrate_cascade((ex(1, "a", "small"),))
    too_many = tuple(ex(i, f"t{i}", "small") for i in range(MAX_CALIBRATION_EXAMPLES + 1))
    with pytest.raises(ValueError, match="CALIBRATION_EXAMPLE_COUNT_INVALID"):
        calibrate_cascade(too_many)
