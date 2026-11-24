"""Development-only calibration of kNN and Cascade routing policy (pure, offline).

Inputs are labeled *development* examples. Each example's ``tier`` is the
smallest model tier judged sufficient for it. Held-out evaluation/golden data
must never be passed here; the CLI additionally enforces the F4 isolation guard
against the active evaluation dataset.

Outputs contain only IDs, counts and rates, never example text.

* kNN: leave-one-out grid search over ``k`` x ``minimum_similarity``, using the
  same ``knn_vote`` logic as ``KnnRouter``. Under-routing (a large-labeled
  example sent to the small model, i.e. quality risk) must stay within a
  ceiling; among feasible settings the most accurate wins, then fewer
  under-routed, then the higher (more conservative) threshold, then smaller k.
* Cascade: a scenario escalates after a successful small answer when its
  large-label share exceeds a risk ceiling, or when it has too few examples.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from llm_routing_eval.core import EmbeddingPort

from .strategies import ModelTier, _check_vector, rank_neighbours, vote_ranked

MAX_CALIBRATION_EXAMPLES = 2_000  # leave-one-out is O(n^2) similarities
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 7)
# KnnSettings.minimum_similarity is constrained to [0, 1].
DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(0, 19))


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    example_id: str
    input_text: str
    scenario_type: str
    tier: ModelTier
    split: Literal["development"] = "development"

    def __post_init__(self) -> None:
        if not self.example_id.strip() or not self.input_text.strip():
            raise ValueError("CALIBRATION_EXAMPLE_EMPTY")
        if not self.scenario_type.strip():
            raise ValueError("CALIBRATION_SCENARIO_EMPTY")
        if self.tier not in ("small", "large"):
            raise ValueError("CALIBRATION_TIER_INVALID")
        if self.split != "development":
            raise ValueError("HELD_OUT_DATA_FORBIDDEN_IN_CALIBRATION")


@dataclass(frozen=True, slots=True)
class KnnGridPoint:
    k: int
    minimum_similarity: float
    correct: int
    under_routed: int  # large-labeled -> small (quality risk)
    over_routed: int  # small-labeled -> large (cost)
    small_selected: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    @property
    def under_routing_rate(self) -> float:
        return self.under_routed / self.total


@dataclass(frozen=True, slots=True)
class KnnCalibration:
    selected: KnnGridPoint | None
    max_under_routing_rate: float
    grid: tuple[KnnGridPoint, ...]

    @property
    def feasible(self) -> bool:
        return self.selected is not None


@dataclass(frozen=True, slots=True)
class ScenarioEvidence:
    scenario_type: str
    examples: int
    large_labels: int
    escalate: bool
    reason: Literal["LARGE_SHARE_ABOVE_CEILING", "INSUFFICIENT_EVIDENCE", "ACCEPT_SMALL"]


@dataclass(frozen=True, slots=True)
class CascadeCalibration:
    escalate_scenarios: tuple[str, ...]
    max_small_risk: float
    min_scenario_examples: int
    scenarios: tuple[ScenarioEvidence, ...]


def _validate_examples(examples: Sequence[CalibrationExample]) -> None:
    if not 2 <= len(examples) <= MAX_CALIBRATION_EXAMPLES:
        raise ValueError("CALIBRATION_EXAMPLE_COUNT_INVALID")
    if len({ex.example_id for ex in examples}) != len(examples):
        raise ValueError("CALIBRATION_DUPLICATE_EXAMPLE_ID")
    if any(ex.split != "development" for ex in examples):
        raise ValueError("HELD_OUT_DATA_FORBIDDEN_IN_CALIBRATION")


async def calibrate_knn(
    examples: Sequence[CalibrationExample],
    embedder: EmbeddingPort,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    max_under_routing_rate: float = 0.05,
) -> KnnCalibration:
    """Leave-one-out: each example is classified by all *other* examples."""
    _validate_examples(examples)
    if not math.isfinite(max_under_routing_rate) or not 0 <= max_under_routing_rate <= 1:
        raise ValueError("UNDER_ROUTING_CEILING_INVALID")
    valid_k = sorted({k for k in k_values if not isinstance(k, bool) and 1 <= k < len(examples)})
    valid_t = sorted({float(t) for t in thresholds if math.isfinite(t) and 0 <= t <= 1})
    if not valid_k or not valid_t:
        raise ValueError("CALIBRATION_GRID_EMPTY")

    vectors = await embedder.embed(tuple(ex.input_text for ex in examples))
    if len(vectors) != len(examples):
        raise ValueError("CALIBRATION_EMBEDDING_COUNT_MISMATCH")
    for vector in vectors:
        _check_vector(vector, len(vectors[0]))
    references = tuple(
        (vector, ex.example_id, ex.tier) for vector, ex in zip(vectors, examples, strict=True)
    )
    # Rankings depend on neither k nor the threshold: rank once, reuse per grid point.
    max_k = valid_k[-1]
    rankings = tuple(
        rank_neighbours(vectors[i], references[:i] + references[i + 1 :])[:max_k]
        for i in range(len(examples))
    )

    grid: list[KnnGridPoint] = []
    for k in valid_k:
        for threshold in valid_t:
            correct = under = over = small = 0
            for ranked, example in zip(rankings, examples, strict=True):
                predicted, _ = vote_ranked(ranked, k=k, minimum_similarity=threshold)
                small += predicted == "small"
                if predicted == example.tier:
                    correct += 1
                elif predicted == "small":
                    under += 1
                else:
                    over += 1
            grid.append(KnnGridPoint(k, threshold, correct, under, over, small, len(examples)))

    feasible = [p for p in grid if p.under_routing_rate <= max_under_routing_rate]
    selected = (
        min(feasible, key=lambda p: (-p.correct, p.under_routed, -p.minimum_similarity, p.k))
        if feasible
        else None
    )
    return KnnCalibration(
        selected=selected, max_under_routing_rate=max_under_routing_rate, grid=tuple(grid)
    )


def calibrate_cascade(
    examples: Sequence[CalibrationExample],
    *,
    max_small_risk: float = 0.2,
    min_scenario_examples: int = 3,
    known_scenarios: Sequence[str] = (),
) -> CascadeCalibration:
    """Calibrate known scenarios; unseen configured risks remain escalated."""
    _validate_examples(examples)
    if not math.isfinite(max_small_risk) or not 0 <= max_small_risk <= 1:
        raise ValueError("SMALL_RISK_CEILING_INVALID")
    if isinstance(min_scenario_examples, bool) or not 1 <= min_scenario_examples <= 1000:
        raise ValueError("MIN_SCENARIO_EXAMPLES_INVALID")
    if any(not scenario.strip() for scenario in known_scenarios):
        raise ValueError("CALIBRATION_SCENARIO_EMPTY")
    # A configured escalation with no development evidence must not disappear.
    counts: dict[str, list[int]] = {scenario: [0, 0] for scenario in known_scenarios}
    for ex in examples:
        entry = counts.setdefault(ex.scenario_type, [0, 0])
        entry[0] += 1
        entry[1] += ex.tier == "large"
    evidence: list[ScenarioEvidence] = []
    for scenario in sorted(counts):
        total, large = counts[scenario]
        if total < min_scenario_examples:
            evidence.append(ScenarioEvidence(scenario, total, large, True, "INSUFFICIENT_EVIDENCE"))
        elif large / total > max_small_risk:
            evidence.append(
                ScenarioEvidence(scenario, total, large, True, "LARGE_SHARE_ABOVE_CEILING")
            )
        else:
            evidence.append(ScenarioEvidence(scenario, total, large, False, "ACCEPT_SMALL"))
    return CascadeCalibration(
        escalate_scenarios=tuple(e.scenario_type for e in evidence if e.escalate),
        max_small_risk=max_small_risk,
        min_scenario_examples=min_scenario_examples,
        scenarios=tuple(evidence),
    )
