"""Interchangeable, typed RouterPort implementations for offline benchmarking.

Routers receive already-sanitized cases. Golden references and evaluator
results must never be passed to routing algorithms.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from llm_routing_eval.core import (
    BenchmarkCase,
    CallStatus,
    EmbeddingPort,
    RoutingAction,
    RoutingContext,
    RoutingDecision,
    StrategyId,
)

ModelTier = Literal["small", "large"]
_SAFE_RULE_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,39}$")


@dataclass(frozen=True, slots=True)
class ModelPair:
    small: str
    large: str

    def __post_init__(self) -> None:
        if not self.small.strip() or not self.large.strip() or self.small == self.large:
            raise ValueError("Small and large model IDs must be nonempty and distinct")

    def select(self, tier: ModelTier) -> str:
        return self.small if tier == "small" else self.large


@dataclass(frozen=True, slots=True)
class ScenarioRule:
    """First matching rule wins; configuration order is the priority order."""

    code: str
    scenario_types: tuple[str, ...]
    tier: ModelTier

    def __post_init__(self) -> None:
        if _SAFE_RULE_CODE.fullmatch(self.code) is None:
            raise ValueError("Rule identifiers must use safe uppercase codes")
        if not self.scenario_types or any(not value.strip() for value in self.scenario_types):
            raise ValueError("Rule scenario types must be nonempty")
        if self.tier not in ("small", "large"):
            raise ValueError("Unknown model tier")


@dataclass(frozen=True, slots=True)
class DevelopmentExample:
    """Calibration-only labeled example: held-out evaluation data is forbidden."""

    example_id: str
    input_text: str
    tier: ModelTier
    split: Literal["development"] = "development"

    def __post_init__(self) -> None:
        if not self.example_id.strip() or not self.input_text.strip():
            raise ValueError("A development example needs an ID and text")
        if self.split != "development":
            raise ValueError("Held-out data must not enter the routing index")
        if self.tier not in ("small", "large"):
            raise ValueError("Unknown model tier")


def _decision(
    strategy: StrategyId,
    context: RoutingContext,
    models: ModelPair,
    *,
    tier: ModelTier | None,
    reason: str,
    start: float,
) -> RoutingDecision:
    return RoutingDecision(
        strategy=strategy,
        action=RoutingAction.INVOKE_MODEL if tier is not None else RoutingAction.FINISH,
        selected_model=models.select(tier) if tier is not None else None,
        stage_index=context.current_stage,
        reason_code=reason,
        decision_time_ms=(perf_counter() - start) * 1000,
    )


class AlwaysLargeRouter:
    """Reference baseline: exactly one selection of the configured large model."""

    def __init__(self, models: ModelPair) -> None:
        self._models = models

    async def decide(self, case: BenchmarkCase, context: RoutingContext) -> RoutingDecision:
        _ = case
        start = perf_counter()
        tier: ModelTier | None = None if context.prior_attempts else "large"
        return _decision(
            StrategyId.ALWAYS_LARGE,
            context,
            self._models,
            tier=tier,
            reason="BASELINE_DONE" if tier is None else "ALWAYS_LARGE",
            start=start,
        )


class RuleBasedRouter:
    """Explicit, priority-ordered routing by scenario_type (not free-text rules)."""

    def __init__(
        self,
        models: ModelPair,
        rules: Sequence[ScenarioRule],
        *,
        default_tier: ModelTier = "large",
    ) -> None:
        if default_tier not in ("small", "large"):
            raise ValueError("Invalid default tier")
        if len({rule.code for rule in rules}) != len(rules):
            raise ValueError("Duplicate rule code")
        self._models = models
        self._rules = tuple(rules)
        self._default_tier = default_tier

    async def decide(self, case: BenchmarkCase, context: RoutingContext) -> RoutingDecision:
        start = perf_counter()
        if context.prior_attempts:
            return _decision(
                StrategyId.RULE_BASED,
                context,
                self._models,
                tier=None,
                reason="BASELINE_DONE",
                start=start,
            )
        for rule in self._rules:
            if case.scenario_type in rule.scenario_types:
                return _decision(
                    StrategyId.RULE_BASED,
                    context,
                    self._models,
                    tier=rule.tier,
                    reason="RULE_" + rule.code,
                    start=start,
                )
        return _decision(
            StrategyId.RULE_BASED,
            context,
            self._models,
            tier=self._default_tier,
            reason="RULE_DEFAULT",
            start=start,
        )


def _check_vector(vector: tuple[float, ...], dimension: int | None = None) -> None:
    if not vector or (dimension is not None and len(vector) != dimension):
        raise ValueError("Missing or inconsistent embedding dimension")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("Embedding contains non-finite values")
    if not any(value != 0 for value in vector):
        raise ValueError("Embedding cannot be an all-zero vector")


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    return numerator / (norm_l * norm_r)


class KnnRouter:
    """Prepared kNN classifier; stable tie breaks and conservative large fallback."""

    def __init__(
        self,
        models: ModelPair,
        embedder: EmbeddingPort,
        examples: tuple[DevelopmentExample, ...],
        vectors: tuple[tuple[float, ...], ...],
        *,
        k: int,
        minimum_similarity: float,
    ) -> None:
        self._models = models
        self._embedder = embedder
        self._examples = examples
        self._vectors = vectors
        self._k = k
        self._minimum_similarity = minimum_similarity

    @classmethod
    async def prepare(
        cls,
        models: ModelPair,
        embedder: EmbeddingPort,
        examples: Sequence[DevelopmentExample],
        *,
        k: int = 1,
        minimum_similarity: float = 0.35,
    ) -> KnnRouter:
        if not examples or len({ex.example_id for ex in examples}) != len(examples):
            raise ValueError("Development examples must be nonempty with unique IDs")
        if isinstance(k, bool) or not 1 <= k <= len(examples):
            raise ValueError("Invalid k")
        if not math.isfinite(minimum_similarity) or not -1 <= minimum_similarity <= 1:
            raise ValueError("Similarity threshold must be within [-1, 1]")
        if any(ex.split != "development" for ex in examples):
            raise ValueError("Held-out data cannot be indexed")
        snapshot = tuple(examples)
        vectors = await embedder.embed(tuple(ex.input_text for ex in snapshot))
        if len(vectors) != len(snapshot):
            raise ValueError("Embedding adapter returned incorrect count")
        for vector in vectors:
            _check_vector(vector, len(vectors[0]))
        return cls(
            models,
            embedder,
            snapshot,
            vectors,
            k=k,
            minimum_similarity=minimum_similarity,
        )

    async def decide(self, case: BenchmarkCase, context: RoutingContext) -> RoutingDecision:
        start = perf_counter()
        if context.prior_attempts:
            return _decision(
                StrategyId.KNN,
                context,
                self._models,
                tier=None,
                reason="BASELINE_DONE",
                start=start,
            )
        query_vectors = await self._embedder.embed((case.input_text,))
        if len(query_vectors) != 1:
            raise ValueError("Embedding adapter returned incorrect query count")
        vector = query_vectors[0]
        _check_vector(vector, len(self._vectors[0]))
        candidates = sorted(
            (
                (_cosine(vector, reference), ex.example_id, ex.tier)
                for reference, ex in zip(self._vectors, self._examples, strict=True)
            ),
            key=lambda item: (-item[0], item[1]),
        )[: self._k]
        if candidates[0][0] < self._minimum_similarity:
            tier: ModelTier = "large"
            reason = "KNN_LOW_SIMILARITY"
        else:
            small_vote = sum(max(0.0, score) for score, _, t in candidates if t == "small")
            large_vote = sum(max(0.0, score) for score, _, t in candidates if t == "large")
            tier = "small" if small_vote > large_vote else "large"
            reason = "KNN_SMALL" if tier == "small" else "KNN_LARGE"
        return _decision(
            StrategyId.KNN,
            context,
            self._models,
            tier=tier,
            reason=reason,
            start=start,
        )


class CascadeRouter:
    """Small-first, optional large-model escalation using explicit scenario policy.

    No claim of confidence-based escalation: the existing RoutingFeedback
    contract has call status but no validated output confidence signal.
    """

    def __init__(
        self,
        models: ModelPair,
        *,
        escalate_scenarios: frozenset[str] = frozenset(),
        escalate_on_failure: bool = True,
    ) -> None:
        if any(not item.strip() for item in escalate_scenarios):
            raise ValueError("Escalation scenarios must be nonempty")
        self._models = models
        self._escalate_scenarios = escalate_scenarios
        self._escalate_on_failure = escalate_on_failure

    async def decide(self, case: BenchmarkCase, context: RoutingContext) -> RoutingDecision:
        start = perf_counter()
        if context.current_stage == 0 and not context.prior_attempts:
            tier: ModelTier | None = "small"
            reason = "CASCADE_SMALL_FIRST"
        elif context.current_stage == 1 and context.prior_attempts:
            last = context.prior_attempts[-1]
            if last.status is CallStatus.SUCCESS:
                escalate = case.scenario_type in self._escalate_scenarios
                reason = "CASCADE_SCENARIO" if escalate else "CASCADE_ACCEPT_SMALL"
            else:
                escalate = self._escalate_on_failure
                reason = "CASCADE_FAILOVER" if escalate else "CASCADE_STOP_FAILURE"
            tier = "large" if escalate else None
        else:
            tier = None
            reason = "CASCADE_DONE"
        return _decision(
            StrategyId.CASCADE,
            context,
            self._models,
            tier=tier,
            reason=reason,
            start=start,
        )
