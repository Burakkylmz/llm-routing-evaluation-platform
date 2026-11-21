"""Offline-only RouterPort composition; application/Core never import adapters.

No provider/credential wiring is allowed here. Only approved development
examples can be indexed for deterministic offline evaluation demonstrations.
"""

from __future__ import annotations

from collections.abc import Sequence

from llm_routing_eval.application import BenchmarkInputError
from llm_routing_eval.cli.configuration import OfflineSettings
from llm_routing_eval.core import BenchmarkCase, RouterPort, StrategyId
from llm_routing_eval.routing import (
    AlwaysLargeRouter,
    CascadeRouter,
    DevelopmentExample,
    HashingEmbedding,
    KnnRouter,
    ModelPair,
    RuleBasedRouter,
    ScenarioRule,
)


def enforce_development_isolation(
    settings: OfflineSettings,
    cases: Sequence[BenchmarkCase],
) -> None:
    """Reject obvious held-out leakage before creating the kNN index."""

    case_ids = {case.case_id for case in cases}
    case_texts = {case.input_text.strip() for case in cases}
    for example in settings.routing.knn.examples:
        if example.example_id in case_ids or example.input_text.strip() in case_texts:
            raise BenchmarkInputError("KNN_DEVELOPMENT_TEST_LEAKAGE")


async def build_offline_router(
    settings: OfflineSettings,
    strategy: StrategyId,
) -> RouterPort:
    """Construct approved offline strategies from validated settings only."""

    pair = ModelPair(
        small=settings.routing.models.small,
        large=settings.routing.models.large,
    )
    try:
        if strategy is StrategyId.ALWAYS_LARGE:
            return AlwaysLargeRouter(pair)
        if strategy is StrategyId.RULE_BASED:
            rules = tuple(
                ScenarioRule(
                    code=rule.code,
                    scenario_types=rule.scenario_types,
                    tier=rule.tier,
                )
                for rule in settings.routing.rule_based.rules
            )
            return RuleBasedRouter(
                pair,
                rules,
                default_tier=settings.routing.rule_based.default_tier,
            )
        if strategy is StrategyId.KNN:
            knn = settings.routing.knn
            examples = tuple(
                DevelopmentExample(
                    example_id=example.example_id,
                    input_text=example.input_text,
                    tier=example.tier,
                    split=example.split,
                )
                for example in knn.examples
            )
            return await KnnRouter.prepare(
                pair,
                HashingEmbedding(dimensions=knn.embedding_dimensions),
                examples,
                k=knn.k,
                minimum_similarity=knn.minimum_similarity,
            )
        if strategy is StrategyId.CASCADE:
            cascade = settings.routing.cascade
            return CascadeRouter(
                pair,
                escalate_scenarios=frozenset(cascade.escalate_scenarios),
                escalate_on_failure=cascade.escalate_on_failure,
            )
    except (ValueError, TypeError) as exc:
        raise BenchmarkInputError("ROUTER_CONFIGURATION_INVALID") from exc
    raise BenchmarkInputError("UNSUPPORTED_ROUTING_STRATEGY")
