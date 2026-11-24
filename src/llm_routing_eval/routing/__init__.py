"""Routing algorithms behind the shared Core router/embedding contracts."""

from .cascade_policy import (
    CascadeEscalationDecision,
    CascadeEscalationPolicy,
    ConfiguredCascadeEscalationPolicy,
)
from .embeddings import HashingEmbedding
from .strategies import (
    AlwaysLargeRouter,
    CascadeRouter,
    DevelopmentExample,
    KnnRouter,
    ModelPair,
    RuleBasedRouter,
    ScenarioRule,
)

__all__ = [
    "AlwaysLargeRouter",
    "CascadeEscalationDecision",
    "CascadeEscalationPolicy",
    "ConfiguredCascadeEscalationPolicy",
    "CascadeRouter",
    "DevelopmentExample",
    "HashingEmbedding",
    "KnnRouter",
    "ModelPair",
    "RuleBasedRouter",
    "ScenarioRule",
]
