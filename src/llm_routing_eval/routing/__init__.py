"""Routing algorithms behind the shared Core router/embedding contracts."""

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
    "CascadeRouter",
    "DevelopmentExample",
    "HashingEmbedding",
    "KnnRouter",
    "ModelPair",
    "RuleBasedRouter",
    "ScenarioRule",
]
