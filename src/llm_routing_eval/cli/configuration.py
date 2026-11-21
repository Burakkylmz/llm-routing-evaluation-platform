"""Strict configuration for the synthetic, network-disabled routing CLI.

The Groq/NVIDIA adapters cannot be enabled by this module. Provider credentials
and actual outbound requests are deliberately not part of offline composition.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llm_routing_eval.application import BenchmarkInputError, RunOptions
from llm_routing_eval.core import StrategyId


class _SettingsBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ExecutionSettings(_SettingsBase):
    mode: Literal["offline", "live"] = "offline"
    live_enabled: bool = False
    max_model_calls_per_case: int = Field(default=2, ge=1, le=20)
    max_retries: int = Field(default=0, ge=0, le=3)


class BenchmarkSettings(_SettingsBase):
    strategy: StrategyId = StrategyId.ALWAYS_LARGE
    domain: str = Field(min_length=1)
    locale: str = Field(min_length=1)


class ModelPairSettings(_SettingsBase):
    small: str = Field(default="offline-small", min_length=1)
    large: str = Field(default="offline-large", min_length=1)

    @model_validator(mode="after")
    def distinct_models(self) -> Self:
        if not self.small.strip() or not self.large.strip() or self.small == self.large:
            raise ValueError("ROUTER_MODELS_MUST_BE_DISTINCT")
        return self


class RuleSettings(_SettingsBase):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,39}$")
    scenario_types: tuple[str, ...] = Field(min_length=1)
    tier: Literal["small", "large"]

    @model_validator(mode="after")
    def valid_scenarios(self) -> Self:
        if any(not item.strip() for item in self.scenario_types):
            raise ValueError("RULE_SCENARIO_INVALID")
        return self


class RuleBasedSettings(_SettingsBase):
    default_tier: Literal["small", "large"] = "large"
    rules: tuple[RuleSettings, ...] = ()


class DevelopmentExampleSettings(_SettingsBase):
    example_id: str = Field(min_length=1)
    input_text: str = Field(min_length=1, repr=False)
    tier: Literal["small", "large"]
    split: Literal["development"]
    synthetic: Literal[True]

    @model_validator(mode="after")
    def valid_example(self) -> Self:
        if not self.example_id.strip() or not self.input_text.strip():
            raise ValueError("EMPTY_DEVELOPMENT_EXAMPLE")
        return self


class KnnSettings(_SettingsBase):
    k: int = Field(default=1, ge=1, le=100)
    minimum_similarity: float = Field(default=0.35, ge=0, le=1, allow_inf_nan=False)
    embedding_dimensions: int = Field(default=256, ge=16, le=4096)
    examples: tuple[DevelopmentExampleSettings, ...] = ()

    @model_validator(mode="after")
    def unique_example_ids(self) -> Self:
        ids = [example.example_id for example in self.examples]
        if len(set(ids)) != len(ids):
            raise ValueError("DUPLICATE_DEVELOPMENT_IDS")
        return self


class CascadeSettings(_SettingsBase):
    escalate_scenarios: tuple[str, ...] = ()
    escalate_on_failure: bool = True

    @model_validator(mode="after")
    def valid_scenarios(self) -> Self:
        if any(not name.strip() for name in self.escalate_scenarios):
            raise ValueError("INVALID_ESCALATION_SCENARIO")
        return self


class RouterSettings(_SettingsBase):
    models: ModelPairSettings = Field(default_factory=ModelPairSettings)
    rule_based: RuleBasedSettings = Field(default_factory=RuleBasedSettings)
    knn: KnnSettings = Field(default_factory=KnnSettings)
    cascade: CascadeSettings = Field(default_factory=CascadeSettings)


class OfflineSettings(_SettingsBase):
    schema_version: Literal["1.0.0"]
    execution: ExecutionSettings
    benchmark: BenchmarkSettings
    routing: RouterSettings = Field(default_factory=RouterSettings)


def read_settings(path: Path, *, mode_override: str | None = None) -> OfflineSettings:
    if not path.is_file():
        raise BenchmarkInputError("CONFIG_NOT_FOUND")
    try:
        source = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise BenchmarkInputError("CONFIG_INVALID_FORMAT")
        settings = OfflineSettings.model_validate(source)
        raw_mode = mode_override if mode_override is not None else os.getenv("BENCHMARK_MODE")
        if raw_mode is not None:
            updated = settings.model_dump()
            updated["execution"]["mode"] = raw_mode
            settings = OfflineSettings.model_validate(updated)
        return settings
    except (yaml.YAMLError, UnicodeError, OSError, ValidationError, ValueError) as exc:
        raise BenchmarkInputError("CONFIG_INVALID") from exc


def demo_options(
    settings: OfflineSettings,
    *,
    strategy_override: StrategyId | None = None,
) -> RunOptions:
    if settings.execution.mode != "offline" or settings.execution.live_enabled:
        raise BenchmarkInputError("LIVE_MODE_NOT_IMPLEMENTED")
    strategy = strategy_override if strategy_override is not None else settings.benchmark.strategy
    if strategy is StrategyId.KNN and (
        not settings.routing.knn.examples
        or settings.routing.knn.k > len(settings.routing.knn.examples)
    ):
        raise BenchmarkInputError("KNN_DEVELOPMENT_EXAMPLES_REQUIRED")
    if strategy is StrategyId.CASCADE and settings.execution.max_model_calls_per_case < 2:
        raise BenchmarkInputError("CASCADE_REQUIRES_TWO_CALL_SLOTS")
    return RunOptions(
        strategy=strategy,
        provider="fake",
        allowed_models=(settings.routing.models.small, settings.routing.models.large),
        max_model_calls_per_case=settings.execution.max_model_calls_per_case,
        max_retries=settings.execution.max_retries,
    )
