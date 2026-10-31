"""Small CLI configuration validator for the offline Member 01 smoke test."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_routing_eval.application import BenchmarkInputError, RunOptions
from llm_routing_eval.core import StrategyId


class ExecutionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["offline", "live"] = "offline"
    live_enabled: bool = False
    max_model_calls_per_case: int = Field(default=2, ge=1, le=20)
    max_retries: int = Field(default=0, ge=0, le=3)


class BenchmarkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: StrategyId = StrategyId.ALWAYS_LARGE
    domain: str = Field(min_length=1)
    locale: str = Field(min_length=1)


class OfflineSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0.0"]
    execution: ExecutionSettings
    benchmark: BenchmarkSettings


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


def demo_options(settings: OfflineSettings) -> RunOptions:
    if settings.execution.mode != "offline" or settings.execution.live_enabled:
        raise BenchmarkInputError("LIVE_MODE_NOT_IMPLEMENTED")
    if settings.benchmark.strategy is not StrategyId.ALWAYS_LARGE:
        raise BenchmarkInputError("ONLY_ALWAYS_LARGE_DEMO_AVAILABLE")
    return RunOptions(
        strategy=settings.benchmark.strategy,
        provider="fake", allowed_models=("openai/gpt-oss-120b",),
        max_model_calls_per_case=settings.execution.max_model_calls_per_case,
        max_retries=settings.execution.max_retries,
    )
