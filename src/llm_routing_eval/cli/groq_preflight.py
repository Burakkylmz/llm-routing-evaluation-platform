"""Offline Groq live-readiness preflight. Never calls the network; never reads key values.

The preflight inspects local configuration (providers, pricing, settings) and
the environment, and lists every blocker that stands between this repository
and a live Groq benchmark. The status is **always BLOCKED**: approvals that only
other owners can give (privacy, expert evaluation, account access) cannot be
inferred from a local toggle or from a working API key, so there is
intentionally no code path that reports READY.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_routing_eval.cli.configuration import OfflineSettings

_KEY_ENV = "GROQ_API_KEY"
_MAX_CONFIG_BYTES = 131_072

# Gates no local file or variable can satisfy; each needs a named owner's sign-off.
EXTERNAL_BLOCKERS: tuple[tuple[str, str], ...] = (
    ("OUTBOUND_PRIVACY_AUTHORIZATION_NOT_APPROVED", "Member 04"),
    ("EXPERT_EVALUATION_NOT_VERIFIED", "Member 03"),
    ("MODEL_ACCOUNT_ACCESS_NOT_VERIFIED", "Member 02 (live smoke test record)"),
    ("LIVE_CLI_PATH_NOT_IMPLEMENTED", "Member 01"),
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class _ModelPair(_Frozen):
    small: str = Field(min_length=1)
    large: str = Field(min_length=1)


class _GroqEntry(_Frozen):
    enabled: bool = False
    api_key_env: Literal["GROQ_API_KEY"]
    candidate_models: _ModelPair


class _OtherProvider(_Frozen):
    enabled: bool = False
    api_key_env: str
    candidate_models: dict[str, str] = Field(default_factory=dict)


class _Providers(_Frozen):
    groq: _GroqEntry
    nvidia_build: _OtherProvider


class ProviderManifest(_Frozen):
    schema_version: Literal["1.0.0"]
    providers: _Providers


class PricingManifest(_Frozen):
    schema_version: Literal["1.0.0"]
    currency: str = Field(min_length=3, max_length=3)
    status: Literal["unverified", "verified"]
    effective_date: str | None = None
    source: str | None = None
    rates: dict[str, dict[str, float]] = Field(default_factory=dict)


class PreflightCheck(_Frozen):
    name: str
    passed: bool
    blocker: str | None = None


class GroqPreflight(_Frozen):
    """Machine-readable, secret-free result. Keys and prompts are never represented."""

    status: Literal["BLOCKED"] = "BLOCKED"
    checks: tuple[PreflightCheck, ...]
    blockers: tuple[str, ...] = Field(min_length=1)
    external_owners: dict[str, str]


def _read_yaml(path: Path, error: str) -> object:
    try:
        if not path.is_file() or path.stat().st_size > _MAX_CONFIG_BYTES:
            raise ValueError(error)
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(error) from exc


def read_provider_manifest(path: Path) -> ProviderManifest:
    try:
        return ProviderManifest.model_validate(_read_yaml(path, "PROVIDER_CONFIG_INVALID"))
    except ValidationError as exc:
        raise ValueError("PROVIDER_CONFIG_INVALID") from exc


def read_pricing_manifest(path: Path) -> PricingManifest:
    try:
        return PricingManifest.model_validate(_read_yaml(path, "PRICING_CONFIG_INVALID"))
    except ValidationError as exc:
        raise ValueError("PRICING_CONFIG_INVALID") from exc


def assess_groq_preflight(
    providers: ProviderManifest,
    pricing: PricingManifest,
    settings: OfflineSettings,
) -> GroqPreflight:
    """Diagnose local readiness. There is intentionally no path to READY."""
    groq = providers.providers.groq
    candidates = (groq.candidate_models.small, groq.candidate_models.large)
    routing_models = (settings.routing.models.small, settings.routing.models.large)
    priced = pricing.status == "verified" and all(
        model in pricing.rates
        and {"input_per_million", "output_per_million"} <= set(pricing.rates[model])
        for model in candidates
    )
    # Presence only: the key value is never read into the result or printed.
    credential = bool(os.getenv(_KEY_ENV, "").strip())
    raw_checks: tuple[PreflightCheck, ...] = (
        PreflightCheck(
            name="groq_provider_enabled", passed=groq.enabled, blocker="GROQ_PROVIDER_DISABLED"
        ),
        PreflightCheck(
            name="routing_models_match_groq_candidates",
            passed=routing_models == candidates,
            blocker="MODEL_CONFIGURATION_MISMATCH",
        ),
        PreflightCheck(
            name="execution_live_enabled",
            passed=settings.execution.mode == "live" and settings.execution.live_enabled,
            blocker="EXECUTION_NOT_LIVE",
        ),
        PreflightCheck(
            name="groq_credential_present",
            passed=credential,
            blocker="GROQ_CREDENTIAL_NOT_CONFIGURED",
        ),
        PreflightCheck(
            name="pricing_verified_for_candidates",
            passed=priced,
            blocker="VERIFIED_PRICING_NOT_AVAILABLE",
        ),
    )
    checks = tuple(
        check.model_copy(update={"blocker": None}) if check.passed else check
        for check in raw_checks
    )
    blockers = tuple(check.blocker for check in checks if check.blocker is not None) + tuple(
        code for code, _ in EXTERNAL_BLOCKERS
    )
    return GroqPreflight(
        checks=checks,
        blockers=blockers,
        external_owners=dict(EXTERNAL_BLOCKERS),
    )
