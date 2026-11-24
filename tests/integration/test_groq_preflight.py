"""groq-preflight: offline, secret-free, and never READY."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from llm_routing_eval.cli.app import app
from llm_routing_eval.cli.configuration import read_settings
from llm_routing_eval.cli.groq_preflight import (
    EXTERNAL_BLOCKERS,
    assess_groq_preflight,
    read_pricing_manifest,
    read_provider_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "settings.yaml"
PROVIDERS = ROOT / "config" / "providers.yaml"
PRICING = ROOT / "config" / "pricing.yaml"
EXTERNAL = [code for code, _ in EXTERNAL_BLOCKERS]
SECRET = "SYNTHETIC_SECRET_VALUE_DO_NOT_PRINT"
runner = CliRunner()


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("preflight must never open a network connection")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def _invoke(*args: str) -> Any:
    return runner.invoke(app, ["groq-preflight", *args])


def _write(tmp_path: Path, name: str, data: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _ready_looking_configs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Every local gate satisfied: the result must STILL be BLOCKED."""
    providers = yaml.safe_load(PROVIDERS.read_text(encoding="utf-8"))
    providers["providers"]["groq"]["enabled"] = True
    small = providers["providers"]["groq"]["candidate_models"]["small"]
    large = providers["providers"]["groq"]["candidate_models"]["large"]
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings["execution"].update(mode="live", live_enabled=True)
    settings["routing"]["models"] = {"small": small, "large": large}
    pricing = yaml.safe_load(PRICING.read_text(encoding="utf-8"))
    pricing.update(
        status="verified",
        effective_date="2026-11-24",
        source="synthetic-test-fixture",
        rates={
            model: {"input_per_million": 0.1, "output_per_million": 0.5} for model in (small, large)
        },
    )
    return (
        _write(tmp_path, "settings.yaml", settings),
        _write(tmp_path, "providers.yaml", providers),
        _write(tmp_path, "pricing.yaml", pricing),
    )


def test_repository_defaults_are_blocked_with_every_local_blocker() -> None:
    result = _invoke()
    assert result.exit_code == 2
    assert "GROQ_PREFLIGHT_STATUS: BLOCKED" in result.output
    for code in (
        "GROQ_PROVIDER_DISABLED",
        "MODEL_CONFIGURATION_MISMATCH",
        "EXECUTION_NOT_LIVE",
        "GROQ_CREDENTIAL_NOT_CONFIGURED",
        "VERIFIED_PRICING_NOT_AVAILABLE",
        *EXTERNAL,
    ):
        assert code in result.output
    assert "no network call was made" in result.output


def test_credential_presence_is_detected_but_value_never_printed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    for args in ((), ("--json",)):
        result = _invoke(*args)
        assert result.exit_code == 2
        assert SECRET not in result.output
    report = json.loads(_invoke("--json").output)
    checks = {c["name"]: c["passed"] for c in report["checks"]}
    assert checks["groq_credential_present"] is True
    assert "GROQ_CREDENTIAL_NOT_CONFIGURED" not in report["blockers"]


def test_satisfying_every_local_gate_is_still_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is no path to READY: external approvals cannot be inferred locally."""
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    settings, providers, pricing = _ready_looking_configs(tmp_path)
    result = _invoke(
        "--config",
        str(settings),
        "--providers",
        str(providers),
        "--pricing",
        str(pricing),
        "--json",
    )
    assert result.exit_code == 2, result.output
    report = json.loads(result.output)
    assert report["status"] == "BLOCKED"
    assert all(check["passed"] for check in report["checks"])
    assert report["blockers"] == EXTERNAL
    assert report["external_owners"]["OUTBOUND_PRIVACY_AUTHORIZATION_NOT_APPROVED"] == "Member 04"


def test_pricing_must_cover_both_candidate_models(tmp_path: Path) -> None:
    settings, providers, pricing_path = _ready_looking_configs(tmp_path)
    pricing = yaml.safe_load(pricing_path.read_text(encoding="utf-8"))
    pricing["rates"].pop(next(iter(pricing["rates"])))
    result = assess_groq_preflight(
        read_provider_manifest(providers),
        read_pricing_manifest(_write(tmp_path, "pricing2.yaml", pricing)),
        read_settings(settings),
    )
    assert "VERIFIED_PRICING_NOT_AVAILABLE" in result.blockers


@pytest.mark.parametrize(
    ("option", "content", "code"),
    [
        ("--providers", "providers: [not, a, mapping]", "PROVIDER_CONFIG_INVALID"),
        ("--pricing", "schema_version: '1.0.0'\nstatus: maybe", "PRICING_CONFIG_INVALID"),
        ("--config", "schema_version: '9.9.9'", "CONFIG_INVALID"),
    ],
)
def test_invalid_configuration_exits_3(
    tmp_path: Path, option: str, content: str, code: str
) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text(content, encoding="utf-8")
    result = _invoke(option, str(path))
    assert result.exit_code == 3
    assert code in result.output


def test_missing_files_exit_3(tmp_path: Path) -> None:
    result = _invoke("--providers", str(tmp_path / "missing.yaml"))
    assert result.exit_code == 3 and "PROVIDER_CONFIG_INVALID" in result.output
