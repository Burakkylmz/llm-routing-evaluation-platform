"""End-to-end regression: strategy-specific config identity survives JSONL export."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from llm_routing_eval.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "config" / "settings.yaml"


def _run(
    tmp_path: Path,
    settings: dict[str, Any],
    label: str,
    *,
    strategy: str,
) -> tuple[str, list[dict[str, Any]], str]:
    config = tmp_path / f"{label}.yaml"
    output = tmp_path / f"{label}.jsonl"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--strategy",
            strategy,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    raw = output.read_text(encoding="utf-8")
    records = [json.loads(line) for line in raw.splitlines()]
    report = records[-1]
    assert report["record_type"] == "report"
    return report["payload"]["configuration_fingerprint"], records, raw


def _attempted_models(records: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [step["model"] for step in record["payload"]["steps"] if step["kind"] == "PROVIDER_ATTEMPT"]
        for record in records
        if record["record_type"] == "trace"
    ]


def test_cascade_policy_changes_report_fingerprint_and_attempt_count(tmp_path: Path) -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    changed = deepcopy(base)
    changed["routing"]["cascade"]["escalate_scenarios"] = []

    first, first_rows, _ = _run(tmp_path, base, "original", strategy="cascade")
    identical, _, _ = _run(tmp_path, base, "repeat", strategy="cascade")
    second, second_rows, _ = _run(tmp_path, changed, "no-escalation", strategy="cascade")

    assert first == identical
    assert first != second
    assert _attempted_models(first_rows) == [
        ["offline-small"],
        ["offline-small", "offline-large"],
    ]
    assert _attempted_models(second_rows) == [["offline-small"], ["offline-small"]]


def test_rule_policy_changes_fingerprint(tmp_path: Path) -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    changed = deepcopy(base)
    changed["routing"]["rule_based"]["rules"][0]["tier"] = "large"

    first, first_rows, _ = _run(tmp_path, base, "rule-original", strategy="rule_based")
    second, second_rows, _ = _run(tmp_path, changed, "rule-changed", strategy="rule_based")

    assert first != second
    assert _attempted_models(first_rows) == [["offline-large"], ["offline-small"]]
    assert _attempted_models(second_rows) == [["offline-large"], ["offline-large"]]


def test_knn_training_change_affects_fingerprint_but_training_text_is_not_exported(
    tmp_path: Path,
) -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    changed = deepcopy(base)
    secret_marker = "synthetic-training-private-phrase-never-export"
    changed["routing"]["knn"]["examples"][0]["input_text"] = secret_marker

    first, _, baseline_artifact = _run(tmp_path, base, "knn-original", strategy="knn")
    second, _, changed_artifact = _run(tmp_path, changed, "knn-changed", strategy="knn")

    assert first != second
    assert secret_marker not in changed_artifact
    assert "Genel hesap işlemlerine" not in baseline_artifact
    assert changed_artifact.count("configuration_fingerprint") == 1


def test_unselected_policy_does_not_change_baseline_fingerprint(tmp_path: Path) -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    changed = deepcopy(base)
    changed["routing"]["knn"]["examples"][0]["input_text"] = "different synthetic text"

    original, _, _ = _run(tmp_path, base, "baseline-original", strategy="always_large")
    modified, _, _ = _run(tmp_path, changed, "baseline-modified", strategy="always_large")
    assert original == modified
