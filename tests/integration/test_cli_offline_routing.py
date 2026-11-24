"""Network-free end-to-end CLI coverage for YAML-driven Member 02 routers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llm_routing_eval.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "settings.yaml"
DATASET = ROOT / "examples" / "offline_demo_cases.json"
runner = CliRunner()


@pytest.mark.parametrize(
    "strategy",
    ["always_large", "rule_based", "knn", "cascade"],
)
def test_all_routing_strategies_can_be_selected_from_cli(tmp_path: Path, strategy: str) -> None:
    output = tmp_path / f"{strategy}.jsonl"
    response = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(CONFIG),
            "--dataset",
            str(DATASET),
            "--strategy",
            strategy,
            "--output",
            str(output),
        ],
    )
    assert response.exit_code == 0, response.output
    assert "SUCCESS: 2" in response.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    traces = [item["payload"] for item in records if item["record_type"] == "trace"]
    reports = [item["payload"] for item in records if item["record_type"] == "report"]
    assert len(traces) == 2
    assert len(reports) == 1
    assert set(reports[0]["strategy_metrics"]) == {strategy}
    assert all(trace["strategy"] == strategy for trace in traces)
    assert "Banka kartı ile" not in output.read_text(encoding="utf-8")
    assert "OFFLINE_FAKE_RESPONSE" not in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("strategy", ["always_large", "rule_based", "knn", "cascade"])
def test_config_validation_supports_all_four_strategies(strategy: str) -> None:
    result = runner.invoke(
        app, ["validate-config", "--config", str(CONFIG), "--strategy", strategy]
    )
    assert result.exit_code == 0, result.output
    assert "CONFIG_VALID" in result.output


def test_yaml_selected_strategy_is_honored_without_cli_override(tmp_path: Path) -> None:
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings["benchmark"]["strategy"] = "rule_based"
    config = tmp_path / "rule.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    output = tmp_path / "rules.jsonl"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--dataset", str(DATASET), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["payload"]["strategy_metrics"].keys() == {"rule_based"}
    attempts = [
        step["model"]
        for record in records
        if record["record_type"] == "trace"
        for step in record["payload"]["steps"]
        if step["kind"] == "PROVIDER_ATTEMPT"
    ]
    assert attempts == ["offline-large", "offline-small"]


def test_cascade_reports_each_model_call(tmp_path: Path) -> None:
    output = tmp_path / "cascade.jsonl"
    result = runner.invoke(
        app,
        ["run", "--config", str(CONFIG), "--strategy", "cascade", "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    attempts_per_case = [
        [step["model"] for step in record["payload"]["steps"] if step["kind"] == "PROVIDER_ATTEMPT"]
        for record in records
        if record["record_type"] == "trace"
    ]
    assert attempts_per_case == [["offline-small"], ["offline-small", "offline-large"]]


def test_knn_rejects_duplicate_development_test_input_before_run(tmp_path: Path) -> None:
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    example = settings["routing"]["knn"]["examples"][0]
    example["input_text"] = data["cases"][0]["input_text"]
    config = tmp_path / "leak.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    output = tmp_path / "must-not-exist.jsonl"
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--dataset",
            str(DATASET),
            "--strategy",
            "knn",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 2
    assert "KNN_DEVELOPMENT_TEST_LEAKAGE" in result.output
    assert not output.exists()


def test_knn_rejects_duplicate_development_test_id(tmp_path: Path) -> None:
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings["routing"]["knn"]["examples"][0]["example_id"] = "demo-card-001"
    config = tmp_path / "leak.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    result = runner.invoke(app, ["run", "--config", str(config), "--strategy", "knn"])
    assert result.exit_code == 2
    assert "KNN_DEVELOPMENT_TEST_LEAKAGE" in result.output


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s["routing"]["knn"]["examples"][0].update(split="golden"),
        lambda s: s["routing"]["knn"]["examples"][0].update(synthetic=False),
        lambda s: s["routing"]["models"].update(large="offline-small"),
        lambda s: s["routing"]["rule_based"]["rules"][0].update(code="prompt unsafe"),
        lambda s: s["routing"]["knn"].update(embedding_dimensions=4),
        lambda s: s["routing"]["knn"].update(extra_secret="oops"),
    ],
)
def test_unsafe_yaml_is_rejected(tmp_path: Path, mutation: object) -> None:
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert callable(mutation)
    mutation(settings)
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    result = runner.invoke(app, ["validate-config", "--config", str(config)])
    assert result.exit_code == 2
    assert "CONFIG_INVALID" in result.output
    assert "oops" not in result.output


def test_invalid_strategy_or_live_config_fails_closed(tmp_path: Path) -> None:
    invalid = runner.invoke(app, ["run", "--strategy", "unrecognized"])
    assert invalid.exit_code == 2
    assert "UNKNOWN_STRATEGY" in invalid.output

    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings["execution"]["live_enabled"] = True
    config = tmp_path / "live.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    result = runner.invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == 2
    assert "LIVE_MODE_NOT_IMPLEMENTED" in result.output


@pytest.mark.parametrize("collision", ["case_id", "input_text"])
def test_cascade_rejects_calibration_development_overlap_before_run(
    tmp_path: Path, collision: str
) -> None:
    """Cascade must not evaluate cases reused from its development examples."""
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    example = settings["routing"]["knn"]["examples"][0]
    evaluation_case = dataset["cases"][0]
    if collision == "case_id":
        example["example_id"] = evaluation_case["case_id"]
    else:
        example["input_text"] = evaluation_case["input_text"]

    config = tmp_path / "cascade-calibrated.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    output = tmp_path / "must-not-exist.jsonl"
    response = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--dataset",
            str(DATASET),
            "--strategy",
            "cascade",
            "--output",
            str(output),
        ],
    )
    assert response.exit_code == 2, response.output
    assert "KNN_DEVELOPMENT_TEST_LEAKAGE" in response.output
    assert evaluation_case["input_text"] not in response.output
    assert not output.exists()
