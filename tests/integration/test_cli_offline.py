"""CLI tests run in process and use only synthetic fixtures."""

import json
from pathlib import Path

from typer.testing import CliRunner

from llm_routing_eval.cli.app import app

runner = CliRunner()


def config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def dataset_path() -> Path:
    return Path(__file__).resolve().parents[2] / "examples" / "offline_demo_cases.json"


def test_cli_config_validation() -> None:
    result = runner.invoke(app, ["validate-config", "--config", str(config_path())])
    assert result.exit_code == 0, result.output
    assert "CONFIG_VALID" in result.output


def test_cli_creates_machine_readable_jsonl_without_prompt_bodies(tmp_path: Path) -> None:
    output = tmp_path / "demo.jsonl"
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config_path()),
            "--dataset",
            str(dataset_path()),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "SUCCESS: 2" in result.output
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [item["record_type"] for item in records] == [
        "trace",
        "trace",
        "evaluation",
        "evaluation",
        "report",
    ]
    assert records[-1]["payload"]["strategy_metrics"]["always_large"]["cost_amount"] is None
    assert "Banka kartı" not in output.read_text()
    assert "OFFLINE_FAKE_RESPONSE" not in output.read_text()


def test_cli_denies_live_mode() -> None:
    result = runner.invoke(app, ["run", "--live"])
    assert result.exit_code == 2
    assert "LIVE_MODE_NOT_IMPLEMENTED" in result.output


def test_cli_denies_live_env_override(monkeypatch) -> None:
    monkeypatch.setenv("BENCHMARK_MODE", "live")
    result = runner.invoke(app, ["validate-config", "--config", str(config_path())])
    assert result.exit_code == 2
    assert "LIVE_MODE_NOT_IMPLEMENTED" in result.output


def test_cli_rejects_nonsynthetic_dataset(tmp_path: Path) -> None:
    data = json.loads(dataset_path().read_text())
    data["cases"][0]["metadata"] = {"synthetic": False}
    dataset = tmp_path / "unsafe.json"
    dataset.write_text(json.dumps(data))
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config_path()),
            "--dataset",
            str(dataset),
            "--output",
            str(tmp_path / "out.jsonl"),
        ],
    )
    assert result.exit_code == 2
    assert "DEMO_REQUIRES_SYNTHETIC_CASES" in result.output
    assert not (tmp_path / "out.jsonl").exists()


def test_cli_fails_bad_config(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("execution: {mode: offline}\nbenchmark: {}\n")
    result = runner.invoke(app, ["validate-config", "--config", str(config)])
    assert result.exit_code == 2
    assert "CONFIG_INVALID" in result.output
