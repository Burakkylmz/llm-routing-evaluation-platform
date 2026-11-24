"""calibrate-routing CLI: strict input, F4 isolation, loadable output, text-free report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llm_routing_eval.application import BenchmarkInputError
from llm_routing_eval.cli.app import app
from llm_routing_eval.cli.calibration import read_calibration_dataset
from llm_routing_eval.cli.configuration import read_settings

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "synthetic_development_calibration.json"
CONFIG = ROOT / "config" / "settings.yaml"
DATASET = ROOT / "examples" / "offline_demo_cases.json"
runner = CliRunner()


def _invoke(tmp_path: Path, examples: Path, *extra: str) -> tuple[Any, Path, Path]:
    config_out = tmp_path / "settings.yaml"
    report = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "calibrate-routing",
            "--examples",
            str(examples),
            "--config",
            str(CONFIG),
            "--dataset",
            str(DATASET),
            "--output-config",
            str(config_out),
            "--report",
            str(report),
            *extra,
        ],
    )
    return result, config_out, report


def _write(tmp_path: Path, data: dict[str, Any], name: str = "examples.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _synthetic() -> dict[str, Any]:
    return json.loads(EXAMPLES.read_text(encoding="utf-8"))


def test_cli_writes_loadable_settings_and_text_free_report(tmp_path: Path) -> None:
    result, config_out, report = _invoke(tmp_path, EXAMPLES)
    assert result.exit_code == 0, result.output
    assert "DEVELOPMENT_ONLY_CALIBRATION" in result.output

    calibrated = read_settings(config_out)
    original = read_settings(CONFIG)
    assert len(calibrated.routing.knn.examples) == len(_synthetic()["examples"])
    assert calibrated.execution == original.execution  # only routing is changed
    assert calibrated.routing.models == original.routing.models

    evidence = json.loads(report.read_text(encoding="utf-8"))
    assert evidence["knn"]["feasible"] is True
    assert evidence["dataset_fingerprint"].startswith("sha256:")
    assert set(evidence["calibrated_policy_digests"]) == {"knn", "cascade"}
    serialized = report.read_text(encoding="utf-8")
    assert all(e["input_text"] not in serialized for e in _synthetic()["examples"])

    # The calibrated file runs end to end with both calibrated strategies.
    for strategy in ("knn", "cascade"):
        run = runner.invoke(
            app,
            [
                "run",
                "--config",
                str(config_out),
                "--dataset",
                str(DATASET),
                "--strategy",
                strategy,
                "--output",
                str(tmp_path / f"{strategy}.jsonl"),
            ],
        )
        assert run.exit_code == 0, run.output


def test_cli_infeasible_writes_report_but_no_settings(tmp_path: Path) -> None:
    data = _synthetic()
    data["examples"] = [
        {**data["examples"][0], "example_id": "a", "tier": "small"},
        {**data["examples"][0], "example_id": "b", "tier": "large"},
    ]
    result, config_out, report = _invoke(
        tmp_path, _write(tmp_path, data), "--max-under-routing-rate", "0"
    )
    assert result.exit_code == 3, result.output
    assert json.loads(report.read_text(encoding="utf-8"))["knn"]["feasible"] is False
    assert not config_out.exists()


def test_cli_rejects_calibration_data_overlapping_evaluation_cases(tmp_path: Path) -> None:
    """F4 guard: an evaluation prompt must never become a kNN development example."""
    data = _synthetic()
    demo = json.loads(DATASET.read_text(encoding="utf-8"))["cases"][0]["input_text"]
    data["examples"][0]["input_text"] = "  " + demo.replace(" ", "   ") + " !!"  # normalized copy
    result, config_out, report = _invoke(tmp_path, _write(tmp_path, data))
    assert result.exit_code == 2
    assert "KNN_DEVELOPMENT_TEST_LEAKAGE" in result.output
    assert not config_out.exists() and not report.exists()



@pytest.mark.parametrize("overlap_type", ["normalized_text", "case_id"])
def test_infeasible_leaking_calibration_never_writes_report(
    tmp_path: Path, overlap_type: str
) -> None:
    """F4 guard must precede even the infeasible calibration report path."""
    data = _synthetic()
    held_out = json.loads(DATASET.read_text(encoding="utf-8"))["cases"][0]
    first = {**data["examples"][0], "example_id": "synthetic-cal-a", "tier": "small"}
    second = {**data["examples"][0], "example_id": "synthetic-cal-b", "tier": "large"}
    # Identical vectors with opposite labels => no kNN configuration has
    # zero under-routing. Before this fix the CLI wrote an evidence artifact.
    if overlap_type == "normalized_text":
        first["input_text"] = "  " + held_out["input_text"].replace(" ", "   ") + " !!"
        second["input_text"] = first["input_text"]
    else:
        first["example_id"] = held_out["case_id"]
    data["examples"] = [first, second]

    result, config_out, report = _invoke(
        tmp_path, _write(tmp_path, data), "--max-under-routing-rate", "0"
    )

    assert result.exit_code == 2, result.output
    assert "KNN_DEVELOPMENT_TEST_LEAKAGE" in result.output
    assert not config_out.exists() and not report.exists()


def test_calibration_isolation_runs_before_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contaminated development data must never reach the embedding adapter."""
    data = _synthetic()
    held_out = json.loads(DATASET.read_text(encoding="utf-8"))["cases"][0]
    data["examples"][0]["input_text"] = held_out["input_text"]

    async def never_calibrate(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("calibration ran before development/evaluation isolation")

    monkeypatch.setattr("llm_routing_eval.routing.calibration.calibrate_knn", never_calibrate)
    result, config_out, report = _invoke(tmp_path, _write(tmp_path, data))
    assert result.exit_code == 2, result.output
    assert "KNN_DEVELOPMENT_TEST_LEAKAGE" in result.output
    assert not config_out.exists() and not report.exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(split="test"),
        lambda d: d["examples"][0].update(split="held_out"),
        lambda d: d["examples"][0].update(synthetic=False),
        lambda d: d["examples"][0].update(tier="medium"),
        lambda d: d["examples"][1].update(example_id=d["examples"][0]["example_id"]),
        lambda d: d["examples"][0].update(expected_facts=["golden"]),
        lambda d: d.update(examples=d["examples"][:1]),
    ],
)
def test_held_out_or_malformed_data_is_rejected(tmp_path: Path, mutate: Any) -> None:
    data = _synthetic()
    mutate(data)
    path = _write(tmp_path, data)
    with pytest.raises(BenchmarkInputError, match="CALIBRATION_DATA_INVALID"):
        read_calibration_dataset(path)
    result, config_out, _ = _invoke(tmp_path, path)
    assert result.exit_code == 2 and "CALIBRATION_DATA_INVALID" in result.output
    assert not config_out.exists()


def test_errors_never_echo_example_text(tmp_path: Path) -> None:
    data = _synthetic()
    data["examples"][0]["tier"] = "medium"
    data["examples"][0]["input_text"] = "PRIVATE CANARY TEXT"
    result, _, _ = _invoke(tmp_path, _write(tmp_path, data))
    assert "PRIVATE CANARY" not in result.output


def test_cli_invalid_parameters_and_missing_file(tmp_path: Path) -> None:
    missing, config_out, _ = _invoke(tmp_path, tmp_path / "missing.json")
    assert missing.exit_code == 2
    assert "CALIBRATION_DATA_MISSING_OR_TOO_LARGE" in missing.output
    bad, _, _ = _invoke(tmp_path, EXAMPLES, "--max-small-risk", "2")
    assert bad.exit_code == 2 and "CALIBRATION_INPUT_INVALID" in bad.output
    assert not config_out.exists()
