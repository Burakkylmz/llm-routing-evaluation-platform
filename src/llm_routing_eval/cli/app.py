"""Typer composition root for the isolated, safe offline M1 smoke workflow."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from llm_routing_eval.application import (
    BenchmarkInputError,
    BenchmarkOrchestrator,
    BenchmarkPersistenceError,
)
from llm_routing_eval.cli.configuration import demo_options, read_settings
from llm_routing_eval.core import BenchmarkCase
from llm_routing_eval.demo.fakes import (
    AllowlistedDemoPrivacy,
    DemoAlwaysLargeRouter,
    DemoFakeProvider,
    DemoFixtureEvaluator,
    InMemoryDemoRepository,
)

app = typer.Typer(help="LLM routing evaluation: safe offline integration demo.")
DEFAULT_CONFIG = Path("config/settings.yaml")
DEFAULT_DATASET = Path("examples/offline_demo_cases.json")
DEFAULT_OUTPUT = Path("artifacts/offline_demo.jsonl")


def _load_demo_cases(path: Path) -> tuple[str, list[BenchmarkCase]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
            raise BenchmarkInputError("DATASET_INVALID_FORMAT")
        dataset_version = data.get("dataset_version")
        if not isinstance(dataset_version, str) or not dataset_version:
            raise BenchmarkInputError("DATASET_MISSING_VERSION")
        cases = [BenchmarkCase.model_validate(record) for record in data["cases"]]
        # In this interim demo, never accept data other than explicitly marked synthetic.
        if any(case.metadata.get("synthetic") is not True for case in cases):
            raise BenchmarkInputError("DEMO_REQUIRES_SYNTHETIC_CASES")
        return dataset_version, cases
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise BenchmarkInputError("DATASET_INVALID") from exc


def _write_safe_jsonl(path: Path, repository: InMemoryDemoRepository) -> None:
    """Atomic interim demo export; Member 04 owns durable benchmark persistence."""
    temp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            [("trace", item) for item in repository.traces]
            + [("evaluation", item) for item in repository.evaluations]
            + [("report", item) for item in repository.reports]
        )
        # Only schema-safe records: no ModelResponse.content or BenchmarkCase.input_text.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".offline-demo-",
            suffix=".jsonl", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for kind, record in rows:
                handle.write(json.dumps(
                    {"record_type": kind, "payload": record.model_dump(mode="json")},
                    ensure_ascii=False,
                ) + "\n")
        os.replace(temp_path, path)
    except OSError as exc:
        raise BenchmarkPersistenceError("OUTPUT_WRITE_FAILED") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.command("validate-config")
def validate_config(
    config: Annotated[Path, typer.Option(help="YAML settings file")] = DEFAULT_CONFIG,
) -> None:
    """Validate offline settings without invoking providers."""
    try:
        demo_options(read_settings(config))
        typer.echo("CONFIG_VALID: offline demo (no network, no API keys)")
    except BenchmarkInputError as exc:
        typer.echo(f"CONFIG_ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from None


@app.command("run")
def run(
    config: Annotated[Path, typer.Option(help="YAML settings file")] = DEFAULT_CONFIG,
    dataset: Annotated[Path, typer.Option(help="Synthetic demo fixtures JSON")] = DEFAULT_DATASET,
    output: Annotated[Path, typer.Option(help="Safe JSONL result destination")] = DEFAULT_OUTPUT,
    live: Annotated[bool, typer.Option(help="Reserved; live mode is NOT implemented")] = False,
) -> None:
    """Run the offline integration test using deterministic fake components."""
    if live:
        typer.echo("LIVE_MODE_NOT_IMPLEMENTED", err=True)
        raise typer.Exit(code=2)
    try:
        options = demo_options(read_settings(config))
        dataset_version, cases = _load_demo_cases(dataset)
        options = options.model_copy(update={"dataset_version": dataset_version})
        repo = InMemoryDemoRepository()
        orchestrator = BenchmarkOrchestrator(
            router=DemoAlwaysLargeRouter(model=options.allowed_models[0]),
            provider=DemoFakeProvider(), evaluator=DemoFixtureEvaluator(),
            privacy=AllowlistedDemoPrivacy(), repository=repo,
        )
        result = asyncio.run(orchestrator.run(cases, options))
        _write_safe_jsonl(output, repo)
        metric = result.report.strategy_metrics[options.strategy]
        typer.echo(f"RUN_ID: {result.report.run_id}")
        typer.echo(f"CASES: {metric.case_count}, SUCCESS: {metric.successful_cases}, "
                   f"FAILED: {metric.failed_cases}, BLOCKED: {metric.blocked_cases}")
        typer.echo("MODE: OFFLINE_SIMULATION; financial quality and API cost NOT measured")
        typer.echo(f"REPORT: {output}")
    except (BenchmarkInputError, BenchmarkPersistenceError, ValidationError) as exc:
        # Avoid exposing Pydantic validation inputs, file names or secrets in errors.
        reason = (
            str(exc) if isinstance(exc, (BenchmarkInputError, BenchmarkPersistenceError))
            else "INVALID_OPTIONS"
        )
        typer.echo(f"RUN_FAILED: {reason}", err=True)
        raise typer.Exit(code=2) from None


if __name__ == "__main__":
    app()
