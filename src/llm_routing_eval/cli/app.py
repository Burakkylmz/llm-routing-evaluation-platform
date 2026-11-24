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
from llm_routing_eval.cli.development_isolation import enforce_development_isolation
from llm_routing_eval.cli.router_factory import build_offline_router
from llm_routing_eval.core import BenchmarkCase, StrategyId
from llm_routing_eval.demo.fakes import (
    AllowlistedDemoPrivacy,
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
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".offline-demo-",
            suffix=".jsonl",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for kind, record in rows:
                handle.write(
                    json.dumps(
                        {"record_type": kind, "payload": record.model_dump(mode="json")},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        os.replace(temp_path, path)
    except OSError as exc:
        raise BenchmarkPersistenceError("OUTPUT_WRITE_FAILED") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _strategy_override(value: str | None) -> StrategyId | None:
    if value is None:
        return None
    try:
        return StrategyId(value)
    except ValueError as exc:
        raise BenchmarkInputError("UNKNOWN_STRATEGY") from exc


@app.command("validate-config")
def validate_config(
    config: Annotated[Path, typer.Option(help="YAML settings file")] = DEFAULT_CONFIG,
    strategy: Annotated[str | None, typer.Option(help="Optional router strategy override")] = None,
) -> None:
    """Validate config and instantiate the selected offline router, without network."""
    try:
        settings = read_settings(config)
        options = demo_options(settings, strategy_override=_strategy_override(strategy))
        asyncio.run(build_offline_router(settings, options.strategy))
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
    strategy: Annotated[str | None, typer.Option(help="Optional router strategy override")] = None,
) -> None:
    """Run any of the four offline routers with a fake provider and allowlisted cases."""
    if live:
        typer.echo("LIVE_MODE_NOT_IMPLEMENTED", err=True)
        raise typer.Exit(code=2)
    try:
        settings = read_settings(config)
        options = demo_options(settings, strategy_override=_strategy_override(strategy))
        dataset_version, cases = _load_demo_cases(dataset)
        if options.strategy is StrategyId.KNN:
            enforce_development_isolation(settings, cases)
        router = asyncio.run(build_offline_router(settings, options.strategy))
        options = options.model_copy(update={"dataset_version": dataset_version})
        repo = InMemoryDemoRepository()
        orchestrator = BenchmarkOrchestrator(
            router=router,
            provider=DemoFakeProvider(),
            evaluator=DemoFixtureEvaluator(),
            privacy=AllowlistedDemoPrivacy(),
            repository=repo,
        )
        result = asyncio.run(orchestrator.run(cases, options))
        _write_safe_jsonl(output, repo)
        metric = result.report.strategy_metrics[options.strategy]
        typer.echo(f"RUN_ID: {result.report.run_id}")
        typer.echo(
            f"CASES: {metric.case_count}, SUCCESS: {metric.successful_cases}, "
            f"FAILED: {metric.failed_cases}, BLOCKED: {metric.blocked_cases}"
        )
        typer.echo(
            f"STRICT_SUCCESS: {metric.successful_cases - metric.degraded_cases}, "
            f"DEGRADED: {metric.degraded_cases} (lower-tier answer kept after failed escalation)"
        )
        typer.echo("MODE: OFFLINE_SIMULATION; financial quality and API cost NOT measured")
        typer.echo(f"REPORT: {output}")
    except (BenchmarkInputError, BenchmarkPersistenceError, ValidationError) as exc:
        # Avoid exposing Pydantic validation inputs, file names or secrets in errors.
        reason = (
            str(exc)
            if isinstance(exc, (BenchmarkInputError, BenchmarkPersistenceError))
            else "INVALID_OPTIONS"
        )
        typer.echo(f"RUN_FAILED: {reason}", err=True)
        raise typer.Exit(code=2) from None


@app.command("groq-preflight")
def groq_preflight(
    config: Annotated[Path, typer.Option(help="YAML settings file")] = DEFAULT_CONFIG,
    providers: Annotated[Path, typer.Option(help="Provider manifest YAML")] = Path(
        "config/providers.yaml"
    ),
    pricing: Annotated[Path, typer.Option(help="Pricing manifest YAML")] = Path(
        "config/pricing.yaml"
    ),
    json_output: Annotated[
        bool, typer.Option("--json", help="Print machine-readable JSON")
    ] = False,
) -> None:
    """Offline live-readiness check for Groq. Never calls the network; always BLOCKED.

    Exit code 2 means BLOCKED (the expected result today); 3 means invalid config.
    """
    from llm_routing_eval.cli.groq_preflight import (
        assess_groq_preflight,
        read_pricing_manifest,
        read_provider_manifest,
    )

    try:
        result = assess_groq_preflight(
            read_provider_manifest(providers),
            read_pricing_manifest(pricing),
            read_settings(config),
        )
    except (BenchmarkInputError, ValueError) as exc:
        code = str(exc) if isinstance(exc, BenchmarkInputError) else str(exc.args[0])
        typer.echo(f"GROQ_PREFLIGHT_CONFIG_INVALID: {code}", err=True)
        raise typer.Exit(code=3) from None
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(f"GROQ_PREFLIGHT_STATUS: {result.status}")
        for check in result.checks:
            typer.echo(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}")
        typer.echo("BLOCKERS:")
        for blocker in result.blockers:
            owner = result.external_owners.get(blocker)
            typer.echo(f"  - {blocker}" + (f" (needs: {owner})" if owner else ""))
        typer.echo("NOTE: offline diagnostic only; no network call was made.")
    raise typer.Exit(code=2)


@app.command("compare-routing")
def compare_routing(
    config: Annotated[Path, typer.Option(help="YAML settings file")] = DEFAULT_CONFIG,
    dataset: Annotated[Path, typer.Option(help="Synthetic demo fixtures JSON")] = DEFAULT_DATASET,
    output: Annotated[Path, typer.Option(help="Comparison summary JSON destination")] = Path(
        "artifacts/offline_comparison.json"
    ),
) -> None:
    """Run all four strategies on the same synthetic cases; offline, never live."""
    from llm_routing_eval.application import RunOptions, StrategyComparisonRunner
    from llm_routing_eval.cli.calibration import atomic_write

    try:
        settings = read_settings(config)
        dataset_version, cases = _load_demo_cases(dataset)
        # kNN is always part of the comparison, so the F4 guard always applies.
        enforce_development_isolation(settings, cases)

        def options_for(strategy: StrategyId) -> RunOptions:
            options = demo_options(settings, strategy_override=strategy)
            return options.model_copy(update={"dataset_version": dataset_version})

        async def orchestrator_for(strategy: StrategyId) -> BenchmarkOrchestrator:
            return BenchmarkOrchestrator(
                router=await build_offline_router(settings, strategy),
                provider=DemoFakeProvider(),
                evaluator=DemoFixtureEvaluator(),
                privacy=AllowlistedDemoPrivacy(),
                repository=InMemoryDemoRepository(),
            )

        summary = asyncio.run(
            StrategyComparisonRunner(
                orchestrator_factory=orchestrator_for, options_factory=options_for
            ).compare(cases, dataset_version=dataset_version)
        )
        atomic_write(output, summary.model_dump_json(indent=2) + "\n")
        typer.echo("MODE: OFFLINE_SIMULATION (NOT financial quality or provider latency)")
        typer.echo(
            f"{'STRATEGY':<14}{'SUCCESS':>9}{'STRICT':>9}{'DEGRADED':>10}"
            f"{'ATTEMPTS':>10}{'ATT/CASE':>10}  CALLS BY MODEL"
        )
        for row in summary.strategies:
            calls = ", ".join(f"{model}={count}" for model, count in row.calls_by_model.items())
            typer.echo(
                f"{row.strategy.value:<14}"
                f"{row.successful_cases:>5}/{row.case_count:<3}"
                f"{row.strict_successful_cases:>5}/{row.case_count:<3}"
                f"{row.degraded_cases:>10}{row.provider_attempts:>10}"
                f"{row.attempts_per_case:>10.2f}  {calls}"
            )
        typer.echo("STRICT: excludes Cascade answers kept after a failed escalation")
        typer.echo("COST: UNKNOWN (no verified pricing); LATENCY: local application only")
        typer.echo(f"REPORT: {output}")
    except (BenchmarkInputError, BenchmarkPersistenceError) as exc:
        typer.echo(f"COMPARISON_FAILED: {exc}", err=True)
        raise typer.Exit(code=2) from None
    except (ValidationError, ValueError, OSError):
        typer.echo("COMPARISON_FAILED: COMPARISON_INPUT_INVALID", err=True)
        raise typer.Exit(code=2) from None


@app.command("calibrate-routing")
def calibrate_routing(
    examples: Annotated[Path, typer.Option(help="Development-only labeled calibration JSON")],
    config: Annotated[Path, typer.Option(help="Base YAML settings file")] = DEFAULT_CONFIG,
    dataset: Annotated[
        Path, typer.Option(help="Evaluation dataset that calibration data must not overlap")
    ] = DEFAULT_DATASET,
    output_config: Annotated[Path, typer.Option(help="Calibrated settings YAML")] = Path(
        "artifacts/settings.calibrated.yaml"
    ),
    report: Annotated[Path, typer.Option(help="Text-free calibration evidence JSON")] = Path(
        "artifacts/routing_calibration_report.json"
    ),
    max_under_routing_rate: Annotated[
        float, typer.Option(help="kNN ceiling for large-labeled examples routed to small")
    ] = 0.05,
    max_small_risk: Annotated[
        float, typer.Option(help="Cascade: escalate scenarios whose large share exceeds this")
    ] = 0.2,
    min_scenario_examples: Annotated[
        int, typer.Option(help="Cascade: escalate scenarios with fewer development examples")
    ] = 3,
) -> None:
    """Calibrate kNN/Cascade on development data only; offline, no network."""
    from llm_routing_eval.cli.calibration import (
        apply_calibration,
        atomic_write,
        calibration_report,
        dump_settings,
        read_calibration_dataset,
    )
    from llm_routing_eval.cli.configuration import routing_policy_digest
    from llm_routing_eval.routing import HashingEmbedding
    from llm_routing_eval.routing.calibration import calibrate_cascade, calibrate_knn

    try:
        settings = read_settings(config)
        calibration = read_calibration_dataset(examples)
        _, cases = _load_demo_cases(dataset)
        calibration_examples = calibration.calibration_examples()
        knn = asyncio.run(
            calibrate_knn(
                calibration_examples,
                # Same embedder and dimensions as build_offline_router uses for kNN.
                HashingEmbedding(dimensions=settings.routing.knn.embedding_dimensions),
                max_under_routing_rate=max_under_routing_rate,
            )
        )
        cascade = calibrate_cascade(
            calibration_examples,
            max_small_risk=max_small_risk,
            min_scenario_examples=min_scenario_examples,
        )
        evidence = calibration_report(calibration, knn, cascade)
        evidence["base_policy_digests"] = {
            strategy.value: routing_policy_digest(settings, strategy)
            for strategy in (StrategyId.KNN, StrategyId.CASCADE)
        }
        if knn.selected is None:
            atomic_write(report, json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
            typer.echo("KNN_CALIBRATION_INFEASIBLE: no grid point meets the ceiling", err=True)
            typer.echo(f"REPORT: {report}")
            raise typer.Exit(code=3)
        calibrated = apply_calibration(settings, calibration, knn, cascade)
        # Same guards as `run`, before anything is written: F4 isolation against
        # the evaluation dataset, and both calibrated routers must build.
        enforce_development_isolation(calibrated, cases)
        for strategy in (StrategyId.KNN, StrategyId.CASCADE):
            asyncio.run(build_offline_router(calibrated, strategy))
        evidence["calibrated_policy_digests"] = {
            strategy.value: routing_policy_digest(calibrated, strategy)
            for strategy in (StrategyId.KNN, StrategyId.CASCADE)
        }
        atomic_write(report, json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
        atomic_write(output_config, dump_settings(calibrated))
        selected = knn.selected
        typer.echo("MODE: DEVELOPMENT_ONLY_CALIBRATION (NOT a held-out quality result)")
        typer.echo(
            f"KNN: k={selected.k} minimum_similarity={selected.minimum_similarity:.2f} "
            f"accuracy={selected.accuracy:.3f} under_routing={selected.under_routing_rate:.3f}"
        )
        typer.echo("CASCADE_ESCALATE: " + (",".join(cascade.escalate_scenarios) or "<none>"))
        typer.echo(f"CONFIG: {output_config}")
        typer.echo(f"REPORT: {report}")
    except BenchmarkInputError as exc:
        typer.echo(f"CALIBRATION_FAILED: {exc}", err=True)
        raise typer.Exit(code=2) from None
    except (ValueError, ValidationError, OSError):
        typer.echo("CALIBRATION_FAILED: CALIBRATION_INPUT_INVALID", err=True)
        raise typer.Exit(code=2) from None


if __name__ == "__main__":
    app()
