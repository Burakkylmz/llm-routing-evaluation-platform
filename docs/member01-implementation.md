# Member 01 — Implementation Handoff

**Status:** Core contracts, application orchestration, CLI, constructor DI, isolated offline fakes, unit/integration tests and documentation are implemented. **Integration with actual Member 02–04 adapters remains pending**, and the code contracts are not team-reviewed.

## Responsibility boundaries

- `core/`: Versioned Pydantic contracts and six async/sync Protocol interfaces.
- `application/orchestrator.py`: `BenchmarkOrchestrator`, `RunOptions`, reporting aggregates and typed safe-failure boundaries. It imports only Core and stdlib/Pydantic, never a concrete provider or banking module.
- `cli/`: Typer composition root and offline-only config/dataset validation.
- `demo/fakes.py`: Interim **fake-only** adapters; no live service, banking evaluator or actual PII masking. Kept outside Member 02–04 implementation modules.
- `examples/offline_demo_cases.json`: Two synthetic, allowlisted Turkish-language prompts. Not a golden dataset.
- `tests/unit/test_orchestrator.py`, `tests/integration/test_cli_offline.py`: Application/CLI security and error-path tests.

## Run locally

With Python 3.12+ and package-index access:

```bash
uv lock
uv sync --locked
uv run llm-routing-eval validate-config --config config/settings.yaml
uv run llm-routing-eval run --config config/settings.yaml \
  --dataset examples/offline_demo_cases.json \
  --output artifacts/offline_demo.jsonl
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

For a machine with already installed `pydantic`, `typer`, `PyYAML` and `pytest`:

```bash
PYTHONPATH=src python -m llm_routing_eval.cli validate-config
PYTHONPATH=src python -m llm_routing_eval.cli run
PYTHONPATH=src python -m pytest -q
```

**This authoring environment:** Python 3.13.5; the 40 test cases passed and application/CLI/demo statement coverage was approximately 92%. `uv.lock` is **not updated** because the registry is inaccessible; `uv lock --offline` failed. `ruff` and `mypy` are unavailable, and neither Docker nor Python 3.12 was verified. CI/Docker are owned by Member 04 and still show scaffold behavior; no green status is claimed.

## Observed behavior and deliberately missing functionality

- Offline only; `--live` and any `BENCHMARK_MODE=live` are rejected. No credentials or external calls.
- Only the **Always-Large test double** is wired in the CLI. Orchestrator also supports multi-stage Cascade and bounded same-model retries through the Port protocol.
- Before routing, the original case metadata is stripped and input is sanitized. The provider prompt receives a second privacy check. The demo privacy adapter **only allows two fixed synthetic sentences**; it is not a general PII sanitizer.
- Provider attempts, retry events, router decisions, safe normalized failures, unique IDs and elapsed time are saved through a repository Port. The demo repository is in-memory; the CLI writes sanitized JSONL atomically. Member 04 must supply durable JSONL/SQLite persistence and privacy review.
- Blocked/failed cases are not scored or sent to a judge; untrusted evaluator prose is dropped. Real financial metrics, quality gates and the expert-verified golden set remain Member 03's responsibility.
- Demo token usage and model latency are **simulated**. End-to-end elapsed time reflects only the local fake workflow; API costs remain `null`/unknown. P95 is only populated for at least 20 recorded cases.
- Output order is traces, evaluations, report. The write is atomic locally, but the repository Port does not promise transactional multi-case persistence. Member 04 must define crash-recovery and idempotency.

## Adapter handoff

1. **Member 02:** Supply `RouterPort`, `LLMProviderPort`; configure allowed models and provider mapping. The orchestrator validates stage, strategy, model allowlist and provider/attempt IDs. Routers do not call providers. Provider failures use typed status, retryability and safe codes.
2. **Member 03:** Supply `EvaluatorPort` with verified rubric and controlled reference access. Evaluator scores require status/ID consistency; metric keys and critical error identifiers must be safe short codes. Do not put PII/answers in free-text metadata.
3. **Member 04:** Supply `PrivacyPort` and `ResultRepositoryPort`; configure fail-closed PII policy, reports, provenance, execution resource limits, GitHub CI/Docker and non-secret config. Raw case input must not be persisted.

## Remaining team integration gates

- Members 02–04 review the contracts and examples before implementation is merged.
- Replace demo fakes with owned adapters and test interoperation.
- Regenerate and commit `uv.lock` with package-index access; run Ruff, mypy, Python 3.12 and Docker validation.
- Add durable audit provenance for model and prompt hashes, source revision, rubric versions and pricing when those adapters exist. Do not claim live model access or banking quality verification beforehand.
