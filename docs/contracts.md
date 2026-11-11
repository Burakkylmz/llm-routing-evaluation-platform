# Core Contracts (M1.1 — Member 01)

**Implementation status:** Pydantic v2 schemas and Python `Protocol` definitions implemented; adapter implementations and human code review remain pending. Schema version: `1.0.0`.

## Public types

- **Seven versioned records:** `BenchmarkCase`, `GoldenReference`, `RoutingDecision`, `ModelResponse`, `EvaluationResult`, `ExecutionTrace`, `BenchmarkReport`.
- **Value objects:** `Evidence`, `TokenUsage`, `ErrorInfo`, `TraceStep`, `StrategyMetrics`.
- **Internal DTOs:** `ModelRequest`, `RoutingContext`, `RoutingFeedback`, `SanitizedText`.
- **Ports:** `RouterPort`, `LLMProviderPort`, `EvaluatorPort`, `PrivacyPort`, `ResultRepositoryPort`, `EmbeddingPort`.

Generated JSON Schemas are in [`docs/schemas/`](schemas/), with seven matching
synthetic payloads in `tests/fixtures/contract_examples_v1.json`.

Import from `llm_routing_eval.core`. Versioned records reject unexpected fields, invalid enum values, unsupported schema versions and invalid required data. Model attributes are frozen; **nested Python dicts are mutable** and must not be modified after construction. Serialization uses `model_dump_json()`.

## Examples

```python
from llm_routing_eval.core import (
    BenchmarkCase,
    RoutingAction,
    RoutingDecision,
    StrategyId,
)

case = BenchmarkCase(
    case_id="dev-card-fee-001",
    input_text="Kredi kartı yıllık ücreti nedir?",  # synthetic input only
    domain="turkish_banking",
    scenario_type="card_fee",
    metadata={"synthetic": True},
)

decision = RoutingDecision(
    strategy=StrategyId.ALWAYS_LARGE,
    action=RoutingAction.INVOKE_MODEL,
    selected_model="openai/gpt-oss-120b",
    reason_code="BASELINE_LARGE",
    stage_index=0,
    decision_time_ms=0.4,
)
```

## Boundary requirements

1. `BenchmarkCase` does not carry golden answers; only `EvaluatorPort` receives `GoldenReference`. `metadata` is caller-controlled and must be filtered by the dataset loader before router use.
2. `ModelRequest.prompt` must already be sanitized. `SanitizedText` validates blocked/allowed state but does **not** perform masking. Privacy adapter is a separate M1 deliverable.
3. `TraceStep` stores safe attempt metadata and token counts but never embeds `ModelResponse.content`. The producer must restrict `safe_metadata`, `safe_reason` and error messages to sanitized data.
4. A provider failure has `ModelResponse.status != SUCCESS`, `content=None`, and a structured `ErrorInfo`. Missing telemetry remains `None`.
5. `RoutingContext` contains permitted prior attempts only; it has no golden reference or judge score.
6. `EvaluatorPort.evaluate()` receives **keyword-only** `run_id` and `trace_id` so its mandatory result correlation fields can be populated without hidden state. This is a documented correction to the initial logical M0.2 signature.
7. This step defines contracts and checks them offline; no real API calls, benchmark runner, CLI, provider, privacy implementation or report engine were added.

## Review checklist

- Member 02: router context and response semantics, retry/attempt identifiers.
- Member 03: reference visibility, evaluator correlation and score coverage.
- Member 04: trace safety, serialization, timestamps, monetary representations.

Contract changes require affected owners' review and an amendment to `docs/decisions.md`.

## Local verification

```bash
# From repository root, with Pydantic v2 and pytest available:
PYTHONPATH=src python -m pytest -q tests/contract

# With Python 3.12+ and the committed lockfile:
uv lock --check
uv sync --locked
uv run pytest -q tests/contract
```

The committed `uv.lock` is synchronized with `pyproject.toml`; contract tests,
Ruff and strict mypy pass and run in the offline CI workflow. Docker remains
unverified (Member 04).
