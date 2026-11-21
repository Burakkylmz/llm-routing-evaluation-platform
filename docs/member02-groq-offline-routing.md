# Member 02 — YAML routing and Groq compatibility

**Status:** Offline CLI/router wiring and Groq adapter are implemented in a feature branch for review. Real Groq calls, production privacy checks, verified financial quality, durable storage, real costs and approved benchmark access remain **out of scope / not yet done**.

## Offline routing (no network)

The CLI uses the application's existing orchestrator with an **in-process fake LLM provider**, the two existing allowlisted synthetic inputs, the demo-only evaluator and an in-memory repository. The actual Member 02 routing algorithms can now be selected by YAML or `--strategy` without modifying Core interfaces.

```bash
uv sync --locked
uv run --locked llm-routing-eval validate-config --config config/settings.yaml
uv run --locked llm-routing-eval run --config config/settings.yaml \
  --strategy always_large --output artifacts/always-large.jsonl
uv run --locked llm-routing-eval run --config config/settings.yaml \
  --strategy rule_based --output artifacts/rules.jsonl
uv run --locked llm-routing-eval run --config config/settings.yaml \
  --strategy knn --output artifacts/knn.jsonl
uv run --locked llm-routing-eval run --config config/settings.yaml \
  --strategy cascade --output artifacts/cascade.jsonl
```

Use `benchmark.strategy` in the YAML instead of `--strategy` to choose the configured default. `--strategy` changes only the current run; it does **not** enable real providers.

The `routing` YAML key contains:
- `models.small` and `models.large`: distinct fake model identifiers, not Groq/NVIDIA access.
- `rule_based.rules`: first-match ordered scenario types with safe uppercase reason codes; conservative default tier.
- `knn.k`, `minimum_similarity`, `embedding_dimensions`, `examples`: strictly synthetic, explicitly `split: development` examples only. The CLI rejects reused benchmark case IDs and verbatim case text as a basic leak guard. It does not prove statistical independence. kNN hashing is **lexical**, not validated semantic similarity.
- `cascade.escalate_scenarios` and `escalate_on_failure`: small-first, explicit policy. The demo reserves two model-call slots; call/attempt budget belongs to the orchestrator.

The YAML parser forbids unknown configuration keys and invalid enum/rule/split values. Both `--live` and `BENCHMARK_MODE=live` are rejected; live provider settings are not reachable from this CLI. Demo evaluation only checks synthetic fake response equality; it is **not** financial accuracy, quality calibration or production throughput. Demo token usage is simulated; cost remains unknown (not zero).

## Groq adapter (isolated, not wired live)

Implementation: `src/llm_routing_eval/adapters/providers/groq.py` implements async `LLMProviderPort` via a transport `Protocol`, with no Core/CLI changes. It uses Groq's official Chat Completions endpoint:

`POST https://api.groq.com/openai/v1/chat/completions`

Reference: https://console.groq.com/docs/api-reference

Safeguards:
1. `GroqConfig(live_enabled=False, privacy_approved=False)` refuses traffic by default. Both must be deliberately enabled by an approved *future composition root*. This flag is **not** a substitute for Member 04's privacy implementation.
2. `GROQ_API_KEY` is read only from the process environment, only after gates/model validation; no key is stored in a trace or report. Requested model must be in an explicit allowlist.
3. Reject unsupported generation parameters. The adapter handles `temperature`, `max_completion_tokens`, optional `top_p`; individual model capability restrictions still need to be verified with Groq. No automatic substitution of models.
4. One transport attempt per `generate`. The **orchestrator** owns retry, limit and trace policy. HTTP redirects are disabled, response size bounded and raw provider error bodies never surfaced. 429, timeouts, auth/model failures, 4xx and 5xx map to safe error enums.
5. Provider token counts are used when present and valid, otherwise remain `None`; no fabricated pricing, no assumed model availability and no cost estimate based on unverified rates.
6. Successful text is ephemeral. Member 04 must implement actual PII redaction and response persistence restrictions before using any external provider. The stdlib transport's thread-based synchronous I/O can outlive task cancellation; review cancellation, total time budgets and idempotency before any live rollout.

**These unit tests are transport-mocked and never call Groq.** No key or real model execution is needed.

```bash
uv run --locked pytest -q tests/unit/test_groq_provider.py
uv run --locked pytest -q tests/integration/test_cli_offline_routing.py
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
```

## Cross-team acceptance / live Go-No-Go

- Member 01 reviews CLI composition and Core orchestrator call-budget invariants.
- Member 03 reviews disjoint development/held-out split, calibrated scoring and comparative evaluation fairness.
- Member 04 implements/reviews **actual** pre-provider privacy sanitization, safe persistence and retention, budget and secret governance.
- Human approval of synthetic-only live cases, Groq key/model/account access, rate/cost limits, quotas, max attempts/judge costs and actual pricing.
- Controlled live smoke followed by equal-workload multi-strategy evaluation with clearly distinguished observed vs simulated metrics. No CI-triggered live calls.

See [Issue #7](https://github.com/Burakkylmz/llm-routing-evaluation-platform/issues/7) for the integration gates.
