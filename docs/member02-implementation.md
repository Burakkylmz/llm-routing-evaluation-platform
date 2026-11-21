# Member 02 — Offline Routing Implementation

**Scope:** Real offline routing decisions behind the existing Core `RouterPort`.
**Status:** Implemented for review on Member 02's feature PR. Real provider adapters, model access, privacy, durable cost accounting, and expert-verified quality evaluation remain outstanding.

## Architecture and boundaries

- `routing/strategies.py` implements four exchangeable strategies using the shared typed contracts; no direct provider, persistence, CLI, domain-bank, or evaluator imports.
- `routing/embeddings.py` implements a deterministic local SHA-256 lexical feature-hash embedding adapter behind `EmbeddingPort`, without external calls or heavyweight dependencies.
- `ModelPair`, `ScenarioRule`, and `DevelopmentExample` are frozen validated configuration objects. Scenario rules are first-match with configuration-order precedence and a conservative large-model default.
- `KnnRouter.prepare(...)` precomputes an index from explicitly labeled **development-only** examples. Prepared vectors are reused for query routing; stable neighbor ordering and large-model tie/low-similarity fallback are explicit.
- `CascadeRouter` calls the small model first. Successful small responses escalate only for explicitly configured scenario types; failed calls may escalate if policy enables it. No confidence-based decision is claimed, because the current `RoutingFeedback` contract lacks a validated confidence signal.
- The existing `BenchmarkOrchestrator` owns privacy checks, retries, provider invocations, error attribution, trace persistence, and evaluation. Routers only produce `RoutingDecision` objects.

## Run checks

```bash
uv sync --locked
uv run pytest -q tests/unit/test_routing_strategies.py tests/integration/test_real_offline_routers.py
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

The integration test executes all four routers on the **same synthetic, allowlisted offline prompt** with the demo fake provider. The fixture evaluator proves orchestration compatibility, **not financial correctness** or superiority of a strategy.

## Data governance and reproducibility

- No held-out/golden case may be a `DevelopmentExample`; do not load golden references into router processes.
- No routing decision uses evaluator scores or private golden labels.
- All explanations are short allowlisted machine-readable reason codes; user text is never included in them.
- Hashing embeddings are lexical, not semantic, and should **not** be used to make real financial-quality or language-understanding claims. Production-calibrated embeddings require an independently validated adapter.
- Track future benchmark calibration dataset version, index/embedding version, similarity threshold, rule ordering, cascade escalation scenarios, and model identifiers in run provenance before reporting comparative results.
- Do not activate live provider calls until Member 04 privacy controls and Member 03 evaluation datasets/rubrics are integrated and reviewed.

## Remaining Member 02 milestones

1. Review routing semantics and `RouterPort` contracts with Members 01/03/04.
2. Add validated YAML router configuration and application composition integration, keeping the offline-only defaults.
3. Add the Groq provider adapter behind `LLMProviderPort` with safe error normalization, usage attribution, explicit opt-in, and a network-disabled test suite.
4. After MVP, implement and verify a separate NVIDIA Build compatibility adapter.
