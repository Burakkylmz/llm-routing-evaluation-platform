# Member 02 — Offline Routing Implementation

**Scope:** Real offline routing decisions behind the existing Core `RouterPort`.
**Status:** Four offline routers merged to main. YAML/CLI switching and network-gated Groq adapter added in a follow-up PR for review. Live model access, privacy enforcement, durable cost accounting and expert-verified quality evaluation remain outstanding. The NVIDIA Build M5 adapter is separately on main, offline-tested only.

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
2. Validate/review the completed YAML router configuration and offline CLI composition with Member 01.
3. Validate/review the completed Groq adapter behind `LLMProviderPort` with Member 04 privacy and safe persistence implementation before live wiring.
4. Review the existing NVIDIA Build M5 adapter separately before live use; its offline transport tests are complete, but model availability, account access and live benchmark integration are unverified.


## F6 — Turkish lexical casefold and reproducibility (Issue #23)

The offline SHA-256 feature-hash embedding now applies Unicode NFKC and
Turkish dotted/dotless-I case mapping before generic casefold/tokenization.
Examples such as `İŞLEM` / `işlem`, `IŞIK` / `ışık`, and decomposed dotted-I
must yield identical vectors under this **Turkish-domain-specific** rule.
`İ` and `I` are distinct letters and remain distinct after normalization.

This changes kNN token hashing and can change its routing decisions. The
composition-root fingerprint therefore records the versioned embedding ID
`sha256-lexical-feature-hash-tr-casefold-v2`, not the old
`sha256-lexical-feature-hash-v1`. Re-run offline kNN results before comparing
strategies across versions; existing results must not be silently relabeled.
The adapter remains **lexical rather than semantic** and is intended only for
synthetic, offline Turkish banking evaluation. Member 03 must still approve
held-out split provenance and evaluation methodology before any live benchmark.


## F5 — Extensible Cascade escalation policy (Issue #31)

`CascadeRouter` now coordinates the existing small-first, stage-one, then
finish routing sequence while delegating *why to escalate* to a typed local
`CascadeEscalationPolicy`. The immutable default policy
`ConfiguredCascadeEscalationPolicy` preserves the original scenario match
and failure-based escalation choices and exact machine-readable reason codes:
`CASCADE_SCENARIO`, `CASCADE_ACCEPT_SMALL`, `CASCADE_FAILOVER` and
`CASCADE_STOP_FAILURE`. Stage zero and terminal-stage codes are unchanged.

A caller may inject a separately implemented policy into `CascadeRouter`
without editing its stage coordination or the shared `RouterPort` contract.
Injected policy decisions are validated, including a boolean escalation flag
and an allowlisted uppercase reason code; unsafe, ambiguous or invalid
configurations fail closed. Existing YAML, production provider wiring and
routing fingerprints are unchanged: the current offline CLI **only** constructs
the configured built-in policy. No arbitrary policy plugin is loaded from YAML.

**Scope of this F5 deliverable:** the strategy implementation is now open
for controlled local escalation-policy extension. The application
orchestrator still explicitly distinguishes Cascade from one-stage baselines
for retries and multi-stage accounting. That is **Member 01's responsibility**
and cannot safely be removed without a separately agreed Core/RouterPort
capability contract, call-budget semantics, fingerprinting and integration
tests. This refactoring does not claim to resolve that cross-team OCP concern,
nor does it authorize live model calls.
