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

## Opt-in live provider smoke tests (`pytest -m live`)

`tests/live/` holds one smoke test each for Groq and NVIDIA Build. Each test sends a **single fixed synthetic prompt** ("Reply with the single word: OK") through the production path: provider, cancellable HTTPX transport, and an `OutboundAuthorization` that permits only that prompt. The tests are **skipped by default**. They also refuse to run in CI (when `CI` is set or `BENCHMARK_MODE=offline`), even with the opt-in flag present. `tests/unit/test_live_test_gating.py` guards this behaviour offline.

Run them only after the account owner has confirmed model access and the privacy owner (Member 04) has approved a synthetic-only pilot:

```bash
LLM_ROUTING_LIVE_TESTS=1 GROQ_API_KEY=... uv run pytest -m live tests/live -q
```

| Variable | Purpose |
| --- | --- |
| `LLM_ROUTING_LIVE_TESTS=1` | Required opt-in |
| `GROQ_API_KEY`, `GROQ_LIVE_TEST_MODEL` | Groq key; model defaults to `openai/gpt-oss-20b` |
| `NVIDIA_API_KEY`, `NVIDIA_LIVE_TEST_MODEL` | NVIDIA key and model (no default: availability is unverified) |
| `LIVE_TEST_MAX_OUTPUT_TOKENS` | Optional completion budget (default 256) |

A pass proves wire compatibility and account/model access only. It is not evidence of quality, latency SLOs or cost. Record the date, model ID and result in the handoff sign-off.

## F7 — Stable routing fingerprints for behavior-equivalent YAML order (Issue #27)

The offline policy digest now canonicalizes the **set** of scenario types
inside each Rule-Based rule and sorts kNN development examples by their
unique example IDs. Neither operation changes the router's in-memory input,
configured first-match rule priority, kNN neighbor ordering, or selected model.
The order of Rule-Based **rules** remains significant and is never sorted;
kNN example ID, text and assigned model tier remain part of the digest
because they can affect selection and tie breaking. Thresholds, embedding
version and model IDs remain material to policy provenance.

This reduces spurious new fingerprints for behavior-equivalent YAML edits; it
is not a proof that arbitrary routing policies are semantically equivalent.
Historic fingerprint values remain immutable in stored reports. In particular,
previously computed fingerprints must not be rewritten or reassigned to old
results. A SHA-256 run fingerprint is **not a privacy/redaction guarantee**:
privacy approval and held-out data isolation are independent gates.
