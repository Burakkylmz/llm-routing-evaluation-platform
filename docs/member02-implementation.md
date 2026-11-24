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

## F9 — Retry backoff provenance on current main (Issue #29)

PR #34 already introduced positive bounded **equal jitter** retry delays:
`ceiling = min(2.0, 0.5 * 2**index)` seconds; wait is
`ceiling / 2 + (ceiling / 2) * random.random()`. The first retry
therefore waits at least **250 ms**, with a **2 s** maximum.

PR #35 added HTTP 429 `Retry-After` handling. A valid hint is treated as a
minimum wait; if it exceeds the configured `max_retry_wait_seconds`, retries
stop rather than firing earlier than requested. This PR **preserves** both
merged behaviors instead of reintroducing an older, incompatible backoff
function. The application still enforces the configured retry and model-call
budgets, cancellation and fail-closed safeguards.

For reporting reproducibility, the existing policy is labeled
`equal-jitter-0p5s-429-v1` in benchmark report provenance and the run
fingerprint **only when retry attempts are enabled**. Runs with
`max_retries=0` retain the current main branch's fingerprint formula.
Historical results and hashes are immutable; never relabel prior runs.

The attached synthetic offline tests validate backoff bounds, retry sequencing,
no extra calls after terminal outcomes, cancellation propagation and
fingerprint differences. They do not authorize live Groq/NVIDIA benchmarks or
replace Member 01, Member 03 and Member 04 approvals.

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
for retries and multi-stage accounting. That is **Member 01's responsibility** (tracked in Issue #33)
and cannot safely be removed without a separately agreed Core/RouterPort
capability contract, call-budget semantics, fingerprinting and integration
tests. This refactoring does not claim to resolve that cross-team OCP concern,
nor does it authorize live model calls.

## Development-only routing calibration (`calibrate-routing`)

`calibrate-routing` derives the kNN `k` and `minimum_similarity` values and the Cascade `escalate_scenarios` list from labeled **development** examples, instead of hand-picking them. It never uses held-out data.

```bash
uv run llm-routing-eval calibrate-routing --examples examples/synthetic_development_calibration.json
```

- **Input** (Member 03 hand-off; see `examples/synthetic_development_calibration.json`): `schema_version`, `dataset_version`, `split: "development"`, and examples with `example_id`, `input_text`, `scenario_type`, `tier` (the smallest sufficient tier), `split: "development"` and `synthetic: true`. Any other split or extra field (for example golden facts) is rejected.
- **kNN:** leave-one-out over `k` × `minimum_similarity`, with the same embedder and vote as `KnnRouter`. The under-routing rate (large-labeled examples routed to small) must stay within `--max-under-routing-rate` (default 0.05). Among the settings that pass, the most accurate wins; ties go to fewer under-routed examples, then the higher threshold, then the smaller `k`. If no setting passes, the command exits `3`, writes the report and writes no settings file.
- **Cascade:** a scenario escalates when its large-label share exceeds `--max-small-risk` (default 0.2) or when it has fewer than `--min-scenario-examples` examples (default 3).
- **Safety:** before anything is written, the calibrated settings pass the strict loader, the **F4 isolation guard** against `--dataset`, and the router factory for both kNN and Cascade. The report contains IDs, counts, rates and the base/calibrated policy digests, never example text.
- **Limits:** development labels are not a held-out quality result. The embedding is the lexical hashing embedding, not validated semantic retrieval. The input must be `synthetic: true` until a policy for real development data is agreed with Members 01, 03 and 04.

## Four-strategy offline comparison (`compare-routing`)

`compare-routing` runs Always-Large, Rule-Based, kNN and Cascade on the **same** synthetic cases, with the fake provider and evaluator. It writes one JSON report.

```bash
uv run llm-routing-eval compare-routing --output artifacts/offline_comparison.json
```

- **Same rules as a standalone `run`:** each strategy gets its own validated `RunOptions` from `demo_options` (never `model_copy`), so the Cascade budget guard (F3) and the per-strategy policy digest (F7) apply exactly as they would in a standalone run. All four option sets are validated before any strategy runs.
- **F4 isolation:** kNN is always included, so the isolation guard always runs against `--dataset`.
- **Per-strategy report:** success, **strict success** (excludes Cascade answers kept after a failed escalation, F2), degraded cases, provider attempts and attempts per case, calls by model, the simulated token totals and per-attempt averages (only when usage is complete), the fixture-only score, local timings, and the routing policy digest.
- **Workload identity:** the dataset's content digest is recorded. Prompts and answers are never written.
- **Limits:** this is an offline simulation. Cost is *unknown* (no verified pricing), not zero; timings are local, not provider latency; fixture-match is a contract check, not answer accuracy.

## Groq live-readiness preflight (`groq-preflight`)

`groq-preflight` lists every blocker between this repository and a live Groq benchmark. It makes **no network call** and only checks whether `GROQ_API_KEY` is set; it never prints or stores the key.

```bash
uv run llm-routing-eval groq-preflight          # human-readable
uv run llm-routing-eval groq-preflight --json   # machine-readable
```

- **Local checks:** the Groq provider is enabled (`config/providers.yaml`); routing models match the Groq candidate models; execution is `live` with `live_enabled`; `GROQ_API_KEY` is set; pricing is `verified` with input and output rates for both candidate models (`config/pricing.yaml`).
- **External blockers, always listed, each with an owner:** outbound privacy authorization (Member 04), expert evaluation (Member 03), model/account access recorded by a live smoke test (Member 02), and the live CLI path (Member 01).
- **Always BLOCKED.** These approvals cannot be inferred from a local toggle or a working key, so the command always exits `2` with `GROQ_PREFLIGHT_STATUS: BLOCKED`. CI asserts this. Invalid configuration exits `3`.
