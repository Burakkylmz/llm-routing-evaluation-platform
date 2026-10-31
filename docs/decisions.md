# LLM Routing Evaluation Platform — Decisions

**Status:** Design baseline adopted; Member 01 Core, offline Orchestrator and CLI implemented with demo fakes. Member 02–04 modules remain pending.\
**Last updated:** 2026-10-31\
**Decision authority:** Project owner; implementation contracts remain subject to technical review by affected module owners.\
**Scope:** Product decisions, M0.1–M0.3, M1 offline vertical slice, M1.1 contract freeze, and M1.2 repository/module boundaries.\
**Language:** English. All paths, public identifiers, API names, and documentation use English.

> **Status distinction:** Decisions in this file are the agreed *design baseline*. No claim is made that code exists, external models have been verified, a banking expert has signed off, or all four contributors have reviewed the schemas. An implementation change requires an explicit, documented amendment; the project owner has delegated remaining planning choices rather than requesting approval at each step.

## 1. Product, MVP, and Architecture Baseline

| ID | Decision | Reason |
| --- | --- | --- |
| P-01 | Build an open-source **LLM Routing Evaluation Platform**, not a new routing algorithm framework or production gateway. | Validate whether additional routing complexity is worthwhile on a user's own workload. |
| P-02 | MVP domain pack: **Turkish Retail Banking & Customer Support** (loans/interest, cards/accounts, and banking fees). Keep the core domain-independent. | Deep, measurable evaluation without spreading a four-person team across sectors. |
| P-03 | Compare **Always-Large, Rule-Based, kNN, and Cascade** against the same eligible cases and controlled model pool. | Simple baselines are essential to assessing marginal value. |
| P-04 | Hybrid evaluation: deterministic checks, rubric-based LLM judge, and human review. Use **quality-first** comparison and expose safety separately. | An inexpensive but unsafe answer is not a successful optimization. |
| P-05 | Treat cost, response quality, latency, reliability, routing explanations, and audit evidence as separate reported dimensions; no single opaque overall score. | Preserve trade-offs and uncertainty. |
| P-06 | Start with **synthetic** development data; build a separately versioned, held-out golden set with explicit expert-review status. Never describe unreviewed records as expert-verified. | Prevent real-customer-data and evidence-integrity risks. |
| P-07 | Use a **modular monolith** with pragmatic Hexagonal Architecture / Ports & Adapters, OOP, SOLID, Dependency Injection, Strategy for routers, and Adapter for providers. Apply further patterns only when needed. | Testable module boundaries without microservice overhead. |
| P-08 | **CLI-first, offline-by-default** MVP; no FastAPI/web dashboard, PostgreSQL, Kubernetes, fine-tuning, production banking integration, or multi-sector launch. | Control scope and avoid requiring paid services for tests. |
| P-09 | Runtime: **Python 3.12+**, **uv** with committed `uv.lock`, **TOML** (`pyproject.toml`), **YAML** runtime configs, **Docker + Compose**, Pydantic v2, Typer, HTTPX, pytest/pytest-cov, Ruff, mypy, JSONL and later SQLite, GitHub Actions, structured logging. | Reproducible, accessible open-source developer experience. |
| P-10 | **Groq** is the first live provider; **NVIDIA Build** is the follow-on compatibility milestone. Read keys from `GROQ_API_KEY` and `NVIDIA_API_KEY` environment variables only. | Use available credentials without embedding secrets in config or code. |
| P-11 | Small/large model candidates: Groq-hosted `openai/gpt-oss-20b` and `openai/gpt-oss-120b`, **subject to runtime access and pricing verification**. Model IDs must be configurable. | Do not confuse a suggested pair with verified availability or a guaranteed quality ordering. |

## 2. M0.1 — Data Contracts (Adopted)

| ID | Decision |
| --- | --- |
| M01-01 | Exactly seven **top-level benchmark records** in Pydantic v2: `BenchmarkCase`, `GoldenReference`, `RoutingDecision`, `ModelResponse`, `EvaluationResult`, `ExecutionTrace`, `BenchmarkReport`. Small nested value objects are allowed. |
| M01-02 | Use `ConfigDict(frozen=True, extra="forbid")` for persisted records and immutable snapshots/events for updates. Free-form safe `metadata` must still have a defined JSON-serializable type. |
| M01-03 | Include `schema_version: str` (starting at `"1.0.0"`) in every persisted top-level record; breaking schema changes increase the major version and require migration/compatibility tests. |
| M01-04 | Banking-specific labels, evidence, rubrics and scoring live in `domains/turkish_banking`, not in the domain-independent Core. |
| M01-05 | Only evaluators receive `GoldenReference`; routers, embeddings and live provider adapters never receive golden answers, rubrics, expected facts or evaluation scores. |
| M01-06 | An `ExecutionTrace` represents one `(run_id, case_id, strategy)` execution and contains multiple routing decisions, provider attempts/retries, and failures, including Cascade escalation. |
| M01-07 | Privacy-safe data only in persisted fields; no keys, unmasked customer identifiers or raw high-risk inputs in trace/report/log. |

### 2.1 Frozen logical field specifications for implementation

All identifiers are nonempty strings. All times are UTC ISO-8601 when serialized. Durations are milliseconds and nonnegative; monetary amounts use decimal-string values in a documented currency, not binary floating point. `None` means **unknown / not applicable**, never zero. Persisted `metadata` values must be sanitized JSON primitives/arrays/objects.

| Record | Required fields (types) | Optional/conditional fields |
| --- | --- | --- |
| `BenchmarkCase` | `schema_version: str`, `case_id: str`, `input_text: str`, `domain: str`, `scenario_type: str`, `metadata: dict[str, JsonValue]` | `locale: str = "tr-TR"`; case does **not** contain golden reference fields |
| `GoldenReference` | `schema_version: str`, `case_id: str`, `dataset_version: str`, `expected_facts: tuple[str, ...]`, `rubric_version: str`, `verification_status: VerificationStatus` | `evidence: tuple[Evidence, ...]`, `reviewed_at: datetime | None`, `reviewer_role: str | None`; never copy into case payload |
| `RoutingDecision` | `schema_version: str`, `strategy: StrategyId`, `action: RoutingAction`, `reason_code: str`, `stage_index: int`, `decision_time_ms: float` | `selected_model: str | None`, `safe_reason: str | None`; `selected_model` required for `INVOKE_MODEL`, absent for `FINISH` |
| `ModelResponse` | `schema_version: str`, `provider: str`, `model: str`, `attempt_id: str`, `status: CallStatus`, `metric_origin: MetricOrigin` | `content: str | None`, `usage: TokenUsage | None`, `latency_ms: float | None`, `error: ErrorInfo | None`, `provider_request_id: str | None` |
| `EvaluationResult` | `schema_version: str`, `run_id: str`, `case_id: str`, `trace_id: str`, `quality_scores: dict[str, float | None]`, `critical_errors: tuple[str, ...]`, `evaluation_method: str`, `evaluator_version: str`, `status: EvaluationStatus` | `review_status: str | None`, `safe_notes: tuple[str, ...]`, `judge_usage: TokenUsage | None` |
| `ExecutionTrace` | `schema_version: str`, `trace_id: str`, `run_id: str`, `case_id: str`, `strategy: StrategyId`, `steps: tuple[TraceStep, ...]`, `status: ExecutionStatus` | `started_at: datetime`, `ended_at: datetime | None`, `total_latency_ms: float | None`, `error: ErrorInfo | None` |
| `BenchmarkReport` | `schema_version: str`, `run_id: str`, `dataset_version: str`, `configuration_fingerprint: str`, `strategy_metrics: dict[str, StrategyMetrics]`, `generated_at: datetime` | `pricing_version: str | None`, `run_status: str`, `limitations: tuple[str, ...]`, `provenance: dict[str, str]` |

**Nested value objects:** `Evidence` (source identifier, effective/as-of date, verification/provenance), `TokenUsage` (input/output/cached/reasoning counts where provided; nullable fields), `ErrorInfo` (category, safe code, retryability, safe message), `TraceStep` (sequence number, kind, action/attempt association, safe payload/measurements), `StrategyMetrics` (counts, failures, quality/safety, latency percentiles, costs and coverage, each with available sample counts). `ModelRequest` and `RoutingContext` are **internal typed transfer objects**, not additional persisted top-level records.

**Enums / validation:** `RoutingAction = INVOKE_MODEL | FINISH`; `CallStatus = SUCCESS | FAILED | TIMEOUT | RATE_LIMITED`; `ExecutionStatus = COMPLETED | PARTIAL | FAILED | BLOCKED`; `EvaluationStatus = COMPLETED | PARTIAL | NOT_EVALUATED | FAILED`; `VerificationStatus = UNREVIEWED | REVIEW_PENDING | EXPERT_VERIFIED`; `MetricOrigin = SIMULATED | OBSERVED`. Strategy IDs: `always_large`, `rule_based`, `knn`, `cascade`. Nullability on absent provider usage must be preserved. Do not enforce an arbitrary zero default for missing telemetry.

**Safety validation:** `ModelResponse.content` is required for a successful call and absent on failed calls; failed calls require `error`. Trace-event order is strictly increasing. An execution may finish with no model response only if it failed/was blocked. A `BenchmarkReport` identifies the exact run and input snapshot; it must not present synthetic fake metrics as observed performance.

## 3. M0.2 — Port and Orchestration Contracts (Adopted)

| ID | Decision |
| --- | --- |
| M02-01 | Define port abstractions at the Core/Application boundary; implement concrete adapters outside Core. Core cannot import provider SDKs, SQLite, CLI or domain-pack implementations. |
| M02-02 | Use constructor-based Dependency Injection at a single CLI composition root; no service locator or implicit global singleton. |
| M02-03 | Router decides; **orchestrator** owns provider calls, privacy enforcement, retries, tracing and termination. Router never invokes an LLM API itself. |
| M02-04 | A router can be re-entered with a **sanitized**, golden-free `RoutingContext` of previous attempts to support Cascade. Enforce a configured maximum model-call budget/stage count. |
| M02-05 | Golden references are provided to `EvaluatorPort` only after provider execution; no scoring feedback from held-out cases goes back to router/calibration. |
| M02-06 | Privacy guard runs before **external transmission**, embeddings that use external services, and persistence; if sanitization cannot satisfy policy, **fail closed** with a traceable blocked status and no outbound call. |
| M02-07 | Every provider attempt, retry, timeout, rate-limit and failure is a distinct trace event; failures do not silently disappear from benchmark denominators. |
| M02-08 | `EmbeddingPort` is separate from `LLMProviderPort`; embedding implementations and reference stores are replaceable. kNN index construction uses only approved development/calibration data. |

### 3.1 Method signatures (logical Python contracts)

Use asynchronous I/O-facing methods. Pure, deterministic privacy processing is synchronous. The methods below are **interface signatures**, not a request to implement code at this planning stage.

```python
class RouterPort(Protocol):
    async def decide(self, case: BenchmarkCase, context: RoutingContext) -> RoutingDecision: ...

class LLMProviderPort(Protocol):
    async def generate(self, request: ModelRequest) -> ModelResponse: ...

class EvaluatorPort(Protocol):
    async def evaluate(
        self, case: BenchmarkCase, response: ModelResponse | None,
        reference: GoldenReference | None, *, run_id: str, trace_id: str,
    ) -> EvaluationResult: ...

class PrivacyPort(Protocol):
    def sanitize(self, text: str, *, purpose: str) -> SanitizedText: ...

class ResultRepositoryPort(Protocol):
    async def save_trace(self, trace: ExecutionTrace) -> None: ...
    async def save_evaluation(self, result: EvaluationResult) -> None: ...
    async def save_report(self, report: BenchmarkReport) -> None: ...

class EmbeddingPort(Protocol):
    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...
```

`SanitizedText` contains `text`, `policy_version`, and a safe `status`/reason code. `RoutingContext` contains ordered prior **permitted** decisions, safe provider response excerpts and failure metadata, current stage and remaining budget; **not** golden answers or scores. `ModelRequest` contains provider/model ID, sanitized prompt, generation parameters, timeout and correlation IDs; not credentials. Method contracts must carry well-defined cancellation and timeout behavior through application orchestration.

### 3.2 Errors and retry policy

| ID | Error category | Contract and handling |
| --- | --- | --- |
| E-01 | Input/schema validation | Reject invalid case/config before execution; record safe validation code, never silently coerce wrong data. |
| E-02 | Privacy blocked | No API call, no unsafe payload persisted; emit `BLOCKED` trace and explicit coverage accounting. |
| E-03 | Provider timeout / transient network | Emit failed attempt; bounded retry with jittered exponential backoff only if configured; account for all attempts. |
| E-04 | Provider rate limit | Respect `Retry-After` where supplied; bounded retries; report throttling as separate failure reason. |
| E-05 | Routing / evaluation exception | Record safe typed error and failure status; do not fabricate a score or substitute another router without marking it. |
| E-06 | Persistence failure | Fail the run visibly; write neither a success summary nor a misleading partial report. Retry storage operations only when idempotent. |

Expected provider failures are returned as `ModelResponse` with a failure status and `ErrorInfo`. Unexpected adapter exceptions are caught at the orchestrator boundary and converted into typed, sanitized failure events. Do **not** retry privacy or schema failures. For M1, retries default to `0`; live-run retry limits are configurable and finite.

## 4. M0.3 — Engineering and Integration Governance (Adopted)

| ID | Decision |
| --- | --- |
| M03-01 | GitHub Flow: short-lived feature branches and pull requests; no direct commits to protected `main`. |
| M03-02 | At least one reviewer other than author for merge; changes affecting shared contracts additionally require affected owners' review. |
| M03-03 | CI checks: `ruff check`, `ruff format --check`, `mypy`, `pytest` (including smoke/contract tests); all are network-free. |
| M03-04 | OOP/SOLID rules: dependency inversion, small responsibilities, explicit types, composition over accidental inheritance, no unnecessary abstract factories. |
| M03-05 | Version public contracts; changes require examples, tests, documentation and, when breaking, a version increment/migration note. |
| M03-06 | Use `uv sync --locked` for reproducible environment setup and commit `uv.lock`. |
| M03-07 | Public docs, CLI examples, comments, identifiers and repository conventions use English. |
| M03-08 | Keep providers and live judge calls **off** in CI; require explicit live flag, external API key and cost warning. |
| M03-09 | PR Definition of Done: implementation/doc, relevant tests, contract conformity, privacy/secrets review, reproducible usage instructions, and successful quality gates. |

### Ownership

| Contributor | Accountable components |
| --- | --- |
| **Member 01 — Architecture & Core** | Contracts, shared ports, orchestrator, CLI, composition root, architecture consistency and integration. |
| **Member 02 — Routing & Integration** | Always-Large, Rule-Based, kNN, Cascade, fake/live provider adapters, embedding port implementation, safe routing explanations. |
| **Member 03 — Dataset & Evaluation** | Synthetic development fixtures, golden reference isolation, rubrics, deterministic evaluator, hybrid evaluation, expert-review workflow. |
| **Member 04 — Infrastructure & Reporting** | uv/Docker/Compose, YAML/TOML config, secrets, privacy, JSONL/SQLite, trace/report persistence, CI and logging. |

Each member owns unit tests and documentation for their module. Member 01 coordinates shared interfaces but cannot unilaterally redefine another member's semantics. Formal human sign-off on actual implementation contracts remains an **execution gate**, not a claimed completed activity.

## 5. M1 — Offline Vertical Slice (Adopted)

| ID | Decision |
| --- | --- |
| M1-01 | Build only `always_large` first; other strategies are M2. |
| M1-02 | `FakeLLMProvider` returns deterministic content and synthetic token/latency values identified with `metric_origin=SIMULATED`. |
| M1-03 | First evaluator performs deterministic fixture assertions only; no model judge or golden test set in M1. |
| M1-04 | Produce one complete privacy-safe `ExecutionTrace` for each `(run_id, case_id, strategy)` including failures. |
| M1-05 | Persist initial output to JSONL; use SQLite for indexed storage/reporting in M2. |
| M1-06 | Run with **no API key and no network**; live mode requires a separate explicit opt-in at M3. |
| M1-07 | Demonstrate both success and failure paths, including nonzero/unknown fake usage treatment and zero external API charges. |
| M1-08 | CLI command and Docker Compose command must execute the same orchestrated benchmark with identical deterministic fixture outputs. |

**Offline path:** CLI → fixture dataset loader → PrivacyPort → orchestrator → `always_large` RouterPort → fake LLM adapter → deterministic evaluator → privacy-safe trace and JSONL report.

**Work packages:** WP-01 core/contracts/orchestrator/CLI (Member 01); WP-02 Always-Large and fake provider (Member 02); WP-03 small synthetic dev fixture and deterministic evaluator (Member 03); WP-04 privacy, JSONL, Docker and offline CI (Member 04).

### M1 acceptance gates

- One command runs offline end-to-end on a clean installation, including in Docker, without credentials.
- Same fixture and config produce identical **logical** outputs; do not require elapsed wall-clock time or generated UUIDs to match byte-for-byte.
- Trace includes routing action, provider attempt, status, token-source status, error category (if any), and safe correlation identifiers.
- Output distinguishes simulated token counts/latency and *estimated/simulated* cost from zero actual API charges. If no fake pricing rule is configured, estimated cost is `null`.
- Failure and privacy-blocked fixtures are tested; the runner reports non-success rather than marking everything successful.
- `ruff`, `mypy`, `pytest` and contract/integration checks pass offline.

## 6. M1.1 — Contract Freeze Resolution (Adopted Baseline)

| ID | Decision |
| --- | --- |
| CF-01 | Modules exchange only declared versioned contracts/typed port inputs and outputs; no direct cross-module imports of concrete implementations. |
| CF-02 | `RoutingDecision.action` is `INVOKE_MODEL` or `FINISH`; selected model is present only on invocation. Cascade can create multiple routing decisions and model attempts in one trace. |
| CF-03 | Missing token, pricing, cost and latency measurements remain `None`/unknown and **must not** become `0`. |
| CF-04 | Simulated metrics and pricing assumptions are explicitly labeled and not compared as real API observations. |
| CF-05 | Held-out `GoldenReference` travels only from dataset/repository to evaluator; router and embedding services receive case text sanitized for their permitted use. |
| CF-06 | `run_id`, `case_id` and `trace_id` form required cross-record correlation; provider attempts have their own `attempt_id`. |
| CF-07 | Async boundaries for routing, provider, evaluator, embedding and repository; sync privacy sanitizer; orchestrator owns control flow and retries. |
| CF-08 | Timing boundaries are distinct: router decision time; per-provider call latency; end-to-end elapsed time; judge evaluation time separately. Cascade totals include all calls, retries and routing overhead. |

## 7. M1.2 — Repository Structure and Module Boundaries (Adopted)

```text
llm-routing-evaluation-platform/
├── pyproject.toml
├── uv.lock
├── README.md
├── .env.example              # Variable names only; no values/secrets
├── .gitignore
├── .dockerignore
├── Dockerfile
├── compose.yaml
├── config/
│   ├── settings.yaml         # Safe defaults: offline mode
│   ├── providers.yaml        # Model IDs and nonsecret provider settings
│   └── pricing.yaml          # Versioned, explicitly sourced price assumptions
├── src/llm_routing_eval/
│   ├── core/                 # Immutable domain-independent contracts, enums, ports
│   ├── application/          # Benchmark use cases, orchestrator, run controls
│   ├── routing/              # Always-Large, Rule-Based, kNN, Cascade
│   ├── evaluation/           # Generic metric engines and evaluation orchestration
│   ├── adapters/
│   │   ├── providers/        # Fake, Groq; NVIDIA later
│   │   ├── embeddings/       # kNN implementation adapters
│   │   └── persistence/      # JSONL first, SQLite later
│   ├── privacy/              # Sanitization policy and adapter
│   ├── reporting/            # Aggregation, cost attribution, export
│   ├── domains/
│   │   └── turkish_banking/  # Domain pack: rules, fixtures schema, rubrics
│   ├── config/               # Config parsing/validation and precedence
│   └── cli/                  # Typer entry point and dependency composition
├── data/
│   ├── development/         # Public synthetic fixtures
│   └── golden/              # Versioned/curated reference sets, access rules
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   └── fixtures/
├── docs/
│   ├── decisions.md
│   ├── architecture.md
│   ├── evaluation-methodology.md
│   ├── privacy.md
│   └── contributing.md
└── .github/workflows/ci.yaml
```

| ID | Dependency and packaging rule |
| --- | --- |
| RS-01 | `core` may depend on stdlib and Pydantic only; it must never import `application`, `routing`, `domains`, adapters or infra. |
| RS-02 | `application` depends on `core` contracts/ports, not concrete implementations; `cli` is the composition root and may import adapters. |
| RS-03 | Routing and generic evaluation depend on core contracts; adapters implement core ports. Domain-pack validation rules register through interfaces without a Core-to-bank dependency. |
| RS-04 | No business logic in CLI, YAML loader, persistence adapters or provider-specific HTTP clients. |
| RS-05 | Repository remains one installable Python package with `src/` layout; do not introduce service boundaries or plugin-loader frameworks for MVP. |
| RS-06 | `.env.example` documents required variable names; `.env`, local overrides, credentials and generated personal/unsafe traces are ignored by Git. |

## 8. Benchmark and Evaluation Policy (Adopted)

| ID | Policy |
| --- | --- |
| EV-01 | Separate dataset splits for **development/calibration** and **held-out evaluation**. Deduplicate and detect near-duplicates across splits before freezing the golden set. |
| EV-02 | Reference evidence must have source/provenance and effective date; financial numbers or terms without sufficient evidence are marked **unverifiable**, not automatically correct. |
| EV-03 | Deterministic checks assess exact/derived facts and required disclosures; rubric LLM judge assesses language/completeness; human review checks critical cases and samples. Judge scores do not substitute for factual evidence. |
| EV-04 | Quality metrics: factual/financial accuracy, completeness, unsupported claims, and safety/critical-error rate. Report each dimension and evaluation coverage, not one blended number. |
| EV-05 | Quality-first selection: only strategies meeting a **documented, domain-approved** quality/safety gate are eligible for cost/latency comparison. Until thresholds are validated, report descriptive metrics and do not claim certification or “safe for banking.” |
| EV-06 | Fairness requires common cases, consistent small/large model definitions, prompts, temperature and generation limits, routing budgets, and pricing snapshots. Differences and unavailable cases must be explicit. |
| EV-07 | Report all provider call attempts, failures, retries, escalation stages, router overhead, evaluator/judge calls, request counts, model tokens and end-to-end latency separately. |
| EV-08 | Pricing: distinguish **observed/billed**, **estimated from a dated published price sheet**, **simulated**, and **unknown**. Store price version, currency and assumptions. Free-tier consumption is not proof of a zero standardized model cost. |
| EV-09 | End-to-end elapsed time is measured around the whole case execution; provider latency includes network and model delivery where observed. Report P50/P95 with sample sizes and explicit missing-data coverage; avoid performance claims from tiny samples. |
| EV-10 | Store reproducibility provenance: dataset/split hash, config hash, code revision, strategy/threshold versions, model/provider identifiers, generation settings, prompt and rubric versions, evaluator version, pricing snapshot and run timestamp. Repeat live tests when feasible and disclose model non-determinism. |
| EV-11 | Golden data may be published only if licenses, synthetic-data checks and privacy reviews permit it. Expert verification is a verifiable status, not a default claim. |

## 9. Configuration, Privacy, and Execution Safety (Adopted)

| ID | Policy |
| --- | --- |
| SEC-01 | Default `mode: offline`; no live provider, judge or embedding request without explicit `--live` opt-in and a validated key. |
| SEC-02 | Configuration precedence for nonsecret values: **explicit CLI flags > authorized environment overrides > YAML config > typed defaults**. Secrets are read **only from environment**; never interpolate them into reports. |
| SEC-03 | Apply privacy processing before logs, persistence and any third-party call; store sanitized representation and policy/version IDs. Reject requests that cannot satisfy policy. |
| SEC-04 | Do not claim masking ensures irreversible anonymization. Real banking data is outside initial MVP dataset scope; any future personal-data use requires separate risk/legal review. |
| SEC-05 | Disable trace bodies containing raw prompts/responses by default unless explicitly validated as safe; use redacted excerpts or hashes for audit. Avoid leaking PII via exception strings. |
| SEC-06 | Provider/API validation includes strict timeouts, bounded retries, safe error normalization, rate-limit handling, no unsafe automatic fallback and concurrency controls. |
| SEC-07 | Docker and CI never contain baked-in secrets. CI never makes paid/external API calls; live runs are local opt-in and warn about potential charges. |

## 10. Delivery Roadmap and Release Gates (Adopted)

| Milestone | Scope | Exit gate |
| --- | --- | --- |
| **M0** | Data/port/governance design, this document, contract samples | Design baseline recorded; code-level contract review/sign-off still required before parallel implementation |
| **M1** | Offline vertical slice: Always-Large, fake provider, deterministic fixture evaluator, privacy-safe JSONL, CLI/Docker/CI | Offline E2E success + explicit failure cases, no keys/network, tests and lint green |
| **M2** | Rule-Based, kNN, Cascade; held-out dataset and rubrics; SQLite, comprehensive traces/reporting | All four strategies run on equal offline scenarios; held-out isolation and metrics tests pass |
| **M3** | Controlled Groq live experiment and hybrid evaluation | Access checked, explicit user opt-in, budget warnings, observed/estimated costs and complete traces |
| **M4** | Open-source MVP release | Reproducible quickstart, engineering/security/evaluation documentation, supported limitations and passing CI |
| **M5** | NVIDIA Build compatibility (post-MVP) | Adapter contract tests and optional live validation; provider comparisons kept distinct |

**No calendar estimates are committed.** Time and throughput are measured after first implementation slices. The team must not promise expert validation, provider availability, or achieved quality until verified.

## 11. Change Control and Execution Status

- **Adopted**: Product scope, architecture, technical stack, provider order, team roles, M0/M1 design and the specific technical choices in this file.
- **Not yet completed**: Runtime router/provider/evaluator/privacy/reporting/orchestrator implementations, contributor reviews/sign-offs, true golden expert verification, live model access checks, experiments, and MVP release. Core Pydantic/Protocol definitions are implemented.
- **Implementation gate**: Member 01 publishes typed Pydantic/Protocol definitions and minimal example fixtures; affected owners review semantics and tests before parallel work. These are *execution tasks*, not additional open design questions.
- **Amendment rule**: Change an adopted contract only through a documented PR that states the decision ID, trade-off, compatibility impact, affected owners and test migration. Do not silently rewrite the contract.

### Revision history

| Date | Revision |
| --- | --- |
| 2026-10-31 | Implemented M1.1 Pydantic data models and Port protocols, plus JSON Schemas, synthetic examples and offline contract tests. Recorded explicit EvaluatorPort correlation parameters. Lock regeneration and owner reviews pending. |
| 2026-10-31 | Consolidated prior M0 decisions with M1 scope, M1.1 frozen logical contracts, M1.2 repository structure, technical governance, privacy, benchmark semantics and milestone gates; resolved planning questions under delegated decision authority. |

### Implementation note — 2026-10-31

Member 01 implemented the Pydantic v2 schemas and six structural Port protocols under `src/llm_routing_eval/core/`. The `EvaluatorPort` signature now receives keyword-only `run_id` and `trace_id` because `EvaluationResult` requires these identifiers and they cannot be obtained from the original three-argument signature without hidden state. This is a documented, backwards-incompatible correction to the **planning signature** (no existing adapter was implemented). The versioned record schema remains `1.0.0`. See [contracts.md](contracts.md). No contributor review/sign-off is claimed.


## 12. Member 01 Implementation Notes — 2026-10-31

These entries record implementation choices within the approved M1 scope; human module-owner review is still required.

| ID | Implemented decision | Status |
| --- | --- | --- |
| IMP01-01 | `BenchmarkOrchestrator` depends on Core Ports via constructor DI. CLI is the composition root. | Implemented, team review pending |
| IMP01-02 | Offline demo adapters live in `demo/`, separate from Member 02–04 owned production modules. Only fixed synthetic cases are allowlisted. | Implemented |
| IMP01-03 | Privacy gates run before router and provider; blocked cases are not sent to providers or evaluators. Router metadata is stripped. | Implemented; production policy pending |
| IMP01-04 | Unique run/trace/attempt IDs, safe error codes, one trace entry per provider attempt, bounded retries, Cascade re-entry, strict model allowlist and maximum calls. | Implemented and unit tested |
| IMP01-05 | The CLI accepts offline YAML settings, validates demo fixtures and exports schema-safe JSONL atomically. No outbound calls; live flag and live mode are rejected. | Implemented, durable storage pending |
| IMP01-06 | Simulated provider usage is marked SIMULATED; no real billing cost is inferred. Local latency is described as fake-workflow timing. | Implemented |
| IMP01-07 | Evaluation results are included only for successful model calls; unsafe evaluator prose is discarded and invalid metric codes fail evaluation. | Implemented; Member 03 evaluator pending |
| IMP01-08 | Run and data provenance include nonsecret configuration fingerprint and case-identifier hash, but not a full verified dataset content checksum. | Partial; Member 03/04 provenance review pending |

**Verification:** 40 passing Python 3.13 tests, approximately 92% Member 01 application/CLI/demo coverage. Registry access was unavailable; `uv.lock` was not regenerated. Ruff/mypy, Docker and Python 3.12 checks were not run. No formal Member 02–04 sign-off has taken place. These are outstanding release and integration gates, not completed deliverables.
