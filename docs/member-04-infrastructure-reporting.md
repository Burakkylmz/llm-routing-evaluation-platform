# Member 04 — Infrastructure, Privacy & Reporting

**Project:** LLM Routing Evaluation Platform\
**Role:** Infrastructure, Privacy, Reporting & CI/CD Owner\
**Status:** Planned — no implementation started

## Mission

Provide a **reproducible, privacy-conscious execution environment** and **auditable benchmark outputs** for comparing Always-Large, Rule-Based, kNN, and Cascade routing strategies. Make it possible to reproduce *offline* test runs and interpret live-run results without confusing inference costs, routing overhead, and evaluation costs.

## Ownership & Deliverables

| Area | Expected deliverable |
| --- | --- |
| Development environment | Standardize Python 3.12+, `uv`, `pyproject.toml`, committed `uv.lock`, and contributor setup instructions. |
| Configuration | Define validated `settings.yaml` templates and environment-based overrides; document precedence and reject unsafe or invalid settings. |
| Containerization | Create `Dockerfile`, `compose.yaml`, and `.dockerignore` for an **offline-first**, CLI-driven benchmark runner. |
| Secrets management | Read `GROQ_API_KEY` and `NVIDIA_API_KEY` from environment variables; provide a safe `.env.example` without credentials. Never commit keys or bake them into images. |
| Privacy | Implement a privacy-processing adapter/guard before external model or judge calls and before logging/persistence; add masking tests and explicit failure behavior. |
| Results storage | Implement versioned **JSONL + SQLite** persistence behind the shared reporting interface, with run-level and case-level records. |
| Reporting & audit | Generate a machine-readable run manifest and concise strategy comparison covering quality, critical failures, cost, latency, coverage, explanations, and execution errors. |
| Observability | Implement structured, privacy-safe logs, correlation/run IDs, and recorded decision traces without exposing original sensitive content. |
| CI/CD | Add GitHub Actions for offline tests, linting, type checking, and container build checks; keep live API checks explicitly opt-in. |
| Documentation | Provide setup, reproducible offline-run, live-run safety, results interpretation, and data-retention guidance. |

## Required Interface Agreements (Before Parallel Development)

1. **Member 01 — Architecture & Core:** Agree on privacy port placement, validated configuration schema, run lifecycle, run ID, event/result schemas, storage port, and immutable *redacted* configuration snapshot.
2. **Member 02 — Routing & Integration:** Agree on provider-call traces, model identifiers, token usage, estimated vs. billed cost semantics, router overhead, attempts/retries, and secret-safe provider error reporting.
3. **Member 03 — Dataset & Evaluation:** Agree on rubric/dataset versions, quality dimensions and gates, critical-error definitions, unscorable-case handling, judge/human-review metadata, and the separation of inference vs. evaluation expenses.

**Contract rule:** Member 04 owns the mechanisms for storage, privacy, infrastructure, and presentation—not the semantics of router selection or the financial evaluation rubric. Changes to shared contracts require team review.

## Engineering & Security Constraints

- Follow the approved **Modular Monolith + Hexagonal Architecture**, **SOLID**, and **OOP** principles. Implement storage, privacy, and reporting as replaceable adapters rather than hard-coding them into the benchmark orchestrator.
- Use the approved **Python 3.12+ / uv / TOML / YAML / Docker** stack, with Pydantic validation, pytest, Ruff, and mypy. Do not add a web dashboard, production API gateway, or unnecessary infrastructure services to the MVP.
- **Default to synthetic data and no raw-request persistence.** Remove or mask identifiable content before outbound calls and persisted artifacts; fail closed if required privacy checks cannot run. Document limits: masking is **not** a guarantee of full anonymization.
- Ensure secrets never appear in CLI output, container logs, test snapshots, reports, exception messages, or committed configuration. Use explicit safe allowlists for what traces may store.
- Keep **offline runs as the default** and require explicit opt-in for live calls. Provide visible usage/cost warnings and do not assume access to Groq or NVIDIA Build models is guaranteed.
- Distinguish **observed provider charges (when available)**, **estimated standard-rate model inference costs**, **routing overhead**, and **judge/evaluation costs**. Missing prices or usage must be labeled *unknown*, not zero.
- Report **end-to-end latency**, router overhead, provider latency, and **P50/P95** with clearly defined measurement boundaries, sample counts, and failed/timeout attempt handling.
- Record strategy/model/provider versions, dataset split/version, prompts, evaluator/rubric versions, configuration fingerprint (excluding secrets), pricing snapshot, timestamps, and per-attempt statuses. Never imply live LLM responses are perfectly deterministic merely because metadata is reproducible.
- Preserve failures, retries, and partial traces in aggregate coverage/error metrics; do not silently exclude failed cases from the denominator.

## Acceptance Criteria

- [ ] A new contributor can run the **offline benchmark** using documented `uv` commands or Docker Compose, without API keys or external network calls.
- [ ] The committed `uv.lock`, sample YAML configuration, and container configuration provide a documented, repeatable setup.
- [ ] Live execution requires explicit opt-in and uses environment-provided keys; no credentials exist in committed files, images, logs, trace fixtures, or reports.
- [ ] Privacy tests cover common Turkish banking PII examples, masking failures, and log/trace leakage; unsafe outbound or persistence paths are rejected.
- [ ] JSONL and SQLite adapters store agreed versioned run/case records, including **failed or incomplete** calls, with working offline round-trip tests.
- [ ] A benchmark report compares all four strategies on the same workload, showing **per-dimension quality, critical-failure rate, completion/coverage, cost, latency P50/P95, and routing explanations** where available.
- [ ] Reports explicitly separate model inference, routing overhead, and judge/evaluation costs, and distinguish *estimated*, *observed*, and *unknown* cost values.
- [ ] Every report can be traced to a run ID and a redacted configuration/dataset/evaluator provenance record; audit samples do not contain raw sensitive user content.
- [ ] CI runs offline tests, Ruff, mypy, and a container build check; API-dependent integration tests cannot execute automatically on untrusted pull requests.
- [ ] Setup, retention limitations, benchmark reproducibility, and known privacy limitations are documented and reviewed with the other members.

## Dependencies & Out of Scope

**Dependencies:** Member 01 supplies the orchestrator and shared ports; Member 02 supplies provider/route telemetry and usage metadata; Member 03 supplies quality measurements, evaluation metadata, and rubric definitions.

**Not owned by Member 04:** Routing algorithms, provider API adapter implementations, golden dataset creation or labeling, quality scoring rules, judge prompts, production banking integrations, user authentication, public dashboard, Kubernetes, or deployment of a production gateway.

## Suggested Delivery Sequence

1. Align run/result schemas, privacy boundary, secrets handling, and cost/latency semantics with the team.
2. Create the `uv` / Docker / YAML environment and a fully offline CI path.
3. Implement privacy guards and JSONL/SQLite adapters with leakage and round-trip tests.
4. Implement audit manifests and quality-first comparative reports using deterministic fixtures.
5. Integrate with Members 01–03, validate four-strategy report completeness, and gate optional live runs.
