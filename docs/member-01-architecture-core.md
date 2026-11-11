# Member 01 — Architecture & Core

**Project:** LLM Routing Evaluation Platform\
**Role:** Architecture & Core Owner\
**Status:** Member 01 implementation complete and ready for team review — offline CLI/orchestrator/tests implemented; lockfile and Ruff/mypy/pytest quality gates pass and are enforced in CI. Formal Member 02–04 contract sign-off and real adapter integration remain pending.

## Mission

Build a testable, domain-independent foundation for running **fair and reproducible** comparisons of Always-Large, Rule-Based, kNN, and Cascade routing strategies. Coordinate shared contracts before parallel implementation begins.

## Ownership & Deliverables

| Area | Expected deliverable |
| --- | --- |
| Architecture | Document the modular-monolith boundaries and dependency direction using pragmatic Hexagonal Architecture (Ports & Adapters). |
| Domain contracts | Define versioned Pydantic schemas or typed interfaces for benchmark cases, requests/responses, routing decisions, model usage, evaluation results, and run metadata. |
| Application orchestration | Implement the benchmark lifecycle: load validated inputs → invoke privacy boundary → execute selected strategy/provider → evaluate → publish results through a reporting port. |
| CLI | Provide Typer commands to validate configuration and run a benchmark; ensure clear errors and non-zero exit codes for invalid runs. |
| Dependency wiring | Compose module implementations through explicit Dependency Injection; keep business/application logic independent of concrete providers, storage, and domain packs. |
| Quality | Add core unit tests, deterministic fake adapters/fixtures, type checking, and concise architecture documentation. |

## Required Interface Agreements (Before Parallel Development)

1. **Member 02 — Routing & Integration:** Router input/output contract, provider request/response contract, selected model identifier, routing decision reason, routing overhead, token usage, and provider error semantics.
2. **Member 03 — Dataset & Evaluation:** Dataset case schema, separation of development and held-out evaluation data, rubric version, evaluation input/output schema, and missing/invalid score handling.
3. **Member 04 — Infrastructure & Reporting:** Privacy processing boundary (before any external call or persistent log), run ID, immutable configuration snapshot excluding secrets, trace/event contract, and report persistence interface.

**Contract rule:** Agree on schemas, semantics, and representative example payloads together. Member 01 owns the shared definitions, but does not unilaterally choose other members' evaluation logic or infrastructure implementation.

## Engineering Constraints

- **Python 3.12+**, **uv**, **TOML**, **YAML**, **Docker**; use the approved stack and keep the CLI-first MVP scope.
- Apply **SOLID**, OOP encapsulation, and ports/adapters; use **Strategy**, **Adapter**, and Dependency Injection where they solve real coupling problems. Avoid unnecessary factories and abstraction layers.
- No Turkish-banking-specific rules inside core orchestration or contracts; keep them in interchangeable domain packs.
- Support a common workload and model pool across strategies. Preserve model/provider/configuration metadata for reproducibility.
- Do not leak API keys, customer inputs, or unmasked PII into logs, traces, or reports. Route outbound requests through the agreed privacy boundary.
- Maintain explicit error statuses for failed/time-out calls; never silently score failures as valid model answers.

## Acceptance Criteria

- [x] The documented module boundaries follow the agreed dependency direction; domain-specific code is not imported into Core.
- [ ] Member 02–04 approve the shared contracts and working example payloads before integration.
- [x] A CLI-driven **offline benchmark smoke test** runs end to end using fakes, without requiring API keys or network calls.
- [x] The same input and fixed fake outputs produce equivalent routing and evaluation records across repeated runs (apart from explicitly identified volatile metadata).
- [x] Every run emits a unique run ID, strategy/model identifiers, configuration provenance (without secrets), timestamps, and machine-readable success/failure outcomes through the reporting contract.
- [x] Unit tests cover invalid configuration and adapter failures; Ruff (lint + format) and strict mypy pass and are enforced by the offline CI workflow.
- [x] The architecture and CLI usage are documented with a minimal reproducible example.

## Dependencies & Out of Scope

**Dependencies:** Finalized interfaces from all members; Member 02 supplies router/provider implementations, Member 03 supplies evaluators/domain pack, and Member 04 supplies privacy/reporting/storage implementations.

**Not owned by Member 01:** New routing algorithms, production provider implementations, banking dataset labeling, financial scoring rubrics, PII detection algorithms, dashboard, or production API gateway.

## Suggested Delivery Sequence

1. Review the project overview with the team; agree on contracts and dependency boundaries.
2. Build the core domain/application skeleton and offline test doubles.
3. Implement the CLI and benchmark orchestrator; verify the offline workflow.
4. Integrate the other three members' modules against the agreed interfaces.

See [implementation handoff](member01-implementation.md) for the precise completed scope and outstanding verification.
