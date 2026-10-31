# Member 02 — Routing & LLM Integration

**Project:** LLM Routing Evaluation Platform\
**Role:** Routing Strategies & Provider Integration Owner\
**Status:** Planned — no implementation started

## Mission

Implement four **interchangeable, explainable routing strategies** and reliable LLM provider adapters so that their **quality, cost, latency, and operational overhead** can be compared on the same workload. Do not develop a new routing algorithm or a production serving gateway.

## Ownership & Deliverables

| Area | Expected deliverable |
| --- | --- |
| Routing interface | Implement the router contract agreed with Member 01, including selected model(s), decision reason, and measurable routing overhead. |
| Always-Large | Send every eligible case directly to the configured large model; use this as the primary quality/cost baseline. |
| Rule-Based | Route using explicit, configurable rules and priority ordering; provide a trace of which rule matched or which default applied. |
| kNN Routing | Select a model based on similarity to **development-only** labeled examples; document embedding/index setup, scoring, and tie-breaking. |
| Cascade | Try the small model first; escalate to the large model when a **predefined, calibrated** escalation condition is met; retain both invocation records. |
| Provider adapters | Integrate **Groq** first and **NVIDIA Build** in the secondary compatibility phase, behind the shared provider interface. |
| Configuration & testing | Supply example YAML configurations, offline fakes, unit tests, integration tests, and usage documentation. |

## Required Interface Agreements (Before Parallel Development)

1. **Member 01 — Architecture & Core:** Router and provider input/output contracts; decision and error enums; dependency injection; execution order; model configuration; cancellation/time-out semantics.
2. **Member 03 — Dataset & Evaluation:** Development vs. held-out test split; authorized kNN examples, rule thresholds, cascade escalation signals, and quality-evaluation boundaries. **No test-set-informed routing decisions.**
3. **Member 04 — Infrastructure & Reporting:** Pre-invocation privacy boundary; safe trace fields; provider request/response redaction; timing/usage instrumentation; run/config version metadata and secret management.

**Contract rule:** Member 02 implements strategy decisions and provider access; Member 03 defines evaluation criteria. Never use an evaluator's held-out labels or reference answers to make routing decisions during a benchmark.

## Engineering Constraints

- Use **Python 3.12+**, **uv**, **TOML/YAML**, **Docker**, typed contracts, **SOLID/OOP**, and pragmatic **Strategy** and **Adapter** patterns. Keep routing logic independent of CLI, storage, and banking-specific test data.
- Keep small/large model IDs configurable. Initial Groq candidates: `openai/gpt-oss-20b` and `openai/gpt-oss-120b`, **subject to verifying availability and account access**.
- Read secrets **only** from environment variables: `GROQ_API_KEY` and `NVIDIA_API_KEY`. Never log, persist, or include credentials in example configuration.
- Run all outbound model/embedding requests **after** the agreed privacy checks; do not bypass the privacy boundary from the kNN implementation.
- For fair comparisons, use the agreed model pool, prompt controls, input cases, and provider settings. Report router overhead separately from model generation latency.
- Capture **all** calls in cascades, including small-model attempts, escalations, retries, failures, and token usage. Never treat multiple calls as one free invocation.
- Distinguish actual provider-reported usage from missing or estimated usage. Handle rate limits, timeouts, and transient failures explicitly using bounded retries; preserve attempt-level traces.
- Provide readable reason codes for routing decisions without exposing sensitive input text. Record kNN configuration and cascade thresholds for reproducibility.

## Acceptance Criteria

- [ ] All four strategies implement the same approved router interface and can be swapped by configuration without changes to the orchestrator.
- [ ] Deterministic offline fixtures demonstrate correct Rule-Based behavior, kNN neighbor/tie handling, and Cascade escalation and non-escalation paths.
- [ ] The held-out golden test set is excluded from kNN references, rule tuning, and cascade threshold calibration.
- [ ] Every routing decision records strategy ID/version, chosen model(s), non-sensitive explanation, and routing overhead where measurable.
- [ ] Groq adapter passes gated live integration tests when the key and model access are available; without keys, the full offline test suite still passes.
- [ ] NVIDIA Build adapter has a documented, separately testable compatibility path; its unverified model availability is not reported as confirmed.
- [ ] Failed calls and retries yield explicit statuses; Cascade traces correctly attribute **every** attempted call, latency, and usage value.
- [ ] No raw API key or unmasked sensitive data appears in logs, reports, recorded fixtures, or provider calls.
- [ ] Module tests, linting, type checking, and a short reproduction guide pass review.

## Dependencies & Out of Scope

**Dependencies:** Member 01 owns shared contracts and orchestration; Member 03 supplies the approved development examples and evaluation requirements; Member 04 supplies privacy guards, telemetry/reporting interfaces, and environment/CI setup.

**Not owned by Member 02:** Banking dataset labeling, judge rubrics, final quality scoring, PII detection logic, report/dashboard presentation, architecture-wide CLI ownership, production API gateway, model fine-tuning, or novel routing research.

## Suggested Delivery Sequence

1. Agree on router/provider contracts, privacy boundary, fair comparison controls, and development/test separation.
2. Implement **Always-Large** and **Rule-Based** with offline provider fakes; add explanations and trace tests.
3. Implement **kNN** and **Cascade** with isolated development data, calibrated conditions, and call-level accounting.
4. Integrate **Groq** and verify controlled live calls; add **NVIDIA Build** compatibility tests in the secondary phase.
5. Validate end-to-end benchmark integration with Members 01, 03, and 04.
