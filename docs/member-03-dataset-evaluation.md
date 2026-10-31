# Member 03 — Dataset & Evaluation

**Project:** LLM Routing Evaluation Platform\
**Role:** Golden Dataset & Evaluation Engine Owner\
**Status:** Planned — no implementation started

## Mission

Develop a **credible Turkish Retail Banking & Customer Support benchmark** and a **hybrid quality evaluation process** that can identify whether a routing strategy is sufficiently accurate and safe before its cost and latency are compared. The evaluation must be independent of the routing strategy under test.

## Ownership & Deliverables

| Area | Expected deliverable |
| --- | --- |
| Scenario taxonomy | Define Turkish retail banking cases covering **loans/interest**, **credit cards**, **account transactions**, and **banking fees**, including ambiguous, incomplete, and potentially unsafe requests. |
| Dataset preparation | Create structured synthetic cases with stable IDs, reference answers or evidence, expected behavior, severity labels, and explicit source/provenance metadata. |
| Dataset governance | Maintain separate **development/calibration** and **held-out golden evaluation** splits, dataset versions, and review status. |
| Financial rubrics | Document criteria for **financial accuracy**, **completeness**, **safety**, and **unsupported or hallucinated claims** with explicit pass/fail or severity definitions. |
| Deterministic evaluation | Implement programmatic checks for specified facts, required disclosures, numeric calculations/tolerances, and structured outputs where applicable. |
| LLM-as-a-Judge | Implement a versioned, rubric-based evaluator for qualitative dimensions; retain judge model/prompt metadata and explicit uncertainty or abstention handling. |
| Human verification | Define an expert-review workflow, resolve disagreements, track reviewer decisions, and prevent unreviewed synthetic cases from being called *expert-verified*. |
| Quality results | Produce per-case and aggregated quality outcomes, critical-failure rates, coverage, and evidence supporting quality-first eligibility decisions. |
| Testing and documentation | Deliver offline fixtures, evaluator unit tests, calibration examples, and a short methodology/limitations guide. |

## Required Interface Agreements (Before Parallel Development)

1. **Member 01 — Architecture & Core:** Define benchmark case schema, reference/evidence representation, evaluator input/output contract, rubric and dataset version identifiers, missing-result semantics, and quality-gate integration.
2. **Member 02 — Routing & Integration:** Share only **development-approved** cases for kNN references, rules, and Cascade calibration. Agree on which generation traces the evaluator receives, while keeping held-out labels and reference answers inaccessible to routers.
3. **Member 04 — Infrastructure & Reporting:** Agree on privacy-safe dataset handling, evidence and judge-trace retention, versioned result fields, aggregate metric definitions, and how expert review outcomes are recorded without exposing protected data.

**Contract rule:** Member 03 owns evaluation definitions and dataset labels; Member 02 owns routing decisions. **The held-out golden set must never influence routing rules, kNN examples, Cascade thresholds, or evaluator calibration.**

## Evaluation Rules & Engineering Constraints

- Follow the approved **Python 3.12+ / uv / TOML / YAML / Docker** baseline and shared **SOLID/OOP**, typed ports/adapters, and domain-pack boundaries. Banking-specific rubrics belong in the domain pack, **not** Domain Core.
- Use the approved **Hybrid Evaluation** approach: deterministic checks + LLM-as-a-Judge + human review. Do not treat an LLM judge as independent ground truth.
- Separate **per-case dimension scores**, **critical safety failures**, and **unscorable/failed invocations**. Never silently convert missing responses into valid low-quality answers or exclude failures from coverage statistics.
- Make quality-first gates **explicit and versioned**. Do not collapse cost, latency, financial accuracy, and safety into a single opaque score.
- For time-sensitive financial facts (rates, fees, regulations), attach an **as-of date**, source/evidence, and validity assumptions; prefer scenario-provided facts over claims about current bank policies. Do not invent live banking data.
- Calibrate rubrics and judge prompts on development cases. Use documented checks for judge consistency, disagreement, and potential preference/position bias; record judge inference as a **separate evaluation cost**, not router inference cost.
- Support expert sign-off for critical cases and a documented human-reviewed sample of other cases. Track unresolved disagreements and mark insufficiently verified cases clearly.
- Design for **synthetic data first**; follow Member 04's privacy boundary before external judge calls or persistence.
- Record dataset, case, rubric, reference, evaluator, judge-model/prompt, and threshold versions to support reproducibility.

## Acceptance Criteria

- [ ] A versioned synthetic Turkish retail banking dataset covers all agreed scenario groups and includes documented labels, references/evidence, and review status.
- [ ] Development/calibration and held-out evaluation splits are explicit, with tests or checks that detect split leakage.
- [ ] Critical financial cases have documented expert review status; no case is represented as expert-verified without evidence of review.
- [ ] Financial correctness, completeness, safety, and hallucination rubrics are precise enough to reproduce example grading decisions.
- [ ] Deterministic checks pass unit tests for correct, incorrect, borderline, and not-applicable cases (including defined numeric tolerance where needed).
- [ ] The LLM judge adapter produces rubric-aligned structured decisions and records judge configuration, uncertainty, and failures; offline tests require no API key.
- [ ] Quality reporting separates dimension-level metrics, critical errors, coverage, abstentions, and unscorable responses; defined quality gates are auditable.
- [ ] Evaluation results never become inputs to the router during the same held-out benchmark, and tuning does not use held-out references.
- [ ] No fabricated live rates/policies, secrets, or unmasked sensitive data appear in cases, judge payloads, logs, or evaluation artifacts.
- [ ] Unit tests, linting, type checks, and a concise evaluation-methodology guide pass review.

## Dependencies & Out of Scope

**Dependencies:** Member 01 defines shared schemas and execution contracts; Member 02 produces model responses and traces without access to held-out gold labels; Member 04 provides privacy, storage, and reporting mechanisms.

**Not owned by Member 03:** Routing algorithms, provider transport/adapters, CLI orchestration, infrastructure deployment, PII masking implementation, final report presentation, production banking integrations, model training, or multi-industry datasets.

## Suggested Delivery Sequence

1. Agree on dataset/evaluation contracts, leakage boundaries, and quality-gate semantics with the team.
2. Define scenario taxonomy, draft synthetic development fixtures, and finalize versioned scoring rubrics.
3. Implement deterministic evaluator and offline tests; calibrate judge criteria on development cases.
4. Prepare held-out golden cases and obtain/record expert verification; freeze the evaluation version.
5. Integrate judge and human-review outputs, then validate quality-first results with the full benchmark workflow.
