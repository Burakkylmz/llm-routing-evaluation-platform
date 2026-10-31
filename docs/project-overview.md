# LLM Routing Evaluation Platform — Project Overview

## 1. Purpose

An open-source, privacy-conscious platform for benchmarking LLM routing strategies on the **same workload**. It helps teams determine whether a more complex routing strategy provides a meaningful improvement in **response quality, cost, and latency** over simpler baselines.

The project is an **evaluation platform**, not a new routing algorithm framework or a production API gateway.

## 2. MVP Scope

- **Initial domain:** Turkish Retail Banking & Customer Support.
- **Test scenarios:** Loan and interest questions, card and account transactions, banking fees.
- **Dataset:** Synthetic Turkish cases with planned expert verification and a versioned golden evaluation set.
- **Routing strategies:** Always-Large, Rule-Based, kNN, and Cascade.
- **Providers:** Groq (primary); NVIDIA Build (secondary compatibility phase).
- **Model setup:** Configurable small/large model pair; initial Groq candidates are GPT-OSS 20B and GPT-OSS 120B, subject to access verification.
- **Outputs:** Comparable quality, cost, latency, routing explanations, and auditable experiment reports.

## 3. Evaluation Principles

- **Hybrid evaluation:** Deterministic checks + LLM-as-a-Judge + human review for critical or sampled cases.
- **Quality first:** Compare cost and latency only after identifying strategies that meet defined quality and safety requirements.
- **Fair comparisons:** Shared test set, model pool, prompt/configuration controls, and documented execution conditions.
- **Data separation:** Prevent leakage between router configuration/tuning examples and the held-out evaluation set.
- **Reproducibility:** Record dataset versions, prompts, models, provider settings, evaluator versions, and run metadata.
- **Transparent costs:** Distinguish actual API charges, estimated standard-rate costs, and routing/evaluation overhead.

## 4. High-Level Architecture

**Modular Monolith + Hexagonal Architecture (Ports & Adapters)**, with SOLID and OOP principles.

| Module | Responsibility |
| --- | --- |
| Domain Core | Domain contracts, entities, and core rules |
| Application / Orchestration | Benchmark lifecycle and use cases |
| Routing Engine | Four interchangeable routing strategies |
| Provider Adapters | Groq and NVIDIA Build integrations |
| Evaluation Engine | Quality, cost, latency, reliability |
| Privacy | PII masking and safe data handling |
| Reporting & Audit | Explainable, traceable benchmark results |
| Domain Packs | Domain-specific datasets, rubrics, and checks |

**Design patterns:** Strategy for router selection, Adapter for provider integration, Dependency Injection for replaceable components; additional patterns only when justified. The core architecture remains domain-independent.

## 5. Approved Technology Baseline

- **Runtime:** Python 3.12+
- **Environment/dependencies:** uv, `pyproject.toml`, `uv.lock`
- **Configuration:** TOML for project/tool settings; YAML for benchmark/provider settings
- **Containerization:** Docker + Docker Compose
- **Data contracts:** Pydantic v2
- **CLI and HTTP:** Typer, HTTPX
- **Testing and quality:** pytest, pytest-cov, Ruff, mypy
- **Results storage:** JSONL + SQLite
- **CI and logging:** GitHub Actions + structured logging
- **Secrets:** Environment variables (`GROQ_API_KEY`, `NVIDIA_API_KEY`); never commit or log credentials

The MVP is **CLI-first**. A full web dashboard, FastAPI service, Kubernetes, model fine-tuning, multi-sector datasets, and production banking integrations are out of scope.

## 6. Team Ownership (4 Members)

| Member | Primary ownership |
| --- | --- |
| Member 01 | Architecture, core contracts, orchestration, CLI |
| Member 02 | Routing strategies, LLM provider adapters |
| Member 03 | Golden dataset, banking rubrics, hybrid evaluation |
| Member 04 | Infrastructure, privacy, audit trail, reports, CI/CD |

All members own tests and documentation for their modules. Shared interfaces must be reviewed and agreed upon before parallel implementation.

## 7. Delivery Approach

1. Agree on component contracts and benchmark/evaluation data schemas.
2. Develop modules in parallel against those contracts.
3. Integrate the complete benchmark workflow and verify fairness, privacy, and reproducibility.
4. Publish the MVP with documented limitations and reproducible example runs.

**Status:** Product scope, methodology, architecture, technology baseline, provider plan, and role allocation approved. Implementation has not started.
