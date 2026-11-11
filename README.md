# LLM Routing Evaluation Platform - V1.1

**Status:** Member 01 core + application + CLI **offline demo implemented**. Real routing, provider, privacy, dataset evaluation and durable reporting adapters are not yet implemented.

An open-source, domain-independent framework to evaluate whether LLM routing strategies improve cost, answer quality and latency on controlled workloads. The first domain pack targets **Turkish Retail Banking & Customer Support**.

## Approved MVP architecture

- Python 3.12+, uv, TOML, YAML, Docker
- Modular monolith, SOLID/OOP and pragmatic Ports & Adapters
- Pydantic v2 contracts, Typer CLI, async dependency injection through Ports
- Strategies: Always-Large, Rule-Based, kNN, Cascade (not yet implemented by Member 02)
- Primary live provider: Groq (later, opt-in); NVIDIA Build (post-MVP)

## Member 01 offline smoke test

The CLI is deliberately limited to two allowlisted synthetic examples and deterministic test doubles. **It does not judge financial correctness and never contacts a real API.**

```bash
# In the repository root, with PyPI access and Python 3.12+:
uv sync --locked
uv run llm-routing-eval validate-config --config config/settings.yaml
uv run llm-routing-eval run --config config/settings.yaml \
  --dataset examples/offline_demo_cases.json \
  --output artifacts/offline_demo.jsonl
uv run pytest -q
```

The generated JSONL contains a redacted execution trace, fixture-only evaluation data and a comparison report. Cost is `null` (unknown), model telemetry is simulated and p95 is withheld for very small samples.

`--live` and `BENCHMARK_MODE=live` **fail closed** because live support is not implemented. Do not use customer data in this demo.

### Current verification constraints

Member 01 quality gates (Python 3.12, `uv sync --locked`): `uv lock --check`, `ruff check .`, `ruff format --check .`, `mypy` (strict, `src/`) and 40 `pytest` tests pass with approximately 94% statement coverage. The GitHub Actions workflow runs the same offline gates plus a CLI smoke test. No live API calls were made. Docker remains a scaffold and is unverified (Member 04).

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q --cov=llm_routing_eval
```

## Documentation

- [Architecture](docs/architecture.md)
- [Decision log](docs/decisions.md)
- [Member 01 implementation and handoff](docs/member01-implementation.md)
- [Core contract specification](docs/contracts.md)
- [Development roadmap](docs/development-roadmap.md)
- [Contributor guidance](docs/contributing.md)

## Ownership

| Owner     | Area                                                                 |
| --------- | -------------------------------------------------------------------- |
| Member 01 | `core/`, `application/`, `cli/`; `demo/` temporary integration fakes |
| Member 02 | `routing/`, `adapters/providers/`, `adapters/embeddings/`            |
| Member 03 | `evaluation/`, `domains/turkish_banking/`, labeled datasets          |
| Member 04 | `privacy/`, `reporting/`, `adapters/persistence/`, Docker/CI         |

Interfaces require review by all affected owners before integration.
