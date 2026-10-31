# Architecture

**Status:** Member 01 offline application and CLI vertical slice implemented with isolated test doubles. Members 02–04 real modules and the integrated production-grade benchmark are pending.

```text
Typer CLI (composition root)
  ├─ load/validate YAML and synthetic fixtures
  └─ constructor-inject Port implementations
     └─ BenchmarkOrchestrator (application)
         ├─ PrivacyPort.sanitize(input) -> fail closed
         ├─ RouterPort.decide(sanitized case, routing context)
         ├─ PrivacyPort.sanitize(provider prompt) -> fail closed
         ├─ LLMProviderPort.generate(request) [bounded attempts]
         ├─ EvaluatorPort.evaluate(safe case, response, golden ref)
         └─ ResultRepositoryPort.save_trace/evaluation/report

core/ (contracts and Protocols) <- application/ <- cli/ and outer adapters
```

**Invariant:** No Core or Application import of provider-specific SDKs, business-domain packages, CLI, persistence, or test doubles. The CLI assembles dependencies. Always-Large, Rule-Based, kNN and Cascade are interchangeable Router Port implementations; the application does not choose routing algorithms.

## M1 offline smoke demo

`llm-routing-eval run` uses `demo/` test doubles only: an allowlisted synthetic prompt privacy adapter, a deterministic Always-Large router, a fake model, a simple fixture-match evaluator and an in-memory repository. The CLI exports interim safe JSONL. These are not the Member 02–04 production modules.

`--live` is rejected and fake telemetry is explicitly labeled **SIMULATED**. The local end-to-end timer does not represent real model latency. Run input is never embedded in trace/report payloads. Router input metadata is removed. Golden references are sent only to the evaluator.

## Error model

- Invalid dataset/config: CLI exits nonzero and avoids displaying caller data.
- Privacy denial: `BLOCKED` trace, no provider and no remote judge.
- Router mismatch: normalized `ROUTING_FAILED`, without a provider request.
- Provider failure/timeout: one trace event per attempt, bounded same-model retry if enabled; no implicit fallback.
- Evaluation error: `PARTIAL` trace and `FAILED` evaluation, not a fabricated quality score.
- Persistence error: raises `BenchmarkPersistenceError`; no successful report is returned.

**Limitations:** The initial demo exporter does not provide transactionality or crash recovery. Config fingerprints do not include secrets. Pricing, financial evaluation, expert review, large-scale parallel execution, live retry-after handling and durable audit storage remain future integrations.

See [Member 01 implementation](member01-implementation.md) and [decisions.md](decisions.md).
