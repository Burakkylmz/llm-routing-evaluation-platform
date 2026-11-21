# M5 — NVIDIA Build Compatibility Adapter

**Status:** Provider adapter implemented behind `LLMProviderPort`; offline transport tests added. **Not live-verified or enabled in the CLI.** This deliverable does not authorize a live benchmark or imply model/account access.

## Purpose and scope

This is the secondary provider compatibility milestone described in `docs/development-roadmap.md`, independent of first-provider Groq benchmarking. The adapter lives at `src/llm_routing_eval/adapters/providers/nvidia_build.py` and leaves Core contracts, routing strategies, banking quality rubrics, persistence and CLI unchanged.

Hosted NVIDIA NIM Chat Completions API (documented by NVIDIA):
- Endpoint: `POST https://integrate.api.nvidia.com/v1/chat/completions`
- Authentication: `Authorization: Bearer $NVIDIA_API_KEY`
- Model ID: supply an exact, account-accessible ID selected and reviewed from NVIDIA Build.
- References: https://docs.api.nvidia.com/nim/reference/llm-apis and https://build.nvidia.com/

## Architecture and safeguards

- Implements the existing async `LLMProviderPort.generate(ModelRequest) -> ModelResponse`.
- Interchangeable `NvidiaTransportPort` permits deterministic mocked responses. The default stdlib HTTPS implementation runs off the event loop, blocks redirects and bounds response size. The orchestrator owns retries; the provider issues **one** HTTP attempt per `generate`.
- **Default denied:** `NvidiaBuildConfig(live_enabled=False, privacy_approved=False)` cannot send requests. Even after flags are deliberately enabled, only explicitly approved model IDs and `request.provider == "nvidia_build"` are permitted.
- `NVIDIA_API_KEY` is resolved from the process environment only, never from a config file. Missing keys produce a safe error without network traffic. No key is stored on the adapter or in `ModelResponse`.
- The provider assumes `ModelRequest.prompt` has already passed the real Member 04 privacy boundary and requires a distinct *composition-time* `privacy_approved` flag; this flag is **not** a privacy sanitizer or a substitute for code review. There is no CLI path that sets it today.
- The caller controls model identifiers; `config/providers.yaml` intentionally keeps NVIDIA disabled with no unverified model defaults. Do not silently substitute a Groq model ID.
- Supports common `temperature`, `max_tokens`, and optional `top_p` only. Rejects unexpected `generation_parameters` rather than forwarding unvalidated provider-specific fields. Individual model capabilities remain subject to separate verification.
- Parses the first text completion only when `finish_reason == "stop"`; truncated, filtered or malformed completions fail explicitly and have no success content.
- Maps HTTP 429 to `RATE_LIMITED`, timeouts to `TIMEOUT`, 5xx/network errors to sanitized retryable failures, and 401/403/404/4xx to nonretryable failures. Never includes request prompts, responses from failed HTTP requests, raw exception text or keys in an `ErrorInfo`.
- Reads provider-reported usage when valid; absent/invalid counts remain `None` (**unknown**, not zero). No pricing is inferred from tokens; model pricing and charges require separate evidence.
- Successful response text is **ephemeral**. The receiving evaluator/persistence pipeline must apply the agreed redaction and retention controls before writing anything.

## Reproduce without external calls

```bash
uv sync --locked
uv run --locked pytest -q tests/unit/test_nvidia_build_provider.py
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
```

The test suite uses a fake `NvidiaTransportPort`. No NVIDIA credentials or network are needed, and no real model quality is measured. The existing offline CLI continues to use its demo fake provider and rejects live mode.

## Live activation prerequisites (not completed)

1. Member 01/02/03/04 approve shared contracts, member 04's actual outbound privacy sanitizer and safe logging/persistence paths, and Member 03's quality/dataset split policy (see Issue #7).
2. Verify an actual NVIDIA API key, endpoint/account access, current model ID(s), per-model generation parameter support, quotas and pricing; freeze the model pool for each provider-level experiment. A model being listed in the catalog is **not** evidence of account access.
3. Complete a reviewed CLI/composition flow that validates `config/providers.yaml` and requires explicit user opt-in plus an approved privacy flag. Maintain offline-by-default mode; CI must not invoke live calls.
4. Set a fixed synthetic case set, explicit attempt limit, token limits, rate-limit policy and cost ceiling, including judge calls and cascade retries. Verify rate-limited and partial output behavior.
5. Record model/provider version, dataset/rubric/prompt versions, cost origin, usage coverage, attempt-level traces, latency distributions and critical failures. Compare providers only with the experimental controls and documented limitations.
6. Perform a manually approved small live smoke test; do not merge or label the connector production-ready solely based on offline CI.

## Ownership

Member 02: provider adapter, transport and compatibility tests. Member 04: privacy, secrets, persistence, CI and cost reporting. Member 03: evaluation fairness, scoring and test data. Member 01: composition, run lifecycle and contract sign-off. NVIDIA Build remains **M5**, a follow-on after MVP readiness, not a replacement for M3 Groq validation.
