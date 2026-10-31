"""Async boundary contracts. The orchestrator owns policy, retries and routing flow."""

from typing import Protocol

from .models import (
    BenchmarkCase,
    BenchmarkReport,
    EvaluationResult,
    ExecutionTrace,
    GoldenReference,
    ModelRequest,
    ModelResponse,
    RoutingContext,
    RoutingDecision,
    SanitizedText,
)


class RouterPort(Protocol):
    async def decide(self, case: BenchmarkCase, context: RoutingContext) -> RoutingDecision: ...


class LLMProviderPort(Protocol):
    async def generate(self, request: ModelRequest) -> ModelResponse: ...


class EvaluatorPort(Protocol):
    async def evaluate(
        self,
        case: BenchmarkCase,
        response: ModelResponse | None,
        reference: GoldenReference | None,
        *,
        run_id: str,
        trace_id: str,
    ) -> EvaluationResult: ...


class PrivacyPort(Protocol):
    def sanitize(self, text: str, *, purpose: str) -> SanitizedText: ...


class ResultRepositoryPort(Protocol):
    async def save_trace(self, trace: ExecutionTrace) -> None: ...

    async def save_evaluation(self, result: EvaluationResult) -> None: ...

    async def save_report(self, report: BenchmarkReport) -> None: ...


class EmbeddingPort(Protocol):
    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...
