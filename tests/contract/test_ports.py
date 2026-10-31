"""Ports are importable structural Protocols; no provider calls are made."""

import inspect
from typing import get_type_hints

from llm_routing_eval.core import (
    EmbeddingPort,
    EvaluatorPort,
    LLMProviderPort,
    PrivacyPort,
    ResultRepositoryPort,
    RouterPort,
)


def test_async_port_boundaries() -> None:
    assert inspect.iscoroutinefunction(RouterPort.decide)
    assert inspect.iscoroutinefunction(LLMProviderPort.generate)
    assert inspect.iscoroutinefunction(EvaluatorPort.evaluate)
    assert inspect.iscoroutinefunction(ResultRepositoryPort.save_trace)
    assert inspect.iscoroutinefunction(ResultRepositoryPort.save_evaluation)
    assert inspect.iscoroutinefunction(ResultRepositoryPort.save_report)
    assert inspect.iscoroutinefunction(EmbeddingPort.embed)
    assert not inspect.iscoroutinefunction(PrivacyPort.sanitize)


def test_evaluator_receives_explicit_correlation_identifiers() -> None:
    assert {"run_id", "trace_id"}.issubset(inspect.signature(EvaluatorPort.evaluate).parameters)
    assert "GoldenReference" in str(get_type_hints(EvaluatorPort.evaluate)["reference"])
