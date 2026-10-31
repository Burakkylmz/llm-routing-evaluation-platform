"""Published schema examples remain parseable as versioned records."""

import json
from pathlib import Path

from llm_routing_eval.core import (
    BenchmarkCase,
    BenchmarkReport,
    EvaluationResult,
    ExecutionTrace,
    GoldenReference,
    ModelResponse,
    RoutingDecision,
)


def test_example_records_are_still_compatible() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "contract_examples_v1.json"
    examples = json.loads(fixture_path.read_text(encoding="utf-8"))
    for contract in (
        BenchmarkCase,
        GoldenReference,
        RoutingDecision,
        ModelResponse,
        EvaluationResult,
        ExecutionTrace,
        BenchmarkReport,
    ):
        model = contract.model_validate_json(json.dumps(examples[contract.__name__]))
        assert model.schema_version == "1.0.0"


def test_published_json_schemas_have_seven_records() -> None:
    schema_dir = Path(__file__).resolve().parents[2] / "docs" / "schemas"
    schemas = list(schema_dir.glob("*.schema.json"))
    assert len(schemas) == 7
    for file in schemas:
        schema = json.loads(file.read_text(encoding="utf-8"))
        assert schema["type"] == "object"
