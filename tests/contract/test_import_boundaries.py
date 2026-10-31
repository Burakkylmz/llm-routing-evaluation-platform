"""Core must stay independent of application and implementation layers."""

import ast
from pathlib import Path


def test_core_imports_only_stdlib_and_pydantic() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "llm_routing_eval" / "core"
    for path in root.glob("*.py"):
        syntax = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(syntax):
            if isinstance(node, ast.Import):
                for item in node.names:
                    assert not item.name.startswith("llm_routing_eval.")
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module is not None
                assert node.module.split(".")[0] in {
                    "__future__", "typing", "datetime", "decimal", "enum", "pydantic",
                }
