"""Ports-and-adapters dependency direction, enforced statically.

Current, intended graph (``cli`` is the only composition root):

    routing              -> core                (no network / process libraries)
    adapters/providers   -> core
    adapters/embeddings  -> core
    application          -> core                (no network libraries)
    cli                  -> anything            (composition root, not checked)

Imports are resolved properly (absolute, relative, ``from llm_routing_eval import x``),
and direct dynamic imports (``importlib.import_module``, ``__import__``) are
rejected in guarded layers. This is a regression guard, not a sandbox: it cannot
prove the absence of every dynamic or reflective import.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "src" / "llm_routing_eval"
PACKAGE = "llm_routing_eval"

NETWORK_MODULES = frozenset(
    {"httpx", "requests", "urllib", "urllib3", "aiohttp", "socket", "http", "ssl", "groq", "openai"}
)
PROCESS_MODULES = frozenset({"subprocess", "multiprocessing"})
# No guarded layer needs dynamic imports; forbidding the module also catches aliases.
DYNAMIC_IMPORT_MODULES = frozenset({"importlib"})

# layer path (relative to the package) -> (allowed project layers, forbidden third-party roots)
LAYERS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "routing": (
        frozenset({"core", "routing"}),
        NETWORK_MODULES | PROCESS_MODULES | DYNAMIC_IMPORT_MODULES,
    ),
    "adapters/providers": (
        frozenset({"core", "adapters/providers"}),
        PROCESS_MODULES | DYNAMIC_IMPORT_MODULES,
    ),
    "adapters/embeddings": (
        frozenset({"core", "adapters/embeddings"}),
        PROCESS_MODULES | DYNAMIC_IMPORT_MODULES,
    ),
    "application": (
        frozenset({"core", "application"}),
        NETWORK_MODULES | PROCESS_MODULES | DYNAMIC_IMPORT_MODULES,
    ),
}


def _layer_of(module: str) -> str:
    """Map a project module name to its layer key, e.g. adapters/providers."""
    parts = module.split(".")[1:]
    if not parts:
        return "<root>"
    if parts[0] == "adapters" and len(parts) > 1:
        return "adapters/" + parts[1]
    return parts[0]


def _imports(tree: ast.AST, package: str) -> list[str]:
    """Fully qualified imported module names, including root-namespace member imports."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            else:
                base = node.module or ""
            if base == PACKAGE or base == "llm_routing_eval.adapters":
                # `from llm_routing_eval import routing` imports the routing layer.
                found.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
                if any(alias.name == "*" for alias in node.names):
                    found.append(base + ".*")
            elif base:
                found.append(base)
    return found


def _has_dynamic_import(tree: ast.AST) -> bool:
    """Reject direct and common aliased dynamic-import calls without executing code."""
    nodes = tuple(ast.walk(tree))
    builtins_names = {"builtins"}
    import_functions = {"__import__", "import_module"}
    for node in nodes:
        if isinstance(node, ast.Import):
            builtins_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "builtins"
            )
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "builtins":
            import_functions.update(
                alias.asname or alias.name for alias in node.names if alias.name == "__import__"
            )

    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in import_functions:
            return True
        if isinstance(func, ast.Attribute):
            if func.attr == "import_module":
                return True
            if (
                func.attr == "__import__"
                and isinstance(func.value, ast.Name)
                and func.value.id in builtins_names
            ):
                return True
    return False


def violations(source: str, *, layer: str, package: str) -> list[str]:
    allowed, forbidden = LAYERS[layer]
    tree = ast.parse(source)
    problems: list[str] = []
    if _has_dynamic_import(tree):
        problems.append("DYNAMIC_IMPORT")
    for module in _imports(tree, package):
        if module == PACKAGE or module.startswith(PACKAGE + "."):
            target = _layer_of(module)
            if module.endswith(".*") or target not in allowed:
                problems.append(f"FORBIDDEN_PROJECT_IMPORT:{module}")
        elif module.split(".")[0] in forbidden:
            problems.append(f"FORBIDDEN_LIBRARY_IMPORT:{module}")
    return problems


def _module_package(path: Path) -> str:
    rel = path.relative_to(PACKAGE_DIR).with_suffix("")
    parts = (PACKAGE, *rel.parts)
    return ".".join(parts[:-1])


@pytest.mark.parametrize("layer", sorted(LAYERS))
def test_layer_respects_dependency_direction(layer: str) -> None:
    directory = PACKAGE_DIR / layer
    assert directory.is_dir(), f"missing layer {layer}"
    problems = {
        str(path.relative_to(ROOT)): found
        for path in sorted(directory.rglob("*.py"))
        if (
            found := violations(
                path.read_text(encoding="utf-8"), layer=layer, package=_module_package(path)
            )
        )
    }
    assert not problems, problems


@pytest.mark.parametrize(
    ("layer", "package", "source"),
    [
        ("routing", "llm_routing_eval.routing", "import llm_routing_eval.adapters.providers"),
        ("routing", "llm_routing_eval.routing", "from llm_routing_eval import adapters"),
        ("routing", "llm_routing_eval.routing", "from llm_routing_eval import cli"),
        ("routing", "llm_routing_eval.routing", "from ..application import orchestrator"),
        ("routing", "llm_routing_eval.routing", "from ..adapters.providers import groq"),
        ("routing", "llm_routing_eval.routing", "import httpx"),
        ("routing", "llm_routing_eval.routing", "from urllib.request import urlopen"),
        ("routing", "llm_routing_eval.routing", "import subprocess"),
        ("routing", "llm_routing_eval.routing", "from llm_routing_eval import *"),
        ("adapters/providers", "llm_routing_eval.adapters.providers", "from ... import routing"),
        ("adapters/providers", "llm_routing_eval.adapters.providers", "from ..embeddings import x"),
        (
            "adapters/providers",
            "llm_routing_eval.adapters.providers",
            "import llm_routing_eval.cli",
        ),
        (
            "adapters/embeddings",
            "llm_routing_eval.adapters.embeddings",
            "from ..providers import groq",
        ),
        ("application", "llm_routing_eval.application", "from llm_routing_eval import routing"),
        ("application", "llm_routing_eval.application", "from ..adapters.providers import groq"),
        ("application", "llm_routing_eval.application", "from ..cli import app"),
        ("application", "llm_routing_eval.application", "import httpx"),
        ("routing", "llm_routing_eval.routing", "import importlib\nimportlib.import_module(n)"),
        (
            "application",
            "llm_routing_eval.application",
            "from importlib import import_module as f\nf(n)",
        ),
        ("adapters/providers", "llm_routing_eval.adapters.providers", "__import__(name)"),
        (
            "routing",
            "llm_routing_eval.routing",
            'import builtins\nbuiltins.__import__("llm_routing_eval.adapters.providers.groq")',
        ),
        (
            "application",
            "llm_routing_eval.application",
            'import builtins as alias\nalias.__import__("llm_routing_eval.adapters.providers.groq")',
        ),
        (
            "routing",
            "llm_routing_eval.routing",
            "from builtins import __import__ as load\nload(name)",
        ),
    ],
)
def test_known_bypasses_are_detected(layer: str, package: str, source: str) -> None:
    assert violations(source, layer=layer, package=package)


@pytest.mark.parametrize(
    ("layer", "package", "source"),
    [
        ("routing", "llm_routing_eval.routing", "from .strategies import ModelPair"),
        ("routing", "llm_routing_eval.routing", "from llm_routing_eval.core import RouterPort"),
        ("routing", "llm_routing_eval.routing", "from llm_routing_eval import core"),
        (
            "adapters/providers",
            "llm_routing_eval.adapters.providers",
            "from .groq import GroqProvider",
        ),
        ("adapters/providers", "llm_routing_eval.adapters.providers", "import httpx"),
        ("application", "llm_routing_eval.application", "from .orchestrator import RunOptions"),
        ("application", "llm_routing_eval.application", "import pydantic"),
        ("routing", "llm_routing_eval.routing", "import builtins\nbuiltins.len([1, 2])"),
        (
            "application",
            "llm_routing_eval.application",
            "from builtins import len as safe_len\nsafe_len([1, 2])",
        ),
    ],
)
def test_allowed_imports_pass(layer: str, package: str, source: str) -> None:
    assert violations(source, layer=layer, package=package) == []
