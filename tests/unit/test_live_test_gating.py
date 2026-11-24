"""Offline guard: live provider tests must skip unless explicitly enabled outside CI.

Runs ``pytest tests/live`` in a subprocess with controlled environments and
asserts that nothing runs (all skipped). No network access is attempted: every
skip happens before any provider or HTTP client is constructed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIVE_ENV = (
    "LLM_ROUTING_LIVE_TESTS",
    "CI",
    "BENCHMARK_MODE",
    "GROQ_API_KEY",
    "NVIDIA_API_KEY",
    "NVIDIA_LIVE_TEST_MODEL",
)


def _run_live_suite(extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in LIVE_ENV}
    env.update(extra)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "tests/live"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ({}, "live tests disabled"),
        ({"GROQ_API_KEY": "FAKE", "NVIDIA_API_KEY": "FAKE"}, "live tests disabled"),
        ({"LLM_ROUTING_LIVE_TESTS": "1", "CI": "true", "GROQ_API_KEY": "FAKE"}, "never run in CI"),
        (
            {"LLM_ROUTING_LIVE_TESTS": "1", "BENCHMARK_MODE": "offline", "GROQ_API_KEY": "FAKE"},
            "never run in CI",
        ),
        ({"LLM_ROUTING_LIVE_TESTS": "1"}, "missing live-test environment"),
    ],
)
def test_live_suite_skips_without_explicit_opt_in(extra: dict[str, str], reason: str) -> None:
    result = _run_live_suite(extra)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "2 skipped" in output, output
    assert "passed" not in output and "failed" not in output, output
    assert reason in output, output


def test_live_marker_is_registered() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'markers = ["live:' in pyproject
