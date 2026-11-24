"""Offline-only regression tests for review #12 F4 kNN split leakage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llm_routing_eval.application import BenchmarkInputError
from llm_routing_eval.cli.app import app
from llm_routing_eval.cli.configuration import OfflineSettings
from llm_routing_eval.cli.development_isolation import enforce_development_isolation
from llm_routing_eval.core import BenchmarkCase

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "settings.yaml"
DATASET = ROOT / "examples" / "offline_demo_cases.json"


def _settings(text: str, *, example_id: str = "different-development-id") -> OfflineSettings:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["routing"]["knn"]["examples"][0]["example_id"] = example_id
    raw["routing"]["knn"]["examples"][0]["input_text"] = text
    return OfflineSettings.model_validate(raw)


def _case(text: str, *, case_id: str = "heldout-001") -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        input_text=text,
        domain="turkish_banking",
        scenario_type="account_general",
        metadata={"synthetic": True},
    )


@pytest.mark.parametrize(
    ("development", "evaluation"),
    [
        (
            "  HESAP   açmak için gerekli belgeler, nelerdir ! ",
            "hesap açmak için gerekli belgeler nelerdir?",
        ),
        (
            "İŞLEM yapmak için bilgi alabilir miyim?",
            "işlem yapmak için bilgi alabilir miyim?",
        ),
        (
            "IŞIK altında kart kullanımını anlatır mısınız?",
            "ışık altında kart kullanımını anlatır mısınız?",
        ),
        (
            "I\u0307stanbul bankacılık işlemleri nasıl yapılır?",
            "İstanbul bankacılık işlemleri nasıl yapılır?",
        ),
        (
            "Banka\u00a0kartı ile ödeme yapmak mümkün mü?",
            "Banka kartı ile ödeme yapmak mümkün mü?",
        ),
        (
            "Kredi kartı yıllık aidat koşullarını nerden öğrenebilirim?",
            "Kredi kartı yıllık aidat koşullarını nereden öğrenebilirim?",
        ),
    ],
)
def test_f4_rejects_normalized_and_near_duplicate_content(
    development: str, evaluation: str
) -> None:
    with pytest.raises(BenchmarkInputError, match="KNN_DEVELOPMENT_TEST_LEAKAGE"):
        enforce_development_isolation(_settings(development), [_case(evaluation)])


def test_f4_rejects_casefolded_unicode_equivalent_ids() -> None:
    settings = _settings(
        "Geliştirme için farklı eğitim örneği.",
        example_id=" İŞLEM-001 ",
    )
    with pytest.raises(BenchmarkInputError, match="KNN_DEVELOPMENT_TEST_LEAKAGE"):
        enforce_development_isolation(
            settings,
            [_case("Bu test senaryosunda başka bir soru var.", case_id="işlem-001")],
        )


def test_f4_accepts_genuinely_different_development_examples() -> None:
    settings = _settings("Kredi kartı limit artışı için örnek eğitim açıklaması.")

    enforce_development_isolation(
        settings,
        [_case("Banka hesabına gelen havale nasıl görüntülenir?")],
    )


@pytest.mark.parametrize(
    ("development", "reason"),
    [
        ("! ! ?", "KNN_ISOLATION_EMPTY_NORMALIZED_TEXT"),
        ("a" * 4097, "KNN_ISOLATION_INPUT_TOO_LONG"),
    ],
)
def test_f4_fails_closed_on_uncomparable_text(development: str, reason: str) -> None:
    with pytest.raises(BenchmarkInputError, match=reason):
        enforce_development_isolation(
            _settings(development),
            [_case("Banka hesap özetini nereden indirebilirim?")],
        )


@pytest.mark.parametrize(
    ("configured_strategy", "near_copy"),
    [(False, False), (True, True)],
)
def test_f4_cli_rejects_leak_before_output(
    tmp_path: Path,
    configured_strategy: bool,
    near_copy: bool,
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source = json.loads(DATASET.read_text(encoding="utf-8"))["cases"][0]["input_text"]
    if configured_strategy:
        config["benchmark"]["strategy"] = "knn"
    config["routing"]["knn"]["examples"][0]["input_text"] = (
        source.replace("fark", "farkı")
        if near_copy
        else source.replace("i", "İ").replace("ı", "I").upper().replace(" ", "   ")
    )
    file_path = tmp_path / "config.yaml"
    file_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    output = tmp_path / "must-not-exist.jsonl"
    args = [
        "run",
        "--config",
        str(file_path),
        "--dataset",
        str(DATASET),
        "--output",
        str(output),
    ]
    if not configured_strategy:
        args.extend(("--strategy", "knn"))

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 2
    assert "KNN_DEVELOPMENT_TEST_LEAKAGE" in result.output
    assert source not in result.output
    assert not output.exists()
