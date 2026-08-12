from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar

import pytest

from saar_mcp import cli


class FakeReport:
    errors: ClassVar[list[str]] = []
    changed_files = 1
    removed_files = 0

    def as_dict(self):
        return {"changed_files": 1, "errors": []}


class FakeResponse:
    def as_dict(self, *, text_limit=None):
        return {"result_count": 1, "text_limit": text_limit}


class FakeVectorStore:
    def ensure_collection(self):
        return None


class FakeEngine:
    closed = 0

    def __init__(self, settings):
        self.settings = settings
        self.vector_store = FakeVectorStore()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        type(self).closed += 1

    def status(self):
        return {"qdrant": {"reachable": True}, "embedding": {"provider": "fake"}}

    def index(self, **kwargs):
        return FakeReport()

    def search(self, *args, **kwargs):
        return FakeResponse()

    def rebuild(self):
        return FakeReport()


def _create_config(tmp_path: Path, corpus: Path) -> Path:
    config = tmp_path / "saar.yaml"
    exit_code = cli.main(
        [
            "--config",
            str(config),
            "init",
            "--source",
            str(corpus),
            "--scope",
            "test",
        ]
    )
    assert exit_code == 0
    return config


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("saar ")


def test_model_environment_uses_writable_cache_and_quiet_stdio(tmp_path, monkeypatch):
    cache = tmp_path / "models"
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache))
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)

    cli._configure_model_environment(quiet_progress=True)

    assert os.environ["HF_XET_CACHE"] == str(cache / "xet")
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"


def test_init_and_config_only_doctor(tmp_path, corpus, capsys):
    config = _create_config(tmp_path, corpus)
    assert cli.main(["--config", str(config), "doctor", "--config-only"]) == 0
    output = capsys.readouterr().out
    assert '"configuration": "valid"' in output
    assert cli.main(["--config", str(config), "init", "--source", str(corpus)]) == 2


def test_cli_runtime_commands_with_injected_engine(tmp_path, corpus, monkeypatch, capsys):
    config = _create_config(tmp_path, corpus)
    monkeypatch.setattr(cli, "RetrievalEngine", FakeEngine)
    assert cli.main(["--config", str(config), "doctor"]) == 0
    assert cli.main(["--config", str(config), "index", "--scope", "test", "--force"]) == 0
    assert (
        cli.main(
            [
                "--config",
                str(config),
                "search",
                "question",
                "--content-type",
                "document,code",
                "--limit",
                "2",
            ]
        )
        == 0
    )
    assert cli.main(["--config", str(config), "rebuild"]) == 2
    assert cli.main(["--config", str(config), "rebuild", "--yes"]) == 0
    assert '"result_count": 1' in capsys.readouterr().out


def test_watch_stops_cleanly(tmp_path, corpus, monkeypatch):
    config = _create_config(tmp_path, corpus)
    monkeypatch.setattr(cli, "RetrievalEngine", FakeEngine)

    def interrupt(_interval):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", interrupt)
    assert cli.main(["--config", str(config), "watch", "--interval", "0.01"]) == 0


def test_serve_network_guard_and_loopback(tmp_path, corpus, monkeypatch):
    config = _create_config(tmp_path, corpus)
    calls = []

    class FakeServer:
        def run(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(cli, "build_server", lambda settings: FakeServer())
    assert cli.main(["--config", str(config), "serve"]) == 0
    assert calls[-1]["transport"] == "stdio"
    assert (
        cli.main(
            [
                "--config",
                str(config),
                "serve",
                "--transport",
                "http",
                "--host",
                "0.0.0.0",
            ]
        )
        == 2
    )
    assert (
        cli.main(
            [
                "--config",
                str(config),
                "serve",
                "--transport",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
            ]
        )
        == 0
    )
    assert calls[-1]["port"] == 9000


def test_evaluate_compare_and_provider_catalog(tmp_path, corpus, monkeypatch, capsys):
    config = _create_config(tmp_path, corpus)
    questions = tmp_path / "questions.yaml"
    questions.write_text(
        "queries:\n  - query: test\n    expected_sources: [guide.md]\n", encoding="utf-8"
    )
    expected_report = {
        "summary": {"mrr@8": 1.0},
        "provider": {"provider": "fake"},
        "results": [],
    }
    monkeypatch.setattr(cli, "RetrievalEngine", FakeEngine)
    monkeypatch.setattr(cli, "evaluate", lambda *args, **kwargs: expected_report)
    report_path = tmp_path / "report.json"
    assert (
        cli.main(
            [
                "--config",
                str(config),
                "evaluate",
                str(questions),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    assert json.loads(report_path.read_text()) == expected_report
    assert cli.main(["compare", str(report_path), str(report_path)]) == 0

    monkeypatch.setattr(
        "fastembed.TextEmbedding.list_supported_models",
        lambda: [{"model": "local", "dim": 3, "license": "MIT", "size_in_GB": 0.1}],
    )
    assert cli.main(["providers"]) == 0
    assert '"provider": "ollama"' in capsys.readouterr().out
