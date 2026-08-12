from __future__ import annotations

from pathlib import Path

import pytest

from saar_mcp.config import Settings, load_settings, read_secret
from saar_mcp.exceptions import ConfigurationError


def test_load_settings_resolves_paths_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CORPUS_PATH", "docs")
    (tmp_path / "docs").mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        """qdrant:
  path: data/qdrant
index:
  state_path: data/state.sqlite3
sources:
  - scope: project
    path: ${CORPUS_PATH}
""",
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.sources[0].path == (tmp_path / "docs").resolve()
    assert settings.qdrant.path == (tmp_path / "data/qdrant").resolve()


def test_unknown_config_key_is_rejected(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("sources: [{scope: x, path: .}]\nembeding: {}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="embeding"):
        load_settings(config)


def test_missing_environment_secret_names_variable_without_value(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("PRIVATE_TEST_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="PRIVATE_TEST_KEY"):
        read_secret("PRIVATE_TEST_KEY", required=True)


def test_contextual_voyage_requires_non_overlapping_chunks(tmp_path: Path):
    with pytest.raises(ValueError, match="chunk_overlap_chars"):
        Settings.model_validate(
            {
                "embedding": {"provider": "voyage", "model": "voyage-context-4"},
                "sources": [{"scope": "x", "path": tmp_path}],
            }
        )
