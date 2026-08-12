from __future__ import annotations

import pytest
from conftest import FakeEmbedder, FakeLexicalEncoder

from saar_mcp.config import QdrantSettings
from saar_mcp.engine import RetrievalEngine
from saar_mcp.exceptions import StorageError
from saar_mcp.vector_store import VectorStore


def test_client_side_rrf_fallback(settings, monkeypatch):
    engine = RetrievalEngine(
        settings,
        embedder=FakeEmbedder(),
        lexical_encoder=FakeLexicalEncoder(),
    )
    try:
        engine.index()
        original = engine.vector_store.client.query_points

        def query_points(*args, **kwargs):
            if "prefetch" in kwargs:
                raise RuntimeError("native weighted RRF unavailable")
            return original(*args, **kwargs)

        monkeypatch.setattr(engine.vector_store.client, "query_points", query_points)
        response = engine.search("high priority request response")
        assert response.fusion == "weighted_rrf_client_fallback"
        assert response.results[0].source_file == "request-handling.md"
    finally:
        engine.close()


def test_dense_only_search(settings):
    settings.lexical.enabled = False
    engine = RetrievalEngine(settings, embedder=FakeEmbedder())
    try:
        engine.index()
        response = engine.search("hardware authentication")
        assert response.fusion == "dense"
        assert response.results
    finally:
        engine.close()


def test_collection_fingerprint_prevents_silent_model_mix(tmp_path):
    qdrant = QdrantSettings(path=tmp_path / "qdrant", collection="fingerprint-test")
    first = VectorStore(
        qdrant,
        embedding_fingerprint="model-a",
        dimension=8,
        lexical_fingerprint=None,
    )
    first.ensure_collection()
    first.close()

    second = VectorStore(
        qdrant,
        embedding_fingerprint="model-b",
        dimension=8,
        lexical_fingerprint=None,
    )
    try:
        with pytest.raises(StorageError, match="incompatible"):
            second.ensure_collection()
        assert second.diagnostics()["collection_exists"] is True
        second.recreate()
        assert second.diagnostics()["status"] == "green"
    finally:
        second.close()


def test_chunking_change_reindexes_same_content(settings):
    first = RetrievalEngine(
        settings,
        embedder=FakeEmbedder(),
        lexical_encoder=FakeLexicalEncoder(),
    )
    try:
        assert first.index().changed_files == 3
    finally:
        first.close()

    settings.index.max_chunk_chars = 420
    second = RetrievalEngine(
        settings,
        embedder=FakeEmbedder(),
        lexical_encoder=FakeLexicalEncoder(),
    )
    try:
        report = second.index()
        assert report.changed_files == 3
        assert report.unchanged_files == 0
    finally:
        second.close()
