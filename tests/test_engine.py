from __future__ import annotations

import pytest
from conftest import FakeEmbedder, FakeLexicalEncoder

from saar_mcp.engine import RetrievalEngine


@pytest.fixture
def engine(settings):
    instance = RetrievalEngine(
        settings,
        embedder=FakeEmbedder(),
        lexical_encoder=FakeLexicalEncoder(),
    )
    yield instance
    instance.close()


def test_incremental_index_and_hybrid_search(engine):
    first = engine.index()
    assert first.changed_files == 3
    assert first.indexed_chunks >= 3
    assert first.errors == []

    second = engine.index()
    assert second.changed_files == 0
    assert second.unchanged_files == 3

    response = engine.search("priority request acknowledged business hours", limit=3)
    assert response.fusion in {"weighted_rrf", "weighted_rrf_client_fallback"}
    assert response.results[0].source_file == "request-handling.md"
    assert "two business hours" in response.results[0].text
    assert all(not item.source_file.startswith("/") for item in response.results)


def test_scope_and_content_filters(engine):
    engine.index()
    response = engine.search("calculate retry delay", content_types=["code"])
    assert response.results
    assert {item.content_type for item in response.results} == {"code"}


def test_frontmatter_metadata_is_searchable_and_filterable(engine):
    engine.index()
    response = engine.search(
        "request handling",
        document_types=["process-guide"],
        dates=["2026-08-12"],
        people=["Alex Example"],
    )
    assert response.results
    hit = response.results[0]
    assert hit.document_type == "process-guide"
    assert hit.date == "2026-08-12"
    assert hit.people == ["Alex Example"]
    assert "service-level" in hit.tags

    absent = engine.search("request handling", people=["Nobody Else"])
    assert absent.results == []
    wrong_date = engine.search("request handling", dates=["2026-08-11"])
    assert wrong_date.results == []


def test_verified_feedback_changes_exact_repeated_query(engine):
    engine.index()
    first = engine.search("access credentials", limit=4)
    assert len(first.results) >= 2
    target = first.results[-1]
    engine.record_feedback(
        first.query_id,
        target.point_id,
        "relevant",
        human_verified=True,
    )
    repeated = engine.search("  ACCESS   credentials ", limit=4)
    assert repeated.results[0].point_id == target.point_id
    assert repeated.results[0].feedback == "relevant"


def test_removed_file_is_pruned(engine, corpus):
    engine.index()
    (corpus / "security.md").unlink()
    report = engine.index()
    assert report.removed_files == 1
    assert "operations/security.md" not in engine.state.source_keys()


def test_status_never_exposes_source_paths(engine, settings):
    engine.index()
    status = engine.status()
    assert status["embedding"]["provider"] == "fake"
    assert str(settings.sources[0].path) not in str(engine.list_sources())
