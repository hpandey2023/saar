from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from saar_mcp.models import SearchHit
from saar_mcp.state import StateStore


def _hit() -> SearchHit:
    return SearchHit(
        point_id="point-1",
        score=0.5,
        scope="project",
        source_file="guide.md",
        content_type="document",
        title="Guide",
        section="Rules",
        citation="project/guide.md#rules",
        status="active",
        text="Relevant text",
    )


def test_search_state_hashes_query_and_requires_verified_feedback(tmp_path):
    state = StateStore(tmp_path / "state.sqlite3")
    secret_query = "Confidential acquisition plan"
    query_id = state.record_search(secret_query, ["project"], 12.5, [_hit()])
    with closing(sqlite3.connect(state.path)) as connection:
        stored = connection.execute("SELECT query_hash FROM searches").fetchone()[0]
    assert stored != secret_query
    assert "acquisition" not in stored

    with pytest.raises(ValueError, match="verified"):
        state.record_feedback(query_id, "point-1", "relevant", human_verified=False)
    state.record_feedback(query_id, "point-1", "relevant", human_verified=True)
    assert state.feedback_for_query(secret_query) == {"point-1": "relevant"}


def test_feedback_must_reference_returned_result(tmp_path):
    state = StateStore(tmp_path / "state.sqlite3")
    query_id = state.record_search("question", ["project"], 1.0, [_hit()])
    with pytest.raises(ValueError, match="returned search result"):
        state.record_feedback(query_id, "other", "irrelevant", human_verified=True)
