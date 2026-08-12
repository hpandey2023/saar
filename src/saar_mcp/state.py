"""Small local state database for incremental indexing and verified feedback."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .models import SearchHit, SourceDocument

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS indexed_files (
    source_key TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    source_file TEXT NOT NULL,
    revision TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_ns INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_indexed_files_scope ON indexed_files(scope);
CREATE TABLE IF NOT EXISTS searches (
    query_id TEXT PRIMARY KEY,
    query_hash TEXT NOT NULL,
    scopes_json TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    result_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_searches_query_hash ON searches(query_hash);
CREATE TABLE IF NOT EXISTS search_results (
    query_id TEXT NOT NULL REFERENCES searches(query_id) ON DELETE CASCADE,
    rank INTEGER NOT NULL,
    point_id TEXT NOT NULL,
    citation TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (query_id, rank)
);
CREATE TABLE IF NOT EXISTS feedback (
    query_id TEXT NOT NULL,
    point_id TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK(verdict IN ('relevant', 'irrelevant')),
    human_verified INTEGER NOT NULL CHECK(human_verified = 1),
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    PRIMARY KEY (query_id, point_id),
    FOREIGN KEY (query_id) REFERENCES searches(query_id) ON DELETE CASCADE
);
"""


def normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


def query_hash(query: str) -> str:
    return hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._session() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_metadata(self, key: str) -> str | None:
        with self._lock, self._session() as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def set_metadata(self, key: str, value: str) -> None:
        with self._lock, self._session() as connection:
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def revision_for(self, source_key: str) -> str | None:
        with self._lock, self._session() as connection:
            row = connection.execute(
                "SELECT revision FROM indexed_files WHERE source_key = ?", (source_key,)
            ).fetchone()
        return str(row["revision"]) if row else None

    def record_file(self, document: SourceDocument, chunk_count: int) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._session() as connection:
            connection.execute(
                """
                INSERT INTO indexed_files(
                    source_key, scope, source_file, revision, chunk_count,
                    size_bytes, modified_ns, indexed_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    revision=excluded.revision,
                    chunk_count=excluded.chunk_count,
                    size_bytes=excluded.size_bytes,
                    modified_ns=excluded.modified_ns,
                    indexed_at=excluded.indexed_at
                """,
                (
                    document.source_key,
                    document.scope,
                    document.source_file,
                    document.sha256,
                    chunk_count,
                    document.size_bytes,
                    document.modified_ns,
                    now,
                ),
            )

    def source_keys(self, scope: str | None = None) -> set[str]:
        query = "SELECT source_key FROM indexed_files"
        parameters: tuple[str, ...] = ()
        if scope:
            query += " WHERE scope = ?"
            parameters = (scope,)
        with self._lock, self._session() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return {str(row["source_key"]) for row in rows}

    def remove_file(self, source_key: str) -> None:
        with self._lock, self._session() as connection:
            connection.execute("DELETE FROM indexed_files WHERE source_key = ?", (source_key,))

    def list_sources(self) -> list[dict[str, object]]:
        with self._lock, self._session() as connection:
            rows = connection.execute(
                """
                SELECT scope, COUNT(*) AS files, SUM(chunk_count) AS chunks,
                       SUM(size_bytes) AS bytes, MAX(indexed_at) AS last_indexed_at
                FROM indexed_files GROUP BY scope ORDER BY scope
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def record_search(
        self, query: str, scopes: list[str], duration_ms: float, results: list[SearchHit]
    ) -> str:
        identifier = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self._lock, self._session() as connection:
            connection.execute(
                "INSERT INTO searches VALUES(?, ?, ?, ?, ?, ?)",
                (identifier, query_hash(query), json.dumps(scopes), duration_ms, len(results), now),
            )
            connection.executemany(
                "INSERT INTO search_results VALUES(?, ?, ?, ?, ?)",
                [
                    (identifier, rank, item.point_id, item.citation, item.score)
                    for rank, item in enumerate(results, start=1)
                ],
            )
        return identifier

    def record_feedback(
        self,
        query_id: str,
        point_id: str,
        verdict: str,
        *,
        human_verified: bool,
        note: str = "",
    ) -> None:
        if verdict not in {"relevant", "irrelevant"}:
            raise ValueError("verdict must be 'relevant' or 'irrelevant'")
        if not human_verified:
            raise ValueError("feedback must be explicitly verified by a human")
        now = datetime.now(UTC).isoformat()
        with self._lock, self._session() as connection:
            exists = connection.execute(
                "SELECT 1 FROM search_results WHERE query_id = ? AND point_id = ?",
                (query_id, point_id),
            ).fetchone()
            if not exists:
                raise ValueError("query_id and point_id do not identify a returned search result")
            connection.execute(
                """
                INSERT INTO feedback(query_id, point_id, verdict, human_verified, note, created_at)
                VALUES(?, ?, ?, 1, ?, ?)
                ON CONFLICT(query_id, point_id) DO UPDATE SET
                    verdict=excluded.verdict, note=excluded.note, created_at=excluded.created_at
                """,
                (query_id, point_id, verdict, note[:1000], now),
            )

    def feedback_for_query(self, query: str) -> dict[str, str]:
        with self._lock, self._session() as connection:
            rows = connection.execute(
                """
                SELECT f.point_id, f.verdict, MAX(f.created_at) AS latest
                FROM feedback f
                JOIN searches s ON s.query_id = f.query_id
                WHERE s.query_hash = ? AND f.human_verified = 1
                GROUP BY f.point_id, f.verdict
                ORDER BY latest
                """,
                (query_hash(query),),
            ).fetchall()
        return {str(row["point_id"]): str(row["verdict"]) for row in rows}

    def counts(self) -> dict[str, int]:
        with self._lock, self._session() as connection:
            files = connection.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0]
            chunks = connection.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) FROM indexed_files"
            ).fetchone()[0]
            feedback = connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        return {"files": int(files), "chunks": int(chunks), "verified_feedback": int(feedback)}
