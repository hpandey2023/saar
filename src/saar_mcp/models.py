"""Domain models used by indexing, retrieval, and MCP serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceDocument:
    scope: str
    source_file: str
    source_key: str
    content_type: str
    title: str
    status: str
    text: str
    sha256: str
    size_bytes: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    point_id: str
    scope: str
    source_file: str
    source_key: str
    content_type: str
    title: str
    section: str
    citation: str
    status: str
    text: str
    revision: str
    chunk_index: int
    char_start: int
    char_end: int

    def payload(self) -> dict[str, Any]:
        """Return the Qdrant payload without exposing an absolute local path."""
        return {
            "kind": "chunk",
            "scope": self.scope,
            "source_file": self.source_file,
            "source_key": self.source_key,
            "content_type": self.content_type,
            "title": self.title,
            "section": self.section,
            "citation": self.citation,
            "status": self.status,
            "text": self.text,
            "revision": self.revision,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass(slots=True)
class SearchHit:
    point_id: str
    score: float
    scope: str
    source_file: str
    content_type: str
    title: str
    section: str
    citation: str
    status: str
    text: str
    retrieval_score: float | None = None
    reranker_score: float | None = None
    feedback: str | None = None

    @classmethod
    def from_payload(cls, point_id: object, score: float, payload: dict[str, Any]) -> SearchHit:
        return cls(
            point_id=str(point_id),
            score=float(score),
            scope=str(payload.get("scope", "")),
            source_file=str(payload.get("source_file", "")),
            content_type=str(payload.get("content_type", "document")),
            title=str(payload.get("title", "")),
            section=str(payload.get("section", "")),
            citation=str(payload.get("citation", "")),
            status=str(payload.get("status", "active")),
            text=str(payload.get("text", "")),
            retrieval_score=float(score),
        )

    def as_dict(self, *, text_limit: int | None = None) -> dict[str, Any]:
        result = asdict(self)
        if text_limit and len(self.text) > text_limit:
            result["text"] = f"{self.text[:text_limit].rstrip()}…"
            result["truncated"] = True
        result = {key: value for key, value in result.items() if value is not None}
        return result


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query_id: str
    query: str
    results: list[SearchHit]
    duration_ms: float
    scopes: list[str]
    provider: str
    model: str
    fusion: str

    def as_dict(self, *, text_limit: int | None = None) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "result_count": len(self.results),
            "duration_ms": round(self.duration_ms, 2),
            "scopes": self.scopes,
            "embedding_provider": self.provider,
            "embedding_model": self.model,
            "fusion": self.fusion,
            "results": [item.as_dict(text_limit=text_limit) for item in self.results],
        }


@dataclass(slots=True)
class IndexReport:
    scanned_files: int = 0
    changed_files: int = 0
    unchanged_files: int = 0
    removed_files: int = 0
    skipped_files: int = 0
    indexed_chunks: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
