"""Incremental indexing and feedback-aware hybrid retrieval engine."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, TypeVar

from .chunking import chunk_document
from .config import Settings
from .exceptions import ConfigurationError, IndexingError
from .models import DocumentChunk, IndexReport, SearchHit, SearchResponse
from .providers.base import Embedder, LexicalEncoder, Reranker
from .providers.factory import build_embedder, build_lexical_encoder, build_reranker
from .scanner import scan_source
from .state import StateStore
from .vector_store import VectorStore

T = TypeVar("T")
INDEX_RECIPE_VERSION = 2


def _batched(items: list[T], size: int) -> Iterable[list[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class RetrievalEngine:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        lexical_encoder: LexicalEncoder | None = None,
        reranker: Reranker | None = None,
        state: StateStore | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.settings = settings
        self.embedder = embedder or build_embedder(settings.embedding)
        self.lexical_encoder = (
            lexical_encoder
            if lexical_encoder is not None
            else build_lexical_encoder(settings.lexical)
        )
        self.reranker = reranker or build_reranker(settings.reranker)
        self.state = state or StateStore(settings.index.state_path)
        self.vector_store = vector_store or VectorStore(
            settings.qdrant,
            embedding_fingerprint=self.embedder.fingerprint,
            dimension=self.embedder.dimension,
            lexical_fingerprint=(
                self.lexical_encoder.fingerprint if self.lexical_encoder else None
            ),
        )
        self._index_recipe = self._build_index_recipe()

    def close(self) -> None:
        close = getattr(self.vector_store, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> RetrievalEngine:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def index(self, *, scopes: list[str] | None = None, force: bool = False) -> IndexReport:
        started = time.perf_counter()
        report = IndexReport()
        self.vector_store.ensure_collection()
        selected = self._validated_scopes(scopes)
        for source in self.settings.sources:
            if source.scope not in selected:
                continue
            previous_keys = self.state.source_keys(source.scope)
            seen: set[str] = set()
            try:
                documents = scan_source(source, self.settings.index)
                for scanned_document in documents:
                    document = replace(
                        scanned_document,
                        sha256=self._effective_revision(scanned_document.sha256),
                    )
                    report.scanned_files += 1
                    seen.add(document.source_key)
                    if (
                        not force
                        and self.state.revision_for(document.source_key) == document.sha256
                    ):
                        report.unchanged_files += 1
                        continue
                    try:
                        chunks = chunk_document(document, self.settings.index)
                        if not chunks:
                            self.vector_store.delete_source(document.source_key)
                            self.state.record_file(document, 0)
                            report.skipped_files += 1
                            continue
                        self._index_chunks(chunks)
                        self.vector_store.delete_old_revisions(document.source_key, document.sha256)
                        self.state.record_file(document, len(chunks))
                        report.changed_files += 1
                        report.indexed_chunks += len(chunks)
                    except Exception as exc:
                        report.errors.append(f"{document.source_key}: {exc}")
            except ConfigurationError as exc:
                report.errors.append(f"{source.scope}: {exc}")

            for removed in sorted(previous_keys - seen):
                try:
                    self.vector_store.delete_source(removed)
                    self.state.remove_file(removed)
                    report.removed_files += 1
                except Exception as exc:
                    report.errors.append(f"{removed}: {exc}")
        report.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        return report

    def _build_index_recipe(self) -> str:
        qdrant_location = (
            str(self.settings.qdrant.path)
            if self.settings.qdrant.mode == "embedded"
            else self.settings.qdrant.url
        )
        value = {
            "version": INDEX_RECIPE_VERSION,
            "embedding": self.embedder.fingerprint,
            "lexical": self.lexical_encoder.fingerprint if self.lexical_encoder else None,
            "collection": self.settings.qdrant.collection,
            "qdrant_mode": self.settings.qdrant.mode,
            "qdrant_location": qdrant_location,
            "max_chunk_chars": self.settings.index.max_chunk_chars,
            "chunk_overlap_chars": self.settings.index.chunk_overlap_chars,
            "min_chunk_chars": self.settings.index.min_chunk_chars,
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _effective_revision(self, content_revision: str) -> str:
        return hashlib.sha256(f"{content_revision}\x1f{self._index_recipe}".encode()).hexdigest()

    def _index_chunks(self, chunks: list[DocumentChunk]) -> None:
        batch_size = self.settings.index.batch_size
        if getattr(self.embedder, "contextual_documents", False):
            batch_size = min(batch_size, self.settings.embedding.contextual_group_size)
        for batch in _batched(chunks, batch_size):
            texts = [item.embedding_text() for item in batch]
            dense = self.embedder.embed_documents(texts)
            if len(dense) != len(batch) or any(
                len(vector) != self.embedder.dimension for vector in dense
            ):
                raise IndexingError("embedding provider returned an unexpected vector shape")
            sparse = self.lexical_encoder.embed_documents(texts) if self.lexical_encoder else None
            self.vector_store.upsert_chunks(batch, dense, sparse)

    def search(
        self,
        query: str,
        *,
        scopes: list[str] | None = None,
        content_types: list[str] | None = None,
        document_types: list[str] | None = None,
        dates: list[str] | None = None,
        people: list[str] | None = None,
        limit: int | None = None,
    ) -> SearchResponse:
        normalized = " ".join(query.split())
        if not normalized:
            raise ValueError("query cannot be empty")
        chosen_limit = limit or self.settings.retrieval.default_limit
        if chosen_limit < 1 or chosen_limit > self.settings.retrieval.max_limit:
            raise ValueError(f"limit must be between 1 and {self.settings.retrieval.max_limit}")
        selected = self._validated_scopes(scopes)
        valid_types = {"document", "code"}
        if content_types and not set(content_types).issubset(valid_types):
            raise ValueError("content_types may contain only 'document' and 'code'")
        document_types = [item.strip().casefold() for item in document_types or [] if item.strip()]
        dates = [item.strip() for item in dates or [] if item.strip()]
        people = [item.strip().casefold() for item in people or [] if item.strip()]

        started = time.perf_counter()
        self.vector_store.ensure_collection()
        dense_query = self.embedder.embed_query(normalized)
        if len(dense_query) != self.embedder.dimension:
            raise IndexingError("embedding provider returned an unexpected query vector shape")
        sparse_query = (
            self.lexical_encoder.embed_query(normalized) if self.lexical_encoder else None
        )
        hits, fusion = self.vector_store.search(
            dense_query,
            sparse_query,
            scopes=selected,
            content_types=content_types,
            document_types=document_types or None,
            dates=dates or None,
            people=people or None,
            retrieval=self.settings.retrieval,
        )
        hits = self._rerank(normalized, hits)
        hits = self._apply_feedback(normalized, hits)
        hits = self._diversify(hits, chosen_limit)
        duration_ms = (time.perf_counter() - started) * 1000
        query_id = self.state.record_search(normalized, selected, duration_ms, hits)
        return SearchResponse(
            query_id=query_id,
            query=normalized,
            results=hits,
            duration_ms=duration_ms,
            scopes=selected,
            provider=self.embedder.info.provider,
            model=self.embedder.info.model,
            fusion=fusion,
        )

    def _rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if self.reranker.info.provider == "none" or not hits:
            return hits
        candidates = hits[: self.settings.reranker.top_k]
        scores = self.reranker.score(query, [item.text for item in candidates])
        if len(scores) != len(candidates) or any(not math.isfinite(score) for score in scores):
            raise IndexingError("reranker returned invalid scores")
        for hit, score in zip(candidates, scores, strict=True):
            hit.reranker_score = score
            hit.score = score
        threshold = self.settings.reranker.min_score
        if threshold is not None:
            candidates = [item for item in candidates if item.score >= threshold]
        candidates.sort(key=lambda item: item.score, reverse=True)
        return [*candidates, *hits[len(scores) :]]

    def _apply_feedback(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not self.settings.feedback.enabled:
            return hits
        feedback = self.state.feedback_for_query(query)
        retained: list[SearchHit] = []
        for item in hits:
            verdict = feedback.get(item.point_id)
            if verdict == "irrelevant":
                continue
            if verdict == "relevant":
                item.feedback = verdict
            retained.append(item)
        retained.sort(key=lambda item: (item.feedback == "relevant", item.score), reverse=True)
        return retained

    def _diversify(self, hits: list[SearchHit], limit: int) -> list[SearchHit]:
        counts: Counter[str] = Counter()
        selected: list[SearchHit] = []
        for item in hits:
            key = f"{item.scope}/{item.source_file}"
            if counts[key] >= self.settings.retrieval.max_per_source:
                continue
            counts[key] += 1
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def record_feedback(
        self,
        query_id: str,
        point_id: str,
        verdict: str,
        *,
        human_verified: bool,
        note: str = "",
    ) -> dict[str, Any]:
        if not self.settings.feedback.enabled:
            raise ConfigurationError("feedback is disabled")
        if self.settings.feedback.require_verified and not human_verified:
            raise ValueError("human_verified must be true")
        self.state.record_feedback(
            query_id,
            point_id,
            verdict,
            human_verified=human_verified,
            note=note,
        )
        return {
            "recorded": True,
            "query_id": query_id,
            "point_id": point_id,
            "verdict": verdict,
            "human_verified": True,
        }

    def list_sources(self) -> dict[str, Any]:
        configured = [
            {"scope": item.scope, "path_available": item.path.is_dir()}
            for item in self.settings.sources
        ]
        return {"configured": configured, "indexed": self.state.list_sources()}

    def status(self) -> dict[str, Any]:
        return {
            "embedding": {
                "provider": self.embedder.info.provider,
                "model": self.embedder.info.model,
                "license": self.embedder.info.license,
                "local": self.embedder.info.local,
                "dimension": self.embedder.dimension,
            },
            "reranker": {
                "provider": self.reranker.info.provider,
                "model": self.reranker.info.model,
                "license": self.reranker.info.license,
                "local": self.reranker.info.local,
            },
            "lexical_enabled": self.lexical_encoder is not None,
            "state": self.state.counts(),
            "qdrant": self.vector_store.diagnostics(),
        }

    def rebuild(self) -> IndexReport:
        self.vector_store.recreate()
        for source_key in self.state.source_keys():
            self.state.remove_file(source_key)
        return self.index(force=True)

    def _validated_scopes(self, scopes: list[str] | None) -> list[str]:
        available = {source.scope for source in self.settings.sources}
        selected = sorted(available if not scopes else set(scopes))
        unknown = set(selected) - available
        if unknown:
            raise ValueError(
                f"unknown scope(s): {', '.join(sorted(unknown))}; "
                f"available: {', '.join(sorted(available))}"
            )
        return selected
