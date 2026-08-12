"""Qdrant collection management and hybrid retrieval."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from .config import QdrantSettings, RetrievalSettings, read_secret
from .exceptions import StorageError
from .models import DocumentChunk, SearchHit
from .providers.base import SparseValues

SCHEMA_VERSION = 1
DENSE_VECTOR = "dense"
LEXICAL_VECTOR = "lexical"


class VectorStore:
    def __init__(
        self,
        settings: QdrantSettings,
        *,
        embedding_fingerprint: str,
        dimension: int,
        lexical_fingerprint: str | None,
    ):
        self.settings = settings
        self.embedding_fingerprint = embedding_fingerprint
        self.dimension = dimension
        self.lexical_fingerprint = lexical_fingerprint
        try:
            if settings.mode == "embedded":
                settings.path.mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=str(settings.path))
            else:
                self.client = QdrantClient(
                    url=settings.url,
                    api_key=read_secret(settings.api_key_env, required=False),
                    prefer_grpc=settings.prefer_grpc,
                    timeout=60,
                )
        except Exception as exc:
            raise StorageError(f"could not connect to Qdrant: {exc}") from exc

    @property
    def collection(self) -> str:
        return self.settings.collection

    @property
    def schema_metadata(self) -> dict[str, Any]:
        return {
            "saar_schema": SCHEMA_VERSION,
            "embedding_fingerprint": self.embedding_fingerprint,
            "lexical_fingerprint": self.lexical_fingerprint or "disabled",
        }

    def collection_exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection)
        except Exception as exc:
            raise StorageError(
                f"could not inspect Qdrant collection {self.collection!r}: {exc}"
            ) from exc

    def ensure_collection(self) -> None:
        if not self.collection_exists():
            sparse = None
            if self.lexical_fingerprint:
                sparse = {LEXICAL_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)}
            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config={
                        DENSE_VECTOR: models.VectorParams(
                            size=self.dimension, distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config=sparse,
                    metadata=self.schema_metadata,
                )
                if self.settings.mode == "server":
                    for field in (
                        "kind",
                        "scope",
                        "content_type",
                        "document_type",
                        "date",
                        "people_norm",
                        "status",
                        "source_key",
                        "revision",
                    ):
                        self.client.create_payload_index(
                            self.collection,
                            field_name=field,
                            field_schema=models.PayloadSchemaType.KEYWORD,
                            wait=True,
                        )
            except Exception as exc:
                raise StorageError(
                    f"could not create Qdrant collection {self.collection!r}: {exc}"
                ) from exc
            return
        self._validate_collection()

    def _validate_collection(self) -> None:
        try:
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            dense = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
            sparse = info.config.params.sparse_vectors or {}
            metadata = info.config.metadata or {}
        except Exception as exc:
            raise StorageError(f"could not read Qdrant collection schema: {exc}") from exc

        problems: list[str] = []
        if dense is None or int(dense.size) != self.dimension:
            found = getattr(dense, "size", "missing")
            problems.append(f"dense dimension is {found}, expected {self.dimension}")
        has_lexical = LEXICAL_VECTOR in sparse
        if has_lexical != bool(self.lexical_fingerprint):
            problems.append("lexical vector configuration differs")
        for key, expected in self.schema_metadata.items():
            if metadata.get(key) != expected:
                problems.append(f"{key} differs")
        if problems:
            detail = "; ".join(problems)
            raise StorageError(
                f"collection {self.collection!r} is incompatible ({detail}). "
                "Use a new collection name or run 'saar rebuild --yes'."
            )

    @staticmethod
    def _sparse(value: SparseValues) -> models.SparseVector:
        return models.SparseVector(indices=value.indices, values=value.values)

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        dense_vectors: Sequence[list[float]],
        sparse_vectors: Sequence[SparseValues] | None,
    ) -> None:
        if len(chunks) != len(dense_vectors):
            raise StorageError("chunk and dense-vector counts differ")
        if sparse_vectors is not None and len(chunks) != len(sparse_vectors):
            raise StorageError("chunk and sparse-vector counts differ")
        points: list[models.PointStruct] = []
        for index, chunk in enumerate(chunks):
            vector: dict[str, Any] = {DENSE_VECTOR: dense_vectors[index]}
            if sparse_vectors is not None:
                vector[LEXICAL_VECTOR] = self._sparse(sparse_vectors[index])
            points.append(
                models.PointStruct(id=chunk.point_id, vector=vector, payload=chunk.payload())
            )
        try:
            self.client.upsert(self.collection, points=points, wait=True)
        except Exception as exc:
            raise StorageError(f"Qdrant upsert failed: {exc}") from exc

    @staticmethod
    def _equals(key: str, value: str) -> models.FieldCondition:
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    def delete_old_revisions(self, source_key: str, current_revision: str) -> None:
        selector = models.Filter(
            must=[self._equals("source_key", source_key)],
            must_not=[self._equals("revision", current_revision)],
        )
        self._delete(selector)

    def delete_source(self, source_key: str) -> None:
        self._delete(models.Filter(must=[self._equals("source_key", source_key)]))

    def _delete(self, selector: models.Filter) -> None:
        try:
            self.client.delete(self.collection, points_selector=selector, wait=True)
        except Exception as exc:
            raise StorageError(f"Qdrant deletion failed: {exc}") from exc

    def _filter(
        self,
        *,
        scopes: list[str],
        content_types: list[str] | None,
        document_types: list[str] | None,
        dates: list[str] | None,
        people: list[str] | None,
        exclude_statuses: list[str],
    ) -> models.Filter:
        must: list[Any] = [self._equals("kind", "chunk")]
        if scopes:
            must.append(
                models.FieldCondition(key="scope", match=models.MatchAny(any=sorted(set(scopes))))
            )
        if content_types:
            must.append(
                models.FieldCondition(
                    key="content_type", match=models.MatchAny(any=sorted(set(content_types)))
                )
            )
        if document_types:
            must.append(
                models.FieldCondition(
                    key="document_type", match=models.MatchAny(any=sorted(set(document_types)))
                )
            )
        if dates:
            must.append(
                models.FieldCondition(key="date", match=models.MatchAny(any=sorted(set(dates))))
            )
        if people:
            must.append(
                models.FieldCondition(
                    key="people_norm", match=models.MatchAny(any=sorted(set(people)))
                )
            )
        must_not: list[Any] = []
        if exclude_statuses:
            must_not.append(
                models.FieldCondition(
                    key="status", match=models.MatchAny(any=sorted(set(exclude_statuses)))
                )
            )
        return models.Filter(must=must, must_not=must_not or None)

    def search(
        self,
        dense_query: list[float],
        sparse_query: SparseValues | None,
        *,
        scopes: list[str],
        content_types: list[str] | None,
        document_types: list[str] | None,
        dates: list[str] | None,
        people: list[str] | None,
        retrieval: RetrievalSettings,
    ) -> tuple[list[SearchHit], str]:
        query_filter = self._filter(
            scopes=scopes,
            content_types=content_types,
            document_types=document_types,
            dates=dates,
            people=people,
            exclude_statuses=retrieval.exclude_statuses,
        )
        if sparse_query is None:
            try:
                response = self.client.query_points(
                    self.collection,
                    query=dense_query,
                    using=DENSE_VECTOR,
                    query_filter=query_filter,
                    limit=retrieval.candidate_limit,
                    with_payload=True,
                )
                return self._hits(response.points), "dense"
            except Exception as exc:
                raise StorageError(f"Qdrant dense search failed: {exc}") from exc

        try:
            response = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_query,
                        using=DENSE_VECTOR,
                        filter=query_filter,
                        limit=retrieval.candidate_limit,
                    ),
                    models.Prefetch(
                        query=self._sparse(sparse_query),
                        using=LEXICAL_VECTOR,
                        filter=query_filter,
                        limit=retrieval.candidate_limit,
                    ),
                ],
                query=models.RrfQuery(
                    rrf=models.Rrf(
                        k=60,
                        weights=[retrieval.dense_weight, retrieval.lexical_weight],
                    )
                ),
                limit=retrieval.candidate_limit,
                with_payload=True,
            )
            return self._hits(response.points), "weighted_rrf"
        except Exception:
            return self._manual_rrf(
                dense_query,
                sparse_query,
                query_filter=query_filter,
                retrieval=retrieval,
            ), "weighted_rrf_client_fallback"

    def _manual_rrf(
        self,
        dense_query: list[float],
        sparse_query: SparseValues,
        *,
        query_filter: models.Filter,
        retrieval: RetrievalSettings,
    ) -> list[SearchHit]:
        try:
            dense = self.client.query_points(
                self.collection,
                query=dense_query,
                using=DENSE_VECTOR,
                query_filter=query_filter,
                limit=retrieval.candidate_limit,
                with_payload=True,
            ).points
            lexical = self.client.query_points(
                self.collection,
                query=self._sparse(sparse_query),
                using=LEXICAL_VECTOR,
                query_filter=query_filter,
                limit=retrieval.candidate_limit,
                with_payload=True,
            ).points
        except Exception as exc:
            raise StorageError(f"Qdrant hybrid search failed: {exc}") from exc

        scores: dict[str, float] = {}
        payloads: dict[str, tuple[object, dict[str, Any]]] = {}
        for weight, rows in (
            (retrieval.dense_weight, dense),
            (retrieval.lexical_weight, lexical),
        ):
            for rank, point in enumerate(rows, start=1):
                identifier = str(point.id)
                scores[identifier] = scores.get(identifier, 0.0) + weight / (60 + rank)
                payloads[identifier] = (point.id, point.payload or {})
        ordered = sorted(scores, key=scores.__getitem__, reverse=True)[: retrieval.candidate_limit]
        return [
            SearchHit.from_payload(payloads[item][0], scores[item], payloads[item][1])
            for item in ordered
        ]

    @staticmethod
    def _hits(points: Sequence[Any]) -> list[SearchHit]:
        return [
            SearchHit.from_payload(point.id, point.score, point.payload or {}) for point in points
        ]

    def recreate(self) -> None:
        try:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
            self.ensure_collection()
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"could not recreate Qdrant collection: {exc}") from exc

    def diagnostics(self) -> dict[str, Any]:
        details: dict[str, Any] = {
            "mode": self.settings.mode,
            "collection": self.collection,
            "embedding_dimension": self.dimension,
            "schema": self.schema_metadata,
        }
        try:
            exists = self.client.collection_exists(self.collection)
            details["reachable"] = True
            details["collection_exists"] = exists
            if exists:
                info = self.client.get_collection(self.collection)
                details["points"] = int(info.points_count or 0)
                details["status"] = str(info.status.value)
        except Exception as exc:
            details["reachable"] = False
            details["error"] = str(exc)
        return json.loads(json.dumps(details, default=str))

    def close(self) -> None:
        self.client.close()
