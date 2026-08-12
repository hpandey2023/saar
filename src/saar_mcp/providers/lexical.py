"""Local BM25 sparse-vector adapter."""

from __future__ import annotations

import hashlib

from ..config import LexicalSettings
from ..exceptions import ProviderError
from .base import SparseValues


class FastEmbedBm25Encoder:
    def __init__(self, settings: LexicalSettings):
        try:
            import fastembed
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(
                model_name=settings.model,
                language=settings.language,
                disable_stemmer=settings.disable_stemmer,
            )
        except Exception as exc:
            raise ProviderError(f"failed to load local BM25 encoder: {exc}") from exc
        self.settings = settings
        self._fastembed_version = str(getattr(fastembed, "__version__", "unknown"))

    @property
    def fingerprint(self) -> str:
        payload = (
            f"{self._fastembed_version}\x1f{self.settings.model}\x1f"
            f"{self.settings.language}\x1f{self.settings.disable_stemmer}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    @staticmethod
    def _convert(item: object) -> SparseValues:
        try:
            return SparseValues(
                indices=[int(value) for value in item.indices],  # type: ignore[attr-defined]
                values=[float(value) for value in item.values],  # type: ignore[attr-defined]
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProviderError(f"invalid BM25 sparse embedding: {exc}") from exc

    def embed_documents(self, texts: list[str]) -> list[SparseValues]:
        try:
            return [self._convert(item) for item in self._model.embed(texts)]
        except Exception as exc:
            raise ProviderError(f"BM25 document embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> SparseValues:
        return self.embed_documents([text])[0]
