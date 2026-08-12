"""Optional second-stage reranking adapters."""

from __future__ import annotations

import importlib

from ..config import RerankerSettings, read_secret
from ..exceptions import ConfigurationError, ProviderError
from .base import ProviderInfo, Reranker


class NoReranker:
    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(provider="none", model="none", local=True)

    def score(self, query: str, documents: list[str]) -> list[float]:
        del query
        return [float(len(documents) - index) for index in range(len(documents))]


class FastEmbedReranker:
    def __init__(self, settings: RerankerSettings):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=settings.model, **settings.kwargs)
            metadata = next(
                (
                    item
                    for item in TextCrossEncoder.list_supported_models()
                    if item["model"] == settings.model
                ),
                {},
            )
        except Exception as exc:
            raise ProviderError(
                f"failed to load FastEmbed reranker {settings.model!r}: {exc}"
            ) from exc
        self.settings = settings
        self._license = str(metadata.get("license", "unknown"))

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            provider="fastembed", model=self.settings.model, license=self._license, local=True
        )

    def score(self, query: str, documents: list[str]) -> list[float]:
        try:
            return [float(value) for value in self._model.rerank(query, documents)]
        except Exception as exc:
            raise ProviderError(f"FastEmbed reranking failed: {exc}") from exc


class VoyageReranker:
    def __init__(self, settings: RerankerSettings):
        try:
            import voyageai
        except ImportError as exc:
            raise ConfigurationError(
                "Voyage reranker requires: pip install 'saar-mcp[voyage]'"
            ) from exc
        self.settings = settings
        self._client = voyageai.Client(api_key=read_secret(settings.api_key_env, required=True))

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(provider="voyage", model=self.settings.model)

    def score(self, query: str, documents: list[str]) -> list[float]:
        try:
            result = self._client.rerank(
                query=query,
                documents=documents,
                model=self.settings.model,
                top_k=len(documents),
            )
            scores = [0.0] * len(documents)
            for row in result.results:
                scores[int(row.index)] = float(row.relevance_score)
            return scores
        except Exception as exc:
            raise ProviderError(f"Voyage reranking failed: {exc}") from exc


def load_python_reranker(settings: RerankerSettings) -> Reranker:
    module_name, separator, object_name = (settings.class_path or "").partition(":")
    if not separator:
        raise ConfigurationError("class_path must use 'module:Class' syntax")
    try:
        provider_class = getattr(importlib.import_module(module_name), object_name)
        provider = provider_class(settings=settings, **settings.kwargs)
    except Exception as exc:
        raise ProviderError(f"failed to load custom reranker {settings.class_path}: {exc}") from exc
    if not isinstance(provider, Reranker):
        raise ConfigurationError(
            f"custom reranker {settings.class_path} does not implement the Reranker contract"
        )
    return provider
