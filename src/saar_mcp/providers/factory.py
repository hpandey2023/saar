"""Validated provider construction."""

from __future__ import annotations

from ..config import EmbeddingSettings, LexicalSettings, RerankerSettings
from ..exceptions import ConfigurationError
from .base import Embedder, LexicalEncoder, Reranker
from .embeddings import (
    FastEmbedEmbedder,
    OllamaEmbedder,
    OpenAICompatibleEmbedder,
    VoyageEmbedder,
    load_python_embedder,
)
from .lexical import FastEmbedBm25Encoder
from .rerankers import FastEmbedReranker, NoReranker, VoyageReranker, load_python_reranker


def build_embedder(settings: EmbeddingSettings) -> Embedder:
    providers = {
        "fastembed": FastEmbedEmbedder,
        "ollama": OllamaEmbedder,
        "openai_compatible": OpenAICompatibleEmbedder,
        "voyage": VoyageEmbedder,
    }
    if settings.provider == "python":
        return load_python_embedder(settings)
    try:
        return providers[settings.provider](settings)
    except KeyError as exc:  # pragma: no cover - guarded by Pydantic
        raise ConfigurationError(f"unsupported embedding provider: {settings.provider}") from exc


def build_lexical_encoder(settings: LexicalSettings) -> LexicalEncoder | None:
    return FastEmbedBm25Encoder(settings) if settings.enabled else None


def build_reranker(settings: RerankerSettings) -> Reranker:
    if settings.provider == "none":
        return NoReranker()
    if settings.provider == "fastembed":
        return FastEmbedReranker(settings)
    if settings.provider == "voyage":
        return VoyageReranker(settings)
    if settings.provider == "python":
        return load_python_reranker(settings)
    raise ConfigurationError(f"unsupported reranker provider: {settings.provider}")
