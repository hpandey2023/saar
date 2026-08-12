"""Built-in dense embedding adapters."""

from __future__ import annotations

import hashlib
import importlib
import warnings
from functools import cached_property
from typing import Any

import httpx

from ..config import EmbeddingSettings, read_secret
from ..exceptions import ConfigurationError, ProviderError
from .base import Embedder, ProviderInfo


def _fingerprint(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _float_vectors(values: Any) -> list[list[float]]:
    return [[float(item) for item in vector] for vector in values]


class FastEmbedEmbedder:
    def __init__(self, settings: EmbeddingSettings):
        try:
            import fastembed
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - declared core dependency
            raise ProviderError("fastembed is not installed") from exc

        self.settings = settings
        self._fastembed_version = str(getattr(fastembed, "__version__", "unknown"))
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"The model .* now uses mean pooling instead of CLS embedding.*",
                    category=UserWarning,
                )
                self._model = TextEmbedding(model_name=settings.model, **settings.kwargs)
        except Exception as exc:
            raise ProviderError(
                f"failed to load FastEmbed model {settings.model!r}: {exc}"
            ) from exc
        self._model_metadata = next(
            (
                item
                for item in TextEmbedding.list_supported_models()
                if item["model"] == settings.model
            ),
            {},
        )

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            provider="fastembed",
            model=self.settings.model,
            license=str(self._model_metadata.get("license", "unknown")),
            local=True,
        )

    @cached_property
    def dimension(self) -> int:
        if self.settings.dimensions:
            return self.settings.dimensions
        configured = self._model_metadata.get("dim")
        if configured:
            return int(configured)
        return len(self.embed_query("dimension probe"))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            "fastembed",
            self._fastembed_version,
            self.settings.model,
            self.dimension,
            self.settings.query_prefix,
            self.settings.document_prefix,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepared = [f"{self.settings.document_prefix}{text}" for text in texts]
        try:
            return _float_vectors(self._model.passage_embed(prepared))
        except Exception as exc:
            raise ProviderError(f"FastEmbed document embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return _float_vectors(self._model.query_embed([f"{self.settings.query_prefix}{text}"]))[
                0
            ]
        except Exception as exc:
            raise ProviderError(f"FastEmbed query embedding failed: {exc}") from exc


class OllamaEmbedder:
    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self._client = httpx.Client(timeout=settings.timeout_seconds)

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(provider="ollama", model=self.settings.model, local=True)

    @cached_property
    def dimension(self) -> int:
        return self.settings.dimensions or len(self.embed_query("dimension probe"))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            "ollama",
            self.settings.model,
            self.dimension,
            self.settings.query_prefix,
            self.settings.document_prefix,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        endpoint = f"{self.settings.base_url.rstrip('/')}/api/embed"
        try:
            response = self._client.post(
                endpoint, json={"model": self.settings.model, "input": texts}
            )
            response.raise_for_status()
            vectors = response.json()["embeddings"]
            return _float_vectors(vectors)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"Ollama embedding failed at {endpoint}; verify Ollama is running and model "
                f"{self.settings.model!r} is installed: {exc}"
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([f"{self.settings.document_prefix}{text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"{self.settings.query_prefix}{text}"])[0]


class OpenAICompatibleEmbedder:
    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        api_key = read_secret(settings.api_key_env, required=False)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=settings.timeout_seconds, headers=headers)

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(provider="openai_compatible", model=self.settings.model)

    @cached_property
    def dimension(self) -> int:
        return self.settings.dimensions or len(self.embed_query("dimension probe"))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            "openai_compatible",
            self.settings.base_url,
            self.settings.model,
            self.dimension,
            self.settings.query_prefix,
            self.settings.document_prefix,
        )

    def _endpoint(self) -> str:
        base = self.settings.base_url.rstrip("/")
        return base if base.endswith("/embeddings") else f"{base}/embeddings"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.settings.model, "input": texts}
        if self.settings.dimensions:
            payload["dimensions"] = self.settings.dimensions
        endpoint = self._endpoint()
        try:
            response = self._client.post(endpoint, json=payload)
            response.raise_for_status()
            rows = sorted(response.json()["data"], key=lambda item: item.get("index", 0))
            return _float_vectors(item["embedding"] for item in rows)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ProviderError(f"OpenAI-compatible embedding failed at {endpoint}: {exc}") from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([f"{self.settings.document_prefix}{text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"{self.settings.query_prefix}{text}"])[0]


class VoyageEmbedder:
    def __init__(self, settings: EmbeddingSettings):
        try:
            import voyageai
        except ImportError as exc:
            raise ConfigurationError(
                "Voyage provider requires: pip install 'saar-mcp[voyage]'"
            ) from exc
        self.settings = settings
        self._client = voyageai.Client(
            api_key=read_secret(settings.api_key_env, required=True),
            timeout=settings.timeout_seconds,
        )

    @property
    def contextual_documents(self) -> bool:
        return self.settings.model.startswith("voyage-context-")

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(provider="voyage", model=self.settings.model)

    @cached_property
    def dimension(self) -> int:
        return self.settings.dimensions or len(self.embed_query("dimension probe"))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            "voyage",
            self.settings.model,
            self.dimension,
            self.settings.query_prefix,
            self.settings.document_prefix,
            self.settings.contextual_group_size,
        )

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        try:
            result = self._client.embed(
                texts,
                model=self.settings.model,
                input_type=input_type,
                output_dimension=self.settings.dimensions,
            )
            return _float_vectors(result.embeddings)
        except Exception as exc:
            raise ProviderError(f"Voyage embedding failed: {exc}") from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepared = [f"{self.settings.document_prefix}{text}" for text in texts]
        if self.contextual_documents:
            try:
                result = self._client.contextualized_embed(
                    inputs=[prepared],
                    model=self.settings.model,
                    input_type="document",
                    output_dimension=self.settings.dimensions,
                )
                return _float_vectors(result.results[0].embeddings)
            except Exception as exc:
                raise ProviderError(f"Voyage contextual document embedding failed: {exc}") from exc
        return self._embed(prepared, "document")

    def embed_query(self, text: str) -> list[float]:
        prepared = f"{self.settings.query_prefix}{text}"
        if self.contextual_documents:
            try:
                result = self._client.contextualized_embed(
                    inputs=[[prepared]],
                    model=self.settings.model,
                    input_type="query",
                    output_dimension=self.settings.dimensions,
                )
                return _float_vectors(result.results[0].embeddings)[0]
            except Exception as exc:
                raise ProviderError(f"Voyage contextual query embedding failed: {exc}") from exc
        return self._embed([prepared], "query")[0]


def load_python_embedder(settings: EmbeddingSettings) -> Embedder:
    module_name, separator, object_name = (settings.class_path or "").partition(":")
    if not separator:
        raise ConfigurationError("class_path must use 'module:Class' syntax")
    try:
        provider_class = getattr(importlib.import_module(module_name), object_name)
        provider = provider_class(settings=settings, **settings.kwargs)
    except Exception as exc:
        raise ProviderError(f"failed to load custom embedder {settings.class_path}: {exc}") from exc
    if not isinstance(provider, Embedder):
        raise ConfigurationError(
            f"custom embedder {settings.class_path} does not implement the Embedder contract"
        )
    return provider
