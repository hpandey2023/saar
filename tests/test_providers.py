from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import httpx
import pytest

from saar_mcp.config import EmbeddingSettings, LexicalSettings, RerankerSettings
from saar_mcp.exceptions import ConfigurationError, ProviderError
from saar_mcp.providers.base import ProviderInfo
from saar_mcp.providers.embeddings import (
    FastEmbedEmbedder,
    OllamaEmbedder,
    OpenAICompatibleEmbedder,
    VoyageEmbedder,
    load_python_embedder,
)
from saar_mcp.providers.factory import (
    build_embedder,
    build_lexical_encoder,
    build_reranker,
)
from saar_mcp.providers.lexical import FastEmbedBm25Encoder
from saar_mcp.providers.rerankers import (
    FastEmbedReranker,
    NoReranker,
    VoyageReranker,
    load_python_reranker,
)

VECTOR = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class FakeResponse:
    def __init__(self, value, *, fail=False):
        self.value = value
        self.fail = fail

    def raise_for_status(self):
        if self.fail:
            raise httpx.HTTPError("failed")

    def json(self):
        return self.value


def test_ollama_adapter_batches_and_reports_errors(monkeypatch):
    class Client:
        def __init__(self, **kwargs):
            pass

        def post(self, endpoint, json):
            assert endpoint.endswith("/api/embed")
            return FakeResponse({"embeddings": [VECTOR for _ in json["input"]]})

    monkeypatch.setattr(httpx, "Client", Client)
    settings = EmbeddingSettings(
        provider="ollama", model="qwen", dimensions=8, document_prefix="doc: "
    )
    adapter = OllamaEmbedder(settings)
    assert adapter.info.local is True
    assert adapter.dimension == 8
    assert len(adapter.embed_documents(["a", "b"])) == 2
    assert adapter.embed_query("a") == VECTOR
    assert len(adapter.fingerprint) == 20

    adapter._client.post = lambda *args, **kwargs: FakeResponse({}, fail=True)
    with pytest.raises(ProviderError, match="Ollama embedding failed"):
        adapter.embed_query("failure")


def test_openai_compatible_adapter_sorts_response_and_uses_key(monkeypatch):
    monkeypatch.setenv("TEST_EMBEDDING_KEY", "not-a-real-key")

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["headers"]["Authorization"].startswith("Bearer ")

        def post(self, endpoint, json):
            assert endpoint == "https://example.test/v1/embeddings"
            assert json["dimensions"] == 8
            return FakeResponse(
                {
                    "data": [
                        {"index": 1, "embedding": [2.0] * 8},
                        {"index": 0, "embedding": [1.0] * 8},
                    ]
                }
            )

    monkeypatch.setattr(httpx, "Client", Client)
    adapter = OpenAICompatibleEmbedder(
        EmbeddingSettings(
            provider="openai_compatible",
            model="internal",
            base_url="https://example.test/v1",
            api_key_env="TEST_EMBEDDING_KEY",
            dimensions=8,
        )
    )
    assert adapter.embed_documents(["first", "second"])[0] == [1.0] * 8
    assert adapter.info.provider == "openai_compatible"


def _fake_voyage_module():
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "not-a-real-key"

        def embed(self, texts, **kwargs):
            return SimpleNamespace(embeddings=[VECTOR for _ in texts])

        def contextualized_embed(self, inputs, **kwargs):
            return SimpleNamespace(
                results=[SimpleNamespace(embeddings=[VECTOR for _ in group]) for group in inputs]
            )

        def rerank(self, query, documents, model, top_k):
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=index, relevance_score=0.9 - index * 0.1)
                    for index in reversed(range(len(documents)))
                ]
            )

    return types.SimpleNamespace(Client=Client)


def test_voyage_standard_contextual_and_reranking(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "not-a-real-key")
    monkeypatch.setitem(sys.modules, "voyageai", _fake_voyage_module())

    standard = VoyageEmbedder(EmbeddingSettings(provider="voyage", model="voyage-4", dimensions=8))
    assert standard.contextual_documents is False
    assert standard.embed_documents(["a"])[0] == VECTOR
    assert standard.embed_query("a") == VECTOR

    contextual = VoyageEmbedder(
        EmbeddingSettings(provider="voyage", model="voyage-context-4", dimensions=8)
    )
    assert contextual.contextual_documents is True
    assert len(contextual.embed_documents(["one", "two"])) == 2
    assert contextual.embed_query("question") == VECTOR
    assert contextual.info.model == "voyage-context-4"

    reranker = VoyageReranker(RerankerSettings(provider="voyage", model="rerank-2.5"))
    assert reranker.score("q", ["a", "b"]) == [0.9, 0.8]


def test_fastembed_wrappers_with_mock_models(monkeypatch):
    class TextEmbedding:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def list_supported_models():
            return [{"model": "fake", "dim": 8, "license": "MIT"}]

        def passage_embed(self, texts):
            return [VECTOR for _ in texts]

        def query_embed(self, texts):
            return [VECTOR for _ in texts]

    class SparseTextEmbedding:
        def __init__(self, **kwargs):
            pass

        def embed(self, texts):
            return [SimpleNamespace(indices=[1, 2], values=[1.0, 0.5]) for _ in texts]

    class TextCrossEncoder:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def list_supported_models():
            return [{"model": "fake-reranker", "license": "MIT"}]

        def rerank(self, query, documents):
            return [0.25 + index for index, _ in enumerate(documents)]

    monkeypatch.setattr("fastembed.TextEmbedding", TextEmbedding)
    monkeypatch.setattr("fastembed.SparseTextEmbedding", SparseTextEmbedding)
    monkeypatch.setattr("fastembed.rerank.cross_encoder.TextCrossEncoder", TextCrossEncoder)

    embedder = FastEmbedEmbedder(EmbeddingSettings(provider="fastembed", model="fake"))
    assert embedder.dimension == 8
    assert embedder.info.license == "MIT"
    assert embedder.embed_documents(["a"])[0] == VECTOR
    assert embedder.embed_query("a") == VECTOR

    lexical = FastEmbedBm25Encoder(LexicalSettings())
    assert lexical.embed_query("a").indices == [1, 2]
    assert len(lexical.fingerprint) == 20

    reranker = FastEmbedReranker(RerankerSettings(provider="fastembed", model="fake-reranker"))
    assert reranker.info.license == "MIT"
    assert reranker.score("q", ["a", "b"]) == [0.25, 1.25]


def test_custom_python_adapters_and_factories(monkeypatch):
    module = types.ModuleType("test_custom_provider")

    class CustomEmbedder:
        info = ProviderInfo("custom", "model", "MIT", True)
        dimension = 8
        fingerprint = "custom-v1"

        def __init__(self, settings, **kwargs):
            pass

        def embed_documents(self, texts):
            return [VECTOR for _ in texts]

        def embed_query(self, text):
            return VECTOR

    class CustomReranker:
        info = ProviderInfo("custom", "reranker", "MIT", True)

        def __init__(self, settings, **kwargs):
            pass

        def score(self, query, documents):
            return [1.0] * len(documents)

    class Invalid:
        def __init__(self, settings, **kwargs):
            pass

    module.CustomEmbedder = CustomEmbedder
    module.CustomReranker = CustomReranker
    module.Invalid = Invalid
    monkeypatch.setitem(sys.modules, module.__name__, module)

    embed_settings = EmbeddingSettings(
        provider="python", class_path="test_custom_provider:CustomEmbedder"
    )
    rerank_settings = RerankerSettings(
        provider="python", class_path="test_custom_provider:CustomReranker"
    )
    assert load_python_embedder(embed_settings).dimension == 8
    assert load_python_reranker(rerank_settings).score("q", ["a"]) == [1.0]
    assert build_embedder(embed_settings).info.provider == "custom"
    assert build_reranker(rerank_settings).info.provider == "custom"
    assert build_reranker(RerankerSettings(provider="none")).score("q", ["a", "b"]) == [
        2.0,
        1.0,
    ]
    assert build_lexical_encoder(LexicalSettings(enabled=False)) is None

    with pytest.raises(ConfigurationError, match="does not implement"):
        load_python_embedder(
            EmbeddingSettings(provider="python", class_path="test_custom_provider:Invalid")
        )


def test_no_reranker_preserves_order():
    reranker = NoReranker()
    assert reranker.info.local is True
    assert reranker.score("ignored", ["a", "b", "c"]) == [3.0, 2.0, 1.0]
