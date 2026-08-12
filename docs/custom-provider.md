# Custom providers

A custom embedding adapter must expose provider metadata, a stable dimension and fingerprint, and separate document/query methods. The constructor receives the validated embedding settings.

```python
from saar_mcp.providers.base import ProviderInfo


class CustomEmbedder:
    def __init__(self, settings, endpoint="http://localhost:9000"):
        self.settings = settings
        self.endpoint = endpoint

    @property
    def info(self):
        return ProviderInfo(
            provider="custom",
            model=self.settings.model,
            license="internal",
            local=True,
        )

    @property
    def dimension(self):
        return self.settings.dimensions or 768

    @property
    def fingerprint(self):
        return f"custom:{self.settings.model}:{self.dimension}:v1"

    def embed_documents(self, texts):
        return self._request(texts, input_type="document")

    def embed_query(self, text):
        return self._request([text], input_type="query")[0]

    def _request(self, texts, input_type):
        # Call the approved local or remote service and return list[list[float]].
        raise NotImplementedError
```

Make the module importable in the same virtual environment as the MCP server, then configure it:

```yaml
embedding:
  provider: python
  model: internal-v1
  dimensions: 768
  class_path: custom_search.embedding:CustomEmbedder
  kwargs:
    endpoint: http://localhost:9000
```

The runtime validates the adapter against the `Embedder` protocol and validates every returned vector shape. Keep credentials in named environment variables and convert operational failures to `ProviderError` without including secret values.

A custom reranker similarly implements `info` and `score(query, documents)`. Configure it with `reranker.provider: python` and `reranker.class_path`.

If an existing system already exposes an OpenAI-compatible embeddings endpoint, prefer the built-in adapter. It is less code to maintain.
