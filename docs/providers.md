# Provider configuration

Voyage is an embedding provider, not the vector database. Saar keeps Qdrant and hybrid retrieval consistent while allowing the embedding and reranking layers to change.

After changing models, use a new `qdrant.collection` or run the explicit rebuild command. Even models with equal dimensions do not share the same vector space.

## FastEmbed: free local default

```yaml
embedding:
  provider: fastembed
  model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
reranker:
  provider: none
```

The default model is Apache-2.0, 384-dimensional, multilingual, and roughly 220 MB. It is a practical starting point for laptops. Run `saar providers` to inspect the FastEmbed models supported by the installed version and review each license before production use.

A larger local option is:

```yaml
embedding:
  provider: fastembed
  model: intfloat/multilingual-e5-large
  query_prefix: "query: "
  document_prefix: "document: "
```

This uses more memory and disk. Evaluate it rather than assuming larger means better for the corpus.

## Ollama and Qwen3: free local

Install Ollama, then pull the smaller official model:

```bash
ollama pull qwen3-embedding:0.6b
```

```yaml
embedding:
  provider: ollama
  model: qwen3-embedding:0.6b
  base_url: http://localhost:11434
  dimensions: 1024
  query_prefix: "Instruct: Retrieve passages that answer the query.\nQuery: "
reranker:
  provider: none
```

The 0.6B Ollama artifact is about 639 MB and supports more than 100 languages. Larger 4B and 8B tags trade memory and latency for potential quality. See the [official Ollama model page](https://ollama.com/library/qwen3-embedding) and [Ollama embedding API](https://docs.ollama.com/capabilities/embeddings).

## Voyage standard embeddings

Install the optional client and provide the key only through the environment:

```bash
python -m pip install 'saar-mcp[voyage]'
export VOYAGE_API_KEY='...'
```

```yaml
embedding:
  provider: voyage
  model: voyage-4
  api_key_env: VOYAGE_API_KEY
```

Do not place the key itself in YAML. Document chunks and search queries leave the machine when this provider is used. Review the [Voyage embedding documentation](https://docs.voyageai.com/docs/embeddings) and current pricing/data terms.

## Voyage contextual embeddings

`voyage-context-4` uses a different API. The package detects the model name, sends ordered groups of chunks from the same document to `contextualized_embed()`, and embeds queries as single-item groups.

```yaml
embedding:
  provider: voyage
  model: voyage-context-4
  api_key_env: VOYAGE_API_KEY
  dimensions: 1024
  contextual_group_size: 32
index:
  chunk_overlap_chars: 0
```

Zero overlap is enforced because Voyage recommends non-overlapping chunks for contextualized embedding. The group size bounds request size; it also becomes part of the collection fingerprint. See the [official contextualized embedding guide](https://docs.voyageai.com/docs/contextualized-chunk-embeddings).

## OpenAI-compatible endpoint

Use this for an internally hosted service, LocalAI, a vLLM-compatible gateway, or another endpoint implementing `POST /embeddings`:

```yaml
embedding:
  provider: openai_compatible
  model: internal-embedding-model
  base_url: https://embeddings.example.internal/v1
  api_key_env: EMBEDDING_API_KEY
  dimensions: 1024
```

The key is optional when the endpoint does not require one. Network trust, certificates, licensing, and data retention remain the operator's responsibility.

## Reranking

The default is no reranker, which minimizes latency and model downloads. Hybrid retrieval still runs.

Free local English reranker:

```yaml
reranker:
  provider: fastembed
  model: Xenova/ms-marco-MiniLM-L-6-v2
  top_k: 30
```

Paid Voyage reranker:

```yaml
reranker:
  provider: voyage
  model: rerank-2.5
  api_key_env: VOYAGE_API_KEY
  top_k: 30
```

Confirm language fit and license before enabling a local reranker. A weak reranker can make good first-stage retrieval worse.

## Custom Python provider

Set `provider: python` and supply an importable `module:Class` implementing the documented protocol. This supports in-house models and providers not built into the package. See [custom providers](custom-provider.md).
