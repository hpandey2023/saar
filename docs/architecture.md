# Architecture

Saar separates source selection, indexing, retrieval, and MCP delivery so providers can change without changing agent behavior.

```text
designated folders
      │
      ▼
safe scanner ──► structure-aware chunks ──► embedding provider
      │                    │                         │
      │                    └────────► local BM25 ───┤
      │                                              ▼
      └── incremental state (SQLite)              Qdrant
                                                     │
agent ──MCP──► search_context ──► weighted RRF ──► optional reranker
                                                     │
                                                     ▼
                                          diversified, cited results
```

## Indexing

Each configured source has a short scope and a root path. Files are addressed in the index as `scope/relative/path`; absolute source paths are not stored in Qdrant payloads. A SHA-256 revision identifies content changes.

The scanner applies extension and size allowlists, default exclusions, source-specific include/exclude patterns, UTF-8 and binary checks, and root containment. Markdown headings and common code symbols become section boundaries. Long sections are split at paragraph, line, sentence, or word boundaries.

New points are written before the older revision is deleted. A failed update therefore leaves the previous searchable revision intact. Removed files are pruned after a successful source scan.

SQLite records only file revisions, aggregate state, query hashes, returned point identifiers, citations, and verified feedback. It does not retain query text.

## Collection schema

The Qdrant collection uses named vectors:

- `dense`: cosine-distance embeddings from the selected provider;
- `lexical`: BM25 sparse vectors with Qdrant's IDF modifier, when enabled.

Collection metadata stores a schema version and provider fingerprints. A model, dimension, prefix, contextual-group, or BM25 change that would make existing vectors incompatible fails with a migration message. The operator must select a new collection or explicitly rebuild.

## Retrieval

Dense and lexical candidates are fused with weighted reciprocal-rank fusion. Native weighted RRF is used where supported; a deterministic client-side implementation is the compatibility fallback. Optional reranking operates on the fused candidate set. Results are then filtered by human-verified exact-query feedback and limited per source to avoid one long file monopolizing context.

No retrieval system can guarantee zero misses. The included labeled-query evaluation makes recall, rank quality, and latency visible before a provider or configuration is adopted.

## MCP boundary

The read-only tools are:

- `search_context`
- `list_sources`
- `get_status`

`record_search_feedback` writes only when feedback is enabled and `human_verified=true`. It does not train an embedding model or accept an agent's own output as truth.
