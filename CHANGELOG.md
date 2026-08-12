# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-08-12

### Added

- Retrieval metadata from YAML frontmatter: document type, date, people, and tags.
- Exact `document_types`, `dates`, and `people` filters for the CLI and MCP `search_context` tool.
- Metadata-aware dense and lexical embeddings while keeping cited result text unchanged.

### Changed

- The index recipe version now forces a safe re-index so existing collections gain metadata payloads.

## [0.1.0] - 2026-08-12

### Added

- Local-first FastEmbed and embedded-Qdrant default.
- Dense plus BM25 retrieval with weighted reciprocal-rank fusion.
- Ollama, OpenAI-compatible, Voyage, and custom Python providers.
- Native Voyage `voyage-context-4` contextualized chunk support.
- Incremental indexing, source pruning, filters, citation-safe payloads, and MCP tools.
- Human-verified exact-query feedback without retaining query text.
- Evaluation, Docker Compose, continuous integration, and security guidance.
- A reproducible retrieval and agent-level A/B evaluation method.
