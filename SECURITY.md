# Security policy

## Supported versions

Security fixes are applied to the latest released minor version. Until the first stable release, use the newest tagged `0.x` release.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability or exposed secret. Use GitHub's private vulnerability reporting for this repository. If that feature is unavailable, contact the repository security owner through a private reporting channel.

Include the affected version, deployment mode, reproduction steps, and likely impact. Do not include real credentials or confidential documents.

## Security model

Saar assumes that:

- source folders are selected by a trusted operator;
- the operating-system account may read those folders;
- MCP clients allowed to use the server may read every indexed scope exposed by that server;
- embedded Qdrant and SQLite files are protected by local filesystem permissions;
- shared Qdrant and HTTP deployments are placed behind appropriate network and access controls.

The MCP server does not implement per-user document authorization. Use separate deployments or collections where users have different access rights.

## Data flow

With the default FastEmbed provider, source text, vectors, query hashes, and search state remain local. Selecting Voyage or another remote endpoint sends document chunks and queries to that provider. Review its data-handling terms and the applicable policy first.

Search text is returned to the calling MCP client. Query text is not retained in SQLite; a SHA-256 hash of normalized text is stored to support exact-query feedback. Search result identifiers and citations are retained.

## Deployment requirements

- Keep source mounts read-only where possible.
- Store API keys in environment variables or an approved secrets manager.
- Never put credentials directly in YAML or commit `.env` files.
- Keep Qdrant and MCP endpoints on loopback unless protected by authentication, TLS, authorization, rate limits, and network policy.
- Back up only encrypted state and vector data where required.
- Run separate deployments when users have different source permissions.
- Treat indexed data as sensitive as the source corpus.

The `--allow-network` flag only acknowledges the risk; it does not add authentication or TLS.
