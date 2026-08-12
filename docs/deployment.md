# Deployment

## Local stdio

The lowest-risk deployment is one dedicated virtual environment and embedded Qdrant instance:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/saar --config saar.yaml doctor
.venv/bin/saar --config saar.yaml index
```

Configure the MCP client to run `saar --config ... serve`. The client owns the process lifecycle. Run `watch` separately only when near-real-time filesystem changes are required.

Embedded Qdrant permits one process to open its storage directory at a time. Use Qdrant server mode when several MCP processes share a collection.

## Qdrant server

```yaml
qdrant:
  mode: server
  url: http://127.0.0.1:6333
  collection: saar-shared
  api_key_env: QDRANT_API_KEY
```

The key may be omitted for a loopback-only development service. Require authentication and TLS for remote Qdrant. Back up Qdrant and SQLite together if verified feedback and incremental state must be preserved.

Only one indexer should update a collection at a time. Multiple read-only MCP processes may search it.

## Docker Compose trial

The included Compose file starts Qdrant and HTTP MCP on loopback. By default it indexes the synthetic sample corpus.

```bash
docker compose up -d qdrant
docker compose run --rm index
docker compose up -d mcp
```

To use a configured source folder:

```bash
CONTEXT_SOURCE=/absolute/path/to/resources docker compose run --rm index
CONTEXT_SOURCE=/absolute/path/to/resources docker compose up -d mcp
```

On macOS and Windows, Docker Desktop must be allowed to mount that folder. The source is mounted read-only.

The MCP endpoint is `http://127.0.0.1:8000/mcp`. The Compose deployment is for local evaluation. Before a wider deployment, add an authenticated reverse proxy, TLS, user/service authorization, rate limiting, audit logging appropriate to the indexed content, secret injection, monitoring, and encrypted backups.

## Upgrade procedure

1. Read `CHANGELOG.md` for schema or provider changes.
2. Back up the SQLite state and Qdrant collection.
3. Install the new package in a fresh virtual environment or build a new container image.
4. Run `doctor` and the labeled retrieval evaluation against a staging collection.
5. Switch MCP clients after checks pass.

Do not rebuild a production collection solely because a package version changed. Rebuild only when the compatibility check requires it or an intended indexing change needs new vectors.
