# Operations

## Routine checks

Run these after changing configuration, provider credentials, or model infrastructure:

```bash
saar doctor
saar index
saar search "known validation question"
```

`doctor` reports provider metadata, dimensions, source availability, state counts, and Qdrant health without returning secret values.

## Index updates

`index` hashes approved files and embeds only changes. It also removes deleted files from the collection. `watch` performs the same operation on a polling interval.

Use `index --force` after changing chunking behavior without changing the collection fingerprint. Use a new collection or `rebuild --yes` after changing an embedding provider, model, dimension, or contextual configuration.

## Backups

For embedded mode, stop MCP and index processes before copying the Qdrant directory and SQLite database. For server mode, use Qdrant's supported snapshot mechanism and copy SQLite at the same logical time. Encrypt backups according to the source corpus classification.

## Failure messages

- `collection ... is incompatible`: model or schema differs; select a new collection or explicitly rebuild.
- `required secret environment variable is not set`: export the named variable in the MCP process environment.
- `source path is not a directory`: correct the resolved source path and host/container mount.
- `Connection closed` in an MCP client: run the exact configured command directly; it usually exited before MCP initialization.
- embedded storage already accessed: stop the other process or move to Qdrant server mode.

## Quality drift

Keep a versioned evaluation set without confidential answer text when possible. Run it before and after provider, chunk, filter, or ranking changes. Human feedback helps exact repeated questions, but it is not a substitute for corpus-wide evaluation and never modifies the underlying embedding model.
