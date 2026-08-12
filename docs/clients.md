# MCP client setup

Index the corpus and pass a successful `saar doctor` before adding the MCP server to a client. Always use absolute executable and configuration paths. The MCP process writes protocol messages to standard output, so wrapper scripts must not print banners there.

## Generic JSON configuration

Many desktop clients accept this shape:

```json
{
  "mcpServers": {
    "saar": {
      "command": "/absolute/path/.venv/bin/saar",
      "args": ["--config", "/absolute/path/saar.yaml", "serve"]
    }
  }
}
```

Reload MCP servers after changing the configuration. Tool prefixes vary by client; inspect the client's tool list rather than relying on a specific generated prefix.

## Agent instruction

Add a short instruction to agents allowed to search the corpus:

```text
For questions that may depend on indexed resources or source code, call search_context first.
Cite the returned citation values. If evidence is incomplete or conflicting, say so.
Record feedback only when a person explicitly verifies the result.
```

MCP makes tools available; it does not guarantee an agent will call them. Client instructions and an evaluation set are both necessary when retrieval should be the default for corpus-dependent work.

`search_context` accepts optional `document_types`, `dates`, and `people` lists. These match YAML frontmatter exactly (with case normalization where appropriate), which lets an agent prefer canonical evidence, target a specific meeting date, or narrow a search to material associated with a named person.
