# Saar

**Connect every agent to the knowledge already in your files.**

Saar (सार, “essence”) is a retrieval-augmentation layer for AI agents. It indexes selected folders containing transcripts, discussions, documents, configuration, and code, then returns only the sections relevant to the agent's current task. Those sections enter the agent's context through the open [Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro), with citations back to the source files. Together with the connected agent, this creates a retrieval-augmented generation (RAG) workflow.

The retrieval pipeline combines semantic embeddings, which find related meaning, with BM25, which protects exact names, identifiers, and phrases. Weighted reciprocal rank fusion merges both result sets, and an optional reranker can refine their final order. This hybrid approach follows the same dense-plus-sparse retrieval pattern documented by [Qdrant](https://qdrant.tech/documentation/search/hybrid-queries/). The result is a small, relevant evidence set instead of an entire archive competing for the agent's context window.

We built Saar after our own transcripts and project discussions had grown across too many files to connect reliably by hand. Loading everything into an agent was wasteful, while choosing files manually was slow and easy to get wrong. Saar turns that accumulated work into reusable infrastructure: keep the source folders current, index them once, and any configured MCP-compatible agent can retrieve the evidence it needs through the same interface.

Your files remain the source of truth. Saar does not replace an agent's reasoning; it gives that reasoning a better starting point.

## What we verified

On August 12, 2026, we tested one agent model on 10 fixed questions under three retrieval conditions. Eight questions were answerable from the indexed sources and two were deliberately unanswerable controls.

| Result | Without indexed retrieval | Initial retrieval | Current retrieval |
|---|---:|---:|---:|
| Overall benchmark score | 20.0% | 76.22% | **94.75%** |
| Fact recall | 0% | 71.50% | **94.51%** |
| Supported answers, out of 8 | 0 | 7 | **8** |
| Incorrect assertions | — | 4 | **0** |
| Median end-to-end time | 5.36 s | 35.38 s | **24.37 s** |

The current pipeline won 5 questions and tied 5 against the initial retrieval version, with no losses. It reduced median end-to-end time by 31.1% compared with that first version. The run without retrieval was faster, but it recovered none of the 41 expected facts in the answerable questions; its 20% overall score came from correctly refusing the two unanswerable controls.

The overall score is the mean of the 10 case scores. Fact recall is the share of the 41 rubric-defined facts recovered across the eight answerable questions. The source CSV, adjudication, summary, and their saved SHA-256 hashes were checked again before publication; the figures above reproduced exactly.

This was an internal benchmark, not a universal guarantee. Each question was run once, scoring was manual and not blind, and the no-retrieval condition had no other access to the indexed sources. The test also cannot separate the individual contribution of index cleanup, BM25, fusion, reranking, version filtering, and the improved search-tool contract. The public aggregate record is in [benchmarks/agent-model-2026-08-12.json](benchmarks/agent-model-2026-08-12.json).

## Install

Python 3.11 or newer is required.

Keep the material agents should use in one folder, such as `resources/`. Any useful subfolder structure is fine.

```bash
cd /absolute/path/to/saar
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/saar init --source /absolute/path/to/resources --scope resources
.venv/bin/saar doctor
.venv/bin/saar index
```

Run `saar index` after changes, or keep `saar watch` running for automatic incremental updates.

Add Saar to an MCP-compatible agent using absolute paths:

```json
{
  "mcpServers": {
    "saar": {
      "command": "/absolute/path/to/saar/.venv/bin/saar",
      "args": [
        "--config",
        "/absolute/path/to/saar/saar.yaml",
        "serve"
      ]
    }
  }
}
```

Then give the agent this rule:

```text
When a task may depend on indexed files, call search_context first.
Cite the returned sources. If evidence is incomplete or conflicting, say so.
```

MCP makes the tool available; the rule makes its use consistent. Each agent or client must be configured separately.

## Included

- Hybrid semantic and BM25 retrieval with weighted rank fusion
- Structured chunking for documents, configuration, and source code
- Citations, source scopes, filters, and stale-content exclusion
- Incremental indexing and a folder watcher
- Embedded or server-hosted Qdrant
- Retrieval evaluation and guarded, human-verified feedback
- Stdio and loopback HTTP MCP transports

The default setup is free and local: FastEmbed, BM25, and embedded Qdrant. You can instead use Qwen3 through Ollama, Voyage, an OpenAI-compatible endpoint, or a custom Python provider. Remote providers receive the text they process, so review data requirements before enabling one.

## Useful commands

```bash
saar doctor              # check the setup
saar index               # index new and changed files
saar watch               # keep the index current
saar search "question"   # inspect retrieval directly
saar evaluate eval.yaml  # measure retrieval quality
saar serve               # run the MCP server
```

See [client setup](docs/clients.md), [provider choices](docs/providers.md), [evaluation](docs/evaluation.md), and [security](SECURITY.md) for details.

Apache-2.0 licensed.
