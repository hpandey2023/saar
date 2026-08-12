# Contributing

## Set up

Use Python 3.11 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Run the same checks as continuous integration:

```bash
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/pytest --cov=saar_mcp --cov-report=term-missing
.venv/bin/python -m build
```

Tests must not use real source documents, credentials, paid APIs, or network services. Use synthetic fixtures and injected providers.

## Changes

Keep changes focused. Update documentation and example configuration when behavior changes. New provider adapters must:

- keep credentials in environment variables;
- expose a stable model fingerprint and vector dimension;
- preserve query/document input semantics;
- convert provider failures into actionable `ProviderError` messages;
- include offline tests with a fake or mocked client;
- document model licensing and whether text leaves the machine.

Do not make autonomous feedback learn from model output. Only explicitly human-verified signals may change future ranking.

## Pull requests

Explain the user-visible change, security and migration impact, and verification performed. A model change that invalidates an existing collection must include migration guidance.
