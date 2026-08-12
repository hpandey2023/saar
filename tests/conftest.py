from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from saar_mcp.config import Settings
from saar_mcp.providers.base import ProviderInfo, SparseValues


class FakeEmbedder:
    dimension = 32
    fingerprint = "fake-embedder-v1"
    info = ProviderInfo(provider="fake", model="hashed-words", license="MIT", local=True)

    @staticmethod
    def _one(text: str) -> list[float]:
        vector = [0.0] * FakeEmbedder.dimension
        for word in re.findall(r"[a-z0-9]+", text.casefold()):
            index = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % len(vector)
            vector[index] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)


class FakeLexicalEncoder:
    fingerprint = "fake-lexical-v1"

    @staticmethod
    def _one(text: str) -> SparseValues:
        counts: dict[int, float] = {}
        for word in re.findall(r"[a-z0-9]+", text.casefold()):
            index = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 100_000 + 1
            counts[index] = counts.get(index, 0.0) + 1.0
        return SparseValues(indices=sorted(counts), values=[counts[key] for key in sorted(counts)])

    def embed_documents(self, texts: list[str]) -> list[SparseValues]:
        return [self._one(text) for text in texts]

    def embed_query(self, text: str) -> SparseValues:
        return self._one(text)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "request-handling.md").write_text(
        """---
title: Request-handling guide
status: active
document_type: process-guide
date: 2026-08-12
people:
  - Alex Example
tags:
  - service-level
---
# Priority response

A high-priority request must be acknowledged within two business hours and assigned to a named owner.

# Escalation

If no owner accepts the request within four business hours, escalate it to the duty lead.
""",
        encoding="utf-8",
    )
    (root / "security.md").write_text(
        """# Access control

All production access requires hardware-backed multi-factor authentication and a ticket.
Never place credentials in source files or search-index configuration.
""",
        encoding="utf-8",
    )
    (root / "retry_policy.py").write_text(
        """def calculate_retry_delay(attempt: int, base_seconds: int = 2) -> int:
    \"\"\"Return the exponential delay before the next request attempt.\"\"\"
    return base_seconds * (2**attempt)
""",
        encoding="utf-8",
    )
    (root / ".env").write_text("API_TOKEN=never-index-this\n", encoding="utf-8")
    archive = root / "archive"
    archive.mkdir()
    (archive / "obsolete.md").write_text(
        "# Old rule\n\nHigh-priority requests may wait five business days.\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def settings(tmp_path: Path, corpus: Path) -> Settings:
    return Settings.model_validate(
        {
            "embedding": {"provider": "fastembed"},
            "reranker": {"provider": "none"},
            "lexical": {"enabled": True},
            "qdrant": {
                "mode": "embedded",
                "path": tmp_path / "qdrant",
                "collection": "test-context",
            },
            "index": {
                "state_path": tmp_path / "state.sqlite3",
                "max_chunk_chars": 500,
                "chunk_overlap_chars": 50,
                "min_chunk_chars": 20,
                "batch_size": 8,
            },
            "retrieval": {
                "default_limit": 4,
                "max_limit": 10,
                "candidate_limit": 12,
                "max_per_source": 2,
            },
            "sources": [{"scope": "operations", "path": corpus}],
        }
    )
