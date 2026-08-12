"""Provider contracts shared by built-in and bring-your-own adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    provider: str
    model: str
    license: str = "unknown"
    local: bool = False


@dataclass(frozen=True, slots=True)
class SparseValues:
    indices: list[int]
    values: list[float]


@runtime_checkable
class Embedder(Protocol):
    @property
    def info(self) -> ProviderInfo: ...

    @property
    def dimension(self) -> int: ...

    @property
    def fingerprint(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class LexicalEncoder(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> list[SparseValues]: ...

    def embed_query(self, text: str) -> SparseValues: ...


@runtime_checkable
class Reranker(Protocol):
    @property
    def info(self) -> ProviderInfo: ...

    def score(self, query: str, documents: list[str]) -> list[float]: ...
