"""Validated YAML configuration and secret resolution."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .exceptions import ConfigurationError

ENV_REF = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")

DEFAULT_EXCLUDES = [
    "**/.git/**",
    "**/.svn/**",
    "**/.hg/**",
    "**/.venv/**",
    "**/venv/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/dist/**",
    "**/build/**",
    "**/target/**",
    "**/.cache/**",
    "**/archive/**",
    "**/backups/**",
    "**/.env",
    "**/.env.*",
    "**/*credential*",
    "**/*secret*",
    "**/*token*",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/*.pfx",
    "**/id_rsa*",
]

DEFAULT_EXTENSIONS = [
    ".md",
    ".mdx",
    ".txt",
    ".rst",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".sql",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".css",
    ".scss",
    ".sh",
    ".ps1",
]


class StrictModel(BaseModel):
    """Reject misspelled configuration keys instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid")


class EmbeddingSettings(StrictModel):
    provider: Literal["fastembed", "ollama", "openai_compatible", "voyage", "python"] = "fastembed"
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimensions: int | None = Field(default=None, ge=8, le=65536)
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    query_prefix: str = ""
    document_prefix: str = ""
    contextual_group_size: int = Field(default=32, ge=1, le=256)
    class_path: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provider_fields(self) -> EmbeddingSettings:
        if self.provider in {"ollama", "openai_compatible"} and not self.base_url:
            default = "http://localhost:11434" if self.provider == "ollama" else None
            if default:
                self.base_url = default
            else:
                raise ValueError("openai_compatible embedding requires base_url")
        if self.provider == "voyage" and not self.api_key_env:
            self.api_key_env = "VOYAGE_API_KEY"
        if self.provider == "python" and not self.class_path:
            raise ValueError("python embedding provider requires class_path='module:Class'")
        return self


class RerankerSettings(StrictModel):
    provider: Literal["none", "fastembed", "voyage", "python"] = "none"
    model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    api_key_env: str | None = None
    top_k: int = Field(default=30, ge=1, le=200)
    min_score: float | None = None
    class_path: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provider_fields(self) -> RerankerSettings:
        if self.provider == "voyage" and not self.api_key_env:
            self.api_key_env = "VOYAGE_API_KEY"
        if self.provider == "python" and not self.class_path:
            raise ValueError("python reranker requires class_path='module:Class'")
        return self


class LexicalSettings(StrictModel):
    enabled: bool = True
    model: str = "Qdrant/bm25"
    language: str = "english"
    disable_stemmer: bool = True


class QdrantSettings(StrictModel):
    mode: Literal["embedded", "server"] = "embedded"
    path: Path = Path(".saar/qdrant")
    url: str = "http://localhost:6333"
    api_key_env: str | None = None
    collection: str = "saar"
    prefer_grpc: bool = False

    @field_validator("collection")
    @classmethod
    def validate_collection(cls, value: str) -> str:
        if not COLLECTION_RE.fullmatch(value):
            raise ValueError("collection must contain only letters, numbers, '.', '_' or '-'")
        return value


class IndexSettings(StrictModel):
    state_path: Path = Path(".saar/state.sqlite3")
    extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_EXTENSIONS))
    max_file_bytes: int = Field(default=2_000_000, ge=1_000, le=100_000_000)
    max_chunk_chars: int = Field(default=2_400, ge=300, le=50_000)
    chunk_overlap_chars: int = Field(default=240, ge=0, le=10_000)
    min_chunk_chars: int = Field(default=80, ge=1, le=5_000)
    batch_size: int = Field(default=64, ge=1, le=512)
    follow_symlinks: bool = False

    @field_validator("extensions")
    @classmethod
    def normalize_extensions(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            extension = value.lower().strip()
            if not extension.startswith("."):
                extension = f".{extension}"
            normalized.append(extension)
        return sorted(set(normalized))

    @model_validator(mode="after")
    def validate_overlap(self) -> IndexSettings:
        if self.chunk_overlap_chars >= self.max_chunk_chars:
            raise ValueError("chunk_overlap_chars must be smaller than max_chunk_chars")
        return self


class RetrievalSettings(StrictModel):
    default_limit: int = Field(default=8, ge=1, le=50)
    max_limit: int = Field(default=20, ge=1, le=100)
    candidate_limit: int = Field(default=40, ge=5, le=500)
    dense_weight: float = Field(default=1.0, gt=0, le=10)
    lexical_weight: float = Field(default=1.0, gt=0, le=10)
    max_per_source: int = Field(default=3, ge=1, le=20)
    result_text_chars: int = Field(default=3_500, ge=200, le=50_000)
    exclude_statuses: list[str] = Field(
        default_factory=lambda: ["archived", "obsolete", "superseded"]
    )

    @model_validator(mode="after")
    def validate_limits(self) -> RetrievalSettings:
        if self.default_limit > self.max_limit:
            raise ValueError("default_limit cannot exceed max_limit")
        if self.candidate_limit < self.max_limit:
            raise ValueError("candidate_limit must be at least max_limit")
        return self


class FeedbackSettings(StrictModel):
    enabled: bool = True
    require_verified: bool = True


class SourceSettings(StrictModel):
    scope: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    path: Path
    include: list[str] = Field(default_factory=lambda: ["**/*"])
    exclude: list[str] = Field(default_factory=list)


class ServerSettings(StrictModel):
    name: str = "saar"
    instructions: str = (
        "Use search_context for questions that depend on indexed resources or source code. "
        "Cite returned citations. If evidence is incomplete or conflicting, say so rather than "
        "guessing. Record feedback only after a human verifies the result."
    )


class Settings(StrictModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    lexical: LexicalSettings = Field(default_factory=LexicalSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    feedback: FeedbackSettings = Field(default_factory=FeedbackSettings)
    sources: list[SourceSettings] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_scopes(self) -> Settings:
        scopes = [source.scope for source in self.sources]
        if len(scopes) != len(set(scopes)):
            raise ValueError("source scope names must be unique")
        if (
            self.embedding.provider == "voyage"
            and self.embedding.model.startswith("voyage-context-")
            and self.index.chunk_overlap_chars != 0
        ):
            raise ValueError("Voyage contextual embeddings require index.chunk_overlap_chars: 0")
        return self


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if resolved is None:
            missing.append(name)
            return match.group(0)
        return resolved

    expanded = ENV_REF.sub(replace, value)
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise ConfigurationError(f"missing environment variable(s): {names}")
    return expanded


def load_settings(path: str | Path) -> Settings:
    """Load, expand, validate, and path-resolve a YAML configuration file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        raw = _expand_environment(raw)
        settings = Settings.model_validate(raw)
    except ConfigurationError:
        raise
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ConfigurationError(f"invalid configuration {config_path}: {exc}") from exc

    base = config_path.parent
    settings.qdrant.path = _resolve_path(settings.qdrant.path, base)
    settings.index.state_path = _resolve_path(settings.index.state_path, base)
    for source in settings.sources:
        source.path = _resolve_path(source.path, base)
    return settings


def _resolve_path(path: Path, base: Path) -> Path:
    expanded = path.expanduser()
    return (base / expanded).resolve() if not expanded.is_absolute() else expanded.resolve()


def read_secret(env_name: str | None, *, required: bool) -> str | None:
    """Resolve a secret from the environment without ever logging its value."""
    if not env_name:
        if required:
            raise ConfigurationError("provider requires api_key_env")
        return None
    value = os.environ.get(env_name)
    if not value and required:
        raise ConfigurationError(f"required secret environment variable is not set: {env_name}")
    return value


def effective_excludes(source: SourceSettings) -> list[str]:
    return [*DEFAULT_EXCLUDES, *source.exclude]
