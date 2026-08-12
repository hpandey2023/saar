"""Allowlist-based, path-safe source discovery."""

from __future__ import annotations

import fnmatch
import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from .config import IndexSettings, SourceSettings, effective_excludes
from .exceptions import ConfigurationError
from .models import SourceDocument

CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".sql",
    ".css",
    ".scss",
    ".sh",
    ".ps1",
}


def _matches(relative: str, pattern: str) -> bool:
    normalized = pattern.replace("\\", "/").lstrip("./")
    candidates = {normalized}
    if normalized.startswith("**/"):
        candidates.add(normalized[3:])
    return any(fnmatch.fnmatchcase(relative, candidate) for candidate in candidates)


def _matches_any(relative: str, patterns: list[str]) -> bool:
    return any(_matches(relative, pattern) for pattern in patterns)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _has_symlink_component(path: Path, root: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    closing = text.find("\n---\n", 4)
    if closing < 0:
        return {}, text
    try:
        metadata = yaml.safe_load(text[4:closing]) or {}
    except yaml.YAMLError:
        return {}, text
    return (metadata if isinstance(metadata, dict) else {}), text[closing + 5 :]


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip()[:200] for item in values if str(item).strip()]


def scan_source(source: SourceSettings, settings: IndexSettings) -> Iterator[SourceDocument]:
    """Yield readable, allowlisted files rooted under one configured source."""
    root = source.path.resolve()
    if not root.is_dir():
        raise ConfigurationError(f"source path is not a directory: {source.path}")

    excludes = effective_excludes(source)
    extensions = set(settings.extensions)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if not _matches_any(relative, source.include) or _matches_any(relative, excludes):
            continue
        if path.suffix.lower() not in extensions:
            continue
        if not settings.follow_symlinks and _has_symlink_component(path, root):
            continue
        resolved = path.resolve()
        if not _is_within(resolved, root):
            continue
        try:
            stat = resolved.stat()
            if stat.st_size > settings.max_file_bytes:
                continue
            raw = resolved.read_bytes()
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        metadata, body = _frontmatter(text)
        content_type = "code" if path.suffix.lower() in CODE_EXTENSIONS else "document"
        fallback_title = path.stem.replace("-", " ").replace("_", " ").strip().title()
        title = str(metadata.get("title") or _title_from_text(body, fallback_title))[:300]
        status = str(metadata.get("status") or "active").strip().lower()[:100]
        retrieval_metadata = {
            "document_type": str(metadata.get("document_type") or content_type)
            .strip()
            .lower()[:100],
            "date": str(metadata.get("date") or "").strip()[:40],
            "people": _string_list(metadata.get("people")),
            "tags": _string_list(metadata.get("tags")),
        }
        digest = hashlib.sha256(raw).hexdigest()
        yield SourceDocument(
            scope=source.scope,
            source_file=relative,
            source_key=f"{source.scope}/{relative}",
            content_type=content_type,
            title=title,
            status=status,
            text=body,
            sha256=digest,
            size_bytes=stat.st_size,
            modified_ns=stat.st_mtime_ns,
            metadata=retrieval_metadata,
        )
