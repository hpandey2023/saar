"""Structure-aware text chunking with stable citations."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from .config import IndexSettings
from .models import DocumentChunk, SourceDocument

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
SYMBOL_RE = re.compile(
    r"^\s*(?:(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:def|class|function)|"
    r"(?:public|private|protected|static|final|abstract|async)\s+(?:class|function)|fn)"
    r"\s+([A-Za-z_$][\w$]*)[^\n]*",
    re.MULTILINE,
)
POINT_NAMESPACE = uuid.UUID("b137c0ef-cff9-5d3c-81f3-0f4ac560f290")


@dataclass(frozen=True, slots=True)
class _Section:
    label: str
    start: int
    end: int
    text: str


def _sections(document: SourceDocument) -> list[_Section]:
    pattern = SYMBOL_RE if document.content_type == "code" else HEADING_RE
    matches = list(pattern.finditer(document.text))
    if not matches:
        return [_Section(document.title, 0, len(document.text), document.text)]

    sections: list[_Section] = []
    if matches[0].start() > 0 and document.text[: matches[0].start()].strip():
        sections.append(
            _Section(document.title, 0, matches[0].start(), document.text[: matches[0].start()])
        )
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document.text)
        label = (
            match.group(2).strip() if document.content_type != "code" else match.group(1).strip()
        )
        sections.append(
            _Section(label[:300], match.start(), end, document.text[match.start() : end])
        )
    return sections


def _split_window(text: str, start: int, maximum: int, overlap: int) -> list[tuple[int, int, str]]:
    if len(text) <= maximum:
        return [(start, start + len(text), text)]
    chunks: list[tuple[int, int, str]] = []
    cursor = 0
    while cursor < len(text):
        hard_end = min(cursor + maximum, len(text))
        end = hard_end
        if hard_end < len(text):
            candidates = [
                text.rfind("\n\n", cursor + maximum // 2, hard_end),
                text.rfind("\n", cursor + maximum // 2, hard_end),
                text.rfind(". ", cursor + maximum // 2, hard_end),
                text.rfind(" ", cursor + maximum // 2, hard_end),
            ]
            boundary = max(candidates)
            if boundary > cursor:
                end = boundary + (2 if text[boundary : boundary + 2] in {"\n\n", ". "} else 1)
        value = text[cursor:end].strip()
        if value:
            leading = len(text[cursor:end]) - len(text[cursor:end].lstrip())
            absolute_start = start + cursor + leading
            chunks.append((absolute_start, absolute_start + len(value), value))
        if end >= len(text):
            break
        cursor = max(cursor + 1, end - overlap)
    return chunks


def chunk_document(document: SourceDocument, settings: IndexSettings) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for section in _sections(document):
        for char_start, char_end, text in _split_window(
            section.text,
            section.start,
            settings.max_chunk_chars,
            settings.chunk_overlap_chars,
        ):
            if len(text) < settings.min_chunk_chars:
                continue
            chunk_index = len(chunks)
            identity = f"{document.source_key}\x1f{document.sha256}\x1f{chunk_index}\x1f{text}"
            point_id = str(uuid.uuid5(POINT_NAMESPACE, identity))
            section_fragment = re.sub(r"\s+", "-", section.label.strip().lower())[:100].strip("-")
            citation = (
                f"{document.source_key}#{section_fragment}"
                if section_fragment
                else document.source_key
            )
            chunks.append(
                DocumentChunk(
                    point_id=point_id,
                    scope=document.scope,
                    source_file=document.source_file,
                    source_key=document.source_key,
                    content_type=document.content_type,
                    title=document.title,
                    section=section.label,
                    citation=citation,
                    status=document.status,
                    text=text,
                    revision=document.sha256,
                    chunk_index=chunk_index,
                    char_start=char_start,
                    char_end=char_end,
                    metadata=document.metadata,
                )
            )
    return chunks
