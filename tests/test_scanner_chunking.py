from __future__ import annotations

from saar_mcp.chunking import chunk_document
from saar_mcp.scanner import scan_source


def test_scanner_enforces_default_exclusions(settings):
    source = settings.sources[0]
    documents = list(scan_source(source, settings.index))
    paths = {item.source_file for item in documents}
    assert paths == {"request-handling.md", "retry_policy.py", "security.md"}
    assert all(not item.source_file.startswith("/") for item in documents)
    assert all("never-index-this" not in item.text for item in documents)


def test_markdown_chunks_have_stable_relative_citations(settings):
    document = next(
        item
        for item in scan_source(settings.sources[0], settings.index)
        if item.source_file == "request-handling.md"
    )
    first = chunk_document(document, settings.index)
    second = chunk_document(document, settings.index)
    assert [item.point_id for item in first] == [item.point_id for item in second]
    assert any("priority-response" in item.citation for item in first)
    assert all(item.citation.startswith("operations/request-handling.md#") for item in first)
    assert all(len(item.text) <= settings.index.max_chunk_chars for item in first)
    assert all("tmp" not in item.payload() for item in first)


def test_code_chunk_uses_symbol_as_section(settings):
    document = next(
        item
        for item in scan_source(settings.sources[0], settings.index)
        if item.source_file == "retry_policy.py"
    )
    chunks = chunk_document(document, settings.index)
    assert chunks[0].section == "calculate_retry_delay"
    assert chunks[0].content_type == "code"
