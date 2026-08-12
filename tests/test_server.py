from __future__ import annotations

import asyncio

from conftest import FakeEmbedder, FakeLexicalEncoder
from fastmcp import Client

from saar_mcp.engine import RetrievalEngine
from saar_mcp.server import build_server


def test_mcp_tools_are_callable_in_process(settings):
    engine = RetrievalEngine(
        settings,
        embedder=FakeEmbedder(),
        lexical_encoder=FakeLexicalEncoder(),
    )
    engine.index()
    server = build_server(settings, engine=engine)

    async def exercise() -> None:
        async with Client(server) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools}
            assert names == {
                "get_status",
                "list_sources",
                "record_search_feedback",
                "search_context",
            }
            search = next(tool for tool in tools if tool.name == "search_context")
            assert search.annotations is not None
            assert search.annotations.readOnlyHint is True
            result = await client.call_tool(
                "search_context",
                {
                    "query": "priority request acknowledgement",
                    "document_types": ["process-guide"],
                    "dates": ["2026-08-12"],
                    "people": ["Alex Example"],
                    "limit": 2,
                },
            )
            assert result.data["result_count"] >= 1
            assert result.data["results"][0]["citation"].startswith("operations/")
            assert result.data["results"][0]["document_type"] == "process-guide"

    try:
        asyncio.run(exercise())
    finally:
        engine.close()
