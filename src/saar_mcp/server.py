"""FastMCP tool surface for retrieving context from configured resources."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from .config import Settings
from .engine import RetrievalEngine


def build_server(settings: Settings, *, engine: RetrievalEngine | None = None) -> FastMCP:
    retrieval = engine or RetrievalEngine(settings)
    server = FastMCP(name=settings.server.name, instructions=settings.server.instructions)

    @server.tool(
        annotations={
            "title": "Search indexed resources and code",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    def search_context(
        query: str,
        scopes: list[str] | None = None,
        content_types: list[str] | None = None,
        document_types: list[str] | None = None,
        dates: list[str] | None = None,
        people: list[str] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Search indexed context. Cite each result's `citation` in the final answer.

        Args:
            query: A specific natural-language question or search phrase.
            scopes: Optional configured source scopes. Omit to search every scope.
            content_types: Optional `document` and/or `code` filter.
            document_types: Optional frontmatter types, such as `transcript-evidence`.
            dates: Optional exact ISO dates from frontmatter, such as `2026-08-12`.
            people: Optional exact attendee or author names from frontmatter.
            limit: Maximum results to return, bounded by server configuration.
        """
        response = retrieval.search(
            query,
            scopes=scopes,
            content_types=content_types,
            document_types=document_types,
            dates=dates,
            people=people,
            limit=limit,
        )
        return response.as_dict(text_limit=settings.retrieval.result_text_chars)

    @server.tool(
        annotations={
            "title": "List context sources",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    def list_sources() -> dict[str, Any]:
        """List configured scopes and aggregate indexed-file counts."""
        return retrieval.list_sources()

    @server.tool(
        annotations={
            "title": "Get retrieval status",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    def get_status() -> dict[str, Any]:
        """Report providers, index counts, and Qdrant health without returning secrets."""
        return retrieval.status()

    if settings.feedback.enabled:

        @server.tool(
            annotations={
                "title": "Record human-verified search feedback",
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            }
        )
        def record_search_feedback(
            query_id: str,
            point_id: str,
            verdict: str,
            human_verified: bool,
            note: str = "",
        ) -> dict[str, Any]:
            """Record relevant/irrelevant feedback only after a person verifies the result.

            This never trains a model or learns from an agent's unverified output. For an exact
            repeated query, verified relevant results are promoted and irrelevant results hidden.
            """
            return retrieval.record_feedback(
                query_id,
                point_id,
                verdict,
                human_verified=human_verified,
                note=note,
            )

    return server
