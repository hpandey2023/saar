from __future__ import annotations

from conftest import FakeEmbedder, FakeLexicalEncoder

from saar_mcp.engine import RetrievalEngine
from saar_mcp.evaluation import compare_reports, evaluate


def test_evaluation_and_comparison(settings):
    engine = RetrievalEngine(
        settings,
        embedder=FakeEmbedder(),
        lexical_encoder=FakeLexicalEncoder(),
    )
    try:
        engine.index()
        report = evaluate(
            engine,
            [
                {
                    "query": "When should a priority request be acknowledged?",
                    "expected_sources": ["request-handling.md"],
                },
                {
                    "query": "How is retry delay calculated?",
                    "expected_sources": ["retry_policy.py"],
                },
            ],
            limit=4,
        )
        assert report["summary"]["hit_rate@4"] == 1.0
        comparison = compare_reports(report, report)
        assert comparison["metrics"]["mrr@4"]["delta"] == 0
    finally:
        engine.close()
