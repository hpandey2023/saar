"""Repeatable retrieval evaluation using representative labeled questions."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from .engine import RetrievalEngine


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load_questions(path: Path) -> list[dict[str, Any]]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read evaluation file {path}: {exc}") from exc
    questions = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(questions, list) or not questions:
        raise ValueError("evaluation file must contain a non-empty 'queries' list")
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict) or not str(question.get("query", "")).strip():
            raise ValueError(f"evaluation query {index} requires non-empty 'query'")
        expected = question.get("expected_sources") or question.get("expected_citations")
        if not isinstance(expected, list) or not expected:
            raise ValueError(
                f"evaluation query {index} requires expected_sources or expected_citations"
            )
    return questions


def evaluate(
    engine: RetrievalEngine, questions: list[dict[str, Any]], *, limit: int
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    for question in questions:
        expected = [
            str(item).casefold()
            for item in (question.get("expected_sources") or question.get("expected_citations"))
        ]
        response = engine.search(
            str(question["query"]),
            scopes=question.get("scopes"),
            content_types=question.get("content_types"),
            limit=limit,
        )
        rank = None
        for position, hit in enumerate(response.results, start=1):
            searchable = f"{hit.scope}/{hit.source_file} {hit.citation}".casefold()
            if any(value in searchable for value in expected):
                rank = position
                break
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        latencies.append(response.duration_ms)
        rows.append(
            {
                "query": question["query"],
                "expected": expected,
                "matched_rank": rank,
                "top_citations": [item.citation for item in response.results],
                "duration_ms": round(response.duration_ms, 2),
            }
        )
    hits = sum(value > 0 for value in reciprocal_ranks)
    return {
        "summary": {
            "queries": len(rows),
            f"hit_rate@{limit}": round(hits / len(rows), 4),
            f"mrr@{limit}": round(statistics.fmean(reciprocal_ranks), 4),
            "latency_ms_p50": round(_percentile(latencies, 0.5), 2),
            "latency_ms_p95": round(_percentile(latencies, 0.95), 2),
        },
        "provider": engine.status()["embedding"],
        "results": rows,
    }


def compare_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_summary = left.get("summary", {})
    right_summary = right.get("summary", {})
    metrics = sorted(set(left_summary) & set(right_summary))
    comparison: dict[str, Any] = {}
    for metric in metrics:
        before = left_summary[metric]
        after = right_summary[metric]
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            comparison[metric] = {
                "left": before,
                "right": after,
                "delta": round(after - before, 4),
            }
    return {
        "left_provider": left.get("provider"),
        "right_provider": right.get("provider"),
        "metrics": comparison,
    }


def load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read report {path}: {exc}") from exc
    if not isinstance(value, dict) or "summary" not in value:
        raise ValueError(f"not a Saar evaluation report: {path}")
    return value
