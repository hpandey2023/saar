"""Command-line interface for setup, indexing, evaluation, and MCP serving."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from . import __version__
from .config import DEFAULT_EXCLUDES, Settings, load_settings
from .engine import RetrievalEngine
from .evaluation import compare_reports, evaluate, load_questions, load_report
from .exceptions import SaarError
from .server import build_server

DEFAULT_CONFIG = "saar.yaml"


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _configure_model_environment(*, quiet_progress: bool = False) -> None:
    """Keep model caches writable and stdio protocol output free of progress bars."""
    default_cache = Path(tempfile.gettempdir()) / "fastembed_cache"
    fastembed_cache = Path(os.environ.get("FASTEMBED_CACHE_PATH", default_cache))
    os.environ.setdefault("HF_XET_CACHE", str(fastembed_cache / "xet"))
    if quiet_progress:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _config_path(arguments: argparse.Namespace) -> Path:
    return Path(arguments.config or os.environ.get("SAAR_CONFIG", DEFAULT_CONFIG)).expanduser()


def _load(arguments: argparse.Namespace) -> Settings:
    return load_settings(_config_path(arguments))


def _split_csv(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


def command_init(arguments: argparse.Namespace) -> int:
    destination = _config_path(arguments)
    if destination.exists() and not arguments.force:
        raise ValueError(f"refusing to overwrite {destination}; pass --force to replace it")
    source = Path(arguments.source).expanduser().resolve()
    config = {
        "server": {"name": arguments.name},
        "embedding": {
            "provider": "fastembed",
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
        "reranker": {"provider": "none"},
        "lexical": {"enabled": True, "model": "Qdrant/bm25"},
        "qdrant": {
            "mode": "embedded",
            "path": ".saar/qdrant",
            "collection": arguments.name,
        },
        "index": {"state_path": ".saar/state.sqlite3"},
        "retrieval": {"default_limit": 8, "candidate_limit": 40},
        "feedback": {"enabled": True, "require_verified": True},
        "sources": [{"scope": arguments.scope, "path": str(source)}],
    }
    Settings.model_validate(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _json(
        {
            "created": str(destination.resolve()),
            "next": [
                f"saar --config {destination} doctor",
                f"saar --config {destination} index",
                f"saar --config {destination} serve",
            ],
        }
    )
    return 0


def command_doctor(arguments: argparse.Namespace) -> int:
    settings = _load(arguments)
    result: dict[str, Any] = {
        "configuration": "valid",
        "config": str(_config_path(arguments).resolve()),
        "sources": [
            {
                "scope": source.scope,
                "exists": source.path.is_dir(),
                "path": str(source.path),
            }
            for source in settings.sources
        ],
        "security_defaults": {
            "default_exclusion_rules": len(DEFAULT_EXCLUDES),
            "feedback_requires_human_verification": settings.feedback.require_verified,
            "absolute_paths_in_search_results": False,
        },
    }
    if arguments.config_only:
        _json(result)
        return 0 if all(item["exists"] for item in result["sources"]) else 1
    with RetrievalEngine(settings) as engine:
        engine.vector_store.ensure_collection()
        result["runtime"] = engine.status()
    result["ok"] = (
        all(item["exists"] for item in result["sources"])
        and result["runtime"]["qdrant"]["reachable"]
    )
    _json(result)
    return 0 if result["ok"] else 1


def command_index(arguments: argparse.Namespace) -> int:
    with RetrievalEngine(_load(arguments)) as engine:
        report = engine.index(scopes=_split_csv(arguments.scope), force=arguments.force)
    _json(report.as_dict())
    return 1 if report.errors else 0


def command_search(arguments: argparse.Namespace) -> int:
    settings = _load(arguments)
    with RetrievalEngine(settings) as engine:
        response = engine.search(
            arguments.query,
            scopes=_split_csv(arguments.scope),
            content_types=_split_csv(arguments.content_type),
            document_types=_split_csv(arguments.document_type),
            dates=_split_csv(arguments.date),
            people=_split_csv(arguments.person),
            limit=arguments.limit,
        )
    _json(response.as_dict(text_limit=settings.retrieval.result_text_chars))
    return 0


def command_watch(arguments: argparse.Namespace) -> int:
    engine = RetrievalEngine(_load(arguments))
    interval = arguments.interval
    print(
        f"Watching configured sources every {interval:g}s; press Ctrl-C to stop.", file=sys.stderr
    )
    try:
        while True:
            report = engine.index(scopes=_split_csv(arguments.scope))
            if report.changed_files or report.removed_files or report.errors:
                _json(report.as_dict())
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0
    finally:
        engine.close()


def command_serve(arguments: argparse.Namespace) -> int:
    settings = _load(arguments)
    server = build_server(settings)
    if arguments.transport == "stdio":
        server.run(transport="stdio", show_banner=False)
        return 0
    safe_hosts = {"127.0.0.1", "::1", "localhost"}
    if arguments.host not in safe_hosts and not arguments.allow_network:
        raise ValueError(
            "refusing to expose indexed content on a network interface; use a loopback host or "
            "pass --allow-network after placing authentication and TLS in front of the service"
        )
    server.run(
        transport="http",
        host=arguments.host,
        port=arguments.port,
        show_banner=False,
    )
    return 0


def command_rebuild(arguments: argparse.Namespace) -> int:
    if not arguments.yes:
        raise ValueError("rebuild deletes the configured collection; rerun with --yes")
    with RetrievalEngine(_load(arguments)) as engine:
        report = engine.rebuild()
    _json(report.as_dict())
    return 1 if report.errors else 0


def command_evaluate(arguments: argparse.Namespace) -> int:
    with RetrievalEngine(_load(arguments)) as engine:
        report = evaluate(
            engine,
            load_questions(Path(arguments.questions)),
            limit=arguments.limit,
        )
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _json(report)
    return 0


def command_compare(arguments: argparse.Namespace) -> int:
    _json(compare_reports(load_report(Path(arguments.left)), load_report(Path(arguments.right))))
    return 0


def command_providers(arguments: argparse.Namespace) -> int:
    del arguments
    try:
        from fastembed import TextEmbedding

        local_models = [
            {
                "model": item["model"],
                "dimensions": item.get("dim"),
                "license": item.get("license", "unknown"),
                "size_gb": item.get("size_in_GB"),
            }
            for item in TextEmbedding.list_supported_models()
        ]
    except Exception:
        local_models = []
    _json(
        {
            "built_in": [
                {
                    "provider": "fastembed",
                    "cost": "free/local",
                    "default_model": (
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    ),
                },
                {"provider": "ollama", "cost": "free/local"},
                {"provider": "openai_compatible", "cost": "depends on endpoint"},
                {"provider": "voyage", "cost": "paid API"},
                {"provider": "python", "cost": "bring your own adapter"},
            ],
            "fastembed_models": local_models,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saar",
        description="Local-first hybrid context retrieval for MCP-compatible agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help=f"configuration file (default: {DEFAULT_CONFIG})")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a safe starter configuration")
    init.add_argument("--source", default=".", help="folder to index")
    init.add_argument("--scope", default="project", help="short source-scope name")
    init.add_argument("--name", default="saar", help="server and collection name")
    init.add_argument("--force", action="store_true", help="replace an existing config")
    init.set_defaults(handler=command_init)

    doctor = commands.add_parser("doctor", help="validate configuration and provider runtime")
    doctor.add_argument("--config-only", action="store_true", help="skip model and Qdrant loading")
    doctor.set_defaults(handler=command_doctor)

    index = commands.add_parser("index", help="incrementally index configured sources")
    index.add_argument("--scope", action="append", help="scope or comma-separated scopes")
    index.add_argument("--force", action="store_true", help="re-index unchanged files")
    index.set_defaults(handler=command_index)

    search = commands.add_parser("search", help="run a retrieval query")
    search.add_argument("query")
    search.add_argument("--scope", action="append", help="scope or comma-separated scopes")
    search.add_argument(
        "--content-type", action="append", help="document, code, or comma-separated values"
    )
    search.add_argument(
        "--document-type",
        action="append",
        help="frontmatter document type or comma-separated values",
    )
    search.add_argument(
        "--date", action="append", help="exact frontmatter ISO date or comma-separated values"
    )
    search.add_argument(
        "--person", action="append", help="exact frontmatter person name or comma-separated values"
    )
    search.add_argument("--limit", type=int)
    search.set_defaults(handler=command_search)

    watch = commands.add_parser("watch", help="poll sources and index changes")
    watch.add_argument("--scope", action="append", help="scope or comma-separated scopes")
    watch.add_argument("--interval", type=float, default=10, help="seconds between scans")
    watch.set_defaults(handler=command_watch)

    serve = commands.add_parser("serve", help="start the MCP server")
    serve.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--allow-network", action="store_true")
    serve.set_defaults(handler=command_serve)

    rebuild = commands.add_parser("rebuild", help="replace the configured index")
    rebuild.add_argument("--yes", action="store_true", help="confirm collection replacement")
    rebuild.set_defaults(handler=command_rebuild)

    evaluation = commands.add_parser("evaluate", help="measure retrieval against labeled queries")
    evaluation.add_argument("questions", help="YAML evaluation set")
    evaluation.add_argument("--limit", type=int, default=8)
    evaluation.add_argument("--output", help="optional JSON report path")
    evaluation.set_defaults(handler=command_evaluate)

    compare = commands.add_parser("compare", help="compare two evaluation JSON reports")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.set_defaults(handler=command_compare)

    providers = commands.add_parser("providers", help="list provider choices and local models")
    providers.set_defaults(handler=command_providers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    _configure_model_environment(
        quiet_progress=arguments.command == "serve" and arguments.transport == "stdio"
    )
    try:
        return int(arguments.handler(arguments))
    except (SaarError, ValidationError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
