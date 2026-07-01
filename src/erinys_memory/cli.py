"""ERINYS JSON CLI.

The CLI is the automation-first entry point. MCP remains useful as an
agent-facing adapter, but both paths should call the same ERINYS functions.
"""

from __future__ import annotations

import argparse
import json
import os
from ._sqlite import sqlite3
import sys
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Literal, NoReturn, TypeAlias
from urllib.parse import quote

from pydantic import BaseModel, Field, ValidationError, model_validator


DEFAULT_PROJECT = "default"
DEFAULT_LIMIT = 10
DEFAULT_DREAM_COLLISIONS = 10
DEFAULT_PRUNE_THRESHOLD = 0.1
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
JSON_INDENT = 2
EXCERPT_LENGTH = 240
LOCAL_COMMANDS = {"stats", "health", "undistilled", "doctor"}
READONLY_COMMANDS = {"search", "recall", "context"}
WRITE_COMMANDS = {"save", "summary", "distill", "dream", "prune"}
SANDBOX_DB_WRITE_REQUIRED = "SANDBOX_DB_WRITE_REQUIRED"
DB_WRITE_UNAVAILABLE = "DB_WRITE_UNAVAILABLE"
READONLY_DB_MARKERS = (
    "attempt to write a readonly database",
    "readonly database",
    "operation not permitted",
    "unable to open database file",
)

ValidCommand: TypeAlias = Literal[
    "save",
    "summary",
    "get",
    "search",
    "recall",
    "context",
    "stats",
    "undistilled",
    "distill",
    "dream",
    "prune",
    "health",
    "doctor",
]
ValidType: TypeAlias = Literal[
    "manual",
    "decision",
    "architecture",
    "bugfix",
    "pattern",
    "config",
    "discovery",
    "learning",
    "anti_pattern",
    "meta_knowledge",
]
ValidDistillLevel: TypeAlias = Literal["concrete", "abstract", "meta"]


class CliArgs(BaseModel):
    command: ValidCommand
    title: str | None = None
    content: str | None = None
    content_file: str | None = None
    query: str | None = None
    observation_id: int | None = None
    type: ValidType = "manual"
    project: str | None = DEFAULT_PROJECT
    scope: str = "project"
    topic_key: str | None = None
    session_id: str | None = None
    metadata_json: str | None = None
    limit: int = Field(default=DEFAULT_LIMIT, ge=1)
    level: ValidDistillLevel = "abstract"
    max_collisions: int = Field(default=DEFAULT_DREAM_COLLISIONS, ge=1)
    threshold: float = Field(default=DEFAULT_PRUNE_THRESHOLD, ge=0.0)
    dry_run: bool = True
    readonly: bool = False
    deep: bool = False
    confirm_global: bool = False
    json_output: bool = True

    @model_validator(mode="after")
    def validate_command_payload(self) -> "CliArgs":
        if self.command in {"save", "summary"} and not self.payload_source:
            raise ValueError("--content or --content-file is required")
        if self.command == "save" and not self.title:
            raise ValueError("save requires --title")
        if self.command == "search" and not self.query:
            raise ValueError("search requires query")
        if self.command in {"get", "distill"} and self.observation_id is None:
            raise ValueError(f"{self.command} requires observation_id")
        if self.command == "prune" and not self.dry_run and not self.confirm_global:
            raise ValueError("prune --execute is a GLOBAL operation across all projects; re-run with --confirm-global")
        return self

    @property
    def payload_source(self) -> str | None:
        return self.content or self.content_file


CliArgs.model_rebuild(
    _types_namespace={
        "ValidCommand": ValidCommand,
        "ValidType": ValidType,
        "ValidDistillLevel": ValidDistillLevel,
    }
)


class JsonArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that keeps the --json contract on usage errors."""

    def error(self, message: str) -> NoReturn:
        write_json(error_result("USAGE", message))
        raise SystemExit(EXIT_USAGE)


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=JsonArgumentParser)
    add_save_parser(subparsers)
    add_summary_parser(subparsers)
    add_get_parser(subparsers)
    add_search_parser(subparsers)
    add_project_limit_parser(subparsers, "recall", readonly=True)
    add_project_limit_parser(subparsers, "context", readonly=True)
    add_project_parser(subparsers, "stats")
    add_undistilled_parser(subparsers)
    add_distill_parser(subparsers)
    add_dream_parser(subparsers)
    add_prune_parser(subparsers)
    add_health_parser(subparsers)
    add_doctor_parser(subparsers)
    return parser


def add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", dest="json_output", action="store_true", default=True)


def add_project_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=DEFAULT_PROJECT)


def add_content_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--content")
    parser.add_argument("--content-file")


def add_save_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("save")
    parser.add_argument("--title", required=True)
    add_content_args(parser)
    add_project_arg(parser)
    parser.add_argument("--type", default="manual")
    parser.add_argument("--scope", default="project")
    parser.add_argument("--topic-key")
    parser.add_argument("--session-id")
    parser.add_argument("--metadata-json")
    add_json_flag(parser)


def add_summary_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("summary")
    add_content_args(parser)
    add_project_arg(parser)
    parser.add_argument("--session-id")
    add_json_flag(parser)


def add_get_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("get")
    parser.add_argument("observation_id", type=int)
    add_json_flag(parser)


def add_readonly_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--readonly",
        action="store_true",
        default=False,
        help="read via SQLite mode=ro without server import / migration / audit log",
    )


def add_search_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("search")
    parser.add_argument("query")
    add_project_arg(parser)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    add_readonly_flag(parser)
    add_json_flag(parser)


def add_project_limit_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    readonly: bool = False,
) -> None:
    parser = subparsers.add_parser(name)
    add_project_arg(parser)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    if readonly:
        add_readonly_flag(parser)
    add_json_flag(parser)


def add_undistilled_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("undistilled", help="oldest undistilled observation IDs (read-only)")
    add_project_arg(parser)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    add_json_flag(parser)


def add_health_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("health")
    add_project_arg(parser)
    parser.add_argument(
        "--deep",
        action="store_true",
        default=False,
        help="also verify server import and run a search smoke test (authoritative check)",
    )
    add_json_flag(parser)


def add_doctor_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="diagnose the environment: Python, SQLite/sqlite-vec, embeddings, deps, DB",
    )
    add_project_arg(parser)
    parser.add_argument(
        "--deep",
        action="store_true",
        default=False,
        help="also import the MCP server and run a search smoke test",
    )
    add_json_flag(parser)


def add_project_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
) -> None:
    parser = subparsers.add_parser(name)
    add_project_arg(parser)
    add_json_flag(parser)


def add_distill_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("distill")
    parser.add_argument("observation_id", type=int)
    parser.add_argument("--level", choices=["concrete", "abstract", "meta"], default="abstract")
    add_json_flag(parser)


def add_dream_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("dream", help="GLOBAL operation: collides observations across ALL projects")
    parser.add_argument("--max-collisions", type=int, default=DEFAULT_DREAM_COLLISIONS)
    add_json_flag(parser)


def add_prune_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("prune", help="GLOBAL operation: prunes decayed observations across ALL projects")
    parser.add_argument("--threshold", type=float, default=DEFAULT_PRUNE_THRESHOLD)
    parser.add_argument("--execute", dest="dry_run", action="store_false", default=True)
    parser.add_argument(
        "--confirm-global",
        action="store_true",
        default=False,
        help="required with --execute: acknowledge deletion applies to ALL projects, not just --project",
    )
    add_json_flag(parser)


def parse_args(argv: list[str] | None = None) -> CliArgs:
    namespace = build_parser().parse_args(argv)
    return CliArgs.model_validate(vars(namespace))


def import_server() -> ModuleType:
    from erinys_memory import server

    server._CONFIG.db_backup_on_init = False
    return server


def read_content(args: CliArgs) -> str:
    if args.content is not None:
        return args.content
    if args.content_file == "-":
        return sys.stdin.read()
    return Path(str(args.content_file)).read_text(encoding="utf-8")


def parse_metadata(raw: str | None) -> dict[str, object] | None:
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--metadata-json must decode to an object")
    return dict(value)


def tool(server: object, name: str) -> Callable[..., object]:
    candidate = getattr(server, name)
    if not callable(candidate):
        raise RuntimeError(f"ERINYS tool is not callable: {name}")
    return candidate


def call_tool(server: object, name: str, **kwargs: object) -> dict[str, object]:
    result = tool(server, name)(**kwargs)
    if isinstance(result, dict):
        return dict(result)
    return error_result("INVALID_RESULT", f"{name} returned a non-object result")


def run_save(args: CliArgs, server: object | None) -> dict[str, object]:
    return call_tool(
        server,
        "erinys_save",
        title=args.title,
        content=read_content(args),
        type=args.type,
        project=args.project,
        scope=args.scope,
        topic_key=args.topic_key,
        session_id=args.session_id,
        metadata=parse_metadata(args.metadata_json),
    )


def run_summary(args: CliArgs, server: object | None) -> dict[str, object]:
    return call_tool(
        server,
        "erinys_session_summary",
        content=read_content(args),
        project=args.project,
        session_id=args.session_id,
    )


def run_get(args: CliArgs, server: object | None) -> dict[str, object]:
    return call_tool(server, "erinys_get", id=args.observation_id)


def run_search(args: CliArgs, server: object | None) -> dict[str, object]:
    if server is None:
        return readonly_search(args)
    return call_tool(server, "erinys_search", query=args.query, project=args.project, limit=args.limit)


def run_recall(args: CliArgs, server: object | None) -> dict[str, object]:
    if server is None:
        return readonly_recall(args)
    return call_tool(server, "erinys_recall", project=args.project, limit=args.limit)


def run_context(args: CliArgs, server: object | None) -> dict[str, object]:
    if server is None:
        return readonly_context(args)
    return call_tool(server, "erinys_context", project=args.project, limit=args.limit)


def row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {key: row[key] for key in row.keys()}


OBSERVATION_EXCERPT_COLUMNS = f"id, title, type, project, created_at, substr(content, 1, {EXCERPT_LENGTH}) AS excerpt"


def fts_match_expression(query: str) -> str:
    terms = [term.replace('"', '""') for term in query.split()]
    return " OR ".join(f'"{term}"' for term in terms)


def fts_search_rows(
    conn: sqlite3.Connection, query: str, project: str | None, limit: int
) -> list[dict[str, object]]:
    sql = (
        "SELECT o.id, o.title, o.type, o.project, o.created_at, "
        f"substr(o.content, 1, {EXCERPT_LENGTH}) AS excerpt "
        "FROM observations_fts f JOIN observations o ON o.id = f.rowid "
        "WHERE observations_fts MATCH ?"
    )
    params: list[object] = [fts_match_expression(query)]
    if project is not None:
        sql += " AND o.project = ?"
        params.append(project)
    params.append(limit)
    return [row_to_dict(row) for row in conn.execute(f"{sql} ORDER BY rank LIMIT ?", params).fetchall()]


def like_search_rows(
    conn: sqlite3.Connection, query: str, project: str | None, limit: int
) -> list[dict[str, object]]:
    sql = f"SELECT {OBSERVATION_EXCERPT_COLUMNS} FROM observations WHERE (title LIKE ? OR content LIKE ?)"
    pattern = f"%{query}%"
    params: list[object] = [pattern, pattern]
    if project is not None:
        sql += " AND project = ?"
        params.append(project)
    params.append(limit)
    return [row_to_dict(row) for row in conn.execute(f"{sql} ORDER BY created_at DESC, id DESC LIMIT ?", params).fetchall()]


def keyword_search_rows(
    conn: sqlite3.Connection, query: str, project: str | None, limit: int
) -> list[dict[str, object]]:
    if table_exists(conn, "observations_fts"):
        try:
            rows = fts_search_rows(conn, query, project, limit)
            if rows:
                return rows
        except sqlite3.OperationalError:
            pass
    return like_search_rows(conn, query, project, limit)


def readonly_search(args: CliArgs) -> dict[str, object]:
    with connect_readonly_db(db_path()) as conn:
        rows = keyword_search_rows(conn, str(args.query), args.project, args.limit)
    data = {"query": args.query, "results": rows, "mode": "readonly-keyword"}
    return {"ok": True, "data": data, "error": None}


def recent_observation_rows(
    conn: sqlite3.Connection, project: str | None, limit: int
) -> list[dict[str, object]]:
    sql = f"SELECT {OBSERVATION_EXCERPT_COLUMNS} FROM observations"
    params: list[object] = []
    if project is not None:
        sql += " WHERE project = ?"
        params.append(project)
    params.append(limit)
    return [row_to_dict(row) for row in conn.execute(f"{sql} ORDER BY created_at DESC, id DESC LIMIT ?", params).fetchall()]


def readonly_recall(args: CliArgs) -> dict[str, object]:
    with connect_readonly_db(db_path()) as conn:
        rows = recent_observation_rows(conn, args.project, args.limit)
    return {"ok": True, "data": {"observations": rows, "mode": "readonly"}, "error": None}


def recent_session_rows(
    conn: sqlite3.Connection, project: str | None, limit: int
) -> list[dict[str, object]]:
    sql = "SELECT id, project, directory, started_at, ended_at, summary FROM sessions"
    params: list[object] = []
    if project is not None:
        sql += " WHERE project = ?"
        params.append(project)
    params.append(limit)
    return [row_to_dict(row) for row in conn.execute(f"{sql} ORDER BY started_at DESC LIMIT ?", params).fetchall()]


def readonly_context(args: CliArgs) -> dict[str, object]:
    with connect_readonly_db(db_path()) as conn:
        sessions = recent_session_rows(conn, args.project, args.limit)
        observations = recent_observation_rows(conn, args.project, min(args.limit, DEFAULT_LIMIT))
    data = {"project": args.project, "sessions": sessions, "observations": observations, "mode": "readonly"}
    return {"ok": True, "data": data, "error": None}


def db_path() -> Path:
    return Path(os.environ.get("ERINYS_DB_PATH", "~/.erinys/memory.db")).expanduser()


def readonly_db_uri(path: Path) -> str:
    # quote so paths containing '?' or '#' are not parsed as URI components
    return f"file:{quote(str(path))}?mode=ro"


def connect_readonly_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(readonly_db_uri(path), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    try:
        row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", [table]).fetchone()
        return row is not None
    except sqlite3.OperationalError as exc:
        if "no such module" in str(exc):
            return False
        raise


def table_count(conn: sqlite3.Connection, table: str) -> int:
    if not table_exists(conn, table):
        return 0
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"]) if row is not None else 0


def observation_count(conn: sqlite3.Connection, project: str | None) -> int:
    if project is None:
        return table_count(conn, "observations")
    row = conn.execute("SELECT COUNT(*) AS count FROM observations WHERE project = ?", [project]).fetchone()
    return int(row["count"]) if row is not None else 0


UNDISTILLED_FILTER = """
    distillation_level IS NULL
    AND id NOT IN (
        SELECT DISTINCT distilled_from
        FROM observations
        WHERE distilled_from IS NOT NULL
    )
"""


def undistilled_query(project: str | None, select: str, suffix: str = "") -> tuple[str, list[object]]:
    query = f"SELECT {select} FROM observations WHERE {UNDISTILLED_FILTER}"
    params: list[object] = []
    if project is not None:
        query += " AND project = ?"
        params.append(project)
    return query + suffix, params


def undistilled_count(conn: sqlite3.Connection, project: str | None) -> int:
    if not column_exists(conn, "observations", "distillation_level"):
        return 0
    query, params = undistilled_query(project, "COUNT(*) AS count")
    row = conn.execute(query, params).fetchone()
    return int(row["count"]) if row is not None else 0


def undistilled_rows(conn: sqlite3.Connection, project: str | None, limit: int) -> list[dict[str, object]]:
    if not column_exists(conn, "observations", "distillation_level"):
        return []
    query, params = undistilled_query(project, "id, title, created_at", " ORDER BY created_at ASC, id ASC LIMIT ?")
    params.append(limit)
    return [row_to_dict(row) for row in conn.execute(query, params).fetchall()]


def run_undistilled(args: CliArgs, server: object | None) -> dict[str, object]:
    with connect_readonly_db(db_path()) as conn:
        rows = undistilled_rows(conn, args.project, args.limit)
    data = {
        "project": args.project,
        "count": len(rows),
        "ids": [row["id"] for row in rows],
        "items": rows,
    }
    return {"ok": True, "data": data, "error": None}


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    if not table_exists(conn, table):
        return False
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row["name"]) == column for row in rows)


def project_breakdown(conn: sqlite3.Connection, project: str | None) -> list[dict[str, int]]:
    if project is not None:
        return [{project: observation_count(conn, project)}]
    rows = conn.execute("SELECT COALESCE(project, '__NULL__') AS name, COUNT(*) AS count FROM observations GROUP BY name").fetchall()
    return [{str(row["name"]): int(row["count"])} for row in rows]


def latest_timestamp(conn: sqlite3.Connection, table: str, column: str) -> str | None:
    if not table_exists(conn, table):
        return None
    row = conn.execute(f"SELECT MAX({column}) AS value FROM {table}").fetchone()
    return str(row["value"]) if row is not None and row["value"] is not None else None


def dream_days_ago(last_collision_at: str | None) -> float | None:
    if last_collision_at is None:
        return None
    parsed = parse_sqlite_datetime(last_collision_at)
    elapsed = datetime.now(timezone.utc) - parsed
    return round(elapsed.total_seconds() / 86_400, 2)


def parse_sqlite_datetime(value: str) -> datetime:
    normalized = value.replace(" ", "T")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def vector_health(conn: sqlite3.Connection) -> dict[str, object]:
    if not table_exists(conn, "vec_observations"):
        return vector_unavailable()
    try:
        orphan = conn.execute("SELECT COUNT(*) AS count FROM vec_observations WHERE rowid NOT IN (SELECT id FROM observations)").fetchone()
        missing = conn.execute("SELECT COUNT(*) AS count FROM observations WHERE id NOT IN (SELECT rowid FROM vec_observations)").fetchone()
        return {"orphan_vectors": int(orphan["count"]), "missing_vectors": int(missing["count"]), "status": "ok"}
    except sqlite3.OperationalError:
        return vector_unavailable()


def vector_unavailable() -> dict[str, object]:
    return {
        "orphan_vectors": 0,
        "missing_vectors": 0,
        "status": "unavailable",
        "reason": "sqlite-vec extension unavailable in this runtime",
    }


def sqlite_stats(project: str | None) -> dict[str, object]:
    path = db_path()
    with connect_readonly_db(path) as conn:
        data = stats_data(conn, project, path)
    return {"ok": True, "data": data, "error": None}


def stats_data(conn: sqlite3.Connection, project: str | None, path: Path) -> dict[str, object]:
    return {
        "project": project,
        "db_path": str(path),
        **observation_stats(conn, project),
        **count_stats(conn),
        **collision_stats(conn),
        **auxiliary_stats(conn, project, path),
    }


def observation_stats(conn: sqlite3.Connection, project: str | None) -> dict[str, object]:
    observation_total = observation_count(conn, project)
    return {
        "observation_count": observation_total,
        "total": observation_total,
        "undistilled": undistilled_count(conn, project),
    }


def count_stats(conn: sqlite3.Connection) -> dict[str, object]:
    return {
        "session_count": table_count(conn, "sessions"),
        "edge_count": table_count(conn, "edges"),
        "prompt_count": table_count(conn, "prompts"),
    }


def collision_stats(conn: sqlite3.Connection) -> dict[str, object]:
    collision_total = table_count(conn, "collisions")
    last_collision_at = latest_timestamp(conn, "collisions", "created_at")
    return {
        "collision_count": collision_total,
        "collisions": collision_total,
        "last_collision_at": last_collision_at,
        "dream_days_ago": dream_days_ago(last_collision_at),
    }


def auxiliary_stats(conn: sqlite3.Connection, project: str | None, path: Path) -> dict[str, object]:
    return {
        "projects": project_breakdown(conn, project),
        "vector_health": vector_health(conn),
        "db_size_bytes": path.stat().st_size if path.exists() else 0,
    }


def run_stats(args: CliArgs, server: object | None) -> dict[str, object]:
    if server is None:
        return sqlite_stats(args.project)
    return call_tool(server, "erinys_stats", project=args.project)


def run_distill(args: CliArgs, server: object | None) -> dict[str, object]:
    return call_tool(server, "erinys_distill", observation_id=args.observation_id, level=args.level)


def run_dream(args: CliArgs, server: object | None) -> dict[str, object]:
    return call_tool(server, "erinys_dream", max_collisions=args.max_collisions)


def run_prune(args: CliArgs, server: object | None) -> dict[str, object]:
    return call_tool(server, "erinys_prune", threshold=args.threshold, dry_run=args.dry_run)


def vector_status(stats_data: dict[str, object]) -> str:
    vector = stats_data.get("vector_health")
    if not isinstance(vector, dict):
        return "unknown"
    if "status" in vector:
        return str(vector["status"])
    # server _vector_health() returns bare counts; healthy means no orphan/missing vectors
    healthy = vector.get("orphan_vectors") == 0 and vector.get("missing_vectors") == 0
    return "ok" if healthy else "degraded"


def deep_search_check(server: object | None) -> str:
    if server is None:
        return "fail: server unavailable"
    result = call_tool(server, "erinys_search", query="health deep probe", limit=1)
    return "ok" if result.get("ok") else f"fail: {result.get('error')}"


def health_checks(args: CliArgs, stats_data: dict[str, object], server: object | None) -> dict[str, str]:
    checks = {"stats": "ok", "vector": vector_status(stats_data)}
    if args.deep:
        checks["server_import"] = "ok" if server is not None else "fail"
        checks["search_smoke"] = deep_search_check(server)
    return checks


def run_health(args: CliArgs, server: object | None) -> dict[str, object]:
    stats = run_stats(args, server)
    if not stats.get("ok"):
        return stats
    stats_data = stats.get("data")
    checks = health_checks(args, stats_data if isinstance(stats_data, dict) else {}, server)
    status = "ok" if all(value == "ok" for value in checks.values()) else "degraded"
    data = {"status": status, "interface": "cli", "deep": args.deep, "checks": checks, "stats": stats_data}
    if status == "ok":
        return {"ok": True, "data": data, "error": None}
    error = {"code": "DEGRADED", "message": "health status=degraded; see data.checks (light health cannot verify vectors without sqlite-vec — use --deep for the authoritative check)"}
    return {"ok": False, "data": data, "error": error}


def _check(status: str, detail: str, fix: str | None = None) -> dict[str, object]:
    entry: dict[str, object] = {"status": status, "detail": detail}
    if fix is not None:
        entry["fix"] = fix
    return entry


def _doctor_python() -> dict[str, object]:
    version = ".".join(str(part) for part in sys.version_info[:3])
    return _check("ok", f"Python {version} on {sys.platform}")


def _doctor_sqlite() -> dict[str, object]:
    impl = "stdlib sqlite3" if sqlite3.__name__ == "sqlite3" else sqlite3.__name__
    supports_ext = hasattr(sqlite3.connect(":memory:"), "enable_load_extension")
    detail = f"{impl} (SQLite {sqlite3.sqlite_version}); loadable extensions: {supports_ext}"
    if supports_ext:
        return _check("ok", detail)
    return _check(
        "fail",
        detail,
        "Use a Python with extension support (python.org / Homebrew / conda), or "
        "install a compatible pysqlite3 (`pip install 'erinys-memory[fallback]'` "
        "where wheels exist). See docs/LIMITATIONS.md.",
    )


def _doctor_sqlite_vec() -> dict[str, object]:
    try:
        from erinys_memory.config import ErinysConfig
        from erinys_memory.db import get_db

        conn = get_db(ErinysConfig(db_path=":memory:", db_backup_on_init=False))
        try:
            vec_version = conn.execute("SELECT vec_version()").fetchone()[0]
        finally:
            conn.close()
        return _check("ok", f"sqlite-vec loaded (vec_version {vec_version})")
    except Exception as exc:  # noqa: BLE001 — surface the real remediation to the user
        return _check(
            "fail",
            f"sqlite-vec cannot load: {exc}",
            "Vector search is unavailable until this is fixed. See the sqlite check above.",
        )


def _doctor_embedding() -> dict[str, object]:
    from erinys_memory.config import ErinysConfig

    model = ErinysConfig().embedding_model
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return _check("fail", "fastembed is not installed", "pip install fastembed")
    return _check(
        "ok",
        f"fastembed available; model '{model}' (downloaded to cache on first embed)",
    )


def _doctor_dependencies() -> dict[str, object]:
    from importlib.metadata import PackageNotFoundError, version

    required = ["fastmcp", "sqlite-vec", "fastembed", "pydantic"]
    found: dict[str, str] = {}
    missing: list[str] = []
    for name in required:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            missing.append(name)
    if missing:
        return _check(
            "fail",
            f"missing: {', '.join(missing)}; present: {found}",
            "pip install erinys-memory",
        )
    return _check("ok", "; ".join(f"{k} {v}" for k, v in found.items()))


def _doctor_db(project: str | None) -> dict[str, object]:
    path = db_path()
    if not path.exists():
        return _check(
            "warn",
            f"no DB at {path} yet (created on first save)",
        )
    try:
        with connect_readonly_db(path) as conn:
            obs = observation_count(conn, None)
            vec = vector_health(conn)
        size_mb = round(path.stat().st_size / 1_048_576, 2)
        vec_status = vec.get("status", "unknown")
        status = "ok" if vec_status == "ok" else "warn"
        return _check(status, f"{path} ({size_mb} MB, {obs} observations, vectors: {vec_status})")
    except Exception as exc:  # noqa: BLE001
        return _check("fail", f"cannot read DB at {path}: {exc}")


def run_doctor(args: CliArgs, server: object | None) -> dict[str, object]:
    checks: dict[str, object] = {
        "python": _doctor_python(),
        "sqlite": _doctor_sqlite(),
        "sqlite_vec": _doctor_sqlite_vec(),
        "embedding": _doctor_embedding(),
        "dependencies": _doctor_dependencies(),
        "db": _doctor_db(args.project),
    }
    if args.deep:
        checks["server_import"] = _check("ok" if server is not None else "fail", "MCP server import")
        smoke = deep_search_check(server)
        checks["search_smoke"] = _check("ok" if smoke == "ok" else "fail", f"search smoke: {smoke}")
    statuses = [str(entry.get("status")) for entry in checks.values() if isinstance(entry, dict)]
    overall = "fail" if "fail" in statuses else ("degraded" if "warn" in statuses else "ok")
    data = {"status": overall, "interface": "cli", "deep": args.deep, "checks": checks}
    if overall == "ok":
        return {"ok": True, "data": data, "error": None}
    code = "UNHEALTHY" if overall == "fail" else "DEGRADED"
    return {"ok": False, "data": data, "error": {"code": code, "message": f"doctor status={overall}; see data.checks (each failing check includes a 'fix')"}}


HANDLERS: dict[ValidCommand, Callable[[CliArgs, object | None], dict[str, object]]] = {
    "save": run_save,
    "summary": run_summary,
    "get": run_get,
    "search": run_search,
    "recall": run_recall,
    "context": run_context,
    "stats": run_stats,
    "undistilled": run_undistilled,
    "distill": run_distill,
    "dream": run_dream,
    "prune": run_prune,
    "health": run_health,
    "doctor": run_doctor,
}


def dispatch(args: CliArgs, server: object | None) -> dict[str, object]:
    result = HANDLERS[args.command](args, server)
    if args.command in WRITE_COMMANDS:
        return normalize_write_result(result)
    return result


def server_for(args: CliArgs) -> object | None:
    if args.command in {"health", "doctor"}:
        return import_server() if args.deep else None
    if args.command in LOCAL_COMMANDS:
        return None
    if args.readonly and args.command in READONLY_COMMANDS:
        return None
    return import_server()


def error_result(code: str, message: str) -> dict[str, object]:
    return {"ok": False, "data": None, "error": {"code": code, "message": message}}


def is_codex_sandbox() -> bool:
    return bool(os.environ.get("CODEX_SANDBOX"))


def result_error_message(result: dict[str, object]) -> str:
    error = result.get("error")
    if isinstance(error, dict):
        return str(error.get("message", ""))
    return str(error or "")


def has_readonly_db_error(result: dict[str, object]) -> bool:
    message = result_error_message(result).lower()
    return any(marker in message for marker in READONLY_DB_MARKERS)


def write_access_error(detail: str) -> dict[str, object]:
    sandboxed = is_codex_sandbox()
    code = SANDBOX_DB_WRITE_REQUIRED if sandboxed else DB_WRITE_UNAVAILABLE
    message = (
        "ERINYS DB write needs an escalated Codex tool call."
        if sandboxed
        else "ERINYS DB path is not writable."
    )
    data = {
        "db_path": str(db_path()),
        "requires_escalation": sandboxed,
        "sandbox_permissions": "require_escalated" if sandboxed else None,
        "detail": detail,
    }
    return {"ok": False, "data": data, "error": {"code": code, "message": message}}


def normalize_write_result(result: dict[str, object]) -> dict[str, object]:
    if result.get("ok") or not has_readonly_db_error(result):
        return result
    return write_access_error(result_error_message(result))


def error_from_exception(exc: Exception) -> dict[str, object]:
    code = "VALIDATION" if isinstance(exc, (ValidationError, ValueError)) else "CLI_ERROR"
    return error_result(code, str(exc))


def write_json(result: dict[str, object]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=JSON_INDENT, default=json_default))


def json_default(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = dispatch(args, server_for(args))
    except Exception as exc:
        result = error_from_exception(exc)
    write_json(result)
    return EXIT_OK if result.get("ok") else EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
