"""Tests for the ERINYS JSON CLI adapter."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory import cli  # noqa: E402


HEALTHY_VECTOR = {"orphan_vectors": 0, "missing_vectors": 0}


class FakeServer:
    def __init__(self, vector_health: dict[str, object] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.vector_health = vector_health if vector_health is not None else dict(HEALTHY_VECTOR)

    def erinys_save(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("save", kwargs))
        return {"ok": True, "data": {"id": 1, "status": "created"}, "error": None}

    def erinys_search(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("search", kwargs))
        return {"ok": True, "data": {"results": []}, "error": None}

    def erinys_stats(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("stats", kwargs))
        data = {"observation_count": 0, "vector_health": self.vector_health}
        return {"ok": True, "data": data, "error": None}

    def erinys_prune(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("prune", kwargs))
        return {"ok": True, "data": {"candidates": []}, "error": None}


class ReadonlyErrorServer(FakeServer):
    def erinys_save(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("save", kwargs))
        error = {"code": "DB_ERROR", "message": "attempt to write a readonly database"}
        return {"ok": False, "data": None, "error": error}


def test_save_dispatch_passes_validated_payload() -> None:
    server = FakeServer()
    args = cli.CliArgs(
        command="save",
        title="CLI save",
        content="What: verify CLI.",
        type="decision",
        metadata_json='{"source":"test"}',
    )

    result = cli.dispatch(args, server)

    assert result["ok"] is True
    assert server.calls[0][0] == "save"
    assert server.calls[0][1]["title"] == "CLI save"
    assert server.calls[0][1]["metadata"] == {"source": "test"}


def test_save_normalizes_readonly_db_error(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_SANDBOX", "seatbelt")
    server = ReadonlyErrorServer()
    args = cli.CliArgs(command="save", title="CLI save", content="What: verify CLI.")

    result = cli.dispatch(args, server)

    assert result["ok"] is False
    assert result["error"]["code"] == "SANDBOX_DB_WRITE_REQUIRED"
    assert result["data"]["requires_escalation"] is True
    assert result["data"]["sandbox_permissions"] == "require_escalated"


def test_search_parser_accepts_json_flag_after_command() -> None:
    args = cli.parse_args(["search", "Buffer DNS", "--project", "my-project", "--json"])

    assert args.command == "search"
    assert args.query == "Buffer DNS"
    assert args.json_output is True


def test_health_uses_stats_as_probe() -> None:
    server = FakeServer()
    args = cli.CliArgs(command="health", project="my-project")

    result = cli.dispatch(args, server)

    assert result["ok"] is True
    assert result["data"]["status"] == "ok"
    assert result["data"]["checks"]["vector"] == "ok"
    assert server.calls[0] == ("stats", {"project": "my-project"})


def test_health_reports_degraded_when_vector_unavailable() -> None:
    server = FakeServer(vector_health={"status": "unavailable"})
    args = cli.CliArgs(command="health", project="my-project")

    result = cli.dispatch(args, server)

    assert result["ok"] is False
    assert result["data"]["status"] == "degraded"
    assert result["error"]["code"] == "DEGRADED"


def test_health_deep_runs_search_smoke_test() -> None:
    server = FakeServer()
    args = cli.CliArgs(command="health", project="my-project", deep=True)

    result = cli.dispatch(args, server)

    assert result["ok"] is True
    assert result["data"]["checks"]["server_import"] == "ok"
    assert result["data"]["checks"]["search_smoke"] == "ok"
    assert ("search", {"query": "health deep probe", "limit": 1}) in server.calls


def test_usage_error_emits_json(capsys: CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.parse_args(["distill", "1", "--level", "nope"])

    assert excinfo.value.code == cli.EXIT_USAGE
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "USAGE"


def test_prune_defaults_to_dry_run() -> None:
    server = FakeServer()
    args = cli.parse_args(["prune", "--threshold", "0.2", "--json"])

    result = cli.dispatch(args, server)

    assert result["ok"] is True
    assert server.calls[0] == ("prune", {"threshold": 0.2, "dry_run": True})


def test_prune_execute_requires_confirm_global(capsys: CaptureFixture[str]) -> None:
    exit_code = cli.main(["prune", "--execute", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_ERROR
    assert payload["ok"] is False
    assert payload["error"]["code"] == "VALIDATION"
    assert "confirm-global" in payload["error"]["message"]


def test_prune_execute_with_confirm_global_passes() -> None:
    server = FakeServer()
    args = cli.parse_args(["prune", "--execute", "--confirm-global", "--json"])

    result = cli.dispatch(args, server)

    assert result["ok"] is True
    assert server.calls[0] == ("prune", {"threshold": 0.1, "dry_run": False})


def test_stats_can_read_sqlite_without_server(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    create_stats_db(db_path)
    monkeypatch.setenv("ERINYS_DB_PATH", str(db_path))
    args = cli.CliArgs(command="stats", project="my-project")

    result = cli.dispatch(args, None)

    data = result["data"]
    assert result["ok"] is True
    assert data["observation_count"] == 1
    assert data["total"] == 1
    assert data["undistilled"] == 1
    assert data["session_count"] == 1
    assert data["collisions"] == 1
    assert data["last_collision_at"] == "2026-06-01 00:00:00"
    assert data["dream_days_ago"] is not None


def create_stats_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE observations(
                id INTEGER PRIMARY KEY,
                project TEXT,
                distillation_level TEXT,
                distilled_from INTEGER
            );
            CREATE TABLE sessions(id TEXT PRIMARY KEY);
            CREATE TABLE collisions(id INTEGER PRIMARY KEY, created_at DATETIME);
            INSERT INTO observations(id, project) VALUES (1, 'my-project');
            INSERT INTO sessions(id) VALUES ('session-1');
            INSERT INTO collisions(id, created_at) VALUES (1, '2026-06-01 00:00:00');
        """)


def create_readonly_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE observations(
                id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT,
                type TEXT,
                project TEXT,
                created_at DATETIME,
                distillation_level TEXT,
                distilled_from INTEGER
            );
            CREATE TABLE sessions(
                id TEXT PRIMARY KEY,
                project TEXT,
                directory TEXT,
                started_at DATETIME,
                ended_at DATETIME,
                summary TEXT
            );
            INSERT INTO observations(id, title, content, type, project, created_at)
            VALUES
              (1, 'Buffer DNS fix', 'Fixed the Buffer DNS issue.', 'bugfix', 'my-project', '2026-06-01 00:00:00'),
              (2, 'CLI design', 'CLI-first migration decision.', 'decision', 'my-project', '2026-06-02 00:00:00'),
              (3, 'Other project note', 'Unrelated note.', 'manual', 'OtherProject', '2026-06-03 00:00:00');
            INSERT INTO sessions(id, project, started_at, summary)
            VALUES ('session-1', 'my-project', '2026-06-02 09:00:00', 'worked on CLI');
        """)


def test_readonly_search_uses_like_fallback(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    create_readonly_db(db_path)
    monkeypatch.setenv("ERINYS_DB_PATH", str(db_path))
    args = cli.parse_args(["search", "Buffer", "--project", "my-project", "--readonly", "--json"])

    assert cli.server_for(args) is None
    result = cli.dispatch(args, None)

    assert result["ok"] is True
    assert result["data"]["mode"] == "readonly-keyword"
    assert [row["id"] for row in result["data"]["results"]] == [1]


def test_readonly_recall_returns_recent_observations(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    create_readonly_db(db_path)
    monkeypatch.setenv("ERINYS_DB_PATH", str(db_path))
    args = cli.parse_args(["recall", "--project", "my-project", "--limit", "1", "--readonly", "--json"])

    result = cli.dispatch(args, None)

    assert result["ok"] is True
    assert [row["id"] for row in result["data"]["observations"]] == [2]


def test_readonly_context_returns_sessions_and_observations(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    create_readonly_db(db_path)
    monkeypatch.setenv("ERINYS_DB_PATH", str(db_path))
    args = cli.parse_args(["context", "--project", "my-project", "--limit", "5", "--readonly", "--json"])

    result = cli.dispatch(args, None)

    assert result["ok"] is True
    assert [row["id"] for row in result["data"]["sessions"]] == ["session-1"]
    assert len(result["data"]["observations"]) == 2


def test_readonly_db_uri_escapes_special_characters(tmp_path: Path) -> None:
    weird_dir = tmp_path / "erinys?cache#1"
    weird_dir.mkdir()
    db_path = weird_dir / "memory.db"
    create_readonly_db(db_path)

    uri = cli.readonly_db_uri(db_path)

    assert "?mode=ro" in uri
    assert uri.count("?") == 1
    assert "#" not in uri
    with cli.connect_readonly_db(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) AS count FROM observations").fetchone()["count"] == 3


def test_readonly_does_not_modify_db(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    create_readonly_db(db_path)
    before = db_path.read_bytes()
    monkeypatch.setenv("ERINYS_DB_PATH", str(db_path))

    cli.dispatch(cli.parse_args(["search", "Buffer", "--readonly", "--json"]), None)
    cli.dispatch(cli.parse_args(["recall", "--readonly", "--json"]), None)

    assert db_path.read_bytes() == before


def test_undistilled_returns_oldest_ids(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    create_readonly_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE observations SET distillation_level = 'meta' WHERE id = 2")
    monkeypatch.setenv("ERINYS_DB_PATH", str(db_path))
    args = cli.parse_args(["undistilled", "--project", "my-project", "--limit", "10", "--json"])

    assert cli.server_for(args) is None
    result = cli.dispatch(args, None)

    assert result["ok"] is True
    assert result["data"]["ids"] == [1]
    assert result["data"]["count"] == 1


def test_invalid_metadata_returns_validation_error() -> None:
    result = cli.error_from_exception(ValueError("--metadata-json must decode to an object"))

    assert result["ok"] is False
    assert result["error"]["code"] == "VALIDATION"


def test_write_json_serializes_datetime(capsys: CaptureFixture[str]) -> None:
    timestamp = datetime(2026, 6, 8, tzinfo=timezone.utc)

    cli.write_json({"ok": True, "data": {"created_at": timestamp}, "error": None})

    assert "2026-06-08T00:00:00+00:00" in capsys.readouterr().out
