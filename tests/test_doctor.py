"""Tests for the ERINYS `doctor` diagnostic command."""

from __future__ import annotations

import sys
from pathlib import Path

from pytest import MonkeyPatch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory import cli  # noqa: E402


def _doctor(tmp_path: Path, monkeypatch: MonkeyPatch, deep: bool = False) -> dict:
    monkeypatch.setenv("ERINYS_DB_PATH", str(tmp_path / "doctor.db"))
    args = cli.CliArgs(command="doctor", project="default", deep=deep)
    return cli.run_doctor(args, None)


def test_doctor_reports_all_environment_checks(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    result = _doctor(tmp_path, monkeypatch)
    checks = result["data"]["checks"]
    for name in ("python", "sqlite", "sqlite_vec", "embedding", "dependencies", "db"):
        assert name in checks
        assert checks[name]["status"] in {"ok", "warn", "fail"}
        assert checks[name]["detail"]


def test_doctor_core_runtime_is_healthy_under_test(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # The test interpreter can always load sqlite-vec (stdlib or pysqlite3 fallback),
    # so these core checks must pass — this guards the portability fix.
    checks = _doctor(tmp_path, monkeypatch)["data"]["checks"]
    for name in ("sqlite", "sqlite_vec", "dependencies"):
        assert checks[name]["status"] == "ok", (name, checks[name])


def test_doctor_missing_db_is_warn_not_fail(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    result = _doctor(tmp_path, monkeypatch)
    assert result["data"]["checks"]["db"]["status"] == "warn"
    # No DB yet is degraded overall, never a hard failure.
    assert result["data"]["status"] in {"degraded", "ok"}


def test_doctor_failing_check_carries_a_fix(monkeypatch: MonkeyPatch) -> None:
    # Force the sqlite check to fail and confirm it surfaces an actionable fix.
    monkeypatch.setattr(cli, "_doctor_sqlite", lambda: cli._check("fail", "simulated", "do X"))
    args = cli.CliArgs(command="doctor", project="default")
    result = cli.run_doctor(args, None)
    assert result["ok"] is False
    assert result["error"]["code"] == "UNHEALTHY"
    assert result["data"]["checks"]["sqlite"]["fix"] == "do X"
