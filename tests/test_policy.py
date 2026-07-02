"""P0-7 memory access gate tests (policy module + opt-in server wiring)."""

from __future__ import annotations

import pytest

import erinys_memory.server as server
from erinys_memory.policy import (
    admit_memory,
    extract_principal,
    extract_project,
    extract_scope,
    normalize_scope_set,
    policy_is_active,
    retrieve_policy,
)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def test_policy_is_active_distinguishes_none_from_empty() -> None:
    assert policy_is_active(None, None) is False, "no constraints means no policy"
    assert policy_is_active([], None) is True, "empty allow-list is a real deny-all constraint, still active"
    assert policy_is_active(None, ["project"]) is True, "any single constraint activates the policy"


def test_normalize_scope_set_lowercases_and_drops_blanks() -> None:
    assert normalize_scope_set(None) is None, "None stays unconstrained"
    assert normalize_scope_set([" Project ", "GLOBAL", ""]) == frozenset({"project", "global"}), (
        "scopes should be trimmed, lower-cased, and blanks dropped"
    )


def test_extractors_read_project_scope_and_principal() -> None:
    row = {
        "project": " alpha ",
        "scope": "Personal",
        "metadata": {"provenance": {"principal": " codex "}},
    }
    assert extract_project(row) == "alpha", "project is trimmed"
    assert extract_scope(row) == "personal", "scope is lower-cased"
    assert extract_principal(row) == "codex", "principal comes from metadata.provenance"


def test_extract_scope_defaults_to_project_when_missing() -> None:
    assert extract_scope({}) == "project", "missing scope defaults to the DB default 'project'"
    assert extract_project({}) is None, "missing project is None (unscoped)"


def test_extract_principal_falls_back_to_top_level_key() -> None:
    assert extract_principal({"principal": "claude"}) == "claude", (
        "write payloads carry principal at top level before provenance is materialized"
    )
    assert extract_principal({}) is None, "no principal anywhere returns None"


# --------------------------------------------------------------------------- #
# retrieve_policy — the read-side trust boundary
# --------------------------------------------------------------------------- #

def _rows() -> list[dict[str, object]]:
    return [
        {"id": 1, "project": "A", "scope": "project", "title": "a-proj"},
        {"id": 2, "project": "B", "scope": "project", "title": "b-proj"},
        {"id": 3, "project": "A", "scope": "personal", "title": "a-personal"},
        {"id": 4, "project": "B", "scope": "global", "title": "b-global"},
        {"id": 5, "project": None, "scope": "project", "title": "unscoped"},
    ]


def test_retrieve_policy_no_config_is_passthrough() -> None:
    rows = _rows()
    out = retrieve_policy(rows)
    assert [r["id"] for r in out] == [1, 2, 3, 4, 5], "no policy configured returns every row unchanged"
    assert out is not rows and out[0] is not rows[0], "output must be fresh dicts, not aliases of the inputs"


def test_retrieve_policy_restricts_principal_to_project_a() -> None:
    out = retrieve_policy(_rows(), principal="alice", allowed_projects=["A"])
    ids = {r["id"] for r in out}
    assert 2 not in ids, "a principal restricted to project A must NOT receive project B project-scoped rows"
    assert 1 in ids and 3 in ids, "project A rows (any non-global scope) remain visible"
    assert 4 in ids, "global-scoped rows are shared knowledge and bypass the project allow-list"
    assert 5 in ids, "unscoped (project=None) rows are not withheld by a project allow-list"


def test_retrieve_policy_scope_filter_drops_personal() -> None:
    out = retrieve_policy(_rows(), allowed_scopes=["project", "global"])
    scopes = {r["scope"] for r in out}
    assert "personal" not in scopes, "a caller not allowed 'personal' scope must not see personal rows"
    assert {"project", "global"} <= scopes, "allowed scopes still flow through"


def test_retrieve_policy_combined_project_and_scope() -> None:
    out = retrieve_policy(_rows(), allowed_projects=["A"], allowed_scopes=["project"])
    ids = {r["id"] for r in out}
    assert ids == {1, 5}, (
        "combined gate: project A + scope project keeps A/project (1) and unscoped/project (5), "
        "drops B (2), personal (3), and even global (4) because scope 'global' is not allowed"
    )


def test_retrieve_policy_empty_allowlist_denies_all_project_rows() -> None:
    out = retrieve_policy(_rows(), allowed_projects=[])
    ids = {r["id"] for r in out}
    assert ids == {4, 5}, "empty project allow-list denies every project-scoped row but keeps global and unscoped"


# --------------------------------------------------------------------------- #
# admit_memory — the write-side trust boundary
# --------------------------------------------------------------------------- #

def test_admit_memory_no_policy_admits_everything() -> None:
    ok, reason = admit_memory({"project": "anything", "scope": "project"})
    assert ok is True and reason == "no-policy", "with no constraint every write is admitted unchanged"


def test_admit_memory_blocks_disallowed_scope() -> None:
    ok, reason = admit_memory(
        {"project": "A", "scope": "personal"},
        allowed_scopes=["project"],
    )
    assert ok is False, "writing scope 'personal' must be blocked when only 'project' is allowed"
    assert reason.startswith("scope-denied"), "reason should name the scope violation"


def test_admit_memory_blocks_disallowed_project() -> None:
    ok, reason = admit_memory(
        {"project": "B", "scope": "project"},
        allowed_projects=["A"],
    )
    assert ok is False and reason.startswith("project-denied"), (
        "writing into project B must be blocked when only A is allowed"
    )


def test_admit_memory_allows_global_write_despite_project_gate() -> None:
    ok, _ = admit_memory({"project": "B", "scope": "global"}, allowed_projects=["A"])
    assert ok is True, "global writes are shared knowledge and bypass the project allow-list"


def test_admit_memory_admits_in_scope_write() -> None:
    ok, reason = admit_memory(
        {"project": "A", "scope": "project"},
        allowed_projects=["A"],
        allowed_scopes=["project"],
    )
    assert ok is True and reason == "admitted", "an in-project, in-scope write is admitted"


# --------------------------------------------------------------------------- #
# Server wiring — opt-in, backward-compatible
# --------------------------------------------------------------------------- #

def test_search_without_policy_sees_everything(db, engine) -> None:
    server.erinys_save(title="Alpha secret", content="shared token phrase alpha", project="A")
    server.erinys_save(title="Beta secret", content="shared token phrase beta", project="B")
    result = server.erinys_search(query="shared token phrase", limit=10)
    projects = {r["project"] for r in result["data"]["results"]}
    assert result["ok"] is True
    assert projects == {"A", "B"}, "an unrestricted caller sees rows from every project (unchanged behavior)"


def test_search_allowed_projects_arg_drops_other_project(db, engine) -> None:
    server.erinys_save(title="Alpha secret", content="shared token phrase alpha", project="A")
    server.erinys_save(title="Beta secret", content="shared token phrase beta", project="B")
    result = server.erinys_search(query="shared token phrase", limit=10, allowed_projects=["A"])
    projects = {r["project"] for r in result["data"]["results"]}
    assert projects == {"A"}, "a caller restricted to project A must not retrieve project B rows"


def test_search_allowed_projects_via_env(db, engine, monkeypatch) -> None:
    server.erinys_save(title="Alpha secret", content="shared token phrase alpha", project="A")
    server.erinys_save(title="Beta secret", content="shared token phrase beta", project="B")
    monkeypatch.setenv("ERINYS_ALLOWED_PROJECTS", "A")
    result = server.erinys_search(query="shared token phrase", limit=10)
    projects = {r["project"] for r in result["data"]["results"]}
    assert projects == {"A"}, "ERINYS_ALLOWED_PROJECTS env should gate retrieval when no explicit arg is passed"


def test_search_global_scope_survives_project_gate(db, engine) -> None:
    server.erinys_save(title="Beta local", content="policy scope token", project="B", scope="project")
    server.erinys_save(title="Shared rule", content="policy scope token", project="B", scope="global")
    result = server.erinys_search(query="policy scope token", limit=10, allowed_projects=["A"])
    titles = {r["title"] for r in result["data"]["results"]}
    assert "Beta local" not in titles, "project B's project-scoped row is hidden from a project-A caller"
    assert "Shared rule" in titles, "global-scoped row remains visible across the project gate"


def test_save_blocked_by_scope_policy_returns_envelope_error(db, engine) -> None:
    result = server.erinys_save(
        title="Personal note",
        content="should be blocked by scope policy",
        project="A",
        scope="personal",
        allowed_scopes=["project"],
    )
    assert result["ok"] is False, "a disallowed-scope write must fail"
    assert result["error"]["code"] == "POLICY_DENIED", "blocked writes surface a POLICY_DENIED envelope"
    stored = db.execute("SELECT COUNT(*) FROM observations WHERE title = 'Personal note'").fetchone()[0]
    assert stored == 0, "a blocked write must not persist any row"


def test_save_blocked_by_project_env_policy(db, engine, monkeypatch) -> None:
    monkeypatch.setenv("ERINYS_ALLOWED_PROJECTS", "A")
    result = server.erinys_save(title="Into B", content="write into disallowed project B", project="B")
    assert result["ok"] is False and result["error"]["code"] == "POLICY_DENIED", (
        "ERINYS_ALLOWED_PROJECTS should block writes into a project outside the allow-list"
    )
    stored = db.execute("SELECT COUNT(*) FROM observations WHERE title = 'Into B'").fetchone()[0]
    assert stored == 0, "blocked env-gated write must not persist"


def test_save_admitted_when_in_policy(db, engine) -> None:
    result = server.erinys_save(
        title="In A",
        content="allowed write into project A",
        project="A",
        allowed_projects=["A"],
        allowed_scopes=["project"],
    )
    assert result["ok"] is True, "an in-policy write should succeed normally"
    stored = db.execute("SELECT COUNT(*) FROM observations WHERE title = 'In A'").fetchone()[0]
    assert stored == 1, "an admitted write persists exactly one row"


def test_save_without_policy_unchanged(db, engine) -> None:
    result = server.erinys_save(title="Free", content="no policy configured at all", project="whatever")
    assert result["ok"] is True, "with no policy configured, saves behave exactly as before"
    stored = db.execute("SELECT COUNT(*) FROM observations WHERE title = 'Free'").fetchone()[0]
    assert stored == 1, "unrestricted write persists"
