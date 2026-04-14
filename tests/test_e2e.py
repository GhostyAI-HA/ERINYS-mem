from __future__ import annotations

import erinys_memory.server as server


def test_save_search_cycle(db, engine) -> None:
    saved = server.erinys_save(title="Cycle", content="save search loop", project="alpha")
    searched = server.erinys_search(query="loop", project="alpha")
    ids = [row["id"] for row in searched["data"]["results"]]
    assert saved["data"]["id"] in ids, "save → search cycle should return the observation that was just saved"


def test_full_lifecycle(db, engine) -> None:
    start = server.erinys_session_start(id="e2e-session", project="alpha", directory="/tmp/alpha")
    first = server.erinys_save(title="One", content="alpha lifecycle one", project="alpha", session_id="e2e-session")
    second = server.erinys_save(title="Two", content="alpha lifecycle two", project="alpha", session_id="e2e-session")
    third = server.erinys_save(title="Three", content="alpha lifecycle three", project="alpha", session_id="e2e-session")
    searched = server.erinys_search(query="lifecycle", project="alpha", limit=3)
    reinforced = server.erinys_reinforce(first["data"]["id"])
    ended = server.erinys_session_end(id="e2e-session", summary="finished lifecycle")
    recalled = server.erinys_recall(project="alpha", limit=3)
    assert start["ok"] is True, "full lifecycle should start a session successfully"
    assert len(searched["data"]["results"]) == 3, "full lifecycle search should find all saved observations"
    assert reinforced["data"]["observation"]["access_count"] == 1, "full lifecycle should reinforce one observation"
    assert ended["data"]["session"]["ended_at"] is not None, "full lifecycle should end the session with an ended_at timestamp"
    recalled_ids = {row["id"] for row in recalled["data"]["observations"]}
    expected_ids = {first["data"]["id"], second["data"]["id"], third["data"]["id"]}
    assert recalled_ids == expected_ids, "full lifecycle recall should return the observations created during the session"
