from __future__ import annotations

from datetime import datetime, timezone

from conftest import insert_test_observation
from erinys_memory.temporal import query_as_of
import erinys_memory.server as server


def test_supersede(db, engine) -> None:
    original = server.erinys_save(title="Feature Flag", content="Feature is enabled", project="alpha", topic_key="flag")
    result = server.erinys_supersede(old_id=original["data"]["id"], new_content="Feature is disabled", reason="rolled back")
    old_record = result["data"]["old"]
    new_record = result["data"]["new"]
    assert old_record["superseded_by"] == new_record["id"], "supersede should link the old observation to the new observation"
    assert old_record["valid_until"] is not None, "supersede should close the validity window of the old observation"
    assert new_record["topic_key"] == "flag", "supersede should preserve the topic_key on the new observation"


def test_timeline_current(db, engine) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    switch = datetime(2026, 2, 1, tzinfo=timezone.utc)
    insert_test_observation(
        db,
        engine,
        "Release Fact",
        "Feature is enabled",
        project="alpha",
        valid_from=start,
        valid_until=switch,
    )
    current = insert_test_observation(
        db,
        engine,
        "Release Fact",
        "Feature is disabled",
        project="alpha",
        valid_from=switch,
    )
    results = query_as_of(db, "Feature", as_of=None, project="alpha")
    ids = [row["id"] for row in results]
    assert int(current["id"]) in ids, "timeline current should include the currently valid observation"
    assert len(ids) == 1, "timeline current should exclude superseded historical observations"


def test_timeline_historical(db, engine) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    switch = datetime(2026, 2, 1, tzinfo=timezone.utc)
    historical = insert_test_observation(
        db,
        engine,
        "Historical Fact",
        "System uses Provider A",
        project="alpha",
        valid_from=start,
        valid_until=switch,
    )
    insert_test_observation(
        db,
        engine,
        "Historical Fact",
        "System uses Provider B",
        project="alpha",
        valid_from=switch,
    )
    results = query_as_of(db, "Provider", as_of="2026-01-15T00:00:00+00:00", project="alpha")
    ids = [row["id"] for row in results]
    assert ids == [int(historical["id"])], "timeline historical should return the fact valid at the requested past timestamp"


def test_supersede_chain(db, engine) -> None:
    first = server.erinys_save(title="Chain Fact", content="Version A", project="alpha", topic_key="chain")
    second = server.erinys_supersede(old_id=first["data"]["id"], new_content="Version B", reason="update")
    third = server.erinys_supersede(old_id=second["data"]["new"]["id"], new_content="Version C", reason="update")
    latest = third["data"]["new"]
    middle = second["data"]["new"]
    reloaded_middle = server.erinys_get(id=middle["id"])["data"]["observation"]
    assert second["data"]["old"]["superseded_by"] == middle["id"], "first link in the supersede chain should point to the middle observation"
    assert reloaded_middle["superseded_by"] == latest["id"], "middle observation should point to the final superseding observation"
    assert latest["content"] == "Version C", "supersede chain should preserve the final content at the tail"
