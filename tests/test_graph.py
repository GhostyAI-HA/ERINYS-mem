from __future__ import annotations

from conftest import insert_test_observation
from erinys_memory.db import delete_observation_with_embedding
from erinys_memory.graph import create_edge, traverse
import erinys_memory.server as server


def test_create_edge(db, engine) -> None:
    left = insert_test_observation(db, engine, "Left", "left graph node")
    right = insert_test_observation(db, engine, "Right", "right graph node")
    edge = create_edge(db, int(left["id"]), int(right["id"]), "relates_to", 0.75)
    assert edge["relation"] == "relates_to", "create_edge should keep the requested relation"
    assert edge["weight"] == 0.75, "create_edge should keep the requested weight"


def test_duplicate_edge_update(db, engine) -> None:
    left = insert_test_observation(db, engine, "Left", "left graph node")
    right = insert_test_observation(db, engine, "Right", "right graph node")
    create_edge(db, int(left["id"]), int(right["id"]), "references", 0.2)
    updated = create_edge(db, int(left["id"]), int(right["id"]), "references", 0.9)
    count = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    assert count == 1, "duplicate edge upsert should keep only one edge row"
    assert updated["weight"] == 0.9, "duplicate edge upsert should overwrite the edge weight"


def test_edge_cascade_delete(db, engine) -> None:
    left = insert_test_observation(db, engine, "Left", "left graph node")
    right = insert_test_observation(db, engine, "Right", "right graph node")
    create_edge(db, int(left["id"]), int(right["id"]), "contains")
    delete_observation_with_embedding(db, int(left["id"]))
    count = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    assert count == 0, "deleting an observation should cascade-delete dependent edges"


def test_traverse_depth_1(db, engine) -> None:
    start = insert_test_observation(db, engine, "Start", "start node")
    middle = insert_test_observation(db, engine, "Middle", "middle node")
    end = insert_test_observation(db, engine, "End", "end node")
    create_edge(db, int(start["id"]), int(middle["id"]), "relates_to")
    create_edge(db, int(middle["id"]), int(end["id"]), "relates_to")
    graph = traverse(db, int(start["id"]), max_depth=1)
    titles = [node["title"] for node in graph["nodes"]]
    assert titles == ["Middle"], "depth=1 traversal should only include direct neighbors"


def test_traverse_depth_2(db, engine) -> None:
    start = insert_test_observation(db, engine, "Start", "start node")
    middle = insert_test_observation(db, engine, "Middle", "middle node")
    end = insert_test_observation(db, engine, "End", "end node")
    create_edge(db, int(start["id"]), int(middle["id"]), "depends_on")
    create_edge(db, int(middle["id"]), int(end["id"]), "depends_on")
    graph = traverse(db, int(start["id"]), max_depth=2)
    depths = {node["title"]: node["depth"] for node in graph["nodes"]}
    assert depths == {"Middle": 1, "End": 2}, "depth=2 traversal should include indirect neighbors with depth metadata"


def test_traverse_relation_filter(db, engine) -> None:
    start = insert_test_observation(db, engine, "Start", "start node")
    keep = insert_test_observation(db, engine, "Keep", "keep node")
    skip = insert_test_observation(db, engine, "Skip", "skip node")
    create_edge(db, int(start["id"]), int(keep["id"]), "references")
    create_edge(db, int(start["id"]), int(skip["id"]), "contains")
    graph = traverse(db, int(start["id"]), max_depth=1, relation_filter=["references"])
    titles = [node["title"] for node in graph["nodes"]]
    assert titles == ["Keep"], "relation_filter should only traverse matching edge relations"


def test_contradicts_edge(db, engine) -> None:
    left = insert_test_observation(db, engine, "Claim A", "feature is enabled")
    right = insert_test_observation(db, engine, "Claim B", "feature is disabled")
    edge = create_edge(db, int(left["id"]), int(right["id"]), "contradicts")
    assert edge["relation"] == "contradicts", "graph should allow contradicts relations"


def test_supersedes_edge(db, engine) -> None:
    old = server.erinys_save(title="Fact", content="Old fact", project="alpha", topic_key="fact")
    result = server.erinys_supersede(old_id=old["data"]["id"], new_content="New fact", reason="updated")
    edge = result["data"]["edge"]
    assert edge["relation"] == "supersedes", "superseding an observation should create a supersedes edge"
    count = db.execute("SELECT COUNT(*) FROM edges WHERE relation = 'supersedes'").fetchone()[0]
    assert count == 1, "superseding an observation should persist one supersedes edge"
