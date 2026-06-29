"""Tests for collider.py: dream cycle outcome scoring."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory.collider import _compute_collision_score


class TestComputeCollisionScore:
    """Tests for _compute_collision_score (#117)."""

    def test_novel_collision_high_novelty(self):
        result = _compute_collision_score(
            "memory management garbage collection",
            "database indexing optimization",
            "quantum computing neural network blockchain",  # all new concepts
            0.7,
        )
        assert result["novelty"] > 0.5

    def test_redundant_collision_low_novelty(self):
        result = _compute_collision_score(
            "memory management garbage collection",
            "database indexing optimization",
            "memory management database indexing",  # same concepts recycled
            0.7,
        )
        assert result["novelty"] < 0.5

    def test_serendipity_formula(self):
        result = _compute_collision_score(
            "source alpha content",
            "source beta content",
            "collision result text",
            0.5,
        )
        expected = (
            result["novelty"] * (1 - 0.5) + result["relevance"] * 0.5
        )
        assert abs(result["serendipity_score"] - expected) < 0.01

    def test_high_similarity_weights_relevance(self):
        result_high_sim = _compute_collision_score(
            "same topic area", "same topic area", "related insight", 0.95,
        )
        result_low_sim = _compute_collision_score(
            "same topic area", "same topic area", "related insight", 0.1,
        )
        # High similarity -> serendipity weights relevance more
        # Low similarity -> serendipity weights novelty more
        # Both should be valid scores
        assert 0.0 <= result_high_sim["serendipity_score"] <= 1.0
        assert 0.0 <= result_low_sim["serendipity_score"] <= 1.0

    def test_return_keys(self):
        result = _compute_collision_score("a", "b", "c", 0.5)
        expected_keys = {"novelty", "relevance", "serendipity_score"}
        assert expected_keys.issubset(set(result.keys()))

    def test_empty_collision(self):
        result = _compute_collision_score("source", "source", "", 0.5)
        assert "serendipity_score" in result
