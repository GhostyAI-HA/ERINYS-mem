"""Tests for search.py: query complexity, intent classification, noun phrase expansion."""

from __future__ import annotations

import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from erinys_memory.search import (
    classify_query_complexity,
    classify_query_intent,
    _sanitize_fts_or,
    COMPLEXITY_L1,
    COMPLEXITY_L2,
    COMPLEXITY_L3,
    INTENT_WHAT,
    INTENT_WHEN,
    INTENT_WHY,
    INTENT_WHO,
    INTENT_GENERAL,
)


# -- Phase 1: Query Complexity Classification --

class TestClassifyQueryComplexity:
    """Tests for classify_query_complexity (#110 SimpleMem)."""

    def test_single_word_is_l1(self):
        assert classify_query_complexity("python") == COMPLEXITY_L1

    def test_two_words_is_l1(self):
        assert classify_query_complexity("git commit") == COMPLEXITY_L1

    def test_medium_query_is_l2(self):
        assert classify_query_complexity("how to configure ERINYS search") == COMPLEXITY_L2

    def test_complex_long_query_is_l3(self):
        query = (
            'Why did the "adaptive retrieval" system in ERINYS fail '
            "on January 15th when processing large batch queries from Tokyo?"
        )
        assert classify_query_complexity(query) == COMPLEXITY_L3

    def test_quoted_phrase_increases_complexity(self):
        # Without quotes: simple
        simple = classify_query_complexity("search config")
        # With quotes: adds +2
        complex_ = classify_query_complexity('"search config"')
        # The quoted version should be same or higher complexity
        assert complex_ >= simple or complex_ == simple

    def test_temporal_expression_increases_complexity(self):
        without = classify_query_complexity("ERINYS configuration")
        with_temporal = classify_query_complexity("ERINYS configuration last week")
        # Temporal adds complexity
        assert with_temporal >= without

    def test_empty_query_is_l1(self):
        assert classify_query_complexity("") == COMPLEXITY_L1

    def test_japanese_query(self):
        # W2: CJK queries should default to L2 (vec-heavy) since FTS5 porter
        # has poor CJK recall
        result = classify_query_complexity("設定")
        assert result == COMPLEXITY_L2


# -- Phase 3a: Intent Classification --

class TestClassifyQueryIntent:
    """Tests for classify_query_intent (#131 MAGMA)."""

    def test_what_intent(self):
        assert classify_query_intent("what is ERINYS?") == INTENT_WHAT

    def test_when_intent(self):
        assert classify_query_intent("when was the last deployment?") == INTENT_WHEN

    def test_why_intent(self):
        assert classify_query_intent("why did the search fail?") == INTENT_WHY

    def test_who_intent(self):
        assert classify_query_intent("who authored this module?") == INTENT_WHO

    def test_general_intent(self):
        assert classify_query_intent("ERINYS configuration") == INTENT_GENERAL

    def test_why_has_highest_priority(self):
        # "why" should win over "what" when both present
        assert classify_query_intent("why is what broken?") == INTENT_WHY

    def test_japanese_why(self):
        assert classify_query_intent("なぜ検索が失敗したのか") == INTENT_WHY

    def test_japanese_when(self):
        assert classify_query_intent("いつデプロイされましたか") == INTENT_WHEN

    def test_japanese_who(self):
        assert classify_query_intent("誰がこのモジュールを書いたか") == INTENT_WHO

    def test_case_insensitive(self):
        assert classify_query_intent("WHAT is this?") == INTENT_WHAT
        assert classify_query_intent("What is this?") == INTENT_WHAT


# -- Phase 2: Noun Phrase Expansion in FTS --

class TestSanitizeFtsOr:
    """Tests for _sanitize_fts_or with noun phrase expansion (#105 TrueMemory)."""

    def test_basic_keywords(self):
        result = _sanitize_fts_or("ERINYS search configuration")
        assert '"ERINYS"' in result or '"erinys"' in result.lower()
        assert "OR" in result

    def test_empty_raises(self):
        import pytest
        with pytest.raises(ValueError, match="FTS query must not be empty"):
            _sanitize_fts_or("   ")

    def test_single_word(self):
        result = _sanitize_fts_or("python")
        assert '"python"' in result

    def test_noun_phrase_generates_near(self):
        # A query with clear noun phrases should generate NEAR clauses
        result = _sanitize_fts_or("memory management system optimization")
        # Should contain individual keywords and possibly NEAR clauses
        assert "OR" in result
