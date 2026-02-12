"""Tests for news alert pipeline — theme parsing and dedup."""

import pytest

from portfolio_advisor.scheduler.alerts import (
    _is_similar_to_existing,
    _parse_themes,
    _theme_similarity,
)


class TestThemeParsing:
    """Tests for _parse_themes — extracting themes from research agent output."""

    def test_parses_json_with_themes_key(self):
        raw = '{"themes": [{"theme": "Fed Rate Cut", "impact": "high"}]}'
        themes = _parse_themes(raw)
        assert len(themes) == 1
        assert themes[0]["theme"] == "Fed Rate Cut"

    def test_parses_json_array(self):
        raw = '[{"theme": "AI Boom", "impact": "medium"}]'
        themes = _parse_themes(raw)
        assert len(themes) == 1

    def test_parses_markdown_code_fence(self):
        raw = """Here are the themes:
```json
{"themes": [{"theme": "Oil Price Surge", "impact": "high"}]}
```
"""
        themes = _parse_themes(raw)
        assert len(themes) == 1
        assert themes[0]["theme"] == "Oil Price Surge"

    def test_parses_json_embedded_in_text(self):
        raw = 'Based on analysis: {"themes": [{"theme": "Tech Selloff", "impact": "high"}]} end.'
        themes = _parse_themes(raw)
        assert len(themes) == 1

    def test_empty_input_returns_empty(self):
        assert _parse_themes("") == []
        assert _parse_themes(None) == []

    def test_garbage_input_returns_empty(self):
        assert _parse_themes("This is not JSON at all.") == []

    def test_nested_json_with_no_themes_key(self):
        raw = '{"data": {"info": "no themes here"}}'
        themes = _parse_themes(raw)
        assert themes == []


class TestThemeSimilarity:
    """Tests for theme similarity and deduplication."""

    def test_identical_themes_score_1(self):
        score = _theme_similarity("fed rate cut", "fed rate cut")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_very_different_themes_score_low(self):
        score = _theme_similarity("oil price surge", "tech earnings beat")
        assert score < 0.5

    def test_similar_themes_score_high(self):
        score = _theme_similarity(
            "federal reserve rate cut expected",
            "fed rate cut expectations rise",
        )
        assert score > 0.5

    def test_is_similar_to_existing_detects_duplicate(self):
        existing = ["fed rate cut expected", "tech selloff deepens"]
        assert _is_similar_to_existing("fed rate cut expected soon", existing)

    def test_is_similar_to_existing_allows_novel(self):
        existing = ["fed rate cut", "tech selloff deepens"]
        assert not _is_similar_to_existing("china gdp growth slows", existing)

    def test_is_similar_with_empty_existing(self):
        assert not _is_similar_to_existing("any theme", [])
