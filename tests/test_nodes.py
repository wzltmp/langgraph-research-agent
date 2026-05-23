"""Offline unit tests for src/nodes.py — no Anthropic, Tavily, or HTTP calls."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src import nodes
from src.state import Source


# ---------- _parse_plan_json ----------
class TestParsePlanJson:
    def test_clean_json(self):
        raw = '{"sub_queries": ["a", "b", "c"]}'
        assert nodes._parse_plan_json(raw) == ["a", "b", "c"]

    def test_json_with_leading_prose(self):
        raw = 'Sure! Here you go:\n{"sub_queries": ["a", "b"]}\nLet me know if you want changes.'
        assert nodes._parse_plan_json(raw) == ["a", "b"]

    def test_markdown_fenced_json(self):
        raw = '```json\n{"sub_queries": ["a"]}\n```'
        assert nodes._parse_plan_json(raw) == ["a"]

    def test_no_json_raises_value_error(self):
        with pytest.raises(ValueError):
            nodes._parse_plan_json("nope, no json here")

    def test_missing_sub_queries_key_raises(self):
        with pytest.raises(ValueError):
            nodes._parse_plan_json('{"something_else": [1, 2]}')


# ---------- plan_node fallback ----------
def test_plan_node_falls_back_to_original_query_on_parse_failure():
    with patch.object(nodes, "_chat", return_value="not json at all"):
        out = nodes.plan_node({"query": "test question"})
    assert out == {"plan": ["test question"]}


# ---------- search_node dedup ----------
def test_search_node_dedupes_across_iterations():
    """An existing source URL should not be re-appended when Tavily returns the same URL again."""
    state = {
        "plan": ["a query"],
        "sources": [
            Source(url="https://example.com/a", title="A", snippet="", content="already-read")
        ],
    }
    fake_tavily_response = {
        "results": [
            {"url": "https://example.com/a", "title": "A", "content": "snip-a"},  # dup
            {"url": "https://example.com/b", "title": "B", "content": "snip-b"},  # new
        ]
    }

    class _FakeTavily:
        def search(self, query, max_results):  # noqa: ARG002
            return fake_tavily_response

    with patch.object(nodes, "_get_tavily", return_value=_FakeTavily()):
        out = nodes.search_node(state)

    urls = [s["url"] for s in out["sources"]]
    assert urls == ["https://example.com/a", "https://example.com/b"]
    # The pre-existing source's content is preserved (not overwritten by snippet)
    assert out["sources"][0]["content"] == "already-read"


# ---------- read_node budget + skip ----------
def test_read_node_skips_already_read_sources():
    """Sources whose content is non-empty must not be re-fetched or re-summarized."""
    state = {
        "query": "Q",
        "sources": [
            Source(url="u1", title="T1", snippet="", content="already-read-1"),
            Source(url="u2", title="T2", snippet="snippet-2", content=""),
        ],
        "notes": ["existing-note-from-prior-iter"],
    }

    fetch_calls = []

    def _fake_fetch(url):
        fetch_calls.append(url)
        return f"clean-text-for-{url}"

    def _fake_chat(model, prompt, max_tokens=300):  # noqa: ARG001
        return "fake-summary"

    with patch.object(nodes, "_fetch_clean_text", side_effect=_fake_fetch), patch.object(
        nodes, "_chat", side_effect=_fake_chat
    ):
        out = nodes.read_node(state)

    # Only u2 was fetched (u1 had content already)
    assert fetch_calls == ["u2"]
    # u1 still has its prior content
    assert out["sources"][0]["content"] == "already-read-1"
    # u2 was just enriched
    assert out["sources"][1]["content"].startswith("clean-text-for-u2")
    # Notes were appended (not overwritten)
    assert out["notes"][0] == "existing-note-from-prior-iter"
    assert "fake-summary" in out["notes"][1]


def test_read_node_respects_max_sources_budget():
    """Across iterations, total reads must never exceed MAX_SOURCES_READ."""
    already_read = [
        Source(url=f"u{i}", title=f"T{i}", snippet="", content=f"c{i}")
        for i in range(nodes.MAX_SOURCES_READ - 1)
    ]
    fresh = [
        Source(url=f"new{i}", title=f"NT{i}", snippet="s", content="")
        for i in range(3)  # 3 fresh sources, but budget allows only 1 more
    ]
    state = {"query": "Q", "sources": already_read + fresh, "notes": []}

    with patch.object(nodes, "_fetch_clean_text", return_value="X"), patch.object(
        nodes, "_chat", return_value="summary"
    ):
        out = nodes.read_node(state)

    read_count = sum(1 for s in out["sources"] if s.get("content"))
    assert read_count == nodes.MAX_SOURCES_READ


# ---------- critique_node iteration logic ----------
def test_critique_node_stops_at_max_iterations_even_with_gaps():
    """At MAX_ITERATIONS, critique must return empty critique to stop the loop."""
    state = {"query": "Q", "draft": "draft", "iterations": nodes.MAX_ITERATIONS - 1}
    with patch.object(nodes, "_chat", return_value="follow up A, follow up B"):
        out = nodes.critique_node(state)
    assert out["critique"] == ""
    assert out["iterations"] == nodes.MAX_ITERATIONS


def test_critique_node_returns_ok_when_critic_satisfied():
    state = {"query": "Q", "draft": "draft", "iterations": 0}
    with patch.object(nodes, "_chat", return_value="OK"):
        out = nodes.critique_node(state)
    assert out["critique"] == ""
    assert out["iterations"] == 1


def test_critique_node_loops_with_new_plan():
    state = {"query": "Q", "draft": "draft", "iterations": 0}
    with patch.object(nodes, "_chat", return_value="follow up A, follow up B"):
        out = nodes.critique_node(state)
    assert out["critique"]
    assert out["plan"] == ["follow up A", "follow up B"]
    assert out["iterations"] == 1


# ---------- should_continue ----------
def test_should_continue_loops_only_when_critique_non_empty():
    assert nodes.should_continue({"critique": "needs more"}) == "search"
    assert nodes.should_continue({"critique": ""}) == "end"
    assert nodes.should_continue({}) == "end"
