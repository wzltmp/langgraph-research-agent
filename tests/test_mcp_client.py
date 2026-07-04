"""Offline unit tests for src/mcp_client.py — no real network calls to the MCP server."""
from __future__ import annotations

from unittest.mock import patch

from src import mcp_client


async def _fake_ok(url: str, n_bullets: int) -> str:
    del url, n_bullets
    return "- bullet one\n- bullet two"


async def _fake_raises(url: str, n_bullets: int) -> str:
    del url, n_bullets
    raise RuntimeError("boom")


def test_summarize_via_mcp_returns_summary_on_success():
    with patch.object(mcp_client, "_summarize_via_mcp_async", _fake_ok):
        result = mcp_client.summarize_via_mcp("https://example.com", n_bullets=2)
    assert result == "- bullet one\n- bullet two"


def test_summarize_via_mcp_returns_none_on_tool_error():
    with patch.object(mcp_client, "_summarize_via_mcp_async", _fake_raises):
        result = mcp_client.summarize_via_mcp("https://example.com")
    assert result is None
