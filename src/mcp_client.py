"""Client for the deployed MCP Automations server's ``summarize_url`` tool.

``read_node`` tries this first instead of running its own fetch+summarize
logic (see README "Key design decisions"). Any failure here — network error,
tool error, timeout — must not raise; callers fall back to the local path.
"""
from __future__ import annotations

import asyncio
import logging
import os

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

log = logging.getLogger(__name__)

MCP_AUTOMATIONS_URL = os.environ.get(
    "MCP_AUTOMATIONS_URL", "https://mcp-automations.fly.dev/mcp"
)
MCP_CALL_TIMEOUT_SEC = 10


def summarize_via_mcp(url: str, n_bullets: int = 3) -> str | None:
    """Call the deployed MCP Automations server's ``summarize_url`` tool.

    Returns the bullet summary on success, or ``None`` on any failure so the
    caller can fall back to a local implementation.
    """
    try:
        return asyncio.run(_summarize_via_mcp_async(url, n_bullets))
    except Exception:
        log.warning("MCP summarize_url call failed for %s; falling back", url, exc_info=True)
        return None


async def _summarize_via_mcp_async(url: str, n_bullets: int) -> str:
    async with streamablehttp_client(
        MCP_AUTOMATIONS_URL, timeout=MCP_CALL_TIMEOUT_SEC
    ) as (read_stream, write_stream, _), ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        result = await session.call_tool(
            "summarize_url", {"url": url, "n_bullets": n_bullets}
        )
        if result.isError:
            raise RuntimeError(f"summarize_url tool error: {result.content}")
        structured = result.structuredContent
        if not structured or "summary" not in structured:
            raise RuntimeError("summarize_url returned no structured summary")
        return str(structured["summary"])
