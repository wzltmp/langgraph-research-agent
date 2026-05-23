"""Shared state for the research-agent graph.

LangGraph passes a single ``AgentState`` dict between nodes. Each node returns
a *partial* dict whose keys overwrite the existing state. ``total=False`` lets
nodes return only the keys they actually update.
"""
from __future__ import annotations

from typing import TypedDict


class Source(TypedDict):
    """A single web source. ``content`` is empty until the read node fetches it."""

    url: str
    title: str
    snippet: str
    content: str


class AgentState(TypedDict, total=False):
    query: str
    plan: list[str]          # sub-queries from planner or critique
    sources: list[Source]    # accumulated across iterations, deduped by url
    notes: list[str]         # 3-bullet summary per source, one entry per read
    draft: str               # current report draft
    critique: str            # critic's feedback; empty string = draft accepted
    iterations: int          # completed write→critique passes (capped by MAX_ITERATIONS)
