"""Shared state passed between LangGraph nodes."""
from __future__ import annotations

from typing import TypedDict


class Source(TypedDict):
    url: str
    title: str
    snippet: str
    content: str  # cleaned full text (may be empty)


class AgentState(TypedDict, total=False):
    query: str
    plan: list[str]          # sub-queries
    sources: list[Source]
    notes: list[str]         # one per source
    draft: str               # current report draft
    critique: str            # critic's feedback ('' if happy)
    iterations: int          # how many plan→write loops so far
