"""Build and compile the research-agent graph.

Usage:
    python -m src.graph "<your research question>"
"""
from __future__ import annotations

import sys
from typing import Any

from langgraph.graph import END, StateGraph

from src.nodes import (
    critique_node,
    plan_node,
    read_node,
    search_node,
    should_continue,
    write_node,
)
from src.state import AgentState


def build_graph() -> Any:
    """Wire the plan→search→read→write→critique graph with a conditional loop edge.

    Returns:
        A compiled LangGraph runnable. Typed as ``Any`` because langgraph's
        ``CompiledStateGraph`` is generic in ways that don't add value here.
    """
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("search", search_node)
    g.add_node("read", read_node)
    g.add_node("write", write_node)
    g.add_node("critique", critique_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "read")
    g.add_edge("read", "write")
    g.add_edge("write", "critique")
    g.add_conditional_edges("critique", should_continue, {"search": "search", "end": END})

    return g.compile()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    default_q = "What changed in retrieval-augmented generation between 2024 and 2026?"
    query = sys.argv[1] if len(sys.argv) > 1 else default_q
    result = build_graph().invoke({"query": query})
    print(result.get("draft", "(no draft produced)"))
