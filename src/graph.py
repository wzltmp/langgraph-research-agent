"""Build the research agent graph."""
from __future__ import annotations

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


def build_graph():
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
    app = build_graph()
    result = app.invoke({"query": "What changed in retrieval-augmented generation between 2024 and 2026?"})
    print(result["draft"])
