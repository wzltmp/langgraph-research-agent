"""Smoke tests for the compiled LangGraph — no node execution."""
from __future__ import annotations

from src.graph import build_graph


def test_graph_compiles_and_has_expected_nodes():
    graph = build_graph()
    g = graph.get_graph()
    node_names = {n for n in g.nodes if n not in {"__start__", "__end__"}}
    assert node_names == {"plan", "search", "read", "write", "critique"}


def test_graph_entry_point_is_plan():
    graph = build_graph()
    g = graph.get_graph()
    successors_of_start = {e.target for e in g.edges if e.source == "__start__"}
    assert successors_of_start == {"plan"}


def test_graph_has_conditional_edge_from_critique():
    graph = build_graph()
    g = graph.get_graph()
    targets_from_critique = {e.target for e in g.edges if e.source == "critique"}
    # critique routes to either search (loop) or END
    assert "search" in targets_from_critique
    assert "__end__" in targets_from_critique
