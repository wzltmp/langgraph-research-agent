"""Streamlit UI for the LangGraph research agent — streams node-by-node progress."""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit Cloud runs the entrypoint as a script, not a package, so `from src.X`
# fails without this shim. Lesson from project 1.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from src.graph import build_graph

load_dotenv()

st.set_page_config(page_title="LangGraph Research Agent", page_icon=":mag:", layout="wide")

GITHUB_URL = "https://github.com/wzltmp/langgraph-research-agent"

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "A stateful research agent built with **LangGraph**. "
        "Given a question, it plans sub-queries, searches the web with Tavily, "
        "reads sources, drafts a cited report, and self-critiques up to 2 iterations."
    )
    st.markdown(f"[Source on GitHub]({GITHUB_URL})")
    st.markdown("---")
    st.markdown("### Graph")
    st.markdown(
        """```mermaid
graph TD
    A([plan]) --> B([search])
    B --> C([read])
    C --> D([write])
    D --> E{critique}
    E -- gap found --> B
    E -- OK --> F([END])
```"""
    )
    st.markdown("---")
    st.markdown("### Stack")
    st.markdown(
        "- LangGraph 1.x state machine\n"
        "- Claude Sonnet 4.6 (writer) + Haiku 4.5 (cheap nodes)\n"
        "- Tavily Search + trafilatura for clean article text\n"
        "- Streamlit Cloud"
    )

# ---- Main ----
st.title(":mag: LangGraph Research Agent")
st.caption("Type a research question. The agent plans, searches, reads, drafts, and critiques.")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

query = st.text_input(
    "Research question",
    placeholder="e.g., How has retrieval-augmented generation evolved from 2024 to 2026?",
)

run = st.button("Run", type="primary", disabled=not query)

if run and query:
    plan_box = st.empty()
    search_box = st.empty()
    read_box = st.empty()
    write_box = st.empty()
    critique_box = st.empty()

    final_state: dict = {}
    iteration_seen = 0

    with st.status("Researching...", expanded=True) as status:
        for chunk in st.session_state.graph.stream({"query": query}, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                final_state.update(node_state)

                if node_name == "plan":
                    plan_items = node_state.get("plan", [])
                    plan_box.markdown(
                        "**Plan**\n\n"
                        + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(plan_items))
                    )
                elif node_name == "search":
                    n = len(node_state.get("sources", []))
                    search_box.markdown(f"**Searched** — {n} unique sources collected so far.")
                elif node_name == "read":
                    sources = node_state.get("sources", [])
                    read = [s for s in sources if s.get("content")]
                    read_box.markdown(
                        f"**Read & summarized** — {len(read)} / {len(sources)} sources."
                    )
                elif node_name == "write":
                    draft = node_state.get("draft", "")
                    write_box.markdown("**Draft ready** — running critic next.")
                elif node_name == "critique":
                    iteration_seen = node_state.get("iterations", iteration_seen)
                    crit = node_state.get("critique", "")
                    if crit:
                        critique_box.markdown(
                            f"**Critic (iter {iteration_seen})** — found gaps, looping:\n\n> {crit}"
                        )
                    else:
                        critique_box.markdown(
                            f"**Critic (iter {iteration_seen})** — draft accepted."
                        )

        status.update(label=f"Done in {iteration_seen} iteration(s)", state="complete")

    # ---- Final report ----
    st.markdown("## Report")
    st.markdown(final_state.get("draft", "*(no draft produced)*"))

    sources = final_state.get("sources", [])
    if sources:
        with st.expander(f"Sources ({len(sources)})", expanded=False):
            for i, s in enumerate(sources[:8]):  # only first 8 are actually cited
                st.markdown(f"**[{i + 1}]** [{s['title']}]({s['url']})")
            if len(sources) > 8:
                st.caption(
                    f"({len(sources) - 8} additional sources were collected but not cited.)"
                )
