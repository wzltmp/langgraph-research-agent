"""LangGraph nodes for the research agent."""
from __future__ import annotations

import json
import os

import httpx
import trafilatura
from anthropic import Anthropic
from dotenv import load_dotenv
from tavily import TavilyClient

from src.state import AgentState, Source

load_dotenv()

CHEAP_MODEL = "claude-haiku-4-5-20251001"
WRITER_MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 2

_anthropic = Anthropic()


def _chat(model: str, prompt: str, max_tokens: int = 1024) -> str:
    msg = _anthropic.messages.create(
        model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


def plan_node(state: AgentState) -> AgentState:
    prompt = (
        "Break the user's research question into 3-5 specific, search-friendly sub-queries. "
        "Each sub-query must be a self-contained search phrase that a web search engine could answer — "
        "include exact entity names, dates, and technical terms from the question. "
        "Do NOT use vague phrases like 'recent developments' or 'overview of X'.\n\n"
        "GOOD example:\n"
        "Question: How has retrieval-augmented generation evolved between 2024 and 2026?\n"
        "{\"sub_queries\": [\n"
        "  \"RAG architecture changes 2024 vs 2025\",\n"
        "  \"new retrieval-augmented generation benchmarks 2025 2026\",\n"
        "  \"long-context LLMs replacing RAG 2025\",\n"
        "  \"hybrid search BM25 dense vector RAG 2025\"\n"
        "]}\n\n"
        "BAD example (too vague, would return off-topic results):\n"
        "{\"sub_queries\": [\"AI advancements\", \"recent NLP research\", \"language model trends\"]}\n\n"
        "Return ONLY JSON: {\"sub_queries\": [...]}\n\n"
        f"Question: {state['query']}"
    )
    raw = _chat(CHEAP_MODEL, prompt)
    plan = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])["sub_queries"]
    return {"plan": plan}


def search_node(state: AgentState) -> AgentState:
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    existing = list(state.get("sources", []))
    seen = {s["url"] for s in existing}
    for sq in state["plan"]:
        res = tavily.search(sq, max_results=3)
        for r in res["results"]:
            if r["url"] in seen:
                continue
            existing.append(
                Source(url=r["url"], title=r["title"], snippet=r.get("content", ""), content="")
            )
            seen.add(r["url"])
    return {"sources": existing}


def read_node(state: AgentState) -> AgentState:
    existing_notes = list(state.get("notes", []))
    sources = list(state["sources"])
    reads_done = sum(1 for s in sources if s.get("content"))
    budget = max(0, 8 - reads_done)  # global cap across iterations

    for idx, s in enumerate(sources[:8 + reads_done]):
        if s.get("content"):
            continue  # already read in a prior iteration
        if budget == 0:
            break
        budget -= 1
        try:
            html = httpx.get(s["url"], timeout=10, follow_redirects=True).text
            text = trafilatura.extract(html) or s["snippet"]
        except Exception:
            text = s["snippet"]
        text = text or ""
        sources[idx] = {**s, "content": text[:4000]}
        summary = _chat(
            CHEAP_MODEL,
            f"Summarize the key facts from this source in 3 bullets. Question context: {state['query']}\n\n{text[:4000]}",
            max_tokens=300,
        )
        existing_notes.append(f"({s['title']}) {summary}")
    return {"sources": sources, "notes": existing_notes}


def write_node(state: AgentState) -> AgentState:
    sources_block = "\n".join(
        f"[{i + 1}] {s['title']} -- {s['url']}" for i, s in enumerate(state["sources"][:8])
    )
    notes_block = "\n\n".join(state["notes"])
    prompt = (
        "Write a concise (250-400 word) report answering the question. "
        "Cite sources inline as [1], [2] using the numbering below. Be specific.\n\n"
        f"Question: {state['query']}\n\n"
        f"Sources:\n{sources_block}\n\n"
        f"Notes:\n{notes_block}"
    )
    draft = _chat(WRITER_MODEL, prompt, max_tokens=900)
    return {"draft": draft}


def critique_node(state: AgentState) -> AgentState:
    iterations = state.get("iterations", 0) + 1  # just finished this write→critique pass
    prompt = (
        "You are a tough editor. Read the draft and decide if it has unsupported claims, missing angles, "
        "or weak citations. If it's solid, respond with exactly 'OK'. Otherwise return 2-3 sub-queries "
        "(comma-separated) that would fill the gaps.\n\n"
        f"Question: {state['query']}\n\nDraft:\n{state['draft']}"
    )
    out = _chat(CHEAP_MODEL, prompt, max_tokens=200).strip()
    if out.upper().startswith("OK") or iterations >= MAX_ITERATIONS:
        return {"critique": "", "iterations": iterations}
    new_plan = [q.strip() for q in out.split(",") if q.strip()]
    return {"critique": out, "plan": new_plan, "iterations": iterations}


def should_continue(state: AgentState) -> str:
    """Conditional edge: re-plan if the critic isn't happy. critique_node is responsible for capping."""
    return "search" if state.get("critique") else "end"
