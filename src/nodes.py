"""LangGraph nodes for the research agent.

The graph is ``plan → search → read → write → critique`` with a conditional
edge from ``critique`` back to ``search`` for up to ``MAX_ITERATIONS`` passes.
Each node reads from and writes to a shared :class:`AgentState`.
"""
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

# ---- Models ----
CHEAP_MODEL = "claude-haiku-4-5-20251001"
WRITER_MODEL = "claude-sonnet-4-6"

# ---- Tunables ----
MAX_ITERATIONS = 2          # write→critique passes before forced stop
TAVILY_RESULTS_PER_QUERY = 3
MAX_SOURCES_READ = 8        # cap across all iterations to control cost / latency
CONTENT_CHAR_LIMIT = 4000   # cap text passed to summarizer per source
FETCH_TIMEOUT_SEC = 10


# ---- Lazy clients (module-level globals, instantiated on first use) ----
_anthropic: Anthropic | None = None
_tavily: TavilyClient | None = None


def _get_anthropic() -> Anthropic:
    global _anthropic
    if _anthropic is None:
        _anthropic = Anthropic()
    return _anthropic


def _get_tavily() -> TavilyClient:
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _tavily


# ---- Helpers ----
def _chat(model: str, prompt: str, max_tokens: int = 1024) -> str:
    """Single-turn chat returning the first text content block.

    Raises:
        RuntimeError: if the response has no text content (refusal or empty).
    """
    msg = _get_anthropic().messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise RuntimeError(f"No text block in response (stop_reason={msg.stop_reason!r})")


def _parse_plan_json(raw: str) -> list[str]:
    """Extract ``sub_queries`` from the planner's reply.

    Tries strict JSON first, then falls back to brace-slicing. The function is
    deliberately liberal because the planner is a cheap (Haiku) model.
    """
    try:
        return json.loads(raw)["sub_queries"]
    except (json.JSONDecodeError, KeyError):
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])["sub_queries"]
        except (json.JSONDecodeError, KeyError):
            pass
    raise ValueError(f"could not parse sub_queries from: {raw!r}")


# ---- Nodes ----
def plan_node(state: AgentState) -> AgentState:
    """Decompose the user's question into 3-5 search-friendly sub-queries."""
    prompt = (
        "Break the user's research question into 3-5 specific, search-friendly sub-queries. "
        "Each sub-query must be a self-contained search phrase that a web search engine could answer — "
        "include exact entity names, dates, and technical terms from the question. "
        "Do NOT use vague phrases like 'recent developments' or 'overview of X'.\n\n"
        "GOOD example:\n"
        "Question: How has retrieval-augmented generation evolved between 2024 and 2026?\n"
        '{"sub_queries": [\n'
        '  "RAG architecture changes 2024 vs 2025",\n'
        '  "new retrieval-augmented generation benchmarks 2025 2026",\n'
        '  "long-context LLMs replacing RAG 2025",\n'
        '  "hybrid search BM25 dense vector RAG 2025"\n'
        "]}\n\n"
        "BAD example (too vague, would return off-topic results):\n"
        '{"sub_queries": ["AI advancements", "recent NLP research", "language model trends"]}\n\n'
        'Return ONLY JSON: {"sub_queries": [...]}\n\n'
        f"Question: {state['query']}"
    )
    try:
        plan = _parse_plan_json(_chat(CHEAP_MODEL, prompt))
    except ValueError:
        plan = [state["query"]]  # fallback: search the original question directly
    return {"plan": plan}


def search_node(state: AgentState) -> AgentState:
    """Tavily-search each sub-query; merge new sources with existing, dedupe by URL."""
    tavily = _get_tavily()
    sources = list(state.get("sources", []))
    seen: set[str] = {s["url"] for s in sources}
    for sub_query in state["plan"]:
        results = tavily.search(sub_query, max_results=TAVILY_RESULTS_PER_QUERY)
        for r in results.get("results", []):
            if r["url"] in seen:
                continue
            sources.append(
                Source(
                    url=r["url"],
                    title=r["title"],
                    snippet=r.get("content", ""),
                    content="",
                )
            )
            seen.add(r["url"])
    return {"sources": sources}


def read_node(state: AgentState) -> AgentState:
    """Fetch + extract + summarize fresh sources, skipping any already read.

    Caps total reads at :data:`MAX_SOURCES_READ` across all iterations.
    """
    sources = [dict(s) for s in state["sources"]]
    notes = list(state.get("notes", []))
    budget = MAX_SOURCES_READ - sum(1 for s in sources if s.get("content"))

    for idx, source in enumerate(sources):
        if budget <= 0:
            break
        if source.get("content"):
            continue
        text = _fetch_clean_text(source["url"]) or source["snippet"] or ""
        sources[idx]["content"] = text[:CONTENT_CHAR_LIMIT]
        summary = _chat(
            CHEAP_MODEL,
            (
                "Summarize the key facts from this source in 3 bullets. "
                f"Question context: {state['query']}\n\n{text[:CONTENT_CHAR_LIMIT]}"
            ),
            max_tokens=300,
        )
        notes.append(f"({source['title']}) {summary}")
        budget -= 1
    return {"sources": sources, "notes": notes}


def _fetch_clean_text(url: str) -> str:
    """Best-effort HTML fetch + trafilatura extract. Returns empty string on any network error."""
    try:
        html = httpx.get(url, timeout=FETCH_TIMEOUT_SEC, follow_redirects=True).text
    except (httpx.HTTPError, OSError):
        return ""
    return trafilatura.extract(html) or ""


def write_node(state: AgentState) -> AgentState:
    """Synthesize the cited report from the source summaries."""
    sources_block = "\n".join(
        f"[{i + 1}] {s['title']} -- {s['url']}"
        for i, s in enumerate(state["sources"][:MAX_SOURCES_READ])
    )
    notes_block = "\n\n".join(state["notes"])
    prompt = (
        "Write a concise (250-400 word) report answering the question. "
        "Cite sources inline as [1], [2] using the numbering below. Be specific.\n\n"
        f"Question: {state['query']}\n\n"
        f"Sources:\n{sources_block}\n\n"
        f"Notes:\n{notes_block}"
    )
    return {"draft": _chat(WRITER_MODEL, prompt, max_tokens=900)}


def critique_node(state: AgentState) -> AgentState:
    """Decide if the draft is acceptable. Increments iterations; caps at :data:`MAX_ITERATIONS`."""
    iterations = state.get("iterations", 0) + 1
    prompt = (
        "You are a tough editor. Read the draft and decide if it has unsupported claims, "
        "missing angles, or weak citations. If it's solid, respond with exactly 'OK'. "
        "Otherwise return 2-3 sub-queries (comma-separated) that would fill the gaps.\n\n"
        f"Question: {state['query']}\n\nDraft:\n{state['draft']}"
    )
    out = _chat(CHEAP_MODEL, prompt, max_tokens=200).strip()
    if out.upper().startswith("OK") or iterations >= MAX_ITERATIONS:
        return {"critique": "", "iterations": iterations}
    new_plan = [q.strip() for q in out.split(",") if q.strip()]
    return {"critique": out, "plan": new_plan, "iterations": iterations}


def should_continue(state: AgentState) -> str:
    """Conditional edge — loop only if :func:`critique_node` returned a non-empty critique."""
    return "search" if state.get("critique") else "end"
