"""LangGraph nodes for the research agent.

The graph is ``plan → search → read → write → critique`` with a conditional
edge from ``critique`` back to ``search`` for up to :data:`MAX_ITERATIONS`
passes. Each node reads from and writes to a shared :class:`AgentState`.

Sub-queries are searched concurrently via a thread pool, which is the largest
single latency win in this pipeline.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import trafilatura
from anthropic import Anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv
from tavily import TavilyClient

from src.exceptions import EmptyLLMResponseError, PlanParseError
from src.mcp_client import summarize_via_mcp
from src.state import AgentState, Source

load_dotenv()

log = logging.getLogger(__name__)

# ---- Models ----
CHEAP_MODEL = "claude-haiku-4-5-20251001"
WRITER_MODEL = "claude-sonnet-4-6"

# ---- Tunables ----
MAX_ITERATIONS = 2
TAVILY_RESULTS_PER_QUERY = 3
MAX_SOURCES_READ = 8
CONTENT_CHAR_LIMIT = 4000
FETCH_TIMEOUT_SEC = 10
SEARCH_PARALLELISM = 5

# ---- Lazy clients (instantiated on first use) ----
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
        EmptyLLMResponseError: when the response has no text content (refusal/empty).
    """
    msg = _get_anthropic().messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if isinstance(block, TextBlock):
            return block.text
    raise EmptyLLMResponseError(
        f"no text block in response (model={model}, stop_reason={msg.stop_reason!r})"
    )


def _parse_plan_json(raw: str) -> list[str]:
    """Extract ``sub_queries`` from the planner's reply.

    Tries strict JSON first, then falls back to brace-slicing for chatty models.
    """
    for candidate in (raw, raw[raw.find("{") : raw.rfind("}") + 1] if "{" in raw else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("sub_queries"), list):
            return [str(q) for q in parsed["sub_queries"]]
    raise PlanParseError(f"could not parse sub_queries from: {raw!r}")


def _fetch_clean_text(url: str) -> str:
    """Best-effort HTML fetch + trafilatura extract. Returns empty string on any network error."""
    try:
        response = httpx.get(url, timeout=FETCH_TIMEOUT_SEC, follow_redirects=True)
    except (httpx.HTTPError, OSError) as exc:
        log.info("fetch failed for %s: %s", url, exc)
        return ""
    return trafilatura.extract(response.text) or ""


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
    except PlanParseError as exc:
        log.warning("planner output unparseable, falling back to original query: %s", exc)
        plan = [state["query"]]
    log.info("planned %d sub-queries", len(plan))
    return {"plan": plan}


def search_node(state: AgentState) -> AgentState:
    """Tavily-search each sub-query in parallel; merge results with state, dedupe by URL."""
    tavily = _get_tavily()
    sub_queries = state["plan"]
    sources = list(state.get("sources", []))
    seen: set[str] = {s["url"] for s in sources}

    def _search(query: str) -> dict[str, Any]:
        result: dict[str, Any] = tavily.search(query, max_results=TAVILY_RESULTS_PER_QUERY)
        return result

    workers = min(len(sub_queries), SEARCH_PARALLELISM) or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = list(pool.map(_search, sub_queries))

    new_count = 0
    for response in responses:
        for r in response.get("results", []):
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
            new_count += 1
    log.info("search added %d new sources (%d total)", new_count, len(sources))
    return {"sources": sources}


def read_node(state: AgentState) -> AgentState:
    """Fetch + extract + summarize fresh sources, skipping any already read.

    Caps total reads at :data:`MAX_SOURCES_READ` across all iterations.
    """
    sources: list[Source] = [Source(**s) for s in state["sources"]]
    notes = list(state.get("notes", []))
    already_read = sum(1 for s in sources if s["content"])
    budget = MAX_SOURCES_READ - already_read

    for idx, source in enumerate(sources):
        if budget <= 0:
            break
        if source["content"]:
            continue

        mcp_summary = summarize_via_mcp(source["url"], n_bullets=3)
        if mcp_summary is not None:
            sources[idx] = Source(
                url=source["url"],
                title=source["title"],
                snippet=source["snippet"],
                content=mcp_summary[:CONTENT_CHAR_LIMIT],
            )
            notes.append(f"({source['title']}) {mcp_summary}")
            budget -= 1
            continue

        text = _fetch_clean_text(source["url"]) or source["snippet"] or ""
        sources[idx] = Source(
            url=source["url"],
            title=source["title"],
            snippet=source["snippet"],
            content=text[:CONTENT_CHAR_LIMIT],
        )
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

    log.info(
        "read %d new sources (budget remaining: %d)",
        MAX_SOURCES_READ - already_read - budget,
        budget,
    )
    return {"sources": sources, "notes": notes}


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
    draft = _chat(WRITER_MODEL, prompt, max_tokens=900)
    log.info("wrote draft (%d chars)", len(draft))
    return {"draft": draft}


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
        log.info("critique accepted at iteration %d", iterations)
        return {"critique": "", "iterations": iterations}

    new_plan = [q.strip() for q in out.split(",") if q.strip()]
    log.info("critique found gaps at iteration %d, re-planning with %d sub-queries", iterations, len(new_plan))
    return {"critique": out, "plan": new_plan, "iterations": iterations}


def should_continue(state: AgentState) -> str:
    """Conditional edge — loop only if :func:`critique_node` returned a non-empty critique."""
    return "search" if state.get("critique") else "end"
