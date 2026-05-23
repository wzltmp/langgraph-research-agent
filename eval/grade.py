"""Eval harness — run the agent and a Sonnet+web_search baseline on every
question in ``queries.jsonl``, score both with an LLM judge, emit a CSV summary.

Usage:
    python -m eval.grade                  # all 20 queries
    python -m eval.grade --limit 3        # first 3 only
    python -m eval.grade --ids f04,m05    # specific ids
    python -m eval.grade --skip-agent     # re-grade cached agent runs only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.graph import build_graph  # noqa: E402

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

JUDGE_MODEL = "claude-sonnet-4-6"
BASELINE_MODEL = "claude-sonnet-4-6"
BASELINE_MAX_SEARCHES = 5
BASELINE_MAX_TOKENS = 1500
JUDGE_MAX_TOKENS = 300

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


# ---------- agent + baseline runners ----------
def run_agent(query: str) -> dict:
    result = build_graph().invoke({"query": query})
    return {
        "draft": result.get("draft", ""),
        "plan": result.get("plan", []),
        "sources": [
            {"url": s["url"], "title": s["title"]} for s in result.get("sources", [])
        ],
        "iterations": result.get("iterations", 0),
    }


def run_baseline(query: str) -> dict:
    """Single Sonnet call with the built-in ``web_search`` tool — same model, different architecture."""
    msg = _get_client().messages.create(
        model=BASELINE_MODEL,
        max_tokens=BASELINE_MAX_TOKENS,
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": BASELINE_MAX_SEARCHES,
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"{query}\n\n"
                    "Write a 250-400 word answer. Cite sources inline as [1], [2] "
                    "using the web search results you find."
                ),
            }
        ],
    )

    text_parts: list[str] = []
    sources: list[dict] = []
    seen_urls: set[str] = set()
    web_searches = 0

    for block in msg.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
            for citation in getattr(block, "citations", None) or []:
                _maybe_add_source(citation, sources, seen_urls)
        elif btype == "server_tool_use":
            web_searches += 1
        elif btype == "web_search_tool_result":
            results = block.content if isinstance(block.content, list) else []
            for r in results:
                _maybe_add_source(r, sources, seen_urls)

    return {
        "draft": "".join(text_parts),
        "sources": sources,
        "web_searches": web_searches,
    }


def _maybe_add_source(obj: Any, sources: list[dict], seen_urls: set[str]) -> None:
    url = getattr(obj, "url", None)
    if not url or url in seen_urls:
        return
    sources.append({"url": url, "title": getattr(obj, "title", "") or ""})
    seen_urls.add(url)


# ---------- judge ----------
def judge(query: str, draft: str, sources: list[dict]) -> dict:
    sources_block = "\n".join(
        f"[{i + 1}] {s['title']} -- {s['url']}" for i, s in enumerate(sources)
    )
    prompt = (
        "You are evaluating a research-agent response. Score on three axes (integer 1-5):\n"
        "  factual_accuracy: claims supported by cited sources (5=every claim cited & plausible, 1=hallucinated)\n"
        "  completeness: does it actually answer the question (5=fully, 1=misses)\n"
        "  citation_quality: inline [n] citations used and source URLs look credible (5=clean, 1=missing/broken)\n\n"
        "Return ONLY JSON: "
        '{"factual_accuracy": N, "completeness": N, "citation_quality": N, "reason": "one short sentence"}\n\n'
        f"Question: {query}\n\n"
        f"Sources available to the responder:\n{sources_block or '(none)'}\n\n"
        f"Response:\n{draft or '(empty)'}"
    )
    msg = _get_client().messages.create(
        model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text
    return _extract_json(raw)


def _extract_json(raw: str) -> dict:
    """Strict JSON parse with a single brace-slice fallback for chatty models."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        return json.loads(raw[start : end + 1])
    raise json.JSONDecodeError("no JSON object found", raw, 0)


# ---------- driver ----------
def _load_queries(ids: str | None, limit: int | None) -> list[dict]:
    lines = (EVAL_DIR / "queries.jsonl").read_text().splitlines()
    queries = [json.loads(line) for line in lines if line.strip()]
    if ids:
        keep = set(ids.split(","))
        queries = [q for q in queries if q["id"] in keep]
    if limit:
        queries = queries[:limit]
    return queries


def _run_one(label: str, runner, query: str, cache_path: Path, use_cache: bool) -> dict:
    if use_cache and cache_path.exists():
        print(f"  [{label}] cached")
        return json.loads(cache_path.read_text())
    t0 = time.time()
    try:
        result = runner(query)
        result["elapsed_sec"] = time.time() - t0
    except Exception as exc:  # noqa: BLE001 — last-resort: capture, don't crash the loop
        result = {
            "draft": "",
            "sources": [],
            "error": repr(exc),
            "elapsed_sec": time.time() - t0,
        }
        print(f"  [{label}] FAILED: {exc}")
    cache_path.write_text(json.dumps(result, indent=2))
    return result


def _score_one(label: str, query: str, result: dict) -> dict:
    try:
        score = judge(query, result["draft"], result["sources"])
        print(
            f"  [{label} score] fa={score['factual_accuracy']} "
            f"cm={score['completeness']} cq={score['citation_quality']}"
        )
        return score
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"  [{label} judge] FAILED: {exc}")
        return {
            "factual_accuracy": 0,
            "completeness": 0,
            "citation_quality": 0,
            "reason": f"judge error: {exc!r}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N queries")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated list of query ids")
    parser.add_argument("--skip-agent", action="store_true", help="Re-grade cached agent runs only")
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Re-grade cached baseline runs only"
    )
    args = parser.parse_args()

    queries = _load_queries(args.ids, args.limit)
    rows = []

    for q in queries:
        qid = q["id"]
        print(f"\n=== {qid} ({q['category']}) ===\n{q['query']}")

        agent_path = RESULTS_DIR / f"agent-{qid}.json"
        baseline_path = RESULTS_DIR / f"baseline-{qid}.json"

        agent_result = _run_one("agent", run_agent, q["query"], agent_path, args.skip_agent)
        if "error" not in agent_result:
            print(
                f"  [agent] {len(agent_result['sources'])} sources, "
                f"{agent_result['elapsed_sec']:.1f}s, "
                f"iters={agent_result.get('iterations', '?')}"
            )

        baseline_result = _run_one(
            "baseline", run_baseline, q["query"], baseline_path, args.skip_baseline
        )
        if "error" not in baseline_result:
            print(
                f"  [baseline] {len(baseline_result['sources'])} sources, "
                f"{baseline_result.get('web_searches', '?')} searches, "
                f"{baseline_result['elapsed_sec']:.1f}s"
            )

        agent_score = _score_one("agent", q["query"], agent_result)
        baseline_score = _score_one("baseline", q["query"], baseline_result)

        must = q.get("must_mention", [])
        a_hit = sum(1 for m in must if m.lower() in agent_result["draft"].lower())
        b_hit = sum(1 for m in must if m.lower() in baseline_result["draft"].lower())

        rows.append(
            {
                "id": qid,
                "category": q["category"],
                "agent_fa": agent_score["factual_accuracy"],
                "agent_cm": agent_score["completeness"],
                "agent_cq": agent_score["citation_quality"],
                "agent_must": f"{a_hit}/{len(must)}" if must else "-",
                "agent_src": len(agent_result["sources"]),
                "agent_sec": round(agent_result.get("elapsed_sec", 0), 1),
                "base_fa": baseline_score["factual_accuracy"],
                "base_cm": baseline_score["completeness"],
                "base_cq": baseline_score["citation_quality"],
                "base_must": f"{b_hit}/{len(must)}" if must else "-",
                "base_src": len(baseline_result["sources"]),
                "base_sec": round(baseline_result.get("elapsed_sec", 0), 1),
                "agent_judge_reason": agent_score.get("reason", ""),
                "baseline_judge_reason": baseline_score.get("reason", ""),
            }
        )

    if not rows:
        print("No queries ran.")
        return

    summary_path = RESULTS_DIR / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def avg(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    print(f"\n\n=== AGGREGATE (n={len(rows)}) ===")
    print(f"  AGENT    fa={avg('agent_fa')}  cm={avg('agent_cm')}  cq={avg('agent_cq')}")
    print(f"  BASELINE fa={avg('base_fa')}  cm={avg('base_cm')}  cq={avg('base_cq')}")
    print(f"  wrote {summary_path}")


if __name__ == "__main__":
    main()
