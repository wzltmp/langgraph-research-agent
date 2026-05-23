"""Eval harness: run agent + Sonnet+web_search baseline on queries.jsonl, LLM-judge both, emit summary CSV."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

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

_client = Anthropic()


def run_agent(query: str) -> dict:
    graph = build_graph()
    result = graph.invoke({"query": query})
    return {
        "draft": result.get("draft", ""),
        "plan": result.get("plan", []),
        "sources": [
            {"url": s["url"], "title": s["title"]} for s in result.get("sources", [])
        ],
        "iterations": result.get("iterations", 0),
    }


def run_baseline(query: str) -> dict:
    """Single Sonnet call with built-in web_search — apples-to-apples architecture comparison."""
    msg = _client.messages.create(
        model=BASELINE_MODEL,
        max_tokens=1500,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
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
            for c in (getattr(block, "citations", None) or []):
                url = getattr(c, "url", None)
                if url and url not in seen_urls:
                    sources.append({"url": url, "title": getattr(c, "title", "") or ""})
                    seen_urls.add(url)
        elif btype == "server_tool_use":
            web_searches += 1
        elif btype == "web_search_tool_result":
            results = block.content if isinstance(block.content, list) else []
            for r in results:
                url = getattr(r, "url", None)
                if url and url not in seen_urls:
                    sources.append({"url": url, "title": getattr(r, "title", "") or ""})
                    seen_urls.add(url)

    return {
        "draft": "".join(text_parts),
        "sources": sources,
        "web_searches": web_searches,
    }


def judge(query: str, draft: str, sources: list) -> dict:
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
    out = _client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = out.content[0].text
    return json.loads(raw[raw.find("{") : raw.rfind("}") + 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N queries")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated list of query ids to run")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    args = parser.parse_args()

    queries = [
        json.loads(line)
        for line in (EVAL_DIR / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if args.ids:
        keep = set(args.ids.split(","))
        queries = [q for q in queries if q["id"] in keep]
    if args.limit:
        queries = queries[: args.limit]

    rows = []
    for q in queries:
        qid = q["id"]
        print(f"\n=== {qid} ({q['category']}) ===\n{q['query']}")

        agent_path = RESULTS_DIR / f"agent-{qid}.json"
        baseline_path = RESULTS_DIR / f"baseline-{qid}.json"

        # --- AGENT ---
        if args.skip_agent and agent_path.exists():
            agent_result = json.loads(agent_path.read_text())
            print("  [agent] cached")
        else:
            t0 = time.time()
            try:
                agent_result = run_agent(q["query"])
                agent_result["elapsed_sec"] = time.time() - t0
                agent_path.write_text(json.dumps(agent_result, indent=2))
                print(
                    f"  [agent] {len(agent_result['sources'])} sources, "
                    f"{agent_result['elapsed_sec']:.1f}s, iters={agent_result.get('iterations', '?')}"
                )
            except Exception as e:
                agent_result = {
                    "draft": "",
                    "sources": [],
                    "error": str(e),
                    "elapsed_sec": time.time() - t0,
                }
                agent_path.write_text(json.dumps(agent_result, indent=2))
                print(f"  [agent] FAILED: {e}")

        # --- BASELINE ---
        if args.skip_baseline and baseline_path.exists():
            baseline_result = json.loads(baseline_path.read_text())
            print("  [baseline] cached")
        else:
            t0 = time.time()
            try:
                baseline_result = run_baseline(q["query"])
                baseline_result["elapsed_sec"] = time.time() - t0
                baseline_path.write_text(json.dumps(baseline_result, indent=2))
                print(
                    f"  [baseline] {len(baseline_result['sources'])} sources, "
                    f"{baseline_result['web_searches']} searches, "
                    f"{baseline_result['elapsed_sec']:.1f}s"
                )
            except Exception as e:
                baseline_result = {
                    "draft": "",
                    "sources": [],
                    "error": str(e),
                    "elapsed_sec": time.time() - t0,
                }
                baseline_path.write_text(json.dumps(baseline_result, indent=2))
                print(f"  [baseline] FAILED: {e}")

        # --- JUDGE ---
        try:
            agent_score = judge(q["query"], agent_result["draft"], agent_result["sources"])
            print(
                f"  [agent score] fa={agent_score['factual_accuracy']} "
                f"cm={agent_score['completeness']} cq={agent_score['citation_quality']}"
            )
        except Exception as e:
            agent_score = {
                "factual_accuracy": 0, "completeness": 0, "citation_quality": 0,
                "reason": f"judge error: {e}",
            }
            print(f"  [agent judge] FAILED: {e}")

        try:
            baseline_score = judge(
                q["query"], baseline_result["draft"], baseline_result["sources"]
            )
            print(
                f"  [baseline score] fa={baseline_score['factual_accuracy']} "
                f"cm={baseline_score['completeness']} cq={baseline_score['citation_quality']}"
            )
        except Exception as e:
            baseline_score = {
                "factual_accuracy": 0, "completeness": 0, "citation_quality": 0,
                "reason": f"judge error: {e}",
            }
            print(f"  [baseline judge] FAILED: {e}")

        must = q.get("must_mention", [])
        a_hit = sum(1 for m in must if m.lower() in agent_result["draft"].lower())
        b_hit = sum(1 for m in must if m.lower() in baseline_result["draft"].lower())

        rows.append({
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
        })

    if not rows:
        print("No queries ran.")
        return

    summary_path = RESULTS_DIR / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def avg(k):
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else 0

    print("\n\n=== AGGREGATE (n=%d) ===" % len(rows))
    print(f"  AGENT    fa={avg('agent_fa')}  cm={avg('agent_cm')}  cq={avg('agent_cq')}")
    print(f"  BASELINE fa={avg('base_fa')}  cm={avg('base_cm')}  cq={avg('base_cq')}")
    print(f"  wrote {summary_path}")


if __name__ == "__main__":
    main()
