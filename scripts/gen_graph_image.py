"""Regenerate ``assets/graph.png`` from the compiled LangGraph.

Run from the project root:
    python scripts/gen_graph_image.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph import build_graph  # noqa: E402


def main() -> None:
    out = ROOT / "assets" / "graph.png"
    out.parent.mkdir(exist_ok=True)
    png = build_graph().get_graph().draw_mermaid_png()
    out.write_bytes(png)
    print(f"wrote {out} ({len(png)} bytes)")


if __name__ == "__main__":
    main()
