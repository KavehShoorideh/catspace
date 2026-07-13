#!/usr/bin/env python
"""
experiments/viz/build_gallery.py — scan artifacts/generated/ for *.html and
write index.html: a dark-styled list of links with mtimes and descriptions.
No payload/template step needed (this page has no data to inject) -- rerun
after any builder to refresh the index.
"""
from __future__ import annotations

import datetime
from pathlib import Path

from catspace.io.paths import generated_dir

DESCRIPTIONS = {
    "training-dashboard.html": "Loss / top1 / top8 / throughput curves parsed from training logs.",
    "fullboard-viewer.html": "Real-board games: reach + eval-head curves ply-by-ply on a projected embedding map.",
    "decision-viewer.html": "FBBoardPolicy candidate moves, scores, and feared opponent replies per ply.",
    "decompose-viewer.html": "M1.5 meet-in-the-middle decomposition trees over real middlegames.",
    "embedding-atlas.html": "Holdout F-embedding projection, step 2000 vs step 30000, switchable coloring.",
    "divergence-explorer.html": "Descriptive vs normative eval divergence (trap-position finder).",
    "eval-dashboard.html": "ROC / reliability / per-ply / per-Elo AUC for the F/B/FB eval-head ablation.",
    "krk-viewer.html": "Toy KRk endgame cone viewer (pre-real-board milestone).",
    "krkn-linked-viewer-pca.html": "Toy KRkn linked cone + PCA map viewer (pre-real-board milestone).",
}

STYLE = """
body { background:#14161a; color:#cfd3da; font:14px/1.5 -apple-system, "Segoe UI", sans-serif; margin:0; padding:32px; }
h1 { font-size:18px; color:#e8eaee; margin:0 0 20px; }
.card { background:#1b1e24; border:1px solid #2a2e36; border-radius:8px; padding:14px 18px; margin-bottom:10px; }
.card a { color:#7aa2ff; font-size:15px; text-decoration:none; font-weight:600; }
.card a:hover { text-decoration:underline; }
.desc { color:#8a90a0; font-size:12px; margin-top:4px; }
.meta { color:#5a6070; font-size:11px; margin-top:4px; }
.empty { color:#8a90a0; }
"""


def main():
    out_dir = generated_dir()
    htmls = sorted((p for p in out_dir.glob("*.html") if p.name != "index.html"),
                   key=lambda p: p.stat().st_mtime, reverse=True)

    cards = []
    for p in htmls:
        mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes")
        size_kb = p.stat().st_size / 1024
        desc = DESCRIPTIONS.get(p.name, "")
        cards.append(f'<div class="card"><a href="{p.name}">{p.name}</a>'
                    f'<div class="desc">{desc}</div>'
                    f'<div class="meta">updated {mtime}  ·  {size_kb:.0f} KB</div></div>')

    body = "\n".join(cards) if cards else '<div class="empty">no viewers built yet</div>'
    html = (f'<!doctype html><html><head><meta charset="utf-8">'
           f'<title>catspace — viz gallery</title><style>{STYLE}</style></head>'
           f'<body><h1>catspace — viz gallery ({len(htmls)})</h1>{body}</body></html>')

    out = out_dir / "index.html"
    out.write_text(html)
    print(f"wrote {out}  ({len(htmls)} viewers)")


if __name__ == "__main__":
    main()
