"""experiments/viz/live_curves.py -- a tiny live training-curve dashboard so a long run is
WATCHABLE, not just logged (Kaveh's "measure but in a way I can also see"). The trainer calls
log_and_render(stem, step, metrics) at every probe: it appends the row to <stem>.jsonl and
re-renders <stem>.png + <stem>.html (auto-refreshing) so a browser tab shows the curves live.
No server needed -- open the .html and it refreshes itself.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def log_and_render(stem, step: int, metrics: dict, title: str = "", refresh: int = 8):
    """Append one row and re-render the dashboard. `metrics` is a flat {name: float} dict;
    each key gets its own panel plotted against step."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    row = {"step": int(step), **{k: (None if v is None else float(v)) for k, v in metrics.items()}}
    with open(stem.with_suffix(".jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")

    rows = [json.loads(l) for l in stem.with_suffix(".jsonl").read_text().splitlines() if l.strip()]
    keys = [k for k in rows[-1] if k != "step"]
    steps = [r["step"] for r in rows]
    n = len(keys)
    ncol = min(3, n); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.2 * nrow), squeeze=False,
                             facecolor="#0f1115")
    for i, k in enumerate(keys):
        ax = axes[i // ncol][i % ncol]; ax.set_facecolor("#0f1115")
        xs = [s for s, r in zip(steps, rows) if r.get(k) is not None]
        ys = [r[k] for r in rows if r.get(k) is not None]
        ax.plot(xs, ys, "-o", ms=3, color="#4fa3ff")
        if ys:
            ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), color="#e6e6e6", fontsize=9,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title(k, color="#e6e6e6", fontsize=10)
        ax.tick_params(colors="#6b7280", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#2a2e37")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{title}  (step {step})", color="#e6e6e6", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = stem.with_suffix(".png")
    fig.savefig(png, dpi=100, facecolor="#0f1115"); plt.close(fig)

    b64 = base64.b64encode(png.read_bytes()).decode()
    stem.with_suffix(".html").write_text(
        f"<!doctype html><html><head><meta charset=utf-8>"
        f"<meta http-equiv='refresh' content='{refresh}'>"
        f"<title>{title or stem.name}</title></head>"
        f"<body style='margin:0;background:#0f1115;text-align:center'>"
        f"<img style='max-width:100%' src='data:image/png;base64,{b64}'></body></html>")
    return png
