"""catspace/research/tools/figures/figlib.py -- shared figure style for the probe suite (dataviz-skill
conventions: recessive grid/axes, thin marks, one accent hue, validated
categorical order, sequential = single hue, text in ink not series color).

Palette validated 2026-07-30 (six-check validator, light surface): all PASS.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CAT = ["#3B6BA5", "#D45D2C", "#2FA089", "#9C5BD1"]      # fixed order, never cycled
ACCENT = CAT[0]
INK = "#333333"
MUTED = "#8a8a8a"
GRID = "#e6e6e3"
SURFACE = "#fcfcfb"
SEQ_CMAP = "Blues"                                        # sequential: one hue

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11,
    "lines.linewidth": 2.0, "figure.dpi": 150,
})


def new_fig(ncols=1, nrows=1, w=4.2, h=3.0):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    return fig, axes


def save(fig, path, title=None):
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")
