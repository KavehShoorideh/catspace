#!/usr/bin/env python
"""
experiments/viz/merged_paper_figures.py — figures for the merged committor-
planner paper (writing/adversarial_reachability.md).

Two families:
  SCHEMATIC (design): two-pole geometry, region-necessity, component diagram,
    search-to-certainty, box-method plan. Ported/merged from the derivation
    draft's TikZ figures.
  DATA (measured): the wall-generated near-mate gradient, and the capacity
    forensics. Numbers are verbatim from journaled VERDICTs (sourced inline).

Output: writing/figures/mp_*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "writing" / "figures"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "axes.titlesize": 12, "svg.fonttype": "none",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/name}")


# ---------------------------------------------------------------- Fig 1
def fig_two_pole():
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.01",
                                fc="none", ec="#888", lw=1.2))
    # draw region (top band)
    ax.add_patch(mp.Ellipse((0.5, 0.83), 0.72, 0.16, fc="#dcdcdc", ec="none", alpha=0.8))
    ax.text(0.5, 0.83, "draws — fortresses, dead equality (far from both poles)",
            ha="center", va="center", fontsize=9, style="italic", color="#444")
    # basins
    ax.add_patch(mp.Ellipse((0.30, 0.42), 0.40, 0.44, fc="none", ec="#2c6fbb",
                            ls="--", lw=1.6))
    ax.add_patch(mp.Ellipse((0.70, 0.42), 0.40, 0.44, fc="none", ec="#c0392b",
                            ls="--", lw=1.6))
    ax.text(0.16, 0.66, "our winning basin", color="#2c6fbb", fontsize=9)
    ax.text(0.68, 0.66, "their winning basin", color="#c0392b", fontsize=9)
    # contested ridge
    ax.add_patch(mp.Ellipse((0.5, 0.42), 0.16, 0.30, fc="#555", ec="none", alpha=0.30))
    ax.text(0.5, 0.42, "contested\n(sharp):\nbasin\noverlap", ha="center", va="center",
            fontsize=8, color="#222")
    # poles
    ax.plot(0.30, 0.34, "*", ms=20, color="#2c6fbb")
    ax.plot(0.70, 0.34, "*", ms=20, color="#c0392b")
    ax.text(0.30, 0.24, r"$\mathcal{T}_P$ (we mate)", ha="center", color="#2c6fbb", fontsize=10)
    ax.text(0.70, 0.24, r"$\mathcal{T}_O$ (we are mated)", ha="center", color="#c0392b", fontsize=10)
    ax.annotate("", xy=(0.34, 0.34), xytext=(0.66, 0.34),
                arrowprops=dict(arrowstyle="<->", color="#999", lw=1))
    ax.text(0.5, 0.30, r"$d(\mathcal{T}_P\!\to\!\mathcal{T}_O)=\infty$ (absorption)",
            ha="center", fontsize=8, color="#777")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Two-pole committor geometry: basins may overlap (sharpness);\n"
                 "draws are a third region; poles separate automatically", fontsize=11)
    save(fig, "mp_two_pole.png")


# ---------------------------------------------------------------- Fig 2
def fig_region_necessity():
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.plot(0.08, 0.5, "o", ms=9, mfc="white", mec="#333")
    ax.text(0.08, 0.60, "s  (we move)", ha="center", fontsize=9)
    ax.plot(0.30, 0.5, "o", ms=9, mfc="#333", mec="#333")
    ax.text(0.30, 0.40, "s'  (they move)", ha="center", fontsize=9)
    ax.annotate("", xy=(0.29, 0.5), xytext=(0.10, 0.5),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
    ax.text(0.19, 0.545, "our forcing\nmove a", ha="center", fontsize=8)
    ys = [0.74, 0.5, 0.26]
    for i, y in enumerate(ys):
        ax.annotate("", xy=(0.60, y), xytext=(0.31, 0.5),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3))
        ax.text(0.45, (y + 0.5) / 2 + 0.01, f"$b_{i+1}$", color="#c0392b", fontsize=9)
        ax.plot(0.62, y, "o", ms=8, mfc="#2e8b57", mec="#2e8b57")
    ax.add_patch(mp.Ellipse((0.64, 0.5), 0.16, 0.62, fc="#2e8b57", ec="#2e8b57",
                            alpha=0.15, lw=1.5))
    ax.text(0.64, 0.06, "region G (forceable)", ha="center", color="#2e8b57", fontsize=9)
    ax.plot(0.90, 0.5, "o", ms=9, mfc="white", mec="#c0392b")
    ax.plot([0.87, 0.93], [0.47, 0.53], color="#c0392b", lw=1.6)
    ax.plot([0.87, 0.93], [0.53, 0.47], color="#c0392b", lw=1.6)
    ax.text(0.90, 0.38, "single g\n(NOT forceable)", ha="center", color="#c0392b", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.9); ax.axis("off")
    ax.set_title("Why subgoals must be regions: the defender chooses the reply, so play\n"
                 r"is steerable into the set $G$ but not to any single position $g$", fontsize=11)
    save(fig, "mp_region_necessity.png")


# ---------------------------------------------------------------- Fig 4
def fig_search_to_certainty():
    fig, ax = plt.subplots(figsize=(8.4, 4.2))

    def node(x, y, kind):
        c = {"g": "#222", "c": "#2e8b57", "o": "white"}[kind]
        e = {"g": "#222", "c": "#2e8b57", "o": "#888"}[kind]
        mk = "s" if kind == "g" else "o"
        ax.plot(x, y, mk, ms=11, mfc=c, mec=e, mew=1.5)

    edges = [((0.5, 0.92), (0.22, 0.72)), ((0.5, 0.92), (0.5, 0.72)),
             ((0.5, 0.92), (0.78, 0.72)),
             ((0.5, 0.72), (0.40, 0.5)), ((0.5, 0.72), (0.62, 0.5)),
             ((0.62, 0.5), (0.55, 0.28)), ((0.62, 0.5), (0.72, 0.28)),
             ((0.72, 0.28), (0.66, 0.08)), ((0.72, 0.28), (0.80, 0.08))]
    for a, b in edges:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-", color="#bbb", lw=1))
    node(0.5, 0.92, "o")
    node(0.22, 0.72, "c"); node(0.5, 0.72, "o"); node(0.78, 0.72, "c")
    node(0.40, 0.5, "g"); node(0.62, 0.5, "o")
    node(0.55, 0.28, "c"); node(0.72, 0.28, "o")
    node(0.66, 0.08, "g"); node(0.80, 0.08, "g")
    ax.text(0.05, 0.5, "quiet lines\nstop early", fontsize=9, color="#2e8b57", ha="center")
    ax.text(0.93, 0.16, "sharp lines\nrun to truth", fontsize=9, color="#222", ha="center")
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker="s", color="w", mfc="#222", label="grounded (mate/tablebase): exact"),
           Line2D([], [], marker="o", color="w", mfc="#2e8b57", label="confident leaf (H ≤ η): trusted"),
           Line2D([], [], marker="o", color="w", mfc="white", mec="#888", label="uncertain: keep expanding")]
    ax.legend(handles=leg, loc="upper left", fontsize=8, frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Search-to-certainty: variable-depth tree; backed-up root inherits leaf accuracy\n"
                 "(non-expansive amplification, Prop. H.2) — sharp deep, quiet shallow", fontsize=11)
    save(fig, "mp_search_to_certainty.png")


# ---------------------------------------------------------------- Fig 3
def fig_component():
    fig, ax = plt.subplots(figsize=(9.6, 4.8))

    def box(x, y, w, h, text, fc="#eef3fb", ec="#2c6fbb"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                    fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", color="#555",
                                     lw=1.1, mutation_scale=11))
    box(0.01, 0.44, 0.16, 0.13, "encoder  φ(s, ω)\n(wide, L1-sparse metric)", fc="#fdf0e8", ec="#c0392b")
    box(0.24, 0.66, 0.22, 0.13, "IQE quasimetric  d(s→G)\n(triangle ineq. by construction)")
    box(0.24, 0.26, 0.22, 0.13, "committor heads  d_W, d_D, d_L\n= −ln P(hit surface first)")
    box(0.53, 0.66, 0.20, 0.13, "search-to-certainty\n(exact dynamics)", fc="#eaf6ec", ec="#2e8b57")
    box(0.53, 0.44, 0.20, 0.11, "episodic memory\n(kNN, sandwich bounds)", fc="#eaf6ec", ec="#2e8b57")
    box(0.53, 0.24, 0.20, 0.12, "entropy / disagreement\n→ regime signal", fc="#f4eefb", ec="#7a3fb0")
    box(0.79, 0.44, 0.19, 0.13, "planner over regions\n+ meta-game repertoire", fc="#fdf0e8", ec="#c0392b")
    arrow((0.17, 0.52), (0.24, 0.70)); arrow((0.17, 0.50), (0.24, 0.34))
    arrow((0.46, 0.72), (0.53, 0.72)); arrow((0.46, 0.32), (0.53, 0.30))
    arrow((0.63, 0.66), (0.63, 0.55)); arrow((0.53, 0.49), (0.46, 0.44))
    arrow((0.63, 0.44), (0.63, 0.36)); arrow((0.73, 0.50), (0.79, 0.50))
    arrow((0.73, 0.30), (0.83, 0.44))
    ax.text(0.63, 0.60, "distill (slow)", fontsize=7, color="#2e8b57", ha="center")
    ax.set_xlim(0, 1); ax.set_ylim(0.18, 0.84); ax.axis("off")
    ax.set_title("Architecture: one encoder → quasimetric + committor heads; search over exact\n"
                 "dynamics grounds values into memory, distilled back into the field", fontsize=11)
    save(fig, "mp_component.png")


# ---------------------------------------------------------------- Fig 5 (DATA)
def fig_wall_gradient():
    # SOURCE: JOURNAL 2026-07-16 rim_staircase VERDICTs (KRRvk, tb-White eps=0.15)
    bins = ["1–2", "3–4", "5–6", "7–8"]
    x = np.arange(len(bins))
    phat = [0.868, 0.878, 0.765, 0.729]; phat_e = [0.163, 0.117, 0.132, 0.152]
    dW = [0.8702, 0.8967, 0.8781, 0.8618]; dW_e = [0.1116, 0.1009, 0.1064, 0.0834]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    a1.errorbar(x, phat, yerr=phat_e, fmt="o-", color="#2e8b57", capsize=4, lw=2)
    a1.set_xticks(x, bins); a1.set_ylim(0.55, 1.0)
    a1.set_xlabel("true distance to mate (|DTZ| bin)")
    a1.set_ylabel("empirical conversion  P̂  (own ε-play)")
    a1.set_title("TARGET: real, wall-generated gradient\nSpearman(−P̂, DTZ) = +0.29  CI[+0.16,+0.51]")
    a1.grid(alpha=0.3)
    a2.errorbar(x, dW, yerr=dW_e, fmt="s-", color="#c0392b", capsize=4, lw=2)
    a2.set_xticks(x, bins); a2.set_ylim(0.55, 1.0)
    a2.set_xlabel("true distance to mate (|DTZ| bin)")
    a2.set_ylabel("learned distance  d_W")
    a2.set_title("LEARNED FIELD: flat\nSpearman(d_W, DTZ) = −0.01  CI[−0.11,+0.09]")
    a2.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.suptitle("The near-mate gradient exists in the data but the learned field is flat — because "
                 "history-blind\naggregation / representation / search hide the draw walls that generate it "
                 "(KRR-vs-k, tablebase truth)", y=0.99, fontsize=11)
    save(fig, "mp_wall_gradient.png")


# ---------------------------------------------------------------- Fig 6 (DATA)
def fig_capacity():
    # SOURCE: JOURNAL 2026-07-16 capacity_forensics VERDICTs
    steps = [30, 60, 90, 120, 150]
    rank = [5.69, 6.19, 6.25, 6.81, 6.79]
    drift = [0.97, 1.17, 1.11, 1.13, 1.71]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    a1.plot(steps, rank, "o-", color="#2c6fbb", lw=2, ms=7)
    a1.axhline(64, color="#bbb", ls=":", lw=1); a1.text(90, 60, "embedding width = 64", fontsize=8, color="#999")
    a1.set_ylim(0, 68); a1.set_xlabel("training step (×1000)")
    a1.set_ylabel("effective rank of F")
    a1.set_title("Capacity never opens:\n~7 of 64 dims used, the whole run")
    a1.grid(alpha=0.3)
    a2.axhline(1.0, color="#444", lw=1)
    a2.plot(steps, drift, "s-", color="#c0392b", lw=2, ms=7)
    a2.annotate("rare regime dragged 1.7×\nby frequent-regime gradients",
                xy=(150, 1.71), xytext=(78, 1.55), fontsize=9, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
    a2.set_xlabel("training step (×1000)")
    a2.set_ylabel("feature-drift ratio  rare / common")
    a2.set_title("Rare regime is undefended collateral:\nlate gradients (zero rook info) overwrite it")
    a2.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.suptitle("Why play peaks early then degrades: a ~7-dim metric has no room to hold a rare "
                 "regime\nseparate from the frequent one — motivating a wide embedding with an L1-priced metric",
                 y=0.99, fontsize=11)
    save(fig, "mp_capacity.png")


if __name__ == "__main__":
    fig_two_pole()
    fig_region_necessity()
    fig_search_to_certainty()
    fig_component()
    fig_wall_gradient()
    fig_capacity()
    print("SOURCES: schematic figs are design; mp_wall_gradient + mp_capacity are "
          "verbatim from JOURNAL 2026-07-16 rim_staircase / capacity_forensics VERDICTs")
