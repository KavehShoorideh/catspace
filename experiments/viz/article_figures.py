#!/usr/bin/env python
"""
experiments/viz/article_figures.py — figures for the writing/ articles.

Provenance rule (journal_numbers_must_be_verdicts): every number here either
(a) is read live from a git-tracked artifact under artifacts/experiments/, or
(b) is a verdict copied VERBATIM from a printed script VERDICT line recorded
    in JOURNAL.md — each such number carries its source entry in SOURCED below.
No number is hand-derived or remembered.

Output: writing/figures/*.png (git-tracked; the articles embed them).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "artifacts" / "experiments"
OUT = ROOT / "writing" / "figures"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

# ---------------------------------------------------------------------------
# Verdict-sourced numbers (verbatim from JOURNAL.md printed VERDICT lines).
# Each entry: value(s) + the journal entry the printed verdict lives in.
# ---------------------------------------------------------------------------
SOURCED = {
    "regime_beam": {
        # incumbent lichess_fb_4gb_qm_plygap_only, KRRvKBP won starts,
        # deterministic tablebase defender, beam readout
        "nodes": [200, 800, 2000],
        "rate": [0.175, 0.325, 0.312],
        "source": "JOURNAL 2026-07-14 'PIVOTAL: SEARCH-LIMITED' + 'TWO REGIMES' "
                  "(200n=0.175, 800n=0.325, 2000n=0.312, n=80-120)",
    },
    "regime_mcts": {
        "nodes": [200, 800],
        "rate": [0.292, 0.388],
        "source": "JOURNAL 2026-07-14 'FIRST CI-REAL PLAY WIN' (MCTS 0.292@200n, "
                  "n=120, diff CI=[+0.042,+0.192]) + 800n leg (0.388, n=80)",
    },
    "scaling_toy": {
        # fixed-start own-play certainty tables, distill per size,
        # money test = paired MCTS 200n vs incumbent, n=120
        "states": [3100, 5200, 10000, 21000],
        "labels": ["K=4", "K=8", "K=16", "K=32"],
        "rho": [0.470, 0.395, 0.369, 0.370],
        "play_diff": [-0.092, -0.017, 0.167, 0.050],
        "k16_ci": [0.050, 0.275],
        "source": "JOURNAL 2026-07-14 'SCALING CURVE CROSSED' + 'Confirmatory: "
                  "K=16 ... winner's curse' (full curve verdict block)",
    },
    "scaling_followups": {
        # (label, diff, lo, hi, significant)
        "rows": [
            ("K=16 confirmatory @200n\n(fresh seed-778 set)", 0.050, -0.050, 0.150, False),
            ("K=16 regime look @800n", 0.225, 0.117, 0.333, True),
            ("K=16 confirmatory @800n\n(fresh seed-779 set)", 0.208, 0.108, 0.317, True),
        ],
        "source": "JOURNAL 2026-07-14 'Confirmatory: K=16 ... did NOT confirm' + "
                  "'CONFIRMED at 800n: certainty field promotion'",
    },
    "proxy_vs_play": {
        # nearest-exemplar KRvK rho vs KRRvKBP n=60 play, same set/seed
        "rows": [
            ("qm_wpov (incumbent r12)", 0.165, 0.558),
            ("qm_gen2 (endgame curriculum)", 0.252, 0.433),
            ("qm_asym015 (asymmetry hinge)", 0.121, 0.367),
            ("qm_gen3 (higher dose)", 0.154, 0.342),
            ("qm_plygap_only (promoted r18)", 0.256, 0.567),
        ],
        "source": "JOURNAL 2026-07-13 round-17 table ('nearest-exemplar rho does "
                  "not predict play') + round-18 recap (0.567); plygap rho from "
                  "artifacts/experiments/qm_fitness_qm_plygap_only.json",
    },
    "confirmed_effects": {
        # (label, diff, lo, hi) — paired mate-rate diffs, deterministic defender
        "rows": [
            ("MCTS readout vs beam @200n\n(confirmatory, seed-777)", 0.217, 0.133, 0.308),
            ("Certainty distill (10k states) @800n\n(confirmatory, seed-779)", 0.208, 0.108, 0.317),
            ("Closed-loop round 2 @1600n\n(confirmatory, seed-780)", 0.075, -0.025, 0.167),
            ("Pole fine-tune V6 (powered A/B)", -0.017, -0.092, 0.058),
            ("Region-bank goal readout", -0.112, -0.200, -0.025),
            ("Two-channel v1 (pure plies + S-head)\n@800n", -0.242, -0.342, -0.142),
        ],
        "source": "JOURNAL: 'MCTS readout CONFIRMED'; 'CONFIRMED at 800n'; 'GATE 2 "
                  "verdict'; 'powered playout confirms V6'; 'region-bank goal "
                  "CONFIRMED worse'; 'Two-channel v1 FALSIFIED'",
    },
}


def logx_ticks(ax, ticks):
    from matplotlib.ticker import NullLocator
    ax.set_xscale("log")
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())


def savefig(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
def fig_regimes():
    b, m = SOURCED["regime_beam"], SOURCED["regime_mcts"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(b["nodes"], b["rate"], "o-", color="#888", lw=2, ms=7, label="beam minimax readout")
    ax.plot(m["nodes"], m["rate"], "s-", color="#c0392b", lw=2, ms=8, label="MCTS readout (same field)")
    ax.axhspan(0.30, 0.35, color="#c9a13b", alpha=0.15)
    ax.text(1250, 0.362, "beam ceiling ~0.31–0.35 (embedding-limited)", fontsize=9,
            color="#7a6220", ha="center")
    ax.text(390, 0.215, "search-limited:\n4× budget ≈ 2× conversion", fontsize=9,
            color="#555", ha="center")
    ax.text(210, 0.415, "MCTS @200 ≈ beam @800\n(~4× compute efficiency)", fontsize=9,
            color="#c0392b")
    logx_ticks(ax, [200, 800, 2000])
    ax.set_xlim(170, 2700)
    ax.set_ylim(0.10, 0.45)
    ax.set_xlabel("search budget (fresh network evaluations per move)")
    ax.set_ylabel("mate conversion rate")
    ax.set_title("Same embedding, three search budgets, two search shapes\n"
                 "KRR vs KBP won positions, tablebase-optimal defender")
    ax.legend(loc="lower right", frameon=False)
    savefig(fig, "fig_regimes.png")


def fig_scaling():
    s, f = SOURCED["scaling_toy"], SOURCED["scaling_followups"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    a1.plot(s["states"], s["rho"], "o-", color="#2c6fbb", lw=2, ms=7)
    for x, y, lab in zip(s["states"], s["rho"], s["labels"]):
        dx = -30 if lab == "K=32" else 6
        a1.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, 6), fontsize=9)
    logx_ticks(a1, [3000, 5000, 10000, 21000])
    a1.set_xlabel("certainty-table size (own-play states)")
    a1.set_ylabel("held-out Spearman ρ (field vs certainty target)")
    a1.set_title("Field calibration: flat in data size")
    a1.set_ylim(0.3, 0.55)

    a2.axhline(0, color="#999", lw=1)
    a2.plot(s["states"], s["play_diff"], "o-", color="#2c6fbb", lw=2, ms=7,
            label="money test @200n (n=120)")
    k16 = s["states"][2]
    a2.errorbar([k16], [s["play_diff"][2]],
                yerr=[[s["play_diff"][2] - s["k16_ci"][0]], [s["k16_ci"][1] - s["play_diff"][2]]],
                fmt="none", ecolor="#2c6fbb", capsize=4)
    offs, cols, marks = [1.25, 1.55, 1.9], ["#888", "#c0392b", "#c0392b"], ["v", "^", "D"]
    for (lab, d, lo, hi, sig), off, c, mk in zip(f["rows"], offs, cols, marks):
        x = k16 * off
        a2.errorbar([x], [d], yerr=[[d - lo], [hi - d]], fmt=mk, color=c, ms=7, capsize=4,
                    label=lab.replace("\n", " "))
    for x, y, lab in zip(s["states"], s["play_diff"], s["labels"]):
        dx = -26 if lab == "K=32" else -2
        a2.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, 8), fontsize=9)
    logx_ticks(a2, [3000, 5000, 10000, 21000])
    a2.set_xlim(2600, 30000)
    a2.set_xlabel("certainty-table size (own-play states)")
    a2.set_ylabel("paired conversion diff vs incumbent")
    a2.set_title("Play: crosses zero at ~10k states,\nconfirms only at the 800-node regime")
    a2.legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Data-scaling curve for certainty distillation (fixed-start toy, MCTS readout)",
                 y=1.02, fontsize=12)
    savefig(fig, "fig_scaling.png")


def fig_horizon():
    files = [
        ("qm_wpov", "incumbent (round 12)", "#2c6fbb", "o"),
        ("qm_asym015", "+ asymmetry hinge (round 16)", "#c0392b", "s"),
        ("qm_plygap_only", "promoted incumbent (round 18)", "#2e8b57", "^"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, label, color, mark in files:
        d = json.loads((EXP / f"qm_fitness_{name}.json").read_text())["horizon_retrieval"]
        ks = sorted(int(k.split("=")[1]) for k in d)
        acc = [d[f"k={k}"]["acc"] for k in ks]
        ax.plot(ks, acc, mark + "-", color=color, lw=2, ms=6, label=label)
    chance = json.loads((EXP / "qm_fitness_qm_wpov.json").read_text())[
        "horizon_retrieval"]["k=1"]["chance"]
    ax.axhline(chance, color="#999", lw=1, ls="--")
    ax.text(1.05, chance + 0.02, "chance (1/64)", fontsize=9, color="#777")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.set_xlabel("horizon k (plies into the future)")
    ax.set_ylabel("retrieval accuracy (true future among 63 decoys)")
    ax.set_title("How far ahead the field can see: sharp to ~10 plies, cliff by 50\n"
                 "(the asymmetry hinge traded the k=1 sharpness play depends on for the tail)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=False)
    savefig(fig, "fig_horizon.png")


def fig_sharpness():
    import chess
    import chess.syzygy
    from scipy.stats import spearmanr

    rows = json.loads((EXP / "sharpness_table.json").read_text())["rows"]
    tb = chess.syzygy.open_tablebase(str(ROOT / "data" / "syzygy"))
    S, dtz, ex = [], [], []
    for r in rows:
        ex.append(r["existence"])
        try:
            z = tb.probe_dtz(chess.Board(r["fen"]))
        except (KeyError, ValueError):
            continue
        S.append(max(r["S"], 0.0))
        dtz.append(abs(z))
    tb.close()
    S, dtz, ex = np.array(S), np.array(dtz), np.array(ex)
    rho, _ = spearmanr(S, dtz)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    a1.scatter(dtz, S, s=6, alpha=0.15, color="#2c6fbb", edgecolors="none")
    bins = np.array([0, 2, 4, 6, 9, 13, 20, 40])
    mids, meds = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (dtz >= lo) & (dtz < hi)
        if m.sum() >= 20:
            mids.append((lo + hi) / 2)
            meds.append(np.median(S[m]))
    a1.plot(mids, meds, "o-", color="#c0392b", lw=2, ms=6, label="median S per |DTZ| bin")
    a1.set_xlabel("|DTZ| — true distance to conversion (tablebase)")
    a1.set_ylabel("identified sharpness S (nats per unit ε)")
    a1.set_title(f"Risk does not accumulate with distance\nSpearman ρ(S, |DTZ|) = {rho:+.3f}"
                 f"  (n={len(S)})")
    a1.set_ylim(-0.5, 25)
    a1.legend(frameon=False)

    a2.hist(ex, bins=60, range=(-1, 2), color="#2c6fbb", alpha=0.75)
    a2.axvline(0, color="#444", lw=1.5, ls="--")
    a2.axvline(np.median(ex), color="#c0392b", lw=1.5)
    a2.text(np.median(ex) + 0.05, a2.get_ylim()[1] * 0.9,
            f"median {np.median(ex):+.3f}", color="#c0392b", fontsize=9)
    a2.text(0.02, a2.get_ylim()[1] * 0.97, "truth (path exists) = 0", fontsize=9, color="#444")
    a2.set_xlabel("existence intercept (−ln P̂ extrapolated to ε → 0)")
    a2.set_ylabel("states")
    a2.set_title("Path-existence identification: rankable, biased slightly up")
    fig.suptitle("Multi-ε identification: −ln P̂ ≈ existence + S·ε on 4,373 tb-White rollout states",
                 y=1.02, fontsize=12)
    savefig(fig, "fig_sharpness.png")


def fig_proxy_vs_play():
    rows = SOURCED["proxy_vs_play"]["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for lab, rho, play in rows:
        promoted = "promoted" in lab or "incumbent" in lab
        ax.scatter([rho], [play], s=90, color="#2e8b57" if promoted else "#c0392b",
                   zorder=3)
        ax.annotate(lab.split(" (")[0], (rho, play), textcoords="offset points",
                    xytext=(8, -4), fontsize=9)
    ax.annotate("best calibration of its era,\nplays 0.12 below the incumbent",
                xy=(0.252, 0.433), xytext=(0.20, 0.48), fontsize=9, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
    ax.set_xlabel("nearest-exemplar KRvK calibration ρ (structural instrument)")
    ax.set_ylabel("KRR vs KBP conversion (n=60, same set)")
    ax.set_title("Structural calibration dissociates from play\n"
                 "(green = promoted incumbents; red = never promoted)")
    ax.set_xlim(0.08, 0.31)
    savefig(fig, "fig_proxy_vs_play.png")


def fig_confirmed_effects():
    rows = SOURCED["confirmed_effects"]["rows"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, d, lo, hi) in zip(ys, rows):
        sig = lo > 0 or hi < 0
        color = ("#2e8b57" if d > 0 else "#c0392b") if sig else "#888"
        ax.errorbar([d], [y], xerr=[[d - lo], [hi - d]], fmt="o", color=color,
                    ms=8, capsize=4, lw=2)
    ax.axvline(0, color="#444", lw=1)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(-0.40, 0.40)
    ax.set_xlabel("paired conversion difference vs incumbent (95% CI)")
    ax.set_title("Every intervention, judged the same way\n"
                 "paired playouts vs tablebase-optimal defender; green/red = CI excludes zero")
    savefig(fig, "fig_confirmed_effects.png")


if __name__ == "__main__":
    fig_regimes()
    fig_scaling()
    fig_horizon()
    fig_sharpness()
    fig_proxy_vs_play()
    fig_confirmed_effects()
    print("SOURCES:")
    for k, v in SOURCED.items():
        print(f"  {k}: {v['source']}")
