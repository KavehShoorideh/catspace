#!/usr/bin/env python
"""
catspace/research/components/planner/approaches/committor_value/experiments/value_fixed_point.py — V* vs V^pi on the KRRvKBP tablebase.

Kaveh, 2026-07-13: is the "blunder pulls a won position toward the loss pole" a
mislabel, or the CORRECT on-policy value? "Needs to be mathematically
investigated." KRRvKBP is a finite MDP with an exact Syzygy tablebase, so we can
compute both fixed points and look:

  V*(s)  = OPTIMAL hitting value (both sides play tablebase-optimal) = the
           game-theoretic win/draw/loss. White-POV: win 1.0 / draw 0.5 / loss 0.
           This is the Bellman-OPTIMALITY fixed point (max at each node).
  V^pi(s)= ON-POLICY value when WHITE plays eps-greedy over optimal (blunders a
           uniform-random legal move w.p. eps) and BLACK plays optimal. This is
           the Bellman-EXPECTATION fixed point (average over the policy's
           transitions). Estimated by Monte-Carlo rollout to absorption.

The MC result-label used in training is ONE sample of V^pi(behavior). Here we
average many rollouts to get V^pi itself, and sweep eps (the competence knob).

What we're testing:
  (1) Do the three outcome classes stay SEPARATED under V* (values pinned at
      1/0.5/0) and BLUR toward the middle under V^pi as eps grows? -> whether
      "regions apart" needs optimal value, and whether on-policy value smears them.
  (2) Is the gap V* - V^pi largest exactly on the hard/sharp states? -> whether
      the gap IS the competence/difficulty signal we've been chasing.
"""
from __future__ import annotations

import argparse
import base64
import sys
from functools import lru_cache
from pathlib import Path

import chess
import chess.syzygy
import numpy as np


from catspace.research.components.memory.approaches.experience_store.experiments.selfplay_generate import random_endgame_start


# MOVED to catspace/tb.py (2026-07-23 refactor) -- canonical home; these re-exports
# keep the ~10 existing cross-experiment imports working.
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, rollout, tb_best_move, white_pov_value  # noqa: F401
from catspace.io import paths


def v_pi(start, eps_white, tb, rng, n_rollouts):
    return float(np.mean([rollout(start, eps_white, tb, rng, ) for _ in range(n_rollouts)]))


def sample_positions(tb, rng, per_class, targets=(1.0, 0.5), max_tries=60000):
    """Random White-to-move KRRvKBP positions bucketed by V*. Default targets are
    win/draw only: our on-demand Syzygy set lacks the pawn-promotion results
    (KRRvKQ/KRRvKR...), so rollouts that promote leave coverage -- which corrupts
    draw/loss defence. WON positions convert by CAPTURE (into covered
    simplifications), so their on-policy value is measured cleanly."""
    buckets = {t: [] for t in targets}
    tries = 0
    while tries < max_tries and any(len(v) < per_class for v in buckets.values()):
        tries += 1
        b = random_endgame_start(rng, material="krrkbp")
        if b is None or b.turn != chess.WHITE:
            continue
        v = white_pov_value(b, tb)
        if v not in buckets or len(buckets[v]) >= per_class:
            continue
        buckets[v].append(b)
    return buckets


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--syzygy-dir", default=str(paths.syzygy_dir()))
    ap.add_argument("--per-class", type=int, default=30, help="positions per W/D/L class")
    ap.add_argument("--rollouts", type=int, default=40, help="MC rollouts per (position, eps)")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.35, 0.5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=paths.generated("value_fixed_point.png"))
    args = ap.parse_args()

    tb = TB(args.syzygy_dir)
    rng = np.random.default_rng(args.seed)
    buckets = sample_positions(tb, rng, args.per_class)
    classes = {1.0: "win", 0.5: "draw", 0.0: "loss"}
    positions, vstar = [], []
    for v, boards in buckets.items():
        for b in boards:
            positions.append(b); vstar.append(v)
    vstar = np.array(vstar)
    print(f"sampled positions: " +
          ", ".join(f"{classes[v]}={len(buckets[v])}" for v in (1.0, 0.5, 0.0)))

    # V^pi(eps) for every position
    vpi = np.zeros((len(positions), len(args.eps)))
    for j, e in enumerate(args.eps):
        for i, b in enumerate(positions):
            vpi[i, j] = v_pi(b, e, tb, np.random.default_rng([args.seed, i, j]), args.rollouts)
        # per-class mean at this eps
        means = {classes[v]: float(vpi[vstar == v, j].mean()) for v in (1.0, 0.5, 0.0)}
        gap = float(np.mean(vstar - vpi[:, j]))
        print(f"  eps={e:.2f}  mean V^pi by V*-class: "
              f"win={means['win']:.3f} draw={means['draw']:.3f} loss={means['loss']:.3f}  "
              f"| mean gap V*-V^pi = {gap:+.3f}")
    tb.close()

    _plot(args, positions, vstar, vpi, classes)


def _plot(args, positions, vstar, vpi, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {1.0: "#33aa55", 0.5: "#8b93a3", 0.0: "#d24b4b"}
    eps = args.eps
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#0f1115")

    # panel 1: mean V^pi per V*-class vs eps -> the "regions blur" curves
    ax = axes[0]; ax.set_facecolor("#0f1115")
    for v in (1.0, 0.5, 0.0):
        m = [vpi[vstar == v, j].mean() for j in range(len(eps))]
        ax.plot(eps, m, "-o", color=colors[v], label=f"{classes[v]} (V*={v})")
        ax.axhline(v, ls=":", color=colors[v], alpha=0.4)
    ax.set_xlabel("eps  (White blunder rate)", color="#e6e6e6")
    ax.set_ylabel("mean on-policy value  V^pi", color="#e6e6e6")
    ax.set_title("Do the three regions collapse as the policy weakens?", color="#e6e6e6")

    # panel 2: gap V*-V^pi vs V*  (is the gap the competence signal?)
    ax2 = axes[1]; ax2.set_facecolor("#0f1115")
    jmid = len(eps) // 2
    gap = vstar - vpi[:, jmid]
    for v in (1.0, 0.5, 0.0):
        mask = vstar == v
        jitter = (np.random.default_rng(1).random(mask.sum()) - 0.5) * 0.12
        ax2.scatter(vstar[mask] + jitter, gap[mask], s=18, c=colors[v], alpha=0.6,
                    label=f"{classes[v]}")
    ax2.set_xlabel("V*  (optimal value)", color="#e6e6e6")
    ax2.set_ylabel(f"competence gap  V* - V^pi  (eps={eps[jmid]})", color="#e6e6e6")
    ax2.set_title("Where is the gap largest?", color="#e6e6e6")

    for a in axes:
        a.tick_params(colors="#6b7280"); [s.set_color("#2a2e37") for s in a.spines.values()]
        leg = a.legend(framealpha=0.2)
        for t in leg.get_texts():
            t.set_color("#e6e6e6")
    fig.suptitle("V* (optimal) vs V^pi (on-policy) on the KRRvKBP tablebase",
                 color="#e6e6e6", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, facecolor="#0f1115"); plt.close(fig)
    b64 = base64.b64encode(out.read_bytes()).decode()
    out.with_suffix(".html").write_text(
        f"<!doctype html><meta charset=utf-8><body style='margin:0;background:#0f1115'>"
        f"<img style='max-width:100%' src='data:image/png;base64,{b64}'></body>")
    print(f"-> {out}\n-> {out.with_suffix('.html')}")


if __name__ == "__main__":
    main()
