#!/usr/bin/env python
"""experiments/basin_tent_fullgames.py -- Kaveh 2026-08-03: replay WHOLE games ply-by-ply and draw
their trajectories descending the tent.

Why this is not just a nicer version of basin_tent.py. The stored dataset is a per-game SUBSAMPLE
(--stride 6 --per-game 8 --tail 4), which has two consequences that full replay removes outright:

  * the ply axis is a COMB with a fixed phase, so density plots alias and trajectories are jagged;
  * stride samples stop at ply 42, so past ply ~54 the dataset is 100% game-endings and the ply
    axis silently swaps population.

A replayed game has EVERY ply, so neither applies: the ply axis here is complete and honest for
its whole length, and a trajectory is a real continuous path rather than 8 scattered dots.

Cost: this must run the frozen trunk itself (these positions are not in the precomputed feature
cache), so it is ~700 positions/sec. That is why it samples games rather than replaying all
200,000 of them -- a few hundred games is ~20k positions and takes well under a minute.

Games are sampled STRATIFIED by result, deliberately: SF-vs-SF is 81% draws, so a uniform sample
would show almost no decisive engine games and the comparison of how the two populations reach a
result would be lost.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from catspace.research.tools.training_infra.losses import basin_logp, WIN, DRAW, LOSS
from catspace.research.tools.embeddings.basin_tent import white_pov_x, COLOR_WHITE_WIN, COLOR_BLACK_WIN
from catspace.research.tools.embeddings.basin_simplex_chart import INK, MUTED, COLOR_DRAW

RESULT_NAME = {1: "White won", 0: "drawn", -1: "Black won"}


def parity_smooth(x, w=2):
    """Centered box filter of width w=2. Chosen for a reason, not for looks: measured lag-1
    autocorrelation of x along a game is FAR lower than lag-2 (+0.42 vs +0.63 human, +0.22 vs
    +0.57 SF), i.e. positions two plies apart -- same side to move -- agree much better than
    adjacent ones. A width-2 mean cancels a pure ply-parity alternation exactly while leaving the
    real drift untouched. The raw path is still drawn underneath so the jitter is visible, not
    hidden: the field is trained on positions 6 plies apart with NO temporal-smoothness term, so
    nothing ever asked consecutive plies to agree, and that is a finding rather than a nuisance."""
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")
RESULT_COLOR = {1: COLOR_WHITE_WIN, 0: COLOR_DRAW, -1: COLOR_BLACK_WIN}

# Endpoint SHAPE encodes HOW the game ended; colour still encodes WHO won. Two independent
# encodings, so neither is carried by colour alone.
ENDING_MARKER = {
    "checkmate": "X", "resign": "v", "flagged": "s", "stalemate": "D",
    "threefold": "^", "fifty-move": "*", "insufficient": "h",
    "agreement": "o", "adjudicated": "P", "truncated": "|", "unknown": ".",
}


def classify_ending(board, termination, result, hit_cap):
    """How did this game end? Board state first, metadata second.

    The two sources carry different evidence and neither alone is enough:
      * human (lichess) has a `termination` column, but it only distinguishes 'Time forfeit' from
        'Normal' -- and 'Normal' lumps mate, stalemate, repetition, fifty-move and RESIGNATION
        together. Resignation is therefore an INFERENCE: a decisive 'Normal' game whose final
        position is not checkmate. That is what resigning is in a move list -- the loser simply
        stops -- so it cannot be read off the board directly, only deduced.
      * SF-vs-SF has no metadata at all; gen_opening_pool_sfsf.py runs to
        `is_game_over(claim_draw=True)` or a ply cap, so engines never resign or flag, and the cap
        gives an 'adjudicated' class that has no human counterpart.
    """
    if board.is_checkmate():
        return "checkmate"
    if termination == "Time forfeit":
        return "flagged"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient"
    # can_claim_*, NOT is_repetition(3) / is_fifty_moves(). This distinction is the whole reason
    # engine draws were landing in "agreement", which is nonsense -- engines never agree to a draw.
    # gen_opening_pool_sfsf.py stops on is_game_over(claim_draw=True), i.e. as soon as a draw is
    # CLAIMABLE, which includes the case where the claim only becomes available AFTER a legal move.
    # is_repetition(3) asks the strictly narrower question "has this exact position already
    # occurred three times", so it misses precisely the positions the generator stopped on.
    if board.can_claim_threefold_repetition():
        return "threefold"
    if board.can_claim_fifty_moves():
        return "fifty-move"
    if hit_cap:
        return "adjudicated"
    if result != 0:
        return "resign"                                    # decisive, not mate, not flagged
    return "agreement"


def sf_games(tsv, per_result, rng):
    """[(gid, result, [uci,...])] stratified by result."""
    buckets = {1: [], 0: [], -1: []}
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            buckets[int(p[1])].append((int(p[0]), int(p[1]), p[2].split(), ""))
    out = []
    for r, b in buckets.items():
        if b:
            out += [b[i] for i in rng.choice(len(b), min(per_result, len(b)), replace=False)]
    return out


def human_games(records_dir, per_result, rng):
    import pyarrow.parquet as pq
    buckets = {1: [], 0: [], -1: []}
    for shard in sorted(Path(records_dir).glob("records_*.parquet")):
        d = pq.read_table(shard, columns=["game_id", "result", "moves", "termination"]).to_pydict()
        for gid, r, mv, tm in zip(d["game_id"], d["result"], d["moves"], d["termination"]):
            if len(buckets[int(r)]) < per_result * 8:
                buckets[int(r)].append((int(gid), int(r), mv.split(), tm))
        if all(len(b) >= per_result * 8 for b in buckets.values()):
            break
    out = []
    for r, b in buckets.items():
        if b:
            out += [b[i] for i in rng.choice(len(b), min(per_result, len(b)), replace=False)]
    return out


def uniform_games(loader_out, n, rng):
    """n games drawn UNIFORMLY from a stratified loader's pool, restoring population proportions."""
    idx = rng.choice(len(loader_out), min(n, len(loader_out)), replace=False)
    return [loader_out[i] for i in idx]


def replay(ucis, max_ply):
    """UCI list -> ((P,112,8,8) uint8 planes one per ply, final board, whether the cap truncated)."""
    import chess
    from lczerolens import LczeroBoard
    b = LczeroBoard()
    out = []
    for ply, u in enumerate(ucis[:max_ply]):
        try:
            b.push(chess.Move.from_uci(u))
        except Exception:
            break
        out.append(b.to_input_tensor().to(dtype=torch.uint8).numpy())
    truncated = len(ucis) > max_ply                        # we stopped early, the GAME did not end
    return (np.asarray(out) if out else None), b, truncated


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/iqe_poles_both_latest.pt")
    ap.add_argument("--onnx", default="assets/engines/lc0/t1-256x10.onnx")
    ap.add_argument("--sf-moves", default="data/derived/opening_pool_sfsf_moves.tsv")
    ap.add_argument("--human-records", default="data/records/lichess_2019-01")
    ap.add_argument("--per-result", type=int, default=40,
                    help="games per outcome per source for the TRAJECTORY figure (stratified)")
    ap.add_argument("--n-uniform", type=int, default=1200,
                    help="games per source, sampled UNIFORMLY, for the DENSITY tent. Uniform, not "
                         "stratified: the trajectory figure over-samples decisive engine games on "
                         "purpose so the splitting is visible, but a density must be "
                         "population-representative or the draw column is understated.")
    ap.add_argument("--max-ply", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_fullgames")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField
    field = ReachabilityField(onnx=args.onnx, head=args.ckpt)
    if not field.has_poles:
        raise SystemExit(f"{args.ckpt} has no trained poles")
    print(f"field loaded [{time.time()-t0:.0f}s]", flush=True)

    rng = np.random.default_rng(args.seed)
    sources = {"human": human_games(args.human_records, args.per_result, rng),
               "SF-vs-SF": sf_games(args.sf_moves, args.per_result, rng)}

    traj = {}
    for name, games in sources.items():
        rows = []
        for gid, res, ucis, tm in games:
            planes, endboard, truncated = replay(ucis, args.max_ply)
            if planes is None or len(planes) < 6:
                continue
            xs = []
            with torch.no_grad():
                for i in range(0, len(planes), args.batch):
                    phi = field.phi_from_planes(list(planes[i:i + args.batch].astype(np.float32)))
                    p = basin_logp(field.head.d_poles(phi), field.head.temperature).exp().cpu().numpy()
                    xs.append(p)
            p = np.concatenate(xs)
            ply = np.arange(len(p))
            ending = ("truncated" if truncated else
                      classify_ending(endboard, tm, res, hit_cap=False))
            rows.append(dict(gid=gid, result=res, ply=ply, x=white_pov_x(p, ply), ending=ending))
        traj[name] = rows
        n_pos = sum(len(r["ply"]) for r in rows)
        print(f"  {name}: {len(rows)} full games, {n_pos:,} positions "
              f"(median {int(np.median([len(r['ply']) for r in rows]))} plies) "
              f"[{time.time()-t0:.0f}s]", flush=True)

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: full trajectories descending the tent -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True, sharex=True)
    for ax, (name, rows) in zip(axes, traj.items()):
        for r in rows:
            c = RESULT_COLOR[r["result"]]
            ax.plot(r["x"], r["ply"], "-", color=c, lw=0.4, alpha=0.13)     # raw, jitter visible
            ax.plot(parity_smooth(r["x"]), r["ply"], "-", color=c, lw=0.9, alpha=0.6)
            ax.plot(r["x"][-1], r["ply"][-1], ENDING_MARKER.get(r["ending"], "."),
                    color=c, ms=6, alpha=0.95, mec="black", mew=0.4, ls="none")
        ax.axvline(0, color=MUTED, lw=0.7, ls=":")
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("P(White wins) - P(Black wins)")
        ax.set_title(f"{name}  ({len(rows)} whole games)", color=INK)
    axes[0].set_ylabel("ply  (start at the top, game descends)")
    axes[0].set_ylim(args.max_ply, 0)
    from matplotlib.lines import Line2D
    res_h = [Line2D([], [], color=RESULT_COLOR[k], lw=2, label=RESULT_NAME[k]) for k in (1, 0, -1)]
    seen = [e for e in ENDING_MARKER if any(r["ending"] == e for rows in traj.values() for r in rows)]
    end_h = [Line2D([], [], color=INK, marker=ENDING_MARKER[e], ls="none", ms=6,
                    mec="black", mew=0.4, label=e) for e in seen]
    axes[0].legend(handles=res_h, fontsize=8, frameon=False, loc="lower left", title="who won")
    axes[1].legend(handles=end_h, fontsize=8, frameon=False, loc="lower right", title="how it ended")
    fig.suptitle("Whole games descending the tent -- every ply replayed, no subsampling\n"
                 "bold = 2-ply smoothed (cancels the ply-parity alternation); faint = raw per-ply")
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}_1_trajectories.png", dpi=140)

    # ---- Figure 2: envelope -- how wide is the tent, from COMPLETE trajectories -------------
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.6))
    bins = np.arange(0, args.max_ply + 1, 5)
    for name, rows in traj.items():
        c = "#2a78d6" if name == "human" else "#e34948"
        allply = np.concatenate([r["ply"] for r in rows])
        allx = np.abs(np.concatenate([r["x"] for r in rows]))
        mid, med, hi = [], [], []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (allply >= a) & (allply < b)
            if m.sum() >= 40:
                mid.append((a + b) / 2); med.append(np.median(allx[m])); hi.append(np.quantile(allx[m], 0.9))
        ax2.plot(mid, med, "-", color=c, lw=2, label=f"{name} median")
        ax2.plot(mid, hi, "--", color=c, lw=1.1, label=f"{name} 90th pct")
    ax2.set_xlabel("ply"); ax2.set_ylabel("|P(White wins) - P(Black wins)|"); ax2.set_ylim(0, 1)
    ax2.set_title("Tent width from COMPLETE games -- no stride comb, no population swap", color=INK)
    ax2.legend(fontsize=8, frameon=False)
    fig2.tight_layout(); fig2.savefig(f"{args.out_prefix}_2_envelope.png", dpi=140)

    print("\nHOW GAMES ENDED (sampled games; stratified by result, so not population shares)")
    import collections
    allk = []
    for name, rows in traj.items():
        allk += [r["ending"] for r in rows]
    cats = [e for e in ENDING_MARKER if e in set(allk)]
    print(f"  {'ending':>14s} " + " ".join(f"{n:>10s}" for n in traj))
    for e in cats:
        print(f"  {e:>14s} " + " ".join(
            f"{sum(1 for r in rows if r['ending']==e):>10d}" for rows in traj.values()))

    print("\nTENT WIDTH from complete games (median |x|)")
    print(f"  {'ply':>10s} {'human':>9s} {'SF-vs-SF':>9s}")
    for a, b in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120), (120, args.max_ply)]:
        row = []
        for name, rows in traj.items():
            allply = np.concatenate([r["ply"] for r in rows])
            allx = np.abs(np.concatenate([r["x"] for r in rows]))
            m = (allply >= a) & (allply < b)
            row.append(np.median(allx[m]) if m.sum() >= 40 else np.nan)
        print(f"  {a:>4d}-{b:<5d} {row[0]:>9.3f} {row[1]:>9.3f}")
    np.savez(f"{args.out_prefix}_data.npz", **{
        f"{n}_{i}_{k}": v for n, rows in traj.items() for i, r in enumerate(rows)
        for k, v in [("ply", r["ply"]), ("x", r["x"]), ("result", np.array([r["result"]]))]})
    print(f"wrote {args.out_prefix}_{{1_trajectories,2_envelope}}.png [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
