#!/usr/bin/env python
"""experiments/basin_tb_anchors.py -- Kaveh's ask (2026-08-02): use the exactly
6-piece Syzygy tablebase classes as ANCHORS (ground-truth win/draw/loss
endgame outcomes under perfect play) and test whether, under near-perfect
SF-vs-SF play, positions organize into ~3 basins matching those anchors --
using the ACTUAL trained asymmetric quasimetric distance (IQE's own
`pairwise()`), not a symmetric UMAP/TICA projection of it. Kaveh: "forcing an
asymmetric distance into Euclidean just breaks it... I don't wanna [retrain
for cones]... what I really wanna find is the places where we cross the
boundaries of the basins."

Scope, per Kaveh's explicit call (2026-08-02): only exactly-6-piece classes
("stop at exactly six... we don't care about [going] below six"), and only
the 4 that are ALREADY local (`KBNPvKR`, `KRRvKBN`, `KRRvKBP`, `KRRvKNP`) --
NOT the full 150GB 6-piece Syzygy set (Kaveh chose "proceed with the 4 we
have" over downloading more).

Anchors are mined, not hand-picked: for each of the 4 material classes,
generate random LEGAL positions of that exact material signature, probe the
local tablebase (data/syzygy), and for decisive ones roll out tablebase-
optimal play (both sides, `catspace.research.components.planner.approaches.endgame_groundtruth.src.tb.rollout_line`) to
the ACTUAL mate -- that mate position is the anchor, not an arbitrary
mid-class position. Drawn instances keep the position itself (a drawn TB
position doesn't converge onward). Caveat stated once, applies throughout:
these anchor FENs have no real game history before them, so their φ is
computed from a "cold" 8-ply-history-of-just-itself board -- same convention
UCI engines use for "position fen X" with no move list, but a real (if
usually small) train/anchor distribution mismatch vs. positions reached via
an actual game.

Core measurement: d(s -> anchor) via the field's OWN trained asymmetric
IQE.pairwise(), for every sampled SF-vs-SF / human position against every
anchor. Two things done with that (N, K) matrix:
  A. Cluster the K anchors by correlation of their distance-PROFILE columns
     (do positions that are close to anchor i tend to also be close to
     anchor j?) and check whether the resulting groups agree with the
     anchors' OWN ground-truth TB label -- this is the actual "do they roll
     up into 3 basins" test, and the ground truth is free (it's the TB).
  B. Per-ply distance-to-every-anchor trajectories for sample games -- the
     literal "crossing a basin boundary" event is where argmin_k d(s->anchor_k)
     changes label. Pure scalar-distance plot, no projection at all.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, rollout_line   # noqa: E402
from experiments.basin_umap_compare import (   # noqa: E402
    COLOR_WIN, COLOR_DRAW, COLOR_LOSS, load_sample,
    select_n_human_games_full, select_n_sf_games_full, replay_full_game,
)

PIECE_TYPE = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK,
              "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}
SIX_PIECE_CLASSES = ["KBNPvKR", "KRRvKBN", "KRRvKBP", "KRRvKNP"]


def random_position(material_sig, rng, max_tries=300):
    """A random LEGAL board with exactly this material signature (e.g.
    'KRRvKBN' -> White K,R,R / Black K,B,N), or None if max_tries exhausted."""
    white_syms, black_syms = material_sig.split("v")
    for _ in range(max_tries):
        board = chess.Board.empty()
        squares = rng.permutation(64)
        idx = 0
        ok = True
        for sym in white_syms:
            sq = int(squares[idx]); idx += 1
            if sym == "P" and chess.square_rank(sq) in (0, 7):
                ok = False; break
            board.set_piece_at(sq, chess.Piece(PIECE_TYPE[sym], chess.WHITE))
        if not ok:
            continue
        for sym in black_syms:
            sq = int(squares[idx]); idx += 1
            if sym == "P" and chess.square_rank(sq) in (0, 7):
                ok = False; break
            board.set_piece_at(sq, chess.Piece(PIECE_TYPE[sym], chess.BLACK))
        if not ok:
            continue
        board.turn = bool(rng.integers(2))
        board.castling_rights = 0
        if board.is_valid():
            return board
    return None


def mine_anchors(classes, tb, rng, n_tries_per_class=400, target_per_label=6):
    """-> [(fen, white_pov_label, material_class), ...]. Mines random legal
    positions, probes the TB, rolls decisive ones out to the actual mate
    (tb-optimal both sides), keeps drawn ones as-is. Labels are white-POV
    +1/0/-1, capped at `target_per_label` per (class, label) to keep the
    anchor set balanced rather than skewed toward whichever label is easiest
    to randomly sample."""
    anchors = []
    for cls in classes:
        got = {1: 0, 0: 0, -1: 0}
        tries = 0
        skipped_rollout_fail = 0
        while tries < n_tries_per_class and sum(got.values()) < 3 * target_per_label:
            tries += 1
            b = random_position(cls, rng)
            if b is None:
                continue
            w, _ = tb.wdl_dtz(b)
            if w is None:
                continue
            mover_label = 1 if w > 0 else (-1 if w < 0 else 0)
            if mover_label != 0:
                line = rollout_line(b, tb, cap=120)
                if line is None:
                    skipped_rollout_fail += 1
                    continue
                final = line[-1]
                outcome = final.outcome(claim_draw=True)
                if outcome is None or outcome.winner is None:
                    continue
                white_label = 1 if outcome.winner == chess.WHITE else -1
                fen = final.fen()
            else:
                white_label = 0
                fen = b.fen()
            if got[white_label] >= target_per_label:
                continue
            anchors.append((fen, white_label, cls))
            got[white_label] += 1
        print(f"    {cls}: {got} (tries={tries}, rollout-fails={skipped_rollout_fail})", flush=True)
    return anchors


def main():
    t0 = time.time()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed = 0
    rng = np.random.default_rng(seed)

    print("[1/4] mining TB anchors (exactly-6-piece classes, local set only) ...", flush=True)
    tb = TB()
    anchors = mine_anchors(SIX_PIECE_CLASSES, tb, rng)
    tb.close()
    print(f"  total anchors: {len(anchors)} [{time.time()-t0:.0f}s]", flush=True)

    print("[2/4] computing phi for anchors + density samples (SF + human) ...", flush=True)
    from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField
    from lczerolens import LczeroBoard
    field = ReachabilityField()
    anchor_planes = np.stack([LczeroBoard(fen).to_input_tensor().numpy() for fen, _, _ in anchors])
    phi_anchors = field.phi_from_planes(list(anchor_planes.astype(np.float32))).cpu().numpy()
    anchor_labels = np.array([lab for _, lab, _ in anchors])
    anchor_classes = [cls for _, _, cls in anchors]

    n_density = 15000
    human = load_sample("data/derived/field_std_v1.npz", n_density, seed, 0)
    sf = load_sample("data/derived/opening_pool_sfsf.npz", n_density, seed, 1)
    phi_human = field.phi_from_planes(list(human["planes"].astype(np.float32))).cpu().numpy()
    phi_sf = field.phi_from_planes(list(sf["planes"].astype(np.float32))).cpu().numpy()
    phi_pool = np.concatenate([phi_human, phi_sf], 0)
    print(f"  anchors phi {phi_anchors.shape}, pool phi {phi_pool.shape} [{time.time()-t0:.0f}s]", flush=True)

    print("[3/4] real asymmetric distance d(s -> anchor) via the trained IQE.pairwise() ...", flush=True)
    import torch
    with torch.no_grad():
        D = field.head.iqe.pairwise(
            torch.as_tensor(phi_pool, device=field.dev),
            torch.as_tensor(phi_anchors, device=field.dev),
        ).cpu().numpy()   # (N, K): d(pool_i -> anchor_k)
    print(f"  D {D.shape} [{time.time()-t0:.0f}s]", flush=True)

    # ==== A. do the K anchors roll up into ~3 basins? =====================
    print("[4/4] clustering anchors by distance-profile correlation ...", flush=True)
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform
    corr = np.corrcoef(D.T)   # (K,K): how similarly two anchors are approached across the pool
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2, 0, None)
    Z = linkage(squareform(dist, checks=False), method="average")
    cluster3 = fcluster(Z, t=3, criterion="maxclust")
    # agreement check: does the 3-cluster split line up with the TRUE anchor label?
    from collections import Counter
    print("    3-cluster composition (true label breakdown per cluster):")
    for c in sorted(set(cluster3)):
        labs = anchor_labels[cluster3 == c]
        print(f"      cluster {c}: n={len(labs)} labels={dict(Counter(labs.tolist()))}")

    label_color = {1: COLOR_WIN, 0: COLOR_DRAW, -1: COLOR_LOSS}
    fig1, ax1 = plt.subplots(figsize=(12, 5.5))
    leaf_labels = [f"{cls}#{i}" for i, (_, _, cls) in enumerate(anchors)]
    dn = dendrogram(Z, ax=ax1, labels=leaf_labels, leaf_rotation=90, leaf_font_size=6,
                     link_color_func=lambda k: "#8a8985")
    for lbl, leaf_idx in zip(ax1.get_xticklabels(), dn["leaves"]):
        lbl.set_color(label_color[int(anchor_labels[leaf_idx])])
    ax1.set_title("Do the TB anchors roll up into ~3 basins?\n"
                   "hierarchical clustering of anchors by correlation of their d(s->anchor) profile\n"
                   "across 30k sampled SF+human positions -- leaf color = TRUE tablebase outcome\n"
                   "(green=White wins, gray=draw, red=Black wins) -- ground truth, not inferred")
    ax1.set_ylabel("1 - correlation (distance-profile dissimilarity)")
    fig1.tight_layout()
    Path("artifacts/experiments").mkdir(exist_ok=True, parents=True)
    Path("docs/figures").mkdir(exist_ok=True, parents=True)
    fig1.savefig("artifacts/experiments/tb_anchor_1_basin_rollup_dendrogram.png", dpi=140)
    fig1.savefig("docs/figures/tb_anchor_1_basin_rollup_dendrogram.png", dpi=140)

    # ==== B. per-ply distance-to-every-anchor trajectories (pure scalar, ====
    # no projection -- the literal asymmetric-distance-native trajectory view) ====
    print("plotting per-ply distance-to-anchor trajectories (asymmetric, unprojected) ...", flush=True)
    n_traj = 4
    human_games = select_n_human_games_full("data/records/lichess_2019-01", "data/derived/field_std_v1.npz", n_traj, seed, 20)
    sf_games = select_n_sf_games_full("data/derived/opening_pool_sfsf_moves.tsv", n_traj, seed, 21)

    def traj_distances(games_moves, label):
        out = []
        for gid, moves in games_moves:
            planes = replay_full_game(moves)
            phi = field.phi_from_planes(list(planes.astype(np.float32))).cpu().numpy()
            with torch.no_grad():
                d = field.head.iqe.pairwise(
                    torch.as_tensor(phi, device=field.dev),
                    torch.as_tensor(phi_anchors, device=field.dev),
                ).cpu().numpy()   # (n_ply, K)
            out.append((gid, d))
        return out

    human_traj_d = traj_distances(human_games, "human")
    sf_traj_d = traj_distances(sf_games, "sf")

    fig2, axes2 = plt.subplots(2, n_traj, figsize=(4.2 * n_traj, 7.5), sharey=True)
    for row, (src_traj, src_name) in enumerate([(sf_traj_d, "SF-vs-SF"), (human_traj_d, "Human")]):
        for col in range(n_traj):
            ax = axes2[row, col]
            gid, d = src_traj[col]
            x = np.arange(len(d))
            for k in range(len(anchors)):
                ax.plot(x, d[:, k], lw=1.0, alpha=0.6, color=label_color[int(anchor_labels[k])])
            nearest_label = anchor_labels[np.argmin(d, axis=1)]
            change_points = np.flatnonzero(np.diff(nearest_label) != 0)
            for cp in change_points:
                ax.axvline(cp + 0.5, color="black", lw=0.7, linestyle=":", alpha=0.6)
            ax.set_title(f"{src_name} game {gid}\n({len(change_points)} basin crossings)", fontsize=8)
            if row == 1:
                ax.set_xlabel("ply")
            if col == 0:
                ax.set_ylabel("d(position -> anchor)\n(real asymmetric IQE distance)")
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], color=COLOR_WIN, lw=1.5, label="anchor: White-wins TB endgame"),
                    Line2D([0], [0], color=COLOR_DRAW, lw=1.5, label="anchor: drawn TB endgame"),
                    Line2D([0], [0], color=COLOR_LOSS, lw=1.5, label="anchor: Black-wins TB endgame")]
    fig2.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig2.suptitle("Distance to every TB anchor across a game (unprojected, real asymmetric quasimetric)\n"
                   "dotted vertical line = a basin-crossing (nearest anchor's label changes)")
    fig2.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig2.savefig("artifacts/experiments/tb_anchor_2_trajectories.png", dpi=140)
    fig2.savefig("docs/figures/tb_anchor_2_trajectories.png", dpi=140)

    # ==== C. summary: crossing RATE, SF vs human =====================
    def crossing_rate(traj_d_list):
        rates = []
        for gid, d in traj_d_list:
            nearest_label = anchor_labels[np.argmin(d, axis=1)]
            crossings = np.sum(np.diff(nearest_label) != 0)
            rates.append(crossings / max(1, len(nearest_label) - 1))
        return rates

    # use a larger sample (not just the n_traj plotted) for a real rate estimate
    n_rate = 25
    human_games_r = select_n_human_games_full("data/records/lichess_2019-01", "data/derived/field_std_v1.npz", n_rate, seed, 30)
    sf_games_r = select_n_sf_games_full("data/derived/opening_pool_sfsf_moves.tsv", n_rate, seed, 31)
    human_rates = crossing_rate(traj_distances(human_games_r, "human"))
    sf_rates = crossing_rate(traj_distances(sf_games_r, "sf"))
    print(f"  basin-crossing rate (fraction of plies where nearest-anchor label flips): "
          f"SF mean={np.mean(sf_rates):.4f} human mean={np.mean(human_rates):.4f} "
          f"[{time.time()-t0:.0f}s]")

    np.savez("artifacts/experiments/tb_anchor_data.npz",
             anchor_fens=np.array([a[0] for a in anchors]), anchor_labels=anchor_labels,
             anchor_classes=np.array(anchor_classes), D=D, corr=corr,
             sf_rates=np.array(sf_rates), human_rates=np.array(human_rates))
    print(f"wrote artifacts/experiments/tb_anchor_{{1_basin_rollup_dendrogram,2_trajectories}}.png "
          f"(+ docs/figures copies) + tb_anchor_data.npz")
    print(f"DONE basin_tb_anchors [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
