#!/usr/bin/env python
"""export_ply_umap.py -- ONE shared UMAP of the learned space, exported as per-position JSON so an
interactive page can slide through PLY CROSS-SECTIONS of it.

Kaveh: "i want actually a slider that lets me go through ply cross section and see embeddings at
that ply ... i mean like a umap of them".

THE ONE DESIGN DECISION THAT MATTERS: the UMAP is fitted ONCE on positions from every ply together,
and the slider then FILTERS that single map. Fitting a separate UMAP per ply would be much easier
and completely misleading -- UMAP axes carry no meaning across independent fits, so consecutive
plies would jump around for purely algorithmic reasons and any apparent "motion" would be an
artifact of refitting. With one shared fit, what the slider shows is a genuine cross-section: the
same coordinates throughout, so a cloud that moves really moved.

Exports x, y, ply, piece count, source (human/SF), and outcome per position, so the page can colour
by any of them without recomputing anything.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.experiments.plot_strata_figures import (
    embed)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=paths.experiment("reach_vit_v1_latest.pt"))
    ap.add_argument("--per-ply", type=int, default=400, help="positions sampled at each ply")
    ap.add_argument("--max-ply", type=int, default=210,
                    help="p99 of game length is 201 plies, so 210 covers 99%% of games")
    ap.add_argument("--n-term", type=int, default=9000, help="terminal positions to include")
    ap.add_argument("--n-trace", type=int, default=200, help="games traced per population")
    ap.add_argument("--dims", type=int, default=3,
                    help="3 = a rotatable 3D map. A 2D projection of a 64-dim quasimetric\n"
                         "space flattens the very layering the strata question is about")
    ap.add_argument("--neighbors", type=int, default=25)
    ap.add_argument("--min-dist", type=float, default=0.08)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=paths.experiment("reach_vit_v1_ply_umap.json"))
    args = ap.parse_args()

    t0 = time.time()
    net, payload = load_net(args.ckpt, args.device)
    c = payload["cfg"]
    tr = T.build(n_human=c["games"] // 2, n_sf=c["games"] // 2, seed=c["traj_seed"],
                 max_plies=c["max_plies"], verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    keep_games = np.flatnonzero(split == 2)
    game, ply, pc = tr.game_of_row(), tr.ply_of_row(), tr.piece_count()
    outcome = tr.outcome_of_row()
    src = np.repeat(tr.source, tr.length)
    in_test = np.isin(game, keep_games)
    rng = np.random.default_rng(0)

    # BALANCED BY PLY on purpose: a uniform sample would be dominated by the ply range where most
    # games happen to sit, and the slider would then show a few dense plies and a scatter of empty
    # ones. Equal counts per ply make each cross-section equally readable.
    rows = []
    for p in range(0, args.max_ply + 1):
        cand = np.flatnonzero(in_test & (ply == p))
        if len(cand) == 0:
            continue
        # DEDUPLICATE BY POSITION IDENTITY. There is exactly ONE position at ply 0 and TWENTY at
        # ply 1, so a flat 400-per-ply sample drew each of them ~20-400 times over. UMAP renders
        # each DISTINCT position as its own island, so those duplicate draws showed up as a scatter
        # of specks around the periphery -- an artifact of the sampler pretending there were 400
        # positions where chess offers 20. Sampling unique positions lets the early plies
        # contribute their true handful.
        h = T.position_hash(tr.tok[cand], tr.glob[cand])
        cand = cand[np.unique(h, return_index=True)[1]]
        take = cand if len(cand) <= args.per_ply else rng.choice(cand, args.per_ply, replace=False)
        rows.append(take)
    rows = np.concatenate(rows)
    # TERMINALS ARE ONE ROW PER GAME, so uniform per-ply sampling barely catches any -- and they
    # are the whole point of the "arrived" colouring, since they are the ABSORBING states. Add them
    # explicitly (test games only) rather than hoping the sampler finds them.
    t_rows, t_term = tr.terminal_rows()
    t_keep = np.isin(game[t_rows], keep_games) & (ply[t_rows] <= args.max_ply)
    t_rows, t_term = t_rows[t_keep], t_term[t_keep]
    if len(t_rows) > args.n_term:
        pick = rng.choice(len(t_rows), args.n_term, replace=False)
        t_rows, t_term = t_rows[pick], t_term[pick]
    rows = np.unique(np.concatenate([rows, t_rows]))
    # ARRIVED: -1 = game still in progress at this position; 0/1/2 = it ENDED here, in that
    # outcome. Distinct from `out`, which back-projects the eventual result onto every ply.
    arrived = np.full(tr.n_positions, -1, np.int8)
    arrived[t_rows] = T.TERM_OUTCOME[t_term]
    print(f"[umap] + {len(t_rows):,} terminal positions "
          f"(W {int((T.TERM_OUTCOME[t_term]==T.WIN).sum()):,} "
          f"D {int((T.TERM_OUTCOME[t_term]==T.DRAW).sum()):,} "
          f"L {int((T.TERM_OUTCOME[t_term]==T.LOSS).sum()):,})", flush=True)
    print(f"[umap] {len(rows):,} positions across {len(np.unique(ply[rows]))} plies", flush=True)

    # TRACE ROWS ARE COLLECTED BEFORE THE FIT AND INCLUDED IN IT (Kaveh: "shouldn't the umap be
    # defined on ... at least the data being plotted?"). Fitting on the cross-sections alone and
    # projecting traces in with transform() is legitimate, but it can only map new points ONTO
    # already-fitted structure -- a game visiting a region the fit never saw gets squashed onto the
    # nearest thing instead of revealing it, and traces are exactly where that would mislead.
    # Fitting the union removes the approximation. (Fitting ALL 18.9M positions is not on: UMAP at
    # that scale is hours and out of memory, so a representative sample is the honest ceiling.)
    # PGN for the traced games. The store keeps POSITIONS, not moves, so the move lists are
    # reloaded from source by game_id (the loaders are deterministic given the same seed, so the
    # ids line up) and converted UCI -> SAN by replay. Cheaper and far less error-prone than
    # inferring each move from consecutive token boards.
    import chess
    uci_by_id = {}
    for g in T.load_human_games(c["games"] // 2, 0, None, 400):
        uci_by_id[(T.HUMAN, g[0])] = g[2]
    for g in T.load_sf_games(c["games"] // 2, 0, None, 400):
        uci_by_id[(T.SF, g[0])] = g[2]

    def san_of(gi):
        ucis = uci_by_id.get((int(tr.source[gi]), int(tr.game_id[gi])))
        if not ucis:
            return []
        b, out = chess.Board(), []
        for u in ucis:
            try:
                mv = chess.Move.from_uci(u)
                out.append(b.san(mv)); b.push(mv)
            except Exception:
                break
        return out

    trace_rows, trace_meta = [], []
    for pop in (T.HUMAN, T.SF):
        gs = np.flatnonzero((tr.source == pop) & np.isin(np.arange(len(tr)), keep_games))
        if not len(gs):
            continue
        for gi in rng.choice(gs, min(args.n_trace, len(gs)), replace=False):
            st, ln = int(tr.start[gi]), int(tr.length[gi])
            if ln < 4:
                continue
            gr = np.arange(st, st + min(ln, args.max_ply + 1))
            trace_rows.append(gr)
            trace_meta.append((int(pop), int(ply[gr[0]]), int(tr.term[gi]), len(gr), san_of(gi)))
    all_trace = np.concatenate(trace_rows) if trace_rows else np.zeros(0, np.int64)
    fit_rows = np.concatenate([rows, all_trace])
    print(f"[umap] fitting on {len(fit_rows):,} = {len(rows):,} cross-section "
          f"+ {len(all_trace):,} trace positions", flush=True)

    Z = embed(net, tr, fit_rows, args.device).numpy()
    import umap
    red = umap.UMAP(n_neighbors=args.neighbors, min_dist=args.min_dist, n_components=args.dims,
                    metric="euclidean", random_state=0, verbose=False)
    XY_all = red.fit_transform(Z)
    print(f"[umap] fitted ONE shared projection [{time.time()-t0:.0f}s]", flush=True)

    # normalise to a stable [0,1] box so the page needs no autoscaling per ply -- otherwise each
    # cross-section would rescale to its own extent and cloud motion would be invisible
    lo, hi = XY_all.min(0), XY_all.max(0)
    XY_all = (XY_all - lo) / np.maximum(hi - lo, 1e-9)
    XY = XY_all[:len(rows)]                       # cross-section points
    TXY = XY_all[len(rows):]                      # trace points, same fit, no transform() needed

    # ---- TRACES: whole games projected into the SAME map -------------------------------------
    # umap.transform() is what makes this honest. Fitting a second UMAP on the trajectories would
    # put them in unrelated coordinates and any apparent path through the cross-sections would be
    # fiction. Transforming into the ALREADY-FITTED embedding keeps traces and cross-sections in
    # one coordinate system, so a game that appears to move through a region really does.
    # GAME PHASE, the engine-standard definition: non-pawn material weighted N=1 B=1 R=2 Q=4,
    # summed over both sides, 24 at the starting position and 0 in a bare-king endgame. Material
    # alone cannot separate opening from middlegame (both are near 24), so ply carries that split --
    # which is the real distinction, since "opening" is about development, not material.
    W = {2: 1, 3: 1, 4: 2, 5: 4, 8: 1, 9: 1, 10: 2, 11: 4}
    phase_val = np.zeros(tr.n_positions, np.int16)
    for pid, wt in W.items():
        phase_val += wt * (tr.tok == pid).sum(1).astype(np.int16)
    _ply_all = tr.ply_of_row()
    # ENDGAME = MAJOR PIECES OFF THE BOARD (Kaveh's definition): no queens and no rooks left, for
    # either side. Majors are Q and R; N and B are minors. Stated because it is stricter than the
    # usual material-threshold rule -- notably ROOK endings, the most common endgame type in
    # practice, land in "middlegame" under this definition. That is a deliberate choice, not an
    # oversight, and `phv` (raw 0-24 non-pawn material) is exported alongside so the alternative
    # threshold can be applied at any time without re-exporting.
    # ENDGAME by MATERIAL THRESHOLD, not by 'majors off'. Kaveh raised the stricter rule
    # (no queens and no rooks) then withdrew it himself for the right reason: K+R endings
    # are among the most common endgames in chess, and 'majors off' would file every one
    # of them as a middlegame. phase_val <= 6 admits rook and queen endings while still
    # excluding positions that merely traded a couple of pieces. `phv` (raw 0-24) is
    # exported too, so another threshold can be applied without re-exporting.
    # THRESHOLD 10, not 6. At 6 a position needed to be down to roughly a queen's worth of
    # non-pawn material, which stretched 'middlegame' out to ply 170 with material 7 -- a
    # rook and a minor each, 85 moves in, is an endgame by any reading. 10 admits
    # rook+minor endings and queen endings while still excluding positions that have
    # merely traded a couple of pieces.
    #
    # SAMPLING CAVEAT, which is a SEPARATE issue and is NOT fixed here: rows are sampled
    # 400-per-ply out to 210 while median game length is 91, so late plies are massively
    # over-represented relative to real games and the phase MIX in this export is not the
    # corpus mix. That is deliberate -- the slider exists for per-ply comparison, which
    # needs every cross-section equally populated -- but it means the endgame share here
    # says nothing about how common endgames are.
    phase = np.where(phase_val <= 10, 2,                                      # endgame
                     np.where((_ply_all <= 24) & (phase_val >= 20), 0, 1))    # 0 open, 1 middle
    phase = phase.astype(np.int8)

    castle = ((tr.glob[:, 1] > 0) * 8 + (tr.glob[:, 2] > 0) * 4
              + (tr.glob[:, 3] > 0) * 2 + (tr.glob[:, 4] > 0)).astype(np.int16)
    traces, off = [], 0
    for (pop, p0, end, n, san) in trace_meta:
        seg = TXY[off:off + n]; off += n
        traces.append({"pop": pop, "p0": p0, "end": end, "san": san,
                       "x": [round(float(v), 4) for v in seg[:, 0]],
                       "y": [round(float(v), 4) for v in seg[:, 1]],
                       "z": ([round(float(v), 4) for v in seg[:, 2]]
                             if seg.shape[1] > 2 else None)})
    print(f"[umap] {len(traces)} games CO-FITTED (not transformed) into the same map", flush=True)

    data = {
        "step": int(payload.get("step", -1)),
        "n": int(len(rows)),
        "dims": int(args.dims),
        "x": [round(float(v), 4) for v in XY[:, 0]],
        "y": [round(float(v), 4) for v in XY[:, 1]],
        "z": [round(float(v), 4) for v in XY[:, 2]] if args.dims > 2 else None,
        "ply": [int(v) for v in ply[rows]],
        "pc": [int(v) for v in pc[rows]],
        "src": [int(v) for v in src[rows]],           # 0 human, 1 sf
        "out": [int(v) for v in outcome[rows]],       # 0 win, 1 draw, 2 loss, -1 censored
        # arrived: did the game ACTUALLY END at this position, and in what? -1 = still in progress
        "arr": [int(v) for v in arrived[rows]],
        # castling rights as a 4-bit code (K,Q,k,q). The lobe diagnostic found this is the SINGLE
        # strongest predictor of cluster membership (AMI 0.357, against 0.004 for eventual
        # outcome) -- castling rights are irreversible, like material, and the model organised its
        # space along irreversibility rather than around who wins.
        "cas": [int(v) for v in castle[rows]],
        "ph": [int(v) for v in phase[rows]],          # 0 opening, 1 middlegame, 2 endgame
        "phv": [int(v) for v in phase_val[rows]],     # raw 0-24 non-pawn material
        "traces": traces,
    }
    with open(args.out, "w") as f:
        json.dump(data, f)
    import os
    print(f"[umap] -> {args.out}  ({os.path.getsize(args.out)/2**20:.1f} MB) "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
