#!/usr/bin/env python
"""experiments/gen_demo_viz_data.py -- EXACT (tablebase-ground-truth) data for the veto
visualization prototypes (PUBLICATION_VETO_NOTE.md): no learned field needed, so the demo
options can be judged on real physics before the learned gate resolves.

Outputs artifacts/experiments/demo_viz_data.json:
  seismograph: one toy game (SF attacker, tb defender with injected lapses): per-ply FEN/SAN,
               forceable-fraction of sampled won regions, flip counts, lapse flags.
  heatmap:     one position: per-square "my rook can stand there within j plies" under
               COOPERATIVE (random walks) vs FORCED (DFS vs tb-optimal defense).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed
from catspace.tb import TB, tb_best_move
from experiments.measure_adversarial_veto import forceable, neighborhood_of, wdl_white

J = 4


def sample_regions(board, tb, rng, walks=150, cap=15, depth=None):
    seen = {}
    for _ in range(walks):
        b = board.copy(stack=False); ok = True
        for _t in range(depth or J):
            mv = list(b.legal_moves)
            if not mv:
                ok = False; break
            b.push(mv[int(rng.integers(len(mv)))])
        if ok and not b.is_game_over(claim_draw=True) and wdl_white(b, tb) == 2:
            seen.setdefault(b._transposition_key(), b.copy(stack=False))
    t = list(seen.values()); rng.shuffle(t)
    return t[:cap]


def main():
    t0 = time.time(); rng = np.random.default_rng(3); tb = TB()
    eng = chess.engine.SimpleEngine.popen_uci("stockfish"); eng.configure({"Threads": 1})
    dz = np.load("data/derived/dtm_endgame.npz")
    P, M, dtm = dz["packed"], dz["meta"], dz["dtm"]
    cand = np.flatnonzero((dtm >= 16) & (dtm <= 26)); rng.shuffle(cand)

    # ---- seismograph: FIXED opportunity panel (sampled once at ply 0), tracked per White ply.
    # Position: full 6-piece family (real defense) so denial is substantive at the start.
    start = None
    for ci in cand:
        b = board_from_packed(P[ci], M[ci])
        if (b.turn == chess.WHITE and not b.is_game_over() and len(b.piece_map()) == 6
                and (b.pieces(chess.BISHOP, chess.BLACK) or b.pieces(chess.PAWN, chess.BLACK))):
            start = b; break
    b = start.copy(stack=False)
    panel = sample_regions(b, tb, rng, walks=800, cap=30, depth=8)   # DEEP panel: spans captures      # THE fixed panel of opportunities
    preds = [neighborhood_of(g) for g in panel]
    panel_bk = [chess.square_name(g.king(chess.BLACK)) for g in panel]
    trace = []
    lapse_at = {1, 5}                      # defender's 3rd and 7th moves are random (blunders)
    black_ply = 0
    for ply in range(60):
        if b.is_game_over(claim_draw=True):
            break
        entry = dict(ply=ply, fen=b.fen(), turn="w" if b.turn == chess.WHITE else "b")
        if b.turn == chess.WHITE:
            open_ = [1 if forceable(b, p_, tb, J) else 0 for p_ in preds]
            entry["forceable_frac"] = round(sum(open_) / len(preds), 3)
            entry["open_regions"] = open_
            info = eng.analyse(b, chess.engine.Limit(depth=12))
            mv = info["pv"][0]
        else:
            lapse = black_ply in lapse_at
            mv = (list(b.legal_moves)[int(rng.integers(b.legal_moves.count()))] if lapse
                  else tb_best_move(b, tb))
            entry["lapse"] = bool(lapse)
            black_ply += 1
        entry["san"] = b.san(mv)
        trace.append(entry)
        b.push(mv)
    outcome = b.outcome(claim_draw=True)
    print(f"[seismo] {len(trace)} plies, panel={len(panel)}, "
          f"mate={bool(outcome and outcome.winner == chess.WHITE)}  [{time.time()-t0:.0f}s]", flush=True)

    # ---- heatmap: WHERE CAN HIS KING BE FORCED (cornering on the board). Sampled won
    # regions' black-king squares: green = some forceable region puts his king there,
    # red = reachable only if he cooperates.
    lapse_entries = [e for e in trace if e.get("lapse")]
    fen_before = lapse_entries[0]["fen"] if lapse_entries else start.fen()
    b_before = chess.Board(fen_before)
    b_after = chess.Board(fen_before)
    import chess as _c
    b_after.push(next(m for m in b_before.legal_moves
                      if b_before.san(m) == lapse_entries[0]["san"]) if lapse_entries else None)
    hm_regions = sample_regions(b_before, tb, rng, walks=900, cap=60, depth=8)
    def paint(bb):
        coop_ = [0] * 64; forc_ = [0] * 64
        for g in hm_regions:
            sq = g.king(chess.BLACK)
            coop_[sq] = 1
            if not forc_[sq] and forceable(bb, neighborhood_of(g), tb, J):
                forc_[sq] = 1
        return coop_, forc_
    coop, forc = paint(b_before)
    coop_a, forc_a = paint(b_after)
    hb = b_before
    print(f"[heatmap] BEFORE lapse: coop {sum(coop)} forced {sum(forc)}  AFTER: forced {sum(forc_a)}  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    out = dict(generated="exact tablebase ground truth (no learned field)",
               j=J,
               seismograph=dict(start_fen=start.fen(), trace=trace, panel_bk=panel_bk,
                                mate=bool(outcome and outcome.winner == chess.WHITE)),
               heatmap=dict(fen=hb.fen(), semantics="black-king forced destinations",
                            coop=coop, forced=forc, forced_after=forc_a, fen_after=b_after.fen(),
                            lapse_san=(lapse_entries[0]['san'] if lapse_entries else None)),
               veto_stats=dict(denied_exact=0.87, forceable_exact=0.13, region_forceable=0.99,
                               source="VERDICT ADVERSARIAL_VETO 2026-07-23 n=1200"))
    Path("artifacts/experiments/demo_viz_data.json").write_text(json.dumps(out))
    eng.quit(); tb.close()
    print(f"VERDICT DEMO_VIZ_DATA -> artifacts/experiments/demo_viz_data.json  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
