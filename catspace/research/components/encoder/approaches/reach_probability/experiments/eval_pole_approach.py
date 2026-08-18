#!/usr/bin/env python
"""eval_pole_approach.py -- does Stockfish move TOWARD the pole, in this geometry?

"how well does sf agree with direction of approach to poles? do we have poles in this iteration"
(Kaveh 2026-08-18)

Poles exist in the DATA -- `terminal_rows` gives every board-caused ending, with time forfeits
censored so a dead-won flagged position never anchors to a loss. They were NOT used to train
cqm1: that geometry saw only the adjacent-ply constraint and the independent-pair objective. So
this asks whether pole structure emerged on its own, which is a stronger question than whether it
was fitted.

Each game's own terminal position is its pole -- concrete, per-game, no clustering needed and no
material signature leaking in.

  1 APPROACH     along a game, does d(s_t -> pole) fall as the game runs? If the geometry has any
                 notion of "getting there", this is where it shows.
  2 SF DIRECTION for each position, delta = d(child -> pole) - d(parent -> pole). Does the move
                 Stockfish actually played reduce the distance, and how does it rank among its
                 siblings by that reduction?
  3 BASELINES    a random sibling, and the best possible sibling, on identical rows.

    .venv/bin/python -m ...eval_pole_approach
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qm", default="artifacts/experiments/cqm1_latest.pt")
    ap.add_argument("--concept-ckpt", default="artifacts/experiments/cef3_step10000.pt")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import chess
    from scipy.stats import spearmanr
    from catspace.research.tools.training_infra.concepts_reference import (
        TOK_PIECES, board_from_row)
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_event_field import ConceptEvent
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import factor_vocab
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_quasimetric import ConceptQuasimetric
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    dev = args.device
    ck = torch.load(args.concept_ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n = len(tr.tok)
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    cr, ce, co, egs = factor_vocab(vocab)
    enc = ConceptEvent(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                       square_codes=A["square_codes"], hidden=A["hidden"],
                       n_res=len(vocab), factored=True, n_eg=len(egs),
                       cls_reach=cr, cls_eg=ce, cls_out=co).to(dev)
    enc.load_state_dict(ck["state_dict"]); enc.eval()
    qk = torch.load(args.qm, map_location="cpu", weights_only=False)
    QA = qk["args"] if isinstance(qk.get("args"), dict) else vars(qk["args"])
    qm = ConceptQuasimetric(A["heads"] * 32 + 64 * 32, d_iqe=QA["d_iqe"],
                            components=QA["components"], hidden=QA["hidden"],
                            leak_beta=QA["leak_beta"]).to(dev)
    qm.load_state_dict(qk["state_dict"]); qm.eval()

    term_rows, term_kind = tr.terminal_rows()
    pole_of_game = {}
    gor = tr.game_of_row()
    for r, k in zip(term_rows, term_kind):
        pole_of_game[int(gor[r])] = int(r)
    print(f"\n  poles available: {len(pole_of_game):,} games with a board-caused ending "
          f"(of {len(tr.start):,}); time forfeits censored by design")

    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True

    @torch.no_grad()
    def enc_np(tk, gb):
        z, _, _, _ = enc.encode(torch.from_numpy(tk.astype(np.int64)).to(dev),
                                torch.from_numpy(gb.astype(np.float32)).to(dev))
        return z

    def tok_glob_of(b):
        t = np.zeros(64, np.uint8)
        for q, p in b.piece_map().items():
            t[q] = TOK_PIECES.index((p.piece_type, p.color)) + 1
        g = np.array([int(b.turn), int(b.has_kingside_castling_rights(chess.WHITE)),
                      int(b.has_queenside_castling_rights(chess.WHITE)),
                      int(b.has_kingside_castling_rights(chess.BLACK)),
                      int(b.has_queenside_castling_rights(chess.BLACK)),
                      0 if b.ep_square is None else chess.square_file(b.ep_square) + 1], np.uint8)
        return t, g

    # ---- 1 does d fall along a game? --------------------------------------------------
    games = [g for g in np.flatnonzero(test_g) if int(g) in pole_of_game]
    rng = np.random.default_rng(args.seed)
    gs = rng.choice(games, min(150, len(games)), replace=False)
    rhos, firsts, lasts = [], [], []
    for g in gs:
        s0, ln = int(tr.start[g]), int(tr.length[g])
        if ln < 12:
            continue
        pr = pole_of_game[int(g)]
        rows = np.arange(s0, s0 + ln - 1)
        if len(rows) > 120:
            rows = rows[np.linspace(0, len(rows) - 1, 120).astype(int)]
        z = enc_np(tr.tok[rows], tr.glob[rows])
        zp = enc_np(tr.tok[pr][None], tr.glob[pr][None])
        with torch.no_grad():
            d = qm.dist(z, zp.expand(len(z), -1)).cpu().numpy()
        rhos.append(spearmanr(np.arange(len(d)), d).statistic)
        firsts.append(d[0]); lasts.append(d[-1])
    rhos = np.array([r for r in rhos if np.isfinite(r)])
    print(f"\n  1 APPROACH along {len(rhos)} held-out games")
    print(f"      rank corr of d(s_t -> pole) with ply t:  {rhos.mean():+.3f} "
          f"(negative = approaching)")
    print(f"      games where it decreases overall:        {100*np.mean(np.array(lasts) < np.array(firsts)):.1f}%")
    print(f"      mean d at the START {np.mean(firsts):.2f}  ->  at the END {np.mean(lasts):.2f}")

    # ---- 2 does SF move toward the pole? ----------------------------------------------
    cand = np.flatnonzero(test_g[gor])
    cand = cand[[int(gor[r]) in pole_of_game and r < pole_of_game[int(gor[r])] - 1
                 for r in cand]]
    rows = cand[rng.choice(len(cand), min(args.n, len(cand)), replace=False)]
    red_sf, rank_sf, top_sf, red_best, red_rand, n_moves = [], [], [], [], [], []
    for r in rows:
        b = board_from_row(tr.tok[r], tr.glob[r])
        mv = list(b.legal_moves)
        if len(mv) < 2:
            continue
        nxt = tr.tok[r + 1]; played = None
        tks, gbs = [], []
        for m in mv:
            b.push(m); t, g = tok_glob_of(b); b.pop()
            tks.append(t); gbs.append(g)
            if played is None and np.array_equal(t, nxt):
                played = len(tks) - 1
        if played is None:
            continue
        pr = pole_of_game[int(gor[r])]
        zp = enc_np(tr.tok[pr][None], tr.glob[pr][None])
        z0 = enc_np(tr.tok[r][None], tr.glob[r][None])
        zc = enc_np(np.stack(tks), np.stack(gbs))
        with torch.no_grad():
            d0 = float(qm.dist(z0, zp)[0])
            dc = qm.dist(zc, zp.expand(len(zc), -1)).cpu().numpy()
        delta = dc - d0                                   # negative = moved toward the pole
        red_sf.append(delta[played]); red_best.append(delta.min())
        red_rand.append(float(delta.mean()))
        order = np.argsort(delta)
        rank_sf.append(int(np.where(order == played)[0][0]) / max(len(mv) - 1, 1))
        top_sf.append(order[0] == played); n_moves.append(len(mv))

    red_sf = np.array(red_sf)
    print(f"\n  2 SF's DIRECTION on {len(red_sf):,} positions ({np.mean(n_moves):.1f} legal moves)")
    print(f"      SF's move reduces d to the pole:      {100*(red_sf < 0).mean():.1f}% of the time")
    print(f"      mean change from SF's move:           {red_sf.mean():+.3f}")
    print(f"      mean change, random sibling:          {np.mean(red_rand):+.3f}")
    print(f"      mean change, BEST sibling:            {np.mean(red_best):+.3f}")
    print(f"      SF picks the best approacher:         {100*np.mean(top_sf):.1f}%  "
          f"(random {100*np.mean(1/np.array(n_moves)):.1f}%)")
    print(f"      SF's percentile among siblings:       {100*np.mean(rank_sf):.1f}%  "
          f"(0 = best approach, 50 = random)")


if __name__ == "__main__":
    main()
