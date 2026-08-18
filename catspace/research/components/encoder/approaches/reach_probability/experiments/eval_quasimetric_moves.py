#!/usr/bin/env python
"""eval_quasimetric_moves.py -- can the geometry tell SIBLING MOVES apart, and does it play?

"so how does this quasimetric play? can it distinguish sibling moves?" (Kaveh 2026-08-18)

The prior question is not whether it ranks moves WELL but whether it can rank them AT ALL. The
concept state is a vector of DISCRETE codes -- 16 global plus one per square -- so two children of
the same position that happen to quantise identically are, to this geometry, the same state. Any
distance between them is then exactly zero by construction, and no amount of training fixes it.
So distinguishability is measured first and separately from quality.

Then the operational test. The corpus is Stockfish vs Stockfish, so the move actually played is a
strong move. For each position we generate every legal move, encode every child, and score it by

    d(child -> goal)        goal = where the game actually got to, K plies later

and ask where the played move lands in that ranking. Against random (1/n_legal) and against a
material-greedy baseline, both computed on identical rows.

NOTE the children are NOT corpus rows, so the frozen TRUNK must run here -- the cached codes
cover played positions only. That is the one place in stage 2 where the encoder is unavoidable.

    .venv/bin/python -m ...eval_quasimetric_moves --qm artifacts/experiments/cqm1_latest.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qm", default="artifacts/experiments/cqm1_latest.pt")
    ap.add_argument("--concept-ckpt", default="artifacts/experiments/cef3_step10000.pt")
    ap.add_argument("--n", type=int, default=1200, help="positions to expand")
    ap.add_argument("--goal-k", type=int, default=16, help="plies ahead the goal sits")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import chess
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
    d_state = A["heads"] * 32 + 64 * 32
    qm = ConceptQuasimetric(d_state, d_iqe=QA["d_iqe"], components=QA["components"],
                            hidden=QA["hidden"], leak_beta=QA["leak_beta"]).to(dev)
    qm.load_state_dict(qk["state_dict"]); qm.eval()

    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    is_test = test_g[gor]
    rng = np.random.default_rng(args.seed)
    cand = np.flatnonzero(is_test)
    cand = cand[cand + args.goal_k < n]
    cand = cand[gor[cand + args.goal_k] == gor[cand]]
    rows = cand[rng.choice(len(cand), min(args.n, len(cand)), replace=False)]

    def tok_glob_of(b):
        t = np.zeros(64, np.uint8)
        for q, p in b.piece_map().items():
            t[q] = TOK_PIECES.index((p.piece_type, p.color)) + 1
        g = np.array([int(b.turn),
                      int(b.has_kingside_castling_rights(chess.WHITE)),
                      int(b.has_queenside_castling_rights(chess.WHITE)),
                      int(b.has_kingside_castling_rights(chess.BLACK)),
                      int(b.has_queenside_castling_rights(chess.BLACK)),
                      0 if b.ep_square is None else chess.square_file(b.ep_square) + 1],
                     np.uint8)
        return t, g

    VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9,
           chess.KING: 0}

    @torch.no_grad()
    def encode(tk, gb):
        zq, ids, ids_sq, _ = enc.encode(torch.from_numpy(tk.astype(np.int64)).to(dev),
                                        torch.from_numpy(gb.astype(np.float32)).to(dev))
        return zq, ids, ids_sq

    n_legal, n_distinct, ranks, tops, rand_tops, mat_tops = [], [], [], [], [], []
    d_spread, zero_gap = [], []
    used = 0
    for r in rows:
        b = board_from_row(tr.tok[r], tr.glob[r])
        moves = list(b.legal_moves)
        if len(moves) < 2:
            continue
        played_to = None
        # the move actually played is the one whose child board matches the next corpus row
        nxt = tr.tok[r + 1]
        tks, gbs, mats = [], [], []
        for mv in moves:
            cap = b.piece_at(mv.to_square)
            mats.append(VAL[cap.piece_type] if cap else 0)
            b.push(mv)
            t, g = tok_glob_of(b)
            tks.append(t); gbs.append(g)
            if played_to is None and np.array_equal(t, nxt):
                played_to = len(tks) - 1
            b.pop()
        if played_to is None:
            continue
        used += 1
        tks = np.stack(tks); gbs = np.stack(gbs)
        _, ids, ids_sq = encode(tks, gbs)
        codes = np.concatenate([ids.cpu().numpy(), ids_sq.cpu().numpy()], 1)
        uniq = len(np.unique(codes, axis=0))
        n_legal.append(len(moves)); n_distinct.append(uniq)

        gt, gg = tr.tok[r + args.goal_k][None], tr.glob[r + args.goal_k][None]
        zg, _, _ = encode(gt, gg)
        zc, _, _ = encode(tks, gbs)
        with torch.no_grad():
            d = qm.dist(zc, zg.expand(len(zc), -1)).cpu().numpy()
        d_spread.append(float(d.std()))
        order = np.argsort(d)                      # closest to the goal first
        rank = int(np.where(order == played_to)[0][0])
        ranks.append(rank / max(len(moves) - 1, 1))
        tops.append(rank == 0)
        rand_tops.append(1.0 / len(moves))
        mat_tops.append(int(np.argmax(mats) == played_to))
        # how many siblings are EXACTLY tied with the best (indistinguishable)?
        zero_gap.append(float((np.abs(d - d.min()) < 1e-6).sum()))

    nl = np.array(n_legal); nd = np.array(n_distinct)
    print(f"\n  {used:,} positions expanded (held-out games), goal = the position "
          f"{args.goal_k} plies later\n")
    print(f"  1 CAN IT SEE THE DIFFERENCE?")
    print(f"      legal moves per position          {nl.mean():.1f}")
    print(f"      DISTINCT concept states among them {nd.mean():.1f}  "
          f"({100*(nd/nl).mean():.1f}% of siblings)")
    print(f"      positions where ALL siblings collapse to one state: "
          f"{100*(nd == 1).mean():.1f}%")
    print(f"      siblings exactly tied at the minimum distance: {np.mean(zero_gap):.1f}")
    print(f"      spread of d across siblings (std)  {np.mean(d_spread):.3f}")
    print(f"\n  2 DOES IT PLAY?  (rank of the move Stockfish actually played)")
    print(f"      picks the played move first       {100*np.mean(tops):.1f}%")
    print(f"      random baseline                   {100*np.mean(rand_tops):.1f}%")
    print(f"      material-greedy baseline          {100*np.mean(mat_tops):.1f}%")
    print(f"      mean percentile of the played move {100*np.mean(ranks):.1f}%  "
          f"(0% = ranked best, 50% = random)")


if __name__ == "__main__":
    main()
