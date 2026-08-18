#!/usr/bin/env python
"""eval_pole_identity.py -- WHICH pole is ours? Distance to win / draw / loss, from the mover.

"we have the poles. we know if they're white winning or black winning or draw. we know whose move
it is. why don't we know which pole is ours?" (Kaveh 2026-08-18)

A fair objection to the previous pole test, which used each game's OWN terminal as its pole and so
measured TIME TO THE END regardless of whether that end was a win, a loss or a draw for the side
to move. The outcome labels were available and unused.

Here the poles are CLASSES, relativised to the mover:
    our pole    terminals of games the side to move WON
    their pole  terminals of games the side to move LOST
    draw pole   drawn terminals
Poles are drawn from TRAIN games only; every scored position is from a held-out game, so nothing
is scored against its own ending.

    d_our(s)   = min over sampled our-poles   of d(s -> pole)
    d_their(s) = min over sampled their-poles
    d_draw(s)  = min over sampled draw-poles

WHAT IS TESTED
  1 SEPARATION   does d_their - d_our predict the actual result? A geometry that only knows
                 time-to-end cannot do this at all; one that knows WHICH pole can.
  2 COMMITTOR    softmax(-d/T) over the three poles, scored as a 3-way classifier against the
                 true result, and against the base rate.
  3 SF DIRECTION does SF's move reduce d_our and INCREASE d_their -- i.e. move toward winning
                 rather than merely toward finishing?

    .venv/bin/python -m ...eval_pole_identity
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qm", default="artifacts/experiments/cqm1_latest.pt")
    ap.add_argument("--concept-ckpt", default="artifacts/experiments/cef3_step10000.pt")
    ap.add_argument("--n", type=int, default=4000, help="held-out positions to score")
    ap.add_argument("--n-poles", type=int, default=128, help="poles sampled per class")
    ap.add_argument("--n-moves", type=int, default=400, help="positions expanded for SF test")
    ap.add_argument("--temp", type=float, default=1.0)
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
    qm = ConceptQuasimetric(A["heads"] * 32 + 64 * 32, d_iqe=QA["d_iqe"],
                            components=QA["components"], hidden=QA["hidden"],
                            leak_beta=QA["leak_beta"]).to(dev)
    qm.load_state_dict(qk["state_dict"]); qm.eval()

    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    term_rows, _ = tr.terminal_rows()
    res_of_game = tr.result                       # +1 white wins, 0 draw, -1 black wins

    @torch.no_grad()
    def enc_np(tk, gb):
        z, _, _, _ = enc.encode(torch.from_numpy(tk.astype(np.int64)).to(dev),
                                torch.from_numpy(gb.astype(np.float32)).to(dev))
        return z

    # ---- poles by OUTCOME, from TRAIN games only ------------------------------------
    rng = np.random.default_rng(args.seed)
    tg = ~test_g[gor[term_rows]]
    pole = {}
    for name, want in (("W", 1), ("D", 0), ("B", -1)):
        sel = term_rows[tg & (res_of_game[gor[term_rows]] == want)]
        sel = sel[rng.choice(len(sel), min(args.n_poles, len(sel)), replace=False)]
        pole[name] = enc_np(tr.tok[sel], tr.glob[sel])
        print(f"  pole {name}: {len(sel)} terminals from train games")

    @torch.no_grad()
    def dmin(z, P):
        """min over poles of d(s -> pole). (B,d) x (P,d) -> (B,)"""
        out = torch.full((len(z),), float("inf"), device=dev)
        for i in range(len(P)):
            out = torch.minimum(out, qm.dist(z, P[i:i + 1].expand(len(z), -1)))
        return out

    # ---- 1 & 2 separation and committor ---------------------------------------------
    rows = np.flatnonzero(test_g[gor])
    rows = rows[rng.choice(len(rows), min(args.n, len(rows)), replace=False)]
    white_to_move = tr.glob[rows, 0].astype(bool)
    res = res_of_game[gor[rows]].astype(int)
    # result FROM THE MOVER'S SIDE: +1 mover wins, 0 draw, -1 mover loses
    rel = np.where(white_to_move, res, -res)

    Z = enc_np(tr.tok[rows], tr.glob[rows])
    dW, dD, dB = dmin(Z, pole["W"]), dmin(Z, pole["D"]), dmin(Z, pole["B"])
    dW, dD, dB = dW.cpu().numpy(), dD.cpu().numpy(), dB.cpu().numpy()
    d_our = np.where(white_to_move, dW, dB)
    d_their = np.where(white_to_move, dB, dW)

    print(f"\n  scored {len(rows):,} held-out positions "
          f"({100*white_to_move.mean():.0f}% white to move)")
    print(f"\n  1 SEPARATION   mean distance by TRUE result (from the mover's side)")
    print(f"      {'result':>12} {'n':>7} {'d_our':>8} {'d_draw':>8} {'d_their':>8} "
          f"{'d_their-d_our':>14}")
    for tag, want in (("mover WINS", 1), ("draw", 0), ("mover LOSES", -1)):
        m = rel == want
        if m.sum():
            print(f"      {tag:>12} {int(m.sum()):>7,} {d_our[m].mean():>8.2f} "
                  f"{dD[m].mean():>8.2f} {d_their[m].mean():>8.2f} "
                  f"{(d_their-d_our)[m].mean():>+14.2f}")
    gap = d_their - d_our
    from scipy.stats import spearmanr
    print(f"      rank corr of (d_their - d_our) with the true result: "
          f"{spearmanr(gap, rel).statistic:+.3f}")
    dec = rel != 0
    if dec.sum():
        y = (rel[dec] > 0).astype(int)
        o = np.argsort(gap[dec]); r = np.empty(len(o)); r[o] = np.arange(len(o))
        auc = (r[y == 1].mean() - (y == 1).sum() / 2 + 0.5) / max((y == 0).sum(), 1)
        auc = ((r[y == 1].sum() - (y == 1).sum() * ((y == 1).sum() - 1) / 2)
               / max((y == 1).sum() * (y == 0).sum(), 1))
        print(f"      AUC of (d_their - d_our) for mover-wins vs mover-loses: {auc:.3f}")

    P = torch.softmax(-torch.from_numpy(np.stack([d_our, dD, d_their], 1)) / args.temp, 1).numpy()
    pred = P.argmax(1)                                 # 0 our-win, 1 draw, 2 their-win
    truth = np.where(rel > 0, 0, np.where(rel == 0, 1, 2))
    base = np.bincount(truth, minlength=3).max() / len(truth)
    print(f"\n  2 COMMITTOR    softmax(-d/T) as a 3-way classifier")
    print(f"      accuracy {100*(pred == truth).mean():.1f}%   majority baseline "
          f"{100*base:.1f}%")

    # ---- 3 does SF move toward OUR pole? ---------------------------------------------
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

    mrows = rows[rng.choice(len(rows), min(args.n_moves, len(rows)), replace=False)]
    sf_our, sf_their, rnd_our, rnd_their, top, nmv = [], [], [], [], [], []
    for r in mrows:
        if r + 1 >= n or gor[r + 1] != gor[r]:
            continue
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
        wm = bool(tr.glob[r, 0])
        # after our move it is the OPPONENT to move, so from the child's own perspective the
        # poles swap. Keep everything in the ORIGINAL mover's frame.
        our_P = pole["W"] if wm else pole["B"]
        their_P = pole["B"] if wm else pole["W"]
        zc = enc_np(np.stack(tks), np.stack(gbs))
        z0 = enc_np(tr.tok[r][None], tr.glob[r][None])
        do0 = float(dmin(z0, our_P)[0]); dt0 = float(dmin(z0, their_P)[0])
        do = dmin(zc, our_P).cpu().numpy() - do0
        dt = dmin(zc, their_P).cpu().numpy() - dt0
        sf_our.append(do[played]); sf_their.append(dt[played])
        rnd_our.append(do.mean()); rnd_their.append(dt.mean())
        score = do - dt                                # want our pole nearer, theirs further
        top.append(int(np.argmin(score) == played)); nmv.append(len(mv))

    print(f"\n  3 SF DIRECTION on {len(sf_our):,} positions, in the ORIGINAL mover's frame")
    print(f"      {'':22} {'d_our change':>14} {'d_their change':>16}")
    print(f"      {'SF played move':22} {np.mean(sf_our):>+14.3f} {np.mean(sf_their):>+16.3f}")
    print(f"      {'average legal move':22} {np.mean(rnd_our):>+14.3f} {np.mean(rnd_their):>+16.3f}")
    print(f"      SF picks the best (d_our down, d_their up): {100*np.mean(top):.1f}%  "
          f"(random {100*np.mean(1/np.array(nmv)):.1f}%)")


if __name__ == "__main__":
    main()
