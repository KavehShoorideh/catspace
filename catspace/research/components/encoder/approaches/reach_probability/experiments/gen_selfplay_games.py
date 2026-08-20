#!/usr/bin/env python
"""gen_selfplay_games.py -- self-play game generator for the gdv improvement loop.

Kaveh 2026-08-20: "find something that is able to respectably play, and then improve upon
it with self-play." The gdv prior is that something (+88=12-0 vs random, +81=19-0 vs
material-greedy). This generator produces the loop's raw material: games where WE own both
the played-move identities and the outcomes -- so imitation labels and outcome labels are
ours by construction, no oracle anywhere.

WHAT IS RECORDED per game: every position's (tok, glob), the move played (from,to,promo),
the mover, and the final result; optionally BRANCHES -- at a random ply the policy's OWN
2nd-ranked move is played instead and the game continues, giving the same paired
within-parent contrast the Stockfish counterfactuals gave (played->main outcome,
alt->branch outcome), SF-free.

Diversity: first --open-plies moves are sampled from the policy softmax at --temp; argmax
after. Opponent pool: --opp-ckpt can point at a PAST checkpoint so generations do not
collapse into self-mirror draws (both sides current net if omitted).

    .venv/bin/python -m ...gen_selfplay_games --ckpt artifacts/experiments/gdv1_latest.pt \
        --games 200 --branch-frac 0.5 --out data/derived/selfplay_g1
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/gdv1_latest.pt")
    ap.add_argument("--opp-ckpt", default="", help="past checkpoint for the black side "
                    "(colors alternate per game); empty = same net both sides")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--open-plies", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--branch-frac", type=float, default=0.5,
                    help="fraction of games that also play a 2nd-choice branch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True, help="npz path prefix")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    import chess
    from catspace.research.tools.training_infra.concepts_reference import TOK_PIECES
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_delta_value import GraphDeltaValue, batch_of
    from catspace.research.tools.training_infra.gen_move_graphs import (
        parent_graph, glob_of_board)

    def load(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
        net = GraphDeltaValue(d=A["d_model"], layers=A["layers"], heads=A["attn_heads"])
        net.load_state_dict(ck["state_dict"]); net.eval()
        return net

    net = load(args.ckpt)
    opp = load(args.opp_ckpt) if args.opp_ckpt else net
    PRC = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}

    @torch.no_grad()
    def ranked_moves(model, b):
        """legal moves sorted by prior desc (the proven converter is PRIOR-ONLY)."""
        g, _ = parent_graph(b)
        D = {k: torch.from_numpy(np.asarray(g[k])[None])
             for k in ("sq", "mv", "mut", "vac", "dsc", "att")}
        D["gl"] = torch.from_numpy(glob_of_board(b)[None])
        D["n_mv"] = torch.tensor([g["n_mv"]], dtype=torch.uint8)
        D["n_att"] = torch.tensor([g["n_att"]], dtype=torch.uint16)
        sqf, mvf, edges, mask = batch_of(D, torch.arange(1), "cpu")
        prior, _ = model(sqf, mvf, edges, mask)
        n = int(g["n_mv"])
        mv = list(b.legal_moves)[:n]
        pr = prior[0, :n].numpy()
        order = np.argsort(-pr)
        return [mv[i] for i in order], pr[order]

    def tok_of(b):
        t = np.zeros(64, np.uint8)
        for q, p in b.piece_map().items():
            t[q] = TOK_PIECES.index((p.piece_type, p.color)) + 1
        return t

    def result_of(b, cap_hit):
        out = b.outcome(claim_draw=True)
        if out is not None:
            return {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[out.result()]
        d = 0
        for q, p in b.piece_map().items():
            d += (1 if p.color else -1) * {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}[p.piece_type]
        return 1 if d >= 3 else (-1 if d <= -3 else 0)

    def play_from(b, white_net, black_net, rng, ply0, forced_first=None):
        """play to the end; returns (result, [(tok, glob, from, to, promo, stm)])."""
        rec = []
        b = b.copy()
        forced = forced_first
        for ply in range(ply0, args.max_plies):
            if b.outcome(claim_draw=True) is not None:
                break
            model = white_net if b.turn else black_net
            if forced is not None:
                m = forced; forced = None
            else:
                mvs, pr = ranked_moves(model, b)
                if ply < args.open_plies and len(mvs) > 1:
                    z = (pr - pr.max()) / max(args.temp, 1e-3)
                    p = np.exp(z); p /= p.sum()
                    m = mvs[rng.choice(len(mvs), p=p)]
                else:
                    m = mvs[0]
            rec.append((tok_of(b), glob_of_board(b), m.from_square, m.to_square,
                        PRC.get(m.promotion, 0), int(b.turn)))
            b.push(m)
        return result_of(b, False), rec

    rng = np.random.default_rng(args.seed)
    TOK, GLB, FRM, TO, PRO, STM, GID, RES = [], [], [], [], [], [], [], []
    B_GID, B_PLY, B_RES, B_FRM, B_TO = [], [], [], [], []
    t0 = time.time()
    for gi in range(args.games):
        w, bl = (net, opp) if gi % 2 == 0 else (opp, net)
        res, rec = play_from(chess.Board(), w, bl, rng, 0)
        for (t, g, f, to, pc, sm) in rec:
            TOK.append(t); GLB.append(g); FRM.append(f); TO.append(to); PRO.append(pc)
            STM.append(sm); GID.append(gi)
        RES.append(res)
        # branch: replay to a random ply, play the 2nd choice, continue -- the SF-free pair
        if rng.random() < args.branch_frac and len(rec) > 6:
            bp = int(rng.integers(2, len(rec) - 2))
            b2 = chess.Board()
            for (t, g, f, to, pc, sm) in rec[:bp]:
                promo = {0: None, 1: chess.KNIGHT, 2: chess.BISHOP,
                         3: chess.ROOK, 4: chess.QUEEN}[pc]
                b2.push(chess.Move(f, to, promotion=promo))
            model = w if b2.turn else bl
            mvs, _ = ranked_moves(model, b2)
            if len(mvs) > 1:
                res2, _ = play_from(b2, w, bl, rng, bp, forced_first=mvs[1])
                B_GID.append(gi); B_PLY.append(bp); B_RES.append(res2)
                B_FRM.append(mvs[1].from_square); B_TO.append(mvs[1].to_square)
        if (gi + 1) % 50 == 0:
            el = time.time() - t0
            w_ = sum(r == 1 for r in RES); d_ = sum(r == 0 for r in RES)
            print(f"  {gi+1}/{args.games} games  {el:.0f}s ({el/(gi+1):.2f}s/game) | "
                  f"white +{w_} ={d_} -{len(RES)-w_-d_} | rows {len(TOK):,} "
                  f"branches {len(B_GID)}", flush=True)

    p = f"{args.out}_{args.games}g_{len(TOK)}r.npz"
    np.savez_compressed(
        p, tok=np.array(TOK, np.uint8), glob=np.array(GLB, np.uint8),
        mv_from=np.array(FRM, np.uint8), mv_to=np.array(TO, np.uint8),
        promo=np.array(PRO, np.uint8), stm=np.array(STM, np.uint8),
        game=np.array(GID, np.int32), result=np.array(RES, np.int8),
        br_game=np.array(B_GID, np.int32), br_ply=np.array(B_PLY, np.int32),
        br_result=np.array(B_RES, np.int8), br_from=np.array(B_FRM, np.uint8),
        br_to=np.array(B_TO, np.uint8))
    dr = np.mean(np.array(RES) == 0)
    print(f"\n  wrote {p}\n  {len(RES)} games ({100*dr:.0f} percent draws), {len(TOK):,} "
          f"positions, {len(B_GID)} branch pairs [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
