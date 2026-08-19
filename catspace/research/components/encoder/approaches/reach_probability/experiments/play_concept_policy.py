#!/usr/bin/env python
"""play_concept_policy.py -- does the emergent concept ranking actually PLAY?

"how well does it play?" (Kaveh 2026-08-19)

cpl1 is a 1-ply policy prior: for each legal move, encode the child through the frozen concept
model and take the argmax of the learned atom utilities. No search, no value, no lookahead --
so this measures the RANKING itself, not a chess program built around it. Pairings:

    vs random          sanity floor -- any signal at all beats this
    vs material-greedy the standing baseline it drew with on the winning-track test; games are
                       the sterner version because errors COMPOUND over 100+ plies
    material vs random calibrates the arena itself
    vs stockfish       skill 0, 10ms -- an external anchor; an opponent, not a label

Opening diversity: for the first 8 plies the concept policy SAMPLES from its softmax (temp 1)
rather than argmax, so games differ; deterministic argmax afterwards. Draws by rule
(repetition/50-move/stalemate/insufficient); at the 240-ply cap the game is adjudicated to the
side up >= 3 points of material, else drawn.

    .venv/bin/python -m ...play_concept_policy --games 100
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


VAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy-ckpt", default="artifacts/experiments/cpl1_latest.pt")
    ap.add_argument("--concept-ckpt", default="artifacts/experiments/cef3_step10000.pt")
    ap.add_argument("--games", type=int, default=100, help="per pairing, colors alternated")
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--open-plies", type=int, default=8)
    ap.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--sf-skill", type=int, default=0)
    ap.add_argument("--sf-ms", type=int, default=10)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import chess
    import chess.engine
    from catspace.research.tools.training_infra.concepts_reference import TOK_PIECES
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_event_field import ConceptEvent
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import factor_vocab
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_policy_head import PolicyScorer
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    dev = args.device
    ck = torch.load(args.concept_ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{len(tr.tok)}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    cr, ce, co, egs = factor_vocab(vocab)
    enc = ConceptEvent(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                       square_codes=A["square_codes"], hidden=A["hidden"],
                       n_res=len(vocab), factored=True, n_eg=len(egs),
                       cls_reach=cr, cls_eg=ce, cls_out=co).to(dev)
    enc.load_state_dict(ck["state_dict"]); enc.eval()
    pk = torch.load(args.policy_ckpt, map_location="cpu", weights_only=False)
    PA = pk["args"] if isinstance(pk.get("args"), dict) else vars(pk["args"])
    scorer = PolicyScorer(A["heads"] * 32 + 64 * 32, hidden=PA["hidden"], depth=PA["depth"],
                          arch=PA.get("arch", "mlp"), n_heads=A["heads"], n_codes=A["codes"],
                          n_sq_codes=A["square_codes"]).to(dev)
    scorer.load_state_dict(pk["state_dict"]); scorer.eval()

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

    @torch.no_grad()
    def cpl_move(b, rng, ply):
        mv = list(b.legal_moves)
        if len(mv) == 1:
            return mv[0]
        tks, gbs = [], []
        for m in mv:
            b.push(m); t, g = tok_glob_of(b); b.pop()
            tks.append(t); gbs.append(g)
        z, ig, isq, _ = enc.encode(torch.from_numpy(np.stack(tks).astype(np.int64)).to(dev),
                                   torch.from_numpy(np.stack(gbs).astype(np.float32)).to(dev))
        sc = scorer(z, ig, isq).cpu().numpy()
        if ply < args.open_plies:
            p = np.exp(sc - sc.max()); p /= p.sum()
            return mv[rng.choice(len(mv), p=p)]
        return mv[int(np.argmax(sc))]

    def material_move(b, rng, ply):
        mv = list(b.legal_moves)
        gain = []
        for m in mv:
            cap = b.piece_at(m.to_square)
            gain.append(VAL[cap.piece_type] if cap else 0)
        gain = np.array(gain)
        best = np.flatnonzero(gain == gain.max())
        return mv[rng.choice(best)]

    def random_move(b, rng, ply):
        mv = list(b.legal_moves)
        return mv[rng.integers(0, len(mv))]

    eng = None

    def sf_move(b, rng, ply):
        return eng.play(b, chess.engine.Limit(time=args.sf_ms / 1000)).move

    def material_diff(b):
        d = 0
        for q, p in b.piece_map().items():
            d += (1 if p.color else -1) * VAL[p.piece_type]
        return d

    def play(white, black, seed):
        rng = np.random.default_rng(seed)
        b = chess.Board()
        for ply in range(args.max_plies):
            out = b.outcome(claim_draw=True)
            if out is not None:
                return {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[out.result()]
            b.push((white if b.turn else black)(b, rng, ply))
        out = b.outcome(claim_draw=True)
        if out is not None:
            return {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[out.result()]
        d = material_diff(b)
        return 1 if d >= 3 else (-1 if d <= -3 else 0)

    def series(name, us, them, need_engine=False):
        nonlocal eng
        if need_engine:
            eng = chess.engine.SimpleEngine.popen_uci(args.engine)
            try:
                eng.configure({"Skill Level": args.sf_skill})
            except Exception:
                pass
        w = d = l = 0
        for gi in range(args.games):
            if gi % 2 == 0:
                r = play(us, them, seed=args.seed * 10000 + gi)
            else:
                r = -play(them, us, seed=args.seed * 10000 + gi)
            w += r == 1; d += r == 0; l += r == -1
            if (gi + 1) % 25 == 0:
                print(f"    {name}: {gi+1}/{args.games}  +{w} ={d} -{l}", flush=True)
        s = (w + 0.5 * d) / max(w + d + l, 1)
        elo = 400 * np.log10(s / (1 - s)) if 0 < s < 1 else float("inf") * (1 if s == 1 else -1)
        print(f"  {name:<28} +{w} ={d} -{l}   score {100*s:.1f}%   elo {elo:+.0f}", flush=True)
        if need_engine:
            eng.quit(); eng = None
        return w, d, l

    print(f"\n  {args.games} games per pairing, colors alternated, {args.max_plies}-ply cap "
          f"(adjudicated at |material| >= 3)\n")
    series("cpl1 vs RANDOM", cpl_move, random_move)
    series("cpl1 vs MATERIAL-GREEDY", cpl_move, material_move)
    series("material vs RANDOM (calib)", material_move, random_move)
    series(f"cpl1 vs SF skill {args.sf_skill} ({args.sf_ms}ms)", cpl_move, sf_move,
           need_engine=True)


if __name__ == "__main__":
    main()
