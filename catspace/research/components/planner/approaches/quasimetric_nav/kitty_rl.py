#!/usr/bin/env python
"""kitty_rl.py -- the SIMPLE planner (Kaveh 2026-08-10): a small learned move-scorer over BOTH
rulers, trained by reinforcement from self-play outcomes. Not subgoals -- those wait until this
plays "a respectable game."

Per legal move the scorer sees 13 features, all white-POV, all from the FROZEN field:
    child dA -> (Wwin, draw, Bwin)   length ruler: expected plies to each ending
    child dB -> (Wwin, draw, Bwin)   outcome ruler: which ending (committor geometry)
    parent dA -> 3, parent dB -> 3   context: the standing before the move
    side-to-move flag                so the net can learn standing-aware conduct
                                     (losing => the draw is an ASSET, Kaveh 2026-08-09)
A 2-layer network maps features -> score; play = softmax sampling (training) or argmax (eval).
Training: REINFORCE with a mean baseline over paired-opening self-play games, mover-POV
returns (+1 win / 0 draw / -1 loss, no discount). The tiny parameter count (~600) is the
point: it learns the TRADE-OFF policy between the rulers, not chess -- interpretable by
reading which features move the score.

    .venv/bin/python -m ...kitty_rl --ckpt <field.pt> [--iters 30] [--games-per 24]
saves scorer to <field>_rlscorer.pt; --eval-only runs the arena probe vs the db chooser.
"""
from __future__ import annotations

import argparse
import random

import chess
import numpy as np
import torch
import torch.nn as nn

from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize


class Scorer(nn.Module):
    def __init__(self, d_in=13, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FeatureField:
    """frozen field -> per-move feature rows for one position (one batched forward)."""

    def __init__(self, eng: KittyChess):
        self.eng = eng
        pi = eng.pi
        self.cols = [pi["WIN"], pi["DRAW"], pi["LOSS"]]

    def _dists(self, z):
        P = self.eng.poles.to(self.eng.device)
        da = torch.stack([self.eng.net.dA(z, P[[k]].expand(len(z), -1)) for k in self.cols], 1)
        db = torch.stack([self.eng.net.dB(z, P[[k]].expand(len(z), -1)) for k in self.cols], 1)
        return torch.cat([da, db], 1)                     # (n, 6)

    def rows(self, board):
        moves = list(board.legal_moves)
        if not moves:
            return None, []
        toks, globs = [tokenize(board)], []
        globs = [toks[0][1]]; toks = [toks[0][0]]
        for mv in moves:
            board.push(mv)
            tk, gl = tokenize(board)
            toks.append(tk); globs.append(gl)
            board.pop()
        with torch.no_grad():
            z = self.eng._embed(toks, globs)
            D = self._dists(z).float()
        par = D[0:1].expand(len(moves), -1)               # (n, 6)
        stm = torch.full((len(moves), 1), 1.0 if board.turn else -1.0, device=D.device)
        X = torch.cat([D[1:], par, stm], 1)               # (n, 13)
        return X.cpu(), moves


def play_selfplay(ff, scorer, opening, temp=1.0, max_plies=150, start_fen=None):
    """one game, both sides sampling from the scorer; returns records + result."""
    b = chess.Board(start_fen) if start_fen else chess.Board()
    for u in opening:
        b.push_uci(u)
    recs = []                                             # (X, idx, mover_is_white)
    while not b.is_game_over(claim_draw=True) and b.ply() < max_plies:
        X, moves = ff.rows(b)
        if X is None:
            break
        with torch.no_grad():
            logits = scorer(X) / temp
            idx = int(torch.distributions.Categorical(logits=logits).sample())
        recs.append((X, idx, b.turn == chess.WHITE))
        b.push(moves[idx])
    out = b.outcome(claim_draw=True)
    res = 0.0 if out is None or out.winner is None else (1.0 if out.winner else -1.0)
    return recs, res


def arena_probe(ff, scorer, eng, rounds=8, seed=0):
    """paired openings: scorer(argmax) vs the db threat-first chooser."""
    rng = random.Random(seed)
    def scorer_move(b):
        X, moves = ff.rows(b)
        if X is None:
            return None
        with torch.no_grad():
            return moves[int(scorer(X).argmax())]
    score = 0.0
    for _ in range(rounds):
        op = []
        bb = chess.Board()
        for _ in range(6):
            m = rng.choice(list(bb.legal_moves)); op.append(m.uci()); bb.push(m)
        for scorer_white in (True, False):
            b = chess.Board()
            for u in op:
                b.push_uci(u)
            while not b.is_game_over(claim_draw=True) and b.ply() < 200:
                mv = scorer_move(b) if (b.turn == chess.WHITE) == scorer_white else eng.choose(b)
                if mv is None:
                    break
                b.push(mv)
            out = b.outcome(claim_draw=True)
            r = 0.5 if out is None or out.winner is None else \
                (1.0 if (out.winner == chess.WHITE) == scorer_white else 0.0)
            score += r
    return score / (2 * rounds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games-per", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm", type=int, default=0,
                    help="behavior-cloning warm start: N positions labeled by the db "
                         "threat-first chooser before RL (cold self-play cannot finish games)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    eng = KittyChess(args.ckpt, args.device)
    ff = FeatureField(eng)
    scorer = Scorer()
    out_path = (args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt) + "_rlscorer.pt"
    import os
    if os.path.exists(out_path):
        scorer.load_state_dict(torch.load(out_path, map_location="cpu"))
        print(f"[rl] resumed scorer from {out_path}")
    if args.eval_only:
        print(f"[rl] arena probe vs db chooser: {arena_probe(ff, scorer, eng, rounds=15):.1%}")
        return

    opt = torch.optim.Adam(scorer.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    # COLD START (first smoke: 0% decisive from balanced starts -> zero REINFORCE signal):
    # self-play from the HANDICAPPED start pool -- decisive under any play, signal from iter 1.
    from catspace.io import paths as _paths
    import os as _os
    fens = []
    for tsv in ("piecedown_sfsf_all.tsv", "pawndown_sfsf_moves.tsv", "exchdown_sfsf_moves.tsv"):
        fp = _paths.derived(tsv)
        if _os.path.exists(fp):
            fens += [l.split("\t")[2] for l in open(fp) if l.count("\t") >= 3][:20000]
    print(f"[rl] {len(fens):,} handicapped start fens for self-play", flush=True)
    if args.warm > 0:
        # WARM START: play the db chooser against itself from handicapped starts, label every
        # position with its move, fit the scorer by cross-entropy. The scorer begins as a
        # distillation of threat-first (a competent, mate-delivering base) and RL then learns
        # what threat-first cannot express (standing-aware conduct, ruler trade-offs).
        Xs, ys = [], []
        while len(Xs) < args.warm:
            b = chess.Board(rng.choice(fens)) if fens else chess.Board()
            while not b.is_game_over(claim_draw=True) and b.ply() < 150 and len(Xs) < args.warm:
                X, moves = ff.rows(b)
                if X is None:
                    break
                mv = eng.choose(b)
                if mv is None or mv not in moves:
                    break
                Xs.append(X); ys.append(moves.index(mv))
                b.push(mv)
        print(f"[rl] warm start: {len(Xs):,} labeled decisions; fitting...", flush=True)
        wopt = torch.optim.Adam(scorer.parameters(), lr=1e-2)
        for ep in range(60):
            perm = np.random.permutation(len(Xs))
            tot = 0.0
            for j in perm:
                wopt.zero_grad()
                l = torch.nn.functional.cross_entropy(scorer(Xs[j])[None, :],
                                                      torch.tensor([ys[j]]))
                l.backward(); wopt.step(); tot += float(l)
            if ep % 20 == 19:
                acc = float(np.mean([int(scorer(Xs[j]).argmax()) == ys[j]
                                     for j in range(len(Xs))]))
                print(f"[rl]   warm ep {ep+1}: CE {tot/len(Xs):.3f}  imitation acc {acc:.1%}",
                      flush=True)
        torch.save(scorer.state_dict(), out_path)
    for it in range(args.iters):
        batch_recs, returns = [], []
        for g in range(args.games_per):
            if fens and g % 4 != 3:                       # 3 of 4 games: handicapped starts
                recs, res = play_selfplay(ff, scorer, [], temp=args.temp,
                                          start_fen=rng.choice(fens))
            else:                                         # 1 of 4: balanced random opening
                op = []
                bb = chess.Board()
                for _ in range(6):
                    m = rng.choice(list(bb.legal_moves)); op.append(m.uci()); bb.push(m)
                recs, res = play_selfplay(ff, scorer, op, temp=args.temp)
            batch_recs.append(recs); returns.append(res)
        # mover-POV returns; baseline = batch mean of |returns| structure
        flat = []
        for recs, res in zip(batch_recs, returns):
            for X, idx, white in recs:
                flat.append((X, idx, res if white else -res))
        if not flat:
            continue
        base = float(np.mean([r for _, _, r in flat]))
        opt.zero_grad()
        loss = torch.zeros(())
        for X, idx, r in flat:
            logp = torch.log_softmax(scorer(X) / args.temp, 0)[idx]
            loss = loss - logp * (r - base)
        (loss / len(flat)).backward()
        opt.step()
        dec = float(np.mean([abs(r) for r in returns]))
        print(f"[rl] iter {it+1}/{args.iters}: {len(flat)} decisions, decisive {dec:.0%}, "
              f"mean return {np.mean(returns):+.2f}", flush=True)
        if (it + 1) % 10 == 0:
            torch.save(scorer.state_dict(), out_path)
            print(f"[rl] arena probe: {arena_probe(ff, scorer, eng):.1%} vs db chooser", flush=True)
    torch.save(scorer.state_dict(), out_path)
    print(f"[rl] saved {out_path}")


if __name__ == "__main__":
    main()
