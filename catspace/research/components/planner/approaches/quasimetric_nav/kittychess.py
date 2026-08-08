#!/usr/bin/env python
"""kittychess.py -- THE most basic quasimetric-navigation engine (Kaveh 2026-08-08).

Move choice is THREAT-FIRST navigation on the outcome field, exactly as specified:
  'it's about holding back the draw and holding back the loss and going towards the win.'

For every legal move, embed the child and read d(child -> WIN/DRAW/LOSS poles), POV-flipped
(the child's mover is the opponent: our win is their loss). With d_bad = min(d_draw, d_loss):

  primary   maximise  d_bad - d_win      (push the nearest threat out AND pull the win in;
                                          a move that delays the draw but delays our win MORE
                                          worsens the margin and is rejected)
  tie-break maximise  d_bad              (among margin-equal moves, buy time vs the threat)
  <=5 pieces: Syzygy lookup outright     (the hybrid oracle -- no field consulted)

No search. One field readout per legal move. This is deliberately the dumbest possible planner:
its match results are the honest floor of what the geometry alone currently buys.
"""
from __future__ import annotations

import argparse
import random

import chess
import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB


class KittyChess:
    def __init__(self, ckpt, device="mps", cond_elo=None, use_tb=True):
        self.net, pay = load_net(ckpt, device)
        self.cfg = pay["cfg"]
        self.device = device
        self.cond_elo = cond_elo
        pn = self.cfg["pole_names"]
        self.poles = self.net.poles.poles.detach().float()
        self.pi = {n: pn.index(n) for n in ("WIN", "DRAW", "LOSS")}
        # basin-pov 'white' (2026-08-08): poles are colour-fixed (WIN = white wins), so
        # readouts select the pole by colour, not by whose turn it is at the queried node.
        self.white_pov = (self.cfg.get("train_args") or {}).get("basin_pov") == "white"
        # terminal-exemplar committor (2026-08-08): contrastive-mode ckpts train NO poles --
        # net.poles holds untouched init buffers (the frozen 45/65/139 readout). The validated
        # readout is median distance to real terminal boards (export_exemplars.py sidecar,
        # white-POV classes W/D/L). When the sidecar exists it REPLACES the pole readout.
        exp = (ckpt[:-3] if ckpt.endswith(".pt") else ckpt) + "_exemplars.pt"
        self.ex = None
        import os as _os
        if _os.path.exists(exp):
            pay_ex = torch.load(exp, map_location=device, weights_only=False)
            self.ex = {k: pay_ex[k].to(device) for k in ("W", "D", "L")}
        self.tb = TB() if use_tb else None
        if getattr(self.net, "split_head", False):
            self.dist = self.net.dB
        elif getattr(self.net, "dual", False):
            self.dist = self.net.qhead.d_base
        else:
            self.dist = self.net.iqe

    def _exd(self, z):
        """(len(z), 3) median distance to the W/D/L exemplar sets -- white-POV columns."""
        cols = []
        for k in ("W", "D", "L"):
            E = self.ex[k]
            dd = torch.stack([self.dist(z, E[e].expand(len(z), -1)) for e in range(len(E))], 1)
            cols.append(dd.median(1).values)
        return torch.stack(cols, 1)

    def _embed(self, toks, globs):
        tok_t = torch.from_numpy(np.array(toks).astype(np.int64)).to(self.device)
        glob_t = torch.from_numpy(np.array(globs).astype(np.float32)).to(self.device)
        if self.cond_elo is not None and getattr(self.net, "dual", False):
            cval = (self.cond_elo - 1500.0) / 500.0
            cond = torch.full((len(toks), self.net.qhead.proj_delta.in_features
                               - self.net.qhead.proj_base.in_features), cval, device=self.device)
            return self.net.encode_dual(tok_t, glob_t, cond)[1]
        return self.net.encode_q(tok_t, glob_t)

    def _tb_move(self, board):
        best, key = None, None
        for mv in board.legal_moves:
            board.push(mv)
            try:
                w, dz = self.tb.wdl_dtz(board)
            except Exception:
                w = None
            board.pop()
            if w is None:
                return None                      # any unprobeable child -> fall back to field
            # opponent POV: lower w better for us; among our wins prefer fast (small |dtz|),
            # among our losses prefer slow
            k = (w, (abs(dz) if w < 0 else -abs(dz)) if dz is not None else 0)
            if key is None or k < key:
                key, best = k, mv
        return best

    MATE = 1e6

    def margins(self, boards):
        """mover-POV threat-first margin for each board, one batched forward."""
        import numpy as np, torch
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
        toks, globs = zip(*(tokenize(b) for b in boards))
        with torch.no_grad():
            z = self._embed(list(toks), list(globs))
            if self.ex is not None:
                M = self._exd(z).float().cpu().numpy()   # white-POV cols: W / D / L
                out = []
                for i, b in enumerate(boards):
                    mw, ml = (0, 2) if b.turn else (2, 0)
                    out.append(float(min(M[i, 1], M[i, ml]) - M[i, mw]))
                return out
            D = {n: self.dist(z, self.poles[[k]].expand(len(z), -1).to(self.device))
                 .float().cpu().numpy() for n, k in self.pi.items()}
        if self.white_pov:
            out = []
            for i, b in enumerate(boards):
                mw, ml = ("WIN", "LOSS") if b.turn else ("LOSS", "WIN")
                out.append(float(min(D["DRAW"][i], D[ml][i]) - D[mw][i]))
            return out
        # mover-POV poles: my win = d->LOSS pole, my loss = d->WIN, draw = d->DRAW
        return [float(min(D["DRAW"][i], D["WIN"][i]) - D["LOSS"][i]) for i in range(len(boards))]

    def wdl(self, board, tau=5.0):
        """WHITE-POV [P(white wins), P(draw), P(black wins)] -- softmax over the three
        pole distances.

        This IS the eval bar (Kaveh 2026-08-08): no scalar, three distances softmaxed.
        Exact at terminals and in tablebase range; field softmax elsewhere. tau matches
        the basin-CE training temperature. Returns (probs, dists) with dists=None when
        the answer is exact.

        KNOWN LIMIT (2026-08-08 diagnostic): every current checkpoint trained the basin
        CE at w=10 where the force audit demanded ~1000x, so d(z->pole) is position-
        invariant (~0.1 over startpos..bare-kings) and the field part of this bar is
        flat until a re-based checkpoint lands."""
        wtm = board.turn == chess.WHITE
        if board.is_checkmate():                        # mover is mated
            return ([0.0, 0.0, 1.0] if wtm else [1.0, 0.0, 0.0]), None
        if board.is_game_over(claim_draw=True):
            return [0.0, 1.0, 0.0], None
        if self.tb is not None and len(board.piece_map()) <= 5:
            try:
                w, _ = self.tb.wdl_dtz(board)           # mover POV
                if w is not None:
                    if w == 0:
                        return [0.0, 1.0, 0.0], None
                    mover_wins = w > 0
                    return ([1.0, 0.0, 0.0] if mover_wins == wtm
                            else [0.0, 0.0, 1.0]), None
            except Exception:
                pass
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
        toks, globs = tokenize(board)
        with torch.no_grad():
            z = self._embed([toks], [globs])
            if self.ex is not None:                     # committor from geometry: white-POV cols
                d = self._exd(z)[0].float().cpu().numpy()
            else:
                D = {n: float(self.dist(z, self.poles[[k]].to(self.device)).float().cpu())
                     for n, k in self.pi.items()}
                if self.white_pov:                      # colour-fixed poles: direct readout
                    d = np.array([D["WIN"], D["DRAW"], D["LOSS"]])
                else:                                   # mover-POV reachability reading, then flip
                    d = np.array([D["LOSS"], D["DRAW"], D["WIN"]])
                    if not wtm:
                        d = d[::-1].copy()
        e = np.exp(-(d - d.min()) / tau)
        return [float(x) for x in (e / e.sum())], [float(x) for x in d]

    def _terminal_value(self, board):
        """exact value at game end / tablebase, mover POV, on the MATE scale."""
        if board.is_checkmate():
            return -self.MATE                       # mover is mated
        if board.is_game_over(claim_draw=True):
            return 0.0                              # stalemate/draw rules
        if self.tb is not None and len(board.piece_map()) <= 5:
            try:
                w, dz = self.tb.wdl_dtz(board)
                if w is not None:
                    if w == 0:
                        return 0.0
                    sign = 1 if w > 0 else -1
                    return sign * (self.MATE / 2 - abs(dz or 0))   # prefer faster wins
            except Exception:
                pass
        return None

    def bellman_residual(self, board):
        """|d(s -> mover's win pole) - 1 - min over moves d(child -> same pole)| in the field.

        The field's local self-consistency at this position. ~0 = the geometry's gradient is a
        valid plan here (trust it, search shallow); large = the map contradicts itself here
        (deliberate). White-POV pole selection; None for terminals or non-white-pov ckpts."""
        if not self.white_pov or board.is_game_over(claim_draw=True):
            return None
        moves = list(board.legal_moves)
        if not moves:
            return None
        toks, globs = [tokenize(board)], []
        globs = [toks[0][1]]; toks = [toks[0][0]]
        for mv in moves:
            board.push(mv)
            tk, gl = tokenize(board)
            toks.append(tk); globs.append(gl)
            board.pop()
        pole = self.poles[[self.pi["WIN" if board.turn else "LOSS"]]].to(self.device)
        with torch.no_grad():
            z = self._embed(toks, globs)
            d = self.dist(z, pole.expand(len(z), -1)).float().cpu().numpy()
        return float(d[0] - 1.0 - d[1:].min())

    class SearchStop(Exception):
        """raised mid-search when the caller's stop() turns true (navigation cancelled it)."""

    def negamax(self, board, depth, alpha, beta, stop=None):
        """alpha-beta minimax where the EVALUATION IS THE MARGIN (Kaveh 2026-08-08:
        'minimax alpha beta pruning but instead of evaluate we do it from the margin').
        Children are scored in ONE batched forward per node for move ordering, which is also
        what makes the pruning bite. Returns (value mover-POV, pv list of Moves)."""
        if stop is not None and stop():
            raise KittyChess.SearchStop
        tv = self._terminal_value(board)
        if tv is not None:
            return tv, []
        moves = list(board.legal_moves)
        if not moves:
            return 0.0, []
        children = []
        for m in moves:
            board.push(m); children.append(board.copy(stack=False)); board.pop()
        cm = self.margins(children)                 # child margins, OPPONENT'S POV
        order = sorted(range(len(moves)), key=lambda i: cm[i])   # their worst first
        if depth <= 1:
            best = order[0]
            return -cm[best], [moves[best]]
        best_v, best_pv = -float("inf"), []
        for i in order:
            v, pv = self.negamax(children[i], depth - 1, -beta, -alpha, stop)
            v = -v
            if v > best_v:
                best_v, best_pv = v, [moves[i]] + pv
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best_v, best_pv

    def search(self, board, depth=3, top=3, stop=None, progress=None,
               adaptive=False, max_extra=1, resid_scale=2.0):
        """root search: rows sorted by searched value, each with its PV. `stop` (callable) makes
        the search ABORTABLE lichess-style: checked at every node; on stop the rows finished so
        far are returned (partial MultiPV beats a frozen UI). `progress(rows_sorted)` fires after
        every completed root move for live streaming.

        `adaptive` (Kaveh 2026-08-08, both Bellman uses): the field's own Bellman residual at
        the root gates the depth -- consistent field, base depth; contradictory field, up to
        `max_extra` deeper (one extra ply per `resid_scale` of |residual|). The residual is
        attached to every returned row as row['resid']; search budget falls automatically as
        the field's consistency improves ([[adaptive_search_budget]])."""
        resid = None
        if adaptive:
            resid = self.bellman_residual(board)
            if resid is not None:
                depth += min(max_extra, int(abs(resid) / resid_scale))
        tv = self._terminal_value(board)
        moves = list(board.legal_moves)
        if not moves:
            return []
        rows = []
        for m in moves:
            board.push(m)
            try:
                v, pv = self.negamax(board, depth - 1, -float("inf"), float("inf"), stop)
            except KittyChess.SearchStop:
                board.pop()
                break
            board.pop()
            rows.append({"mv": m, "value": -v, "pv": [m] + pv,
                         "resid": resid, "depth_used": depth})
            if progress is not None:
                progress(sorted(rows, key=lambda r: r["value"], reverse=True))
        rows.sort(key=lambda r: r["value"], reverse=True)
        return rows

    def choose(self, board):
        moves = list(board.legal_moves)
        if not moves:
            return None
        if self.tb is not None and len(board.piece_map()) <= 5:
            mv = self._tb_move(board)
            if mv is not None:
                return mv
        toks, globs = [], []
        for mv in moves:
            board.push(mv)
            tk, gl = tokenize(board)
            toks.append(tk); globs.append(gl)
            board.pop()
        with torch.no_grad():
            z = self._embed(toks, globs)
            if self.ex is not None:
                M = self._exd(z).float().cpu().numpy()   # white-POV cols
                ow, ot = (0, 2) if board.turn else (2, 0)
                d_win, d_draw, d_loss = M[:, ow], M[:, 1], M[:, ot]
                D = None
            else:
                D = {n: self.dist(z, self.poles[[k]].expand(len(z), -1).to(self.device))
                     .float().cpu().numpy() for n, k in self.pi.items()}
        if D is None:
            pass
        elif self.white_pov:
            # colour-fixed poles: our win pole is by OUR colour, same at every child
            ours, theirs = ("WIN", "LOSS") if board.turn else ("LOSS", "WIN")
            d_win, d_draw, d_loss = D[ours], D["DRAW"], D[theirs]
        else:
            # POV flip at the child: our win = child-mover's LOSS
            d_win, d_draw, d_loss = D["LOSS"], D["DRAW"], D["WIN"]
        d_bad = np.minimum(d_draw, d_loss)
        margin = d_bad - d_win
        order = np.lexsort((-d_bad, -margin))    # primary margin desc, then d_bad desc
        return moves[int(order[0])]


def play(engine, opponent, engine_white, max_plies=300, start_fen=None):
    b = chess.Board(start_fen) if start_fen else chess.Board()
    while not b.is_game_over(claim_draw=True) and b.ply() < max_plies:
        mv = engine.choose(b) if b.turn == engine_white else opponent(b)
        if mv is None:
            break
        b.push(mv)
    out = b.outcome(claim_draw=True)
    if out is None or out.winner is None:
        return 0.5
    return 1.0 if out.winner == engine_white else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--opponent", default="random", choices=["random"])
    ap.add_argument("--cond-elo", type=float, default=None)
    ap.add_argument("--start", default="balanced", choices=["balanced", "piece-up"],
                    help="piece-up: opponent missing a knight -- the conversion test")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    eng = KittyChess(args.ckpt, args.device, args.cond_elo)
    rng = random.Random(0)

    def rand_opp(b):
        ms = list(b.legal_moves)
        return rng.choice(ms) if ms else None

    def start_fen(white):
        if args.start == "balanced":
            return None
        b = chess.Board()
        squares = [sq for sq, pc in b.piece_map().items()
                   if pc.piece_type == chess.KNIGHT and pc.color != white]
        b.remove_piece_at(rng.choice(squares))
        return b.fen()

    score, n = 0.0, 0
    for g in range(args.games):
        white = g % 2 == 0
        score += play(eng, rand_opp, white, start_fen=start_fen(white))
        n += 1
        if n % 10 == 0:
            print(f"[kitty] {n}/{args.games}  score {score/n:.2f}", flush=True)
    print(f"[kitty] FINAL vs {args.opponent} ({args.start}): {score/n:.3f} over {n} games "
          f"(0.5 = parity; random-vs-random ~0.5)")


if __name__ == "__main__":
    main()
