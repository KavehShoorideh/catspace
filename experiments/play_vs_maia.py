#!/usr/bin/env python
"""experiments/play_vs_maia.py -- put the trained full-board field into PLAY against MAIA (human-like
lc0 nets) and measure how it does (Kaveh: "how does the engine play against maia?").

The field player is COMMITTOR-GREEDY on field_fullgame_v3: for each legal move it evaluates the
committor c(s')=P(white win) of the resulting position and picks argmax if White / argmin if Black
(maximise MY win probability). This is the pure 1-ply VALUE policy on the field -- no search, no
opponent model yet (that is the z/T exploitation layer, next). It answers: does the committor, used
greedily, play sensible chess against a human-like opponent, and where does it break.

Maia = lc0 + maia-<elo>.pb.gz at nodes=1 (pure policy head = human-like move). Alternating colors,
optional random opening plies for diversity, W/D/L + score with a baseline, PGN saved.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import chess, chess.engine, chess.pgn
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_clock_field import ClockField
from catspace.train.scaffold import resolve_device


class CommittorGreedy:
    """1-ply committor-greedy field player. Keeps an lczero board mirror for real-history planes."""
    def __init__(self, ckpt, device, tau=0.0):
        self.dev = device; self.tau = tau
        p = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = p.get("cfg", {"d": 64, "ch": 128, "blocks": 8, "in_planes": 112})
        self.net = ClockField(cfg["d"], ch=cfg["ch"], blocks=cfg["blocks"], in_planes=cfg.get("in_planes", 112)).to(device)
        self.net.load_state_dict(p["state_dict"]); self.net.eval()
        from lczerolens import LczeroBoard
        self._LB = LczeroBoard

    def _committor(self, planes_list):
        if not planes_list:
            return np.zeros(0)
        x = torch.from_numpy(np.stack(planes_list)).to(self.dev)
        with torch.no_grad():
            return self.net.committor(x).cpu().numpy()       # P(white win)

    def _term_myval(self, board, my_white):
        r = board.result(claim_draw=True)
        return 1.0 if ((r == "1-0") == my_white) else (0.5 if r == "1/2-1/2" else 0.0)

    def select(self, lcboard, rng, depth=1):
        moves = list(lcboard.legal_moves)
        if not moves:
            return None, 0.5
        my_white = (lcboard.turn == chess.WHITE)
        if depth <= 1:                                       # 1-ply: committor of each successor
            planes, term = [], {}
            for i, m in enumerate(moves):
                lcboard.push(m)
                if lcboard.is_game_over(claim_draw=True):
                    term[i] = self._term_myval(lcboard, my_white)
                else:
                    term[i] = ("leaf", len(planes)); planes.append(lcboard.to_input_tensor().to(torch.float32).numpy())
                lcboard.pop()
            c = self._committor(planes)
            vals = np.array([t if not isinstance(t, tuple) else (c[t[1]] if my_white else 1 - c[t[1]]) for t in term.values()])
        else:                                                # 2-ply MINIMAX: worst-case over Maia replies
            leaves = []; move_reply = []
            for m in moves:
                lcboard.push(m)
                if lcboard.is_game_over(claim_draw=True):
                    move_reply.append([("term", self._term_myval(lcboard, my_white))]); lcboard.pop(); continue
                rr = []
                for r_ in lcboard.legal_moves:
                    lcboard.push(r_)
                    if lcboard.is_game_over(claim_draw=True):
                        rr.append(("term", self._term_myval(lcboard, my_white)))
                    else:
                        rr.append(("leaf", len(leaves))); leaves.append(lcboard.to_input_tensor().to(torch.float32).numpy())
                    lcboard.pop()
                move_reply.append(rr); lcboard.pop()
            c = self._committor(leaves)
            def leafval(t): return t[1] if t[0] == "term" else (c[t[1]] if my_white else 1 - c[t[1]])
            vals = np.array([min(leafval(t) for t in rr) if rr else 0.5 for rr in move_reply])  # Maia minimises my value
        if self.tau > 0:
            p = np.exp((vals - vals.max()) / self.tau); p /= p.sum(); i = rng.choice(len(moves), p=p)
        else:
            i = int(np.argmax(vals))
        return moves[i], float(vals[i])


def play_game(field, maia, field_is_white, opening_plies, max_plies, rng, maia_nodes, depth):
    from lczerolens import LczeroBoard
    board = LczeroBoard()
    # random opening plies (both sides) for diversity
    for _ in range(opening_plies):
        ms = list(board.legal_moves)
        if not ms: break
        board.push(ms[rng.integers(0, len(ms))])
    ply = board.ply()
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        if board.turn == (chess.WHITE if field_is_white else chess.BLACK):
            mv, _ = field.select(board, rng, depth=depth)
        else:
            mv = maia.play(board, chess.engine.Limit(nodes=maia_nodes)).move
        if mv is None: break
        board.push(mv); ply += 1
    res = board.result(claim_draw=True)
    return board, res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/field_fullgame_v3_final.pt")
    ap.add_argument("--maia-elo", type=int, default=1500)
    ap.add_argument("--maia-nodes", type=int, default=1)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--depth", type=int, default=1, help="1=committor-greedy, 2=2-ply minimax")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-pgn", default="artifacts/experiments/field_v3_vs_maia.pgn")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = resolve_device(args.device); rng = np.random.default_rng(args.seed); t0 = time.time()
    field = CommittorGreedy(args.ckpt, dev, tau=args.tau)
    wpath = f"data/engines/maia/maia-{args.maia_elo}.pb.gz"
    maia = chess.engine.SimpleEngine.popen_uci(["lc0", f"--weights={wpath}", "--backend=eigen"])
    print(f"[play] field_v3 (committor-greedy) vs maia-{args.maia_elo} (nodes={args.maia_nodes}) "
          f"| {args.games} games alternating colors depth={args.depth}", flush=True)

    W = D = L = 0; games_pgn = []
    for g in range(args.games):
        field_white = (g % 2 == 0)
        board, res = play_game(field, maia, field_white, args.opening_plies, args.max_plies, rng, args.maia_nodes, args.depth)
        # score from the FIELD's POV
        if res == "1/2-1/2": D += 1; s = 0.5
        elif (res == "1-0") == field_white: W += 1; s = 1.0
        else: L += 1; s = 0.0
        gp = chess.pgn.Game.from_board(board)
        gp.headers["White"] = "field_v3" if field_white else f"maia-{args.maia_elo}"
        gp.headers["Black"] = f"maia-{args.maia_elo}" if field_white else "field_v3"
        gp.headers["Result"] = res
        games_pgn.append(str(gp))
        print(f"  game {g+1}/{args.games} field={'W' if field_white else 'B'} -> {res} "
              f"(field {s}) | running W{W} D{D} L{L} [{time.time()-t0:.0f}s]", flush=True)
    maia.quit()
    n = W + D + L; score = (W + 0.5 * D) / n
    Path(args.save_pgn).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_pgn).write_text("\n\n".join(games_pgn))
    print(f"\nVERDICT field_v3 (committor-greedy, 1-ply) vs maia-{args.maia_elo}: "
          f"{W}W {D}D {L}L in {n} | SCORE {score:.3f} (0.5=even) | [{time.time()-t0:.0f}s]", flush=True)
    print(f"  PGN -> {args.save_pgn}")


if __name__ == "__main__":
    main()
