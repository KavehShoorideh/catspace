#!/usr/bin/env python
"""chessplank_uci.py -- ChessPlank as a standard UCI engine.

Speaks the protocol every off-the-shelf chess tool understands: cutechess-cli tournaments,
GUI analysis boards, lichess-bot bridges. Ignores clocks (the engine is single-readout and
near-instant); `go` always answers with the threat-first choice.

    .venv/bin/python -m ...chessplank_uci --ckpt <ckpt.pt> [--cond-elo E]
then point any UCI host at that command.
"""
from __future__ import annotations

import argparse
import sys

import chess

from catspace.research.components.planner.approaches.quasimetric_nav.chessplank import ChessPlank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cond-elo", type=float, default=None)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    eng = ChessPlank(args.ckpt, args.device, args.cond_elo)
    board = chess.Board()
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "uci":
            print("id name ChessPlank"); print("id author catspace"); print("uciok", flush=True)
        elif cmd == "isready":
            print("readyok", flush=True)
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd.startswith("position"):
            parts = cmd.split()
            board = chess.Board()
            if "fen" in parts:
                i = parts.index("fen")
                board = chess.Board(" ".join(parts[i + 1:i + 7]))
            if "moves" in parts:
                for u in parts[parts.index("moves") + 1:]:
                    board.push_uci(u)
        elif cmd.startswith("go"):
            mv = eng.choose(board)
            print(f"bestmove {mv.uci() if mv else '0000'}", flush=True)
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
