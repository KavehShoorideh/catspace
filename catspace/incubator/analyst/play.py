"""play.py -- the analyst as a player's eyes. Each of OUR turns: analyse the position (idea,
threats, what no longer works), play the idea. Each of THEIR turns: ask an opponent engine
(Maia via lc0 at nodes=1) and let the analyst say what that move did. Writes a transcript + PGN.
    python -m catspace.incubator.analyst.play --maia assets/engines/maia/maia-1100.pb.gz \
        --color white --depth 14 --plies 120 --out tmp/analyst_vs_maia.txt"""
from __future__ import annotations

import argparse
import time

import chess
import chess.engine
import chess.pgn

from catspace.io import paths
from .engine import Engine
from .ideas import analyse_position
from .verbalize import render, COLOR


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--maia", default=paths.engine("maia/maia-1100.pb.gz"))
    ap.add_argument("--color", default="white"); ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--multipv", type=int, default=4); ap.add_argument("--plies", type=int, default=120)
    ap.add_argument("--out", default="tmp/analyst_vs_maia.txt")
    a = ap.parse_args(argv)
    us = a.color == "white"
    t0 = time.perf_counter()
    maia = chess.engine.SimpleEngine.popen_uci(["lc0", f"--weights={a.maia}"])
    lines = [f"analyst ({a.color}, Stockfish depth {a.depth}, multipv {a.multipv}) vs Maia-1100 (lc0 nodes=1)", ""]
    board = chess.Board(); boards, sans = [board.copy()], []
    with Engine(depth=a.depth) as eng:
        while not board.is_game_over(claim_draw=True) and board.ply() < a.plies:
            prev = boards[-3] if len(boards) >= 3 else None
            kb = sans[-1] if sans else None
            if board.turn == us:
                rec = analyse_position(board, eng, prev=prev, killed_by=kb, multipv=a.multipv)
                move = chess.Move.from_uci(rec.idea.uci)
                lines.append(render(rec, alternatives=False)); lines.append(f"  >> we play {board.san(move)}"); lines.append("")
            else:
                move = maia.play(board, chess.engine.Limit(nodes=1)).move
                rec = analyse_position(board, eng, prev=prev, killed_by=kb, played=move, multipv=a.multipv)
                tag = f" ({rec.verdict})" if rec.verdict else ""
                lines.append(f"[{board.fullmove_number}{'.' if board.turn else '...'}] Maia plays {board.san(move)}{tag}")
                if rec.verdict in ("blunder", "mistake", "inaccuracy"):
                    lines.append(render(rec, alternatives=False).split("\n", 1)[1].split("\n")[0])   # the verdict line only
                lines.append("")
            sans.append(board.san(move)); board.push(move); boards.append(board.copy())
        res = board.result(claim_draw=True)
        lines.append(f"RESULT {res} after {board.ply()} plies | {time.perf_counter()-t0:.0f}s | engine calls {eng.calls}, cache hits {eng.hits}")
    maia.quit()
    game = chess.pgn.Game.from_board(board); game.headers["White"] = "analyst" if us else "Maia-1100"; game.headers["Black"] = "Maia-1100" if us else "analyst"; game.headers["Result"] = res
    lines.append(""); lines.append(str(game))
    open(a.out, "w").write("\n".join(lines))
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
