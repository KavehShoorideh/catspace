"""WHERE THE PASSER LEMMA FIRES (Kaveh 2026-10-04: "see if the theorem is useful in action").

On Lichess puzzles of a theme (pawnEndgame, promotion, advancedPawn), at every position of the solver's line: does
the detector say PROVED (the pawn promotes whatever they do), ONEAWAY (one hypothesis fails -- which, and does the
solution's move touch the target it names), or neither. Also on a set of game positions. Counts only; the lemma's
first form needs a bare enemy king and no sliders of ours, so the question is how often real positions are in it.

    python -m catspace.incubator.analyst.lean_passer_puzzles --theme pawnEndgame --n 500 --offset 17
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import chess

from .lean_backrank_puzzles import puzzle_rows
from .lean_ideas import LeanIdeas


def scan(rows: list, li: LeanIdeas, log=print) -> None:
    pos = 0
    status = Counter()
    failing = Counter()
    proved_plies = Counter()
    touch = Counter()                 # one-away: does the solution move touch a target / is it the push itself
    kings_bare = 0
    sliders = 0
    examples = []
    for row in rows:
        b = chess.Board(row["FEN"])
        mv = row["Moves"].split()
        b.push_uci(mv[0])
        sol = [chess.Move.from_uci(u) for u in mv[1:]]
        for i in range(0, len(sol), 2):
            pos += 1
            enemy = not b.turn
            if len(chess.SquareSet(b.occupied_co[enemy])) == 1:
                kings_bare += 1
            if (b.rooks | b.bishops | b.queens) & b.occupied_co[b.turn]:
                sliders += 1
            insts = li.passer(b)
            m = sol[i]
            for inst in insts:
                status[inst.status] += 1
                if inst.status == "PROVED":
                    proved_plies[inst.plies] += 1
                    if len(examples) < 3:
                        examples.append(("PROVED", b.fen(), chess.square_name(inst.square), b.san(m)))
                elif inst.status == "ONEAWAY":
                    failing[inst.failing] += 1
                    T = set(inst.targets)
                    b.push(m)
                    hit = m.to_square in T or m.from_square in T or bool(chess.SquareSet(b.attacks_mask(m.to_square)) & T)
                    b.pop()
                    is_push = m.from_square == inst.square
                    touch[(inst.failing, "push" if is_push else "touches" if hit else "other")] += 1
                    if len(examples) < 6 and inst.failing in ("pathClear", "kingOutside"):
                        examples.append(("ONEAWAY " + inst.failing, b.fen(), chess.square_name(inst.square), b.san(m),
                                         [chess.square_name(s) for s in inst.targets]))
            b.push(m)
            if i + 1 < len(sol):
                b.push(sol[i + 1])
    log(f"OBS passer fire rate: {pos} solver positions in {len(rows)} puzzles; bare enemy king in {kings_bare}, "
        f"our sliders present in {sliders}; pawn instances {dict(status)}; PROVED by plies {dict(sorted(proved_plies.items()))}; "
        f"one-away failing hypothesis {dict(failing)}")
    log(f"OBS one-away vs the solution move: {dict(touch)}")
    for ex in examples:
        log(f"  {ex}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="pawnEndgame")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--stride", type=int, default=97)
    ap.add_argument("--offset", type=int, default=17)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    rows = puzzle_rows(a.n, a.stride, a.offset, None, theme=a.theme)
    print(f"loaded {len(rows)} {a.theme} puzzles in {time.perf_counter() - t0:.0f}s")
    with LeanIdeas() as li:
        scan(rows, li)


if __name__ == "__main__":
    main()
