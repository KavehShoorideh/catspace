"""THE PUZZLE MISSES, CLASSIFIED (Kaveh 2026-09-28: "dig the puzzle misses"). Reads frame_puzzles' dump and, for every
puzzle not solved, looks at the step where it failed: the solution move's kind (quiet, capture, check, promotion), whether
the reply ply looked at it at all (a candidate), the frames' value of the solution and of their pick, and Stockfish's
view of both (the gap confirms the miss). Prints the counts and examples per kind, for the dig.

    python -m catspace.incubator.analyst.puzzle_dig --ledger two_ply_candidates --show 4
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import chess

from catspace.io import paths

from .engine import Engine


def kind(b: chess.Board, m: chess.Move) -> str:
    nb = b.copy(); nb.push(m)
    if nb.is_checkmate():
        return "mate"
    if m.promotion:
        return "promotion"
    if b.gives_check(m):
        return "check" + (" capture" if b.is_capture(m) else "")
    if b.is_capture(m):
        return "capture"
    return "quiet"


def main(argv=None):
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="lichess"); ap.add_argument("--ledger", default="two_ply_candidates")
    ap.add_argument("--show", type=int, default=4); ap.add_argument("--depth", type=int, default=16)
    a = ap.parse_args(argv)
    res = json.load(open(paths.derived(f"frame_puzzles_{a.source}_{a.ledger}.json")))
    misses = [r for r in res if not r["solved"]]
    eng = Engine(threads=8, depth=a.depth)
    rows = []
    try:
        for r in misses:
            k = next(i for i, s in enumerate(r["steps"]) if not s["tied"])
            s = r["steps"][k]; b = chess.Board(s["fen"])
            sol = chess.Move.from_uci(s["sol"]); pick = chess.Move.from_uci(s["best"][0])
            ls = eng.analyse(b, root_moves=[sol], depth=a.depth); lp = eng.analyse(b, root_moves=[pick], depth=a.depth)
            gap = (ls[0].cp - lp[0].cp) if ls and lp else None
            rows.append(dict(r=r, step=k, fen=s["fen"], sol=b.san(sol), pick=b.san(pick), kind=kind(b, sol), pick_kind=kind(b, pick),
                             candidate=s["v_sol"] is not None, v_sol=s["v_sol"], v_top=s["v_top"], gap=gap,
                             sf_line=ls[0].san_line(b, 8) if ls else "", pick_line=lp[0].san_line(b, 6) if lp else ""))
    finally:
        eng.close()
    n = len(rows)
    print(f"[puzzle_dig] {a.source} {a.ledger}: {n} misses of {len(res)}")
    print("OBS the solution move's kind at the failing step: " + ", ".join(f"{k} {c}" for k, c in Counter(x["kind"] for x in rows).most_common()))
    print(f"OBS the solution move was not a candidate for the reply ply in {sum(not x['candidate'] for x in rows)}/{n}; "
          f"the failing step is the first move in {sum(x['step'] == 0 for x in rows)}/{n}")
    print(f"OBS Stockfish prefers the solution to the frames' pick by 1.5+ pawns in {sum(1 for x in rows if x['gap'] is not None and x['gap'] >= 150)}/{n} "
          f"(by less: {sum(1 for x in rows if x['gap'] is not None and x['gap'] < 150)} -- the pick may be fine too)")
    print("OBS themes of the misses: " + ", ".join(f"{t} {c}" for t, c in Counter(t for x in rows for t in x["r"]["themes"]
                                                                                if t not in ("short", "long", "veryLong", "middlegame", "endgame", "crushing", "advantage")).most_common(14)))
    groups = defaultdict(list)
    for x in rows:
        groups[x["kind"] if x["candidate"] else x["kind"] + " (not a candidate)"].append(x)
    for g, xs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"\n== {g}: {len(xs)}")
        for x in xs[:a.show]:
            r = x["r"]
            print(f"  {r['id']} ({r['rating']}; {' '.join(t for t in r['themes'][:4])}) step {x['step'] + 1}: {x['fen']}")
            print(f"    solution {x['sol']} (frames {x['v_sol'] if x['v_sol'] is not None else '-'}; Stockfish {x['sf_line']}); frames' pick {x['pick']} "
                  f"({x['pick_kind']}, frames {x['v_top']:+g}; Stockfish {x['pick_line']}); gap {x['gap']}")


if __name__ == "__main__":
    main()
