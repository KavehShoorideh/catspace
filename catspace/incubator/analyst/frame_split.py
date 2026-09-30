"""IS A DEEP LINE A FEW SHALLOW FRAMES? (Kaveh 2026-09-30: "to resolve all the tactics on the board you might need to
go 16 plies deep, because there might be 16 little tactics all around the board; but if they are organized in
two-tactic pairs, each frame needs only two plies ... that's how we reduce depth by keeping track of the frames --
that's what I'm betting on").

The test of the bet: Stockfish's principal variation for a puzzle step, cut at its PAYOFF PLY (sf_depth: where the
line has secured its gain -- what a search must reach), split into FRAMES by Kaveh's rules -- a reply is on the same
square, anything elsewhere is another frame (an intermezzo): two plies belong to one frame when they share a square
(the same piece moving on, a capture or a reply on a square the other touched), and a check stays with the reply it
forces (the king's frame joins). Printed: how many frames a line touches, and the plies of the longest frame -- if a
16-ply line is several frames of 2-4 plies, depth is composition; if one frame alone runs 10+ plies, it is not.

    python -m catspace.incubator.analyst.frame_split --n 300
"""
from __future__ import annotations

import argparse
import json

import chess

from catspace.io import paths
from .sf_depth import payoff_ply


def _touched(b: chess.Board, m: chess.Move) -> set:
    out = {m.from_square, m.to_square}
    if b.is_en_passant(m):
        out.add(chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square)))
    if b.is_castling(m):
        rank = chess.square_rank(m.from_square); ks = chess.square_file(m.to_square) > chess.square_file(m.from_square)
        out |= {chess.square(7 if ks else 0, rank), chess.square(5 if ks else 3, rank)}
    return out


def split(b: chess.Board, line: list) -> list:
    """The line's plies grouped into frames (lists of ply indices)."""
    parent = list(range(len(line)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def join(x, y):
        parent[find(x)] = find(y)
    bb = b.copy(); touched = []
    for i, m in enumerate(line):
        t = _touched(bb, m)
        for j in range(i):                                            # a reply on the same square: one frame
            if t & touched[j]:
                join(i, j)
        check = bb.gives_check(m)
        touched.append(t)
        bb.push(m)
        if check and i + 1 < len(line):                               # a check and the reply it forces: the king's frame joins
            join(i, i + 1)
    groups = {}
    for i in range(len(line)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values())


def _line(args):
    pid, fen, depth = args
    from .sf_depth import _engine
    b = chess.Board(fen)
    info = _engine().analyse(b, chess.engine.Limit(depth=depth, time=20.0))
    pv = info.get("pv", [])
    pay = payoff_ply(b, pv, info.get("score"))
    return dict(pid=pid, fen=fen, pv=[m.uci() for m in pv], payoff=pay)


def main(argv=None):
    import multiprocessing as mp
    import statistics as st
    import chess.engine  # noqa: F401
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--misses", nargs="*", default=["frame_puzzles_lichess_zone_candidates.json",
                    "frame_puzzles_lichess_zone_candidates_off500.json", "frame_puzzles_lichess_zone_candidates_off600.json"])
    a = ap.parse_args(argv)
    from .gauge_puzzles import sample
    items = []
    for r in sample(str(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst"), a.n, 997):
        b = chess.Board(r["FEN"]); b.push_uci(r["Moves"].split()[0])
        items.append((f"solved-sample:{r['PuzzleId']}", b.fen(), a.depth))
    for name in a.misses:
        try:
            for r in json.load(open(paths.derived(name))):
                if not r["solved"]:
                    s = next((s for s in r["steps"] if not s["tied"]), None)
                    if s is not None:
                        items.append((f"miss:{r['id']}", s["fen"], a.depth))
        except FileNotFoundError:
            pass
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(_line, items, chunksize=2)
    json.dump(rows, open(paths.derived("frame_split.json"), "w"))
    for tag in ("solved-sample", "miss"):
        xs = []
        for r in rows:
            if not r["pid"].startswith(tag + ":") or r["payoff"] is None or r["payoff"] < 2:
                continue
            b = chess.Board(r["fen"]); line = [chess.Move.from_uci(u) for u in r["pv"][:r["payoff"]]]
            fr = split(b, line)
            xs.append((r["payoff"], len(fr), max(len(f) for f in fr)))
        if not xs:
            continue
        long_ = [x for x in xs if x[0] >= 8]
        print(f"OBS {tag}: {len(xs)} lines with a material payoff; payoff ply median {st.median(x[0] for x in xs):g}; frames a line median "
              f"{st.median(x[1] for x in xs):g}; the longest frame's plies median {st.median(x[2] for x in xs):g}")
        if long_:
            print(f"   lines paying off at 8+ plies ({len(long_)}): frames median {st.median(x[1] for x in long_):g}, the longest frame median "
                  f"{st.median(x[2] for x in long_):g} plies; the longest frame 4 plies or fewer in {sum(1 for x in long_ if x[2] <= 4)}, "
                  f"10 or more in {sum(1 for x in long_ if x[2] >= 10)}; the whole line one frame in {sum(1 for x in long_ if x[1] == 1)}")
    from .sf_depth import _ENG
    for e in _ENG.values():
        e.quit()


if __name__ == "__main__":
    main()
