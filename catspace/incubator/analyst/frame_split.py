"""IS A DEEP LINE A FEW SHALLOW FRAMES? (Kaveh 2026-09-30: "to resolve all the tactics on the board you might need to
go 16 plies deep, because there might be 16 little tactics all around the board; but if they are organized in
two-tactic pairs, each frame needs only two plies ... that's how we reduce depth by keeping track of the frames --
that's what I'm betting on").

The test of the bet: Stockfish's principal variation for a puzzle step, cut at its PAYOFF PLY (sf_depth: where the
line has secured its gain -- what a search must reach), split into FRAMES by Kaveh's rules -- a reply is on the same
square, anything elsewhere is another frame (an intermezzo): two plies belong to one frame when they share a square
(the same piece moving on, a capture or a reply on a square the other touched), and a check stays with the reply it
forces (the king's frame joins). The grouping is TRANSITIVE (a ply sharing a square with any ply of a frame joins it),
the most merging reading -- it can only make frames longer than Kaveh's rule, so small frames here are strong support.
Printed: how many frames a line touches, the plies of the longest frame, and the SWITCHES (consecutive plies in
different frames: the intermezzi) -- if a 16-ply line is several frames of 2-4 plies, depth is composition; if one
frame alone runs 10+ plies, it is not.

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


def _reads(b: chess.Board, m: chess.Move) -> set:
    """The squares a move reads: its from- and to-squares and the squares it passes (a slider, a double step, castling)."""
    out = {m.from_square, m.to_square} | set(chess.SquareSet(chess.between(m.from_square, m.to_square)))
    if b.is_en_passant(m):
        out.add(chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square)))
    if b.is_castling(m):
        rank = chess.square_rank(m.from_square); ks = chess.square_file(m.to_square) > chess.square_file(m.from_square)
        rook = chess.square(7 if ks else 0, rank)
        out |= {rook} | set(chess.SquareSet(chess.between(rook, m.from_square)))
    return out


def chain(b: chess.Board, line: list, attacks: bool = False) -> tuple:
    """The line's CRITICAL PATH: the longest chain of plies each enabled by an earlier one -- the depth that cannot be
    split into independent parts. Ply j depends on the LAST ply that changed a square j reads (it moves the piece that
    ply put there, through a square it cleared, onto a square it filled or emptied) and, in check, on the checking ply;
    with `attacks`, also on the last ply whose piece attacks j's from- or to-square (a threat answered, a square
    contested: more links, an upper bound). -> (length in plies, the path's ply indices)."""
    bb = b.copy(); writer = {}; threat = {}; best = []; prev = []
    for j, m in enumerate(line):
        deps = {writer[s] for s in _reads(bb, m) if s in writer}
        if attacks:
            deps |= {threat[s] for s in (m.from_square, m.to_square) if s in threat}
        if bb.is_check() and j > 0:
            deps.add(j - 1)
        k = max(deps, key=lambda i: best[i], default=None)
        best.append(1 + (best[k] if k is not None else 0)); prev.append(k)
        for s in _touched(bb, m):
            writer[s] = j
        bb.push(m)
        for s in chess.SquareSet(bb.attacks_mask(m.to_square)):
            threat[s] = j
    if not line:
        return 0, []
    j = max(range(len(line)), key=lambda i: best[i]); n = best[j]; path = []
    while j is not None:
        path.append(j); j = prev[j]
    return n, sorted(path)


def with_passes(path: list) -> int:
    """The plies a frame-local search needs to play a chain out when the other side may pass elsewhere: the chain's
    plies plus one pass between two consecutive chain plies of the same side (an even gap in the line)."""
    return len(path) + sum(1 for i, k in zip(path, path[1:]) if (k - i) % 2 == 0)


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
    ap.add_argument("--reuse", action="store_true", help="re-read the lines of the last run (frame_split.json); no engine")
    a = ap.parse_args(argv)
    from .gauge_puzzles import sample
    items = []
    for r in sample(str(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst"), a.n, 997):
        b = chess.Board(r["FEN"]); b.push_uci(r["Moves"].split()[0])
        items.append((f"sample:{r['PuzzleId']}", b.fen(), a.depth))
    for name in a.misses:
        try:
            for r in json.load(open(paths.derived(name))):
                if not r["solved"]:
                    s = next((s for s in r["steps"] if not s["tied"]), None)
                    if s is not None:
                        items.append((f"miss:{r['id']}", s["fen"], a.depth))
        except FileNotFoundError:
            pass
    if a.reuse:
        rows = json.load(open(paths.derived("frame_split.json")))
    else:
        with mp.get_context("fork").Pool(a.workers) as pool:
            rows = pool.map(_line, items, chunksize=2)
    for r in rows:                                                    # every line with its frames, for the dig
        if r["payoff"] is not None and r["payoff"] >= 2:
            b = chess.Board(r["fen"]); line = [chess.Move.from_uci(u) for u in r["pv"][:r["payoff"]]]
            lab = {i: k for k, f in enumerate(split(b, line)) for i in f}
            r["frames"] = [lab[i] for i in range(len(line))]
            bb = b.copy(); sans = []
            for m in line:
                sans.append(bb.san(m)); bb.push(m)
            r["san"] = sans
            longest = max(set(r["frames"]), key=r["frames"].count)      # a WALK: only pawn and king moves (a race, a king walk) --
            bb = b.copy(); walk = True                                   # long in plies, but the reachability frame resolves it
            for i, m in enumerate(line):                                 # without search (piece_reach: KPK 474/474)
                if r["frames"][i] == longest and bb.piece_type_at(m.from_square) not in (chess.PAWN, chess.KING):
                    walk = False
                bb.push(m)
            r["longest_walk"] = walk
            r["chain"], r["chain_path"] = chain(b, line)
            r["chain_att"], path_att = chain(b, line, attacks=True)
            r["depth"], r["depth_att"] = with_passes(r["chain_path"]), with_passes(path_att)
            bb = b.copy(); kinds = []
            for m in line:
                kinds.append(bb.piece_type_at(m.from_square)); bb.push(m)
            r["chain_walk"] = all(kinds[i] in (chess.PAWN, chess.KING) for i in r["chain_path"])
    json.dump(rows, open(paths.derived("frame_split.json"), "w"))
    for tag in ("sample", "miss"):
        xs = []
        for r in rows:
            if not r["pid"].startswith(tag + ":") or "frames" not in r:
                continue
            fr = r["frames"]; sizes = [fr.count(k) for k in set(fr)]
            xs.append((len(fr), len(sizes), max(sizes), sum(1 for x, y in zip(fr, fr[1:]) if x != y), r, r["longest_walk"]))
        if not xs:
            continue
        print(f"OBS {tag}: {len(xs)} of {sum(1 for r in rows if r['pid'].startswith(tag + ':'))} lines with a payoff at ply 2+; payoff ply "
              f"median {st.median(x[0] for x in xs):g}; frames a line median {st.median(x[1] for x in xs):g}; the longest frame's plies "
              f"median {st.median(x[2] for x in xs):g}; switches median {st.median(x[3] for x in xs):g}")
        long_ = [x for x in xs if x[0] >= 8]
        if long_:
            print(f"   lines paying off at 8+ plies ({len(long_)}): frames median {st.median(x[1] for x in long_):g}, the longest frame median "
                  f"{st.median(x[2] for x in long_):g} plies; the longest frame 4 plies or fewer in {sum(1 for x in long_ if x[2] <= 4)}, "
                  f"5-9 in {sum(1 for x in long_ if 5 <= x[2] <= 9)}, 10 or more in {sum(1 for x in long_ if x[2] >= 10)}; the whole line "
                  f"one frame in {sum(1 for x in long_ if x[1] == 1)}; switches median {st.median(x[3] for x in long_):g}")
            big = [x for x in long_ if x[2] >= 5]
            print(f"   of the {len(big)} with a longest frame of 5+ plies: a WALK (pawn and king moves only) in {sum(1 for x in big if x[5])}; "
                  f"with pieces (tactics) in {sum(1 for x in big if not x[5])}, their longest frame median "
                  f"{st.median([x[2] for x in big if not x[5]] or [0]):g} plies")
            for x in sorted(long_, key=lambda x: -x[0])[:3]:              # examples: each ply tagged with its frame (A, B, ...)
                r = x[4]
                print(f"   e.g. {r['pid']} ({x[0]} plies): " + " ".join(f"{s}[{chr(65 + k)}]" for s, k in zip(r["san"], r["frames"])))
            # the CRITICAL PATH (the bug check of the frames above: transitive grouping links any two plies that ever touch a
            # common square, so over a long line nearly everything merges -- the longest enabling chain is the real depth)
            for key, what in (("chain", "squares read"), ("chain_att", "squares read + attacks")):
                cs = [x[4][key] for x in long_]
                print(f"   critical path ({what}) of the 8+ lines: median {st.median(cs):g} plies (payoff median {st.median(x[0] for x in long_):g}); "
                      f"4 or fewer in {sum(1 for c in cs if c <= 4)}, 5-7 in {sum(1 for c in cs if 5 <= c <= 7)}, 8+ in {sum(1 for c in cs if c >= 8)}; "
                      f"the path is half the line or less in {sum(1 for x in long_ if 2 * x[4][key] <= x[0])}/{len(long_)}")
            for key, what in (("depth", "squares read"), ("depth_att", "squares read + attacks")):
                ds = [x[4][key] for x in long_]
                print(f"   the chain played out with passes elsewhere ({what}): median {st.median(ds):g} plies; 4 or fewer in "
                      f"{sum(1 for d in ds if d <= 4)}, 5-8 in {sum(1 for d in ds if 5 <= d <= 8)}, 9+ in {sum(1 for d in ds if d >= 9)}; "
                      f"shorter than the line in {sum(1 for x in long_ if x[4][key] < x[0])}/{len(long_)}")
            walks = [x for x in long_ if x[4]["chain_walk"]]
            print(f"   chains that are WALKS (pawn and king moves only; the reachability frame's, not search's): {len(walks)}/{len(long_)}; "
                  f"the others' chain with passes (squares read): median {st.median([x[4]['depth'] for x in long_ if not x[4]['chain_walk']] or [0]):g} plies")
            for x in sorted(long_, key=lambda x: -x[0])[:3]:              # examples: the critical path's plies starred
                r = x[4]; on = set(r["chain_path"])
                print(f"   path {r['pid']} ({r['chain']} of {x[0]}): " + " ".join(f"{s}{'*' if i in on else ''}" for i, s in enumerate(r["san"])))
    from .sf_depth import _ENG
    for e in _ENG.values():
        e.quit()


if __name__ == "__main__":
    main()
