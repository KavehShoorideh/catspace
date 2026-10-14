"""THE PATTERN LEMMAS -- cycles whose net effect is a shift (Kaveh 2026-10-13: "a cycle of moves that ... pushes the king
one square ... and we get to start again ... the net result of the cycle is like translation ... we get to corner the king
with that"; "whatever it took for you to find that these are the right lemmas to add, that needs to become machinery so
that the proposer can suggest these same things. That means the vocabulary needs to be complemented").

The machinery, no hand-written chess:

  WORDS        the relative geometry of the base vocabulary: (dfile X L), (drank X L) -- where each named piece stands from
               their king -- and (fileof L), (rankof L); an ARRANGEMENT is the set of literals "(dfile X L) = a",
               "(drank X L) = b" over the named pieces
  PROPOSE      from any games: the same arrangement again, their king moved by a vector v, k moves of ours later -- a
               candidate CYCLE (arrangement, v, k)
  VERIFY       the arrangement placed everywhere on the board it fits; at each placement our exact solver (goals.rs
               forced_goal, a goal in words) asks: is the same arrangement, their king moved by v, forced within k moves
               against every defence? -- forced / stopped / unknown at every placement, nothing sampled
  THE LEMMA    the placements where it holds are the frame S, their king's distance to the edge along v the measure, k the
               period: the hypothesis of Lean `Ideas.Forced.forced_by_measure` (lean/ideas/Ideas/ForcedByMeasure.lean), checked
               on every position of S -- so the shifted arrangement is forced again and again to the edge, within k moves a
               step; at the edge, the finishing goal (a mate) asked the same way

  python -m catspace.incubator.analyst.discovery.patterns --games 'artifacts/experiments/surprise_loop_*_r*.json' --material KRvK
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import defaultdict

import chess

from catspace.io import paths

NAMED = ("W", "P", "P2", "P3", "O", "O2", "O3")   # every piece named (the completed vocabulary) -- placed relative to their king L
BUDGET = 2_000_000
MATE = 100_000


def _a():
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    return A


def all_named(b: chess.Board) -> bool:
    """Every piece has a word: a king and at most three other pieces a side (P, P2, P3; O, O2, O3)."""
    return all(len([p for p in b.piece_map().values() if p.color == c and p.piece_type != chess.KING]) <= 3 for c in (True, False))


_KINDS = {chess.QUEEN: "queens", chess.ROOK: "rooks", chess.BISHOP: "bishops", chess.KNIGHT: "knights", chess.PAWN: "pawns"}


def arrangement(fen: str) -> tuple:
    """The arrangement in words, the same whichever of two like pieces stands where (2026-10-13: the two rooks of the ladder
    swap ranks every step, and names ordered by square swapped with them -- no cycle was ever seen): our king's offset from
    their king, and for every other piece "a <kind> of <side> stands <df> files, <dr> ranks from their king" -- literals
    ((term, value), ...), sorted."""
    A = _a()
    b = chess.Board(fen)
    lk = b.king(not b.turn)
    lits = [(t, A.vocab_values([fen], t)[0]) for t in ("(dfile W L)", "(drank W L)")]
    up = 1 if b.turn == chess.WHITE else -1
    for sq, p in b.piece_map().items():
        if p.piece_type == chess.KING:
            continue
        df = chess.square_file(sq) - chess.square_file(lk)
        dr = (chess.square_rank(sq) - chess.square_rank(lk)) * up
        side = "our" if p.color == b.turn else "their"
        lits.append((f"(count (and {side}{_KINDS[p.piece_type]} (at (offset L {df} {dr}))))", 1))
    return tuple(sorted(lits))


def king_at(fen: str) -> tuple:
    A = _a()
    return A.vocab_values([fen], "(fileof L)")[0], A.vocab_values([fen], "(rankof L)")[0]


def recurrences(files: list, material: str) -> dict:
    """{(arrangement, v): dict(count, k, example)} -- the same arrangement again with their king moved by v."""
    from catspace.incubator.analyst.material_walk import _material
    out = {}
    for f in files:
        for g in json.load(open(f)):
            seen = defaultdict(list)
            for i, r in enumerate(g["rows"]):
                if r["material"] != material:
                    continue
                b = chess.Board(r["fen"])
                if not all_named(b):
                    continue
                arr, kl = arrangement(r["fen"]), king_at(r["fen"])
                for j, kl0, fen0 in seen[arr]:
                    v = (kl[0] - kl0[0], kl[1] - kl0[1])
                    if v != (0, 0):
                        c = out.setdefault((arr, v), dict(count=0, k=10 ** 9, example=fen0))
                        c["count"] += 1
                        if i - j < c["k"]:
                            c["k"], c["example"] = i - j, fen0
                seen[arr].append((i, kl, r["fen"]))
    return out


def placements(example: str) -> list:
    """The example's pieces kept where they stand from their king, their king put on every square: [(fen, (file, rank))]
    for every legal one (us to move)."""
    b = chess.Board(example)
    lk = b.king(not b.turn)
    rel = [(chess.square_file(s) - chess.square_file(lk), chess.square_rank(s) - chess.square_rank(lk), p) for s, p in b.piece_map().items()]
    out = []
    for f in range(8):
        for r in range(8):
            x = chess.Board(None); x.turn = b.turn
            ok = True
            for df, dr, p in rel:
                ff, rr = f + df, r + dr
                if not (0 <= ff < 8 and 0 <= rr < 8) or (p.piece_type == chess.PAWN and rr in (0, 7)):
                    ok = False; break
                x.set_piece_at(chess.square(ff, rr), p)
            if ok and x.is_valid() and not x.is_game_over():
                out.append((x.fen(), king_at(x.fen())))
    return out


def _verify_one(args):
    fen, lits, k, budget = args
    return _a().forced_goal(fen, lits, k, budget)


def verify_cycle(arr: tuple, v: tuple, k: int, example: str, pool, budget: int = BUDGET) -> dict:
    """At every placement: is the arrangement again, their king moved by v, forced within k moves?"""
    jobs, where = [], []
    for fen, (f, r) in placements(example):
        if not (0 <= f + v[0] < 8 and 0 <= r + v[1] < 8):
            continue
        lits = [(t, x, x) for t, x in arr] + [("(fileof L)", f + v[0], f + v[0]), ("(rankof L)", r + v[1], r + v[1])]
        jobs.append((fen, lits, k, budget)); where.append((fen, (f, r)))
    res = pool.map(_verify_one, jobs)
    holds = [(w, x) for w, x in zip(where, res) if x[0] == "forced"]
    return dict(placements=len(jobs), holds=len(holds), stopped=sum(x[0] == "stopped" for x in res),
                unknown=sum(x[0] == "unknown" for x in res), at=[w[1] for w, _ in holds],
                fewest=min((x[1] for _, x in holds), default=None), line=holds[0][1][2] if holds else None,
                example=holds[0][0][0] if holds else None)


def _mate_one(args):
    fen, depth, budget = args
    r = _a().forced_moves(fen, MATE, depth, budget)
    return min((n for _, st, n, _ in r if st == "forced"), default=None)


def finishing(example: str, v: tuple, holds_at: list, pool, depth: int = 6) -> dict:
    """Where the cycle ends -- the placements one step past the last that holds along v -- is a mate forced?"""
    edge = [(fen, kl) for fen, kl in placements(example) if kl not in holds_at and (kl[0] - v[0], kl[1] - v[1]) in holds_at]
    res = pool.map(_mate_one, [(fen, depth, BUDGET) for fen, _ in edge])
    return dict(edge=len(edge), mated=sum(r is not None for r in res), within=max((r for r in res if r), default=None))


def words(arr: tuple, v: tuple) -> str:
    from . import base_vocabulary as V
    lits = "; ".join(f"{V.gloss(V.parse(t))} = {x}" for t, x in arr)
    return f"while {lits}: the same again with their king {v[0]:+d} file(s), {v[1]:+d} rank(s)"


# ---------------------------------------------------------------------------------------------------------------------
# CYCLES AS FRAMES (Kaveh 2026-10-13: "we should be able to have loops in the frames ... several types of frames, loops or
# cycles and other non-loop frames. I guess it doesn't make sense to categorize it that way"): a loop is a frame whose
# measure falls once a cycle -- a horizon k; nothing else differs (Lean forced_by_measure covers every k). The proposer
# reads games for an arrangement seen again with their king shifted, in VIEWS -- every coordinate, the ranks alone, the
# files alone, with our king or without (the ladder of two rooks: their king wanders along the files, the rooks' ranks
# step up) -- and emits a CONDITIONAL FRAME: the condition the view's words (their king kept where the shift stays on the
# board), the measure their king's distance to the edge along the shift, the horizon the period. Checked EXACTLY on
# positions drawn inside the condition (the solver: the same arrangement again, their king shifted, within k, against
# every defence); kept when it holds on every one, and on held-out ones.
# ---------------------------------------------------------------------------------------------------------------------
VIEWS = ("full", "ranks", "files", "full-W", "ranks-W", "files-W")


def view_literals(fen: str, view: str) -> tuple:
    """The arrangement in one view, as literals (term, lo, hi) -- order-free, like pieces counted together."""
    from collections import Counter
    A = _a()
    b = chess.Board(fen)
    lk = b.king(not b.turn)
    up = 1 if b.turn == chess.WHITE else -1
    part, with_w = view.split("-")[0], not view.endswith("-W")
    lits = []
    if with_w:
        if part in ("full", "files"):
            lits.append(("(dfile W L)",) + (A.vocab_values([fen], "(dfile W L)")[0],) * 2)
        if part in ("full", "ranks"):
            lits.append(("(drank W L)",) + (A.vocab_values([fen], "(drank W L)")[0],) * 2)
    cells = Counter()
    for sq, p in b.piece_map().items():
        if p.piece_type == chess.KING:
            continue
        df = chess.square_file(sq) - chess.square_file(lk)
        dr = (chess.square_rank(sq) - chess.square_rank(lk)) * up
        kind = ("our" if p.color == b.turn else "their") + _KINDS[p.piece_type]
        cell = {"full": f"(at (offset L {df} {dr}))", "ranks": f"(rank (offset L 0 {dr}))", "files": f"(file (offset L {df} 0))"}[part]
        cells[(kind, cell)] += 1
    for (kind, cell), n in sorted(cells.items()):
        lits.append((f"(count (and {kind} {cell}))", n, n))
    return tuple(lits)


def view_shift(v: tuple, view: str) -> tuple:
    part = view.split("-")[0]
    return (v[0] if part in ("full", "files") else 0, v[1] if part in ("full", "ranks") else 0)


def view_recurrences(files: list, material: str) -> dict:
    """{(view, literals, shift): dict(count, k, example)} -- an arrangement in a view seen again, their king moved along it."""
    out = {}
    for f in files:
        for g in json.load(open(f)):
            seen = defaultdict(list)
            for i, r in enumerate(g["rows"]):
                if r["material"] != material:
                    continue
                kl = king_at(r["fen"])
                for view in VIEWS:
                    lits = view_literals(r["fen"], view)
                    for j, kl0, fen0 in seen[(view, lits)]:
                        v = view_shift((kl[0] - kl0[0], kl[1] - kl0[1]), view)
                        if v != (0, 0) and i - j <= 12:
                            c = out.setdefault((view, lits, v), dict(count=0, k=10 ** 9, example=fen0))
                            c["count"] += 1
                            if i - j < c["k"]:
                                c["k"], c["example"] = i - j, fen0
                    seen[(view, lits)].append((i, kl, r["fen"]))
    return out


def frame_of(material: str, lits: tuple, v: tuple) -> tuple:
    """The conditional frame: (key, the measure, the condition's literals with their king where the shift stays on the board)."""
    bounds = []
    if v[0]:
        bounds.append(("(fileof L)", 0, 7 - v[0]) if v[0] > 0 else ("(fileof L)", -v[0], 7))
    if v[1]:
        bounds.append(("(rankof L)", 0, 7 - v[1]) if v[1] > 0 else ("(rankof L)", -v[1], 7))
    cond = [tuple(x[:3]) for x in lits] + bounds
    text = "&&".join(f"{t}>={lo}&&{t}<={hi}" if lo != hi else f"{t}>={lo}&&{t}<={hi}" for t, lo, hi in cond)
    parts = []
    if v[0]:
        parts.append("(diff (fileof L) 7)" if v[0] > 0 else "(fileof L)")
    if v[1]:
        parts.append("(diff (rankof L) 7)" if v[1] > 0 else "(rankof L)")
    measure = parts[0] if len(parts) == 1 else f"(sum {parts[0]} {parts[1]})"
    return f"{material}:if:{text}", measure, cond


def _check_cycle(args):
    """At one position inside the condition: is the arrangement again, their king shifted by v, forced within k? (The
    goal is the arrangement's own literals -- a refinement narrows where the frame applies, not where it must arrive.)"""
    fen, lits, v, k, budget = args
    lits = [x for x in lits if not (len(x) == 4 and x[3] == "refined")]
    f, r = king_at(fen)
    goal = [tuple(x) for x in lits]
    if v[0]:
        goal.append(("(fileof L)", f + v[0], f + v[0]))
    if v[1]:
        goal.append(("(rankof L)", r + v[1], r + v[1]))
    return _a().forced_goal(fen, goal, k, budget)[0]


def refine(good: list, bad: list) -> tuple | None:
    """A word that keeps every position where the loop failed OUT and the most where it held IN (no named square) -- the
    condition's next literal (term, lo, hi); None when no word does."""
    import numpy as np
    from . import explorer as E
    M = E.matrix(good + bad)
    g, b = M[: len(good)], M[len(good):]
    allow = E.layer_mask(0, set())
    best = None
    for op in (">=", "<="):
        th = (b.max(axis=0) + 1) if op == ">=" else (b.min(axis=0) - 1)
        keep = ((g >= th) if op == ">=" else (g <= th)).sum(axis=0)
        keep = np.where(allow, keep, -1)
        j = int(keep.argmax())
        if keep[j] > 0 and (best is None or keep[j] > best[0]):
            best = (int(keep[j]), (E.terms()[j], int(th[j]), 99) if op == ">=" else (E.terms()[j], 0, int(th[j])))
    return best[1] if best else None


def check_frame(material: str, lits: tuple, v: tuple, k: int, pool, rng, n: int, budget: int = 500_000) -> dict:
    """Positions drawn inside the condition, each checked exactly: forced / stopped / unknown."""
    from .explorer import random_positions
    key, measure, cond = frame_of(material, lits, v)
    conds = [dict(term=t, op=op, theta=th) for t, lo, hi in cond for op, th in ((">=", lo), ("<=", hi))]
    fens = random_positions(material, n, rng, conds, tries=4000)
    res = pool.map(_check_cycle, [(f, lits, v, k, budget) for f in fens])
    return dict(key=key, measure=measure, drawn=len(fens), forced=res.count("forced"), stopped=res.count("stopped"),
                unknown=res.count("unknown"), failed=[f for f, x in zip(fens, res) if x != "forced"],
                held=[f for f, x in zip(fens, res) if x == "forced"])


def derive_loops(files: list, materials, top: int = 10, draw: int = 60, held_out: int = 60, workers: int | None = None,
                 log=print) -> dict:
    """THE LOOP FRAMES of the given games: {conditional frame key: [measure, "horizon:k"]} -- each a view seen again with
    their king shifted, checked exactly on positions drawn inside its condition, refined, held out."""
    import random
    from multiprocessing import get_context
    kept = {}
    with get_context("spawn").Pool(workers or max(1, (os.cpu_count() or 8) - 1)) as pool:
        for material in sorted(materials):
            t0 = time.time()
            rec = view_recurrences(files, material)
            cands = sorted(rec.items(), key=lambda kv: (-kv[1]["count"], len(kv[0][1]), kv[1]["k"]))
            log(f"OBS   {material}: {len(rec)} candidate loops ({time.time() - t0:.0f} s); checking the {min(top, len(cands))} seen most often")
            for (view, lits, v), c in cands[:top]:
                r = check_frame(material, lits, v, c["k"], pool, random.Random(f"{view}{lits}{v}"), draw)
                line = (f"OBS     {view}, seen {c['count']}x, period {c['k']}, their king {v}: drawn {r['drawn']} -- forced at {r['forced']}"
                        f", stopped {r['stopped']}, unknown {r['unknown']}")
                for rnd in range(3):
                    if not r["drawn"] or r["forced"] == r["drawn"] or not r["held"]:
                        break
                    w = refine(r["held"], r["failed"])
                    if w is None:
                        break
                    lits = tuple(lits) + ((w[0], w[1], w[2], "refined"),)
                    r = check_frame(material, lits, v, c["k"], pool, random.Random(f"r{rnd}{view}{lits}{v}"), draw)
                    line += f"; refined by {w[0]} [{w[1]}, {w[2]}]: {r['forced']}/{r['drawn']}"
                if r["drawn"] and r["forced"] == r["drawn"]:
                    h = check_frame(material, lits, v, c["k"], pool, random.Random(f"held{view}{lits}{v}"), held_out)
                    line += f"; HELD OUT {h['forced']}/{h['drawn']}"
                    if h["drawn"] and h["forced"] == h["drawn"]:
                        kept[r["key"]] = [r["measure"], f"horizon:{c['k']}"]
                        line += " -> KEPT"
                log(line)
    return kept


def frames_main(argv=None):
    """python -m catspace.incubator.analyst.discovery.patterns frames --games ... --material KRRvK"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--material", required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--draw", type=int, default=60, help="positions drawn inside a condition to check it")
    ap.add_argument("--held-out", type=int, default=60)
    a = ap.parse_args(argv)
    t0 = time.time()
    kept = derive_loops(sorted(glob.glob(a.games)), [a.material], a.top, a.draw, a.held_out)
    out = paths.derived(f"patterns/frames_{a.material}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(kept, open(out, "w"), indent=1)
    print(f"OBS {len(kept)} loop frame(s) kept -> {out} ({time.time() - t0:.0f} s)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="a glob of results JSONs (surprise_loop._game rows)")
    ap.add_argument("--material", required=True)
    ap.add_argument("--top", type=int, default=12, help="the most frequent candidates to verify")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    a = ap.parse_args(argv)
    t0 = time.time()
    rec = recurrences(sorted(glob.glob(a.games)), a.material)
    cands = sorted(rec.items(), key=lambda kv: (-kv[1]["count"], kv[1]["k"]))
    print(f"OBS {len(rec)} candidate cycles (an arrangement again, their king shifted) in {a.material} ({time.time() - t0:.0f} s); "
          f"verifying the {min(a.top, len(cands))} most frequent")
    from multiprocessing import get_context
    found = []
    with get_context("spawn").Pool(a.workers) as pool:
        for (arr, v), c in cands[: a.top]:
            r = verify_cycle(arr, v, c["k"], c["example"], pool)
            fin = finishing(c["example"], v, r["at"], pool) if r["holds"] else None
            found.append(dict(arrangement=arr, v=v, k=c["k"], seen=c["count"], **{k_: r[k_] for k_ in r if k_ != "at"},
                              holds_at=r["at"], finishing=fin))
            print(f"OBS   seen {c['count']}x, k {c['k']}: {words(arr, v)}")
            print(f"OBS     at {r['placements']} placements: forced at {r['holds']}, stopped at {r['stopped']}, unknown at {r['unknown']}"
                  + (f"; e.g. {r['example']}: {' '.join(r['line'][:2 * c['k']])}" if r["holds"] else "")
                  + (f"; where it ends, a mate forced at {fin['mated']} of {fin['edge']} (within {fin['within']})" if fin else ""))
    out = paths.derived(f"patterns/cycles_{a.material}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(found, open(out, "w"), indent=1)
    print(f"OBS -> {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "frames":
        frames_main(sys.argv[2:])
    else:
        main()
