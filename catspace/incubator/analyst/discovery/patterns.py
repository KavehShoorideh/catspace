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


def arrangement(fen: str) -> tuple:
    """The arrangement in words: ((term, value), ...) for every named piece present -- where it stands from their king."""
    A = _a()
    b = chess.Board(fen)
    n_us = sum(p.color == b.turn and p.piece_type != chess.KING for p in b.piece_map().values())
    n_them = sum(p.color != b.turn and p.piece_type != chess.KING for p in b.piece_map().values())
    present = {"W": True, "P": n_us >= 1, "P2": n_us >= 2, "P3": n_us >= 3, "O": n_them >= 1, "O2": n_them >= 2, "O3": n_them >= 3}
    terms = [f"({h} {x} L)" for x in NAMED if present[x] for h in ("dfile", "drank")]
    return tuple((t, A.vocab_values([fen], t)[0]) for t in terms)


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
    main()
