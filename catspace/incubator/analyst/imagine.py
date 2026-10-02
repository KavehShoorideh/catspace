"""IMAGINED FRAMES -- a measurement before a build (Kaveh 2026-10-01: "we see the board, we want to imagine a future
where a frame might exist"; "basically we want high-value lemmas, and find interactions of lemmas where our opponent
cannot see"; "and imagine them, then find a way to them, tactically"; his pick: measure first).

An IMAGINED FRAME is a what-if (whatif.py): an edit of the board -- one more attacker on an enemy piece, an enemy
defender gone, one of our pieces standing elsewhere -- answered EXACTLY by the frames: what the edited frame would win.
High-value: a pawn or more. A move ADVANCES an imagined frame when it completes the edit or takes its first step
(whatif.realizes). For every legal move of the side to move, readings compared by how they rank the puzzle's solution:
- IMAGINED: the summed gains of the high-value frames the move advances (a first step counts half).
- INTERACTIONS: how many distinct enemy pieces the move completes a high-value frame against now (an attacker added
  or a defender removed) -- two or more is a double attack: one reply cannot stop both.
- PRE-EMPT: the opponent's imagined frames against us (guard edits) the move answers.
Baselines: the frames' one-ply values (the player's candidate order), the frames' quiet-threat read (the null-move
what-if), and chance (the mean rank of a random order). Ties take the mean rank. Also: the one-ply order with the
imagined value as its tie-break (how it would enter the player), and how often the solution advances a frame the
frames' quiet-threat read does not yet show (a future the frames cannot see one move ahead).

Why not the lemma library's values (lemma_gen v2, 3 pieces): they are means over each frame's world, not an
instance's stake (a first reading composed them: a free queen read +1.0). The library does SEE a fork -- its 3-piece
frames call every action of a forked queen and rook losing (`--lemma-view` prints a piece's frames) -- so the lemmas
can name what the exact frames measure: the explanation layer.

    python -m catspace.incubator.analyst.imagine --sets fresh300 misses70 left52
    python -m catspace.incubator.analyst.imagine --lemma-view "<fen>" e6
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time

import chess

from catspace.io import paths

HIGH = 1.0                                                          # a pawn
SETS = {"fresh300": ("frame_puzzles_lichess_zone_candidates_off300.json", "all"),
        "misses70": ("frame_puzzles_lichess_zone_candidates_off500.json", "failing"),
        "left52": ("frame_puzzles_lichess_zone_candidates_propose3_budgetdecider_price3e-06_depth_decider_no500_d4-6-8-10-12_pawnmoves_off500_onlymisses.json", "failing")}


def readings(b: chess.Board) -> dict:
    """Every legal move's readings (see the module note)."""
    from .chess_domain import passed
    from .fates import clear_caches, joint_threats, joint_threats_book, square_frame
    from .whatif import edits, realizes
    from . import frame_player as FP
    clear_caches()
    stand = FP.passed(b)
    before, book = joint_threats_book(stand, square_frame, select=FP.SELECT) if stand is not None else ((0.0, None, []), None)
    one = FP._one_ply_values(b, before, book)
    now_q = joint_threats(b, square_frame)[0]
    es = edits(b)
    attack = [e for e in es if e.kind in ("add", "remove", "relocate") and e.gain >= HIGH]
    guard = [e for e in es if e.kind == "guard" and e.gain >= HIGH]
    rows = {}
    for m in b.legal_moves:
        nb = b.copy(stack=False); nb.push(m)
        mate = nb.is_checkmate()
        pb = passed(nb)
        q = (joint_threats(pb, square_frame)[0] - now_q) if pb is not None else 0.0
        img = 0.0; targets = set()
        for e in attack:
            r = realizes(b, m, e)
            if r:
                img += e.gain if r == 1 else e.gain / 2
                if r == 1 and e.kind in ("add", "remove"):
                    targets.add(e.target)
        pre = sum(e.gain for e in guard if realizes(b, m, e) == 1)
        rows[m.uci()] = dict(mate=mate, one=one.get(m.uci(), 0.0), quiet=q, imagined=img, inter=len(targets), preempt=pre)
    return rows


_GOALS: dict = {}
_VALUES: dict = {}


def values(b: chess.Board) -> dict:
    """Every legal move's imagined value (the high-value frames it advances: double attacks counted, then the summed
    gains) -> {uci: (completes frames against 2+ enemy pieces, summed gain)}. Cached by position."""
    key = b.fen()
    if key in _VALUES:
        return _VALUES[key]
    from .whatif import edits, realizes
    attack = [e for e in edits(b) if e.kind in ("add", "remove", "relocate") and e.gain >= HIGH]
    out = {}
    for m in b.legal_moves:
        img = 0.0; targets = set()
        for e in attack:
            r = realizes(b, m, e)
            if r:
                img += e.gain if r == 1 else e.gain / 2
                if r == 1 and e.kind in ("add", "remove"):
                    targets.add(e.target)
        out[m.uci()] = (len(targets) >= 2, img)
    if len(_VALUES) > 256:
        _VALUES.clear()
    _VALUES[key] = out
    return out


def goals(b: chess.Board, k: int = 3) -> tuple:
    """THE IMAGINED FRAMES AS GOALS (Kaveh 2026-10-01: "imagine them, then find a way to them, tactically"; his pick: hand
    them to the search). The moves of the side to move ranked by the high-value imagined frames they advance -- double
    attacks first (frames completed against two or more enemy pieces at once), then the summed gains -- and the top k
    with the squares of the frames they advance: the targets, the removed defenders, the relocations' destinations, and
    the moves' own squares. -> ([ucis], frozenset of squares). Cached by position (the planner and the candidate set
    both ask)."""
    key = (b.fen(), k)
    if key in _GOALS:
        return _GOALS[key]
    from .whatif import edits, realizes
    attack = [e for e in edits(b) if e.kind in ("add", "remove", "relocate") and e.gain >= HIGH]
    scored = []
    for m in b.legal_moves:
        img = 0.0; targets = set(); squares = set()
        for e in attack:
            r = realizes(b, m, e)
            if r:
                img += e.gain if r == 1 else e.gain / 2
                squares |= {e.target} | ({e.arg} if e.kind in ("remove", "relocate") else set())
                if r == 1 and e.kind in ("add", "remove"):
                    targets.add(e.target)
        if img > 0:
            scored.append(((len(targets) >= 2, img), m, squares | {m.from_square, m.to_square}))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]
    out = ([m.uci() for _, m, _ in top], frozenset(q for _, _, sq in top for q in sq))
    if len(_GOALS) > 256:
        _GOALS.clear()
    _GOALS[key] = out
    return out


def position(args) -> dict | None:
    fen, sol, pid, tag = args
    b = chess.Board(fen)
    if b.legal_moves.count() < 2 or chess.Move.from_uci(sol) not in b.legal_moves:
        return None
    t0 = time.perf_counter()
    rows = readings(b)
    return dict(pid=pid, tag=tag, fen=fen, sol=sol, n=len(rows), rows=rows, secs=round(time.perf_counter() - t0, 1))


def _rank(rows: dict, sol: str, key) -> float:
    g = (lambda r: r[key]) if isinstance(key, str) else key
    f = lambda r: (r["mate"], g(r))                                  # a mating move first, whatever the reading
    v = f(rows[sol])
    above = sum(1 for r in rows.values() if f(r) > v); tie = sum(1 for r in rows.values() if f(r) == v)
    return above + (tie + 1) / 2


def items_for(name: str) -> list:
    dump, which = SETS[name]
    out = []
    for r in json.load(open(paths.derived(dump))):
        if which == "all":
            out += [(s["fen"], s["sol"], r["id"], name) for s in r["steps"]]
        elif not r["solved"]:
            s = next((x for x in r["steps"] if not x["tied"]), None)
            if s is not None:
                out.append((s["fen"], s["sol"], r["id"], name))
    return out


ORDERS = [("frames one-ply (the player's order)", "one"), ("frames quiet threat", "quiet"), ("imagined frames advanced", "imagined"),
          ("interactions (targets advanced)", "inter"), ("pre-empt (their frames answered)", "preempt"),
          ("one-ply, imagined as tie-break", lambda r: (r["one"], r["imagined"]))]


def report(recs: list, log=print) -> None:
    for tag in sorted({r["tag"] for r in recs}):
        rs = [r for r in recs if r["tag"] == tag]
        chance = st.mean((r["n"] + 1) / 2 for r in rs)
        log(f"OBS {tag}: {len(rs)} positions, legal moves median {st.median(r['n'] for r in rs):g}; the solution's rank by each reading "
            f"(mean; top-1 / top-3; ties at their mean rank); chance mean {chance:.1f}")
        for name, key in ORDERS:
            rk = [_rank(r["rows"], r["sol"], key) for r in rs]
            log(f"   {name:38s} mean {st.mean(rk):5.1f}  top-1 {sum(1 for x in rk if x <= 1):3d}  top-3 {sum(1 for x in rk if x <= 3):3d}")
        e1 = e3 = 0.0                                                   # the fair control for a tie-break: a RANDOM one (expected)
        for r in rs:
            f = lambda x: (x["mate"], x["one"])
            v = f(r["rows"][r["sol"]]); above = sum(1 for x in r["rows"].values() if f(x) > v); tie = sum(1 for x in r["rows"].values() if f(x) == v)
            e1 += (1.0 / tie) if above == 0 else 0.0
            e3 += max(0.0, min(1.0, (3 - above) / tie))
        log(f"   {'one-ply, RANDOM tie-break (expected)':38s}             top-1 {e1:5.1f}  top-3 {e3:5.1f}")
        sol = [r["rows"][r["sol"]] for r in rs]
        allm = [x for r in rs for x in r["rows"].values()]
        log(f"   the solution advances a high-value imagined frame in {sum(1 for x in sol if x['imagined'] > 0)} (all moves: "
            f"{sum(1 for x in allm if x['imagined'] > 0) / len(allm):.3f}); completes frames against two or more enemy pieces (a double "
            f"attack) in {sum(1 for x in sol if x['inter'] >= 2)} (all moves: {sum(1 for x in allm if x['inter'] >= 2) / len(allm):.3f}); "
            f"mates at once in {sum(1 for x in sol if x['mate'])}")
        unseen = [x for x in sol if x["imagined"] > 0 and x["quiet"] < HIGH and not x["mate"]]
        log(f"   ... of which the frames' quiet-threat read shows no threat after the move (the future unseen one move ahead): {len(unseen)}")


def lemma_view(fen: str, square: str, log=print) -> None:
    """The lemma library's frames on one piece: for each embedded frame with the piece as actor, its verdicts on the
    piece's actions (loses / keeps) and the lemma's typical loss when it loses (its 'holds' value)."""
    from collections import defaultdict
    from .chess_domain import domain
    from .lemma_apply import Applier, load_library
    app = Applier(domain(), load_library(paths.derived("lemmas/library_v2_size3.json"), paths.derived("lemmas/proved_v2_size3.json")))
    b = chess.Board(fen); sq = chess.parse_square(square)
    res = app.apply(b)
    frames = defaultdict(lambda: [0, 0, None])
    for (e, key), vs in res["verdicts"].items():
        if e != sq:
            continue
        for c, pred, fired, ids in vs:
            if ids[c.slot] == sq:
                f = frames[(c.idx, ids)]; f[0 if pred == "loses" else 1] += 1; f[2] = c
    log(f"OBS lemma view {fen} {square}: {sum(1 for m in b.legal_moves if m.from_square == sq)} legal actions; {len(frames)} frames with it as actor; "
        f"the largest frame has {max((c.frame.size for _, _, c in frames.values()), default=0)} pieces")
    for (idx, ids), (lo, ke, c) in sorted(frames.items(), key=lambda kv: (-kv[1][2].frame.size, -kv[1][0]))[:8]:
        v = c.value or {}
        log(f"   loses {lo:2d} keeps {ke:2d}  claim '{c.claim}', world means: holds {v.get('holds', 0):+.1f}, exceptions {v.get('exception', 0):+.1f}  "
            f"{c.text[:110]}  [{' '.join(chess.square_name(i) for i in ids)}]")


def main(argv=None):
    import multiprocessing as mp
    import os
    import sys
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(); ap.add_argument("--sets", nargs="+", default=list(SETS)); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--lemma-view", nargs=2, metavar=("FEN", "SQUARE"), default=None)
    a = ap.parse_args(argv)
    if a.lemma_view:
        return lemma_view(*a.lemma_view)
    out = paths.derived("imagine.jsonl")
    if not a.report_only:
        items = [x for s in a.sets for x in items_for(s)][:a.limit]
        t0 = time.perf_counter()
        with mp.get_context("fork").Pool(a.workers) as pool, open(out, "w") as fh:
            for r in pool.imap_unordered(position, items, chunksize=1):
                if r is not None:
                    fh.write(json.dumps(r) + "\n"); fh.flush()
        print(f"[imagine] {len(items)} positions -> {out} ({time.perf_counter() - t0:.0f}s)")
    report([json.loads(l) for l in open(out)])


if __name__ == "__main__":
    main()
