"""L2 WINDOWS as EXPLANATION TREES (ANALYST_PLAN 41.6, 41.7, phase A stage 1).

  python -m catspace.incubator.analyst.discovery.windows --pgn-index N [--page] [--depth 14 --multipv 6 --margin 60 --branches 8 --rollout 6]

For each material observable (capture / promotion with delta >= 1, or mate) the tree is built BACKWARD along the game: at each ply back the side to move
has REASONABLE moves = the teacher's multipv moves within a margin (centipawns, a dict keyed by Elo band with a default) of its best, plus the move played
(marked, reasonable or not). Each branch is followed by the teacher's line: the multipv PV of that move, its first --rollout plies, extended while the
next PV move is a capture (a quiet end). Outcome: SAME (the victim still loses >= delta, or is mated), AVOIDED (no such loss), OTHER (a different
observable, recorded). Going back stops when the reasonable branches in all would exceed --branches (ply 0 is always kept), or at move one.
INEVITABILITY PLY = the furthest ply back such that every reasonable branch from the observable back to it is SAME (contiguous). The LIKELIHOOD point
needs a band-conditioned policy (not phase A). Windows with the same start position are MERGED, their leaves grouped winning / losing / neutral for the
side to move at the start. The teacher is a pluggable protocol (teachers.py; StockfishTeacher is the checker only), the domain an adapter (observables.ChessDomain); this file
imports neither python-chess nor an engine.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from . import observables as OB
from . import teachers as TE

MARGIN = {"default": 60}          # centipawns by Elo band ("1800" = 1800-1999); the design makes the margin a knob per band


def build_tree(teacher, domain, fens: list, moves: list, obs, band: str, cap: int, rollout: int = 6) -> dict:
    """The backward tree along the game from obs.ply: plies k = 0, 1, ... ; stops when the reasonable branches would exceed cap (ply 0 kept) or at move one."""
    plies, total = [], 0
    for k in range(obs.ply + 1):
        i = obs.ply - k
        fen, played = fens[i], moves[i]
        mover = domain.side_to_move(fen)
        rs = dict(teacher.reasonable(fen, band))
        if k > 0 and total + len(rs) > cap:
            stop = "cap"
            break
        total += len(rs)
        brs = []
        for mv in list(rs) + ([played] if played not in rs else []):
            cp = rs[mv] if mv in rs else teacher.score(domain.apply(fen, mv), mover)
            line = teacher.rollout(fen, mv, rollout)
            b = dict(uci=mv, cp=cp, reasonable=mv in rs, played=mv == played)
            b.update(domain.outcome(fen, mv, line, cp, obs, rollout))
            b["san"] = domain.san(fen, [mv])[0]
            brs.append(b)
        plies.append(dict(k=k, ply=i, fen=fen, mover=mover, branches=brs))
    else:
        stop = "move one"
    # inevitability: the victim's reasonable moves are what can avoid the loss, so only the victim's plies count (the winner's alternatives are its own
    # choice); the strict version (every reasonable branch of both sides SAME) is recorded too. Both are contiguous from the observable; 0 = the observable itself.
    inev = strict = 0
    broke = sbroke = False
    for p in plies[1:]:
        ok = all(b["cls"] == "SAME" for b in p["branches"] if b["reasonable"])
        if p["mover"] == obs.victim and not ok:
            broke = True
        if not ok:
            sbroke = True
        if not broke:
            inev = p["k"]
        if not sbroke:
            strict = p["k"]
    return dict(observable=obs.asdict(), plies=plies, reasonable=total, depth=len(plies) - 1, inevitability=inev, inevitability_strict=strict, stop=stop)


def finish_window(w: dict, domain, cashed=True) -> dict:
    """start position, the leaves valued for the side to move at the start, the cashed flag (the loss fell in the game)."""
    start = w["plies"][-1]
    victim = w["observable"]["victim"]
    base = domain.loss(start["fen"], victim)
    sign = 1 if domain.side_to_move(start["fen"]) == victim else -1
    leaves = []
    for p in w["plies"]:
        for b in p["branches"]:
            v = sign * (b["end_bal"] - base)
            leaves.append(dict(k=p["k"], move=b["san"], cls=b["cls"], reasonable=b["reasonable"], played=b["played"], value=v, group="winning" if v >= 1 else "losing" if v <= -1 else "neutral"))
    w["start_fen"], w["leaves"], w["cashed"] = start["fen"], leaves, cashed
    return w


def merge(windows: list) -> list:
    """windows sharing a start position -> one group, leaves grouped winning / losing / neutral for the side to move at the start."""
    by = {}
    for n, w in enumerate(windows):
        g = by.setdefault(w["start_fen"], dict(start_fen=w["start_fen"], windows=[], leaves={"winning": [], "losing": [], "neutral": []}))
        g["windows"].append(n)
        for l in w["leaves"]:
            g["leaves"][l["group"]].append(dict(l, window=n))
    return list(by.values())


def material_observables(obs: list, domain=OB.DOMAIN) -> list:
    return [o for o in obs if o.kind in domain.material_kinds and (o.delta >= 1 or o.kind == "mate")]


def run_game(index: int, teacher, domain=OB.DOMAIN, cap=8, rollout=6, log=print, page=False, knobs=None) -> dict:
    from catspace.io import paths
    t0 = time.time()
    game = domain.load_game(index)
    gid = domain.game_id(game)
    moves, fens = domain.positions(game)
    obs = domain.stream(game)
    counts = {k: sum(o.kind == k for o in obs) for k in domain.kinds}
    log(f"OBS observables {gid}: " + ", ".join(f"{k} {n}" for k, n in counts.items()))
    band = domain.band_of(game)
    wins = []
    for o in material_observables(obs, domain):
        w = finish_window(build_tree(teacher, domain, fens, moves, o, band, cap, rollout), domain)
        wins.append(w)
        lv = {c: sum(1 for p in w["plies"] for b in p["branches"] if b["cls"] == c) for c in ("SAME", "AVOIDED", "OTHER")}
        log(f"OBS window {gid} {o.kind} ply {o.ply + 1} {o.side} wins {o.delta}: tree {w['reasonable']} reasonable branches, depth {w['depth']} plies back, "
            f"inevitability {w['inevitability']} plies back (strict {w['inevitability_strict']}), stop {w['stop']}, leaves {lv}, cashed {w['cashed']}")
    groups = merge(wins)
    elapsed = round(time.time() - t0, 1)
    log(f"OBS game {gid}: {len(wins)} windows in {len(groups)} merged start positions, band {band}; {elapsed} s")
    res = dict(game=gid, index=index, band=band, knobs=dict(knobs or {}, branches=cap, rollout=rollout), counts=counts,
               observables=[o.asdict() for o in obs], windows=wins, merged=groups, seconds=elapsed, **domain.describe(game))
    out = paths.derived(f"windows/{gid}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    if page:
        from .window_page import write_page
        log(f"OBS page {write_page(res, domain)}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="A handful of games only (the machinery is the deliverable; no mining job).")
    ap.add_argument("--pgn-index", type=int, action="append", help="repeatable")
    ap.add_argument("--games", type=int, default=3, help="when no --pgn-index: the first N filtered games (small)")
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--multipv", type=int, default=6)
    ap.add_argument("--margin", type=int, default=60, help="centipawns; the default of a per-band dict")
    ap.add_argument("--branches", type=int, default=8, help="stop going back when the reasonable branches in all exceed this")
    ap.add_argument("--rollout", type=int, default=6, help="plies of the teacher's line followed after a branch (then through captures)")
    ap.add_argument("--page", action="store_true")
    a = ap.parse_args(argv)
    idx = a.pgn_index or list(range(min(a.games, 10)))
    teacher = TE.StockfishTeacher(a.depth, a.multipv, {"default": a.margin})
    try:
        for i in idx:
            run_game(i, teacher, cap=a.branches, rollout=a.rollout, page=a.page, knobs=dict(depth=a.depth, multipv=a.multipv, margin=a.margin))
    finally:
        teacher.close()


if __name__ == "__main__":
    main()
