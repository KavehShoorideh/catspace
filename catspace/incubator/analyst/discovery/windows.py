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
    leaves = []
    for p in w["plies"]:
        for b in p["branches"]:
            side0 = domain.side_to_move(start["fen"])
            v = domain.balance(b["end_fen"], side0) - domain.balance(start["fen"], side0)
            leaves.append(dict(k=p["k"], end_fen=b["end_fen"], move=b["san"], cls=b["cls"], reasonable=b["reasonable"], played=b["played"], value=v, group="winning" if v >= 1 else "losing" if v <= -1 else "neutral"))
    w["start_fen"], w["leaves"], w["cashed"] = start["fen"], leaves, cashed
    return w


def merge(windows: list, domain=OB.DOMAIN) -> list:
    """windows sharing a start position, or whose start lies inside an earlier window's span (the same forcing line), -> one group (the earliest start); its
    leaves grouped winning / losing / neutral for the side to move at the group's start."""
    groups = []
    for n, w in enumerate(windows):
        s, e = w["plies"][-1]["ply"], w["observable"]["ply"]
        g = next((g for g in groups if g["span"][0] <= s <= g["span"][1]), None)
        if g is None:
            g = dict(start_fen=w["start_fen"], span=[s, e], windows=[])
            groups.append(g)
        g["windows"].append(n)
        g["span"][1] = max(g["span"][1], e)
        if s < g["span"][0]:
            g["span"][0], g["start_fen"] = s, w["start_fen"]
    for g in groups:
        side = domain.side_to_move(g["start_fen"])
        base = domain.balance(g["start_fen"], side)
        g["leaves"] = {"winning": [], "losing": [], "neutral": []}
        for n in g["windows"]:
            for l in windows[n]["leaves"]:
                v = domain.balance(l["end_fen"], side) - base
                g["leaves"]["winning" if v >= 1 else "losing" if v <= -1 else "neutral"].append(dict(l, window=n, value=v))
    return groups


def important_observables(obs: list, domain=OB.DOMAIN) -> list:
    return [o for o in obs if o.kind in domain.important_kinds]


def material_observables(obs: list, domain=OB.DOMAIN) -> list:
    """the earlier, unfiltered set (settled captures / promotions of delta >= 1, and mate): kept only to report how many windows the filter removes."""
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
    before = len(material_observables(obs, domain))
    for o in important_observables(obs, domain):
        w = finish_window(build_tree(teacher, domain, fens, moves, o, band, cap, rollout), domain)
        wins.append(w)
        lv = {c: sum(1 for p in w["plies"] for b in p["branches"] if b["cls"] == c) for c in ("SAME", "AVOIDED", "OTHER")}
        log(f"OBS window {gid} {o.kind} ply {o.ply + 1} {o.side} wins {abs(o.delta)}: tree {w['reasonable']} reasonable branches, depth {w['depth']} plies back, "
            f"inevitability {w['inevitability']} plies back (strict {w['inevitability_strict']}), stop {w['stop']}, leaves {lv}, cashed {w['cashed']}")
    all_san = domain.san(fens[0], moves)
    for w in wins:
        start, end = w["plies"][-1]["ply"], w["observable"]["ply"]
        while end + 1 < len(moves) and domain.is_capture(fens[end], moves[end]) and domain.is_capture(fens[end + 1], moves[end + 1]):
            end += 1                                              # on to the settled end of the capture chain
        w["start_ply"], w["played_san"], w["played_uci"] = start, all_san[start:end + 1], moves[start:end + 1]
        best = max(teacher.reasonable(w["start_fen"], band), key=lambda r: r[1])[0]
        tl = teacher.rollout(w["start_fen"], best, len(w["played_uci"]))[:len(w["played_uci"])]    # the teacher's best play from the start, as long as the played line (or as far as it reaches)
        w["teacher_uci"], w["teacher_san"] = tl, domain.san(w["start_fen"], tl)
        w["diverge"] = next((i for i in range(min(len(tl), len(w["played_uci"]))) if tl[i] != w["played_uci"][i]), None if tl == w["played_uci"][:len(tl)] else 0)
    groups = merge(wins, domain)
    elapsed = round(time.time() - t0, 1)
    log(f"OBS game {gid}: windows before the importance filter {before}, after {len(wins)}; {len(groups)} merged groups ({len(wins) - len(groups)} windows merged into an earlier one), band {band}; {elapsed} s")
    res = dict(game=gid, index=index, band=band, knobs=dict(knobs or {}, branches=cap, rollout=rollout), counts=counts,
               san=all_san, observables=[o.asdict() for o in obs], windows=wins, merged=groups, seconds=elapsed, **domain.describe(game))
    out = paths.derived(f"windows/{gid}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    if page:
        from .window_page import write_page
        log(f"OBS page {write_page(res, domain)}")
    return res


def find_shift(k: int, n_games: int = 60, log=print) -> list:
    """No teacher: list (game index, ply, side, delta) of every material shift of delta >= k in the first `n_games` games that pass the filter
    (the --pgn-index numbering), to pick a window for --layers. -> [(index, ply, side, delta)]."""
    import io
    import subprocess
    import chess.pgn
    from . import ledger as LG
    p = subprocess.Popen(["zstd", "-dc", LG.LICHESS_PGN], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    buf, n, found, t0 = [], 0, [], time.time()

    def games():
        nonlocal buf
        for line in p.stdout:
            if line.startswith("[Event ") and buf:
                yield "".join(buf)
                buf = []
            buf.append(line)
        if buf:
            yield "".join(buf)
    try:
        for text in games():
            g = chess.pgn.read_game(io.StringIO(text))
            if g is None:
                continue
            h = g.headers
            try:
                if "Rated" not in h.get("Event", "") or "Bullet" in h.get("Event", "") or int(h.get("WhiteElo", 0)) < 1600 \
                        or int(h.get("BlackElo", 0)) < 1600 or h.get("Result") not in ("1-0", "0-1"):
                    continue
            except ValueError:
                continue
            if sum(1 for _ in g.mainline_moves()) < 50:
                continue
            for o in OB.stream(g):
                if o.kind == "shift" and abs(o.delta) >= k:
                    found.append((n, o.ply + 1, o.side, abs(o.delta)))
                    log(f"OBS shift game index {n} ({h.get('Site', '').rsplit('/', 1)[-1]}) ply {o.ply + 1}: {o.side} wins {abs(o.delta)}")
            n += 1
            if n >= n_games:
                break
    finally:
        p.kill()
    log(f"OBS scanned {n} games in {round(time.time() - t0)} s: {len(found)} shifts of {k} or more")
    return found


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
    ap.add_argument("--find-shift", type=int, default=0, metavar="K", help="no teacher: list the material shifts of K or more in the first --scan games, then stop")
    ap.add_argument("--scan", type=int, default=60)
    a = ap.parse_args(argv)
    if a.find_shift:
        find_shift(a.find_shift, a.scan)
        return
    idx = a.pgn_index or list(range(min(a.games, 10)))
    teacher = TE.StockfishTeacher(a.depth, a.multipv, {"default": a.margin})
    try:
        for i in idx:
            run_game(i, teacher, cap=a.branches, rollout=a.rollout, page=a.page, knobs=dict(depth=a.depth, multipv=a.multipv, margin=a.margin))
    finally:
        teacher.close()


if __name__ == "__main__":
    main()
