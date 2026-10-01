"""WHY THE SOLUTION IS VALUED BELOW THE TOP -- the anatomy of a miss whose solution IS a candidate (2026-09-30: after
the proposer and the depth decider, 51 of the 57 misses left on the offset-500 replay are of this kind).

For the failing step of every missed puzzle in a frame_puzzles dump: the player's top move T and the solution S, each
read by Stockfish (the score after the move, from the mover's side, and its principal variation), and each line replayed
through the zone search's own rules (its root zone with the planner's hints; a move outside the zone is a move the
search never looks at). The classes, in order:
- ALSO WINS: Stockfish rates T within a pawn of S (or both mate) -- the puzzle wants one answer; not an engine error.
- T OVERRATED, ITS REFUTATION OUTSIDE ITS ZONE: Stockfish's reply line to T leaves T's zone within the search's reach.
- S UNDERRATED, ITS LINE OUTSIDE ITS ZONE: Stockfish's line after S leaves S's zone.
- INSIDE BOTH ZONES: the lines are searchable; the difference is depth, budget or the leaf's material read.

    python -m catspace.incubator.analyst.miss_anatomy --dump <frame_puzzles dump> [--reach 12]
"""
from __future__ import annotations

import argparse
import json
import statistics as st

import chess
import chess.engine

from catspace.io import paths


def _zone_exit(b: chess.Board, line: list, reach: int, pawn_moves: bool = False):
    """The first ply of `line` (line[0] is the root move) outside the zone search's zone, or None when the first
    `reach` plies all stay in."""
    from .fates import clear_caches
    from .frame_planner import MaterialPlanner
    from .frame_player import candidates
    from .zone_search import ZoneSearch
    clear_caches()
    from dataclasses import replace
    zs = ZoneSearch(spec=replace(MaterialPlanner().plan(b, candidates(b)).spec, pawn_moves=pawn_moves))   # the RUN's zone rules
    root = zs.root(b, line[0])
    if root is None:
        return None
    _, bb, zone = root
    for i, m in enumerate(line[1:reach], start=1):
        if m not in zs.moves(bb, zone):
            return i
        bb.push(m)
        nz = zone | {m.from_square, m.to_square}                     # the search's own growth rule (zone_search.search)
        if len(nz) < zs.max_zone:
            nz |= {t for t in chess.SquareSet(bb.attacks_mask(m.to_square)) if bb.piece_at(t) is not None and bb.piece_at(t).color == bb.turn}
        zone = frozenset(nz)
    return None


def _score(info, mover) -> float:
    s = info["score"].pov(mover)
    return 100.0 if s.is_mate() and s.mate() > 0 else -100.0 if s.is_mate() else s.score() / 100.0


def _one(args) -> dict | None:
    pid, fen, sol, top, depth, reached = args
    from .sf_depth import _engine
    b = chess.Board(fen); mover = b.turn
    out = dict(pid=pid, fen=fen, sol=sol, top=top, reached=reached)
    for name, u in (("S", sol), ("T", top)):
        m = chess.Move.from_uci(u); bb = b.copy(); bb.push(m)
        if bb.is_game_over():
            out[name] = (100.0 if bb.is_checkmate() else 0.0, [u]); continue
        info = _engine().analyse(bb, chess.engine.Limit(depth=depth, time=10.0), game=(pid, name))   # a new game: the hash
                                                          # cleared (bug 2026-09-30: one engine's hash carried across positions, so
                                                          # two runs split T/S 24/6 and 17/11)
        out[name] = (_score(info, mover), [u] + [x.uci() for x in info.get("pv", [])])
    return out


def main(argv=None):
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--dump", required=True); ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--reach", type=int, default=12, help="plies of each line checked against the zone (the search's deepest rung)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--pawn-moves", action="store_true", help="the run searched every pawn move (on by itself for a dump tagged _pawnmoves)")
    a = ap.parse_args(argv)
    pawn = a.pawn_moves or "_pawnmoves" in a.dump                    # (bug 2026-09-30: the zone check used its own default spec)
    items = []
    for r in json.load(open(paths.derived(a.dump))):
        if r["solved"]:
            continue
        s = next((x for x in r["steps"] if not x["tied"]), None)
        if s is None or s["v_sol"] is None or not s["best"]:
            continue                                                 # never a candidate: the candidates' gap, not this one
        items.append((r["id"], s["fen"], s["sol"], s["best"][0], a.depth, s.get("depth")))
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = [r for r in pool.map(_one, items, chunksize=1) if r]
    classes = {}
    for r in rows:
        b = chess.Board(r["fen"]); (vs, ls), (vt, lt) = r["S"], r["T"]
        line_s = [chess.Move.from_uci(u) for u in ls]; line_t = [chess.Move.from_uci(u) for u in lt]
        r["exit_S"], r["exit_T"] = _zone_exit(b, line_s, a.reach, pawn), _zone_exit(b, line_t, a.reach, pawn)
        if vt >= vs - 1.0 or (vs >= 100 and vt >= 100):
            c = "also wins"
        elif r["exit_T"] is not None:
            c = "T overrated: refutation outside its zone"
        elif r["exit_S"] is not None:
            c = "S underrated: its line outside its zone"
        else:
            c = "inside both zones"
        r["class"] = c; classes.setdefault(c, []).append(r)
        r["pay_S"] = _payoff(b, line_s, vs); r["pay_T"] = _payoff(b, line_t, -vt, against=True)
    json.dump(rows, open(paths.derived("miss_anatomy.json"), "w"))
    print(f"OBS {a.dump}: {len(rows)} misses with the solution a candidate valued below the top; Stockfish depth {a.depth}, zone reach {a.reach} plies"
          f"{', pawn moves in the zone' if pawn else ''}")
    for c in ("also wins", "T overrated: refutation outside its zone", "S underrated: its line outside its zone", "inside both zones"):
        rs = classes.get(c, [])
        gap = [r["S"][0] - r["T"][0] for r in rs if abs(r["S"][0]) < 100 and abs(r["T"][0]) < 100]
        print(f"   {c}: {len(rs)}" + (f" (Stockfish's S minus T, median {st.median(gap):+.1f} pawns)" if gap else ""))
        if c.startswith("T overrated") and rs:                       # how the refutations leave: the ply, whose move, which piece
            from .fates import clear_caches, passed, joint_threats, square_frame
            exits = {}; side = {"opponent": 0, "ours": 0}; pawn_opp = qt = 0
            for r in rs:
                bb = chess.Board(r["fen"]); line = [chess.Move.from_uci(u) for u in r["T"][1]]; k = r["exit_T"]
                for m in line[:k]:
                    bb.push(m)
                m = line[k]; who = "opponent" if k % 2 == 1 else "ours"; side[who] += 1
                exits[k] = exits.get(k, 0) + 1
                pawn_opp += who == "opponent" and bb.piece_type_at(m.from_square) == chess.PAWN
                clear_caches(); now = joint_threats(bb, square_frame)[0]
                nb = bb.copy(stack=False); nb.push(m); pb = passed(nb)
                qt += pb is not None and joint_threats(pb, square_frame)[0] >= now + 0.5
            print(f"      the refutation leaves the zone at ply " + ", ".join(f"{k}: {n}" for k, n in sorted(exits.items()))
                  + f"; by the opponent's move {side['opponent']} (the side whose missing moves overrate T; a pawn move in {pawn_opp}), "
                  f"by ours {side['ours']}; the leaving move a quiet threat (null-move what-if) in {qt}")
        if c == "inside both zones" and rs:                         # depth or the read: where the lines pay off against the depth the
            reach = lambda r: max(x for x in (r["pay_S"], r["pay_T"]) if x is not None) if (r["pay_S"] or r["pay_T"]) else None   # run reached
            known = [r for r in rs if reach(r) is not None and r["reached"]]
            beyond = [r for r in known if reach(r) > r["reached"]]
            print(f"      where they pay off (Stockfish's lines: the solution's gain, the top's refutation) against the deepest rung the run "
                  f"completed: beyond it in {len(beyond)}, within it in {len(known) - len(beyond)} (unknown {len(rs) - len(known)}); "
                  f"payoff median {st.median([reach(r) for r in known]) if known else '-'} plies, reached median "
                  f"{st.median([r['reached'] for r in known]) if known else '-'}")
            r_in = [r for r in known if reach(r) <= r["reached"]]
            for r in r_in[:2]:
                b = chess.Board(r["fen"]); sx = lambda line: " ".join(_sans(b, line[:10]))
                print(f"      within reach, {r['pid']} (reached {r['reached']}, pays S {r['pay_S']} T {r['pay_T']}): S {sx(r['S'][1])} ({r['S'][0]:+.1f}) | "
                      f"T {sx(r['T'][1])} ({r['T'][0]:+.1f})")
        for r in rs[:2]:
            b = chess.Board(r["fen"])
            sx = lambda line: " ".join(_sans(b, line[:8]))
            print(f"      {r['pid']}: S {sx(r['S'][1])} ({r['S'][0]:+.1f}) | T {sx(r['T'][1])} ({r['T'][0]:+.1f}) | exits S {r['exit_S']} T {r['exit_T']}")
    from .sf_depth import _ENG
    for e in _ENG.values():
        e.quit()


def _payoff(b: chess.Board, line: list, score: float, against: bool = False):
    """The ply of a Stockfish line where its gain is secured (sf_depth.payoff_ply): for the solution, the mover's gain;
    for the top move (against=True), the opponent's -- the refutation's payoff, counted from the root. A mate: its line."""
    from .sf_depth import payoff_ply
    if abs(score) >= 100:
        return len(line) if score > 0 else None
    if not against:
        return payoff_ply(b, line, None)
    bb = b.copy(); bb.push(line[0])
    p = payoff_ply(bb, line[1:], None)
    return None if p is None else p + 1


def _sans(b: chess.Board, ucis: list) -> list:
    bb = b.copy(); out = []
    for u in ucis:
        m = chess.Move.from_uci(u)
        if m not in bb.legal_moves:
            break
        out.append(bb.san(m)); bb.push(m)
    return out


if __name__ == "__main__":
    main()
