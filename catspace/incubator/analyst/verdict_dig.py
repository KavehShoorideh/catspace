"""WRONG DEEP VERDICTS -- where the zone search's own line goes wrong (Kaveh's pick, 2026-09-30, after the depth lever was
spent: the search completes 10-12 plies and still prefers the wrong move; 02jzH: Kg4 +1 at 10 plies, Stockfish +0.3,
the solution Kf3 0 at 10 plies, Stockfish +4.3).

For each miss of a frame_puzzles dump whose solution S is a candidate: the zone search on S and on the player's top T,
with the run's rules (pawn moves when the dump says so; rungs to the depth the run reached; the run's budget), and
Stockfish's reading of the position WHERE EACH ZONE LINE ENDS (cut at its first pass: the stand-pat). Classes:
- T'S LEAF OVERRATED: the zone value of T is 1.5+ pawns above Stockfish's reading of T's leaf -- a horizon read (02jzH:
  a pawn grabbed on the last ply, the counterplay after it unseen).
- S'S LEAF UNDERRATED: Stockfish reads S's leaf 1.5+ pawns above the zone value of S -- the win is there, past the
  horizon, and the leaf's material read cannot see it.
- BOTH LEAVES RIGHT: the leaves are read right, so the error is inside the line (a move the search chose that is not
  best play: the zone, a pruned reply, the tie-breaks).

    python -m catspace.incubator.analyst.verdict_dig --dump <frame_puzzles dump>
"""
from __future__ import annotations

import argparse
import json
import statistics as st

import chess
import chess.engine

from catspace.io import paths

GAP = 1.5


def _sf(b: chess.Board, mover: chess.Color, key, depth: int) -> float:
    from .sf_depth import _engine
    if b.is_checkmate():
        return -100.0 if b.turn == mover else 100.0
    if b.is_game_over():
        return 0.0
    s = _engine().analyse(b, chess.engine.Limit(depth=depth, time=10.0), game=key)["score"].pov(mover)   # a cleared hash
    return 100.0 if s.is_mate() and s.mate() > 0 else -100.0 if s.is_mate() else s.score() / 100.0


def _leaf(b: chess.Board, line: list) -> tuple:
    """The position where a zone line ends (before its first pass) and the plies to it."""
    bb = b.copy(); n = 0
    for m in line:
        if not m:
            break
        bb.push(m); n += 1
    return bb, n


def _one(args) -> dict | None:
    pid, fen, sol, top, reached, pawn, budget, depth, probe = args
    from dataclasses import replace
    from .fates import clear_caches
    from .frame_planner import MaterialPlanner
    from .zone_search import ZoneSearch, _san
    b = chess.Board(fen); mover = b.turn
    S, T = chess.Move.from_uci(sol), chess.Move.from_uci(top)
    clear_caches()
    rungs = tuple(d for d in (4, 6, 8, 10, 12) if d <= max(4, reached or 8))
    spec = replace(MaterialPlanner().plan(b, [S, T]).spec, depths=rungs, budget=budget, pawn_moves=pawn, margin=1e9)
    zs = ZoneSearch(spec=spec); got = zs.rank(b, [S, T])
    out = dict(pid=pid, fen=fen, rungs=rungs)
    from .sf_depth import _engine
    for name, m in (("S", S), ("T", T)):
        v, ln, d = got[m]
        leaf, n = _leaf(b, ln)
        out[name] = dict(zone=float(v), depth=d, line=_san(b, ln), leaf_plies=n, left=d - n,
                         sf_leaf=_sf(leaf, mover, (pid, name, "leaf"), depth), sf_root=_sf(_after(b, m), mover, (pid, name, "root"), depth))
        end, zone = _zone_at_end(zs, b, ln)                     # Stockfish's move where the line ends: would the search look at it?
        if end is not None and not end.is_game_over():
            info = _engine().analyse(end, chess.engine.Limit(depth=depth, time=10.0), game=(pid, name, "best"))
            pv = info.get("pv", [])
            if pv:
                out[name]["sf_best"] = end.san(pv[0]); out[name]["in_zone"] = pv[0] in zs.moves(end, zone)
                if probe:                                       # depth or blindness: from there, does a deeper zone search find it?
                    stand = zs.stand(end, zone); found = None
                    pz = ZoneSearch(spec=replace(spec, depths=(4, 6, 8, 10, 12), budget=150000))
                    pg = pz.rank(end, [pv[0]])
                    if pg[pv[0]][0] >= 1e5:                         # mates at once: rank() never searches it, its history is
                        found = 1                                   # empty (bug 2026-09-30, the ladder's "mates dropped" again)
                    for dd, vv, _ in ([] if found else pz.history.get(pv[0], [])):
                        if vv is not None and vv - stand >= GAP:
                            found = dd; break
                    out[name]["probe"] = found; out[name]["probe_top"] = max((dd for dd, vv, _ in pz.history.get(pv[0], []) if vv is not None), default=0)
                    bb = end.copy(); promo = False                  # what kind of advantage: a promotion in Stockfish's line, a runner
                    for x in pv[:20]:                               # no defender reaches in time (piece_reach.goal_frame, unopposed)
                        promo |= bool(x.promotion); bb.push(x)
                    from .piece_reach import goal_frame
                    runner = any(goal_frame(end, sq, jobs=False).unopposed for c in (chess.WHITE, chess.BLACK) for sq in end.pieces(chess.PAWN, c))
                    out[name]["promo"] = promo; out[name]["runner"] = runner
                    from .sf_depth import payoff_ply                # deep tactics or position: does Stockfish's line from there win
                    out[name]["sf_pay"] = payoff_ply(end, pv, info.get("score"))   # material (or mate), and at which ply
                out[name]["best_kind"] = ("capture" if end.is_capture(pv[0]) else "check" if end.gives_check(pv[0]) else
                                          "pawn" if end.piece_type_at(pv[0].from_square) == chess.PAWN else "quiet piece")
    return out


def _zone_at_end(zs, b: chess.Board, line: list):
    """The zone the search held at the end of its own line (its root zone, grown by its rule along the line's moves up
    to the first pass) -> (board there, zone)."""
    root = zs.root(b, line[0])
    if root is None:
        return None, None
    _, bb, zone = root
    for m in line[1:]:
        if not m:
            break
        bb.push(m)
        nz = zone | {m.from_square, m.to_square}
        if len(nz) < zs.max_zone:
            nz |= {t for t in chess.SquareSet(bb.attacks_mask(m.to_square)) if bb.piece_at(t) is not None and bb.piece_at(t).color == bb.turn}
        zone = frozenset(nz)
    return bb, zone


def _after(b: chess.Board, m: chess.Move) -> chess.Board:
    bb = b.copy(); bb.push(m)
    return bb


def main(argv=None):
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--dump", required=True); ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--budget", type=int, default=20000); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--probe", action="store_true", help="from where each line ends, a zone search (to 12 plies, 150,000 a depth) on Stockfish's move there")
    a = ap.parse_args(argv)
    pawn = "_pawnmoves" in a.dump
    items = []
    for r in json.load(open(paths.derived(a.dump))):
        if r["solved"]:
            continue
        s = next((x for x in r["steps"] if not x["tied"]), None)
        if s is None or s["v_sol"] is None or not s["best"]:
            continue
        items.append((r["id"], s["fen"], s["sol"], s["best"][0], s.get("depth"), pawn, a.budget, a.depth, a.probe))
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = [r for r in pool.map(_one, items, chunksize=1) if r]
    cls = {}
    for r in rows:
        S, T = r["S"], r["T"]
        clip = lambda x: max(-20.0, min(20.0, x))
        if clip(T["zone"]) - clip(T["sf_leaf"]) >= GAP:
            c = "T's leaf overrated"
        elif clip(S["sf_leaf"]) - clip(S["zone"]) >= GAP:
            c = "S's leaf underrated"
        else:
            c = "both leaves read right"
        r["class"] = c; cls.setdefault(c, []).append(r)
    json.dump(rows, open(paths.derived("verdict_dig.json"), "w"))
    agree = sum(1 for r in rows if r["S"]["zone"] < r["T"]["zone"] or r["S"]["zone"] <= r["T"]["zone"])
    print(f"OBS {a.dump}: {len(rows)} misses with the solution a candidate; replayed on S and T alone (pawn moves {pawn}, budget {a.budget}); "
          f"the replay still puts S at or below T in {agree}")
    for c in ("T's leaf overrated", "S's leaf underrated", "both leaves read right"):
        rs = cls.get(c, [])
        print(f"   {c}: {len(rs)}" + (f" (T: zone minus Stockfish at its leaf, median {st.median([r['T']['zone'] - r['T']['sf_leaf'] for r in rs]):+.1f}; "
                                       f"S: Stockfish at its leaf minus zone, median {st.median([r['S']['sf_leaf'] - r['S']['zone'] for r in rs]):+.1f}; "
                                       f"T's leaf after median {st.median([r['T']['leaf_plies'] for r in rs]):g} plies)" if rs else ""))
        side = "T" if c.startswith("T") else "S" if c.startswith("S") else None
        if side and rs:
            known = [r for r in rs if "in_zone" in r[side]]
            kinds = {}
            for r in known:
                if not r[side]["in_zone"]:
                    kinds[r[side]["best_kind"]] = kinds.get(r[side]["best_kind"], 0) + 1
            print(f"      where {side}'s line ends: Stockfish's best move there is in the search's zone in {sum(1 for r in known if r[side]['in_zone'])}/{len(known)}"
                  f" (outside: " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items(), key=lambda x: -x[1])) + f"); the search's depth left there "
                  f"median {st.median([r[side]['left'] for r in rs]):g} plies")
            pr = [r for r in rs if "probe" in r[side]]
            if pr:
                found = {}
                for r in pr:
                    found[r[side]["probe"]] = found.get(r[side]["probe"], 0) + 1
                print(f"      from there, a zone search on Stockfish's move gains {GAP}+ over standing pat first at depth: "
                      + ", ".join(f"{k}: {found[k]}" for k in sorted(k for k in found if k is not None)) + f"; not by 12: {found.get(None, 0)} "
                      f"(deepest completed median {st.median([r[side]['probe_top'] for r in pr]):g})")
                blind = [r for r in pr if r[side]["probe"] is None]
                pays = [r[side].get("sf_pay") for r in blind]
                print(f"      of the {len(blind)} not found by 12, Stockfish's line from there wins material or mates: by ply 12 in "
                      f"{sum(1 for x in pays if x is not None and x <= 12)}, later in {sum(1 for x in pays if x is not None and x > 12)}, "
                      f"not within its line (positional) in {sum(1 for x in pays if x is None)}")
                print(f"      of the {len(blind)} not found by 12: a promotion in Stockfish's line (20 plies) in {sum(1 for r in blind if r[side]['promo'])}; "
                      f"an unopposed runner on the board in {sum(1 for r in blind if r[side]['runner'])}; neither in "
                      f"{sum(1 for r in blind if not r[side]['promo'] and not r[side]['runner'])}")
        for r in rs[:2]:
            print(f"      {r['pid']}: S {r['S']['line']} (zone {r['S']['zone']:+.1f}, SF leaf {r['S']['sf_leaf']:+.1f}, SF root {r['S']['sf_root']:+.1f}) | "
                  f"T {r['T']['line']} (zone {r['T']['zone']:+.1f}, SF leaf {r['T']['sf_leaf']:+.1f}, SF root {r['T']['sf_root']:+.1f})")
    from .sf_depth import _ENG
    for e in _ENG.values():
        e.quit()


if __name__ == "__main__":
    main()
