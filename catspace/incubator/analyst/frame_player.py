"""THE FRAME PLAYER (Kaveh 2026-09-27: "you look at the board, you see a frame of pieces ... you pick the right frames,
you analyze the frames ... these frames may interact with the rest of the board ... you resolve each frame and out of
the resolution ... there's one stable output ... try some positions and see if you make the best move ... if not,
let's dig in"). A move is chosen from the frames alone and its reasoning is kept:

  1. the frames: around every contested square, the pieces that attack or defend it, and those that would once the
     frame's pieces are lifted off (fates.square_frame);
  2. each frame resolved exactly: the capture sequence on its square, each side capturing or passing (alpha-beta);
  3. the stable output: a move's value is what it wins, minus the opponent's best frame after it, plus their best
     frame if we had passed (the ledger); the best value is chosen (ties: every tied move is listed).

Scored on held-out positions against Stockfish (its evaluation of every move, cached): does the choice lose 1.5 pawns
or more against the best move? The misses are printed with their trace, for the dig.

    python -m catspace.incubator.analyst.frame_player --n 200
"""
from __future__ import annotations

import argparse
import json
import time

import chess

from catspace.io import paths

from .fates import captured_value, changed_squares, joint_threats, joint_threats_book, ledger_move_value, passed, square_frame, threats
from .gauge_verdicts import LOSS_CP, positions, truths
from .engine import Engine


JOINT = True                                                        # frames resolved jointly with the pieces they pull in (2026-09-28)
SELECT = None                                                       # the prior's choice of pulled-in pieces (None: all of them)
PLANS = "none"                                                      # the tactical plan layer that breaks ties in the top set (plan_layers)
_LAYER = None


def _layer():
    global _LAYER
    if _LAYER is None or _LAYER.name != PLANS:
        from .plan_layers import get
        _LAYER = get(PLANS)
    return _LAYER


def choose(b: chess.Board) -> dict:
    """-> {'values': {uci: value}, 'best': [ucis with the top value], 'trace': {uci: (their best after, their best if
    we pass)}} -- the frames' own reasoning for every legal move."""
    stand = passed(b)
    if JOINT:
        before, book = joint_threats_book(stand, square_frame, select=SELECT) if stand is not None else ((0.0, None, []), None)
    else:
        before = threats(stand, square_frame) if stand is not None else (0.0, None, [])
    values = {}; trace = {}
    for m in b.legal_moves:
        if JOINT:
            nb = b.copy(stack=False); nb.push(m)
            after = (0.0, None, []) if nb.is_checkmate() else joint_threats(nb, square_frame, select=SELECT, parent=book, changed=changed_squares(b, m))   # the pass position's book: only the frames the move touches
            v = 1000.0 if nb.is_checkmate() else captured_value(b, m) - after[0] + before[0]
            bef = before
        else:
            v, after, bef = ledger_move_value(b, m, square_frame, before=before)
        values[m.uci()] = v; trace[m.uci()] = (after, bef)
    top = max(values.values())
    best = [u for u, v in values.items() if v == top]
    plan_scores = {}
    if PLANS != "none" and len(best) > 1:                           # the plan layer ranks the moves the frames cannot tell apart
        plan_scores = _layer().score_moves(b, [chess.Move.from_uci(u) for u in best])
        hi = max(plan_scores.values())
        if hi > 0:
            best = [u for u in best if plan_scores.get(u, 0.0) == hi]
    return dict(values=values, best=best, trace=trace, before=before, plans=plan_scores)


def _choose_fen(fen: str) -> dict:
    return choose(chess.Board(fen))


def explain(b: chess.Board, u: str, c: dict) -> str:
    m = chess.Move.from_uci(u); after, bef = c["trace"][u]
    nm = lambda t: (f"{t[0]:+.0f} on {chess.square_name(t[1])} by {' '.join(t[2])}" if t[1] is not None else "nothing")
    return f"{b.san(m)} {c['values'][u]:+.1f}: their best frame after it {nm(after)}; if we passed {nm(bef)}"


def main(argv=None):
    global JOINT, PLANS
    import sys
    sys.stdout.reconfigure(line_buffering=True)                       # every OBS line reaches the log as it prints (a crash loses none)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--per-frame", action="store_true", help="resolve each frame alone (no pulled-in pieces), for comparison")
    ap.add_argument("--show", type=int, default=3); ap.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 2) - 1))
    ap.add_argument("--plans", default=PLANS, help="the tactical plan layer that breaks ties in the top set (plan_layers.LAYERS)")
    a = ap.parse_args(argv)
    JOINT = not a.per_frame
    PLANS = a.plans
    t0 = time.perf_counter()
    eng = Engine(threads=8, depth=a.depth)
    try:
        _run(a, eng, t0)
    finally:
        eng.close()                                                   # an open engine keeps the interpreter from exiting after a crash


def _run(a, eng, t0):
    rows = []
    boards = positions(a.n, a.seed)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers) as pool:                  # the frames' choice per position over a pool; Stockfish (cached) in the parent
        chosen = pool.map(_choose_fen, [b.fen() for b in boards], chunksize=2)
    for i, (b, c) in enumerate(zip(boards, chosen)):
        lines = eng.analyse(b, multipv=b.legal_moves.count(), depth=a.depth)
        if not lines:
            continue
        cp = {l.move: l.cp for l in lines}; best_cp = max(cp.values()); sf_best = max(cp, key=cp.get)
        pick = c["best"][0]                                         # the first tied move in move order: the frames cannot tell tied moves apart
        tied_ok = sum(1 for u in c["best"] if cp.get(u, -1e9) > best_cp - LOSS_CP)
        false_safe = sum(1 for u in c["best"] if cp.get(u, -1e9) <= best_cp - LOSS_CP)   # a top move Stockfish calls a loss
        rows.append(dict(i=i, fen=b.fen(), pick=pick, loss=best_cp - cp.get(pick, -10000), sf_best=sf_best, sf_in_top=sf_best in c["best"],
                         n_top=len(c["best"]), tied_ok=tied_ok, false_safe=false_safe, c=c, cp=cp, b=b))
    n = len(rows)
    mistakes = [r for r in rows if r["loss"] >= LOSS_CP]
    print(f"[frame_player] {n} held-out positions, plan layer {PLANS} ({time.perf_counter() - t0:.0f}s)")
    print(f"OBS Stockfish's best move among the frames' top moves: {sum(r['sf_in_top'] for r in rows)}/{n}; the frames' top set is a single move in "
          f"{sum(r['n_top'] == 1 for r in rows)}/{n}")
    print(f"OBS the pick (the first top move) loses 1.5 pawns or more against the best move: {len(mistakes)}/{n}; "
          f"of those, every top move does: {sum(1 for r in mistakes if r['tied_ok'] == 0)}")
    print(f"OBS false-safe top moves (a top move losing 1.5 pawns or more): {sum(r['false_safe'] for r in rows)} in {sum(r['false_safe'] > 0 for r in rows)}/{n} positions; "
          f"own misses (every top move does): {sum(r['false_safe'] == r['n_top'] for r in rows)}")
    print(f"OBS the pick is Stockfish's best move: {sum(r['pick'] == r['sf_best'] for r in rows)}/{n}; the pick's mean loss against the best move "
          f"{sum(min(r['loss'], 1000) for r in rows) / max(1, n):.0f} cp (capped at 1000); positions where the plan layer credited a tied move: "
          f"{sum(1 for r in rows if r['c'].get('plans') and max(r['c']['plans'].values()) > 0)}")
    better = worse = same = 0; gaps = []                              # the plan layer's credit, checked against Stockfish on the tied moves
    for r in rows:
        ps = r["c"].get("plans") or {}
        if not ps or max(ps.values()) <= 0:
            continue
        hi = max(ps.values()); cred = [u for u in ps if ps[u] == hi]; rest = [u for u in ps if ps[u] != hi]
        if not rest:
            continue
        c_cp = sum(r["cp"].get(u, -10000) for u in cred) / len(cred); o_cp = sum(r["cp"].get(u, -10000) for u in rest) / len(rest)
        gaps.append(c_cp - o_cp); better += c_cp > o_cp + 20; worse += c_cp < o_cp - 20; same += abs(c_cp - o_cp) <= 20
    wrong = []
    for r in rows:
        ps = r["c"].get("plans") or {}
        if ps and max(ps.values()) > 0:
            hi = max(ps.values()); cred = [u for u in ps if ps[u] == hi]; rest = [u for u in ps if ps[u] != hi]
            if rest and max(r["cp"].get(u, -10000) for u in cred) < max(r["cp"].get(u, -10000) for u in rest) - 20:
                wrong.append((r, cred, rest))
    for r, cred, rest in wrong[:a.show]:                              # the plan layer's wrong credits, for the dig
        b = r["b"]; nm = lambda us: ", ".join(f"{b.san(chess.Move.from_uci(u))} ({r['cp'].get(u, 0):+d})" for u in us[:6])
        print(f"\nWRONG CREDIT {b.fen()}  credited {nm(cred)} ({r['c']['plans'][cred[0]]:+g}); the best other tied moves {nm(sorted(rest, key=lambda u: -r['cp'].get(u, -10000)))}")
        for line in _layer().explain(b)[:8]:
            print("   ", line)
    if PLANS != "none":
        print(f"OBS the plan layer credited some tied moves over others in {len(gaps)}/{n} positions; Stockfish rates the credited moves above the "
              f"other tied moves by more than 0.2 in {better}, below in {worse}, within 0.2 in {same}; mean gap {sum(gaps) / max(1, len(gaps)):+.0f} cp")
    worst = sorted(mistakes, key=lambda r: -r["loss"])
    json.dump([dict(fen=r["fen"], pick=r["pick"], loss=r["loss"], sf_best=r["sf_best"], n_top=r["n_top"], tied_ok=r["tied_ok"], plans=r["c"].get("plans") or {},
                    cp={u: r["cp"].get(u) for u in (r["c"].get("plans") or {})}) for r in rows],
              open(paths.derived(f"frame_player_positions_{PLANS}.json"), "w"), indent=1)
    shown = 0
    for r in worst:
        if r["tied_ok"] or shown >= a.show:
            continue                                                # a miss the frames own: no tied move was fine
        shown += 1; b = r["b"]; c = r["c"]
        sf_line = eng.analyse(b, root_moves=[chess.Move.from_uci(r["sf_best"])], depth=a.depth)[0]
        my_line = eng.analyse(b, root_moves=[chess.Move.from_uci(r["pick"])], depth=a.depth)[0]
        print(f"\nMISS {b.fen()}  {'white' if b.turn else 'black'} to move")
        print(b.unicode(borders=False, empty_square='·'))
        print(f"  the frames chose {explain(b, r['pick'], c)}   (Stockfish {r['cp'][r['pick']]:+d})")
        print(f"  Stockfish's best {explain(b, r['sf_best'], c)}   (Stockfish {r['cp'][r['sf_best']]:+d})")
        print(f"  Stockfish's line after the frames' move: {my_line.san_line(b, 8)}")
        print(f"  Stockfish's line after its best move:   {sf_line.san_line(b, 8)}")


if __name__ == "__main__":
    main()
