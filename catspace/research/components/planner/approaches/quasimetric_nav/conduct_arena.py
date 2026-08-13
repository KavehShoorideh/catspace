#!/usr/bin/env python
"""conduct_arena.py -- ASYMMETRIC conduct arena (the 2026-08-12 process lesson: the battery
has sharp answers and symmetric-flag arenas hide conduct regressions -- Kaveh's games caught
knight-shuffling, bongclouding and pony-retreats that 60 symmetric games could not).

Plays engine-with-CHANGE vs engine-without-CHANGE from paired openings, SAME checkpoint.
Any behavioral flag/attribute is settable per side via --a/--b (comma k=v lists). Reports
pentanomial pair stats + per-game conduct counters that catch the known pathologies:
    retreats-to-home     minor piece returning to its start square in the first 12 moves
    early-king-walks     non-castling king moves before move 10
    early-edge-lunges    a/h-file pawn pushes or Nh3/Na3-class moves in the first 8 moves

    .venv/bin/python -m ...conduct_arena --ckpt <ckpt> --a forcing_pref=True --b forcing_pref=False
"""
from __future__ import annotations

import argparse
import random

import chess


HOME = {chess.B1, chess.G1, chess.C1, chess.F1, chess.B8, chess.G8, chess.C8, chess.F8}


def conduct_counters(moves, start):
    """the pathology counters, from a movetext replay (start = the opening position the
    game actually began from -- replaying on a fresh board was the first crash)."""
    b = start.copy()
    retreats = kingwalks = lunges = 0
    for i, mv in enumerate(moves):
        pc = b.piece_type_at(mv.from_square)
        if i < 24 and pc in (chess.KNIGHT, chess.BISHOP) and mv.to_square in HOME \
                and b.color_at(mv.from_square) == (mv.to_square in (chess.B1, chess.G1,
                                                                    chess.C1, chess.F1)):
            retreats += 1
        if i < 20 and pc == chess.KING and not b.is_castling(mv):
            kingwalks += 1
        if i < 16:
            f = chess.square_file(mv.to_square)
            if (pc == chess.PAWN and f in (0, 7)) or \
               (pc == chess.KNIGHT and f in (0, 7)):
                lunges += 1
        b.push(mv)
    return retreats, kingwalks, lunges


def set_flags(eng, spec):
    for kv in (spec or "").split(","):
        if not kv.strip():
            continue
        k, v = kv.split("=")
        vv = {"True": True, "False": False}.get(v, None)
        setattr(eng, k.strip(), vv if vv is not None else float(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--a", default="", help="side A flags, e.g. forcing_pref=True")
    ap.add_argument("--b", default="", help="side B flags")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess
    ea = KittyChess(args.ckpt, args.device)
    eb = KittyChess(args.ckpt, args.device)
    ea.concept_eval = eb.concept_eval = False
    set_flags(ea, args.a)
    set_flags(eb, args.b)
    if eb.tb is not None:
        eb.tb = ea.tb                                     # shared TB connection (the sqlite scar)

    rng = random.Random(0)
    def opening():
        b = chess.Board()
        for _ in range(4):
            b.push(rng.choice(list(b.legal_moves)))
        return b

    def play(white, black, b0):
        b = b0.copy()
        while not b.is_game_over(claim_draw=True) and b.ply() < args.max_plies:
            eng = white if b.turn else black
            rows = eng.search_coherent(b, budget=args.budget)
            rows = eng.rank_by_child_E(b, rows)
            if not rows:
                mv = eng._tb_move(b) if eng.tb else None
                if mv is None:
                    break
                b.push(mv)
                continue
            b.push(rows[0]["mv"])
        o = b.outcome(claim_draw=True)
        r = 0.5 if o is None or o.winner is None else (1.0 if o.winner else 0.0)
        return r, list(b.move_stack)[len(b0.move_stack):]

    pairs, condA, condB = [], [0, 0, 0], [0, 0, 0]
    for rd in range(args.rounds):
        op = opening()
        rA, mvA = play(ea, eb, op)          # A white
        rB, mvB = play(eb, ea, op)          # B white
        pairs.append(rA + (1.0 - rB))
        for i, v in enumerate(conduct_counters(mvA, op)):
            condA[i] += v
        for i, v in enumerate(conduct_counters(mvB, op)):
            condB[i] += v
        print(f"[conduct] round {rd+1}/{args.rounds}: A {sum(pairs):.1f}/{2*len(pairs)}",
              flush=True)
    n = 2 * len(pairs)
    names = ("retreats-home", "king-walks", "edge-lunges")
    print(f"\n[conduct] SCORE A {sum(pairs):.1f}/{n} ({sum(pairs)/n:.0%})  "
          f"A=({args.a or 'default'}) vs B=({args.b or 'default'})")
    print(f"[conduct] counters (both sides' games pooled per config):")
    for i, nm in enumerate(names):
        print(f"  {nm:14s} A {condA[i]:3d}   B {condB[i]:3d}")
    print("[conduct] VERDICT: flag a change if its side scores <45% OR any counter is 2x+")


if __name__ == "__main__":
    main()
