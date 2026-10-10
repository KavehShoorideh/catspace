"""THE FIRST WIDER LOOP: WATCHING GAMES (Kaveh 2026-10-10: "start the first little wider loop from watching games").

Every census position where Stockfish has an only-move tactic (tactic_reach.py's label: the best move beats the
second by 150 cp and wins material against the position before the opponent's move) is a context; the question is
"did the planner MISS it?" (no immediate win found). The vocabulary describes the SOLUTION (Stockfish's move) and the
planner's handling of it -- facts about the move (check, capture, quiet, which piece, a pawn target, two new attacks, a
discovered attack, a sacrifice), about the position (an overloaded piece of theirs, an endgame) and about the planner's
pipeline (the solution among its seeds, gated by the lone-attack lemma, proved and failed). The miner
(lemma_miner.analyse) finds what characterizes the misses; the holdout is the other half of the GAMES (split by game
identity); the open cases go to the review between Kaveh and Claude.
"""
from __future__ import annotations

import json
import os
import random
import time

import chess

from catspace.io import paths

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def _attacked_values(b: chess.Board, by: bool) -> dict:
    return {s: VAL[p.piece_type] for s, p in b.piece_map().items() if p.color != by and b.attackers(by, s)}


def _row(args):
    fen, sol = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    b = chess.Board(fen); us = b.turn
    m = chess.Move.from_uci(sol)
    san = b.san(m)
    a = b.copy(stack=False); a.push(m)
    trace, dec, proofs, nodes, ms, imm = analyst_core.plan_trace(fen)
    facts = set()
    piece = b.piece_type_at(m.from_square)
    facts.add(f"the solution moves a {chess.piece_name(piece)}")
    if a.is_check(): facts.add("the solution gives check")
    if b.is_capture(m):
        facts.add("the solution captures")
        cap = b.piece_type_at(m.to_square) or chess.PAWN
        if cap == chess.PAWN: facts.add("the solution captures a pawn")
    if not b.is_capture(m) and not a.is_check(): facts.add("the solution is quiet")
    if m.promotion: facts.add("the solution promotes")
    before = _attacked_values(b, us); after = _attacked_values(a, us)
    new = [s for s, v in after.items() if s not in before and v >= 3]
    if len(new) >= 2: facts.add("the solution attacks two new pieces (worth 3+)")
    if any(s for s in new if m.to_square not in a.attackers(us, s)): facts.add("the solution uncovers an attack (a discovery)")
    if a.attackers(not us, m.to_square) and not a.attackers(us, m.to_square) and piece != chess.PAWN:
        facts.add("the solution leaves the moved piece en prise (a sacrifice)")
    if len(b.piece_map()) <= 12: facts.add("an endgame (12 pieces or fewer)")
    stm = "w" if us else "b"
    if any(o[0] != stm for o in analyst_core.overload_check(fen)): facts.add("they have an overloaded piece")
    seeded = any(san in l and ("moves on a path onto it" in l) for l in trace)
    gated = any(l.strip().startswith(f"{san}:") and "lone attack" in l for l in trace)
    proved_fail = any(l.strip().startswith(f"prove {san} ") and "fails" in l for l in trace)
    facts.add("the planner seeded the solution" if seeded else "not: the planner seeded the solution")
    if gated: facts.add("the lone-attack lemma skipped the solution")
    if proved_fail: facts.add("the planner proved the solution and it failed")
    missed = imm is None
    return frozenset(facts), frozenset({"missed"} if missed else set()), fen, sol


def main():
    from multiprocessing import Pool
    from .tactic_census_measure import load
    from .tactic_reach import is_tactic
    from .lemma_miner import analyse
    t0 = time.time()
    census = load("c300")
    prev = {(r["game"], r["ply"]): r["fen"] for r in census}
    labels = {(r["game"], r["ply"]): r for r in map(json.loads, open(paths.experiment("tactic_labels_c300_d12_pv.jsonl")))}
    items = []
    for r in census:
        k = (r["game"], r["ply"])
        lab = labels.get(k); pf = prev.get((r["game"], r["ply"] - 1))
        if lab and pf and lab.get("move") and is_tactic(lab, pf, r["fen"]):
            items.append((r["game"], r["fen"], lab["move"]))
    games = sorted({g for g, _, _ in items})
    rng = random.Random(7); rng.shuffle(games)
    half = set(games[: len(games) // 2])                              # split BY GAME (identity), not by position
    with Pool(10) as p:
        rows = list(p.imap(_row, [(f, s) for _, f, s in items], chunksize=4))
    data = [r for (g, _, _), r in zip(items, rows) if g in half]
    hold = [r for (g, _, _), r in zip(items, rows) if g not in half]
    n_miss = sum(1 for r in rows if "missed" in r[1])
    print(f"TIME facts: {time.time() - t0:.0f} s -- {len(rows)} Stockfish tactics in {len(games)} games; the planner missed {n_miss}; "
          f"mining half {len(data)}, holdout half {len(hold)}")
    from . import lemma_miner
    lemma_miner.MIN_SUPPORT = 8
    from collections import Counter
    tab = Counter()
    for f, t, _, _ in rows:
        for x in f:
            tab[(x, "missed" in t)] += 1
    print("OBS cross-tab (fact: missed / found, miss rate):")
    for x in sorted({x for x, _ in tab}, key=lambda x: -(tab[(x, True)] + tab[(x, False)])):
        a, b = tab[(x, True)], tab[(x, False)]
        print(f"OBS   {x:52s} {a:4d} / {b:4d}  ({100 * a / max(a + b, 1):.0f}%)")
    # THE CRITICAL CASE for the review: the lone-attack lemma's exceptions -- solutions it skipped as "a lone attack on a
    # piece that can step away" that Stockfish says win (Bain & Muggleton: the exception set is the candidate concept)
    import analyst_core
    lone = [(f, s) for (fct, t, f, s) in rows if "the lone-attack lemma skipped the solution" in fct]
    sec = ["", "## THE FIRST REVIEW: the lone-attack lemma's exceptions", "",
           f"The lemma 'a move attacking a piece that can step away wins nothing (unless check, a second target, a new path "
           f"onto the king, or a capture)' skipped the solution in {len(lone)} Stockfish tactics; the planner then missed "
           f"{sum(1 for (fct, t, f, s) in rows if 'the lone-attack lemma skipped the solution' in fct and 'missed' in t)} of them. "
           "What do these lone attacks have that the lemma does not see? That is the next concept.", "",
           "| position (side to move) | solution | Stockfish | why the planner skipped it |", "|---|---|---|---|"]
    for f, s in lone:
        b = chess.Board(f); san = b.san(chess.Move.from_uci(s))
        lab = next((labels[k] for k in labels if prev.get(k) == f), None)
        why = next((l.strip() for l in analyst_core.plan_trace(f)[0] if l.strip().startswith(f"{san}:")), "")
        sec.append(f"| `{f}` | {san} | {(lab or {}).get('best', '?')} cp | {why[len(san) + 1:].strip()[:90]} |")
    analyse(data, hold, "missed", {}, 1.0e4, "review", f"census games (c300): Stockfish only-move tactics, the planner's misses; split by game",
            near_tolerance=0.10, near_support=8)
    rp = paths.experiment("lemma_review_missed.md")
    with open(rp, "a") as fh:
        fh.write("\n".join(sec) + "\n")
    print(f"the first review (the lone-attack exceptions) appended to {rp}")


if __name__ == "__main__":
    main()
