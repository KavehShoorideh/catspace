"""PLAN 79.27 PART C AND PLAN 79.28 PART C: THE MAKER OF LAYER 4'S GOALS FIXTURE (SYSTEM_SPEC 11 "Layer 4 takes a goal as an input", "Reached means the goal outlasts the reply", "the layers"; ledger rows 36, 14; piece: tactical reasoner).
A PRINTING SCRIPT (it may name the games' units because it only writes a data file). It writes rust/engine/tactics/tests/fixtures/goals.fixture: for two games (the standard game and a variant with a second way to end it), positions met in seeded random games, each with a LIST of
goals that are neither a gain nor the game's end: patterns over the facts of layer 1's graph (relations unit, bear, may_move, on_line, on_edge; layer 1's `graph_facts`), written with placeholders (a word starting with ?) and the constants of the position:
  bears      a unit of the side to move bears on a named unit of the other side (the named unit by its kind and place)
  two bear   two DIFFERENT units of the side bear on one unit of the other (plan 79.28: written with layer 1's `differ` literal, any two units; the unit is the one of the other side that the most units of the side bear on, ties by the seed)
  on a line  a unit of the side stands on a line through the named unit's place (a line is named as the graph names it)
  on edge    a unit of the kind of the side's unit that is nearest the far edge (not a pawn, not a king; ties by the seed) stands on the far edge (where it may change kind)
  may take   a unit of the side may move to the named unit's place while a named unit of the other side does not bear on it (a negated literal)
No verdict is written: the reference in the test decides what is reached. Seeded; the same file every time.

  .venv/bin/python -m catspace.incubator.analyst.discovery.layer4_goals_print OUT_FILE
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import chess
import chess.variant

GAMES = [("chess", chess.Board, "learned/chess/graph_vocabulary.json"), ("kingofthehill", chess.variant.KingOfTheHillBoard, "learned/chess/graph_vocabulary.json")]
KINDS = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
ROLE = {True: "white", False: "black"}
PER_GAME = 40


def lit(rel: str, *args: str, neg: bool = False) -> dict:
    return {"relation": rel, "args": list(args), "neg": neg}


def goals_of(b: "chess.Board", rng: random.Random) -> list:
    me, them = b.turn, not b.turn
    mine = [(sq, p) for sq, p in b.piece_map().items() if p.color == me]
    theirs = [(sq, p) for sq, p in b.piece_map().items() if p.color == them and p.piece_type != chess.KING]
    if not mine or not theirs:
        return []
    out = []
    tsq, tp = rng.choice(theirs)
    t, tk = chess.square_name(tsq), KINDS[tp.piece_type]
    out.append(("bears", [lit("unit", ROLE[them], tk, t), lit("bear", ROLE[me], "?k", "?p", t)]))
    kinds = sorted({KINDS[p.piece_type] for _, p in mine})
    most = max(len(b.attackers(me, sq)) for sq, _ in theirs)
    wsq, wp = rng.choice([(sq, p) for sq, p in theirs if len(b.attackers(me, sq)) == most])
    w, wk = chess.square_name(wsq), KINDS[wp.piece_type]
    out.append(("two_bear", [lit("unit", ROLE[them], wk, w), lit("bear", ROLE[me], "?k1", "?p1", w), lit("bear", ROLE[me], "?k2", "?p2", w), lit("differ", "?p1", "?p2")]))
    f, r = t[0], t[1]
    line = rng.choice([f"line:north@place:{f}1", f"line:east@place:a{r}"])
    out.append(("on_line", [lit("unit", ROLE[me], "?k", "?p"), lit("on_line", "?p", line)]))
    runners = [(sq, p) for sq, p in mine if p.piece_type not in (chess.PAWN, chess.KING)] or mine
    gap = lambda sq: (7 - chess.square_rank(sq)) if me else chess.square_rank(sq)
    nearest = min(gap(sq) for sq, _ in runners)
    k = KINDS[rng.choice([p for sq, p in runners if gap(sq) == nearest]).piece_type]
    out.append(("on_edge", [lit("unit", ROLE[me], k, "?p"), lit("on_edge", "?p", "north" if me else "south")]))
    osq, op = rng.choice(theirs)
    out.append(("may_take", [lit("unit", ROLE[me], "?k", "?p"), lit("may_move", ROLE[me], "?k", "?p", t), lit("bear", ROLE[them], KINDS[op.piece_type], chess.square_name(osq), t, neg=True)]))
    rng.shuffle(out)
    return out


def positions(cls, rng: random.Random, n: int) -> list:
    got = []
    while len(got) < n:
        b = cls()
        want = rng.choice([8, 12, 16, 20, 26, 32, 40, 50])
        for _ in range(want):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over():
                break
            caps = [m for m in moves if b.is_capture(m)]
            b.push(rng.choice(caps) if caps and rng.random() < 0.5 else rng.choice(moves))
        if b.is_game_over() or not list(b.legal_moves):
            continue
        gs = goals_of(b, rng)
        if len(gs) == 5:
            got.append((b.fen(), gs))
    return got


def main(argv: "list | None" = None) -> int:
    out = Path((argv or sys.argv[1:])[0])
    lines = ["note\tplan 79.28 part C (first made in plan 79.27 part C): positions of two games with lists of goals (patterns over layer 1's graph facts); no verdict is written, the reference in the test decides what is reached"]
    total = 0
    for gi, (name, cls, decl) in enumerate(GAMES):
        rng = random.Random(7927 + gi)
        lines.append(f"game\t{name}\t{decl}")
        for i, (fen, gs) in enumerate(positions(cls, rng, PER_GAME)):
            lines.append(f"pos\t{name}-{i}\t{fen}\t" + json.dumps([{"kind": kind, "pattern": pat} for kind, pat in gs], separators=(",", ":")))
            total += len(gs)
    out.write_text("\n".join(lines) + "\n")
    print(f"OBS 79.28 goals fixture: {len(GAMES) * PER_GAME} positions of {len(GAMES)} games, {total} goals, written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
