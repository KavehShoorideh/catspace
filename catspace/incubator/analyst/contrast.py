"""contrast.py -- explain a move BY CONTRAST (Kaveh 2026-09-08): "a knight hop that is a good move in another position
is a bad move here because it ALLOWS the opponent a tactic. That a tactic exists is not interesting; that this move
allowed it is."

Two moves from the same position are both played, both resulting positions go through the same analysis, and the
DIFFERENCE is the explanation: the opponent's threats that exist after move A and not after move B (A allows
them), the ones that exist after B and not after A (A prevents them), our own threats kept or given up, the
material each hands over one reply deep, the holdings each breaks, the structure each changes. The comparison
partner is the best move when the move played was not it, and otherwise the most common human move at the
player's level (Maia 1800 by default) when that differs -- a contrast needs a real alternative.

    python -m catspace.incubator.analyst.contrast --fen "<fen>" --move Nf5 [--against Nd2] [--engine] [--human 1800]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import chess

from .memo import Memo
from .facts import value
from .threats import free_move_threats, mate_in_one
from .subgames import holdings, cost_of_move, holding_text
from .rank import sure_gain
from .structure import structure_diff, structure_changes

HUMAN_LEVEL = 1800


@dataclass
class Side:
    """One move's consequences, in the ledger's currency."""
    san: str
    base: float                       # our material net one reply deep (negative: it hangs something)
    base_why: str
    their_threats: dict               # their move SAN -> (gain, why) with a free move after ours
    our_threats: dict                 # our move SAN -> (gain, why) with a free move after theirs
    broken: list                      # holdings the move breaks, in words
    structure: list                   # structure changes, in words
    mate_threat: str | None = None    # their mate in one after our move, if any
    cp: int | None = None             # the engine's number after the move (side to move = them; shown for us)


@dataclass
class Contrast:
    a: Side
    b: Side
    allows: list[str] = field(default_factory=list)      # their threats after A that B does not allow
    prevents: list[str] = field(default_factory=list)    # their threats after B that A prevents
    gives_up: list[str] = field(default_factory=list)    # our threats after B that A loses
    creates: list[str] = field(default_factory=list)     # our threats after A that B does not create
    lines: list[str] = field(default_factory=list)


def side_of(board: chess.Board, m: chess.Move, memo, engine=None, depth: int = 10) -> Side:
    us = board.turn
    after = board.copy(stack=False); after.push(m)
    v, why = sure_gain(board, m, memo, signed=True)
    theirs = {}
    if not after.is_game_over():
        for mv, g, w in free_move_threats(after, not us, memo, top=8):
            theirs[after.san(mv)] = (g, w)
    ours = {}
    if not after.is_game_over():
        b2 = after.copy(stack=False); b2.turn = us
        if b2.is_valid():
            for mv, g, w in free_move_threats(after, us, memo, top=8):
                ours[b2.san(mv)] = (g, w)
    held = holdings(board, memo)
    broken = [f"the {chess.piece_name(board.piece_at(h.piece).piece_type)} on {chess.square_name(h.piece)} {holding_text(board, h)}" for h in cost_of_move(board, m, held, memo)]
    try:
        created, ended = structure_diff(board, [m])
        struct = structure_changes(created, ended)[:3]
    except Exception:
        struct = []
    mt = None
    if not after.is_game_over():
        mates = mate_in_one(after, not us)
        if mates:
            mt = after.san(mates[0])
    cp = None
    if engine is not None and not after.is_game_over():
        try:
            l = engine.best(after, depth=depth)
            cp = -l.cp if l else None                      # from our side
        except Exception:
            cp = None
    return Side(board.san(m), v, why, theirs, ours, broken, struct, mt, cp)


def _fmt(san: str, g: float) -> str:
    return f"{san} ({'mate' if g >= 1000 else f'+{g:g}'})"


def contrast(board: chess.Board, a: chess.Move, b: chess.Move, memo=None, engine=None) -> Contrast:
    memo = memo if memo is not None else Memo()
    A, B = side_of(board, a, memo, engine), side_of(board, b, memo, engine)
    c = Contrast(A, B)
    BIG = 2.0                                                   # a threat worth naming in a contrast: two pawns or more, or a mate
    def diff(x: dict, y: dict, skip=()):
        out = []
        for san, (g, w) in sorted(x.items(), key=lambda kv: -kv[1][0]):
            if san in skip:
                continue
            gy = y.get(san, (0.0, ""))[0]
            if g >= BIG and g > gy + 0.5:
                out.append(_fmt(san, g))
        return out[:3]
    c.allows = diff(A.their_threats, B.their_threats)
    c.prevents = diff(B.their_threats, A.their_threats)
    c.gives_up = diff(B.our_threats, A.our_threats, skip=(A.san, B.san))   # a threat the partner executed is not 'given up'
    c.creates = diff(A.our_threats, B.our_threats, skip=(A.san, B.san))
    c.lines = narrate(c)
    return c


def narrate(c: Contrast) -> list[str]:
    A, B = c.a, c.b
    out = []
    head = f"{A.san} against {B.san}"
    if A.cp is not None and B.cp is not None:
        head += f": {A.cp / 100:+.1f} against {B.cp / 100:+.1f} by the engine"
    out.append(head + ".")
    if A.mate_threat and not B.mate_threat:
        out.append(f"{A.san} allows {A.mate_threat}, mate; {B.san} does not.")
        c.allows = [x for x in c.allows if not x.startswith(A.mate_threat)]
    if c.allows:
        out.append(f"{A.san} allows " + ", ".join(c.allows) + f" -- after {B.san} " + ("none of that is there." if len(c.allows) > 1 else "it is not there."))
    if A.base < B.base - 0.5:
        out.append(f"{A.san} {A.base_why or f'nets {A.base:+g}'}; {B.san} {B.base_why or f'nets {B.base:+g}'}.")
    elif B.base < A.base - 0.5:
        out.append(f"{B.san} {B.base_why or f'nets {B.base:+g}'}; {A.san} {A.base_why or f'nets {A.base:+g}'}.")
    if c.gives_up:
        out.append(f"{A.san} gives up " + ", ".join(c.gives_up) + f", which {B.san} keeps.")
    if c.creates:
        out.append(f"{A.san} creates " + ", ".join(c.creates) + f", which {B.san} does not.")
    if c.prevents:
        out.append(f"{A.san} prevents " + ", ".join(c.prevents) + f", which {B.san} allows.")
    if A.broken and not B.broken:
        out.append(f"{A.san} breaks what the piece held: " + "; ".join(A.broken) + ".")
    if A.structure or B.structure:
        if A.structure:
            out.append(f"Structure after {A.san}: " + "; ".join(A.structure) + ".")
        if B.structure:
            out.append(f"Structure after {B.san}: " + "; ".join(B.structure) + ".")
    if len(out) == 1:
        out.append("The two positions differ in nothing the ledger names: the difference is positional.")
    return out


# ---------------------------------------------------------------- the partners: the best move, the human move at a level
def partners(board: chess.Board, played: chess.Move | None, best: chess.Move | None, maia=None) -> list[tuple[str, chess.Move, chess.Move]]:
    """(label, A, B) pairs to contrast: the played move against the best when they differ; the best against the
    most common human move at the level when that differs (so a best move played is still explained against what
    a human would do)."""
    out = []
    human = None
    if maia is not None:
        try:
            top = maia.top(board, 1)
            if top:
                human = chess.Move.from_uci(top[0][0])
        except Exception:
            human = None
    if played is not None and best is not None and played != best:
        out.append(("played against best", played, best))
    if best is not None and human is not None and human != best:
        out.append((f"best against the human move ({HUMAN_LEVEL})", best, human))
    return out


def contrast_lines(board: chess.Board, played: chess.Move | None, best: chess.Move | None, memo=None, engine=None, maia=None) -> list[str]:
    out = []
    for label, a, b in partners(board, played, best, maia):
        try:
            c = contrast(board, a, b, memo, engine)
        except Exception:
            continue
        out.append(f"[{label}] " + " ".join(c.lines))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True); ap.add_argument("--move", required=True); ap.add_argument("--against", default="")
    ap.add_argument("--engine", action="store_true"); ap.add_argument("--human", type=int, default=HUMAN_LEVEL)
    a = ap.parse_args(argv)
    b = chess.Board(a.fen)
    eng = None; maia = None
    if a.engine:
        from .engine import Engine
        eng = Engine(depth=12)
    mv = b.parse_san(a.move)
    if a.against:
        other = b.parse_san(a.against)
    else:
        if eng is None:
            from .engine import Engine
            eng = Engine(depth=12)
        other = chess.Move.from_uci(eng.best(b, depth=12).move)
        if other == mv:
            from .maia import Maia
            maia = Maia(a.human)
            other = chess.Move.from_uci(maia.top(b, 1)[0][0])
    c = contrast(b, mv, other, Memo(), eng)
    for l in c.lines:
        print(l)
    if eng:
        eng.close()
    if maia:
        try:
            maia.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
