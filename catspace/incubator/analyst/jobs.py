"""THE JOB LEDGER -- what each piece is holding, kept as knowledge (Kaveh 2026-09-30: "set up the structure so that our
knowledge of the pin, and similar tactics, is leveraged to the maximum, so we are faster and better"; Stockfish knows a
pin to the king by one bit per move and prunes with it, but a pin to the queen only statistically, re-found line after
line; and "knowing the knight is pinned is only useful computationally when it helps with early pruning or with the
search for a solution to the pin").

A JOB: a piece whose presence keeps a frame from winning -- found generically, by removing the piece and reading the
frame books again (the ablation idea, per frame): the frames that improve for the other side are the holder's
DEFENSIVE jobs -- a SHIELD when the attacker's line to the target runs through it (a pin; to the king: absolute), a
GUARD when it defends the target, a BLOCK otherwise -- and the frames that improve for its own side are its
DISCOVERIES (its own piece standing in its own attack's way). Each job: holder, target, attacker, value, footprint
(the holder, the target, the attacker and the squares between) -- valid while those squares hold what they held.

Two uses:
- FASTER: the zone search orders a move that abandons a valid job worth more than it takes LAST (the capture that
  refutes it is a high-value capture and comes first anyway) -- the pin's knowledge in place of re-finding it by search
  in every sibling line; `TacticalSpec.prune_jobs` may also drop such moves near the horizon (off by default).
- BETTER: the jobs name moves -- against the opponent's jobs: attack the holder (pressure), hit one of an overloaded
  holder's targets (deflection); for ours under attack: move the target off the line, interpose, relieve the holder
  with a second guard; our discoveries: the holder's moves that also threaten (a double attack) -- the candidate set's
  job operators (frame_player JOBS).

    python -m catspace.incubator.analyst.jobs --fen "<fen>"
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import chess

from . import fates as F

DECISIVE = 100.0


@dataclass
class Job:
    holder: int
    piece: str
    colour: chess.Color                 # the holder's side
    kind: str                           # "shield" | "guard" | "block" | "discovery"
    target: int                         # the square whose frame flips (for a king: the king's square)
    attacker: int | None                # the piece that cashes in (a discovery: our own attacker behind the holder)
    value: float                        # what the other side gains (a discovery: what we gain) if the holder is gone
    footprint: tuple = field(default=())    # (square, symbol) pairs: the job holds while they are unchanged

    def valid(self, b: chess.Board) -> bool:
        return all((b.piece_at(s).symbol() if b.piece_at(s) else None) == sym for s, sym in self.footprint)


def _book(b: chess.Board, side: chess.Color) -> dict | None:
    """The side's frames with their values: {target square: value} (a king frame keyed by the king's square)."""
    t = b if b.turn == side else F.passed(b)
    if t is None:
        return None
    out = {}
    for key, (_fr, _, v, _ln) in F.frame_book(t).items():
        sq = key[1] if key[0] == "sq" else t.king(key[1])
        if sq is not None:
            out[sq] = max(out.get(sq, 0.0), float(v))
    return out


def _unveiled(before: chess.Board, after: chess.Board, side: chess.Color, target: int, holder: int) -> int | None:
    """A `side` slider that attacks target once the holder is gone, through the holder's square."""
    for a in chess.SquareSet(after.attackers_mask(side, target) & ~before.attackers_mask(side, target)):
        if holder in chess.SquareSet(chess.between(a, target)):
            return a
    return None


def _foot(b: chess.Board, squares) -> tuple:
    return tuple((s, b.piece_at(s).symbol() if b.piece_at(s) else None) for s in sorted(set(squares)))


def ledger(b: chess.Board, min_value: float = 0.5) -> list:
    """Every piece's jobs in the position."""
    base = {c: _book(b, c) for c in (chess.WHITE, chess.BLACK)}
    out = []
    for s, pc in b.piece_map().items():
        if pc.piece_type == chess.KING:
            continue
        c = pc.color
        e = b.copy(stack=False); e.remove_piece_at(s)
        k = e.king(c)
        if k is not None and e.is_attacked_by(not c, k):                  # it shields its own king: an absolute pin
            a = next((x for x in chess.SquareSet(e.attackers_mask(not c, k)) if s in chess.SquareSet(chess.between(x, k))), None)
            if a is not None:
                out.append(Job(s, pc.symbol(), c, "shield", k, a, DECISIVE,
                               _foot(b, {s, k, a} | set(chess.SquareSet(chess.between(a, k))))))
            continue
        for side, own in ((not c, False), (c, True)):
            before, after = base[side], _book(e, side)
            if before is None or after is None:
                continue
            for t, v in after.items():
                gain = v - before.get(t, 0.0)
                if t == s or gain < min_value:
                    continue
                if v >= 500.0:                                            # a mate: the job is on the king, whatever square keyed
                    kk = e.king(not side)                                 # the frame ("Rc8 guards against mate", not "blocks for Nh6")
                    if kk is not None:
                        t = kk
                a = _unveiled(b, e, side, t, s)
                if own:
                    if a is None:
                        continue                                          # our own piece in our way: only a line it blocks
                    out.append(Job(s, pc.symbol(), c, "discovery", t, a, gain,
                                   _foot(b, {s, t, a} | set(chess.SquareSet(chess.between(a, t))))))
                    continue
                if a is not None:
                    kind, foot = "shield", {s, t, a} | set(chess.SquareSet(chess.between(a, t)))
                elif t in chess.SquareSet(b.attacks_mask(s)):
                    kind, foot = "guard", {s, t} | set(chess.SquareSet(b.attackers_mask(side, t)))
                    a = next(iter(chess.SquareSet(b.attackers_mask(side, t))), None)
                else:
                    kind, foot = "block", {s, t}
                out.append(Job(s, pc.symbol(), c, kind, t, a, gain, _foot(b, foot)))
    best = {}                                                         # one job per holder, kind and target
    for j in out:
        k = (j.holder, j.kind, j.target)
        if k not in best or j.value > best[k].value:
            best[k] = j
    out = sorted(best.values(), key=lambda j: -j.value)
    return out


def abandons(b: chess.Board, m: chess.Move, jobs: list) -> float:
    """The value of the valid jobs the move walks away from (its holder leaves; a shield that stays on its line and a
    guard that still guards do not count), less what it takes -- > 0: the move gives away more than it wins."""
    lost = 0.0
    for j in jobs:
        if j.holder != m.from_square or j.kind == "discovery" or not j.valid(b):
            continue
        if j.kind == "shield" and j.attacker is not None and m.to_square in chess.SquareSet(chess.between(j.attacker, j.target)) | {j.attacker}:
            continue
        if j.kind == "guard":
            nb = b.copy(stack=False); nb.push(m)
            if j.target in chess.SquareSet(nb.attacks_mask(m.to_square)):
                continue
        lost = max(lost, j.value)
    return lost - F.captured_value(b, m)


def operators(b: chess.Board, jobs: list) -> list:
    """The moves the jobs name for the side to move (ucis): pressure on the opponent's holders, deflection of an
    overloaded one, a discovery that also threatens; for our own holders under attack, the target off the line, an
    interposition, a second guard."""
    me = b.turn
    out = set()
    theirs = [j for j in jobs if j.colour != me and j.kind != "discovery" and j.valid(b)]
    ours = [j for j in jobs if j.colour == me and j.kind != "discovery" and j.valid(b)]
    load = {}
    for j in theirs:
        load.setdefault(j.holder, []).append(j)
    for m in b.legal_moves:
        nb = b.copy(stack=False); nb.push(m)
        att = set(chess.SquareSet(nb.attacks_mask(m.to_square)))
        for h, js in load.items():
            if h in att:
                out.add(m.uci())                                          # pressure on a holder
            if len(js) >= 2 and any(j.target in att for j in js):
                out.add(m.uci())                                          # an overloaded holder: hit one of its targets
        for j in ours:
            if j.value < 1.0:
                continue
            if m.from_square == j.target:
                out.add(m.uci())                                          # the target off the line / out of the frame
            elif j.kind == "shield" and j.attacker is not None and m.to_square in chess.SquareSet(chess.between(j.attacker, j.target)):
                out.add(m.uci())                                          # interpose
            elif j.kind in ("shield", "guard") and j.target in att and m.from_square != j.holder:
                out.add(m.uci())                                          # a second guard relieves the holder
            elif j.attacker is not None and m.to_square == j.attacker:
                out.add(m.uci())                                          # take the attacker
    for j in jobs:
        if j.colour == me and j.kind == "discovery" and j.valid(b):
            for m in b.legal_moves:
                if m.from_square == j.holder:
                    nb = b.copy(stack=False); nb.push(m)
                    if nb.is_check() or any(nb.piece_at(t) is not None and nb.piece_at(t).color != me
                                            for t in chess.SquareSet(nb.attacks_mask(m.to_square))):
                        out.add(m.uci())                                  # the discovery with a second threat: a double attack
    return sorted(out)


def explain(b: chess.Board, jobs: list) -> list:
    n = lambda s: f"{b.piece_at(s).symbol() if b.piece_at(s) else ''}{chess.square_name(s)}" if s is not None else "?"
    lines = []
    for j in jobs:
        if j.kind == "discovery":
            lines.append(f"{n(j.holder)} stands in the way of {n(j.attacker)} on {n(j.target)}: moving it gains {j.value:+g}")
        else:
            what = {"shield": "shields", "guard": "guards", "block": "blocks for"}[j.kind]
            own_king = b.king(j.colour) == j.target
            cost = ("an absolute pin: it cannot leave the line" if own_king and j.kind == "shield" and j.value == DECISIVE
                    else "losing it allows mate" if j.value >= 400 else f"losing it costs {j.value:+g}")
            lines.append(f"{n(j.holder)} {what} {n(j.target)}" + (f" against {n(j.attacker)}" if j.attacker is not None else "") + f": {cost}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--fen", required=True)
    a = ap.parse_args(argv)
    b = chess.Board(a.fen)
    js = ledger(b)
    for line in explain(b, js):
        print(line)
    print("the jobs' moves for the side to move:", [b.san(chess.Move.from_uci(u)) for u in operators(b, js)])


if __name__ == "__main__":
    main()
