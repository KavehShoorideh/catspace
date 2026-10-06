"""TACTIC RULES FOR THE ZERO-SEARCH PLAYER (Kaveh 2026-10-05, the PoC): rules written by the agent from the player's
misses, each a bounded material claim about a move with hypotheses the board can check, and a bounded LOCAL check inside
the move's own frame (Kaveh: "a bounded local check inside its own frame, like the frames' exchange resolution"; no
tree search over the position). Each ships with its BRIDGE: the claim checked against an exact enumeration on the
positions where it fires (0 wrong before any number is quoted), and is a CONJECTURE until its Lean statement exists.

Rule 1, THE DOUBLE THREAT (fork and discovered attack; from development misses 03R2c Nf4, 0550Q Bxc3, 07i5h Nd3,
03bzK Nf6+, 046Rc Bxh7+, 01wbG Qxg8+): a move that newly attacks two enemy pieces, or one with check. Claim: after
ANY legal reply, we capture for at least k (the exchange on its square, the frames' primitive) one of: the threatened
pieces, the piece that just captured (the recapture), a piece the reply left undefended (the deflection); net of what
the reply took and what the move itself took. k = the minimum over replies. A 3-ply material claim: our move, their
reply, our capture; the exchange on the capture's square is played out, nothing elsewhere.
"""
from __future__ import annotations

import chess

from .fates import captured_value, exchange, square_frame

VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0, chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}
MATE = 1000.0


def material(b: chess.Board, side: chess.Color) -> float:
    return sum(VAL[p.piece_type] * (1 if p.color == side else -1) for p in b.piece_map().values())


def _attacked_by(b: chess.Board, side: chess.Color) -> set[int]:
    """Enemy non-king pieces that `side` attacks."""
    return {sq for sq, p in b.piece_map().items() if p.color != side and p.piece_type != chess.KING and b.is_attacked_by(side, sq)}


def double_threats(b: chess.Board) -> list[tuple[chess.Move, list[int], bool]]:
    """The moves that create two new threats, or one with check: [(move, newly threatened squares, gives check)]."""
    us = b.turn
    before = _attacked_by(b, us)
    out = []
    for m in b.legal_moves:
        nb = b.copy(stack=False)
        nb.push(m)
        if nb.is_checkmate():
            continue                                                     # the ledger values mates itself
        new = sorted((_attacked_by(nb, us) - before) | {t for t in nb.attacks(m.to_square) if (p := nb.piece_at(t)) and p.color != us and p.piece_type != chess.KING})
        check = nb.is_check()
        if len(new) >= 2 or (new and check):
            out.append((m, new, check))
    return out


def local_claim(b: chess.Board, m: chess.Move, targets: list[int]) -> tuple[float, str]:
    """The bounded local check: for every legal reply, our best exchange on a threatened square (the frames'
    `exchange`), net of what the reply and the move itself took. -> (the minimum over replies, the worst reply)."""
    us = b.turn
    won = captured_value(b, m)
    nb = b.copy(stack=False)
    nb.push(m)
    worst, worst_reply = None, None
    for r in nb.legal_moves:
        nb2 = nb.copy(stack=False)
        nb2.push(r)
        lost = captured_value(nb, r)
        gain = 0.0
        if nb2.is_checkmate():
            gain = -MATE
        elif nb2.is_stalemate():
            total = 0.0                                                  # a stalemating reply draws: no material claim (review F2a)
            if worst is None or total < worst:
                worst, worst_reply = total, r
            continue
        else:
            squares = set(targets)
            squares.add(r.to_square)                                     # the recapture of the piece that just took (Nb5 Qxf2+ Kxf2)
            for t, p in nb2.piece_map().items():                         # a piece the reply left undefended (the deflection:
                if p.color != us and p.piece_type != chess.KING and nb2.is_attacked_by(us, t) and not nb2.is_attacked_by(not us, t):   # Qxg8+ Kxg8 Bxa1)
                    squares.add(t)
            for t in squares:
                p = nb2.piece_at(t)
                if p is None or p.color == us or not nb2.is_attacked_by(us, t):
                    continue
                v, line = exchange(nb2, t, set(square_frame(nb2, t)), us)
                if line:                                                 # the truth is 3 plies: our first capture's own outcome
                    nb3 = nb2.copy(stack=False)                          # (a mate is MATE flat, a stalemate a draw: review F2b/F3)
                    nb3.push(chess.Move.from_uci(line[0]))
                    if nb3.is_checkmate():
                        v = MATE
                    elif nb3.is_stalemate():
                        v = 0.0
                gain = max(gain, min(v, MATE))
        total = min(won - lost + gain, MATE)
        if worst is None or total < worst:
            worst, worst_reply = total, r
    if worst is None:                                                    # no legal reply: mate (the ledger's) or stalemate
        return 0.0, ""
    return worst, nb.san(worst_reply)


def fork_claims(b: chess.Board, min_gain: float = 1.0) -> dict:
    """The rule in play: {uci: (claim, why)} for the double threats whose local check wins at least min_gain."""
    out = {}
    for m, targets, check in double_threats(b):
        k, reply = local_claim(b, m, targets)
        if k >= min_gain:
            names = ", ".join(chess.square_name(t) for t in targets)
            out[m.uci()] = (k, f"{'check and ' if check else ''}new threats on {names}: after any reply we take at least {k:+.1f} (worst reply {reply})")
    return out


def exact_3ply(b: chess.Board, m: chess.Move) -> float:
    """THE BRIDGE's truth: material for the mover after our move, their best reply and our best answer (3 plies,
    exact over all legal moves; a mate is +-MATE), minus material now."""
    us = b.turn
    base = material(b, us)
    nb = b.copy(stack=False)
    nb.push(m)
    if nb.is_checkmate():
        return MATE
    if nb.is_stalemate():
        return 0.0
    worst = None
    for r in nb.legal_moves:
        nb2 = nb.copy(stack=False)
        nb2.push(r)
        if nb2.is_checkmate():
            v = -MATE
        elif nb2.is_stalemate():
            v = 0.0
        else:
            v = None
            for m2 in nb2.legal_moves:
                nb3 = nb2.copy(stack=False)
                nb3.push(m2)
                x = MATE if nb3.is_checkmate() else 0.0 if nb3.is_stalemate() else material(nb3, us) - base   # a stalemate draws, at any ply
                v = x if v is None else max(v, x)
        worst = v if worst is None else min(worst, v)
    return worst if worst is not None else 0.0


def bridge(fens: list[str]) -> dict:
    """Every claim the rule makes on these positions against the exact 3-ply value: a claim above the truth is WRONG.
    -> dict(positions, claims, wrong, slack=[truth - claim], examples)."""
    claims = wrong = 0
    slack, examples = [], []
    for f in fens:
        b = chess.Board(f)
        for u, (k, why) in fork_claims(b).items():
            claims += 1
            truth = exact_3ply(b, chess.Move.from_uci(u))
            slack.append(truth - k)
            if k > truth + 1e-9:
                wrong += 1
                if len(examples) < 10:
                    examples.append(dict(fen=f, move=u, claim=k, truth=truth, why=why))
    return dict(positions=len(fens), claims=claims, wrong=wrong, slack=slack, examples=examples)
