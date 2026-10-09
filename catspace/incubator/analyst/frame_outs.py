"""THE OUTS OF A FRAME, with their probabilities (Kaveh 2026-10-09: "this joined frame has multiple different outs, but
there is one out with a non-negligible probability of happening ... bring in Maia and estimate the probability of each
out in my frame given my opponent -- and leave room for the remaining time, but not yet").

For our move m: every reply r of the opponent is an OUT of the frame. Its probability is Maia's policy at the
opponent's rating band (lc0 with the Maia weights, one node; explain.maia_policy). Its OUTCOME for us, in pawns, is the
material r takes from us, net of the largest gain we can then CERTIFY (the rule layer's settled certificates on the
position after r: analyst_core.threat_claims; mate counts 50). Classes: WALKS IN (outcome >= +3: the tactic fires),
NEUTRAL, COSTS US (outcome <= -1). The expected value of m is the probability-weighted outcome; the worst out is the
value against best defence among the replies considered.

`clock` (the opponent's remaining time) is accepted and NOT used yet: the slot for a time-conditioned policy later.
"""
from __future__ import annotations

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def _our_best_certified(b: chess.Board) -> int:
    import analyst_core
    if b.is_checkmate():
        return 0
    return max([k for _, _, k, _, _, framed in analyst_core.threat_claims(b.fen()) if framed] + [0])


def outs(b: chess.Board, m: chess.Move, band: int = 1500, clock: float | None = None, top: int = 12) -> dict:
    """-> dict(outs=[(reply san, probability, outcome, class)], p_walks_in, expected, worst)."""
    from .explain import maia_policy
    _ = clock                                                         # reserved: the opponent's remaining time, later
    nb = b.copy(stack=False); nb.push(m)
    if nb.is_game_over():
        return dict(outs=[], p_walks_in=0.0, expected=0.0, worst=0.0)
    pol = maia_policy(nb, band)
    rows = []
    for uci, p in sorted(pol.items(), key=lambda kv: -kv[1])[:top]:
        r = chess.Move.from_uci(uci)
        if r not in nb.legal_moves:
            continue
        taken = VAL[nb.piece_at(r.to_square).piece_type] if nb.piece_at(r.to_square) else (1 if nb.is_en_passant(r) else 0)
        b2 = nb.copy(stack=False); b2.push(r)
        gain = 0 if b2.is_checkmate() else min(50, _our_best_certified(b2))
        outcome = gain - taken - (50 if b2.is_checkmate() else 0)
        cls = "walks in" if outcome >= 3 else "costs us" if outcome <= -1 else "neutral"
        rows.append((nb.san(r), p, outcome, cls))
    mass = sum(p for _, p, _, _ in rows) or 1.0
    return dict(outs=rows, p_walks_in=sum(p for _, p, _, c in rows if c == "walks in") / mass,
                expected=sum(p * o for _, p, o, _ in rows) / mass, worst=min((o for *_, o, _ in [(x[0], x[1], x[2], x[3]) for x in rows]), default=0.0),
                covered=mass)
