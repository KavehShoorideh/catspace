"""WHAT DID THIS MOVE CREATE? -- matching threat lemmas after a move (Kaveh 2026-10-08: "we always want to recognize the
existence of winning frames ... a knight landing on a fork square is already a winning threat ... if they move this
one, that one dies ... we should be able to prove this using our lemmas ... loop over applicable lemmas and see what
applies in this position ... cornering is a frame type of its own -- the mate frame: we could be mating a king, or
cornering a queen, trapping a rook ... we should be able to mate any piece").

THE PROOF is one kernel certificate for all of them: `Ideas/Fork.lean`, `localCheck` with `localCheck_sound` -- after
move m, EVERY legal reply is answered by one of our captures (or a king move) after which, whatever they reply, we are
up at least k (their mate counts as won). A fork, a trapped (cornered) piece, a single threat that cannot be parried
and a mate threat are all instances: the certificate does not care which; the NAME does, so the claim is named by its
shape:
  fork              two or more enemy pieces newly attacked with a winning static exchange, the gain the second best
  cornered piece    one piece newly attacked, the gain its own value: it cannot escape, be defended, or have its
                    attacker taken or blocked -- "mating a piece" (Kaveh)
  mate threat       a mate next move were the opponent to pass; certified as a gain no material can reach (MATE_K)
  promotion         a pawn newly on its seventh rank with its promotion square safe -- the certificate's gain is the
                    new queen
A claim is CERTIFIED (the kernel's localCheck says yes: proved by localCheck_sound) or only MATCHED (the shape is
there, the certificate fails -- a candidate, not a fact). Only certified claims enter the frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
MATE_K = 50                                                 # a gain no material can reach: only a mate meets it


@dataclass
class Claim:
    name: str                                               # fork / cornered piece / mate threat / promotion
    move: str                                               # UCI
    k: int                                                  # the gain claimed
    targets: tuple                                          # square names
    certified: bool


def _winning_targets(b: chess.Board, side: chess.Color) -> dict:
    """{square: static gain} of `side`'s winning captures were it to move now (on the safe null move if needed)."""
    from .chess_domain import passed
    from .exchange_law import see
    p = b if b.turn == side else passed(b)
    if p is None:                                           # the other side is in check (no null move): for the static
        p = b.copy(stack=False); p.turn = side; p.ep_square = None   # attack reading alone a turn flip is safe (no legality)
    out = {}
    for q, pc in p.piece_map().items():
        if pc.color != side and pc.piece_type != chess.KING and p.attackers(side, q):
            g = see(p, q)[0]
            if g > 0:
                out[q] = g
    return out


def _mate_in_one(b: chess.Board, side: chess.Color) -> bool:
    from .chess_domain import passed
    p = b if b.turn == side else passed(b)
    if p is None:
        return False
    for x in list(p.legal_moves):
        p.push(x)
        mate = p.is_checkmate()
        p.pop()
        if mate:
            return True
    return False


def claims(b: chess.Board, m: chess.Move, certify: bool = True) -> list:
    """The threat lemmas a move matches: what it CREATED (targets not already winning before it), each certified by the
    kernel when `certify`."""
    side = b.turn
    nb = b.copy(stack=False); nb.push(m)
    if nb.is_game_over():
        return []
    before = _winning_targets(b, side)
    after = _winning_targets(nb, side)
    new = {q: g for q, g in after.items() if q not in before or g > before[q]}
    out = []
    if len(new) >= 2:
        vals = sorted(new.values(), reverse=True)
        out.append(Claim("fork", m.uci(), vals[1], tuple(chess.square_name(q) for q in new), False))
    elif len(new) == 1:
        (q, g), = new.items()                               # with check it is a fork of the king and the piece
        out.append(Claim("fork with check" if nb.is_check() else "cornered piece", m.uci(), g, (chess.square_name(q),), False))
    if not _mate_in_one(b, side) and _mate_in_one(nb, side):
        out.append(Claim("mate threat", m.uci(), MATE_K, (chess.square_name(nb.king(not side)),), False))
    pc = b.piece_at(m.from_square)
    if pc is not None and pc.piece_type == chess.PAWN and chess.square_rank(m.to_square) == (6 if side == chess.WHITE else 1):
        out.append(Claim("promotion", m.uci(), VAL[chess.QUEEN] - 1, (chess.square_name(m.to_square),), False))
    if certify and out:
        from .lean_ideas import detector
        li = detector()
        for c in out:
            try:
                c.certified = li.fork(b, m, c.k)
            except Exception:
                c.certified = False
    return out


def certified_gain(b: chess.Board, m: chess.Move) -> int:
    """The largest gain a certified threat lemma proves for m (0 when none): a lower bound on what m wins within three
    plies, whatever the opponent does."""
    return max((c.k for c in claims(b, m) if c.certified), default=0)


def threat_moves(b: chess.Board) -> list:
    """The side to move's quiet moves that a certified threat lemma proves winning: [(move, k)]. Cheap prefilter first:
    a move must newly attack an enemy piece or give a mate threat to be checked by the kernel."""
    out = []
    for m in b.legal_moves:
        if b.is_capture(m) or b.gives_check(m) or m.promotion:
            continue                                        # captures and checks are the frame's own moves
        g = certified_gain(b, m)
        if g > 0:
            out.append((m, g))
    return out
