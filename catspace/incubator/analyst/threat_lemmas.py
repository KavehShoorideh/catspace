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
        for c in out:
            c.certified = local_check(b, m, c.k)
    return out


_CERT: dict = {}                                            # (fen, uci, k) -> bool: a conclusion is computed once
VERIFY = __import__("os").environ.get("CATSPACE_VERIFY_CERT", "") == "1"   # re-check every Rust answer against Lean


def local_check(b: chess.Board, m: chess.Move, k: int) -> bool:
    """The kernel's localCheck (Fork.lean, proved sound by localCheck_sound), answered by its Rust mirror
    (analyst_core.local_check -- Kaveh: hot loops to the Rust layer) and cached; with CATSPACE_VERIFY_CERT=1 every
    answer is re-checked against the Lean binary and a disagreement raises."""
    key = (b.fen(), m.uci(), k)
    hit = _CERT.get(key)
    if hit is not None:
        return hit
    try:
        import analyst_core
        ans = analyst_core.local_check(key[0], key[1], k)
    except Exception:
        ans = None
    if ans is None or VERIFY:
        from .lean_ideas import detector
        lean = detector().fork(b, m, k)
        if ans is not None and ans != lean:
            raise AssertionError(f"Rust localCheck disagrees with the kernel: {key} rust {ans} lean {lean}")
        ans = lean
    _CERT[key] = ans
    return ans


_RUST: dict = {}                                            # fen -> {uci: certified gain}: the rule layer's conclusions, once
FRAMED = __import__("os").environ.get("CATSPACE_FRAMED_CERT", "") == "1"   # also accept the frame-sized certificate
                                                            # (captures resolved at the leaf; Rust-certified, theorem to write)


def rust_gains(b: chess.Board) -> dict | None:
    """{uci: largest certified gain} for every legal move, from the Rust rule layer (Ascent rules + the mirrored
    certificate; analyst_core.threat_claims), cached per position; None if the core is unavailable."""
    fen = b.fen()
    hit = _RUST.get(fen)
    if hit is not None:
        return hit
    try:
        import analyst_core
        rows = analyst_core.threat_claims(fen)
    except Exception:
        return None
    out: dict = {}
    for uci, name, k, cert, _, framed in rows:
        if cert or (FRAMED and framed):
            out[uci] = max(out.get(uci, 0), k)
    _RUST[fen] = out
    return out


def certified_gain(b: chess.Board, m: chess.Move) -> int:
    """The largest gain a certified threat lemma proves for m (0 when none): a lower bound on what m wins within three
    plies, whatever the opponent does. The Rust rule layer when available, else the Python matcher."""
    g = rust_gains(b)
    if g is not None:
        return g.get(m.uci(), 0)
    return max((c.k for c in claims(b, m) if c.certified), default=0)


def threat_moves(b: chess.Board) -> list:
    """The side to move's quiet moves that a certified threat lemma proves winning: [(move, k)]. Cheap prefilter first:
    a move must newly attack an enemy piece or give a mate threat to be checked by the kernel."""
    out = []
    g = rust_gains(b)
    for m in b.legal_moves:
        if b.is_capture(m) or b.gives_check(m) or m.promotion:
            continue                                        # captures and checks are the frame's own moves
        k = g.get(m.uci(), 0) if g is not None else certified_gain(b, m)
        if k > 0:
            out.append((m, k))
    return out
