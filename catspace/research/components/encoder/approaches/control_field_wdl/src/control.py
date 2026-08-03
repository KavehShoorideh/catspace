"""catspace/controlfield/control.py -- the CONTROL FIELD (Kaveh's control-field/
ascent-cone spec, 2026-08-02). Ground-truth labels, no learning: a per-square scalar
measuring net force bearing on that square, so "force concentration" (bringing more
attackers to bear on a point while not losing defense elsewhere) is a measurable
quantity rather than a hand-coded heuristic.

Definitions (kept consistent with the spec):
  control field C(g; s)   -- scalar per square g, positive = side-to-move controls it.
  critical-square mask M(g) -- which squares the mover can't afford to lose control of.

Deviations from the spec, noted explicitly:
  - python-chess 1.11.2 (installed here) has NO Board.see()/see_ge() -- the spec's
    assumption that SEE is library-provided is wrong for this version. Variant B
    hand-implements the standard swap-off SEE algorithm below.
  - The weight table in the spec omits a queen weight. Filled in following the
    stated rationale (cheaper attackers contribute more usable pressure) as the
    next step down from rook, strictly below it: queen 0.3. Documented here, not
    silently assumed -- a config value, not a constant, like the rest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

# Variant A weights: pawn > knight ~= bishop > rook > queen > king. Rationale
# (spec): a cheap attacker contributes more usable pressure because it can capture
# and survive; queen filled in below rook (spec omitted it) since it's the piece
# least willing to trade, more so even than the point value alone suggests.
DEFAULT_ATTACK_WEIGHTS = {
    chess.PAWN: 1.0, chess.KNIGHT: 0.9, chess.BISHOP: 0.9,
    chess.ROOK: 0.6, chess.QUEEN: 0.3, chess.KING: 0.4,
}

# Variant B (SEE): standard material point values for the swap-off algorithm.
SEE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
              chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}

CENTRAL_SQUARES = frozenset({chess.D4, chess.D5, chess.E4, chess.E5})


def _flip_vertical(arr: np.ndarray) -> np.ndarray:
    """Board-vertical flip on a length-64 array indexed by chess.SQUARES: content
    at (file, rank) moves to (file, 7-rank). Same convention as lc0's POV-oriented
    input planes (files never flip, only ranks)."""
    out = np.zeros_like(arr)
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        out[chess.square(f, 7 - r)] = arr[sq]
    return out


def orient(field_white_pov: np.ndarray, mover_is_white: bool) -> np.ndarray:
    """White-POV field -> mover-POV field (spec 2.2): identity if White to move;
    vertical flip + negate if Black to move (so positive always = mover controls)."""
    if mover_is_white:
        return field_white_pov.copy()
    return -_flip_vertical(field_white_pov)


def weighted_attacker_field(board: chess.Board, weights: dict | None = None) -> np.ndarray:
    """Variant A (spec 2.1-A): C_raw(g) = A_white(g) - A_black(g), White POV,
    length-64 float32, indexed by chess.SQUARES."""
    w = weights or DEFAULT_ATTACK_WEIGHTS
    a_white = np.zeros(64, np.float32)
    a_black = np.zeros(64, np.float32)
    for sq in chess.SQUARES:
        for att in board.attackers(chess.WHITE, sq):
            a_white[sq] += w[board.piece_type_at(att)]
        for att in board.attackers(chess.BLACK, sq):
            a_black[sq] += w[board.piece_type_at(att)]
    return a_white - a_black


def _see_square(board: chess.Board, square: int, side: chess.Color) -> int:
    """Standard swap-off Static Exchange Evaluation: net material `side` wins by
    initiating captures on `square`, assuming best play (least-valuable-attacker-
    first) by both sides. Not in python-chess (see module docstring) -- this is the
    textbook algorithm (Fruit/Stockfish "see" routine), no external prior art needed
    beyond the well-known algorithm shape."""
    target = board.piece_at(square)
    if target is None:
        return 0

    def attackers_sorted(bd: chess.Board, sq: int, color: chess.Color):
        atts = []
        for a in bd.attackers(color, sq):
            pt = bd.piece_type_at(a)
            atts.append((SEE_VALUES[pt], a, pt))
        atts.sort(key=lambda t: t[0])
        return atts

    board = board.copy(stack=False)
    gain = [SEE_VALUES[target.piece_type]]      # gain[0] = value of the piece sitting there
    attacker_color = side
    depth = 0
    while True:
        atts = attackers_sorted(board, square, attacker_color)
        if not atts:
            break
        val, from_sq, pt = atts[0]               # val = value of the CAPTURING piece
        if pt == chess.KING:
            defenders = board.attackers(not attacker_color, square)
            if defenders:
                break   # can't capture with king into a defended square
        depth += 1
        gain.append(val - gain[depth - 1])       # net swing if the exchange stopped here
        piece = board.remove_piece_at(from_sq)
        board.set_piece_at(square, piece)
        attacker_color = not attacker_color
        if pt == chess.KING:
            break
    # backward negamax pass: each side only continues capturing if it's an
    # improvement, so fold from the leaves back to the root. depth==1 (a single
    # uncontested capture) folds zero times by construction -- range(0,0,-1) is
    # empty -- leaving gain[0] as the full uncontested value, not the (wrong)
    # once-folded value a naive while-loop would produce.
    for i in range(depth - 1, 0, -1):
        gain[i - 1] = -max(-gain[i - 1], gain[i])
    return gain[0]


def see_field(board: chess.Board) -> np.ndarray:
    """Variant B (spec 2.1-B): C_see(g), White POV, length-64 float32. Nonzero only
    on squares where at least one side has an attacker (sparse by construction --
    the spec's own expectation)."""
    c = np.zeros(64, np.float32)
    for sq in chess.SQUARES:
        white_atts = bool(board.attackers(chess.WHITE, sq))
        black_atts = bool(board.attackers(chess.BLACK, sq))
        if not white_atts and not black_atts:
            continue
        # side to move at THIS square for SEE purposes: whoever would capture
        # first if initiating an exchange here -- White's SEE value if White has
        # any attacker (can choose to initiate), else Black's (mirrored, negated).
        if white_atts:
            c[sq] += _see_square(board, sq, chess.WHITE)
        if black_atts:
            c[sq] -= _see_square(board, sq, chess.BLACK)
    return c / 100.0   # centipawns -> pawns, matching Variant A's ~unit scale


@dataclass
class MaskConfig:
    critical_value: float = 1.0
    other_piece_value: float = 0.6
    central_value: float = 0.3
    critical_piece_types: frozenset = field(
        default_factory=lambda: frozenset({chess.ROOK, chess.QUEEN, chess.KING}))


def critical_square_mask(board: chess.Board, cfg: MaskConfig | None = None) -> np.ndarray:
    """Spec 2.3: M(g) in [0,1], WHITE-POV (oriented like the control field by the
    caller, via orient() -- the mask itself carries no sign, only the square
    permutation applies under orientation, since 0/0.3/0.6/1.0 aren't POV-signed)."""
    cfg = cfg or MaskConfig()
    mover = board.turn
    m = np.zeros(64, np.float32)
    king_sq = board.king(mover)
    king_zone = set()
    if king_sq is not None:
        kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                f, r = kf + df, kr + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    king_zone.add(chess.square(f, r))
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is not None and pc.color == mover and pc.piece_type in cfg.critical_piece_types:
            m[sq] = cfg.critical_value
        elif sq in king_zone:
            m[sq] = cfg.critical_value
        elif pc is not None and pc.color == mover:
            m[sq] = cfg.other_piece_value
        elif sq in CENTRAL_SQUARES:
            m[sq] = cfg.central_value
    return m


def compute_fields(board: chess.Board, weights=None, mask_cfg: MaskConfig | None = None):
    """-> (C_a, C_see, M), all mover-POV, length-64 float32. The one entry point
    Phase 2 (derivative.py) and the data pipeline should call."""
    mover_is_white = board.turn == chess.WHITE
    c_a = orient(weighted_attacker_field(board, weights), mover_is_white)
    c_see = orient(see_field(board), mover_is_white)
    m_white_pov = critical_square_mask(board, mask_cfg)
    m = m_white_pov if mover_is_white else _flip_vertical(m_white_pov)
    return c_a, c_see, m
