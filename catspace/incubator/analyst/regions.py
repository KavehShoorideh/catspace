"""REGIONS (Kaveh 2026-09-19): from the rules of movement, attack, capture, promotion and check ALONE, what a
piece can reach. Two layers: the SAFE closure -- squares reachable by the piece's own moves through squares the
enemy does not attack (the king by the rule of check), the opponent passing; and the CONTESTED frontier --
squares reachable only onto or through attacked squares, each tagged with its attackers, the pieces that would
have to be deflected. A sacrifice is a contested square whose attacker's other duties are worth more; it is a
question for a deeper search, never excluded here. A pawn's region is its path; the promotion squares are the
last rank inside it. Concepts such as "cut off" or "cannot be dislodged" are NOT written here: the relations
between regions are enumerated mechanically (relations()) and a learned model decides which matter."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

BB_ALL = chess.BB_ALL


def attacked_by(board: chess.Board, color: chess.Color) -> int:
    """The squares `color` attacks (pawns by their capture squares), as a bitboard."""
    bb = 0
    for sq in chess.scan_forward(board.occupied_co[color]):
        bb |= int(board.attacks(sq))
    return bb


def _steps_from(board: chess.Board, piece: chess.Piece, sq: int, origin: int) -> int:
    """The squares the piece could move to from `sq` on this board (its own square vacated), as a bitboard:
    empty squares and enemy-occupied squares (a capture ends the walk there). Pawns: pushes and captures."""
    occ = board.occupied & ~chess.BB_SQUARES[origin]
    own = board.occupied_co[piece.color] & ~chess.BB_SQUARES[origin]
    them = board.occupied_co[not piece.color]
    pt, c = piece.piece_type, piece.color
    if pt == chess.PAWN:
        fwd = 8 if c == chess.WHITE else -8
        out = 0
        one = sq + fwd
        if 0 <= one < 64 and not (occ & chess.BB_SQUARES[one]):
            out |= chess.BB_SQUARES[one]
            start_rank = 1 if c == chess.WHITE else 6
            two = sq + 2 * fwd
            if chess.square_rank(sq) == start_rank and 0 <= two < 64 and not (occ & chess.BB_SQUARES[two]):
                out |= chess.BB_SQUARES[two]
        out |= int(chess.BB_PAWN_ATTACKS[c][sq]) & them
        return out
    if pt == chess.KNIGHT:
        att = chess.BB_KNIGHT_ATTACKS[sq]
    elif pt == chess.KING:
        att = chess.BB_KING_ATTACKS[sq]
    else:
        att = 0
        if pt in (chess.BISHOP, chess.QUEEN):
            att |= chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]
        if pt in (chess.ROOK, chess.QUEEN):
            att |= chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ] | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ]
    return int(att) & ~own


@dataclass
class Region:
    square: int
    piece: chess.Piece
    safe: int = 0                                  # bitboard: reachable through unattacked squares (the origin included)
    steps: dict = field(default_factory=dict)      # square -> the fewest moves (safe walk)
    contested: dict = field(default_factory=dict)  # square -> (fewest moves, attackers bitboard) for squares reachable only via attacked ones
    attacks_from: dict = field(default_factory=dict)   # square in the region -> the squares the piece would attack standing there

    @property
    def contested_bb(self) -> int:
        bb = 0
        for s in self.contested:
            bb |= chess.BB_SQUARES[s]
        return bb

    def promotion_squares(self) -> int:
        if self.piece.piece_type != chess.PAWN:
            return 0
        last = chess.BB_RANK_8 if self.piece.color == chess.WHITE else chess.BB_RANK_1
        return (self.safe | self.contested_bb) & last


def region(board: chess.Board, sq: int, max_steps: int = 16) -> Region:
    piece = board.piece_at(sq)
    assert piece is not None
    enemy_att = attacked_by(board, not piece.color)
    r = Region(sq, piece)
    r.safe = chess.BB_SQUARES[sq]; r.steps[sq] = 0
    # the safe walk: breadth first over unattacked squares
    frontier = [sq]
    for k in range(1, max_steps + 1):
        nxt = []
        for s in frontier:
            for t in chess.scan_forward(_steps_from(board, piece, s, sq)):
                if enemy_att & chess.BB_SQUARES[t] or r.safe & chess.BB_SQUARES[t]:
                    continue
                r.safe |= chess.BB_SQUARES[t]; r.steps[t] = k
                if not (board.occupied_co[not piece.color] & chess.BB_SQUARES[t]):        # a capture ends the walk
                    nxt.append(t)
        if not nxt:
            break
        frontier = nxt
    # the contested frontier: squares reachable only via attacked squares, with their attackers; a walk of the
    # same kind that may enter attacked squares but records the deflection needed
    seen = dict(r.steps)
    frontier = [(s, 0) for s in r.steps]
    for k in range(1, max_steps + 1):
        nxt = []
        for s, _ in frontier:
            for t in chess.scan_forward(_steps_from(board, piece, s, sq)):
                if t in seen:
                    continue
                seen[t] = k
                att = 0
                for a in chess.scan_forward(board.occupied_co[not piece.color]):
                    if int(board.attacks(a)) & chess.BB_SQUARES[t]:
                        att |= chess.BB_SQUARES[a]
                r.contested[t] = (k, att)
                if not (board.occupied_co[not piece.color] & chess.BB_SQUARES[t]):
                    nxt.append((t, k))
        if not nxt:
            break
        frontier = nxt
    for s in list(r.steps) + list(r.contested):
        r.attacks_from[s] = _attacks_at(board, piece, s, sq)
    return r


def _attacks_at(board: chess.Board, piece: chess.Piece, s: int, origin: int) -> int:
    occ = board.occupied & ~chess.BB_SQUARES[origin]
    pt, c = piece.piece_type, piece.color
    if pt == chess.PAWN:
        return int(chess.BB_PAWN_ATTACKS[c][s])
    if pt == chess.KNIGHT:
        return int(chess.BB_KNIGHT_ATTACKS[s])
    if pt == chess.KING:
        return int(chess.BB_KING_ATTACKS[s])
    att = 0
    if pt in (chess.BISHOP, chess.QUEEN):
        att |= chess.BB_DIAG_ATTACKS[s][chess.BB_DIAG_MASKS[s] & occ]
    if pt in (chess.ROOK, chess.QUEEN):
        att |= chess.BB_RANK_ATTACKS[s][chess.BB_RANK_MASKS[s] & occ] | chess.BB_FILE_ATTACKS[s][chess.BB_FILE_MASKS[s] & occ]
    return int(att)


def regions(board: chess.Board) -> dict[int, Region]:
    return {sq: region(board, sq) for sq in chess.scan_forward(board.occupied)}


# ---------------------------------------------------------------- the enumerated relations (no names, no chess)
PAIR_RELATIONS = ("safe_meets_safe", "safe_has_square", "safe_attacks", "any_meets_any", "any_has_square", "any_attacks", "safe_meets_promotion", "any_meets_promotion")
UNARY = ("safe_size", "contested_size", "safe_promotion", "any_promotion", "steps_to_enemy_king_safe")


def pair_relations(ra: Region, rb: Region) -> list[int]:
    any_a = ra.safe | ra.contested_bb
    any_b = rb.safe | rb.contested_bb
    sb = chess.BB_SQUARES[rb.square]
    prom_b = rb.promotion_squares()
    return [int(bool(ra.safe & rb.safe)), int(bool(ra.safe & sb)), int(any(ra.attacks_from[s] & sb for s in ra.steps)),
            int(bool(any_a & any_b)), int(bool(any_a & sb)), int(any(ra.attacks_from[s] & sb for s in ra.attacks_from)),
            int(bool(ra.safe & prom_b)), int(bool(any_a & prom_b))]


def unary_relations(r: Region, board: chess.Board) -> list[int]:
    ek = board.king(not r.piece.color)
    ek_zone = chess.BB_KING_ATTACKS[ek] | chess.BB_SQUARES[ek] if ek is not None else 0
    d = min((k for s, k in r.steps.items() if ek_zone & chess.BB_SQUARES[s]), default=99)
    return [chess.popcount(r.safe), len(r.contested), chess.popcount(r.safe & r.promotion_squares()), chess.popcount(r.promotion_squares()), min(d, 20)]


def relations(board: chess.Board, regs: dict[int, Region] | None = None) -> dict:
    """Every ordered pair of pieces and every piece: the relation vectors, keyed by squares (the caller names them)."""
    regs = regs if regs is not None else regions(board)
    pairs = {}
    for a, ra in regs.items():
        for b, rb in regs.items():
            if a != b:
                pairs[(a, b)] = pair_relations(ra, rb)
    return dict(pairs=pairs, unary={s: unary_relations(r, board) for s, r in regs.items()})


def describe(board: chess.Board, regs: dict[int, Region]) -> list[str]:
    out = []
    for s, r in regs.items():
        p = board.piece_at(s)
        who = f"{'White' if p.color else 'Black'} {chess.piece_name(p.piece_type)} on {chess.square_name(s)}"
        cont = sorted(r.contested, key=lambda t: r.contested[t][0])[:6]
        out.append(f"{who}: safe region {chess.popcount(r.safe)} squares ({', '.join(chess.square_name(t) for t in sorted(r.steps, key=r.steps.get)[:10])}...); "
                   f"contested {len(r.contested)} (e.g. {', '.join(chess.square_name(t) + ' by ' + '/'.join(chess.square_name(a) for a in chess.scan_forward(r.contested[t][1])) for t in cont[:3])})")
    return out


if __name__ == "__main__":
    import sys
    b = chess.Board(sys.argv[1] if len(sys.argv) > 1 else "4k3/1R6/4K3/4P3/8/8/8/5r2 b - - 0 58")
    regs = regions(b)
    for line in describe(b, regs):
        print(line)
    rel = relations(b, regs)
    for (a, c), v in rel["pairs"].items():
        if any(v):
            print(f"  {chess.square_name(a)}->{chess.square_name(c)}: " + " ".join(f"{n}={x}" for n, x in zip(PAIR_RELATIONS, v) if x))
