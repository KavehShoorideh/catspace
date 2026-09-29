"""THE ZONE SEARCH (Kaveh 2026-09-28: "I find it hard to believe that going like 6 plies is hard ... humans go 6 plies
(takes takes takes!) without trouble. Either we aren't focusing on the right frames, or counting plies in the most
important frame is too slow"). Prototype. Alpha-beta several plies deep inside the ZONE of an idea -- the squares of
the frames its move touches, growing as the line touches new squares and as a move's threats name new targets -- with
the moves that can matter there: captures, checks, evasions, promotions, moves of zone pieces or onto zone squares, and
moves after which the moved piece bears on the zone (the frame interface). Standing pat is passing: minus what the
opponent then wins in the zone (their threats count; promotions included). Leaves read only the zone's frames, not the
whole board -- which is why it is cheap: Kaveh's Kf5 refutation (Rxe5+ Kxe5 Rc5+ Ke6 Rxh5) in about 300 nodes.

    python -m catspace.incubator.analyst.zone_search 8
"""
import sys, time
import chess
from . import fates as F
from .frame_goals import frame_on

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
MATE = 1000.0
nodes = [0]


def cap(b, m):
    return F.captured_value(b, m)


def gains(b, zone):
    """What the side to move can win now in the zone: a frame on an enemy piece there, or a promotion that survives
    (its gain less what the opponent then wins on the promotion square)."""
    best = 0.0
    for s in zone:
        pc = b.piece_at(s)
        if pc is not None and pc.color != b.turn and pc.piece_type != chess.KING:
            best = max(best, frame_on(b, s)[0])
    for m in b.generate_legal_moves(from_mask=b.pawns & b.occupied_co[b.turn]):
        if m.promotion == chess.QUEEN:
            b.push(m); lost = frame_on(b, m.to_square)[0] if not b.is_checkmate() else 0.0; b.pop()
            best = max(best, cap(b, m) - max(0.0, lost))
    return best


def stand(b, zone):
    """Standing pat is passing: minus what the opponent wins in the zone if we pass (their threats count)."""
    if b.is_check():
        return None
    pb = F.passed(b)
    return 0.0 if pb is None else -gains(pb, zone)


def _att(pt, color, sq, occ):
    if pt == chess.PAWN:
        return chess.BB_PAWN_ATTACKS[color][sq]
    if pt == chess.KNIGHT:
        return chess.BB_KNIGHT_ATTACKS[sq]
    if pt == chess.KING:
        return chess.BB_KING_ATTACKS[sq]
    m = 0
    if pt in (chess.BISHOP, chess.QUEEN):
        m |= chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]
    if pt in (chess.ROOK, chess.QUEEN):
        m |= chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ] | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ]
    return m


def moves_in_zone(b, zone):
    if b.is_check():
        return list(b.legal_moves)
    zm = 0
    for s in zone:
        zm |= chess.BB_SQUARES[s]
    out = []
    for m in b.legal_moves:
        pt = b.piece_type_at(m.from_square)
        occ = (b.occupied & ~chess.BB_SQUARES[m.from_square]) | chess.BB_SQUARES[m.to_square]
        bears = _att(pt, b.turn, m.to_square, occ) & zm                   # the frame interface: it comes to bear on the zone
        if b.is_capture(m) or m.promotion or chess.BB_SQUARES[m.from_square] & zm or chess.BB_SQUARES[m.to_square] & zm or bears or b.gives_check(m):
            out.append(m)
    return out


def search(b, zone, depth, alpha, beta, line_out):
    nodes[0] += 1
    if b.is_checkmate():
        return -MATE
    sp = stand(b, zone)
    if depth == 0:
        return sp if sp is not None else -gains(b, zone)
    best = sp if sp is not None else -1e9
    if sp is not None:
        alpha = max(alpha, sp)
        if alpha >= beta:
            return sp
    ms = moves_in_zone(b, zone)
    ms.sort(key=lambda m: -(cap(b, m) + (5 if b.gives_check(m) else 0)))
    best_line = []
    for m in ms:
        g = cap(b, m)
        b.push(m)
        threatened = {t for t in chess.SquareSet(b.attacks_mask(m.to_square)) if b.piece_at(t) is not None and b.piece_at(t).color == b.turn}
        nz = zone | {m.from_square, m.to_square} | threatened            # a threat's target joins: its replies are searched
        sub = []
        v = g - search(b, nz, depth - 1, -beta + g, -alpha + g, sub)
        b.pop()
        if v > best:
            best = v; best_line = [m] + sub
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    line_out[:] = best_line
    return best if best > -1e9 else 0.0


def zone_of(b, m):
    """The zone of a candidate move: the squares of the frames it touches (its from/to, the pieces bearing on them),
    the kings' squares."""
    z = {m.from_square, m.to_square}
    for s in list(z):
        z |= F.square_frame(b, s)
    for c in (chess.WHITE, chess.BLACK):
        k = b.king(c)
        if k is not None:
            z.add(k)
    return z


if __name__ == "__main__":
    depth = int(sys.argv[1])
    for pid, fen, cands in [("0KCNf", "2R5/6pk/4P1pp/Pn1K4/2p3rP/8/8/8 w - - 4 45", ["e7", "Rxc4"]),
                            ("0M4g4", "8/k2P4/N1n1N1p1/4p3/4P1rp/8/8/3K4 w - - 3 46", ["Nb4", "d8=Q"]),
                            ("Kaveh", "6k1/6p1/8/2r1P2Q/6K1/3P2P1/2r5/8 w - - 1 38", ["Kf5", "Qe8+", "Kf4"]),
                            ("1HnWH", "8/4N1pk/7p/1p3R2/q3n3/4P2P/2r1Q1P1/6K1 w - - 2 41", ["Qg4", "Qxb5"])]:
        b = chess.Board(fen)
        for san in cands:
            m = b.parse_san(san); z = zone_of(b, m)
            nodes[0] = 0; t = time.perf_counter()
            g = cap(b, m); b.push(m); line = []
            v = g - search(b, z | {m.to_square}, depth - 1, -1e9, 1e9, line); b.pop()
            bb = b.copy(); bb.push(m); ls = []
            for x in line:
                ls.append(bb.san(x)); bb.push(x)
            print(f"{pid} {san}: zone search {depth} plies -> {v:+g}  line {san} {' '.join(ls)}  ({nodes[0]} nodes, {time.perf_counter()-t:.1f}s, zone {len(z)} squares)")
