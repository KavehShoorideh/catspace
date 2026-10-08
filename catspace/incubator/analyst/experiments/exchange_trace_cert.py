"""Python mirror of the kernel's settledWithin (forcedWithinB over settledB) that prints the refuting line."""
import sys, chess
from catspace.incubator.analyst.exchange_law import see, participants, cheapest_capture, material

def settled(b, c, s, target):
    if b.turn == c: return "our move"
    if b.is_checkmate(): return True
    if material(b, c) < target: return f"material {material(b, c)} < {target}"
    for u, pc in b.piece_map().items():
        if pc.color == c and pc.piece_type != chess.KING and b.attackers(b.turn, u):
            pays = see(b, u)[0]
            if (u == s and pays > 0) or (u != s and pays > material(b, c) - target):
                return f"their capture on {chess.square_name(u)} pays {pays:+d} (slack {material(b, c) - target})"
    return True

def forced(b, c, goal, n, line):
    g = goal(b)
    if g is True: return True, line
    if n == 0: return False, line + [f"<{g}>"]
    moves = list(b.legal_moves)
    if b.turn == c:
        first = None
        for m in moves:
            nb = b.copy(stack=False); nb.push(m)
            ok, l = forced(nb, c, goal, n - 1, line + [m.uci()])
            if ok: return True, l
            first = first or l
        return False, first or line + ["no move"]
    if not moves: return False, line + ["no reply"]
    for m in moves:
        nb = b.copy(stack=False); nb.push(m)
        ok, l = forced(nb, c, goal, n - 1, line + [m.uci()])
        if not ok: return False, l
    return True, line

for fen, sq in [("8/3r1pk1/3R2p1/pNpP2Pp/6nP/1PK5/P7/8 b - - 2 33", "d6"), ("r6r/p1p1kp2/2pp1N2/4p1pp/4P3/2PP2B1/P1P2P2/R4RK1 w - - 2 19", "h5"),
                ("r2qk1nr/p1p2ppp/2ppb3/4p3/1b2P3/2N2N1P/PPPP1PP1/R1BQ1RK1 b kq - 2 7", "c3")]:
    b = chess.Board(fen); s = chess.parse_square(sq); c = b.turn
    A, D = participants(b, s); m0 = cheapest_capture(b, s, A); n = len(A) + len(D)
    v = see(b, s)[0]; target = material(b, c) + v
    b1 = b.copy(stack=False); b1.push(m0)
    ok, line = forced(b1, c, lambda q: settled(q, c, s, target), n, [m0.uci()])
    print(f"{sq} value {v:+d} n {n}: {'SETTLED' if ok else 'NOT'}; line {line}")
