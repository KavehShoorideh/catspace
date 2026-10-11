"""DIG A DISAGREEMENT (Kaveh 2026-10-10: "keep digging until you can articulate exactly what Stockfish did and why we did
not do it"). For a position, our move and Stockfish's: both lines (Stockfish depth 18, 12 plies), the material at each
ply of each line (for the side to move), the planner's net for both moves (net_of), and our planner's view after each
move (the opponent's best proved win)."""
import sys

import chess
import chess.engine

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def mat(b, c):
    return sum(VAL[p.piece_type] for p in b.piece_map().values() if p.color == c) - sum(VAL[p.piece_type] for p in b.piece_map().values() if p.color != c)


def dig(fen, ours, theirs_best, eng, plies=12):
    import analyst_core
    b = chess.Board(fen); us = b.turn
    out = []
    for name, san in (("ours", ours), ("Stockfish", theirs_best)):
        m = b.parse_san(san)
        a = b.copy(); a.push(m)
        info = eng.analyse(a, chess.engine.Limit(depth=18))
        x = a.copy(); traj = [mat(a, us)]; sans = [san]
        for mv in info.get("pv", [])[: plies - 1]:
            sans.append(x.san(mv)); x.push(mv); traj.append(mat(x, us))
        net = analyst_core.net_of(fen, m.uci())
        out.append(f"  {name:9s} {san:6s} SF {info['score'].pov(us).score(mate_score=3000):+5d}  our net {net[0]:+d}{' (they then have ' + net[1] + ')' if net[1] else ''}\n"
                   f"            line: {' '.join(sans)}\n            material for us along it: {traj}")
    return "\n".join(out)


if __name__ == "__main__":
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    print(dig(sys.argv[1], sys.argv[2], sys.argv[3], eng))
    eng.quit()
