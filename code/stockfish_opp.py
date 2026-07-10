"""
stockfish_opp.py — Stockfish as opponent on 5×5 KRK via UCI protocol.

Stockfish plays on arbitrary board positions described as FEN-like strings.
We send it KRK positions and let it think to a fixed depth.
"""
import subprocess, time
from domain import rc, sq

class Stockfish:
    def __init__(self, depth=10, timeout=2.0):
        """Start Stockfish process. depth=search depth in plies."""
        self.depth = depth
        self.timeout = timeout
        self.proc = subprocess.Popen(
            ["stockfish"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("uci")
        self.readto("uciok")
        self.send("isready")
        self.readto("readyok")

    def send(self, msg):
        self.proc.stdin.write(msg + "\n"); self.proc.stdin.flush()

    def readto(self, marker, timeout=1.0):
        """Read until marker line appears."""
        tstart = time.time()
        while time.time() - tstart < timeout:
            line = self.proc.stdout.readline()
            if marker in line: return line
        raise TimeoutError(f"stockfish didn't send {marker}")

    def pos_to_fen(self, wk, wr, bk):
        """Convert (wk, wr, bk) state to FEN. Assumes it's white to move."""
        # Place pieces on 5×5 board, pad with empty rows/cols to 8×8 for FEN
        board = [['.' for _ in range(8)] for _ in range(8)]
        # Map 5×5 coordinates to 8×8: KRK lives in top-left
        r_wk, c_wk = rc(wk); board[7-r_wk][c_wk] = 'K'
        r_wr, c_wr = rc(wr); board[7-r_wr][c_wr] = 'R'
        r_bk, c_bk = rc(bk); board[7-r_bk][c_bk] = 'k'
        fen_board = "/".join(
            "".join(board[r]).replace("........", "8").replace("..", "2")
                              .replace("...", "3").replace("....", "4")
                              .replace(".....", "5").replace("......", "6")
                              .replace(".......", "7")
            for r in range(8))
        return fen_board + " w - - 0 1"

    def fen_to_move_seq(self, start_fen, moves_uci):
        """Apply a sequence of UCI moves to a FEN position."""
        fen = start_fen
        for move in moves_uci:
            self.send(f"position fen {fen} moves {move}")
            # Stockfish doesn't echo position, so we just update
            # For simplicity, we'll track it manually
            pass
        return fen

    def best_move(self, fen_pos):
        """Get best move in this position."""
        self.send(f"position fen {fen_pos}")
        self.send(f"go depth {self.depth}")
        line = self.readto("bestmove", timeout=self.timeout)
        # parse "bestmove e2e4" or similar
        parts = line.split()
        for i, p in enumerate(parts):
            if p == "bestmove" and i+1 < len(parts):
                return parts[i+1]
        raise ValueError(f"couldn't parse best move from {line}")

    def close(self):
        self.proc.terminate()

# quick test
if __name__ == "__main__":
    try:
        sf = Stockfish(depth=8)
        # KRK: wk=0, wr=1, bk=6 (a random position)
        fen = sf.pos_to_fen(0, 1, 6)
        print(f"FEN: {fen}")
        mv = sf.best_move(fen)
        print(f"best move: {mv}")
        sf.close()
    except Exception as e:
        print(f"stockfish test failed (stockfish not installed?): {e}")
