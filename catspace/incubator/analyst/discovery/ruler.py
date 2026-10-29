"""THE SEARCH RULER (ANALYST_PLAN 46): a plain alpha-beta solver with Stockfish's move ordering, there to be MEASURED against, never to play.

`solve(fen, need, max_depth, trace, answer_uci)`: iterative deepening negamax over ALL legal moves of both sides, then a quiescence search over
captures (and promotions; the first quiescence ply also tries the quiet checks) with stand-pat. An in-check quiescence node tries every evasion
and has no stand-pat. The leaf is the material balance in the store's valuation (plan 76.5: `lemma_store.valuation()`) read from the root side, minus the root's own balance (so a score
is a GAIN); checkmate is +-MATE (the terminal GOAL entry's stake) whatever the distance; stalemate, a repeated position on the path and the 50-move clock are 0.
"forced" = the root's score at the finishing depth is >= `need` against every defence (the smallest depth at which it holds). Each root move is
searched in the null window (need-1, need), so cuts are against the target; that is also why `line` is the best-scoring line the search kept,
not a proof of optimal play. It is a search claim, not a proof: the horizon is material at a quiet leaf, so the exact labeller (Rust
`forced_moves`) stays the judge.

STOCKFISH'S MOVEPICK (src/movepick.cpp), what is TAKEN: the transposition-table (here: previous iteration's best) move first; captures ordered by
MVV-LVA (victim value, then the cheapest attacker) with a static exchange (SEE) filter, the SEE-negative captures kept for the END (after the quiet
moves); promotions; quiet checks; two killer moves per ply; the history heuristic (butterfly table colour x from x to, depth^2 on a quiet beta
cut) for the rest. In quiescence, captures by MVV-LVA, checks at the first quiescence ply only, evasions when in check.
LEFT OUT on purpose (Kaveh wants a FULL search with ordering, so every node visited is one a plain search needs): null-move pruning, late-move
reductions, futility/razoring, check extensions, singular extensions, aspiration windows, TT score cutoffs (the table is used for the move only),
continuation/capture histories. The SEE is static and ignores pins; Stockfish's `see_ge` is the same swap algorithm with the same blindness.

TRACE: the first `trace` nodes visited by the FINISHING iteration (earlier iterations only seed the ordering), in visit order, as
(ply, san path, kind in {main, quiescence}, reason, score); reason in {cut: beta, cut: stand-pat, leaf, mate, draw, repeat, searched}. The score
is from the side to move at that node. Node counts are the TOTAL over all iterations, main and quiescence apart.
Run: .venv/bin/python -m catspace.incubator.analyst.discovery.ruler --set dev_20 --index 0 --trace 200
"""
from __future__ import annotations

import argparse
import shutil

import chess

from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import names as NM

# plan 76.5: the ruler's worths are the store's valuation (the entry the GOAL entries count in) -- its own table and the import-time assertion that it
# equalled selfmine's are gone. The royal's place in the exchange ("the king counts 100": Lean Ideas.ExchangeLaw.valueAt) and a mate's score are the
# terminal GOAL entry's stake (the royal's loss ends the game: above every count).
_V = LS.valuation()
MATE = LS.goal_kinds().terminal.stake
VAL = _V.by_kind()
PAWN_WORTH = _V.pawn
SEE_VAL = {**VAL, NM.kind(_V.royal): MATE}


def material(board: chess.Board, color: bool) -> int:
    return sum(VAL[t] * len(board.pieces(t, color)) for t in VAL)


def see(board: chess.Board, move: chess.Move) -> int:
    """Static exchange on the move's target square (swap algorithm, least valuable attacker first, x-rays by removing pieces; pins ignored)."""
    to = move.to_square
    mover = board.piece_type_at(move.from_square)
    if board.is_en_passant(move):
        victim = PAWN_WORTH
    else:
        vt = board.piece_type_at(to)
        victim = VAL[vt] if vt else 0
    gain = [victim]
    occ = board.occupied ^ chess.BB_SQUARES[move.from_square]
    if board.is_en_passant(move):
        occ ^= chess.BB_SQUARES[to + (-8 if board.turn else 8)]
    on_sq = SEE_VAL[mover]
    side = not board.turn
    while True:
        atk = (board.attackers_mask(side, to, occ)) & occ
        best_sq, best_v = None, None
        for sq in chess.scan_forward(atk):
            v = SEE_VAL[board.piece_type_at(sq)]
            if best_v is None or v < best_v:
                best_sq, best_v = sq, v
        if best_sq is None:
            break
        gain.append(on_sq - gain[-1])
        on_sq = best_v
        occ ^= chess.BB_SQUARES[best_sq]
        side = not side
    for i in range(len(gain) - 1, 0, -1):
        gain[i - 1] = -max(-gain[i - 1], gain[i])
    return gain[0]


class Ruler:
    def __init__(self, board: chess.Board, need: int, trace: int, ours=None):
        self.b = board
        self.ours = ours                                  # plan 51: ours(board) -> list[Move] | (list, reason) | None at nodes where the ROOT's side moves
        self.cuts: list = []
        self.root = board.turn
        self.base = material(board, self.root) - material(board, not self.root)
        self.need = need
        self.cap = trace
        self.killers: dict[int, list] = {}
        self.history: dict[tuple, int] = {}
        self.hashmove: dict = {}
        self.nm = self.nq = 0
        self.trace: list = []
        self.sans: list = []

    # ---------------------------------------------------------------- evaluation and ordering
    def evaluate(self) -> int:
        bal = material(self.b, self.b.turn) - material(self.b, not self.b.turn)
        return bal - (self.base if self.b.turn == self.root else -self.base)

    def order(self, ply: int, hashmove, moves=None) -> list:
        b = self.b
        ks = self.killers.get(ply, [])
        keyed = []
        for m in (moves if moves is not None else b.legal_moves):
            if m == hashmove:
                keyed.append(((0, 0), m))
                continue
            cap = b.is_capture(m)
            if cap or m.promotion:
                if b.is_en_passant(m):
                    victim = PAWN_WORTH
                else:
                    vt = b.piece_type_at(m.to_square)
                    victim = VAL[vt] if vt else 0
                att = SEE_VAL[b.piece_type_at(m.from_square)]
                promo = VAL[m.promotion] - PAWN_WORTH if m.promotion else 0
                mvvlva = -(victim * 16 + promo * 16 - att)
                if cap and (victim < att) and see(b, m) < 0:
                    keyed.append(((6, mvvlva), m))
                elif cap or m.promotion == chess.QUEEN:
                    keyed.append(((1, mvvlva), m))
                else:
                    keyed.append(((2, mvvlva), m))
            elif b.gives_check(m):
                keyed.append(((3, 0), m))
            elif m in ks:
                keyed.append(((4, ks.index(m)), m))
            else:
                keyed.append(((5, -self.history.get((b.turn, m.from_square, m.to_square), 0)), m))
        keyed.sort(key=lambda x: x[0])
        return [m for _, m in keyed]

    def candidates(self, ply: int):
        """The moves to search at a main node: None = all legal moves. Only at nodes where the root's side is to move (their replies are ALWAYS all:
        a restricted max is at most the full max, so a line found is a real forced line); each restriction records a cut (ply, description, moves cut, reason)."""
        if self.ours is None or self.b.turn != self.root:
            return None
        got = self.ours(self.b)
        if got is None:
            return None
        moves, reason = got if isinstance(got, tuple) else (got, "ours")
        total = self.b.legal_moves.count()
        legal = set(self.b.legal_moves)
        moves = [m for m in moves if m in legal]
        if not moves:
            return None
        self.cuts.append((ply, f"{len(moves)} of {total} moves kept", total - len(moves), reason))
        return moves

    # ---------------------------------------------------------------- bookkeeping
    def enter(self, move, kind: str) -> int:
        """Push the move; return the trace slot (-1 when the trace is full)."""
        slot = -1
        if len(self.trace) < self.cap:
            san = self.b.san(move)
            self.sans.append(san)
            slot = len(self.trace)
            self.trace.append([len(self.sans), " ".join(self.sans), kind, "searched", 0])
        else:
            self.sans.append(None)
        self.b.push(move)
        return slot

    def leave(self, slot: int, reason: str, score: int):
        self.b.pop()
        self.sans.pop()
        if slot >= 0:
            self.trace[slot][3] = reason
            self.trace[slot][4] = score

    # ---------------------------------------------------------------- search
    def terminal(self):
        b = self.b
        if not any(True for _ in b.legal_moves):
            return (-MATE, "mate") if b.is_check() else (0, "draw")
        if b.halfmove_clock >= 100:
            return 0, "repeat"
        if b.halfmove_clock >= 4 and b.is_repetition(2):
            return 0, "repeat"
        return None

    def child(self, m, depth: int, alpha: int, beta: int, ply: int, kind: str, qply: int = 0):
        """Play m, score the child from the MOVER's side (negamax), leave; returns (score, line)."""
        slot = self.enter(m, kind)
        if kind == "main":
            self.nm += 1
        else:
            self.nq += 1
        t = self.terminal()
        if t is not None:
            s, reason, line = -t[0], t[1], []
        elif kind == "main":
            s, line = self.search(depth, -beta, -alpha, ply + 1)
            s, reason = -s, self.last_reason
        else:
            s, line, reason = -self.quiesce(-beta, -alpha, ply + 1, qply + 1), [], self.last_reason
        self.leave(slot, reason, -s)
        return s, line

    def search(self, depth: int, alpha: int, beta: int, ply: int):
        """Negamax, fail-soft. Returns (score, line); sets last_reason for the caller's trace entry."""
        b = self.b
        if depth <= 0:
            return self.quiesce(alpha, beta, ply, 0), []
        key = b._transposition_key()
        best, bestline, bestm = -10 ** 6, [], None
        reason = "searched"
        for m in self.order(ply, self.hashmove.get(key), self.candidates(ply)):
            quiet = not b.is_capture(m)
            turn = b.turn
            sc, line = self.child(m, depth - 1, alpha, beta, ply, "main")
            if sc > best:
                best, bestm, bestline = sc, m, [m] + line
            if sc > alpha:
                alpha = sc
            if sc >= beta:
                reason = "cut: beta"
                if quiet:
                    ks = self.killers.setdefault(ply, [])
                    if m in ks:
                        ks.remove(m)
                    ks.insert(0, m)
                    del ks[2:]
                    hk = (turn, m.from_square, m.to_square)
                    self.history[hk] = self.history.get(hk, 0) + depth * depth
                break
        if bestm is not None:
            self.hashmove[key] = bestm
        self.last_reason = reason
        return best, bestline

    def quiesce(self, alpha: int, beta: int, ply: int, qply: int) -> int:
        b = self.b
        in_check = b.is_check()
        if not in_check:
            stand = self.evaluate()
            if stand >= beta:
                self.last_reason = "cut: stand-pat"
                return stand
            alpha = max(alpha, stand)
            best = stand
            if qply == 0:
                cand = [m for m in b.legal_moves if b.is_capture(m) or m.promotion or b.gives_check(m)]
            else:
                cand = [m for m in b.generate_legal_captures()] + [m for m in b.legal_moves if m.promotion and not b.is_capture(m)]
            moves = self.order(ply, None, cand)
        else:
            best = -10 ** 6
            moves = self.order(ply, None)
            if not moves:
                self.last_reason = "mate"
                return -MATE
        reason = "leaf" if not in_check else "searched"
        for m in moves:
            sc, _ = self.child(m, 0, alpha, beta, ply, "quiescence", qply)
            if sc > best:
                best = sc
            if sc > alpha:
                alpha = sc
            if sc >= beta:
                reason = "cut: beta"
                break
        self.last_reason = reason
        return best


def stockfish_reference(fen: str, answer_uci: str, max_depth: int = 20):
    """Lowest depth at which real Stockfish's pv[0] is the puzzle's answer, and its node count there. None when no engine / never."""
    import chess.engine
    exe = shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"
    try:
        eng = chess.engine.SimpleEngine.popen_uci(exe)
    except Exception:
        return None
    try:
        board = chess.Board(fen)
        for d in range(1, max_depth + 1):
            info = eng.analyse(board, chess.engine.Limit(depth=d))
            pv = info.get("pv")
            if pv and pv[0].uci() == answer_uci:
                return dict(depth=d, nodes=int(info.get("nodes", 0)))
        return None
    finally:
        eng.quit()


def solve(fen: str, need: int, max_depth: int = 6, trace: int = 200, answer_uci: str | None = None, ours=None) -> dict:
    """`ours` (plan 51): ours(board) -> list[Move] | (list[Move], reason) | None, called at every main node where the root's side is to move; a list restricts
    the moves searched there (in the ruler's own order), None = all. Quiescence and their replies are unchanged. The result gains `cuts` of the finishing iteration."""
    board = chess.Board(fen)
    r = Ruler(board, need, trace, ours)
    status, line, depth = "stopped", [], 0
    final_trace: list = []
    final_cuts: list = []
    best_line: list = []
    for d in range(1, max_depth + 1):
        r.trace, r.sans, r.cuts = [], [], []
        key = board._transposition_key()
        moves = r.order(0, r.hashmove.get(key), r.candidates(0))
        if not moves:
            break
        bestsc, bestm, bestl = -10 ** 6, None, []
        found = False
        for m in moves:
            sc, line_ = r.child(m, d - 1, need - 1, need, 0, "main")
            if sc > bestsc:
                bestsc, bestm, bestl = sc, m, [m] + line_
            if sc >= need:
                found = True
                break
        r.hashmove[key] = bestm
        depth = d
        final_trace = r.trace
        final_cuts = r.cuts
        best_line = bestl
        if found:
            status = "forced"
            break
    san_line, tmp = [], chess.Board(fen)
    for m in best_line:
        san_line.append(tmp.san(m))
        tmp.push(m)
    ref = stockfish_reference(fen, answer_uci) if answer_uci else None
    return dict(status=status, line=san_line, nodes_main=r.nm, nodes_q=r.nq, nodes=r.nm + r.nq, depth=depth,
                trace=[tuple(x) for x in final_trace], stockfish=ref, cuts=final_cuts)


def format_trace(trace: list, n: int = 40) -> list:
    """'1. Kxc2 .. Qc4+ .. Kd2 leaf 62' style: the path, then reason and score."""
    out = []
    for ply, path, kind, reason, score in trace[:n]:
        sans = path.split()
        out.append(f"{ply}. " + " .. ".join(sans) + f"  [{kind}] {reason} {score}")
    return out


def main():
    from catspace.incubator.analyst.discovery.puzzle_apply import load_set
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev_20")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--id")
    ap.add_argument("--trace", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--no-stockfish", action="store_true")
    a = ap.parse_args()
    ps = load_set(a.set)
    p = next(x for x in ps if x["id"] == a.id) if a.id else ps[a.index]
    res = solve(p["fen"], int(p["value"]), a.max_depth, a.trace, None if a.no_stockfish else p["answer"])
    for line in format_trace(res["trace"], a.trace):
        print(line)
    print(f"OBS ruler {p['id']}: {res['status']}, depth {res['depth']}, nodes_main {res['nodes_main']}, nodes_q {res['nodes_q']}, line {' '.join(res['line'])}")
    sf = res["stockfish"]
    if sf:
        print(f"OBS stockfish {p['id']}: depth {sf['depth']}, nodes {sf['nodes']}")


if __name__ == "__main__":
    main()
