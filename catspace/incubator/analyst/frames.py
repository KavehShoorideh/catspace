"""PROOF BY FRAME ENUMERATION (Kaveh 2026-09-22: "your method is unsound").

The contrast miner PROPOSES; nothing it reports is evidence. A perturbation of one square changes many concepts
at once, and the operator decides which concepts can ever appear (its "material never differs" was an artifact of
moving one piece, not a fact about chess). What establishes a claim is quantification over a FRAME: name the
pieces and the geometry, say what must not be present, and enumerate EVERY placement, checking the claim by the
rules. Then the statement is universal with explicit hypotheses, and the exceptions are complete, not sampled --
which is what a guard is.

The first claim, the pin, decided by the rules alone (no tablebase, so it holds at any material):
    FRAME   an enemy slider S; our piece A on a line from S with nothing between; our piece B further along the
            same line with nothing between A and B; value(B) >= value(A); no pawns among S, A, B (so the frame is
            symmetry-invariant and A's move set does not depend on which way it faces).
    CLAIM   if A moves off the line, S captures B next move and we lose at least value(B) - (what A won).
    Every placement of S, A, B and both kings is enumerated (S's square fixed to the ten canonical squares by the
    board's eightfold symmetry, which the claim respects). Each failure is classified, and the classes ARE the guard.

    python -m catspace.incubator.analyst.frames pin --slider r --a n --b q
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import chess

OUT = Path("data/derived/frames")
CANON = [chess.A1, chess.B1, chess.C1, chess.D1, chess.B2, chess.C2, chess.D2, chess.C3, chess.D3, chess.D4]
VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0, chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 100.0}
SYM = {"p": chess.PAWN, "n": chess.KNIGHT, "b": chess.BISHOP, "r": chess.ROOK, "q": chess.QUEEN, "k": chess.KING}
RAYS = {chess.ROOK: [(0, 1), (0, -1), (1, 0), (-1, 0)],
        chess.BISHOP: [(1, 1), (1, -1), (-1, 1), (-1, -1)],
        chess.QUEEN: [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]}


def _walk(sq: int, d: tuple, n: int) -> int | None:
    f, r = chess.square_file(sq) + d[0] * n, chess.square_rank(sq) + d[1] * n
    return chess.square(f, r) if 0 <= f < 8 and 0 <= r < 8 else None


def frames(slider: int, a_type: int, b_type: int):
    """Every placement of the frame: the slider on a canonical square, A and B along one of its rays with nothing
    between, both kings anywhere legal. Yields (board, s, a, b) with BLACK (the pinned side) to move."""
    for s in CANON:
        for d in RAYS[slider]:
            for i in range(1, 8):
                a = _walk(s, d, i)
                if a is None:
                    break
                for j in range(i + 1, 8):
                    b = _walk(s, d, j)
                    if b is None:
                        break
                    used = {s, a, b}
                    for wk in range(64):
                        if wk in used:
                            continue
                        for bk in range(64):
                            if bk in used or bk == wk or chess.square_distance(wk, bk) <= 1:
                                continue
                            if b_type == chess.KING and bk != b:
                                continue
                            board = chess.Board(None)
                            board.set_piece_at(s, chess.Piece(slider, chess.WHITE))
                            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                            board.set_piece_at(a, chess.Piece(a_type, chess.BLACK))
                            if b_type != chess.KING:
                                board.set_piece_at(b, chess.Piece(b_type, chess.BLACK))
                            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                            board.turn = chess.BLACK
                            if not board.is_valid() or board.is_check():
                                continue                                   # a frame where Black is already in check is a different claim
                            yield board, s, a, b
                            if b_type == chess.KING:
                                break
                        if b_type == chess.KING and bk == b:
                            pass


def pin(slider_sym: str, a_sym: str, b_sym: str, log=print, cap: int = 0):
    slider, a_type, b_type = SYM[slider_sym], SYM[a_sym], SYM[b_sym]
    assert slider in RAYS and a_type != chess.PAWN and b_type != chess.PAWN, "no pawns in the frame: the claim must be symmetry-invariant"
    assert VAL[b_type] >= VAL[a_type], "the frame requires the piece behind to be worth at least as much"
    n_frames = n_moves = holds = 0
    fails = Counter(); examples = {}
    t0 = time.perf_counter()
    for board, s, a, b in frames(slider, a_type, b_type):
        n_frames += 1
        if cap and n_frames > cap:
            break
        line = set(chess.SquareSet.between(s, b)) | {s, b}
        for m in board.legal_moves:
            if m.from_square != a or m.to_square in line:
                continue                                                   # only moves that LEAVE the line
            n_moves += 1
            nb = board.copy(stack=False); nb.push(m)
            won = VAL.get(board.piece_type_at(m.to_square), 0.0) if board.is_capture(m) else 0.0
            take = chess.Move(s, b)
            if nb.is_check():
                fails["A's move gives check"] += 1; examples.setdefault("A's move gives check", (board.fen(), board.san(m)))
                continue
            if take not in nb.legal_moves:
                # the slider cannot take: it is pinned itself, or the king would be left in check, or B moved away
                fails["the slider cannot capture B"] += 1; examples.setdefault("the slider cannot capture B", (board.fen(), board.san(m)))
                continue
            after = nb.copy(stack=False); after.push(take)
            recapture = max((VAL[slider] for x in after.legal_moves if x.to_square == b), default=0.0)
            net = won - VAL[b_type] + recapture                            # Black's material change, with the recapture if there is one
            if net < 0:
                holds += 1
            else:
                kind = ("A won at least as much" if won >= VAL[b_type] else
                        "B is defended: the slider is recaptured" if recapture else "the exchange comes out even")
                fails[kind] += 1; examples.setdefault(kind, (board.fen(), board.san(m)))
    dt = time.perf_counter() - t0
    log(f"[frames/pin] S={slider_sym.upper()} A={a_sym} B={b_sym}: {n_frames} placements of the frame, {n_moves} moves of A off the line, in {dt:.0f}s"
        + (" (CAPPED: not a proof)" if cap and n_frames > cap else " (complete over the frame)"))
    log(f"[frames/pin] THE CLAIM 'A leaves the line => we lose material' HOLDS in {holds}/{n_moves} = {100 * holds / max(1, n_moves):.2f}%")
    log("[frames/pin] the exceptions, complete -- these ARE the guard:")
    for k, c in fails.most_common():
        ex = examples.get(k)
        log(f"    {k:<44} {c:6d} = {100 * c / max(1, n_moves):5.2f}%" + (f"   e.g. {ex[1]} in {ex[0]}" if ex else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(slider=slider_sym, a=a_sym, b=b_sym, frames=n_frames, moves=n_moves, holds=holds,
                   complete=not (cap and n_frames > cap), fails={k: int(v) for k, v in fails.items()},
                   examples={k: list(v) for k, v in examples.items()}), open(OUT / f"pin_{slider_sym}{a_sym}{b_sym}.json", "w"), indent=1)
    return holds, n_moves, fails


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pin"])
    ap.add_argument("--slider", default="r"); ap.add_argument("--a", default="n"); ap.add_argument("--b", default="q")
    ap.add_argument("--cap", type=int, default=0)
    a = ap.parse_args(argv)
    pin(a.slider, a.a, a.b, cap=a.cap)


if __name__ == "__main__":
    main()
