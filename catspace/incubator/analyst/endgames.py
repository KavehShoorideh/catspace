"""THE TABLEBASE AS THE TEACHER (Kaveh 2026-09-19): random legal positions of chosen materials (3-5 men), their
exact result (Syzygy WDL, DTZ), the enumerated region relations (regions.py; for the rooting), one random reply
child with its WDL (for the coherence loss: tokens equal across a reply when the result is equal), and KING-MOVE
SIBLINGS: positions where two king moves lead to different results (the explanation task).
Output: data/derived/endgames/<material>.jsonl rows and a joined npz. The split is by material configuration
(held-out materials test transfer) and by position hash within a material."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import chess
import chess.syzygy
import numpy as np

from .regions import regions, relations, PAIR_RELATIONS, UNARY

OUT = Path("data/derived/endgames")
TB_DIR = Path("assets/tablebases/syzygy")
MATERIALS_DEFAULT = ["KPvK", "KRvK", "KQvK", "KRPvKR", "KPvKP", "KQvKR", "KRPvK", "KPPvK", "KBPvK", "KNPvK", "KQPvKQ", "KRvKP", "KRvKB", "KRvKN", "KQvKP", "KBvKP", "KNvKP", "KPPvKP", "KRPvKP", "KRBvKR", "KRNvKR", "KQvKRP", "KBNvK", "KBBvK", "KRRvKR"]
PIECE = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}


def random_position(material: str, rng: random.Random) -> chess.Board | None:
    w, b = material.split("v")
    board = chess.Board(None)
    squares = rng.sample(range(64), len(w) + len(b))
    k = 0
    for color, side in ((chess.WHITE, w), (chess.BLACK, b)):
        for ch in side:
            sq = squares[k]; k += 1
            if ch == "P" and chess.square_rank(sq) in (0, 7):
                return None
            board.set_piece_at(sq, chess.Piece(PIECE[ch], color))
    board.turn = rng.random() < 0.5
    board.castling_rights = 0
    if not board.is_valid() or board.is_game_over():
        return None
    return board


def wdl_of(tb, board: chess.Board) -> int | None:
    try:
        return tb.probe_wdl(board)                                     # side to move: 2 win, 1 cursed win, 0 draw, -1, -2
    except (KeyError, chess.syzygy.MissingTableError):
        return None


def sign(w: int) -> int:
    return 1 if w > 0 else -1 if w < 0 else 0


def row_of(board: chess.Board, tb, rng: random.Random, material: str) -> dict | None:
    w = wdl_of(tb, board)
    if w is None:
        return None
    try:
        dtz = tb.probe_dtz(board)
    except Exception:
        dtz = None
    regs = regions(board)
    rel = relations(board, regs)
    pieces = [(sq, board.piece_at(sq).piece_type, int(board.piece_at(sq).color == board.turn)) for sq in sorted(regs)]     # 'ours' = the side to move
    # a random reply child, with its WDL in the PARENT's side-to-move view
    legal = list(board.legal_moves)
    child = None
    if legal:
        m = rng.choice(legal); nb = board.copy(stack=False); nb.push(m)
        if nb.is_game_over():
            cw = 0 if not nb.is_checkmate() else -2
            child = dict(move=m.uci(), fen=nb.fen(), wdl=-cw if False else (2 if nb.is_checkmate() else 0))   # mate: the parent's mover won
        else:
            cw = wdl_of(tb, nb)
            if cw is not None:
                child = dict(move=m.uci(), fen=nb.fen(), wdl=-cw)
    # king-move siblings: two king moves with different results (the explanation task)
    ksq = board.king(board.turn)
    kmoves = [m for m in legal if m.from_square == ksq]
    outcomes = {}
    for m in kmoves:
        nb = board.copy(stack=False); nb.push(m)
        if nb.is_game_over():
            outcomes[m.uci()] = 0 if nb.is_stalemate() or nb.is_insufficient_material() else 2
        else:
            cw = wdl_of(tb, nb)
            if cw is not None:
                outcomes[m.uci()] = -cw
    siblings = None
    if len({sign(v) for v in outcomes.values()}) > 1:
        siblings = {u: v for u, v in outcomes.items()}
    return dict(fen=board.fen(), material=material, wdl=w, dtz=dtz, pieces=pieces,
                pairs={f"{chess.square_name(a)}-{chess.square_name(b_)}": v for (a, b_), v in rel["pairs"].items()},
                unary={chess.square_name(s): v for s, v in rel["unary"].items()},
                child=child, siblings=siblings)


def generate(materials: list[str], per_material: int, seed: int = 0, out: Path = OUT, log=print):
    out.mkdir(parents=True, exist_ok=True)
    tb = chess.syzygy.open_tablebase(str(TB_DIR))
    for material in materials:
        rng = random.Random(f"{seed}:{material}")
        path = out / f"{material}.jsonl"
        n = 0; tries = 0; t0 = time.perf_counter(); sib = 0; wdls = {}
        with open(path, "w") as f:
            while n < per_material and tries < per_material * 50:
                tries += 1
                b = random_position(material, rng)
                if b is None:
                    continue
                r = row_of(b, tb, rng, material)
                if r is None:
                    continue
                f.write(json.dumps(r) + "\n"); n += 1; sib += r["siblings"] is not None; wdls[sign(r["wdl"])] = wdls.get(sign(r["wdl"]), 0) + 1
        log(f"[endgames] {material}: {n} positions in {time.perf_counter() - t0:.0f}s ({tries} tries); wdl signs {wdls}; king-move siblings {sib}", flush=True)
    tb.close()


def load(out: Path = OUT, materials=None) -> list[dict]:
    rows = []
    for p in sorted(out.glob("*.jsonl")):
        if materials and p.stem not in materials:
            continue
        with open(p) as f:
            rows.extend(json.loads(l) for l in f)
    return rows


def split_of(row: dict, held_materials: set) -> str:
    if row["material"] in held_materials:
        return "held"
    h = int(hashlib.md5(row["fen"].encode()).hexdigest()[:6], 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "stats"])
    ap.add_argument("--materials", default=",".join(MATERIALS_DEFAULT)); ap.add_argument("--per-material", type=int, default=4000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        generate(a.materials.split(","), a.per_material, a.seed)
    else:
        rows = load()
        print(len(rows), "rows;", sum(r["siblings"] is not None for r in rows), "with king-move siblings;",
              {m: sum(1 for r in rows if r["material"] == m) for m in sorted({r["material"] for r in rows})})


if __name__ == "__main__":
    main()
