"""transitions.py -- the VALUE OF A REGIME TRANSITION from outcomes, conditional on the present (Kaveh 2026-09-15:
"the planner should realize it must avoid trades when down and simplify when ahead" -- and the data's answer is
what it is; no rule is coded).

For each regime coordinate (the phase = piece material on the board, the queens, the pawns, the open files, the
passed pawns), the contrast between games where it FELL by at least a step over the next 20 plies and games
where it did not, in the mover's result, within cells of the present: the material balance, and the level (the
rating band, from the Lichess corpus) or the engine's evaluation (from Kaveh's games). The planner reads the
table: a steering target whose named differences carry a positive contrast in the present cell is supported,
a negative one is an AVOID. Nothing chess-specific beyond the rules' scale; any regime coordinate can be added.

    python -m catspace.incubator.analyst.transitions fit
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess
import numpy as np

OUT = Path("data/derived/transitions.json")
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
KEYS = ("phase", "queens_on", "pawns_ours", "pawns_theirs", "open_files", "passed_ours", "passed_theirs")
STEP = {"phase": 6.0, "queens_on": 1.0, "pawns_ours": 1.0, "pawns_theirs": 1.0, "open_files": 1.0, "passed_ours": 1.0, "passed_theirs": 1.0}
MAT_BINS = (("behind <= -5", -99, -4.5), ("behind -4..-2", -4.5, -1.5), ("even", -1.5, 1.5), ("ahead 2..4", 1.5, 4.5), ("ahead >= 5", 4.5, 99))
ELO_BANDS = (("<1400", 0, 1400), ("1400-1700", 1400, 1700), ("1700-2000", 1700, 2000), (">=2000", 2000, 9999))


def coords(board: chess.Board, us: chess.Color) -> dict:
    out = {"phase": 0.0, "queens_on": 0.0, "pawns_ours": 0.0, "pawns_theirs": 0.0, "open_files": 0.0, "passed_ours": 0.0, "passed_theirs": 0.0, "material": 0.0}
    for sq, p in board.piece_map().items():
        if p.piece_type == chess.KING:
            continue
        v = VAL[p.piece_type]; out["material"] += v if p.color == us else -v
        if p.piece_type != chess.PAWN:
            out["phase"] += v
        if p.piece_type == chess.QUEEN:
            out["queens_on"] += 1
    for c, tag in ((us, "ours"), (not us, "theirs")):
        pawns = list(board.pieces(chess.PAWN, c)); out[f"pawns_{tag}"] = len(pawns)
        for sq in pawns:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            ahead = [t for t in board.pieces(chess.PAWN, not c) if abs(chess.square_file(t) - f) <= 1 and ((chess.square_rank(t) > r) if c == chess.WHITE else (chess.square_rank(t) < r))]
            out[f"passed_{tag}"] += 0 if ahead else 1
    allp = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
    out["open_files"] = sum(1 for f in range(8) if not any(chess.square_file(t) == f for t in allp))
    return out


def contrast(rows, cell_of, min_n: int = 25) -> dict:
    """rows: (cell, coords_now, coords_later, result). Per cell and key: the mean result when the key fell by >=
    a step against when it did not; the difference and the counts."""
    cells = {}
    for cell, c0, c1, res in rows:
        cells.setdefault(cell, []).append((c0, c1, res))
    table = {}
    for cell, rs in cells.items():
        t = {}
        for k in KEYS:
            fell = [r for c0, c1, r in rs if c0[k] - c1[k] >= STEP[k]]; kept = [r for c0, c1, r in rs if c0[k] - c1[k] < STEP[k] * 0.5]
            if len(fell) >= min_n and len(kept) >= min_n:
                t[k] = dict(fell=float(np.mean(fell)), kept=float(np.mean(kept)), diff=float(np.mean(fell) - np.mean(kept)), n_fell=len(fell), n_kept=len(kept))
        table[cell] = t
    return table


def fit(lichess: str = "data/derived/reach_qm/games.jsonl", games_meta: str = "data/derived/knowledge", n_lichess: int = 60000, seed: int = 0, log=print):
    rng = random.Random(seed)
    # Lichess by rating band and material (even trades only: the material balance kept within a pawn)
    rows = []
    for gi, line in enumerate(open(lichess)):
        if gi >= n_lichess:
            break
        g = json.loads(line); res = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(g["result"])
        if res is None or len(g["moves"]) < 46:
            continue
        elo = (g["elo_w"] + g["elo_b"]) / 2; band = next(n for n, lo, hi in ELO_BANDS if lo <= elo < hi)
        boards = [chess.Board()]
        try:
            for u in g["moves"]:
                b = boards[-1].copy(stack=False); b.push_uci(u); boards.append(b)
        except Exception:
            continue
        for p in rng.sample(range(16, len(boards) - 21), min(3, len(boards) - 37)):
            b0, b1 = boards[p], boards[p + 20]; us = b0.turn
            c0, c1 = coords(b0, us), coords(b1, us)
            if abs(c1["material"] - c0["material"]) > 1.0:
                continue
            mb = next(n for n, lo, hi in MAT_BINS if lo <= c0["material"] < hi)
            rows.append((f"lichess {band} | {mb}", c0, c1, res if us == chess.WHITE else 1 - res))
        if gi % 10000 == 0 and gi:
            log(f"[transitions] {gi} Lichess games, {len(rows)} rows", flush=True)
    table = contrast(rows, None)
    # Kaveh's games by material and the engine's evaluation
    from .dataset import load
    meta = load(games_meta, meta_only=True)["meta"]
    rows2 = []
    for m in meta:
        if m["result"] is None or m["ended_within"]["20"] or m["best_cp"] is None:
            continue
        b0 = chess.Board(m["fen"]); us = b0.turn; c0 = coords(b0, us)
        a = m["ahead"]["20"]
        from .dataset import REGIME_KEYS
        c1 = dict(c0); c1["phase"] = a[REGIME_KEYS.index("phase")]; c1["queens_on"] = a[REGIME_KEYS.index("queens_on")]; c1["pawns_ours"] = a[REGIME_KEYS.index("pawns_ours")]; c1["pawns_theirs"] = a[REGIME_KEYS.index("pawns_theirs")]
        c1["open_files"] = a[REGIME_KEYS.index("open_files")]; c1["passed_ours"] = a[REGIME_KEYS.index("passed_ours")]; c1["passed_theirs"] = a[REGIME_KEYS.index("passed_theirs")]; c1["material"] = a[REGIME_KEYS.index("material_diff")]
        if abs(c1["material"] - c0["material"]) > 1.0:
            continue
        mb = next(n for n, lo, hi in MAT_BINS if lo <= c0["material"] < hi)
        cp = m["best_cp"]; eb = "losing" if cp <= -200 else "worse" if cp <= -60 else "level" if cp < 60 else "better" if cp < 200 else "winning"
        rows2.append((f"kaveh {eb} | {mb}", c0, c1, m["result"]))
    table.update(contrast(rows2, None))
    json.dump(dict(keys=KEYS, step=STEP, table=table), open(OUT, "w"), indent=1)
    log(f"[transitions] {len(rows)} Lichess rows, {len(rows2)} of Kaveh's; {len(table)} cells -> {OUT}", flush=True)
    return table


def lookup(table: dict, cell: str, key: str) -> dict | None:
    return table.get(cell, {}).get(key)


def cell_for(board: chess.Board, cp: int | None, elo: float | None) -> list[str]:
    """The cells that apply to this position: Kaveh's by the evaluation when given, Lichess by the band when given."""
    c0 = coords(board, board.turn)
    mb = next(n for n, lo, hi in MAT_BINS if lo <= c0["material"] < hi)
    out = []
    if cp is not None:
        eb = "losing" if cp <= -200 else "worse" if cp <= -60 else "level" if cp < 60 else "better" if cp < 200 else "winning"
        out.append(f"kaveh {eb} | {mb}")
    if elo is not None:
        band = next(n for n, lo, hi in ELO_BANDS if lo <= elo < hi); out.append(f"lichess {band} | {mb}")
    return out


def report(path: Path = OUT) -> str:
    t = json.load(open(path))["table"]
    out = [f"[transitions] the result when a coordinate FELL over 20 plies minus when it did not (even trades), by cell; n in brackets"]
    for cell in sorted(t):
        bits = [f"{k} {v['diff']:+.3f} ({v['n_fell']}/{v['n_kept']})" for k, v in t[cell].items() if k in ("phase", "queens_on", "pawns_ours", "open_files")]
        if bits:
            out.append(f"  {cell:<34} " + "; ".join(bits))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["fit", "report"])
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        fit(); print(report())
    else:
        print(report())


if __name__ == "__main__":
    main()
