"""gauge_puzzles.py -- do L3/L4 NAME the tactic that the Lichess puzzle themes name? (S0 gauge)
Themes are free labels (millions of puzzles). Solver position = FEN after the opponent's first
move; solution = Moves[1:]. A theme counts as NAMED when the corresponding relation, owned by the
solver's colour (victim-side for hanging/overloaded), is present on the solver position or is
CREATED within the first 3 plies of the solution's effects. Beside every recall we print the
DEGENERATE BASELINE: the same detector's fire rate on puzzles that do NOT carry the theme.
Engine themes (mateInN, zugzwang) use Stockfish on smaller samples.
    python -m catspace.incubator.analyst.gauge_puzzles [--n 5000 --engine-n 80 --depth 12]"""
from __future__ import annotations

import argparse
import csv
import io
import time

import chess
import zstandard

from catspace.io import paths
from .relations import relations
from .dynamics import line_effects, zugzwang
from .engine import Engine

THEME_TO_REL = {"fork": ("fork",), "pin": ("pin",), "skewer": ("skewer",),
                "discoveredAttack": ("discovered_attack", "battery"), "hangingPiece": ("hanging",),
                "xRayAttack": ("xray",)}
VICTIM_SIDE = ("hanging", "loose", "overloaded")


def solver_position(row):
    b = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    b.push_uci(moves[0])
    return b, [chess.Move.from_uci(u) for u in moves[1:]]


def detected(board: chess.Board, solution: list[chess.Move]) -> set[str]:
    solver = board.turn
    names = set()
    for r in relations(board):
        if (r.owner != solver) if r.name in VICTIM_SIDE else (r.owner == solver):
            names.add(r.name)
    for e in line_effects(board, solution, 3):
        for r in e.created:
            if (r.owner != solver) if r.name in VICTIM_SIDE else (r.owner == solver):
                names.add(r.name)
    return names


def sample(path: str, n: int, stride: int = 200):
    out = []
    with open(path, "rb") as fh:
        reader = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"))
        for i, row in enumerate(reader):
            if i % stride == 0:
                out.append(row)
                if len(out) >= n:
                    break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000); ap.add_argument("--engine-n", type=int, default=80)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--db", default=str(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst"))
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    rows = sample(a.db, a.n)
    print(f"[gauge] {len(rows)} puzzles sampled in {time.perf_counter()-t0:.0f}s")
    hits = {t: [0, 0] for t in THEME_TO_REL}       # [named among theme, n theme]
    fires = {t: [0, 0] for t in THEME_TO_REL}      # [named among NON-theme, n non-theme]
    t1 = time.perf_counter()
    for row in rows:
        try:
            b, sol = solver_position(row)
            names = detected(b, sol)
        except Exception:
            continue
        themes = set(row["Themes"].split())
        for t, rels in THEME_TO_REL.items():
            fired = any(r in names for r in rels)
            tgt = hits if t in themes else fires
            tgt[t][0] += int(fired); tgt[t][1] += 1
    print(f"[gauge] L3/L4 detectors over {len(rows)} puzzles in {time.perf_counter()-t1:.0f}s")
    for t in THEME_TO_REL:
        h, f = hits[t], fires[t]
        rec = h[0] / max(h[1], 1); fr = f[0] / max(f[1], 1)
        print(f"VERDICT {t:17s} recall {rec:.2f} (n={h[1]})  | fire-rate on non-{t} {fr:.2f} (n={f[1]})  | lift x{rec/max(fr,1e-9):.1f}")
    # ---- engine themes ----
    eng = Engine(depth=a.depth, cache_path=str(paths.derived("analyst_gauge_cache.sqlite")))
    def mate_in(b, k): l = eng.best(b); return bool(l and l.mate == k)
    def zz(b): z = zugzwang(b, eng); return bool(z and z.flagged)
    checks = {"mateIn1": lambda b: mate_in(b, 1), "mateIn2": lambda b: mate_in(b, 2), "zugzwang": zz}
    t2 = time.perf_counter()
    for t, fn in checks.items():
        pos = [r for r in rows if t in r["Themes"].split()][:a.engine_n]
        neg = [r for r in rows if t not in r["Themes"].split()][:a.engine_n]
        def rate(rs):
            k = n = 0
            for r in rs:
                try: b, _ = solver_position(r); k += int(fn(b)); n += 1
                except Exception: pass
            return k / max(n, 1), n
        rp, npos = rate(pos); rn, nneg = rate(neg)
        print(f"VERDICT {t:17s} recall {rp:.2f} (n={npos})  | fire-rate on non-{t} {rn:.2f} (n={nneg})  | lift x{rp/max(rn,1e-9):.1f}")
    print(f"[gauge] engine themes in {time.perf_counter()-t2:.0f}s ({eng.calls} calls); total {time.perf_counter()-t0:.0f}s")
    eng.close()


if __name__ == "__main__":
    main()
