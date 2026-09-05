"""gauge_puzzles.py -- do L3/L4 NAME the tactic that the Lichess puzzle themes name? (S0 gauge)
Themes are free labels (millions of puzzles). Solver position = FEN after the opponent's first
move; solution = Moves[1:]. A theme counts as NAMED when the corresponding relation, owned by the
solver's colour (victim-side for hanging/overloaded), is present on the solver position or is
CREATED within the first 3 plies of the solution's effects. Output lines are OBS (observations from ONE
run on ONE sample -- directional guides, not verdicts; Kaveh 2026-09-04). Beside every recall we print the
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
from .sequences import find_sequences
SEQ_THEMES = {"deflection": ("deflection",), "attraction": ("attraction",), "interference": ("interference",),
              "clearance": ("clearance",), "capturingDefender": ("removal_of_guard",)}

THEME_TO_REL = {"fork": ("fork",), "pin": ("pin",), "skewer": ("skewer",),
                "discoveredAttack": ("discovered_attack", "battery"), "hangingPiece": ("hanging",),
                "xRayAttack": ("xray",)}
VICTIM_SIDE = ("hanging", "loose", "overloaded")


def solver_position(row):
    b = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    b.push_uci(moves[0])
    return b, [chess.Move.from_uci(u) for u in moves[1:]]


def detected(board: chess.Board, solution: list[chess.Move]) -> tuple[set[str], set[str]]:
    """(static-or-created, created-only) relation names credited to the solver. 'hanging' is
    credited only when the solution actually CAPTURES a hanging piece (static presence of a
    hanging piece is near-universal in tactical positions -- the baseline column showed 0.74)."""
    solver = board.turn
    static, created = set(), set()
    hanging_now = {r.squares[0] for r in relations(board) if r.name == "hanging" and r.owner != solver}
    for r in relations(board):
        if r.name == "hanging":
            continue
        if (r.owner != solver) if r.name in VICTIM_SIDE else (r.owner == solver):
            static.add(r.name)
    t = board.copy(stack=False)
    for i, e in enumerate(line_effects(board, solution, 3)):
        mv = solution[i]
        if t.turn == solver and e.captured and mv.to_square in hanging_now:
            created.add("hanging")
        for r in e.created:
            if r.name == "hanging":
                if r.owner != solver:
                    hanging_now.add(r.squares[0])
                continue
            if (r.owner != solver) if r.name in VICTIM_SIDE else (r.owner == solver):
                created.add(r.name)
        t.push(mv)
    return static | created, created


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
    stats = {m: {t: [[0, 0], [0, 0]] for t in THEME_TO_REL} for m in ("static+created", "created-only")}
    t1 = time.perf_counter()
    for row in rows:
        try:
            b, sol = solver_position(row)
            both, created = detected(b, sol)
        except Exception:
            continue
        themes = set(row["Themes"].split())
        for mode, names in (("static+created", both), ("created-only", created)):
            for t, rels in THEME_TO_REL.items():
                fired = any(r in names for r in rels)
                cell = stats[mode][t][0 if t in themes else 1]
                cell[0] += int(fired); cell[1] += 1
    print(f"[gauge] L3/L4 detectors over {len(rows)} puzzles in {time.perf_counter()-t1:.0f}s")
    for mode in stats:
        for t in THEME_TO_REL:
            h, f = stats[mode][t]
            rec = h[0] / max(h[1], 1); fr = f[0] / max(f[1], 1)
            print(f"OBS [{mode:14s}] {t:17s} recall {rec:.2f} (n={h[1]})  | fire-rate on non-{t} {fr:.2f} (n={f[1]})  | lift x{rec/max(fr,1e-9):.1f}")
    # ---- sequence motifs (engine-free): found in the first 3 plies of the solution ----
    sq = {t: [[0, 0], [0, 0]] for t in SEQ_THEMES}
    for row in rows:
        try:
            b, sol = solver_position(row); names = {x.name for x in find_sequences(b, sol, 3)}
        except Exception:
            continue
        themes = set(row["Themes"].split())
        for t, ns in SEQ_THEMES.items():
            cell = sq[t][0 if t in themes else 1]; cell[0] += int(any(n in names for n in ns)); cell[1] += 1
    for t in SEQ_THEMES:
        h, f = sq[t]; rec = h[0] / max(h[1], 1); fr = f[0] / max(f[1], 1)
        print(f"OBS [sequence      ] {t:17s} recall {rec:.2f} (n={h[1]})  | fire-rate on non-{t} {fr:.2f} (n={f[1]})  | lift x{rec/max(fr,1e-9):.1f}")
    # ---- engine themes ----
    eng = Engine(depth=a.depth, cache_path=str(paths.derived("analyst_gauge_cache.sqlite")))
    def mate_in(b, k): l = eng.best(b); return bool(l and l.mate == k)
    def zz(b):
        return False   # replaced below: zugzwang is tested on the position AFTER the solver's move
    checks = {"mateIn1": lambda b: mate_in(b, 1), "mateIn2": lambda b: mate_in(b, 2), "zugzwang": zz}
    t2 = time.perf_counter()
    for t, fn in checks.items():
        pos = [r for r in rows if t in r["Themes"].split()][:a.engine_n]
        neg = [r for r in rows if t not in r["Themes"].split()][:a.engine_n]
        def rate(rs):
            k = n = 0
            for r in rs:
                try:
                    b, sol = solver_position(r)
                    if t == "zugzwang":
                        b.push(sol[0]); z = zugzwang(b, eng); fired = bool(z and z.flagged)
                    else:
                        fired = fn(b)
                    k += int(fired); n += 1
                except Exception:
                    pass
            return k / max(n, 1), n
        rp, npos = rate(pos); rn, nneg = rate(neg)
        print(f"OBS {t:17s} recall {rp:.2f} (n={npos})  | fire-rate on non-{t} {rn:.2f} (n={nneg})  | lift x{rp/max(rn,1e-9):.1f}")
    print(f"[gauge] engine themes in {time.perf_counter()-t2:.0f}s ({eng.calls} calls); total {time.perf_counter()-t0:.0f}s")
    eng.close()


if __name__ == "__main__":
    main()
