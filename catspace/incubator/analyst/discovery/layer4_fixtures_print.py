"""PLAN 79.25 PART B: THE MAKER OF LAYER 4'S FIXTURES (SYSTEM_SPEC 11 "the layers", 4 tactical reasoning, 10 data rules, 15 standing limits; ledger row 36; piece: tactical reasoner).
Made once, by a program that reads the puzzle set (`catspace.incubator.analyst.discovery.ledger_puzzles`, the file plans 78 and 79 graded on) and asks an OUTSIDE ENGINE (Stockfish through the analyst adapter, `catspace.incubator.analyst.engine`)
as a MEASURE only: its numbers are stored in the fixture as the reference verdict and as the reason a position was kept; they are never fitted on and the search never reads them. The layer's test (rust/engine/tactics/tests/layer4.rs) reads the files
this makes as opaque text and calls no outside program. This is one of the places that names the game's objects (a maker script named `*_print.py`). No player's name or id is read, kept or printed (the puzzle's own id is a number of the puzzle set).

THE RULE OF "NOT QUIET" (stated here, written into the fixture as `rule` lines, the counts kept and dropped printed as OBS lines). A puzzle of the set is kept only if
  R1  the outside engine's best move at the puzzle's start equals the solution's first move, and it beats the engine's second move by at least MARGIN centipawns (a forced end is scored as MATE_CP less its length, so a forced end beats any count);
  R2  every move of OURS along the solution is a taking, or a threat to the chief unit (it gives check), or an ONLY MOVE (the engine's best at that position is the solution's move and beats its second by at least MARGIN).
The positions are the position the puzzle's setup move leaves (the side to move is the solver). The engine's reading at the start (its score or its forced end in moves) is stored as the REFERENCE VERDICT. Themes are the puzzle set's own labels.
A second fixture holds QUIET positions: positions of human games that layer 3's own fixture holds as quiet rows (value.fixture, nothing stands to be lost by inclusion), kept again only if the engine says nothing is forced there: its score for the side to move within QUIET_CP of nought, its best
move beating its second by less than QUIET_MARGIN, no forced end, and its best move neither a taking nor a check.

TWO FIXTURES, in rust/engine/tactics/tests/fixtures/ (tab separated lines, the first field the tag):
  puzzle.fixture
    game    <name> <declaration file under rules/>
    rule    <text>                                  the rule above with its numbers, one line a part
    puzzle  <id> <themes, space separated> <our moves in the solution> <reference verdict: m<n> a forced end for us in n moves, m-<n> against us, c<cp> a score> <margin of the best over the second, cp> <position text> <the solution: every move of both sides, space separated>
  quiet.fixture
    game    <name> <declaration file>
    rule    <text>
    quiet   <id> <reference score, cp> <margin of the best over the second, cp> <position text>

  .venv/bin/python -m catspace.incubator.analyst.discovery.layer4_fixtures_print OUT_DIR [--workers K] [--per-theme N] [--quiet-n N] [--depth D]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import io
import multiprocessing
import time
from collections import Counter, defaultdict
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import ledger_puzzles as LP
from catspace.incubator.analyst.engine import MATE_CP, Engine
from catspace.io import paths

MARGIN = 150                  # R1 and R2: the best beats the second by this many centipawns
QUIET_CP = 100                # a quiet position: the engine's score for the side to move is within this of nought
QUIET_MARGIN = 60             # ... and its best move beats its second by less than this
DEPTH = 15                    # the engine's depth (fixed: the same reading every time)
SEED = 7925
DECLARATION = "learned/chess/graph_vocabulary.json"
THEMES = ["mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5", "fork", "pin", "skewer", "discoveredAttack", "hangingPiece", "backRankMate", "trappedPiece", "sacrifice", "deflection",
          "attraction", "doubleCheck", "smotheredMate", "exposedKing", "kingsideAttack", "capturingDefender", "intermezzo", "xRayAttack", "advancedPawn", "promotion", "endgame",
          "middlegame", "opening", "short", "long", "veryLong", "crushing", "equality", "defensiveMove", "zugzwang", "interference", "clearance"]

_ENG: dict = {}


def init(depth: int) -> None:
    pid = os.getpid()
    out = Path(paths.experiment("whole_79_25", "sf_cache")); out.mkdir(parents=True, exist_ok=True)
    _ENG["e"] = Engine(threads=1, hash_mb=64, depth=depth, cache_path=str(out / f"cache_{pid}.sqlite"))
    _ENG["depth"] = depth


def two(board: chess.Board):
    """the engine's best two lines for the side to move"""
    return _ENG["e"].analyse(board, multipv=2, depth=_ENG["depth"])


def reference(line) -> str:
    if line.mate is not None:
        return f"m{line.mate}"
    return f"c{line.cp}"


def candidate_rows(per_theme: int) -> list:
    """per theme a seeded reservoir of the puzzle set's rows, clear of the held-out test sample the earlier plans graded on (row number a multiple of 200); the union, in row order"""
    import random
    import zstandard
    rng = random.Random(SEED)
    keep: dict = {t: [] for t in THEMES}
    seen = Counter()
    with open(LP._csv_path(), "rb") as fh:
        reader = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"))
        for i, row in enumerate(reader):
            if i % 200 == 0:
                continue
            themes = set((row.get("Themes") or "").split())
            for t in themes & keep.keys():
                seen[t] += 1
                if len(keep[t]) < per_theme:
                    keep[t].append((i, row))
                else:
                    j = rng.randrange(seen[t])
                    if j < per_theme:
                        keep[t][j] = (i, row)
    rows: dict = {}
    for t in THEMES:
        for i, row in keep[t]:
            rows[i] = row
    return [rows[i] for i in sorted(rows)]


def judge_puzzle(row: dict) -> dict:
    """the rule of 'not quiet' applied to one puzzle (see the file's doc); returns the verdict and why"""
    out = {"id": row["PuzzleId"], "themes": (row.get("Themes") or "").split()}
    try:
        b, sol, themes = LP._setup(row)
    except Exception as e:                                                   # a row the library cannot read
        return {**out, "keep": False, "why": "unreadable"}
    out["text"] = b.fen()
    out["solution"] = sol
    out["ours"] = (len(sol) + 1) // 2
    if len(sol) < 1:
        return {**out, "keep": False, "why": "no solution"}
    b2 = b.copy(stack=False)
    ref = None
    for k, u in enumerate(sol):
        mv = chess.Move.from_uci(u)
        if mv not in b2.legal_moves:
            return {**out, "keep": False, "why": "solution not legal"}
        if k % 2 == 0:                                                          # one of OURS
            lines = two(b2)
            if len(lines) < 2:
                return {**out, "keep": False, "why": "one legal move"}
            margin = lines[0].cp - lines[1].cp
            taking = b2.is_capture(mv)
            gives_check = b2.gives_check(mv)
            only = lines[0].move == u and margin >= MARGIN
            if k == 0:
                ref = reference(lines[0])
                if lines[0].move != u or margin < MARGIN:
                    return {**out, "keep": False, "why": "R1: the engine's best is not the solution's first move or beats its second by too little"}
                out["reference"], out["margin"] = ref, margin
            if not (taking or gives_check or only):
                return {**out, "keep": False, "why": "R2: a move of ours is no taking, no threat to the chief unit and no only move"}
        b2.push(mv)
    return {**out, "keep": True, "why": "kept"}


def judge_quiet(item: tuple) -> dict:
    sample, game, text = item
    b = chess.Board(text)
    if b.is_game_over() or not b.is_valid():
        return {"keep": False, "why": "unreadable"}
    lines = two(b)
    if len(lines) < 2:
        return {"keep": False, "why": "one legal move"}
    mv = chess.Move.from_uci(lines[0].move)
    margin = lines[0].cp - lines[1].cp
    if lines[0].mate is not None or lines[1].mate is not None:
        return {"keep": False, "why": "a forced end is read"}
    if abs(lines[0].cp) > QUIET_CP:
        return {"keep": False, "why": "the score is not near nought"}
    if margin >= QUIET_MARGIN:
        return {"keep": False, "why": "the best move stands out"}
    if b.is_capture(mv) or b.gives_check(mv):
        return {"keep": False, "why": "the best move is a taking or a threat to the chief unit"}
    return {"keep": True, "why": "kept", "id": f"{sample}-{game}-{hashlib.sha1(text.encode()).hexdigest()[:6]}", "cp": lines[0].cp, "margin": margin, "text": text}


def human_quiet_positions(n: int) -> list:
    """layer 3's own fixture's quiet rows of the human samples, spread evenly (every k-th), in file order; no player field is in that fixture"""
    fx = Path(__file__).resolve().parents[4] / "rust" / "engine" / "value" / "tests" / "fixtures" / "value.fixture"
    lines = fx.read_text().splitlines()
    human = {l.split("\t")[1] for l in lines if l.startswith("sample\t") and l.split("\t")[2] == "human"}
    pos = [l.split("\t") for l in lines if l.startswith("pos\t") and l.split("\t")[1] in human]
    step = max(1, len(pos) // n)
    return [(p[1].split()[0], p[2], p[6]) for p in pos[::step][:n]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--per-theme", type=int, default=45)
    ap.add_argument("--quiet-n", type=int, default=240)
    ap.add_argument("--depth", type=int, default=DEPTH)
    a = ap.parse_args(argv)
    obs = []

    def say(s: str) -> None:
        print(s, flush=True)
        obs.append(s)

    t0 = time.time()
    rows = candidate_rows(a.per_theme)
    say(f"OBS 79.25 maker: {len(rows)} candidate puzzles ({a.per_theme} a theme at most, {len(THEMES)} themes, row numbers a multiple of 200 left out as the held-out test sample, seed {SEED})")
    with multiprocessing.Pool(a.workers, initializer=init, initargs=(a.depth,)) as pool:
        verdicts = pool.map(judge_puzzle, rows, chunksize=4)
        why = Counter(v["why"] for v in verdicts)
        say(f"OBS 79.25 maker: the rule of not quiet (engine depth {a.depth}, margin {MARGIN} cp): kept {why['kept']} of {len(verdicts)}; dropped " + "; ".join(f"{k}: {n}" for k, n in why.most_common() if k != "kept"))
        kept = [v for v in verdicts if v["keep"]]
        quiet_in = human_quiet_positions(a.quiet_n)
        qv = pool.map(judge_quiet, quiet_in, chunksize=4)
    qwhy = Counter(v["why"] for v in qv)
    say(f"OBS 79.25 maker: quiet positions from layer 3's human samples ({len(quiet_in)} read): kept {qwhy['kept']}; dropped " + "; ".join(f"{k}: {n}" for k, n in qwhy.most_common() if k != "kept"))
    by_theme_cand, by_theme_kept = Counter(), Counter()
    for v in verdicts:
        for t in v["themes"]:
            by_theme_cand[t] += 1
            if v["keep"]:
                by_theme_kept[t] += 1
    say("OBS 79.25 maker: share kept by theme (kept of candidates): " + ", ".join(f"{t} {by_theme_kept[t]}/{by_theme_cand[t]}" for t in sorted(by_theme_cand, key=lambda t: -by_theme_kept[t])))
    by_len = Counter(v["ours"] for v in kept)
    say("OBS 79.25 maker: kept by the number of our moves in the solution: " + ", ".join(f"{k} {by_len[k]}" for k in sorted(by_len)))
    refs = Counter("forced end for us" if v["reference"].startswith("m") and not v["reference"].startswith("m-") else ("forced end against us" if v["reference"].startswith("m-") else "a score") for v in kept)
    say("OBS 79.25 maker: reference verdicts of the kept: " + ", ".join(f"{k} {n}" for k, n in refs.items()))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rule = [
        f"R1 the outside engine (depth {a.depth}) names the solution's first move best and it beats the second by at least {MARGIN} cp",
        f"R2 every move of ours in the solution is a taking, or gives check, or is an only move (the engine's best, beating its second by at least {MARGIN} cp)",
        "reference verdict: m<n> the engine reads a forced end for us in n moves, m-<n> against us, c<cp> a score for the side to move",
    ]
    with open(out / "puzzle.fixture", "w") as f:
        f.write(f"game\tchess\t{DECLARATION}\n")
        for r in rule:
            f.write(f"rule\t{r}\n")
        for v in kept:
            f.write("puzzle\t" + "\t".join([v["id"], " ".join(v["themes"]), str(v["ours"]), v["reference"], str(v["margin"]), v["text"], " ".join(v["solution"])]) + "\n")
    qrule = [f"taken from the quiet rows of layer 3's human samples; kept if the outside engine (depth {a.depth}) reads no forced end, a score within {QUIET_CP} cp of nought, a best move beating its second by less than {QUIET_MARGIN} cp, and a best move that is neither a taking nor a check"]
    with open(out / "quiet.fixture", "w") as f:
        f.write(f"game\tchess\t{DECLARATION}\n")
        for r in qrule:
            f.write(f"rule\t{r}\n")
        for v in qv:
            if v["keep"]:
                f.write(f"quiet\t{v['id']}\t{v['cp']}\t{v['margin']}\t{v['text']}\n")
    for p in ("puzzle.fixture", "quiet.fixture"):
        say(f"OBS 79.25 maker: wrote {p}: {(out / p).stat().st_size} bytes")
    say(f"OBS 79.25 maker: {time.time() - t0:.0f} s on {a.workers} workers")
    o = Path(paths.experiment("whole_79_25")); o.mkdir(parents=True, exist_ok=True)
    with open(o / "obs.txt", "a") as f:
        f.write("\n".join(obs) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
