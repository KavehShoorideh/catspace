"""plan 76.0 (b) CROSS-PLAY: the rule spec (rules/chess.gdl, read by discovery/gdl.py) against the fast kernels (python-chess, the kernel `Game.terminal` reads; the Rust
core's `analyst_core.Pos`, shakmaty) on a sample of positions. At every position: the legal move set, the next state after EVERY legal move, the terminal reasons, the
terminal verdict, the goal values. Kaveh 2026-10-28: "if there is a disagreement somewhere, that's a surprise. We look at that disagreement and say what do we need to add
or subtract". Every disagreement is printed with its class; one per (kernel, class, signature) is SHRUNK (units removed, castling rights / the e.p. square / the clock
cleared, while the same disagreement persists and the board stays valid) and FILED under rules/surprises/ as a case the tests replay forever.

  .venv/bin/python scripts/crossplay.py run [--games 40] [--seed 76] [--jobs 11]   # the sample; OBS lines; new surprises filed (status "open")
  .venv/bin/python scripts/crossplay.py replay                                     # every filed surprise against both kernels (what the test does)
  .venv/bin/python scripts/crossplay.py fen "<fen>"                                # one position's comparison, printed

THE SAMPLE (named, with its reason): (1) EVERY distinct position of the 47 puzzle sets (data/derived/ledger_puzzles_*.json: the whole sets, no pick); (2) the rule
CORNERS (trace_identity.CORNERS: castling, promotions, en passant, disambiguation, mates, evasions; and RULE_CORNERS below: the draws, the clock at its limit, castling
through an attacked square, a right lost to a capture); (3) RANDOM PLAY-OUTS: `--games` games from the initial position, every move uniformly random among the legal
ones (seeded), played until the kernel's terminal (`Game.terminal`: mate, stalemate, insufficient material, the fifty-move rule -- the rules end every game, no ply cap),
EVERY position of every game taken. The size: with zero disagreements in N positions the per-position disagreement rate is below 3/N at 95% (the rule of three); the
puzzle sets are middlegames, the play-outs bring the openings' castling and the endgames' draws, the corners the rules a random sample rarely meets.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import chess

import analyst_core as AC
from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import gdl_chess as GC
from catspace.incubator.analyst.discovery import certify as CF
from catspace.incubator.analyst.discovery.game import CHESS
from catspace.io import paths

SURPRISES = GC.SURPRISES
PUZZLE_GLOB = "ledger_puzzles_*.json"

RULE_CORNERS = CF.RULE_CORNERS                      # (plan 77.1 (d) (2): the comparison is the certify stage's -- discovery/certify.py; this script samples, shrinks and files)


# ------------------------------------------------------------------------------------------------------------------------------------------ one position

compare = CF.kernels_against_spec
_move_kind = CF.move_kind
_cover_move = CF.cover_move


def _expected_rust_draw(ns, prog) -> "str | None":
    return GC.expected_rust_draw(ns, prog)


def _material(state) -> str:
    return GC.material_signature(state)


# ------------------------------------------------------------------------------------------------------------------------------------------ the sample

def puzzle_fens() -> list:
    """Every distinct position of the puzzle sets (each set's file, its order; duplicates dropped)."""
    out, seen = [], set()
    for f in sorted(paths.derived_dir().glob(PUZZLE_GLOB)):
        d = json.load(open(f))
        for p in (d["results"] if isinstance(d, dict) else d):
            fen = p.get("fen") if isinstance(p, dict) else None
            if fen and fen not in seen:
                seen.add(fen)
                out.append(fen)
    return out


def corner_fens() -> list:
    import importlib.util
    spec = importlib.util.spec_from_file_location("trace_identity_corners", Path(__file__).resolve().parent / "trace_identity.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return list(m.CORNERS) + list(RULE_CORNERS)


def playout_fens(games: int, seed: int) -> list:
    """Every position of `games` uniformly random games from the initial position (seeded), each played to the kernel's terminal."""
    rng = random.Random(seed)
    out = []
    for _g in range(games):
        b = chess.Board()
        while True:
            out.append(b.fen())
            if CHESS.terminal(b) is not None:
                break
            b.push(rng.choice(list(b.legal_moves)))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ shrinking and filing

def _valid(b: chess.Board) -> bool:
    return b.is_valid()


def _keys(res: dict) -> set:
    return {(k, c, s) for k, c, s, _d in res["disagree"]}


def shrink(fen: str, key: tuple) -> str:
    """The smallest position (greedy, to a fixpoint) that still shows the disagreement `key` = (kernel, class, signature): remove one non-king unit, a castling right,
    the e.p. square, or reset the clock, while the board stays valid and the disagreement persists."""
    cur = chess.Board(fen)
    changed = True
    while changed:
        changed = False
        for cand in _shrink_candidates(cur):
            if not _valid(cand):
                continue
            try:
                res = compare(cand.fen(), rust=(key[0] == "rust"))
            except Exception:
                continue
            if key in _keys(res):
                cur, changed = cand, True
                break
    return cur.fen()


def _shrink_candidates(b: chess.Board):
    for s, p in sorted(b.piece_map().items()):
        if p.piece_type == chess.KING:
            continue
        c = b.copy(stack=False)
        c.remove_piece_at(s)
        yield c
    for s in chess.SquareSet(b.castling_rights):
        c = b.copy(stack=False)
        c.castling_rights &= ~chess.BB_SQUARES[s]
        yield c
    if b.ep_square is not None:
        c = b.copy(stack=False)
        c.ep_square = None
        yield c
    if b.halfmove_clock:
        c = b.copy(stack=False)
        c.halfmove_clock = 0
        yield c


def file_surprise(key: tuple, fen: str, shrunk: str, detail: str) -> Path:
    """A new case under rules/surprises/ (status 'open': the reading of the rules, the side that was wrong and the change are written after reading them)."""
    res = compare(shrunk, rust=(key[0] == "rust"))
    det = [d for k, c, s, d in res["disagree"] if (k, c, s) == key]
    return GC.file_surprise(key, fen, shrunk, det[0] if det else detail)


def load_surprises() -> list:
    return GC.load_surprises()


def replay(case: dict) -> list:
    """A filed surprise against both kernels: -> the disagreements left at its position (expect 'agree': none; expect 'known': the case's own disagreement, and no other
    kernel or class -- a kernel difference reported, not patched: when the kernel changes the replay says so and the case is read again)."""
    res = compare(case["fen"], rust=True)
    return res["disagree"]


# ------------------------------------------------------------------------------------------------------------------------------------------ the run

def _work(fens: list) -> list:
    out = []
    for fen in fens:
        t = time.perf_counter()
        res = compare(fen)
        res["sec"] = time.perf_counter() - t
        res["fen"] = fen
        out.append(res)
    return out


def run(games: int, seed: int, jobs: int) -> int:
    t0 = time.time()
    groups = {"puzzle": puzzle_fens(), "corner": corner_fens(), "playout": playout_fens(games, seed)}
    print(f"OBS crossplay sample: puzzle positions {len(groups['puzzle'])} (every distinct position of the puzzle sets), corners {len(groups['corner'])}, "
          f"play-out positions {len(groups['playout'])} ({games} random games, seed {seed}, every position to the kernel's terminal)")
    invalid = {g: [f for f in fs if not chess.Board(f).is_valid()] for g, fs in groups.items()}
    if any(invalid.values()):
        print("OBS crossplay sample: invalid positions left out (python-chess `is_valid`: both kernels assume a legal position) " +
              ", ".join(f"{g} {len(v)}" for g, v in invalid.items() if v))
    order = [(g, f) for g, fs in groups.items() for f in fs if f not in invalid[g]]
    fens = [f for _g, f in order]
    chunk = 25
    parts = [fens[i:i + chunk] for i in range(0, len(fens), chunk)]
    results = []
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            for r in pool.imap(_work, parts):
                results += r
    else:
        for part in parts:
            results += _work(part)
    wall = time.time() - t0
    cov: Counter = Counter()
    by_class: Counter = Counter()
    first: dict = {}
    n_moves = n_next = 0
    secs = []
    for (g, fen), res in zip(order, results):
        cov += res["coverage"]
        n_moves += res["n_moves"]
        n_next += res["n_next"]
        secs.append(res["sec"])
        for k, c, s, d in res["disagree"]:
            by_class[(k, c)] += 1
            first.setdefault((k, c, s), (fen, d, g))
            print(f"DISAGREE {k} {c} [{s}] at {fen} ({g}): {d}")
    n_bad = sum(1 for res in results if res["disagree"])
    n_bad_py = sum(1 for res in results if any(k == "python" for k, _c, _s, _d in res["disagree"]))
    print(f"OBS crossplay: positions {len(results)}, legal moves (python-chess) {n_moves}, next states compared {n_next} (python-chess) and {cov['rust next']} (Rust), "
          f"Rust draw readings {cov['rust draw read']}; wall {wall:.1f} s on {jobs} processes; the spec's reading per position (all its next states included) "
          f"median {sorted(secs)[len(secs) // 2] * 1000:.1f} ms, mean {sum(secs) / len(secs) * 1000:.1f} ms (python-chess and the Rust readings inside)")
    print("OBS crossplay coverage: " + ", ".join(f"{k} {v}" for k, v in sorted(cov.items()) if not k.startswith("rust")))
    print(f"OBS crossplay disagreements: positions with one {n_bad} of {len(results)} (with python-chess {n_bad_py}); by class "
          + (", ".join(f"{k} {c} {v}" for (k, c), v in sorted(by_class.items())) or "none"))
    print(f"OBS crossplay agreement: {len(results) - n_bad} of {len(results)} positions agree on every reading "
          f"({(len(results) - n_bad) / len(results) * 100:.3f}%); with python-chess {len(results) - n_bad_py} of {len(results)}")
    known = {(c["kernel"], c["class"], c["signature"]) for c in load_surprises()}
    for key, (fen, d, g) in sorted(first.items()):
        if key in known:
            print(f"OBS crossplay surprise already filed: {key}")
            continue
        small = shrink(fen, key)
        p = file_surprise(key, fen, small, d)
        print(f"OBS crossplay surprise FILED {p.name}: {key} at {fen} -> shrunk {small}")
    return 1 if n_bad else 0


def show(fen: str) -> None:
    res = compare(fen)
    print(f"moves {res['n_moves']}, next states {res['n_next']}, coverage {dict(res['coverage'])}")
    for k, c, s, d in res["disagree"]:
        print(f"DISAGREE {k} {c} [{s}]: {d}")
    if not res["disagree"]:
        print("agree")


def replay_main() -> int:
    bad = 0
    cases = load_surprises()
    for case in cases:
        left = replay(case)
        exp_known = case.get("expect") == "known"
        keys = {(k, c, s) for k, c, s, _d in left}
        mine = (case["kernel"], case["class"], case["signature"])
        ok = (mine in keys and {(k, c) for k, c, _s in keys} == {mine[:2]}) if exp_known else not left
        bad += not ok
        print(f"OBS surprise {case['file']}: {case['kernel']} {case['class']} -- {'as expected' if ok else 'DIFFERENT'} ({case.get('expect')}; left {sorted(keys)})")
    print(f"OBS surprises replayed: {len(cases)}, as expected {len(cases) - bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "run":
        def opt(name, default):
            return type(default)(a[a.index(name) + 1]) if name in a else default
        sys.exit(run(opt("--games", 40), opt("--seed", 76), opt("--jobs", os.cpu_count() or 1)))
    elif a[0] == "replay":
        sys.exit(replay_main())
    elif a[0] == "fen":
        show(a[1])
