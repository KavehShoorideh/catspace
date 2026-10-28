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
from catspace.incubator.analyst.discovery.game import CHESS
from catspace.io import paths

SURPRISES = GC.SURPRISES
PUZZLE_GLOB = "ledger_puzzles_*.json"

RULE_CORNERS = [
    "8/8/8/4k3/8/8/8/KN6 w - - 0 1",                 # K+N v K: insufficient (both kernels)
    "8/8/8/4k3/8/8/8/KB6 w - - 0 1",                 # K+B v K: insufficient (both kernels)
    "8/8/8/4k3/8/3b4/8/KB6 w - - 0 1",               # K+B v K+B, the bishops on one colour (b1, d3 light): python-chess insufficient
    "8/8/8/4k3/8/2b5/8/KB6 w - - 0 1",               # K+B v K+B on opposite colours: a mate exists -- sufficient
    "8/8/8/4k3/8/8/3q4/KN6 w - - 0 1",               # K+N v K+Q: the queen can mate -- sufficient
    "8/8/8/4k3/8/8/3n4/KN6 w - - 0 1",               # K+N v K+N: a selfmate exists -- sufficient
    "8/8/3k4/8/8/8/1B6/KB6 w - - 0 1",               # K+BB on two colours v K: sufficient
    "8/8/8/4k3/8/8/2B5/KB6 w - - 0 1",               # K+BB on one colour (b1, c2 light) v K: insufficient
    "8/8/8/4k3/8/8/8/KR6 w - - 100 80",              # the clock at its limit, legal moves: the fifty-move rule
    "8/8/8/4k3/8/8/8/KR6 w - - 99 80",               # the clock one below: a quiet move reaches 100, a capture or a pawn move resets
    "8/8/8/4k3/8/8/8/KR6 w - - 140 80",              # the clock far above the limit (the spec saturates it)
    "R5k1/5ppp/8/8/8/8/8/6K1 b - - 100 80",          # mated with the clock at 100: the mate comes first
    "7k/5Q2/6K1/8/8/8/8/8 b - - 100 80",             # stalemated with the clock at 100: no legal move, so no fifty-move claim
    "r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1",        # castling kingside through f1 attacked: illegal; queenside legal
    "r3k2r/8/8/8/8/8/1r6/R3K2R w KQkq - 0 1",        # queenside with b1 attacked: legal (b1 need only be empty)
    "r3k2r/8/8/8/8/8/6B1/R3K2R w KQkq - 0 1",        # Bxa8 ends Black's queenside right (a capture on the rook's square)
    "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1",            # rights in part
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",      # an e.p. square no pawn can use
    "4k3/8/8/4pP2/8/8/8/4K3 w - e6 0 2",             # f5xe6 e.p.
    "4k3/8/8/r2pP2K/8/8/8/8 w - d6 0 2",             # e.p. that would expose the king along the rank: illegal
    "4k3/1P6/8/8/8/8/6p1/4K2R b K - 0 1",            # Black's g2xh1 promotions (each piece) capture the rook: White's right ends
]


# ------------------------------------------------------------------------------------------------------------------------------------------ one position

def _rust_state(q) -> frozenset:
    """The spec's state of a Rust position (its exact FEN, the raw e.p. square it keeps)."""
    return GC.state_of(chess.Board(q.fen_exact()), ep=q.ep_square())


def _fluents(diff) -> list:
    return sorted({rel.split(".", 1)[1] for rel, _a in diff})


def compare(fen: str, rust: bool = True, prog: "GDL.Program | None" = None) -> dict:
    """The spec against the kernels at one position. -> {'n_moves', 'coverage': Counter, 'disagree': [(kernel, class, signature, detail)]}."""
    prog = prog or GC.program()
    b = chess.Board(fen)
    st = GC.state_of(b)
    r = GC.Reading(st, prog)
    out = []
    cov: Counter = Counter()
    k_legal = sorted(m.uci() for m in b.legal_moves)
    # python-chess: the legal moves
    if r.legal != k_legal:
        spec_only, kern_only = sorted(set(r.legal) - set(k_legal)), sorted(set(k_legal) - set(r.legal))
        sig = "spec+" + ",".join(sorted({_move_kind(b, u) for u in spec_only})) + " kernel+" + ",".join(sorted({_move_kind(b, u) for u in kern_only}))
        out.append(("python", "legal", sig, f"spec only {spec_only}, kernel only {kern_only}"))
    # the next states (every move legal in both)
    n_next = 0
    for m in b.legal_moves:
        u = m.uci()
        if u not in r.legal:
            continue
        ns = GC.next_state(st, u, r.mover, prog)
        c = b.copy(stack=False)
        c.push(m)
        ks = GC.state_of(c)
        n_next += 1
        _cover_move(cov, b, m, ns)
        if ns != ks:
            out.append(("python", "next", "+".join(_fluents(ns ^ ks)) + ":" + _move_kind(b, u),
                        f"after {u}: spec only {sorted(GC.fact_text(f) for f in ns - ks)}, kernel only {sorted(GC.fact_text(f) for f in ks - ns)}"))
    # the terminal reasons, the verdict, the goals
    k_reasons = GC.kernel_reasons(b)
    if r.reasons != k_reasons:
        out.append(("python", "terminal", f"spec {'+'.join(r.reasons) or '-'} kernel {'+'.join(k_reasons) or '-'}", f"spec {r.reasons}, kernel {k_reasons}"))
    kind = CHESS.terminal(b)
    spec_kind = next((x for x in GC.KERNEL_ORDER if x in r.reasons), None)
    if (GC.KERNEL_KIND.get(kind) if kind else None) != spec_kind or r.terminal != (kind is not None):
        out.append(("python", "verdict", f"spec {spec_kind} kernel {kind}", f"spec terminal {r.terminal} {spec_kind}, Game.terminal {kind}"))
    if r.goals != GC.goals_of(kind, r.mover):
        out.append(("python", "goal", f"spec {r.goals} kernel {GC.goals_of(kind, r.mover)}", ""))
    for x in r.reasons:
        cov["reason " + x] += 1
    if r.check:
        cov["check"] += 1
    if rust:
        out += _compare_rust(b, st, r, prog, cov)
    return {"n_moves": len(k_legal), "n_next": n_next, "coverage": cov, "disagree": out}


def _compare_rust(b: chess.Board, st, r, prog, cov) -> list:
    out = []
    try:
        p = AC.Pos(b.fen())
    except ValueError as x:
        return [("rust", "unparsable", "", str(x))]
    rl = sorted(p.legal_ucis())
    if rl != r.legal:
        spec_only, kern_only = sorted(set(r.legal) - set(rl)), sorted(set(rl) - set(r.legal))
        sig = "spec+" + ",".join(sorted({_move_kind(b, u) for u in spec_only})) + " kernel+" + ",".join(sorted({_move_kind(b, u) for u in kern_only}))
        out.append(("rust", "legal", sig, f"spec only {spec_only}, kernel only {kern_only}"))
    for u in rl:
        if u not in r.legal:
            continue
        ns = GC.next_state(st, u, r.mover, prog)
        q = p.push(u)
        ks = _rust_state(q)
        cov["rust next"] += 1
        if ns != ks:
            out.append(("rust", "next", "+".join(_fluents(ns ^ ks)) + ":" + _move_kind(b, u),
                        f"after {u}: spec only {sorted(GC.fact_text(f) for f in ns - ks)}, kernel only {sorted(GC.fact_text(f) for f in ks - ns)}"))
    rm, rs = p.is_checkmate(), p.is_stalemate()
    if rm != ("mate" in r.reasons) or rs != ("stalemate" in r.reasons):
        out.append(("rust", "terminal", f"spec {'+'.join(x for x in r.reasons if x in ('mate', 'stalemate')) or '-'} kernel "
                    f"{'+'.join(x for x, y in (('mate', rm), ('stalemate', rs)) if y) or '-'}", ""))
    # the Rust core's draws (game.rs `drawn`: the fifty-move rule, then insufficient material, then stalemate), read at the successors (`game_draw_check`, no history)
    rdraw = dict(AC.game_draw_check(b.fen(), []))
    for u in rl:
        if u not in r.legal:
            continue
        ns = GC.next_state(st, u, r.mover, prog)
        exp = _expected_rust_draw(ns, prog)
        got = rdraw.get(u)
        if got == "stalemate" and exp is None:
            continue                                      # a stalemate after the move: compared at the position itself (is_stalemate), not here
        if got != exp:
            sig = f"spec {exp} kernel {got}: {_material(ns)}"
            out.append(("rust", "draw", sig, f"after {u}: spec {exp}, Rust game_draw_check {got}"))
        cov["rust draw read"] += 1
    return out


def _expected_rust_draw(ns, prog) -> "str | None":
    return GC.expected_rust_draw(ns, prog)            # (plan 76.1 (a): moved to gdl_chess, read by the ontology shelf's agreement too)


def _material(state) -> str:
    return GC.material_signature(state)


def _move_kind(b: chess.Board, uci: str) -> str:
    m = chess.Move.from_uci(uci)
    p = b.piece_at(m.from_square)
    if p is None:
        return "none"
    if b.is_castling(m):
        return "castle"
    if b.is_en_passant(m):
        return "en passant"
    if m.promotion:
        return "promotion"
    return chess.piece_name(p.piece_type) + (" capture" if b.is_capture(m) else "")


def _cover_move(cov: Counter, b: chess.Board, m: chess.Move, ns) -> None:
    if b.is_castling(m):
        cov["castle"] += 1
    if b.is_en_passant(m):
        cov["en passant"] += 1
    if m.promotion:
        cov["promotion " + chess.piece_name(m.promotion)] += 1
    if b.is_capture(m):
        cov["capture"] += 1
    if any(rel == "true.ep_square" for rel, _a in ns):
        cov["e.p. square set"] += 1
    rights_before = len(chess.SquareSet(b.clean_castling_rights()))
    rights_after = sum(1 for rel, _a in ns if rel == "true.castling_right")
    if rights_after < rights_before:
        cov["castling right lost"] += 1
    clock = next((int(a[0]) for rel, a in ns if rel == "true.halfmove"), None)
    if clock == GC.CLOCK_LIMIT:
        cov["clock at its limit after the move"] += 1


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
