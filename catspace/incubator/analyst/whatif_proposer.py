"""THE WHAT-IF PROPOSER (Kaveh 2026-09-28, chosen: "AlphaGeometry + expert iteration"). AlphaGeometry (Trinh et al.,
Nature 2024): a symbolic engine does the exact work; when it stalls, a learned model proposes an AUXILIARY CONSTRUCTION
("add point E on line AC") and the engine continues; the proposer is trained on synthetic proofs the engine verified,
with no human proofs. Here the frames are the engine and the what-if edits are the constructions ("what if one more
attacker bore on d5", "what if the e6 defender were gone", "what if the rook stood on h6"); an edit is USEFUL when a
move that starts or completes it (whatif.realizes) beats the best standard candidate (captures, checks, the one-ply top
moves) by half a pawn or more, by the frames' two-ply value against every relevant reply -- self-verified, no engine.

    python -m catspace.incubator.analyst.whatif_proposer data --n 400 --offset 499      # training puzzles, disjoint from the test sample
    python -m catspace.incubator.analyst.whatif_proposer train
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time

import chess
import numpy as np

from catspace.io import paths

KINDS = ("add", "remove", "relocate", "guard")
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
FEATURES = ["add", "remove", "relocate", "guard", "static_gain", "target_value", "arg_value", "hops", "n_complete", "n_start",
            "target_defenders", "target_attackers", "target_near_enemy_king", "target_near_own_king", "target_pinned",
            "n_ties", "top_value", "material_balance", "n_pieces", "in_check"]
MARGIN = 0.5


QUIET_THEMES = {"quietMove", "advancedPawn", "defensiveMove", "deflection", "attraction", "interference", "clearance", "zugzwang", "intermezzo"}


def training_rows(n: int, offset: int, stride: int = 997, themes: set | None = QUIET_THEMES) -> list:
    """Lichess puzzles from rows i % stride == offset -- the test sample (gauge_puzzles.sample) takes offset 0 -- whose
    themes name a quiet idea (where a construction is needed: the frames alone stall)."""
    import zstandard
    out = []
    with open(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst", "rb") as fh:
        reader = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"))
        for i, row in enumerate(reader):
            if i % 997 == 0:
                continue                                          # the test sample's rows (gauge_puzzles.sample: stride 997, offset 0)
            if i % stride == offset and (themes is None or themes & set(row["Themes"].split())):
                out.append(row)
                if len(out) >= n:
                    break
    return out


def edit_features(b: chess.Board, e, n_complete: int, n_start: int, ctx: dict) -> list:
    side = b.turn
    tp = b.piece_at(e.target)
    tv = VAL[tp.piece_type] if tp else 0
    argv = e.arg if e.kind in ("add", "guard") else (VAL[b.piece_at(e.arg).piece_type] if e.kind == "remove" and b.piece_at(e.arg) else 0)
    ek = b.king(not side); ok = b.king(side)
    near = lambda k: k is not None and chess.square_distance(e.target, k) <= 2
    return [e.kind == k for k in KINDS] + [e.gain, tv, argv, e.hops if e.kind == "relocate" else 1, n_complete, n_start,
                                           len(b.attackers(tp.color, e.target)) if tp else 0, len(b.attackers(not tp.color, e.target)) if tp else 0,
                                           near(ek), near(ok), bool(tp) and b.is_pinned(tp.color, e.target),
                                           ctx["n_ties"], ctx["top_value"], ctx["material"], ctx["n_pieces"], b.is_check()]


def _material(b: chess.Board) -> float:
    return sum((VAL[p.piece_type] if p.color == b.turn else -VAL[p.piece_type]) for p in b.piece_map().values())


def position_data(fen: str) -> list:
    """Every what-if edit of the side to move with its features and label (does a realizing move beat the standard
    candidates by MARGIN, by the frames' two-ply value)."""
    from .fates import clear_caches
    from .whatif import edits, realizes
    from . import frame_player as FP
    clear_caches()
    b = chess.Board(fen)
    if b.is_game_over():
        return []
    stand = FP.passed(b)
    before, book = FP.joint_threats_book(stand, FP.square_frame) if stand is not None else ((0.0, None, []), None)
    one = FP._one_ply_values(b, before, book)
    top1 = max(one.values()); ranked = sorted(one, key=lambda u: -one[u])
    standard = set(ranked[:6]) | {u for u in one if one[u] == top1} | {m.uci() for m in b.legal_moves if b.is_capture(m) or b.gives_check(m) or m.promotion}
    v2 = {}

    def val(u):
        if u not in v2:
            v2[u] = FP.two_ply_value(b, chess.Move.from_uci(u))[0]
        return v2[u]
    base = max(val(u) for u in standard)
    if base > MARGIN:
        return []                                                  # the frames do not stall here: no construction needed
    ctx = dict(n_ties=sum(1 for u in one if one[u] == top1), top_value=top1, material=_material(b), n_pieces=len(b.piece_map()))
    rows = []
    for e in edits(b):
        comp = [m.uci() for m in b.legal_moves if realizes(b, m, e) == 1]
        start = [m.uci() for m in b.legal_moves if realizes(b, m, e) == 2]
        moves = [u for u in comp + start if u not in standard]
        best = max((val(u) for u in moves), default=None)
        rows.append(dict(fen=fen, kind=e.kind, target=e.target, arg=e.arg, x=edit_features(b, e, len(comp), len(start), ctx),
                         useful=best is not None and best >= max(base, 0.0) + MARGIN, best=best, base=base,
                         moves=sorted(moves, key=lambda u: -val(u))[:3]))
    return rows


def solution_data(row: dict) -> list:
    """Stage 1 (Kaveh: 'solutions first'): at every solver step of a puzzle, every what-if edit of the side to move,
    labelled useful when the puzzle's solution move starts or completes it; with whether the frames alone chose it."""
    from .fates import clear_caches
    from .whatif import edits, realizes
    from . import frame_player as FP
    b = chess.Board(row["FEN"]); moves = row["Moves"].split(); b.push_uci(moves[0]); sol = moves[1:]
    out = []
    for i in range(0, len(sol), 2):
        clear_caches()
        m = chess.Move.from_uci(sol[i])
        FP.LEDGER = "one_ply"
        c = FP.choose(b)
        frames_ok = sol[i] in c["best"]
        one_top = max(c["values"].values())
        ctx = dict(n_ties=len(c["best"]), top_value=one_top, material=_material(b), n_pieces=len(b.piece_map()))
        legal = list(b.legal_moves)
        for e in edits(b):
            r = [realizes(b, x, e) for x in legal]
            out.append(dict(fen=b.fen(), pid=row["PuzzleId"], step=i // 2, kind=e.kind, target=e.target, arg=e.arg,
                            x=edit_features(b, e, sum(v == 1 for v in r), sum(v == 2 for v in r), ctx),
                            useful=realizes(b, m, e) > 0, frames_ok=frames_ok, quiet=not (b.is_capture(m) or b.gives_check(m) or m.promotion)))
        b.push(m)
        if i + 1 < len(sol):
            b.push_uci(sol[i + 1])
    return out


_MODEL = {}


def propose_moves(b: chess.Board, one: dict, k: int = 3, model: str = "whatif_proposer_solutions.pkl") -> dict:
    """Stage 2 (the AlphaGeometry step): the proposer ranks the what-if edits of the side to move; the moves that start
    or complete its top k become candidates for the search to verify. `one`: the one-ply values (uci -> value), the
    context the proposer was trained with. -> {uci: (rank of the edit, edit kind, target square)}."""
    import pickle
    from .whatif import edits, realizes
    if model not in _MODEL:
        _MODEL[model] = pickle.load(open(paths.derived(model), "rb"))
    es = edits(b)
    if not es:
        return {}
    top1 = max(one.values())
    ctx = dict(n_ties=sum(1 for v in one.values() if v == top1), top_value=top1, material=_material(b), n_pieces=len(b.piece_map()))
    legal = list(b.legal_moves)
    real = [[realizes(b, m, e) for m in legal] for e in es]
    X = np.array([edit_features(b, e, sum(v == 1 for v in r), sum(v == 2 for v in r), ctx) for e, r in zip(es, real)], dtype=float)
    p = _MODEL[model].predict_proba(X)[:, 1]
    out = {}
    for rank, i in enumerate(np.argsort(-p)[:k]):
        for m, v in zip(legal, real[i]):
            if v > 0 and m.uci() not in out:
                out[m.uci()] = (rank, es[i].kind, es[i].target)
    return out


def cmd_solutions(a):
    import multiprocessing as mp
    rows = training_rows(a.n, a.offset, stride=a.stride)
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(a.workers) as pool:
        got = pool.map(solution_data, rows, chunksize=1)
    out = [x for g in got for x in g]
    path = paths.derived(f"whatif_proposer_solutions_{a.offset}.jsonl")
    with open(path, "w") as fh:
        for x in out:
            fh.write(json.dumps(x) + "\n")
    steps = {(x["pid"], x["step"]) for x in out}
    with_useful = {(x["pid"], x["step"]) for x in out if x["useful"]}
    missed = {(x["pid"], x["step"]) for x in out if not x["frames_ok"]}
    print(f"[whatif_proposer] {len(rows)} puzzles, {len(steps)} solver steps, {len(out)} edits ({sum(x['useful'] for x in out)} realized by the solution); "
          f"steps whose solution realizes an edit {len(with_useful)}; steps the frames alone missed {len(missed)} "
          f"(with an edit the solution realizes: {len(missed & with_useful)}); {time.perf_counter() - t0:.0f}s -> {path}")


def cmd_data(a):
    import multiprocessing as mp
    rows = training_rows(a.n, a.offset, stride=a.stride)
    fens = []
    for r in rows:
        b = chess.Board(r["FEN"]); b.push_uci(r["Moves"].split()[0]); fens.append(b.fen())
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(a.workers) as pool:
        got = pool.map(position_data, fens, chunksize=1)
    out = [x for g in got for x in g]
    path = paths.derived(f"whatif_proposer_data_{a.offset}.jsonl")
    with open(path, "w") as fh:
        for x in out:
            fh.write(json.dumps(x) + "\n")
    print(f"[whatif_proposer] {len(fens)} positions ({sum(1 for g in got if g)} where the frames stall), {len(out)} edits, {sum(x['useful'] for x in out)} useful "
          f"({sum(x['useful'] for x in out) / max(1, len(out)):.3f}); positions with a useful edit "
          f"{sum(1 for g in got if any(x['useful'] for x in g))}/{len(fens)}; {time.perf_counter() - t0:.0f}s -> {path}")


def cmd_train(a):
    import hashlib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    data = []
    for off in a.offsets:
        data += [json.loads(l) for l in open(paths.derived(f"whatif_proposer_{'solutions' if a.solutions else 'data'}_{off}.jsonl"))]
    key = (lambda d: d["pid"]) if a.solutions else (lambda d: d["fen"])
    test_of = lambda k: int(hashlib.sha1(("prop" + k).encode()).hexdigest()[:8], 16) % 4 == 0
    tr = [d for d in data if not test_of(key(d))]; te = [d for d in data if test_of(key(d))]
    X = np.array([d["x"] for d in tr], dtype=float); y = np.array([d["useful"] for d in tr])
    Xt = np.array([d["x"] for d in te], dtype=float); yt = np.array([d["useful"] for d in te])
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced").fit(X, y)
    p = model.predict_proba(Xt)[:, 1]
    print(f"[whatif_proposer] train {len(tr)} edits ({y.mean():.3f} useful), held out {len(te)} ({yt.mean():.3f}); "
          f"AUC {roc_auc_score(yt, p):.3f} (static gain alone {roc_auc_score(yt, Xt[:, FEATURES.index('static_gain')]):.3f})")
    for label, sel in (("all steps", lambda d: True), ("steps the frames alone missed", lambda d: not d.get("frames_ok", False)),
                       ("steps whose solution is quiet", lambda d: d.get("quiet", False))):
        by = {}
        for d, pp in zip(te, p):
            if sel(d):
                by.setdefault((d["fen"]), []).append((pp, d["x"][FEATURES.index("static_gain")], d["useful"]))
        pos = [v for v in by.values() if any(u for _, _, u in v)]
        if not pos:
            continue
        for k in (1, 3, 5):
            hit = sum(1 for v in pos if any(u for _, _, u in sorted(v, key=lambda t: -t[0])[:k]))
            hs = sum(1 for v in pos if any(u for _, _, u in sorted(v, key=lambda t: -t[1])[:k]))
            print(f"OBS {label}, positions whose solution realizes an edit ({len(pos)}): among the proposer's top {k} in {hit} (by static gain: {hs}; "
                  f"edits per position {np.mean([len(v) for v in pos]):.0f})")
    import pickle
    pickle.dump(model, open(paths.derived(f"whatif_proposer{'_solutions' if a.solutions else ''}.pkl"), "wb"))


def main(argv=None):
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data"); d.add_argument("--n", type=int, default=400); d.add_argument("--offset", type=int, default=499); d.add_argument("--workers", type=int, default=10)
    d.add_argument("--stride", type=int, default=97)
    t = sub.add_parser("train"); t.add_argument("--offsets", type=int, nargs="+", default=[499]); t.add_argument("--solutions", action="store_true")
    so = sub.add_parser("solutions"); so.add_argument("--n", type=int, default=600); so.add_argument("--offset", type=int, default=49)
    so.add_argument("--workers", type=int, default=10); so.add_argument("--stride", type=int, default=97)
    a = ap.parse_args(argv)
    {"data": cmd_data, "train": cmd_train, "solutions": cmd_solutions}[a.cmd](a)


if __name__ == "__main__":
    main()
