"""gauge_threat.py -- is 'respond to the threat' what the best move does, and how much does it constrain the
search? (Kaveh 2026-09-15: a threat is the easily seen loss; its level should constrain the action.)

On held-out game positions: the standing threat is what the opponent's LAST move created, as the threat module
reads it (a mate in one, or an enemy piece that falls by the ledger, with the REQUIRED DEFENCES enumerated by the
goal search's AND node). Reported by threat level (none / material by its value / mate): how often the engine's
best move is one of the required defences (against a random legal move), the size of the defence set as a
fraction of the legal moves (the constraint), and, where the best move is NOT a defence, whether it is a
counter-threat worth at least as much (the exception the gate must allow).

    python -m catspace.incubator.analyst.gauge_threat --n 1200 [--shard i/k]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import chess

from .memo import Memo
from .threats import threats_after, free_move_threats
from . import knowledge_net as K

OUT = Path("data/derived/threat")


def level_of(threats) -> tuple[str, float]:
    if not threats:
        return "none", 0.0
    if any(t.kind == "mate" for t in threats):
        return "mate", 1000.0
    v = max(t.value for t in threats)
    return ("piece" if v >= 3 else "pawn"), v


def run(n: int, shard: str, seed: int = 0, budget: int = 150, log=print):
    i, k = (int(x) for x in shard.split("/"))
    OUT.mkdir(parents=True, exist_ok=True)
    data = K.Data("data/derived/knowledge", meta_only=True)
    ids = [int(x) for x in data.ids("test")]
    rng = random.Random(seed); rng.shuffle(ids)
    ids = [x for x in ids if data.index.get((data.meta[x]["game"], data.meta[x]["ply"] - 1)) is not None and data.meta[x]["best"]][:n][i::k]
    from .engine import Engine
    eng = Engine(depth=12, cache_path=str(OUT / f"engine_{i}.sqlite")); memo = Memo()
    rows = []; t0 = time.perf_counter()
    for nn, x in enumerate(ids, 1):
        m = data.meta[x]; prev = data.meta[data.index[(m["game"], m["ply"] - 1)]]
        b0 = chess.Board(prev["fen"]); their = chess.Move.from_uci(prev["played"])
        board = chess.Board(m["fen"]); best = chess.Move.from_uci(m["best"])
        try:
            ts = threats_after(b0, their, memo, eng, budget=budget)               # what their move threatens against us, with the defences
        except Exception as ex:
            log(f"[threat] {x} error {ex}", flush=True); continue
        level, value = level_of(ts)
        legal = list(board.legal_moves); sans = {board.san(mv): mv for mv in legal}
        defences = set()
        for t in ts:
            defences |= set(t.defences)
        best_san = board.san(best)
        row = dict(i=x, fen=m["fen"], level=level, value=value, n_legal=len(legal), n_defences=len(defences & set(sans)),
                   best_is_defence=(best_san in defences), best_kind=m["best_kind"], random_is_defence=(rng.choice(list(sans)) in defences) if sans else False)
        if ts and not row["best_is_defence"]:
            # the exception: is the best move a counter-threat worth at least the threat? (our free-move gain after it, one reply deep)
            nb = board.copy(stack=False); nb.push(best)
            try:
                ours = free_move_threats(nb, board.turn, memo, top=3) if not nb.is_game_over() else []
                row["counter_value"] = max((g for _, g, _ in ours), default=0.0)
            except Exception:
                row["counter_value"] = 0.0
        rows.append(row)
        if nn % 25 == 0:
            log(f"[threat {shard}] {nn}/{len(ids)}, {time.perf_counter() - t0:.0f}s", flush=True)
    eng.close()
    with open(OUT / f"rows_{i}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"[threat {shard}] done {len(rows)} in {time.perf_counter() - t0:.0f}s", flush=True)


def run_ledger(n: int, shard: str, seed: int = 0, min_threat: float = 1.0, log=print):
    """The null-move definition: the threat is the opponent's best free-move gain by the ledger (any position, not
    only what the last move created); a move MEETS it when the residual gain in the child is at most half a pawn
    or a pawn below the threat; a COUNTER is a move after which our own free-move gain is at least the threat."""
    i, k = (int(x) for x in shard.split("/"))
    OUT.mkdir(parents=True, exist_ok=True)
    data = K.Data("data/derived/knowledge", meta_only=True)
    from .graph import POSITION_QUESTIONS
    import numpy as np
    from .dataset import load
    pos = load("data/derived/knowledge")["position"]
    PQ = {q: j for j, q in enumerate(POSITION_QUESTIONS)}
    ids = [int(x) for x in data.ids("test") if data.meta[int(x)]["best"] and pos[int(x), PQ["threat_theirs"]] >= min_threat]
    rng = random.Random(seed); rng.shuffle(ids); ids = ids[:n][i::k]
    memo = Memo(); rows = []; t0 = time.perf_counter()
    for nn, x in enumerate(ids, 1):
        m = data.meta[x]; board = chess.Board(m["fen"]); best = chess.Move.from_uci(m["best"])
        threat = float(pos[x, PQ["threat_theirs"]]); mate_t = pos[x, PQ["mate_theirs"]] > 0
        legal = list(board.legal_moves)
        resid = {}; counter = {}
        try:
            for mv in legal:
                nb = board.copy(stack=False); nb.push(mv)
                if nb.is_game_over():
                    oc = nb.outcome(); resid[mv] = 0.0 if (oc and oc.winner == board.turn) else (0.0 if oc and oc.winner is None else 1000.0); counter[mv] = 1000.0 if oc and oc.winner == board.turn else 0.0; continue
                theirs = free_move_threats(nb, not board.turn, memo, top=3)              # their free-move gain now (it is their move: what they take)
                resid[mv] = max((g for _, g, _ in theirs), default=0.0)
                ours = free_move_threats(nb, board.turn, memo, top=3)
                counter[mv] = max((g for _, g, _ in ours), default=0.0)
        except Exception as ex:
            log(f"[threat] {x} error {ex}", flush=True); continue
        meets = {mv for mv in legal if resid[mv] <= max(0.5, threat - 1.0)}
        counters = {mv for mv in legal if counter[mv] >= threat and mv not in meets}
        rnd = rng.choice(legal)
        rows.append(dict(i=x, fen=m["fen"], threat=threat, mate=bool(mate_t), n_legal=len(legal), n_meets=len(meets), n_counters=len(counters),
                         best_meets=best in meets, best_counters=best in counters, random_meets=rnd in meets, best_kind=m["best_kind"],
                         best_resid=resid.get(best, -1.0), min_resid=min(resid.values())))
        if nn % 10 == 0:
            log(f"[threat-ledger {shard}] {nn}/{len(ids)}, {time.perf_counter() - t0:.0f}s", flush=True)
    with open(OUT / f"ledger_rows_{i}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"[threat-ledger {shard}] done {len(rows)} in {time.perf_counter() - t0:.0f}s", flush=True)


def report_ledger() -> str:
    rows = []
    for f in sorted(OUT.glob("ledger_rows_*.jsonl")):
        rows.extend(json.loads(l) for l in open(f))
    out = [f"[threat-ledger] {len(rows)} held-out positions with a free-move threat >= 1 by the ledger (the null-move definition)"]
    out.append(f"  {'threat':<10}{'n':>5}{'best meets it':>15}{'random does':>13}{'meets/legal':>13}{'best counters':>15}{'meets or counters':>19}{'best quiet':>12}")
    bands = (("1-2.9", 1.0, 3.0), ("3-8.9", 3.0, 9.0), (">= 9", 9.0, 999.0), ("mate", 999.0, 1e9))
    for name, lo, hi in bands:
        rs = [r for r in rows if lo <= r["threat"] < hi]
        if not rs:
            continue
        out.append(f"  {name:<10}{len(rs):>5}{sum(r['best_meets'] for r in rs) / len(rs):>15.0%}{sum(r['random_meets'] for r in rs) / len(rs):>13.0%}"
                   f"{sum(r['n_meets'] / r['n_legal'] for r in rs) / len(rs):>13.2f}{sum(r['best_counters'] for r in rs) / len(rs):>15.0%}"
                   f"{sum(r['best_meets'] or r['best_counters'] for r in rs) / len(rs):>19.0%}{sum(r['best_kind'] == 'quiet' for r in rs) / len(rs):>12.0%}")
    return "\n".join(out)


def report() -> str:
    rows = []
    for f in sorted(OUT.glob("rows_*.jsonl")):
        rows.extend(json.loads(l) for l in open(f))
    out = [f"[threat] {len(rows)} held-out positions after the opponent's move; the threat as the ledger and the goal search read it"]
    out.append(f"  {'level':<8}{'n':>6}{'share':>7}{'best is a defence':>19}{'random move is':>16}{'defences/legal':>16}{'best = counter-threat':>22}{'best quiet':>12}")
    for lvl in ("none", "pawn", "piece", "mate"):
        rs = [r for r in rows if r["level"] == lvl]
        if not rs:
            continue
        with_def = [r for r in rs if r["n_defences"] > 0]
        bd = sum(r["best_is_defence"] for r in rs) / len(rs); rd = sum(r["random_is_defence"] for r in rs) / len(rs)
        frac = sum(r["n_defences"] / max(1, r["n_legal"]) for r in with_def) / max(1, len(with_def))
        counter = [r for r in rs if not r["best_is_defence"] and "counter_value" in r]
        cv = sum(1 for r in counter if r["counter_value"] >= r["value"]) / max(1, len(counter)) if lvl != "none" else float("nan")
        quiet = sum(1 for r in rs if r["best_kind"] == "quiet") / len(rs)
        out.append(f"  {lvl:<8}{len(rs):>6}{len(rs) / len(rows):>7.0%}{bd:>19.0%}{rd:>16.0%}{frac:>16.2f}{cv:>22.0%}{quiet:>12.0%}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1200); ap.add_argument("--shard", default="0/1"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--mode", default="last-move", choices=["last-move", "ledger"])
    a = ap.parse_args(argv)
    if a.report:
        print(report() if a.mode == "last-move" else report_ledger())
    elif a.mode == "ledger":
        run_ledger(a.n, a.shard)
    else:
        run(a.n, a.shard)


if __name__ == "__main__":
    main()
