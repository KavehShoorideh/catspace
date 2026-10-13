"""THE AUDITOR CHECKED AGAINST REFERENCES (Kaveh 2026-10-12: "you can use Stockfish to verify our auditor or whatever other
tool you trust" -- in tests and measurements only, never in play).

The planner's move audit (plan.rs audit_net: what a move nets in material after their best answer -- the static exchange,
their threats, their forcing search, the game's draws) on every legal move of sampled positions from our games, against

  MATERIAL  Stockfish (with the Syzygy tablebases) after the move: the material balance along its line (up to PLIES
            plies, or mate) -- "the audit says safe, the line loses a piece" (a MISSED loss) and "the audit says a piece
            goes, the line keeps everything" (a FALSE alarm)
  OUTCOME   the tablebase (exact for these piece counts): "the audit says safe, the move throws a win away" and "the audit
            says unsafe, the move keeps the position's result"

Neither reference knows the game's history: a verdict whose reason is a repetition or the fifty-move clock is counted
apart (HISTORY), not as a disagreement.

  python -m catspace.incubator.analyst.audit_oracle --runs carry2,geometry --n 400
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter

from catspace.io import paths

VAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}            # python-chess piece types -> the kernel's values
PLIES = 8
MATE = 1000
HISTORY_WORDS = ("threefold", "repetition", "fifty")

_ENG = _TB = None


def balance(b, white: bool) -> int:
    """Our material minus theirs (we are White when `white`)."""
    s = sum(VAL[p.piece_type] * (1 if p.color == white else -1) for p in b.piece_map().values())
    return s


def line_net(b, us: bool, pv: list, score) -> int:
    """The material change for `us` along the reference's line (up to PLIES plies), or +-MATE when it mates."""
    if score is not None and score.is_mate():
        return MATE if (score.white().mate() or 0) > 0 and us or (score.white().mate() or 0) < 0 and not us else -MATE
    before = balance(b, us)
    x = b.copy(stack=False)
    for m in pv[:PLIES]:
        if m not in x.legal_moves:
            break
        x.push(m)
    return balance(x, us) - before


def _init():
    global _ENG, _TB
    import chess.engine
    import chess.syzygy
    _TB = chess.syzygy.open_tablebase(str(paths.syzygy_dir()))
    _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    _ENG.configure({"SyzygyPath": str(paths.syzygy_dir()), "Threads": 1})


def _one(args):
    import chess
    import chess.engine
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    fen, history, depth = args
    b = chess.Board(fen)
    us = b.turn
    wdl0 = _TB.probe_wdl(b)
    out = []
    for u, net, why in A.audit_check(fen, history):
        m = chess.Move.from_uci(u)
        before = balance(b, us)
        x = b.copy(stack=False)
        x.push(m)
        taken = balance(x, us) - before                       # what the move itself takes
        if x.is_checkmate():
            ref, wdl = MATE, 2
        elif x.is_stalemate() or x.is_insufficient_material():
            ref, wdl = taken, 0
        else:
            info = _ENG.analyse(x, chess.engine.Limit(depth=depth))
            ref = taken + line_net(x, us, info.get("pv", []), info.get("score"))
            wdl = -_TB.probe_wdl(x)
        out.append(dict(fen=fen, move=u, san=b.san(m), net=net, why=why, ref=ref, wdl0=wdl0, wdl=wdl,
                        history=any(w in why for w in HISTORY_WORDS)))
    return out


def positions(runs: list, n: int, seed: int = 0) -> list:
    """(fen, the game's history before it) for our moves while ahead, sampled across the runs' rounds."""
    import chess
    pool = []
    for run in runs:
        for rnd in range(0, 10):
            p = paths.experiment(f"surprise_loop_KRRvKR_{run}_r{rnd}.json")
            if not os.path.exists(p):
                continue
            for g in json.load(open(p)):
                hist = []
                for r in g["rows"]:
                    if r["lead"] > 0:
                        pool.append((r["fen"], list(hist)))
                    hist.append(r["fen"])
                    b = chess.Board(r["fen"]); b.push_san(r["san"]); hist.append(b.fen())
    return random.Random(seed).sample(pool, min(n, len(pool)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="carry2,geometry")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    a = ap.parse_args(argv)
    t0 = time.time()
    pos = positions(a.runs.split(","), a.n)
    from multiprocessing import Pool
    with Pool(a.workers, initializer=_init) as p:
        rows = [r for rs in p.imap_unordered(_one, [(f, h, a.depth) for f, h in pos], chunksize=4) for r in rs]
    kind = Counter()
    ex = {}
    for r in rows:
        if r["history"]:
            k = "history (repetition / fifty moves -- the references cannot see it)"
        elif r["net"] >= 0 and r["ref"] <= -3:
            k = "MISSED: the audit says safe, the line loses a piece or more"
        elif r["net"] <= -3 and r["ref"] >= 0:
            k = "FALSE ALARM: the audit says a piece goes, the line keeps everything"
        else:
            k = "agree (material)"
        kind[k] += 1
        ex.setdefault(k, []).append(r)
        if not r["history"]:
            if r["net"] >= 0 and r["wdl"] < r["wdl0"]:
                kind["OUTCOME MISSED: the audit says safe, the tablebase says the move throws the result away"] += 1
                ex.setdefault("outcome missed", []).append(r)
            elif r["net"] < 0 and r["wdl"] >= r["wdl0"]:
                kind["OUTCOME FALSE ALARM: the audit says unsafe, the tablebase says the result is kept"] += 1
                ex.setdefault("outcome false alarm", []).append(r)
    print(f"OBS the auditor against Stockfish (depth {a.depth}, its line up to {PLIES} plies) and the tablebase: {len(pos)} positions "
          f"of ours while ahead (runs {a.runs}), {len(rows)} moves ({time.time() - t0:.0f} s)")
    for k, v in kind.most_common():
        print(f"OBS   {k}: {v} ({100 * v / max(len(rows), 1):.1f}% of the moves)")
    for k in ("MISSED: the audit says safe, the line loses a piece or more", "FALSE ALARM: the audit says a piece goes, the line keeps everything",
              "outcome missed", "outcome false alarm"):
        for r in random.Random(1).sample(ex.get(k, []), min(4, len(ex.get(k, [])))):
            print(f"OBS   e.g. [{k.split(':')[0]}] {r['fen']} {r['san']}: the audit {r['net']} ({r['why'] or 'nothing of theirs'}), "
                  f"the line {r['ref']}, the tablebase {r['wdl0']} -> {r['wdl']}")
    json.dump(rows, open(paths.experiment(f"audit_oracle_{a.runs.replace(',', '_')}.json"), "w"))


if __name__ == "__main__":
    main()
