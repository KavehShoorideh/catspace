"""THE SURPRISE LOOP (Kaveh 2026-10-11: "every time Stockfish disagrees with you, you get a signal and you do two things:
see if there was a lemma you could have considered that you didn't -- lemma generation -- and the picking of lemmas and
the depth you go in these frames and proofs -- the reinforcement learning itself").

  1 DISAGREEMENTS  the decision review's rows where our decision loses 150+ cp to Stockfish's best
  2 ATTRIBUTION    each replayed under variants -- the stop policy off; the forcing budget x10; the proof budgets x5;
                   all three -- and Stockfish grades every variant's move: the FIRST variant that brings the loss under
                   the bar names the cause (policy / forcing budget / proof depth); none -> a LEMMA IS MISSING
  3 THE TWO OUTPUTS
     lemma mining  the unfixed cases with Stockfish's line in SAN, for reading (artifacts/experiments/surprise_unfixed.json)
     the policy    on the cases the policy caused, the full search's log: every line that PROVED and that the deployed
                   policy would have STOPPED is a hard positive (data/derived/forcing_policy_hard.npz); `train --hard`
                   folds them in with weight
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from catspace.io import paths

BAR = 150
VARIANTS = [("policy off", dict(policy=False)), ("forcing x10", dict(policy=False, forcing=2000)),
            ("proofs x5", dict(policy=False, scale=5)), ("all", dict(policy=False, forcing=2000, scale=5))]

_ENG = None


def _sf(b, depth=12):
    global _ENG
    import chess.engine
    if _ENG is None:
        _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    return _ENG.analyse(b, chess.engine.Limit(depth=depth))


def _loss(fen: str, mv: str) -> int:
    import chess
    b = chess.Board(fen)
    i0 = _sf(b)
    a = b.copy(); a.push_uci(mv)
    i1 = _sf(a)
    return i0["score"].pov(b.turn).score(mate_score=3000) - i1["score"].pov(b.turn).score(mate_score=3000)


def _decide(fen: str, policy: bool = True, forcing: int | None = None, scale: int = 1):
    import analyst_core
    analyst_core.set_forcing_policy_on(policy)
    if forcing is not None:
        analyst_core.forcing_stats(forcing)
    analyst_core.set_proof_scale(scale)
    r = analyst_core.plan_trace(fen)
    analyst_core.set_proof_scale(1)
    if forcing is not None:
        analyst_core.forcing_stats(200)
    analyst_core.set_forcing_policy_on(True)
    dec = r[1]
    if not dec:
        return None
    kind = dec[0]
    return dec[1] if kind in ("", "defend", "cheapest") else kind


def _case(row):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core
    fen = row["fen"]
    out = dict(fen=fen, move=row["move"], loss=row["loss"], best_move=row["best_move"], best_line=row["best_line"], kind=row["kind"], variants=[])
    cause = "lemma missing"
    for name, kw in VARIANTS:
        mv = _decide(fen, **kw)
        loss = _loss(fen, mv) if mv else row["loss"]
        out["variants"].append(dict(name=name, move=mv, loss=loss))
        if mv and loss < BAR and cause == "lemma missing":
            cause = name
    out["cause"] = cause
    # the policy's hard positives: the full search's lines that proved and the deployed policy would have stopped
    if cause == "policy off":
        analyst_core.set_forcing_policy_on(False)
        analyst_core.forcing_stats(1000)
        analyst_core.set_forcing_log(True)
        analyst_core.forcing_log()
        analyst_core.plan_trace(fen)
        rows = [(f, int(p), int(c)) for f, p, _, c in analyst_core.forcing_log() if p]
        analyst_core.set_forcing_log(False)
        analyst_core.forcing_stats(200)
        analyst_core.set_forcing_policy_on(True)
        out["proven_lines"] = rows
    return out


def run(store: str, limit: int, workers: int) -> None:
    import numpy as np
    from multiprocessing import Pool
    rows = [r for r in json.load(open(paths.experiment(store))) if r["loss"] >= BAR][:limit]
    t0 = time.time()
    with Pool(workers) as p:
        cases = list(p.imap_unordered(_case, rows, chunksize=1))
    by = Counter(c["cause"] for c in cases)
    print(f"OBS surprise loop: {len(cases)} disagreements (loss >= {BAR} cp) replayed ({time.time() - t0:.0f} s on {workers} processes); "
          f"cause: " + ", ".join(f"{k} {v} ({100 * v / len(cases):.0f}%)" for k, v in by.most_common()))
    for name, _ in VARIANTS:
        fixed = sum(1 for c in cases for v in c["variants"] if v["name"] == name and v["move"] and v["loss"] < BAR)
        print(f"OBS   variant {name:12s}: brings the loss under {BAR} in {fixed} of {len(cases)}")
    # the policy's hard positives, scored by the deployed policy
    pol = json.load(open(paths.derived("forcing_policy.json")))
    mean, std = np.array(pol["mean"], dtype=np.float32), np.array(pol["std"], dtype=np.float32)
    w1, b1, w2, b2 = (np.array(pol[k], dtype=np.float32) for k in ("w1", "b1", "w2", "b2"))
    def prob(X):
        h = np.maximum((X - mean) / std @ w1.T + b1, 0)
        return 1 / (1 + np.exp(-(h @ w2 + b2)))
    hard = [r for c in cases for r in c.get("proven_lines", [])]
    if hard:
        X = np.array([r[0] for r in hard], dtype=np.float32)
        p = prob(X)
        stopped = p < pol.get("tau", 0.1)
        print(f"OBS   hard positives: {len(hard):,} proving lines in the policy-caused positions; the deployed policy would have stopped {stopped.sum():,} ({100 * stopped.mean():.1f}%)")
        np.savez_compressed(paths.derived("forcing_policy_hard.npz"), X=X, y=np.ones(len(X), dtype=np.int8), cost=np.array([r[2] for r in hard], dtype=np.int32), stopped=stopped)
    unfixed = [c for c in cases if c["cause"] == "lemma missing"]
    import chess
    for c in unfixed:
        b = chess.Board(c["fen"]); sans = []
        for u in c["best_line"]:
            m = chess.Move.from_uci(u)
            if m not in b.legal_moves: break
            sans.append(b.san(m)); b.push(m)
        c["best_san"] = " ".join(sans)
        c["our_san"] = chess.Board(c["fen"]).san(chess.Move.from_uci(c["move"])) if c["move"] else None
    with open(paths.experiment("surprise_unfixed.json"), "w") as fh:
        json.dump(unfixed, fh, indent=1)
    with open(paths.experiment("surprise_cases.json"), "w") as fh:
        json.dump(cases, fh, indent=1)
    print(f"OBS   {len(unfixed)} cases for lemma mining -> {paths.experiment('surprise_unfixed.json')}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="decision_review_policy_on.json")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    run(a.store, a.limit, a.workers)


if __name__ == "__main__":
    main()
