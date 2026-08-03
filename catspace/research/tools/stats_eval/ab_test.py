#!/usr/bin/env python
"""catspace/research/tools/stats_eval/ab_test.py -- ANYTIME-VALID A/B TEST between two model endpoints (Kaveh:
'a separate model endpoint up, a dedicated ab test ui can ping them both, gather evidence,
and decide winner. doesn't need to be fixed horizontally, can be anytime valid').

Design (identification stated per the define-identifications rule):
  - Paired trials: the SAME winnable tablebase position goes to both endpoints'
    /set_fen + /calculate; each model's top move is scored against syzygy ground truth
    (success = the move preserves the win: resulting wdl still lost for the opponent).
  - Concordant pairs carry no comparative evidence and are discarded (classic sign test).
  - Discordant pairs are Bernoulli(theta) with H0: theta = 1/2 (theta = P(B is the
    winner of a discordant pair)). Evidence = the Beta(1,1)-mixture likelihood-ratio
    e-process:  E_n = 2^n * B(k+1, n-k+1) / B(1,1) = 2^n * k!(n-k)!/(n+1)!
    This is a nonnegative martingale with E[E_0]=1 under H0, so by Ville's inequality
    P(sup_n E_n >= 1/alpha) <= alpha AT ANY STOPPING TIME -- peek freely, stop whenever.
  - Decision: E >= 1/alpha -> the side with more discordant wins is declared winner at
    level alpha. No fixed horizon.
Progress + verdicts stream to artifacts/experiments/ab_live.json (the /ab page polls it)
and every pair is appended to the jsonl log with full provenance.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from math import exp, lgamma, log
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
import requests


from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.gen_dtm_data import random_class_start
from catspace.io import paths


def log_e_value(n: int, k: int) -> float:
    """log of the Beta(1,1)-mixture e-process for the discordant sign test."""
    if n == 0:
        return 0.0
    return n * log(2.0) + lgamma(k + 1) + lgamma(n - k + 1) - lgamma(n + 2)


def top_move(endpoint: str, fen: str, nodes: int) -> chess.Move | None:
    r = requests.post(f"{endpoint}/set_fen", json={"fen": fen}, timeout=60)
    if not r.ok:
        return None
    r = requests.post(f"{endpoint}/calculate", json={"nodes": nodes}, timeout=600)
    if not r.ok or not r.json().get("top"):
        return None
    b = chess.Board(fen)
    return b.parse_san(r.json()["top"][0]["san"])


def preserves_win(b: chess.Board, mv: chess.Move, tb) -> bool:
    c = b.copy(stack=False)
    c.push(mv)
    if c.is_checkmate():
        return True
    if c.is_stalemate() or c.is_insufficient_material():
        return False
    try:
        return tb.probe_wdl(c) < 0        # opponent to move and still lost
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint-a", default="http://localhost:8777",
                    help="incumbent")
    ap.add_argument("--endpoint-b", default="http://localhost:8778",
                    help="challenger")
    ap.add_argument("--classes", default="KRRvKB,KRRvKP,KRRvKBP,KQvKR,KRvKN,KRvKB")
    ap.add_argument("--nodes", type=int, default=800)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=paths.experiment("ab_pairs.jsonl"))
    ap.add_argument("--live", default=paths.experiment("ab_live.json"))
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    classes = args.classes.split(",")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    va = requests.get(f"{args.endpoint_a}/state", timeout=30).json().get("version", {})
    vb = requests.get(f"{args.endpoint_b}/state", timeout=30).json().get("version", {})
    print(f"[ab] A={va} vs B={vb}  alpha={args.alpha}  classes={classes}", flush=True)
    boundary = 1.0 / args.alpha

    n_pairs = n_disc = k_b = ok_a_tot = ok_b_tot = 0
    verdict = "gathering"
    with chess.syzygy.open_tablebase(str(paths.syzygy_dir())) as tb:
        while n_pairs < args.max_pairs:
            cls = classes[int(rng.integers(len(classes)))]
            b = random_class_start(rng, cls)
            if b is None or not b.turn:
                continue
            try:
                if tb.probe_wdl(b) <= 0:      # only winnable starts carry ground truth
                    continue
            except Exception:
                continue
            fen = b.fen()
            mv_a = top_move(args.endpoint_a, fen, args.nodes)
            mv_b = top_move(args.endpoint_b, fen, args.nodes)
            if mv_a is None or mv_b is None:
                continue
            ok_a = preserves_win(b, mv_a, tb)
            ok_b = preserves_win(b, mv_b, tb)
            n_pairs += 1; ok_a_tot += ok_a; ok_b_tot += ok_b
            if ok_a != ok_b:
                n_disc += 1
                k_b += int(ok_b)
            loge = log_e_value(n_disc, k_b)
            e = exp(min(loge, 700))
            rec = dict(fen=fen, cls=cls, mv_a=mv_a.uci(), mv_b=mv_b.uci(),
                       ok_a=int(ok_a), ok_b=int(ok_b), n=n_pairs, n_disc=n_disc,
                       k_b=k_b, e=round(e, 4), commit=commit, ts=time.time())
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if e >= boundary:
                verdict = "B" if k_b * 2 > n_disc else "A"
            Path(args.live).write_text(json.dumps(dict(
                a=va, b=vb, n=n_pairs, n_disc=n_disc, k_b=k_b, e=round(e, 4),
                boundary=boundary, ok_a=ok_a_tot, ok_b=ok_b_tot,
                verdict=verdict, alpha=args.alpha, elapsed=round(time.time() - t0))))
            if n_pairs % 10 == 0 or verdict != "gathering":
                print(f"  n={n_pairs} disc={n_disc} k_B={k_b} E={e:.3f} "
                      f"(A ok {ok_a_tot} | B ok {ok_b_tot}) [{time.time()-t0:.0f}s]",
                      flush=True)
            if verdict != "gathering":
                break
    print(f"VERDICT AB_TEST winner={verdict} n={n_pairs} discordant={n_disc} "
          f"k_B={k_b} E={exp(min(log_e_value(n_disc, k_b), 700)):.2f} "
          f"boundary={boundary} A_ok={ok_a_tot}/{n_pairs} B_ok={ok_b_tot}/{n_pairs} "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
