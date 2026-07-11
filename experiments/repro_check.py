#!/usr/bin/env python
"""
experiments/repro_check.py — re-runs the ported pipeline and diffs against
tests/baselines/expected.json: exact match for deterministic quantities
(state counts, DTM extremes, the rank-64/concept-dims numbers, which are
reproduced bit-for-bit by krk_rung1.py), loose bands for anything downstream
of stochastic training (a trained field is a different sample each run --
see tests/baselines/expected.json's own note).

Fast path (default, ~30s): domain counts + krk_rung1.py's deterministic
sections. --full (~10 min) additionally re-trains KRkn from scratch
(experiments/train_krkn.py) and band-checks the curriculum/search-sweep
numbers against the historical run.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from catspace.domains import krk, krkn

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = json.loads((ROOT / "tests" / "baselines" / "expected.json").read_text())

FAILURES: list[str] = []


def check_exact(name, got, expected):
    ok = got == expected
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got} expected={expected}")
    if not ok:
        FAILURES.append(name)


def check_band(name, got, lo, hi):
    ok = lo <= got <= hi
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got:.4f} band=[{lo:.4f},{hi:.4f}]")
    if not ok:
        FAILURES.append(name)


def run_capture(args) -> str:
    result = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True, timeout=300)
    return result.stdout


def grab(pattern: str, text: str, cast=float):
    m = re.search(pattern, text)
    if m is None:
        raise ValueError(f"pattern not found: {pattern!r}")
    return cast(m.group(1))


def check_domains():
    print("\n== domain counts (exact) ==")
    W, B = krk.enumerate_states()
    dtm_w, _ = krk.compute_dtm(W, B)
    exp = EXPECTED["krk_domain"]
    check_exact("krk.w_states", len(W), exp["w_states"])
    check_exact("krk.b_nodes", len(B), exp["b_nodes"])
    check_exact("krk.dtm_max", float(dtm_w.max()), exp["dtm_max"])
    check_exact("krk.forcible_mate_w_states", int(np.isfinite(dtm_w).sum()), exp["forcible_mate_w_states"])

    chain = krkn.build_chain(verbose=False)
    dtm = krkn.compute_dtm(chain)
    exp2 = EXPECTED["krkn_domain"]
    n2 = chain.strata["KRkn"].stop
    n1 = chain.n_live - n2
    check_exact("krkn.n2_krkn_w_states", n2, exp2["n2_krkn_w_states"])
    check_exact("krkn.n1_krk_stratum", n1, exp2["n1_krk_stratum"])
    check_exact("krkn.union_n", chain.n, exp2["union_n"])   # n_live + absorbing (MATE/DRAW/BWIN)
    check_exact("krkn.dtm_max", int(dtm[np.isfinite(dtm)].max()), exp2["dtm_max"])
    # won_fraction is specifically over the KRkn stratum (dtm[:n2]) -- the KRk
    # sub-stratum is 100% forcible-mate and would inflate the combined figure.
    check_band("krkn.won_fraction", float(np.isfinite(dtm[:n2]).mean()),
               exp2["won_fraction"] - 0.01, exp2["won_fraction"] + 0.01)


def check_krk_rung1():
    print("\n== krk_rung1.py (deterministic sections exact; engine rate near-exact) ==")
    out = run_capture(["experiments/krk_rung1.py"])
    exp = EXPECTED["krk_experiment"]
    rank64 = grab(r"d= *64.*reach_rel_err=([\d.]+)", out)
    check_band("rank64_reach_rel_err", rank64, exp["rank64_reach_rel_err"] - 0.005,
               exp["rank64_reach_rel_err"] + 0.005)
    n_strong = grab(r"distinct dims with \|rho\|>0.5: (\d+)", out, int)
    check_exact("distinct_concept_dims_rho_gt_0.5", n_strong, exp["distinct_concept_dims_rho_gt_0.5"])
    mate_32k = grab(r"learned \( 32000 games.*mate-rate=([\d.]+)", out)
    check_band("mate_rate_32000_games", mate_32k, exp["mate_rate_32000_games"] - 0.02,
               exp["mate_rate_32000_games"] + 0.02)
    base_rate = grab(r"random white +: mate-rate=([\d.]+)", out)
    check_band("mate_rate_random_baseline", base_rate, exp["mate_rate_random_baseline"] - 0.02,
               exp["mate_rate_random_baseline"] + 0.02)


def check_krkn_search_sweep():
    print("\n== krkn_search_sweep.py (loose bands -- depends on the CURRENT trained field) ==")
    if not (ROOT / "data" / "derived" / "krkn_F.npy").exists():
        print("SKIP: data/derived/krkn_F.npy missing -- run experiments/train_krkn.py first")
        return
    out = run_capture(["experiments/krkn_search_sweep.py"])
    k0 = grab(r"0 \| +1 +\| +([\d.]+)", out)
    check_band("k0_1ply_conversion (loose)", k0, 0.3, 1.0)
    k2 = grab(r"\n +2 \| +5 +\| +([\d.]+)", out)
    check_band("k2_5ply_collapse (documented pessimistic-collapse pattern)", k2, 0.0, 0.2)


def check_full_curriculum():
    print("\n== [--full] retraining KRkn from scratch (~7 min) ==")
    out = run_capture(["experiments/train_krkn.py"])
    exp = EXPECTED["krkn_curriculum_exp_krkn2"]
    conv = grab(r"conversion=([\d.]+)", out)
    check_band("krkn_curriculum.vs_optimal_mate_rate (loose)", conv,
               exp["vs_optimal_mate_rate"] - 0.2, min(1.0, exp["vs_optimal_mate_rate"] + 0.3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also re-train KRkn from scratch (~10 min)")
    args = ap.parse_args()

    check_domains()
    check_krk_rung1()
    check_krkn_search_sweep()
    if args.full:
        check_full_curriculum()

    print(f"\n{'='*60}")
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("PASS: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
