#!/usr/bin/env python
"""
experiments/compare_methods.py — paired, e-value-tested comparisons between
readout/policy methods on a learned field. Validates the README's own
documented MIN-vs-MEAN effect (switching the reply aggregation from MEAN to
MIN was worth +20 points of conversion) as a rigorous, anytime-valid
statistical claim rather than a single side-by-side run.

Requires experiments/train_krkn.py (or train_krk_pi.py for --domain krk) to
have produced a trained F/B field in data/derived/ first.
"""
from __future__ import annotations

import argparse

import numpy as np

from latentchess.abtest import MethodSpec, compare
from latentchess.chain import exact_P
from latentchess.cone.tabular import TabularFB
from latentchess.domains import krk, krkn
from latentchess.io.paths import load_array
from latentchess.opponents import EpsOptimalDTM, optimal_reply_table
from latentchess.planner.policy import DTMOraclePolicy, RandomPolicy, TablePolicy
from latentchess.planner.readout import ReplyAgg, greedy_policy
from latentchess.scoring import TerminalScores, fill_terminal_state_scores
from latentchess.cone.embedding import make_goal, reach

DOMAINS = {"krk": krk, "krkn": krkn}


def _load_field(domain_name: str):
    dom = DOMAINS[domain_name]
    chain = dom.build_chain(verbose=False)
    try:
        dtm = load_array(f"dtm_{domain_name}") if domain_name != "krk" else None
        F = load_array(f"{domain_name}_F")
        B = load_array(f"{domain_name}_B")
    except FileNotFoundError:
        F = B = dtm = None
    if F is None:
        if domain_name == "krk":
            P = exact_P(chain)
            emb = TabularFB.fit(P, gamma=0.98, d=64, seed=0)
            W, Bs = krk.enumerate_states()
            dtm, _ = krk.compute_dtm(W, Bs)
        else:
            raise FileNotFoundError(
                f"data/derived/{domain_name}_F.npy not found -- run experiments/train_{domain_name}.py first")
    else:
        emb = TabularFB(F=F, B=B)
    return chain, dtm, emb


def _builder(chain, emb, dtm, ts, agg, name):
    if name == "oracle":
        return MethodSpec("oracle", lambda: DTMOraclePolicy(chain, dtm))
    if name == "random":
        return MethodSpec("random", lambda: RandomPolicy())
    if name not in ("mean", "min"):
        raise ValueError(f"unknown method {name!r} (choices: mean, min, oracle, random)")
    goal = make_goal("MATE", np.array([chain.terminals.mate]), emb)
    scores = fill_terminal_state_scores(reach(emb, goal, None), chain, ts)
    agg_enum = ReplyAgg.MEAN if name == "mean" else ReplyAgg.MIN
    table = greedy_policy(scores, chain, agg_enum, ts)
    return MethodSpec(name, lambda t=table: TablePolicy(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), default="krkn")
    ap.add_argument("--methods", default="mean,min")
    ap.add_argument("--n-starts", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--eps-black", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=50)
    args = ap.parse_args()

    chain, dtm, emb = _load_field(args.domain)
    ts = TerminalScores.big()
    method_names = args.methods.split(",")
    methods = {name: _builder(chain, emb, dtm, ts, None, name) for name in method_names}

    b_opt = optimal_reply_table(chain, dtm)
    black_table = b_opt
    black_builder = lambda: EpsOptimalDTM(black_table, eps=args.eps_black)

    rng = np.random.default_rng(args.seed)
    live_dtm = dtm[:chain.n_live] if len(dtm) >= chain.n_live else dtm
    stratum = chain.strata.get("KRkn" if args.domain == "krkn" else None)
    pool = np.arange(chain.n_live) if stratum is None else np.arange(stratum.start, stratum.stop)
    pool = pool[np.isfinite(live_dtm[pool])]
    starts = rng.choice(pool, size=min(args.n_starts, len(pool)), replace=False)

    rows = compare(chain, dtm, methods, black_builder, starts, alpha=args.alpha,
                    batch=args.batch, early_stop=True, base_seed=args.seed)

    print(f"{'A':>8} {'B':>8} {'pairs':>6} {'decisive':>8} {'A-wins':>6} {'B-wins':>6} "
          f"{'e-value':>12} {'rejected':>8} {'mean_diff':>9} {'CI':>20}")
    for row in rows:
        print(f"{row.method_a:>8} {row.method_b:>8} {row.n_pairs:>6} {row.decisive:>8} "
              f"{row.a_wins:>6} {row.b_wins:>6} {row.e_value:>12.3g} {str(row.rejected):>8} "
              f"{row.mean_win_diff:>9.3f} [{row.ci[0]:.3f},{row.ci[1]:.3f}]")

    if set(method_names) == {"mean", "min"}:
        row = rows[0]
        min_is_a = row.method_a == "min"
        min_wins = row.a_wins if min_is_a else row.b_wins
        mean_wins = row.b_wins if min_is_a else row.a_wins
        if row.rejected and min_wins > mean_wins:
            print("\nVALIDATION: min beats mean, e-value rejects H0 -- matches the README's "
                  "documented MIN-vs-MEAN readout effect.")
        else:
            print("\nVALIDATION FAILED: expected min to beat mean with e-value rejection "
                  "within n-starts pairs -- investigate before trusting this harness.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
