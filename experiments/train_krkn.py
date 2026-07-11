#!/usr/bin/env python
"""
experiments/train_krkn.py — curriculum policy-iteration training on KRkn.

Reproduces exp_krkn2.py's schedule and evaluation on top of the new
latentchess package (CurriculumTrainer, unified CSR chain, TerminalScores).
"""
from __future__ import annotations

import argparse

import numpy as np

from latentchess.domains import krkn
from latentchess.opponents import optimal_reply_table
from latentchess.train.curriculum import CurriculumTrainer, CurriculumConfig, Round
from latentchess.planner.readout import ReplyAgg

SCHEDULE = [  # (eps_white, eps_black, n_games, dtm_cap)
    Round(0.50, 1.00, 15000, 5),
    Round(0.40, 0.70, 15000, 9),
    Round(0.30, 0.50, 15000, 13),
    Round(0.25, 0.30, 15000, 19),
    Round(0.20, 0.15, 15000, 27),
    Round(0.20, 0.05, 15000, None),
    Round(0.15, 0.00, 15000, None),
    Round(0.15, 0.00, 15000, None),
]


def goal_region(chain, dtm):
    return np.concatenate([np.where(dtm <= 3)[0], [chain.terminals.mate]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="path to a resumable .npz checkpoint")
    args = ap.parse_args()

    chain = krkn.build_chain(verbose=True)
    dtm = krkn.compute_dtm(chain)
    b_opt = optimal_reply_table(chain, dtm)

    n2 = chain.strata["KRkn"].stop
    cfg = CurriculumConfig(
        schedule=SCHEDULE, gamma=0.93, d=48, goal_region=goal_region,
        eval_n=300, train_cap=120, eval_cap=70, seed=100, agg=ReplyAgg.MEAN,
        track_stratum_cross="KRk", start_pool=np.arange(n2),
    )
    trainer = CurriculumTrainer(chain, dtm, b_opt, cfg, ckpt_path=args.ckpt)
    results = trainer.run()

    final = results[-1]
    print(f"\nfinal round: conversion={final.conversion:.3f} tempo={final.tempo:.2f} "
          f"rook_loss={final.rook_loss:.3f} extra={final.extra}")


if __name__ == "__main__":
    main()
