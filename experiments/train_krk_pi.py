#!/usr/bin/env python
"""
experiments/train_krk_pi.py — curriculum policy-iteration training on KRk.

Reproduces exp_policy_iteration.py's schedule/evaluation on the new
catspace package.
"""
from __future__ import annotations

import argparse

import numpy as np

from catspace.domains import krk
from catspace.opponents import optimal_reply_table
from catspace.train.curriculum import CurriculumTrainer, CurriculumConfig, Round
from catspace.planner.readout import ReplyAgg

SCHEDULE = [  # (eps_white, eps_black, n_games)
    Round(1.00, 1.00, 20000, None),   # pure random vs random (the original data)
    Round(0.30, 0.50, 20000, None),
    Round(0.30, 0.25, 20000, None),
    Round(0.20, 0.10, 20000, None),
    Round(0.20, 0.00, 20000, None),   # black fully optimal
    Round(0.20, 0.00, 20000, None),
]


def goal_region(chain, dtm):
    return np.concatenate([np.where(dtm <= 3)[0], [chain.terminals.mate]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    chain = krk.build_chain()
    W, B = krk.enumerate_states()
    dtm, _ = krk.compute_dtm(W, B)
    b_opt = optimal_reply_table(chain, dtm)

    cfg = CurriculumConfig(
        schedule=SCHEDULE, gamma=0.92, d=64, goal_region=goal_region,
        eval_n=300, train_cap=120, eval_cap=60, seed=100, agg=ReplyAgg.MEAN,
    )
    trainer = CurriculumTrainer(chain, dtm, b_opt, cfg, ckpt_path=args.ckpt)
    results = trainer.run()

    final = results[-1]
    print(f"\nfinal round: conversion={final.conversion:.3f} tempo={final.tempo:.2f}")


if __name__ == "__main__":
    main()
