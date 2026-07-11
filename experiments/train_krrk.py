#!/usr/bin/env python
"""experiments/train_krrk.py — curriculum PI training on KRRk (stratified union chain)."""
from __future__ import annotations

import argparse

import numpy as np

from catspace.domains import krrk
from catspace.opponents import optimal_reply_table
from catspace.train.curriculum import CurriculumTrainer, CurriculumConfig, Round
from catspace.planner.readout import ReplyAgg


SCHEDULE = [
    Round(1.0, 0.7, 15000, None),
    Round(0.3, 0.4, 15000, None),
    Round(0.25, 0.2, 15000, None),
    Round(0.2, 0.0, 15000, None),
    Round(0.2, 0.0, 15000, None),
]


def goal_region(chain, dtm):
    return np.concatenate([np.where(dtm <= 3)[0], [chain.terminals.mate]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    chain = krrk.build_chain(verbose=True)
    dtm = krrk.compute_dtm(chain)
    b_opt = optimal_reply_table(chain, dtm)
    n2 = chain.strata["KRRk"].stop

    cfg = CurriculumConfig(
        schedule=SCHEDULE, gamma=0.90, d=64, goal_region=goal_region,
        eval_n=300, train_cap=120, eval_cap=40, seed=100, agg=ReplyAgg.MEAN,
        track_stratum_cross="KRk", start_pool=np.arange(n2),
    )
    trainer = CurriculumTrainer(chain, dtm, b_opt, cfg, ckpt_path=args.ckpt)
    results = trainer.run()
    final = results[-1]
    print(f"\nfinal round: conversion={final.conversion:.3f} tempo={final.tempo:.2f} extra={final.extra}")


if __name__ == "__main__":
    main()
