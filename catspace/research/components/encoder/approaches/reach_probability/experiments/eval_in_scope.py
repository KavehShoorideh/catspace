#!/usr/bin/env python
"""eval_in_scope.py -- score a concept-dynamics checkpoint on the task it is ACTUALLY FOR.

Kaveh 2026-08-16: "my model's scope is up until tablebase, at which point we look up optimal
play."

WHY THIS EXISTS. `concept_dynamics_field` trains a value head anchored on exact Syzygy WDL, and
reports its accuracy on those anchored positions. But anchored positions are <=5 men -- inside
the tablebase, where the answer is looked up and the model is never consulted. That number
(0.940 against a 0.929 material-lookup baseline) measures how well the model reproduces a table
we already own. It is a sanity check, not a result, and it must never be quoted as one.

    IN SCOPE  (>5 men)   4,364,284   93.81%   <- the model's job
    OUT       (<=5 men)    287,977    6.19%   <- looked up

THE IN-SCOPE TASK WITH EXACT GROUND TRUTH. Of the in-scope positions, 1,083,252 (24.8%) belong
to games that DO eventually reach the tablebase. For those we know with certainty what the game
resolves to -- Syzygy at the entry -- even though the position itself is far above the boundary.
So we can grade "what will this resolve to?" exactly, on the real task.

    in-scope target distribution   W 40.3%  L 37.5%  D 22.3%
    majority-class baseline        40.3%

No material shortcut exists here: at 8+ men, piece counts say very little about which endgame
you convert into. Contrast the 92.9% material baseline that applies INSIDE the tablebase.

Reported by distance-to-resolution, because "what does this become" is a very different question
40 plies out than 4 plies out, and a single averaged number hides that entirely.

    .venv/bin/python -m ...eval_in_scope --ckpt artifacts/experiments/cdf1_step2000.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=20000, help="positions to score")
    ap.add_argument("--split", type=int, default=2, help="0 train / 1 val / 2 test")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_dynamics_field import ConceptDynamics
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"]
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n_pos = len(tr.tok)
    z = np.load(paths.derived(f"tb_entry_5-4-3men_{n_pos}.npz"), allow_pickle=True)
    vocab = [str(x) for x in z["vocab"]]
    cls, plies, men = z["tb_class"].astype(np.int64), z["tb_plies"], z["men"]

    # exact target: the SOLVED outcome at the tablebase entry this game reaches
    lab = np.array([vocab[c] if c >= 0 else "" for c in cls])
    exact = np.array([("=" in v) for v in lab])
    OUT = {"L": 0, "D": 1, "W": 2}
    y = np.full(n_pos, -1, np.int64)
    sel = np.flatnonzero(exact)
    y[sel] = [OUT[lab[i].split("=")[1]] for i in sel]

    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nv = max(1, int(A["val_frac"] * len(tr.start))); nt = max(1, int(A["test_frac"] * len(tr.start)))
    sog = np.zeros(len(tr.start), np.int8); sog[perm[:nv]] = 1; sog[perm[nv:nv + nt]] = 2
    spl = sog[gor]

    IN = (men > 5) & (y >= 0) & (spl == args.split)          # in scope, exactly labelled
    rows = np.flatnonzero(IN)
    rng = np.random.default_rng(0)
    rows = rows[rng.choice(len(rows), min(args.n, len(rows)), replace=False)]
    print(f"[scope] ckpt step {ck.get('step')} | scoring {len(rows):,} IN-SCOPE positions "
          f"(>5 men, exact target, split {args.split})", flush=True)

    net = ConceptDynamics(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                          square_codes=A["square_codes"], hidden=A["hidden"], ema=A["ema"])
    net.load_state_dict(ck["state_dict"]); net.eval()

    preds = []
    with torch.no_grad():
        for a in range(0, len(rows), args.batch):
            r = rows[a:a + args.batch]
            zq, _, _, _ = net.encode(
                torch.from_numpy(tr.tok[r].astype(np.int64)),
                torch.from_numpy(tr.glob[r].astype(np.float32)))
            preds.append(net.value(zq).argmax(-1).numpy())
    pred = np.concatenate(preds); truth = y[rows]

    from collections import Counter
    maj = Counter(truth.tolist()).most_common(1)[0]
    acc = float((pred == truth).mean())
    print(f"\n  IN-SCOPE accuracy          {acc:.3f}")
    print(f"  majority-class baseline    {maj[1]/len(truth):.3f}   (always predict "
          f"{'LDW'[maj[0]]})")
    print(f"  lift over baseline         {100*(acc - maj[1]/len(truth)):+.1f} points")

    # by distance to the resolution -- the averaged number hides the whole story
    print(f"\n  {'plies to resolution':<22} {'n':>7} {'accuracy':>9} {'baseline':>9}")
    pl = plies[rows]
    for lo, hi in ((0, 5), (6, 15), (16, 40), (41, 100), (101, 10 ** 6)):
        m = (pl >= lo) & (pl <= hi)
        if m.sum() < 50:
            continue
        b = Counter(truth[m].tolist()).most_common(1)[0][1] / m.sum()
        print(f"  {f'{lo}-{hi if hi < 10**6 else 'inf'}':<22} {int(m.sum()):>7,} "
              f"{float((pred[m] == truth[m]).mean()):>9.3f} {b:>9.3f}")


if __name__ == "__main__":
    main()
