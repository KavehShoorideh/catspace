#!/usr/bin/env python
"""
experiments/search_tournament.py — which SEARCH is better, statistically?

Kaveh 2026-07-14: "use the e-value harness ... to figure out which of these
searches are better in a well-trained embedding space." Round-robin paired
duels between search readouts at MATCHED eval budget, each duel scored with
catspace.abtest.EValueTest (anytime-valid: we stop a duel EARLY the moment
e >= 1/alpha, the sequential-bandit efficiency) plus a bootstrap CI on the
mate-rate diff at whatever n the duel stopped.

Two field modes:
  --field oracle   the tablebase itself as the reach function -- a PERFECT
                   field. This isolates SEARCH quality with field quality
                   pinned at the ceiling ("well-trained embedding in the
                   limit"). EVAL-ONLY tooling: the oracle never touches
                   training (audit rule). Arms: mcts, anytime (their cores
                   take a bare reach_fn; beam's core is welded to the torch
                   model and already lost to mcts CI-real on a learned field).
  --field ckpt     a learned checkpoint (--ckpt): all three arms via
                   make_search_policy. Rerun on the best distilled ckpt to
                   answer the question on OUR well-trained field.

Oracle reach (White POV, higher = closer to White's mate): wdl=+2 -> 2 + a
DTZ-progress bonus (smaller distance-to-zeroing = closer); draws/cursed -> 0;
losing -> -2. Coarse but monotone toward conversion -- direction, which is
what the searches consume.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catspace.abtest import EValueTest
from experiments.playout_ab import playout
from experiments.value_fixed_point import TB


def oracle_reach_fn(tb):
    def reach(boards):
        out = []
        for b in boards:
            w, d = tb.wdl_dtz(b)
            if w is None:
                out.append(0.0)
                continue
            if b.turn == chess.BLACK:
                w = -w
            if w == 2:
                prog = (100.0 - min(abs(d), 100.0)) / 100.0 if d is not None else 0.0
                out.append(2.0 + prog)
            elif w == -2:
                out.append(-2.0)
            else:
                out.append(0.0)
        return np.array(out)
    return reach


def make_arm(kind, field, nodes, tb, ckpt, device, c_puct, beam):
    """Returns a policy with .move(board, rng)."""
    if field == "oracle":
        reach = oracle_reach_fn(tb)
        if kind == "mcts":
            from catspace.nn.mcts import MCTS

            class P:
                def __init__(self):
                    self.m = MCTS(reach, max_nodes=nodes, c_puct=c_puct)

                def move(self, board, rng):
                    return self.m.best_move(board)
            return P()
        if kind == "anytime":
            from catspace.nn.anytime import AnytimePathSearch

            class P:
                def __init__(self):
                    self.s = AnytimePathSearch(reach, max_nodes=nodes, beam=beam)

                def move(self, board, rng):
                    return self.s.search(board)
            return P()
        raise ValueError(f"arm {kind!r} not available on the oracle field")
    import torch  # noqa: F401
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import make_search_policy
    dev = pick_device(device)
    fb, pay = load_ckpt(Path(ckpt), dev)
    return make_search_policy(kind, fb, pay["zgoals"]["MATE_W"], max_nodes=nodes,
                              beam=beam, c_puct=c_puct, device=dev)


def duel(name_a, name_b, pol_a, pol_b, starts, tb, max_plies, seed, alpha, boot=2000):
    """Paired sequential duel with e-process early stopping."""
    ev = EValueTest()
    a_hist, b_hist = [], []
    for i, fen in enumerate(starts):
        rng_a = np.random.default_rng([seed, i, 0])
        rng_b = np.random.default_rng([seed, i, 1])
        board = chess.Board(fen)
        ra, _ = playout(pol_a, board, tb, rng_a, max_plies)
        rb, _ = playout(pol_b, board, tb, rng_b, max_plies)
        a_hist.append(ra)
        b_hist.append(rb)
        ev.update(rb - ra)
        if ev.reject_at(alpha) and ev.n >= 5:
            break
    a, b = np.array(a_hist), np.array(b_hist)
    n = len(a)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(boot, n))
    bs = b[idx].mean(1) - a[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    stopped = "early-stop" if n < len(starts) else "exhausted"
    print(f"DUEL {name_a} vs {name_b}: rate {a.mean():.3f} vs {b.mean():.3f}  "
          f"diff={b.mean()-a.mean():+.3f} CI=[{lo:+.3f},{hi:+.3f}]  "
          f"e={ev.e:.2f} ({ev.n} decisive, n={n}, {stopped})", flush=True)
    return dict(a=name_a, b=name_b, rate_a=float(a.mean()), rate_b=float(b.mean()),
                e=ev.e, n=n, ci=[float(lo), float(hi)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["mcts", "anytime"])
    ap.add_argument("--field", choices=("oracle", "ckpt"), default="oracle")
    ap.add_argument("--ckpt", default="data/derived/lichess_fb_4gb_qm_plygap_only.pt")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_fixed_test_n200.json")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="artifacts/experiments/search_tournament.jsonl")
    args = ap.parse_args()

    tb = TB(args.syzygy_dir)
    starts = json.loads(Path(args.fixed_set).read_text())["fens"][:args.n]
    print(f"TOURNAMENT {args.label} field={args.field} nodes={args.nodes} "
          f"arms={args.arms} (n<= {len(starts)}, alpha={args.alpha})")
    results = []
    for ka, kb in itertools.combinations(args.arms, 2):
        pa = make_arm(ka, args.field, args.nodes, tb, args.ckpt, args.device,
                      args.c_puct, args.beam)
        pb = make_arm(kb, args.field, args.nodes, tb, args.ckpt, args.device,
                      args.c_puct, args.beam)
        results.append(duel(ka, kb, pa, pb, starts, tb, args.max_plies,
                            args.seed, args.alpha))
    tb.close()
    with open(args.out, "a") as f:
        f.write(json.dumps(dict(label=args.label, field=args.field,
                                nodes=args.nodes, duels=results)) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
