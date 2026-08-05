#!/usr/bin/env python
"""catspace/research/tools/embeddings/expectimax_reachability.py -- reachability where a STEP is two
plies (mine + the opponent's reply), so the metric counts MY MOVES and the opponent is probabilistic
rather than adversarial (Kaveh 2026-08-05).

THE OBJECT. Minimax assumes the opponent plays the best reply. Replace that with an EXPECTATION over
what a population actually plays and the backup becomes expectimax:

    d_my(s) = 1 + min over my legal a of d_opp(s . a)

with the max/min on MY side and the expectation on THEIRS. The two halves live in different places,
which is what makes this trainable from data alone:

  * THE EXPECTATION IS LEARNED. d_opp is the field evaluated at an opponent-to-move node. It is
    trained (train_iqe_head --macro-step) on same-side-to-move pairs from real games, whose gaps
    have already integrated over whatever the opponent did in between. The games ARE samples of the
    opponent's policy, so no policy model is fitted. Which population -- lichess or SF-vs-SF -- is
    a switch on the conditioned field (--n-sources 2), not a different model.
  * THE MAX IS SEARCHED. Expanding my legal moves at the anchor is cheap and exact, and it is the
    only place control enters. Without it the field would measure JOINT OCCUPANCY -- where these
    two players happened to go together -- rather than where I can get to. That distinction is the
    z-tainting failure mode in MILESTONES M3 and it is the reason this script exists at all.

WHAT IT MEASURES, and the honest failure mode. d_opp is trained only on positions those games
visited, while the min over legal a queries positions one ply off that support, where a learned
value is typically OPTIMISTIC -- it will happily claim a move gets somewhere fast precisely where
it knows least. So off-support rate is reported as a first-class number next to the accuracy, not
assumed small. The comparison that matters is expectimax vs the direct field read, both scored
against the REALIZED macro-distance in held-out games.

It also reports the thing that motivated the design -- "this move might take me where all the 2nd
plies go" -- as the SPREAD of d over the opponent's legal replies to a fixed move of mine. That
spread is the branching the macro-step averages over, and a move whose spread is large is one whose
outcome the opponent still controls.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from catspace.research.tools.embeddings.basin_tent_fullgames import (
    replay, population_games_human, population_games_sf)
from catspace.research.tools.embeddings.basin_hazard_field import load_head

SRC = {"human": 0, "sf": 1}


@torch.no_grad()
def phi_of(field, head, boards):
    """LczeroBoards -> phi under the given head, one trunk pass."""
    planes = [b.to_input_tensor().to(dtype=torch.uint8).numpy() for b in boards]
    return head.phi(field.trunk_feats([p.astype(np.float32) for p in planes]))


@torch.no_grad()
def d_to(head, e_src, e_goal):
    """d(src -> goal) for a batch of sources against ONE goal embedding."""
    return head.d_pair_emb(e_src, e_goal.expand(len(e_src), -1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/iqe_macro_v2_latest.pt",
                    help="a --macro-step field; with --n-sources 2 the opponent is a switch")
    ap.add_argument("--onnx", default="assets/engines/lc0/t1-256x10.onnx")
    ap.add_argument("--opponent", choices=["human", "sf"], default="human")
    ap.add_argument("--sf-moves", default="data/derived/opening_pool_sfsf_moves.tsv")
    ap.add_argument("--human-records", default="data/records/lichess_2019-01")
    ap.add_argument("--n-games", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=6, help="goal is this many MACRO steps ahead")
    ap.add_argument("--anchors", type=int, default=6, help="anchor positions sampled per game")
    ap.add_argument("--max-moves", type=int, default=32, help="legal moves expanded per anchor")
    ap.add_argument("--support-k", type=int, default=2000,
                    help="reference phi sample used to measure how far expanded successors sit "
                         "from the positions the field was actually trained on")
    ap.add_argument("--max-ply", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    import chess
    from lczerolens import LczeroBoard
    from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField

    field = ReachabilityField(onnx=args.onnx, head=args.ckpt)
    head = load_head(args.ckpt, field.dev)
    sid = SRC[args.opponent]
    if head.pole_delta is None and sid:
        raise SystemExit(f"{args.ckpt} is not opponent-conditioned; --opponent sf is meaningless")
    print(f"[expectimax] {args.ckpt} | opponent = {args.opponent} "
          f"| horizon {args.horizon} macro steps [{time.time()-t0:.0f}s]", flush=True)

    rng = np.random.default_rng(args.seed)
    games = (population_games_human(args.human_records, args.n_games, rng) if args.opponent == "human"
             else population_games_sf(args.sf_moves, args.n_games, rng))

    ref, direct, expmax, true_d, spread, opt_gap = [], [], [], [], [], []
    for gid, res, ucis, _tm in games:
        planes, _b, _t = replay(ucis, args.max_ply)
        if planes is None or len(planes) < 2 * args.horizon + 4:
            continue
        n = len(planes)
        # Anchors at MY-to-move plies with a same-parity goal `horizon` macro steps later.
        lo, hi = 0, n - 2 * args.horizon - 1
        if hi <= lo:
            continue
        for t in rng.choice(np.arange(lo, hi), size=min(args.anchors, hi - lo), replace=False):
            t = int(t)
            board = LczeroBoard()
            for u in ucis[:t + 1]:
                board.push(chess.Move.from_uci(u))
            goal = LczeroBoard()
            for u in ucis[:t + 1 + 2 * args.horizon]:
                goal.push(chess.Move.from_uci(u))
            legal = list(board.legal_moves)
            if not legal:
                continue
            if len(legal) > args.max_moves:
                legal = [legal[i] for i in rng.choice(len(legal), args.max_moves, replace=False)]
            played = chess.Move.from_uci(ucis[t + 1]) if t + 1 < len(ucis) else None
            if played is not None and played not in legal:
                legal.append(played)

            succ = []
            for mv in legal:
                b2 = board.copy(stack=False)
                b2.push(mv)
                succ.append(b2)
            e_goal = phi_of(field, head, [goal])[0]
            e_anchor = phi_of(field, head, [board])
            e_succ = phi_of(field, head, succ)
            d_anchor = float(d_to(head, e_anchor, e_goal)[0])
            d_succ = d_to(head, e_succ, e_goal).cpu().numpy()

            direct.append(d_anchor)
            expmax.append(1.0 + float(d_succ.min()))
            true_d.append(args.horizon)
            spread.append(float(d_succ.max() - d_succ.min()))
            if played is not None:
                j = legal.index(played)
                opt_gap.append(float(d_succ[j] - d_succ.min()))
            ref.append(e_anchor[0].cpu().numpy())

    direct = np.array(direct); expmax = np.array(expmax); true_d = np.array(true_d, float)
    spread = np.array(spread); opt_gap = np.array(opt_gap)
    print(f"  {len(direct):,} anchors from {len(games)} games [{time.time()-t0:.0f}s]\n")

    print("PREDICTED macro-distance to a goal that is EXACTLY --horizon macro steps away")
    print(f"  {'read':>12s} {'median':>9s} {'mean':>9s} {'MAE vs truth':>13s}")
    for nm, v in [("direct d(s->g)", direct), ("expectimax", expmax)]:
        print(f"  {nm:>12s} {np.median(v):>9.3f} {v.mean():>9.3f} "
              f"{np.abs(v - true_d).mean():>13.3f}")
    print(f"\n  expectimax is BELOW the direct read on {100*(expmax < direct).mean():.1f}% of "
          f"anchors\n  (it should be: min over my legal moves can only find a shorter route than "
          f"the one\n  the anchor's own embedding averages over -- if it is not, the field is not "
          f"ordering successors)")

    print(f"\nBRANCHING -- spread of d over MY legal moves at one anchor "
          f"('where all the 2nd plies go')")
    for q in (10, 50, 90):
        print(f"  p{q:<2d} spread {np.percentile(spread, q):>7.3f}")
    if len(opt_gap):
        print(f"\n  the move ACTUALLY PLAYED sits {np.median(opt_gap):+.3f} above the best "
              f"successor (median);\n  {100*(opt_gap <= 1e-6).mean():.1f}% of the time it WAS the "
              f"field's preferred move")

    print(f"\ndone [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
