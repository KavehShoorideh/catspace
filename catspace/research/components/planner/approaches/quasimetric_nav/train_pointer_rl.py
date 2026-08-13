#!/usr/bin/env python
"""train_pointer_rl.py -- THE RL LOOP (Kaveh 2026-08-13: "do we have a planner that actually
does reinforcement learning or not?"). First runnable episode trainer for the pointer policy:

  per move (policy side): SubgoalFormer certificate over the candidate tokens -> alert set
  (certificate diff vs previous move) -> PointerPolicy picks (pursue g | deny g | hold) AND a
  search budget (b=0 premove-tier = minimal verification) -> goal-conditioned coherent search
  executes -> move. Opponent: the plain champion engine.

  reward per game: R = outcome - lambda * total_leaf_evals   (win with the least effort)
  update: REINFORCE with a running baseline, per-game.

Gradient boundaries hold: field, jqt sidecar and SubgoalFormer are FROZEN here -- only the
pointer policy trains. Headline metric: score vs champion AND evals/move (the strength-per-
node frontier).

    .venv/bin/python -m ...train_pointer_rl --ckpt <champion.pt> [--games 60]
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--opp-budget", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=2e-6)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    import re, os, chess
    from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess
    from catspace.research.components.planner.approaches.quasimetric_nav.subgoal_former import (
        GeoQuery, SubgoalFormer, alert_set)
    from catspace.research.components.planner.approaches.quasimetric_nav.pointer_policy import (
        PointerPolicy, BUDGETS, effort_reward, reinforce_loss)

    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    stem = re.sub(r"_(latest|step\d+)$", "", base)
    eng = KittyChess(args.ckpt, args.device)      # the policy's engine (shared field)
    opp = KittyChess(args.ckpt, args.device)      # the plain champion opponent
    eng.concept_eval = opp.concept_eval = False
    opp.tb = eng.tb
    jqt_path = next(p for p in (base + "_jqt.pt", stem + "_jqt.pt") if os.path.exists(p))
    lev_path = base + "_concept_leverage.npz"
    gq = GeoQuery(eng, jqt_path, lev_path if os.path.exists(lev_path) else None, args.device)
    former = SubgoalFormer(n_head=gq.H, n_code=gq.C)
    fp = base + "_former.pt"
    former.load_state_dict(torch.load(fp, map_location="cpu"))
    former.eval()
    print(f"[rl] substrate: {jqt_path} + {fp} (both FROZEN)", flush=True)

    pol = PointerPolicy()
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    hc = gq.candidates(k_lev=12)
    sides = torch.zeros(len(hc), dtype=torch.long)
    hct = torch.as_tensor(hc)

    rng = random.Random(0)
    baseline = 0.0
    scores, evals_per_move = [], []
    t0 = time.time()
    for game in range(args.games):
        b = chess.Board()
        for _ in range(4):
            b.push(rng.choice(list(b.legal_moves)))
        pol_white = bool(game % 2 == 0)
        logps, total_evals, n_pol_moves = [], 0, 0
        cert_prev = None
        committed = None
        while not b.is_game_over(claim_draw=True) and b.ply() < args.max_plies:
            if (b.turn == chess.WHITE) == pol_white:
                G, F = gq.geometry(b, hc)
                ci = committed if committed is not None else 0
                cert = former.certificate(hct, sides, F, G, committed_idx=ci)
                alerts = alert_set(cert_prev, cert, F, k=12, lev=gq.lev) \
                    if cert_prev is not None else []
                a, bud, logp = pol.act(cert, alerts)
                logps.append(logp)
                if a < len(alerts):                    # point: pursue/deny that concept
                    committed = int(np.flatnonzero(
                        (hc == np.array(alerts[a].hc)).all(1))[0]) \
                        if (hc == np.array(alerts[a].hc)).all(1).any() else committed
                goal = tuple(hc[committed]) if committed is not None else None
                budget = max(0.3, bud)                 # b=0 premove-tier -> minimal verify
                rows = eng.search_coherent(b, budget=budget,
                                           goal=goal if bud > 0 else None)
                rows = eng.rank_by_child_E(b, rows)
                total_evals += eng.last_evals
                n_pol_moves += 1
                cert_prev = cert
                if not rows:
                    mv = eng._tb_move(b)
                    if mv is None:
                        break
                    b.push(mv); continue
                b.push(rows[0]["mv"])
            else:
                rows = opp.search_coherent(b, budget=args.opp_budget)
                rows = opp.rank_by_child_E(b, rows)
                if not rows:
                    mv = opp._tb_move(b)
                    if mv is None:
                        break
                    b.push(mv); continue
                b.push(rows[0]["mv"])
        o = b.outcome(claim_draw=True)
        res_white = 0.5 if o is None or o.winner is None else (1.0 if o.winner else 0.0)
        outcome = res_white if pol_white else 1.0 - res_white
        R = effort_reward(outcome, total_evals, lam=args.lam)
        baseline = 0.9 * baseline + 0.1 * R
        if logps:
            opt.zero_grad()
            reinforce_loss(logps, [R] * len(logps), baseline=baseline).backward()
            opt.step()
        scores.append(outcome)
        epm = total_evals / max(n_pol_moves, 1)
        evals_per_move.append(epm)
        print(f"[rl] game {game+1}/{args.games}: {'W' if outcome==1 else 'D' if outcome==0.5 else 'L'} "
              f"R {R:+.3f} evals/move {epm:,.0f} | running score "
              f"{np.mean(scores):.2f} evals/mv {np.mean(evals_per_move):,.0f} "
              f"[{(time.time()-t0)/60:.0f}m]", flush=True)
    out = args.out or (base + "_pointer.pt")
    torch.save(pol.state_dict(), out)
    print(f"[rl] VERDICT score {np.mean(scores):.3f} vs champion "
          f"(chance 0.5 if policy adds nothing) | mean evals/move "
          f"{np.mean(evals_per_move):,.0f} | frontier point recorded -> {out}")


if __name__ == "__main__":
    main()
