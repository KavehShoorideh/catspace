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
    ap.add_argument("--ckpt-every", type=int, default=10, help="games between policy checkpoints")
    ap.add_argument("--v2", type=int, default=1,
                    help="1 = hierarchical loop: live candidates, executed deny, real premove,"
                         " achievement credit, reach-event logging")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="achievement bonus weight (base-rate corrected, per commitment)")
    ap.add_argument("--commit-window", type=int, default=16,
                    help="plies a commitment has to activate before it is scored abandoned")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    import re, os, chess
    from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess
    from catspace.research.components.planner.approaches.quasimetric_nav.subgoal_former import (
        GeoQuery, SubgoalFormer, alert_set)
    from catspace.research.components.planner.approaches.quasimetric_nav.pointer_policy import (
        PointerPolicy, BUDGETS, effort_reward, reinforce_loss, achievement_bonus)

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
    hc = gq.candidates(k_lev=12)                 # v1 fallback set
    BR = None
    brp = base + "_code_baserates.npy"
    if os.path.exists(brp):
        BR = np.load(brp)
    relog = open(base + "_rl_reach_events.jsonl", "a", buffering=1)

    rng = random.Random(0)
    baseline = 0.0
    scores, evals_per_move = [], []
    t0 = time.time()
    import json as _json
    mlog = open((args.out or (base + "_pointer.pt")).replace(".pt", "_rl.jsonl"), "a",
                buffering=1)          # EVERY GAME, structured (Kaveh 2026-08-13)
    for game in range(args.games):
        b = chess.Board()
        for _ in range(4):
            b.push(rng.choice(list(b.legal_moves)))
        pol_white = bool(game % 2 == 0)
        logps, total_evals, n_pol_moves = [], 0, 0
        n_point, n_hold, bud_hist = 0, 0, {}
        cert_prev = None
        committed = None                     # (goal hc, deny?, commit_ply, [logp indices])
        segments = []                        # closed commitment episodes: (idx list, bonus)
        cur_seg = []
        n_ach = n_aband = 0
        traj_codes = []
        while not b.is_game_over(claim_draw=True) and b.ply() < args.max_plies:
            if (b.turn == chess.WHITE) == pol_white:
                hc_l = gq.candidates_live(b, k=12, k_lev=4) if args.v2 else hc
                sides_l = torch.zeros(len(hc_l), dtype=torch.long)
                hct_l = torch.as_tensor(hc_l)
                G, F = gq.geometry(b, hc_l)
                # committed goal may not be in this position's candidate set: track by VALUE
                ci = 0
                if committed is not None:
                    hits = np.flatnonzero((hc_l == np.array(committed[0])).all(1))
                    ci = int(hits[0]) if len(hits) else 0
                cert = former.certificate(hct_l, sides_l, F, G, committed_idx=ci)
                alerts = alert_set(cert_prev, cert, F, k=12, lev=gq.lev) \
                    if cert_prev is not None else []
                a, bud, logp = pol.act(cert, alerts)
                li = len(logps)
                logps.append(logp)
                bud_hist[str(bud)] = bud_hist.get(str(bud), 0) + 1
                # activation check for the standing commitment (against live codes)
                with torch.no_grad():
                    tokb, glb = __import__("catspace.research.components.encoder.approaches."
                        "jepa_tokenizer.src.jepa", fromlist=["tokenize"]).tokenize(b)
                    phi_b = eng.net.backbone(
                        torch.from_numpy(np.asarray([tokb], dtype="int64")).to(args.device),
                        torch.from_numpy(np.asarray([glb], dtype="float32")).to(args.device))
                    _, ids_b = gq.jqt.target_codes(phi_b)
                ids_b = ids_b[0].cpu().numpy()
                traj_codes.append([int(x) for x in ids_b])
                if committed is not None and args.v2:
                    gh, gc = committed[0]
                    achieved = int(ids_b[gh]) == int(gc)
                    expired = b.ply() - committed[2] > args.commit_window
                    if achieved or expired:
                        base_rate = float(BR[gh, gc]) if BR is not None else 0.05
                        bonus = achievement_bonus(achieved, committed[1], base_rate,
                                                  args.alpha)
                        segments.append((list(cur_seg), float(bonus)))
                        n_ach += int(achieved and not committed[1])
                        n_aband += int(expired and not achieved)
                        committed = None
                        cur_seg = []
                if a < len(alerts):
                    n_point += 1
                    deny = alerts[a].kind == "worry"
                    committed = (tuple(int(x) for x in alerts[a].hc), deny, b.ply(), None)
                    cur_seg = [li]
                else:
                    n_hold += 1
                    if committed is not None:
                        cur_seg.append(li)
                goal = committed[0] if committed is not None else None
                deny_f = committed[1] if committed is not None else False
                if bud == 0 and args.v2 and goal is not None:
                    mv, n_ch = gq.move_toward(b, goal, minimize=deny_f)   # REAL premove tier
                    total_evals += n_ch
                    n_pol_moves += 1
                    cert_prev = cert
                    if mv is None:
                        break
                    b.push(mv)
                    continue
                w_g = -25.0 if deny_f else 25.0        # EXECUTED deny: bias AWAY
                rows = eng.search_coherent(b, budget=max(0.3, bud),
                                           goal=goal if bud > 0 else None, w_goal=w_g)
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
            # HIERARCHICAL CREDIT: every move gets the game advantage; moves inside a
            # commitment episode ALSO get that episode's base-rate-corrected achievement
            # bonus (options-style credit; Kaveh 2026-08-13).
            rewards = [R] * len(logps)
            for idxs, bonus in segments:
                for i2 in idxs:
                    rewards[i2] += bonus
            opt.zero_grad()
            reinforce_loss(logps, rewards, baseline=baseline).backward()
            opt.step()
        relog.write(_json.dumps({"game": game + 1, "outcome": outcome,
                                 "pol_white": pol_white, "plies": b.ply(),
                                 "n_ach": n_ach, "n_abandoned": n_aband,
                                 "traj_codes": traj_codes[-40:]}) + "\n")
        scores.append(outcome)
        epm = total_evals / max(n_pol_moves, 1)
        evals_per_move.append(epm)
        mlog.write(_json.dumps({
            "game": game + 1, "outcome": outcome, "R": round(R, 4),
            "baseline": round(baseline, 4), "evals": int(total_evals),
            "evals_per_move": round(epm, 1), "n_moves": n_pol_moves,
            "pol_white": pol_white, "plies": b.ply(),
            "n_point": n_point, "n_hold": n_hold, "budgets": bud_hist,
            "n_achieved": n_ach, "n_abandoned": n_aband,
            "running_score": round(float(np.mean(scores)), 3),
            "running_epm": round(float(np.mean(evals_per_move)), 1),
            "minutes": round((time.time() - t0) / 60, 1)}) + "\n")
        if (game + 1) % args.ckpt_every == 0:
            _outp = args.out or (base + "_pointer.pt")
            torch.save(pol.state_dict(), _outp)                      # latest
            torch.save({"state_dict": pol.state_dict(),
                        "opt": opt.state_dict(), "game": game + 1,
                        "baseline": baseline,
                        "running_score": float(np.mean(scores)),
                        "running_epm": float(np.mean(evals_per_move))},
                       _outp.replace(".pt", f"_g{game+1}.pt"))       # ladder, never overwritten
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
