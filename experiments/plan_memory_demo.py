#!/usr/bin/env python
"""
experiments/plan_memory_demo.py — plan memory on KRkn: two goals (direct MATE,
and MATE-via-the-KRk-stratum after the knight is traded off), a calibrated
availability threshold, and a per-ply trace showing plans block, get
remembered with why, and wake when a stratum crossing makes them feasible.

Requires experiments/train_krkn.py to have been run at least once (reads
dtm_krkn/krkn_F/krkn_B from data/derived/).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from latentchess.cone.embedding import make_goal, reach
from latentchess.concepts import KMeansVQ
from latentchess.domains import krkn
from latentchess.cone.tabular import TabularFB
from latentchess.game import play_game
from latentchess.io.paths import generated_dir, load_array
from latentchess.opponents import EpsOptimalDTM, optimal_reply_table
from latentchess.planner.move_identity import RegionPairIdentity, SyntacticIdentity
from latentchess.planner.plans import PlanMemory, PlanStore, calibrate_tau
from latentchess.planner.policy import PlanningPolicy
from latentchess.planner.readout import ReplyAgg
from latentchess.scoring import TerminalScores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps-black", type=float, default=0.25)
    args = ap.parse_args()

    chain = krkn.build_chain(verbose=False)
    try:
        dtm = load_array("dtm_krkn")
        F = load_array("krkn_F")
        B = load_array("krkn_B")
    except FileNotFoundError:
        print("data/derived/{dtm_krkn,krkn_F,krkn_B}.npy not found -- "
              "run experiments/train_krkn.py first.", file=sys.stderr)
        return 1

    emb = TabularFB(F=F, B=B)

    mate_region = np.array([chain.terminals.mate])
    mate_goal = make_goal("MATE", mate_region, emb)

    via_range = chain.strata["KRk"]
    via_states = np.array([i for i in via_range if np.isfinite(dtm[i]) and dtm[i] <= 5])
    via_goal = make_goal("VIA_KRK", via_states, emb)

    live_dtm = dtm[:chain.n_live]
    won = np.isfinite(live_dtm)
    reach_mate = reach(emb, mate_goal, None)[:chain.n_live]
    tau = calibrate_tau(reach_mate, won)
    print(f"calibrated tau = {tau:.4f}")

    tokens_fit = KMeansVQ(n_tokens=16, seed=0).fit(F[:chain.n_live])
    tokens = tokens_fit.tokens(F[:chain.n_live])
    identities = [SyntacticIdentity(), RegionPairIdentity(tokens)]

    b_opt = optimal_reply_table(chain, dtm)
    black = EpsOptimalDTM(b_opt, eps=args.eps_black)

    rng = np.random.default_rng(args.seed)
    krkn_stratum = chain.strata["KRkn"]
    candidates = np.array([i for i in krkn_stratum
                            if 10 <= live_dtm[i] <= 25])
    starts = rng.choice(candidates, size=min(args.games, len(candidates)), replace=False)

    store = PlanStore(generated_dir() / "plans_demo")
    saw_availability_flip = False

    for gi, s0 in enumerate(starts):
        memory = PlanMemory(emb, [mate_goal, via_goal], tau=tau)
        policy = PlanningPolicy(chain, emb, memory, ts=TerminalScores.big(),
                                 agg=ReplyAgg.MIN, depth=1, identities=identities)

        on_ply_log = []
        _orig_on_ply = memory.on_ply

        def _logged_on_ply(s, events, F_now, _orig=_orig_on_ply, _log=on_ply_log):
            woken = _orig(s, events, F_now)
            _log.append((events, woken))
            return woken

        memory.on_ply = _logged_on_ply

        print(f"\n=== game {gi} start={int(s0)} dtm={live_dtm[s0]:.0f} ===")
        print(f"{'ply':>3} {'move':>8} {'plan':>6} {'goal':>9} {'reach':>7} "
              f"{'avail(MATE,VIA)':>16} {'events':>10} {'woken':>8}")

        rec = play_game(chain, policy, black, int(s0), cap=100, rng=np.random.default_rng(gi))

        prev_via_available = None
        for ply, (s, mid) in enumerate(zip(rec.states, rec.move_ids)):
            avail = memory.available(s)
            flags = tuple(ok for _, _, ok in avail)
            plan = memory.plans.get(memory.active_id) if memory.active_id else None
            plan_id = plan.plan_id if plan else "-"
            goal_name = plan.goal.name if plan else "-"
            r = memory.reach_of(plan.goal, s) if plan else float("nan")
            events, woken = on_ply_log[ply] if ply < len(on_ply_log) else ([], [])
            ev_str = ",".join(str(e) for e in events) or "-"
            wk_str = ",".join(woken) or "-"
            print(f"{ply:>3} {chain.move_names[mid]:>8} {plan_id:>6} {goal_name:>9} "
                  f"{r:>7.3f} {str(flags):>16} {ev_str:>10} {wk_str:>8}")

            via_ok = flags[1]
            if prev_via_available is False and via_ok is True:
                saw_availability_flip = True
            prev_via_available = via_ok

        print(f"result={rec.result} plies={len(rec.states)}")
        for plan in memory.plans.values():
            store.append(plan)

    store.save_goal_z([mate_goal, via_goal])
    reloaded = store.load_all()
    print(f"\nreloaded {len(reloaded)} plans from {store.dir}")
    for p in reloaded:
        print(f"  {p.plan_id} status={p.status.name} goal={p.subgoals[0].name} "
              f"feasibility0={p.feasibility0:.3f} trace_len={len(p.trace)}")

    if not saw_availability_flip:
        print("\nNOTE: no VIA_KRK availability flip observed in this sample; "
              "try a different --seed or --games.", file=sys.stderr)
        return 1
    print("\nobserved a VIA_KRK availability flip (False -> True) during play.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
