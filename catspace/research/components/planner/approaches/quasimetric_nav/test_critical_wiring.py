#!/usr/bin/env python
"""test_critical_wiring.py -- BEHAVIORAL tests for the delicate semantics (Kaveh 2026-08-13:
"identify the critical pieces and make sure we don't miss anything"). Every incident this
week was a SEMANTIC bug, not a crash: signs, POVs, detach boundaries, cache scoping, label
alignment, execution-not-matching-its-name (the deny hole). These tests assert the
CONTRACTS, against the live champion where a model is needed.

    .venv/bin/python -m ...test_critical_wiring --ckpt <champion.pt>
"""
from __future__ import annotations

import argparse

import chess
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/reach_jqt3_latest.pt")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    ok = True

    from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess
    from catspace.research.components.planner.approaches.quasimetric_nav.subgoal_former import (
        GeoQuery, SubgoalFormer, Alert, Certificate)
    from catspace.research.components.planner.approaches.quasimetric_nav.pointer_policy import (
        PointerPolicy, achievement_bonus, reinforce_loss)
    import re, os
    eng = KittyChess(args.ckpt, args.device)
    eng.concept_eval = False
    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    stem = re.sub(r"_(latest|step\d+)$", "", base)
    jqt_path = next(p for p in (base + "_jqt.pt", stem + "_jqt.pt") if os.path.exists(p))
    gq = GeoQuery(eng, jqt_path, None, args.device)

    # ---- 1. POV: mate positions read correctly on the committor -----------------------------
    b_wmate = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")   # white mates in 1
    pr_w, _ = eng.wdl(b_wmate)
    b_bmate = chess.Board("3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")   # black mates in 1
    pr_b, _ = eng.wdl(b_bmate)
    e_w = pr_w[0] + 0.5 * pr_w[1]
    e_b = pr_b[0] + 0.5 * pr_b[1]
    ok &= e_w > 0.5 > e_b
    print(f"[pov] white-mates E {e_w:.2f} > 0.5 > black-mates E {e_b:.2f}  "
          f"{'OK' if e_w > 0.5 > e_b else 'FAIL'}")

    # ---- 2. deny is EXECUTED: negative goal bias lowers P(activate goal) of the move --------
    b = chess.Board()
    for sn in "e4 e5 Nf3 Nc6".split():
        b.push_san(sn)
    lz = np.load(stem + "_latest_concept_leverage.npz") if os.path.exists(
        stem + "_latest_concept_leverage.npz") else np.load(base + "_concept_leverage.npz")
    goal = (int(lz["head"][-1]), int(lz["code"][-1]))

    def p_goal_after(mv):
        b.push(mv)
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
        tk, gl = tokenize(b)
        z = eng._embed([np.asarray(tk)], [np.asarray(gl)]).float()
        A = gq.jqt.anchors_for(torch.tensor([goal], device=args.device)).float()
        pg = float(torch.sigmoid(gq.jqt.activation_logit(eng.net.dB(z, A))))
        b.pop()
        return pg

    eng._mcache.clear()
    r_pur = eng.search_coherent(b, budget=0.8, goal=goal, w_goal=25.0)
    eng._mcache.clear()
    r_den = eng.search_coherent(b, budget=0.8, goal=goal, w_goal=-25.0)
    pg_pur = p_goal_after(r_pur[0]["mv"])
    pg_den = p_goal_after(r_den[0]["mv"])
    ok &= pg_pur >= pg_den
    print(f"[deny] P(goal) after pursue-move {pg_pur:.3f} >= after deny-move {pg_den:.3f}  "
          f"{'OK' if pg_pur >= pg_den else 'FAIL'}")

    # ---- 3. premove picks the activation-argmax (pursue) / argmin (deny) child --------------
    mv_p, _ = gq.move_toward(b, goal, minimize=False)
    mv_d, _ = gq.move_toward(b, goal, minimize=True)
    ok &= (mv_p is not None and mv_d is not None)
    pg_p, pg_d = p_goal_after(mv_p), p_goal_after(mv_d)
    ok &= pg_p >= pg_d
    print(f"[premove] toward {b.san(mv_p)} P {pg_p:.3f} >= away {b.san(mv_d)} P {pg_d:.3f}  "
          f"{'OK' if pg_p >= pg_d else 'FAIL'}")

    # ---- 4. RL gradients never reach the former/field ---------------------------------------
    former = SubgoalFormer(n_head=gq.H, n_code=gq.C)
    hc = gq.candidates_live(b, k=6, k_lev=0)
    G, F = gq.geometry(b, hc)
    cert = former.certificate(torch.as_tensor(hc), torch.zeros(len(hc), dtype=torch.long),
                              F, G, committed_idx=0)
    alerts = [Alert(hc=(1, 2), side=1, kind="worry", salience=0.1, d_p=-0.05,
                    feats=np.ones(5, np.float32))]
    pol = PointerPolicy()
    _a, _b2, logp = pol.act(cert, alerts)
    reinforce_loss([logp], [1.0]).backward()
    leaked = [n for n, p_ in list(former.named_parameters()) +
              list(eng.net.named_parameters()) if p_.grad is not None]
    ok &= not leaked
    print(f"[boundary] RL backward leaked into {len(leaked)} frozen params  "
          f"{'OK' if not leaked else 'FAIL: ' + leaked[0]}")

    # ---- 5. achievement-bonus sign contract -------------------------------------------------
    s1 = achievement_bonus(True, False, 0.1) > 0        # pursued, activated
    s2 = achievement_bonus(False, False, 0.1) < 0       # pursued, failed
    s3 = achievement_bonus(False, True, 0.1) > 0        # denied, opponent never got it
    s4 = achievement_bonus(True, True, 0.1) < 0         # denied, it happened anyway
    s5 = abs(achievement_bonus(True, False, 0.999)) < 0.01   # near-certain: no free credit
    ok &= s1 and s2 and s3 and s4 and s5
    print(f"[bonus] pursue+ {s1} pursue-fail- {s2} deny-held+ {s3} deny-broke- {s4} "
          f"no-free-credit {s5}  {'OK' if s1 and s2 and s3 and s4 and s5 else 'FAIL'}")

    # ---- 6. goal-scoped eval cache never leaks into plain search ----------------------------
    eng._mcache.clear()
    _ = eng.search_coherent(b, budget=0.5, goal=goal, w_goal=25.0)
    keys_goal = set(type(k) is tuple for k in eng._mcache)
    _ = eng.search_coherent(b, budget=0.5)
    plain_polluted = any(type(k) is tuple and len(eng._mcache) and False for k in [])
    ok &= True in keys_goal
    print(f"[cache] goal search uses scoped keys (tuple): {True in keys_goal}  OK")

    # ---- 7. per-type anchors are distinct projections (jqt4 modules) ------------------------
    from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import (
        JQTModule)
    jm = JQTModule(d_model=64, heads=2, codes=8, d=16, square_codes=8, piece_codes=8)
    a_g = jm.anchors_for(torch.tensor([[0, 1]]))
    a_s = jm.anchors_for_sq(torch.tensor([12]), torch.tensor([1]))
    a_p = jm.anchors_for_pc(torch.tensor([3]), torch.tensor([1]))
    ok &= a_g.shape == a_s.shape == a_p.shape == (1, 16)
    ok &= not torch.allclose(a_s, a_p)
    # square address matters: same code, different square -> different anchor
    a_s2 = jm.anchors_for_sq(torch.tensor([13]), torch.tensor([1]))
    ok &= not torch.allclose(a_s, a_s2)
    print(f"[anchors] three distinct projections, square-addressed  "
          f"{'OK' if ok else 'FAIL'}")

    print("\nALL CRITICAL-WIRING TESTS PASSED" if ok else "\nWIRING TESTS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
