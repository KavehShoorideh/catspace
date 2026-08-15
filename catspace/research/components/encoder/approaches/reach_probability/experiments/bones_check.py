#!/usr/bin/env python
"""bones_check.py -- IS THE LENGTH RULER A RULER? (Kaveh 2026-08-13: "I can't tell if the
flatness of the dA is because of lack of data ... I don't wanna [scale] until the bones are
right.")

Four gates, all on an EXISTING checkpoint, CPU, minutes. They separate "needs more data"
from "structurally broken", which is the decision that gates any cloud spend.

  1 CALIBRATION   for OBSERVED events (hit=1), is predicted dA the true plies-to-first-
                  activation? Spearman + slope + median error, bucketed by true distance.
                  NOTE the training loss (censored_plies_loss) drops censored pairs
                  entirely, so any bias here is toward REACHED goals, not away.
  2 MONOTONICITY  walk backward from a real activation event: does dA fall ~1 per ply?
                  slope ~1.0 and a high monotone fraction = the ruler works GLOBALLY, and
                  one-ply flatness is only a resolution question.
  3 RESOLUTION    across legal moves, spread of dA in RAW PLIES, bucketed by how far the
                  goal actually is. Near goals SHOULD separate; far goals should not
                  (one move cannot change a 60-ply distance -- that is physics, not a bug).
  4 LEARNING CURVE the only test that answers "is it data?": probe frozen embeddings ->
                  plies, at 10/30/100% of labels. Still climbing at 100% = data helps.
                  Plateaued by 30% = more data will NOT fix it.

    .venv/bin/python -m ...bones_check --ckpt artifacts/experiments/reach_jqt3_latest.pt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/reach_jqt3_latest.pt")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--events", type=int, default=3000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import os, re
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import (
        JQTModule, ActivationIndex)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
        load_net)
    dev = args.device
    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    stem = re.sub(r"_(latest|step\d+)$", "", base)
    model, _ = load_net(args.ckpt, dev)
    model.eval()
    pj = torch.load(next(p for p in (base + "_jqt.pt", stem + "_jqt.pt")
                         if os.path.exists(p)), map_location=dev, weights_only=False)
    jqt = JQTModule(d_model=pj["d_in"], heads=pj["heads"], codes=pj["codes"], d=pj["d"],
                    square_codes=pj.get("square_codes", 0),
                    piece_codes=pj.get("piece_codes", 0)).to(dev)
    jqt.load_state_dict(pj["state_dict"], strict=False)
    jqt.eval()
    print(f"[bones] {args.ckpt}  heads={pj['heads']} codes={pj['codes']}", flush=True)

    tr = T.build(n_human=0, n_sf=4000, seed=0, cache=True, max_plies=400,
                 n_piecedown=45906)
    g_of = tr.game_of_row()
    rng = np.random.default_rng(0)
    gsel = rng.choice(int(g_of.max()) + 1, args.games, replace=False)
    gmask = np.isin(g_of, gsel)
    rws = np.flatnonzero(gmask)
    t0 = time.time()
    ids_all = np.empty((len(rws), pj["heads"]), np.int64)
    Z = []
    with torch.no_grad():
        for a in range(0, len(rws), 4096):
            rr = rws[a:a + 4096]
            tok = torch.from_numpy(tr.tok[rr].astype(np.int64)).to(dev)
            glob = torch.from_numpy(tr.glob[rr].astype(np.float32)).to(dev)
            phi = model.backbone(tok, glob)
            ids_all[a:a + 4096] = jqt.target_codes(phi)[1].cpu().numpy()
            Z.append(model.proj_b(phi).float().cpu())
    Z = torch.cat(Z)
    print(f"[bones] coded {len(rws):,} rows from {args.games} games "
          f"[{time.time()-t0:.0f}s]", flush=True)
    row_ix = {int(r): i for i, r in enumerate(rws)}

    idx = ActivationIndex(rng, codes=pj["codes"])
    games = []
    gor = g_of[rws]
    for g in gsel:
        m = gor == g
        if m.sum() >= 6:
            games.append((rws[m], ids_all[m]))       # sample() takes 2-tuples
    idx.refresh(games)
    grows, ghc, gplies, ghit = idx.sample(args.events)
    hc_t = torch.from_numpy(ghc).long()
    with torch.no_grad():
        A = jqt.anchors_for(hc_t).float()
        zb = torch.stack([Z[row_ix[int(r)]] for r in grows])
        dA = model.dA(zb, A).float().numpy()
        dB = model.dB(zb, A).float().numpy()
        p_act = torch.sigmoid(jqt.activation_logit(torch.from_numpy(dB))).numpy()

    # ---- GATE 1: calibration on observed events ------------------------------------------
    hit = ghit > 0.5
    tp, pp = gplies[hit], dA[hit]
    rho = spearman(tp, pp)
    lo = np.polyfit(np.log1p(tp), np.log1p(pp), 1)[0]
    med = float(np.median(np.abs(pp - tp)))
    print(f"\n[gate 1: CALIBRATION] {hit.sum()} observed events")
    print(f"  spearman(true plies, dA) {rho:+.3f} | log-log slope {lo:.2f} (1.0 = a ruler)"
          f" | median |err| {med:.1f} plies")
    for a0, b0 in ((1, 3), (4, 8), (9, 20), (21, 60), (61, 999)):
        m = (tp >= a0) & (tp <= b0)
        if m.sum() > 20:
            print(f"    true {a0:3d}-{b0:<3d} n={int(m.sum()):4d}  dA median "
                  f"{np.median(pp[m]):6.1f}  (mean P(act) {p_act[hit][m].mean():.2f})")
    cens = ~hit
    if cens.sum() > 20:
        print(f"  censored (never activated, DROPPED by the loss): dA median "
              f"{np.median(dA[cens]):.1f} vs observed {np.median(pp):.1f}")

    # ---- GATE 2: monotonicity along real approach trajectories ---------------------------
    K = 10
    slopes, mono = [], []
    with torch.no_grad():
        for (rowsg, C) in games[:200]:
            L = len(rowsg)
            for h in range(pj["heads"]):
                ch = np.flatnonzero(C[1:, h] != C[:-1, h]) + 1
                ch = ch[ch >= K + 1]                  # need K plies of run-up
                for t in ch[:2]:
                    c_new = int(C[t, h])
                    zb_seq = torch.stack([Z[row_ix[int(rowsg[t - k])]]
                                          for k in range(1, K + 1)])
                    Aq = jqt.anchors_for(torch.tensor([[h, c_new]])).float()
                    d = model.dA(zb_seq, Aq.expand(K, -1)).float().numpy()
                    true_k = np.arange(1, K + 1, dtype=float)
                    slopes.append(np.polyfit(true_k, d, 1)[0])
                    mono.append(float((np.diff(d) >= -1e-6).mean()))
                    if len(slopes) >= 400:
                        break
                if len(slopes) >= 400:
                    break
            if len(slopes) >= 400:
                break
    print(f"\n[gate 2: MONOTONICITY] {len(slopes)} approach trajectories, {K} plies back")
    print(f"  mean slope {np.mean(slopes):+.3f} plies of dA per ply of truth (1.0 = ideal)"
          f" | median {np.median(slopes):+.3f}")
    print(f"  fraction of steps moving the RIGHT way {np.mean(mono):.1%} (0.5 = chance)")

    # ---- GATE 3: one-ply resolution vs how far the goal actually is ----------------------
    print(f"\n[gate 3: RESOLUTION] dA by how far the goal ACTUALLY is (observed events)")
    for name, (a0, b0) in (("near (1-3)", (1, 3)), ("mid (4-12)", (4, 12)),
                           ("far (25+)", (25, 999))):
        m = hit & (gplies >= a0) & (gplies <= b0)
        if m.sum() > 20:
            print(f"  {name:11s} n={int(m.sum()):4d}  dA spread across events "
                  f"p10-p90 {np.percentile(dA[m],10):5.1f}-{np.percentile(dA[m],90):5.1f}"
                  f"  P(act) {p_act[m].mean():.2f}")

    # ---- GATE 4: learning curve on frozen embeddings -------------------------------------
    print(f"\n[gate 4: LEARNING CURVE] probe frozen z_B + anchor -> log1p(plies)")
    Xz = zb[hit]
    Xa = A[hit].detach()
    Y = torch.from_numpy(np.log1p(gplies[hit])).float()
    n = len(Y)
    perm = torch.randperm(n)
    n_te = max(200, n // 5)
    te, trn = perm[:n_te], perm[n_te:]
    print(f"  (the trained dA head scores spearman {rho:+.3f} on these same events --"
          f" a probe that MATCHES it is at the representation ceiling)")
    for frac in (0.03, 0.1, 0.3, 0.6, 1.0):
        k = max(50, int(len(trn) * frac))
        sub = trn[:k]
        net = torch.nn.Sequential(torch.nn.Linear(Xz.shape[1] * 2, 128), torch.nn.GELU(),
                                  torch.nn.Linear(128, 1))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        Xtr = torch.cat([Xz[sub], Xa[sub]], -1)
        Xte = torch.cat([Xz[te], Xa[te]], -1)
        for _ in range(400):
            loss = torch.nn.functional.huber_loss(net(Xtr).squeeze(-1), Y[sub])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pr = net(Xte).squeeze(-1).numpy()
        rho_p = spearman(np.expm1(Y[te].numpy()), np.expm1(pr))
        print(f"  {int(frac*100):3d}% of labels (n={k:5d})  held-out spearman {rho_p:+.3f}")
    # ---- GATE 5: NAVIGABILITY on tablebase-optimal lines --------------------------------
    # THE decisive test for the QRL ruler fix. On a TB-optimal line the true distance falls by
    # exactly 1 every ply, so any step where dA RISES is a step greedy descent would get wrong.
    # Correlation completely misses this: jqt5 scored ~0.5 while going the wrong way 34.9% of
    # the time. Measured before the fix: 34.9% wrong-way, median step -0.18 (true: -1.000).
    print(f"\n[gate 5: NAVIGABILITY] dA along TABLEBASE-OPTIMAL lines")
    try:
        import chess
        from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import (
            TB, rollout_line)
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
            tokenize as _tkg)
        from catspace.io import paths as _paths
        _tb = TB(str(_paths.syzygy_dir()))
        _fens = ["8/8/8/4k3/8/8/3Q4/4K3 w - - 0 1", "8/8/8/4k3/8/8/8/R3K3 w - - 0 1",
                 "8/8/8/8/4k3/8/4P3/4K3 w - - 0 1", "4k3/8/8/8/8/8/6PP/4K2R w K - 0 1",
                 "8/5k2/8/8/8/2K5/1Q6/8 w - - 0 1", "3k4/8/3K4/8/8/8/8/5R2 w - - 0 1"]
        P0 = model.poles.poles[:3]
        steps_all, nlines = [], 0
        for _f in _fens:
            _line = rollout_line(chess.Board(_f), _tb, cap=120)
            if not _line or len(_line) < 8:
                continue
            nlines += 1
            _t, _g = zip(*(_tkg(b) for b in _line))
            with torch.no_grad():
                _phi = model.backbone(torch.from_numpy(np.array(_t, dtype="int64")).to(dev),
                                      torch.from_numpy(np.array(_g, dtype="float32")).to(dev))
                _d = model.dA(model.proj_b(_phi).float(),
                              P0[[0]].expand(len(_line), -1)).cpu().numpy()
            steps_all.append(np.diff(_d))
        S = np.concatenate(steps_all)
        print(f"  {nlines} lines | steps going the WRONG way {100*(S>0).mean():.1f}%  "
              f"(perfect ruler 0%; jqt5 measured 34.9%)")
        print(f"  median step {np.median(S):+.3f}  (perfect ruler -1.000; jqt5 -0.180)")
        print(f"  worst backward step {S.max():+.2f}")
    except Exception as e:
        print(f"  skipped ({e})")

    print("\n[bones] READ: gate1 slope ~1 + gate2 slope ~1 => the ruler is a ruler; "
          "gate4 still climbing at 100% => data helps, flat => structure is the limit.")


if __name__ == "__main__":
    main()
