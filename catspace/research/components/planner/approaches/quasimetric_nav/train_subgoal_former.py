#!/usr/bin/env python
"""train_subgoal_former.py -- supervised calibration of the SubgoalFormer on corpus-derived
activation events (Kaveh 2026-08-12: "go for the subgoal supervised training on the corpus").

Labels are FREE and vocabulary-fresh: code every row of sampled games with the champion's
jqt quantizer; for a sampled position and any token (h, c), the label is whether that code
first-activates strictly after the position within its game (hit 0/1, the CDB convention).
Every token in every sample carries a label, so one position supervises the whole token set.

Token sets are sampled per position (top-leverage + random codes) so ALL 512 embeddings
train. Geometry (G, feats) is precomputed once per sample -- the expensive part -- then
epochs are cheap. Gradient boundary: the field and jqt sidecar are FROZEN here; only the
SubgoalFormer's parameters train (docs/SUBGOALFORMER.md).

    .venv/bin/python -m ...train_subgoal_former --ckpt <field.pt> [--samples 20000]
writes <ckpt>_former.pt + prints held-out Brier/AUC (the race battery reads --former).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples", type=int, default=20_000)
    ap.add_argument("--games", type=int, default=3000, help="games to code for labels")
    ap.add_argument("--tokens", type=int, default=14, help="tokens per sample (lev + random)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess
    from catspace.research.components.planner.approaches.quasimetric_nav.subgoal_former import (
        GeoQuery, SubgoalFormer)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
        split_by_game)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.eval_dtz_gate import (
        row_to_board)
    from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T

    t0 = time.time()
    eng = KittyChess(args.ckpt, args.device)
    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    import re, os
    jqt_path = next(p for p in (base + "_jqt.pt",
                                re.sub(r"_(latest|step\d+)$", "", base) + "_jqt.pt")
                    if os.path.exists(p))
    lev_path = base + "_concept_leverage.npz"
    gq = GeoQuery(eng, jqt_path, lev_path if os.path.exists(lev_path) else None, args.device)
    c = eng.cfg
    tr = T.build(n_human=0, n_sf=c["games"], seed=c["traj_seed"], max_plies=c["max_plies"],
                 n_piecedown=c.get("n_piecedown", 0), verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c.get("train_args", {}).get("seed", 0))
    rng = np.random.default_rng(0)

    # ---- code sampled games with the frozen jqt quantizer (labels) --------------------------
    gor = tr.game_of_row()
    fit_pool = np.flatnonzero(split == 0)
    gsel = rng.choice(fit_pool, min(args.games, len(fit_pool)), replace=False)
    gmask = np.isin(gor, gsel)
    rws = np.flatnonzero(gmask)
    ids_all = np.empty((len(rws), gq.H), np.int16)
    with torch.no_grad():
        for a in range(0, len(rws), 4096):
            rr = rws[a:a + 4096]
            tok = torch.from_numpy(tr.tok[rr].astype(np.int64)).to(args.device)
            gl = torch.from_numpy(tr.glob[rr].astype(np.float32)).to(args.device)
            phi = eng.net.backbone(tok, gl)
            _, ids = gq.jqt.target_codes(phi)
            ids_all[a:a + 4096] = ids.cpu().numpy().astype(np.int16)
    row_pos = {int(r): i for i, r in enumerate(rws)}
    games = {}
    for g in gsel:
        m = gor[rws] == g
        if m.sum() >= 6:
            games[int(g)] = (rws[m], ids_all[m].astype(np.int64))
    print(f"[sf-train] coded {len(rws):,} rows across {len(games)} games "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- precompute samples: geometry + per-token activation labels -------------------------
    lev_flat = np.argsort(-np.abs(gq.lev).ravel())[:8] if gq.lev is not None else []
    lev_hc = [(int(i // gq.C), int(i % gq.C)) for i in lev_flat]
    Gs, Fs, HCs, Ys = [], [], [], []
    glist = list(games)
    n_bad = 0
    while len(Gs) < args.samples:
        g = glist[int(rng.integers(0, len(glist)))]
        rws_g, C_g = games[g]
        L = len(rws_g)
        p = int(rng.integers(0, L - 3))
        r = int(rws_g[p])
        b = row_to_board(tr.tok[r], tr.glob[r])
        if not b.is_valid() or b.is_game_over(claim_draw=True):
            n_bad += 1
            continue
        n_rand = args.tokens - len(lev_hc)
        hc = lev_hc + [(int(rng.integers(0, gq.H)), int(rng.integers(0, gq.C)))
                       for _ in range(n_rand)]
        hc = np.array(hc, np.int64)
        y = np.zeros(len(hc), np.float32)
        for ti, (hh, cc) in enumerate(hc):
            fut = C_g[p + 1:, hh]
            prev = C_g[p:-1, hh]
            ev = fut[fut != prev]
            y[ti] = 1.0 if (len(ev) and (ev == cc).any()) else 0.0
        try:
            G, F = gq.geometry(b, hc)
        except Exception:
            n_bad += 1
            continue
        Gs.append(G); Fs.append(F); HCs.append(torch.as_tensor(hc)); Ys.append(torch.as_tensor(y))
        if len(Gs) % 2000 == 0:
            print(f"[sf-train] {len(Gs)}/{args.samples} samples "
                  f"[{(time.time()-t0)/60:.0f}m]", flush=True)
    G = torch.stack(Gs); F = torch.stack(Fs); HC = torch.stack(HCs); Y = torch.stack(Ys)
    pos_rate = float(Y.mean())
    print(f"[sf-train] {len(G):,} samples x {args.tokens} tokens | positive rate "
          f"{pos_rate:.2f} | skipped {n_bad}", flush=True)

    n_val = max(500, len(G) // 10)
    idx = rng.permutation(len(G))
    vi, ti = idx[:n_val], idx[n_val:]

    former = SubgoalFormer(n_head=gq.H, n_code=gq.C).to(args.device)
    opt = torch.optim.Adam(former.parameters(), lr=args.lr)
    sides0 = torch.zeros(args.tokens, dtype=torch.long, device=args.device)
    Gd, Fd = G.to(args.device), F.to(args.device)
    HCd, Yd = HC.to(args.device), Y.to(args.device)

    def run(sel, train):
        tot, n = 0.0, 0
        ps_all, ys_all = [], []
        former.train(train)
        for a in range(0, len(sel), args.batch):
            bb = sel[a:a + args.batch]
            losses = []
            for i in bb:
                p, _ = former(HCd[i], sides0, Fd[i], Gd[i])
                losses.append(torch.nn.functional.binary_cross_entropy(p, Yd[i]))
                if not train:
                    ps_all.append(p.detach().cpu().numpy()); ys_all.append(Yd[i].cpu().numpy())
            loss = torch.stack(losses).mean()
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(bb); n += len(bb)
        if not train:
            ps, ys = np.concatenate(ps_all), np.concatenate(ys_all)
            brier = float(np.mean((ps - ys) ** 2))
            o = np.argsort(ps); rk = np.empty(len(ps)); rk[o] = np.arange(len(ps))
            n1, n0 = int(ys.sum()), int((1 - ys).sum())
            auc = float((rk[ys == 1].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1))
            return tot / n, brier, auc
        return tot / n

    for ep in range(args.epochs):
        tl = run(ti, True)
        vl, brier, auc = run(vi, False)
        print(f"[sf-train] epoch {ep+1}/{args.epochs}  train BCE {tl:.4f}  val BCE {vl:.4f}  "
              f"Brier {brier:.4f}  AUC {auc:.3f}  [{(time.time()-t0)/60:.0f}m]", flush=True)

    out = base + "_former.pt"
    torch.save(former.state_dict(), out)
    print(f"[sf-train] VERDICT  val Brier {brier:.4f} (chance ~{pos_rate*(1-pos_rate):.3f}+)  "
          f"AUC {auc:.3f} (chance 0.5)  -> {out}")


if __name__ == "__main__":
    main()
