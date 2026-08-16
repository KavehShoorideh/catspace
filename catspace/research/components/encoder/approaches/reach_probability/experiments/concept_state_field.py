#!/usr/bin/env python
"""concept_state_field.py -- the IQE moved ONE LAYER DOWNSTREAM (Kaveh 2026-08-15:
"let's replace the iqe and send it one layer downstream. i.e. let's get concepts from some
board trunk. concepts should be predictive of next concepts. then we define states as the set
of concepts active. then we iqe on those states, like we do now, trying to nail down the ply
count poorly but at least get committor right. I don't wanna evaluate sf on every state yet as
learning labels")

THE INVERSION. Until now the quasimetric was primary and concepts were quantised off z_B:

    board -> phi -> z_B -> IQE geometry            concepts quantised the geometry

Here the concepts are primary and the geometry is a readout of the DISCRETE state:

    board -> phi -> VQ concepts -> STATE = the set of active concepts -> IQE

WHAT THIS BUYS. The state is now genuinely discrete: at most K^H distinct states, and the
geometry is PIECEWISE CONSTANT on concept cells. Two positions with the same concepts are the
same point, by construction -- which is the "subgoal = invariance class" definition made
structural rather than hoped for.

WHAT IT COSTS, STATED UP FRONT. Sibling discrimination now depends entirely on the vocabulary
being fine enough. If a blunder and the best move land in the same concept cell, they receive
literally identical value -- not approximately, exactly. probe_value_proxy measured the current
committor at within-parent spearman +0.096 with sibling spread 22% of truth, so this design
does not inherit a good number; it inherits a bad one and makes the cause explicit and
measurable. `--eval-sf` reports whether the partition separates good moves from bad, using the
Stockfish shards STRICTLY AS A YARDSTICK and never as a training label, per the instruction.

NO STOCKFISH IN THE LOSS. Every training signal here is free from the corpus:
    committor   the game's actual W/D/L result            (outcome, the thing to get right)
    dynamics    the next position's own codes             (concepts predict next concepts)
    ply         observed ply gaps inside a game           (deliberately weak; "poorly")
    commitment  the VQ's own term

    .venv/bin/python -m ...concept_state_field --steps 400 --games 400     # smoke
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptStateField(nn.Module):
    """board -> concepts -> state -> quasimetric."""

    def __init__(self, d_model=192, layers=4, tr_heads=8, heads=16, codes=16, d_code=32,
                 d_state=48, components=8, ply_cap=60.0):
        super().__init__()
        from vector_quantize_pytorch import VectorQuantize
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
            JepaEncoder)
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE

        self.heads, self.codes, self.d_code = heads, codes, d_code
        self.trunk = JepaEncoder(d_model, layers, tr_heads)
        self.enc = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(),
                                 nn.Linear(256, heads * d_code))
        self.vq = nn.ModuleList([VectorQuantize(dim=d_code, codebook_size=codes, decay=0.9,
                                                commitment_weight=0.25) for _ in range(heads)])
        # STATE = a function of the CODES ALONE. zq carries the straight-through gradient, and
        # takes only `codes` distinct values per head, so z_state is piecewise constant on
        # concept cells exactly as intended. Head identity is preserved (concatenation, not a
        # permutation-invariant sum) because head h is a specific slot, not an interchangeable
        # set member.
        self.state = nn.Sequential(nn.Linear(heads * d_code, 256), nn.GELU(),
                                   nn.Linear(256, d_state))
        h = d_state // 2
        assert d_state % 2 == 0 and h % components == 0, (
            f"d_state={d_state} splits into two {h}-d rulers; each must divide into "
            f"{components} IQE components (try --d-state {components * 2 * 2})")
        self.iqe_a = IQE(h, components=components)      # LENGTH ruler (plies) -- weak on purpose
        self.iqe_b = IQE(h, components=components)      # OUTCOME ruler -> committor
        self.poles = nn.Parameter(torch.randn(3, d_state) * 0.5)   # W / D / L
        # dynamics: (state, move) -> next position's code in every head
        # all three share a width because they are SUMMED, as MoveHead does it (64x32, 64x32,
        # 5x32 -> one 32-d action vector)
        self.mv_from = nn.Embedding(64, 32); self.mv_to = nn.Embedding(64, 32)
        self.mv_promo = nn.Embedding(5, 32)
        self.dyn = nn.Sequential(nn.Linear(d_state + 32, 512), nn.GELU(),
                                 nn.Linear(512, heads * codes))
        self.ply_cap = ply_cap

    # ---- concepts -------------------------------------------------------------------
    def quantize(self, phi):
        h = self.enc(phi).view(len(phi), self.heads, self.d_code)
        qs, ids, vl = [], [], 0.0
        for i in range(self.heads):
            q, idx, l = self.vq[i](h[:, i])
            qs.append(q); ids.append(idx); vl = vl + l
        return torch.cat(qs, -1), torch.stack(ids, 1), vl

    def encode(self, tok, glob):
        phi = self.trunk(tok, glob)
        zq, ids, vl = self.quantize(phi)
        return self.state(zq), ids, vl, zq

    def dA(self, u, v):
        k = u.shape[-1] // 2
        return self.iqe_a(u[..., :k], v[..., :k])

    def dB(self, u, v):
        k = u.shape[-1] // 2
        return self.iqe_b(u[..., k:], v[..., k:])

    def committor(self, z, temp=5.0):
        """-> (N,3) over (WIN, DRAW, LOSS) from OUTCOME-ruler distance to the three poles."""
        d3 = torch.stack([self.dB(z, self.poles[[k]].expand(len(z), -1)) for k in range(3)], 1)
        return torch.softmax(-d3 / temp, 1)

    def predict_next(self, z, mid):
        e = self.mv_from(mid[:, 0]) + self.mv_to(mid[:, 1]) + self.mv_promo(mid[:, 2])
        return self.dyn(torch.cat([z, e], -1)).view(len(z), self.heads, self.codes)


def perplexity(ids, codes):
    ps = []
    for h in range(ids.shape[1]):
        c = torch.bincount(ids[:, h], minlength=codes).float()
        p = c / c.sum().clamp(min=1)
        ps.append(float(torch.exp(-(p * (p + 1e-9).log()).sum())))
    return float(np.mean(ps)), float(np.mean([len(torch.unique(ids[:, h]))
                                              for h in range(ids.shape[1])]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--codes", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--d-state", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--basin-temp", type=float, default=5.0)
    ap.add_argument("--w-outcome", type=float, default=3.0, help="committor: GET THIS RIGHT")
    ap.add_argument("--w-dyn", type=float, default=2.0, help="concepts predict next concepts")
    ap.add_argument("--w-ply", type=float, default=0.2, help="deliberately weak ('poorly')")
    ap.add_argument("--ply-gap", type=int, default=8, help="max gap for the dA pair sample")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-sf", type=int, default=1,
                    help="report the Stockfish within-parent yardstick. EVALUATION ONLY -- "
                         "never enters the loss (Kaveh: no SF as learning labels yet)")
    ap.add_argument("--sf-parents", type=int, default=400)
    ap.add_argument("--out", default="artifacts/experiments/csf1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    torch.manual_seed(args.seed)
    dev = args.device
    t0 = time.time()
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=0)
    gor = tr.game_of_row(); por = tr.ply_of_row()
    out_w = tr.outcome_of_row_white()          # WIN / DRAW / LOSS per row, white POV
    cls = np.full(len(gor), -1, np.int64)
    cls[out_w == T.WIN] = 0; cls[out_w == T.DRAW] = 1; cls[out_w == T.LOSS] = 2
    ok = np.flatnonzero(cls >= 0)
    # the transitions cache is keyed by ROW COUNT, so a new corpus needs a new one; build it
    # rather than die (no script in the repo generated this file before gen_game_transitions)
    from catspace.research.tools.training_infra import gen_game_transitions as GT
    gz = GT.ensure(tr, T.load_sf_games(args.games, args.seed, max_plies=args.max_plies))
    par, mid = gz["par"], gz["mid"]
    print(f"[csf] {len(tr):,} games | {len(tr.tok):,} positions | {len(ok):,} outcome-labelled "
          f"| {len(par):,} transitions [{time.time()-t0:.0f}s]", flush=True)

    net = ConceptStateField(d_model=args.d_model, heads=args.heads, codes=args.codes,
                            d_state=args.d_state).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"[csf] {sum(p.numel() for p in net.parameters()):,} params | state space "
          f"{args.codes}^{args.heads}", flush=True)

    def batch(rows):
        return (torch.from_numpy(tr.tok[rows].astype(np.int64)).to(dev),
                torch.from_numpy(tr.glob[rows].astype(np.float32)).to(dev))

    rng = np.random.default_rng(args.seed)
    log = open(f"{args.out}_steps.jsonl", "w", buffering=1 << 14)   # NOT append: one run, one file
    for step in range(1, args.steps + 1):
        net.train()
        # --- (1) committor on outcome-labelled positions (the free label) ----------------
        r = ok[rng.integers(0, len(ok), args.batch)]
        tk, gb = batch(r)
        z, ids, vl, _ = net.encode(tk, gb)
        y = torch.from_numpy(cls[r]).to(dev)
        p3 = net.committor(z, args.basin_temp)
        l_out = F.nll_loss((p3 + 1e-8).log(), y)

        # --- (2) concepts predict next concepts ------------------------------------------
        ti = rng.integers(0, len(par), args.batch)
        pr = par[ti]
        tk_p, gb_p = batch(pr)
        tk_c, gb_c = batch(pr + 1)
        z_p, ids_p, vl_p, _ = net.encode(tk_p, gb_p)
        with torch.no_grad():
            _, ids_c, _, _ = net.encode(tk_c, gb_c)          # target codes, detached
        mv = torch.from_numpy(mid[ti].astype(np.int64)).to(dev)
        lg = net.predict_next(z_p, mv)
        l_dyn = F.cross_entropy(lg.reshape(-1, args.codes), ids_c.reshape(-1))

        # --- (3) ply, deliberately weak ---------------------------------------------------
        g = gor[pr]; same = np.flatnonzero(gor[np.clip(pr + args.ply_gap, 0, len(gor) - 1)] == g)
        if len(same) > 8:
            a = pr[same]; k = rng.integers(1, args.ply_gap + 1, len(same))
            b = np.clip(a + k, 0, len(gor) - 1)
            keep = gor[b] == gor[a]
            a, b, k = a[keep], b[keep], k[keep]
            za, _, _, _ = net.encode(*batch(a))
            zb_, _, _, _ = net.encode(*batch(b))
            l_ply = F.mse_loss(net.dA(za, zb_),
                               torch.from_numpy(k.astype(np.float32)).to(dev))
        else:
            l_ply = torch.zeros((), device=dev)

        loss = (args.w_outcome * l_out + args.w_dyn * l_dyn + args.w_ply * l_ply
                + vl + vl_p)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                perp, used = perplexity(ids.detach(), args.codes)
                acc = float((p3.argmax(1) == y).float().mean())
                dyn_acc = float((lg.argmax(-1) == ids_c).float().mean())
            rec = {"s": step, "loss": float(loss.detach()), "outcome": float(l_out.detach()),
                   "dyn": float(l_dyn.detach()), "ply": float(l_ply.detach()),
                   "vq": float(vl.detach()), "perp": perp, "used": used,
                   "out_acc": acc, "dyn_acc": dyn_acc}
            log.write(json.dumps(rec) + "\n")
            if step % 100 == 0 or step == 1:
                print(f"[csf] {step:5d} loss {float(loss):8.3f} | committor acc {acc:.3f} "
                      f"| next-code acc {dyn_acc:.3f} | perp {perp:5.2f} used {used:4.1f}/"
                      f"{args.codes} | {time.time()-t0:.0f}s", flush=True)

        if args.eval_sf and (step % args.eval_every == 0 or step == args.steps):
            sf = sf_yardstick(net, args, dev)
            if sf:
                print(f"[csf] {step:5d} SF YARDSTICK (eval only, never a label): "
                      f"within-parent spearman {sf['sp']:+.3f} | top-1 {sf['top1']:.1%} "
                      f"| distinct concept cells among siblings {sf['cells']:.2f}", flush=True)
                log.write(json.dumps({"s": step, **{f"sf_{k}": v for k, v in sf.items()}}) + "\n")

    torch.save({"state_dict": net.state_dict(), "step": args.steps,
                "args": vars(args)}, f"{args.out}_latest.pt")
    print(f"[csf] done -> {args.out}_latest.pt [{time.time()-t0:.0f}s]", flush=True)


def sf_yardstick(net, args, dev):
    """Stockfish shards as a MEASURING STICK ONLY -- no gradient, never in the loss.

    Reports the question this architecture lives or dies on: among the legal children of one
    position, does the concept state rank the moves? Plus `cells`, the mean number of DISTINCT
    concept states among a parent's children -- if that is ~1 the partition cannot rank moves
    even in principle, which is the structural risk of moving the IQE downstream."""
    import glob as _glob
    from scipy.stats import spearmanr
    fs = sorted(_glob.glob("data/derived/wdl_labels/*.npz"))
    if not fs:
        return None
    toks, globs, wdls, cnt = [], [], [], []
    for f in fs:
        z = np.load(f)
        toks.append(z["tok"]); globs.append(z["glob"]); wdls.append(z["wdl"])
        cnt.append(np.diff(z["off"]))
    toks, globs, wdls = np.concatenate(toks), np.concatenate(globs), np.concatenate(wdls)
    off = np.concatenate([[0], np.cumsum(np.concatenate(cnt))]).astype(np.int64)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(off) - 1, size=min(args.sf_parents, len(off) - 1), replace=False)
    net.eval()
    sps, tops, cells = [], [], []
    with torch.no_grad():
        for pi in pick:
            a, b = int(off[pi]), int(off[pi + 1])
            if b - a < 3:
                continue
            w = wdls[a:b].astype(np.float32)
            e_par = (w[:, 2] + 0.5 * w[:, 1]) / np.clip(w.sum(1), 1.0, None)
            g = globs[a:b]
            e_white = np.where(g[:, 0] == 0, e_par, 1.0 - e_par)
            z, ids, _, _ = net.encode(
                torch.from_numpy(toks[a:b].astype(np.int64)).to(dev),
                torch.from_numpy(g.astype(np.float32)).to(dev))
            p3 = net.committor(z, args.basin_temp)
            e_ours = (p3[:, 0] + 0.5 * p3[:, 1]).cpu().numpy()
            cells.append(len(np.unique(ids.cpu().numpy(), axis=0)))
            if np.ptp(e_white) < 1e-6:
                continue
            s = spearmanr(e_ours, e_white)[0]
            if np.isfinite(s):
                sps.append(s)
            tops.append(float(np.argmax(e_ours) == np.argmax(e_white)))
    net.train()
    if not sps:
        return None
    return {"sp": float(np.mean(sps)), "top1": float(np.mean(tops)),
            "cells": float(np.mean(cells)), "n": len(sps)}


if __name__ == "__main__":
    main()
