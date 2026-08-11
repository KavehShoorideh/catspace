#!/usr/bin/env python
"""concept_vq.py -- vector-quantized CONCEPT extraction (Kaveh 2026-08-11): a discrete
bottleneck trained to reproduce THE FIELD'S OWN evaluations. If a few codes regenerate the
distances and probabilities, the codes ARE the concepts the evaluation uses -- faithfulness
by construction (predict the model, not the world), which is the interpretability thesis's
primary endpoint.

    frozen trunk phi(s) [128] -> enc MLP -> H residual/multi-head VQ codes (H heads x K codes)
    -> dec MLP -> the field's own outputs: dA(->W/D/L) [3] + committor probs [3]

Reported: fidelity (R^2 per output on held-out rows), codebook perplexity (are codes used),
head independence, and a post-hoc NAMING pass -- correlation of each frequent code with cheap
board predicates (material sign, queens on, piece count, ...). Predicates are for NAMING ONLY
(Kaveh: concepts must be latent/learned; python-chess predicates never enter training).

    .venv/bin/python -m ...concept_vq --ckpt <field.pt> [--heads 4] [--codes 64]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


class ConceptVQ(nn.Module):
    def __init__(self, d_in=128, d_code=32, heads=4, codes=64):
        super().__init__()
        from vector_quantize_pytorch import VectorQuantize
        self.enc = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(),
                                 nn.Linear(256, heads * d_code))
        self.heads, self.d_code = heads, d_code
        self.vq = nn.ModuleList([VectorQuantize(dim=d_code, codebook_size=codes,
                                                decay=0.9, commitment_weight=0.25)
                                 for _ in range(heads)])
        self.dec = nn.Sequential(nn.Linear(heads * d_code, 256), nn.GELU(),
                                 nn.Linear(256, 6))

    def forward(self, x):
        h = self.enc(x).view(len(x), self.heads, self.d_code)
        qs, ids, vloss = [], [], 0.0
        for i, vq in enumerate(self.vq):
            q, idx, l = vq(h[:, i])
            qs.append(q); ids.append(idx); vloss = vloss + l
        y = self.dec(torch.cat(qs, -1))
        return y, torch.stack(ids, 1), vloss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--codes", type=int, default=64)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    tr = T.build(n_human=0, n_sf=c["games"], seed=c["traj_seed"], max_plies=c["max_plies"],
                 n_piecedown=c.get("n_piecedown", 0), verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    game = tr.game_of_row()
    pn = c["pole_names"]
    P = net.poles.poles.detach().float().to(args.device)
    pidx = [pn.index(k) for k in ("WIN", "DRAW", "LOSS")]
    rng = np.random.default_rng(0)

    def featurize(rows):
        """frozen field -> (phi, targets[dA3 + prob3]) in chunks."""
        PHI, Y = [], []
        for a in range(0, len(rows), 4096):
            rr = rows[a:a + 4096]
            with torch.no_grad():
                tok = torch.from_numpy(tr.tok[rr].astype(np.int64)).to(args.device)
                glob = torch.from_numpy(tr.glob[rr].astype(np.float32)).to(args.device)
                phi = net.backbone(tok, glob)
                z = net.proj_b(phi)
                DA = torch.stack([net.dA(z, P[[k]].expand(len(z), -1)) for k in pidx], 1)
                DB = torch.stack([net.dB(z, P[[k]].expand(len(z), -1)) for k in pidx], 1)
                pr = torch.softmax(-DB / 5.0, 1)
                PHI.append(phi.float().cpu()); Y.append(torch.cat([DA, pr], 1).float().cpu())
        return torch.cat(PHI), torch.cat(Y)

    fit_rows = np.flatnonzero(np.isin(game, np.flatnonzero(split == 0)))
    val_rows = np.flatnonzero(np.isin(game, np.flatnonzero(split == 1)))
    fit_rows = fit_rows[rng.choice(len(fit_rows), min(args.rows, len(fit_rows)), replace=False)]
    val_rows = val_rows[rng.choice(len(val_rows), min(30_000, len(val_rows)), replace=False)]
    Xf, Yf = featurize(fit_rows)
    Xv, Yv = featurize(val_rows)
    mu, sd = Yf.mean(0), Yf.std(0).clamp(min=1e-6)
    print(f"[vq] fit {len(Xf):,} rows, val {len(Xv):,}; targets: dA(W/D/L) + P(W/D/L)")

    model = ConceptVQ(d_in=Xf.shape[1], heads=args.heads, codes=args.codes).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xf_d, Yf_d = Xf.to(args.device), ((Yf - mu) / sd).to(args.device)
    for step in range(args.steps):
        sel = torch.randint(0, len(Xf_d), (1024,), device=args.device)
        y, ids, vloss = model(Xf_d[sel])
        loss = torch.nn.functional.mse_loss(y, Yf_d[sel]) + vloss
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            print(f"[vq] step {step+1}: recon {float(loss - vloss):.4f}", flush=True)

    with torch.no_grad():
        yv, idv, _ = model(Xv.to(args.device))
        yv = (yv.cpu() * sd + mu)
    names = ["dA->W", "dA->D", "dA->L", "P(W)", "P(D)", "P(B)"]
    print("\n[fidelity] held-out R^2 per output (codes -> evaluation):")
    for j, nm in enumerate(names):
        ss_res = float(((yv[:, j] - Yv[:, j]) ** 2).sum())
        ss_tot = float(((Yv[:, j] - Yv[:, j].mean()) ** 2).sum())
        print(f"  {nm:6s} R^2 = {1 - ss_res / ss_tot:.3f}")
    idv = idv.cpu().numpy()
    for h in range(args.heads):
        cnt = np.bincount(idv[:, h], minlength=args.codes) / len(idv)
        perp = float(np.exp(-(cnt[cnt > 0] * np.log(cnt[cnt > 0])).sum()))
        print(f"[codebook] head {h}: perplexity {perp:.1f}/{args.codes} "
              f"(top code {cnt.max():.0%})")

    # ---- post-hoc NAMING (predicates never enter training) ----
    tok_v = tr.tok[val_rows]
    VAL = {0: 0, 1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0, 7: 1, 8: 3, 9: 3, 10: 5, 11: 9, 12: 0}
    vmat = np.vectorize(VAL.get)(tok_v)
    wm = np.where((tok_v >= 1) & (tok_v <= 6), vmat, 0).sum(1)
    bm = np.where(tok_v >= 7, vmat, 0).sum(1)
    preds = {"mat_diff>2": (wm - bm) > 2, "mat_diff<-2": (wm - bm) < -2,
             "queens_on": ((tok_v == 5).any(1) | (tok_v == 11).any(1)),
             "few_pieces(<=8)": (tok_v > 0).sum(1) <= 8,
             "pawn_endgame": (~(((tok_v >= 2) & (tok_v <= 5)) |
                               ((tok_v >= 8) & (tok_v <= 11))).any(1))}
    print("\n[naming] strongest predicate per frequent code (top 3 codes/head):")
    for h in range(args.heads):
        cnt = np.bincount(idv[:, h], minlength=args.codes)
        for code in np.argsort(-cnt)[:3]:
            m = idv[:, h] == code
            if m.sum() < 100:
                continue
            best, bl = max(((abs(float(p[m].mean()) - float(p.mean())), k)
                            for k, p in preds.items()))
            k = bl
            print(f"  h{h}/c{code} ({m.mean():.0%} of rows): {k} "
                  f"{float(preds[k][m].mean()):.0%} vs base {float(preds[k].mean()):.0%}")


if __name__ == "__main__":
    main()
