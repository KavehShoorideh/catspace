#!/usr/bin/env python
"""catspace/research/tools/figures/fig_hazard.py -- the paper's Figure 4 as a source-agnostic script:
hazards -> survival -> timing -> reachability -> expected discount, exact
arithmetic from whatever emitted the hazards.

Sources:
  --lam npz:key : any stored hazard array -- (H,) one row, or (N,H) averaged
  --ckpt + --fen: a JEPA T1 checkpoint's any-event head on one position

Panels: (a) lambda(h), (b) S(h) survival steps, (c) f(h)=lambda*S(h-1) with the
open "never within horizon" bar, (d) R(h)=1-S(h); gamma printed in the caption
both correct (E[rho^k]) and wrong (rho^E[k]) to display the Jensen gap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from catspace.research.tools.figures import figlib                                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", default="", help="file.npz:key holding hazards")
    ap.add_argument("--ckpt", default="", help="JEPA T1 ckpt (any-event head)")
    ap.add_argument("--fen", default="")
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--fig", default="hazard_fig.png")
    args = ap.parse_args()
    if args.lam:
        path, key = args.lam.rsplit(":", 1)
        lam = np.load(path, allow_pickle=True)[key].astype(float)
        lam = lam.mean(0) if lam.ndim == 2 else lam
        src = f"{Path(path).name}:{key}"
    else:
        import chess
        import torch
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import JepaT1, tokenize
        from catspace.research.tools.training_infra.train.scaffold import resolve_device
        dev = resolve_device("auto")
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        m = JepaT1(**{k: ck["cfg"][k] for k in ("d", "layers", "n_class")}).to(dev)
        m.load_state_dict(ck["state_dict"]); m.eval()
        t, g = tokenize(chess.Board(args.fen))
        with torch.no_grad():
            phi = m.enc(torch.as_tensor(t[None]).to(dev), torch.as_tensor(g[None]).to(dev))
            lam = torch.sigmoid(m.haz(phi, torch.zeros(1, 2, device=dev)))[0].cpu().numpy()
        src = "jepa any-event"
    H = len(lam)
    S = np.cumprod(1 - lam)
    Sprev = np.concatenate([[1.0], S[:-1]])
    f = lam * Sprev
    R = 1 - S
    m_h = np.arange(1, H + 1)
    gamma = float((args.rho ** m_h * f).sum())
    ek = float((m_h * f).sum() + (H + 1) * S[-1])
    gamma_wrong = args.rho ** ek
    print(f"VERDICT hazard-identities[{src}]: R(H) {R[-1]:.3f} | f(inf) {S[-1]:.3f} | "
          f"gamma=E[rho^k] {gamma:.4f} vs rho^E[k] {gamma_wrong:.4f} "
          f"(Jensen gap {gamma - gamma_wrong:+.4f})")
    fig, ax = figlib.new_fig(4, w=3.2, h=2.6)
    x = np.arange(1, H + 1)
    ax[0].bar(x, lam, color=figlib.ACCENT, width=0.7)
    ax[0].set_title("hazards λ(h)"); ax[0].set_xlabel("bucket")
    ax[1].step(np.concatenate([[0], x]), np.concatenate([[1.0], S]),
               where="post", color=figlib.ACCENT)
    ax[1].set_ylim(0, 1.02); ax[1].set_title("survival S(h)"); ax[1].set_xlabel("bucket")
    ax[2].bar(x, f, color=figlib.ACCENT, width=0.7)
    ax[2].bar([H + 1], [S[-1]], color=figlib.SURFACE, edgecolor=figlib.INK, width=0.7)
    ax[2].set_title("timing f(h) + never-bar"); ax[2].set_xlabel("bucket (∞ open)")
    ax[3].plot(x, R, color=figlib.ACCENT, marker="o", ms=4)
    ax[3].set_ylim(0, 1.02); ax[3].set_title("reachability R(h)"); ax[3].set_xlabel("bucket")
    figlib.save(fig, args.fig, f"From hazards to timing, reachability, discount — {src}")


if __name__ == "__main__":
    main()
