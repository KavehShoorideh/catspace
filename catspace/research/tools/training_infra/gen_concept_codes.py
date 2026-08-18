#!/usr/bin/env python
"""gen_concept_codes.py -- freeze a concept model and cache its codes for the whole corpus.

Stage 2 of "concepts first, then geometry" needs the concept STATE for millions of positions,
many times over. Re-running the trunk every step would dominate the quasimetric's training time
(the trunk is 2.35 ms/position single-threaded) and would also be wasted work, because the
encoder is FROZEN by construction at this stage.

So the codes are computed once and cached. Caching the codes rather than the dense state is the
whole trick: 16 global + 64 square indices is 80 bytes per position (~0.5 GB for 6.5M rows),
where the dense state would be 2,560 float32 (~67 GB). The state is rebuilt from the codebooks
by a gather, which is free next to a transformer forward.

    .venv/bin/python -m ...gen_concept_codes --ckpt artifacts/experiments/cef2_latest.pt --save
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/cef2_latest.pt")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_event_field import ConceptEvent
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import factor_vocab
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n = len(tr.tok)
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    cr, ce, co, egs = factor_vocab(vocab)
    net = ConceptEvent(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                       square_codes=A["square_codes"], hidden=A["hidden"],
                       n_res=len(vocab), factored=True, n_eg=len(egs),
                       cls_reach=cr, cls_eg=ce, cls_out=co).to(args.device)
    net.load_state_dict(ck["state_dict"]); net.eval()

    H, SC = A["heads"], A["square_codes"]
    g_out = np.zeros((n, H), np.int16)
    s_out = np.zeros((n, 64), np.int16)
    t0 = time.time()
    with torch.no_grad():
        for a in range(0, n, args.batch):
            b = slice(a, min(a + args.batch, n))
            tok = torch.from_numpy(tr.tok[b].astype(np.int64)).to(args.device)
            glb = torch.from_numpy(tr.glob[b].astype(np.float32)).to(args.device)
            _, ids, ids_sq, _ = net.encode(tok, glb)
            g_out[b] = ids.cpu().numpy().astype(np.int16)
            s_out[b] = ids_sq.cpu().numpy().astype(np.int16)
            if (a // args.batch) % 100 == 0:
                el = time.time() - t0
                done = min(a + args.batch, n)
                print(f"  {done:>9,}/{n:,}  {done/max(el,1e-9):,.0f} pos/s  "
                      f"eta {(n-done)/max(done/max(el,1e-9),1e-9)/60:.1f}m", flush=True)
    el = time.time() - t0
    print(f"\n  encoded {n:,} positions in {el/60:.1f}m ({n/el:,.0f} pos/s)")
    print(f"  global codes {g_out.shape} using {len(np.unique(g_out))} distinct values")
    print(f"  square codes {s_out.shape} using {len(np.unique(s_out))} of {SC}")

    # the codebooks travel with the codes, so the state can be rebuilt without the trunk
    gb = torch.stack([net.vq[h]._codebook.embed[0].detach().cpu() for h in range(H)]).numpy()
    sb = net.sq_vq._codebook.embed[0].detach().cpu().numpy()
    if args.save:
        p = paths.derived(f"concept_codes_{A['out'].split('/')[-1]}_{n}.npz")
        np.savez_compressed(p, g=g_out, s=s_out, g_book=gb, s_book=sb,
                            ckpt=np.array(args.ckpt, dtype=object),
                            step=np.array(ck.get("step", -1)))
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
