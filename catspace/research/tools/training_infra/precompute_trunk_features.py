#!/usr/bin/env python
"""catspace/research/tools/training_infra/precompute_trunk_features.py -- M1: precompute FROZEN Leela-trunk features for the
standard dataset, once, at full MPS throughput (measured ~21k pos/s batched fp32, 23k fp16), so IQE-
head training never re-runs the trunk. Output = float16 OPEN_MEMMAP .npy aligned 1:1 with the input
npz rows (mmap-able at train time -> no decompress-to-RAM; the 7.7GB-in-RAM pattern is retired).
DVC-track the output. Efficiency per MILESTONES locked decision 7 (tensor-batched, no subprocests).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from catspace.io import paths



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", default=paths.engine("maia/maia-1500.onnx"))
    ap.add_argument("--data", default=paths.derived("field_std_v1.npz"))
    ap.add_argument("--out", default="")
    ap.add_argument("--hook", default="", help="module name to hook (default: last trunk relu)")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--rows-mod", type=int, default=0, help="keep rows where game %% mod == 0 (subset by game); 0 = all")
    ap.add_argument("--tokens", action="store_true", help="transformer trunk: hook emits (B*64, C) tokens -> reshape to (B, C, 8, 8)")
    args = ap.parse_args()
    t0 = time.time()
    from lczerolens import LczeroModel

    m = LczeroModel.from_onnx_path(args.onnx).float().to(args.device).eval()
    names = [n for n, _ in m.named_modules() if n]
    trunk = [n for n in names if all(k not in n.lower() for k in ("policy", "value", "wdl", "output", "mlh"))]
    hook_name = args.hook or trunk[-1]
    feats = {}
    dict(m.named_modules())[hook_name].register_forward_hook(lambda mo, i, o: feats.__setitem__("t", o))

    z = np.load(args.data)
    planes = z["planes"]                                     # (N,112,8,8) uint8
    if args.rows_mod:
        rows = np.flatnonzero(z["game"] % args.rows_mod == 0)
        planes = planes[rows]
    else:
        rows = np.arange(len(planes))
    N = len(planes)
    tag = Path(args.onnx).stem
    out = args.out or paths.derived(f"trunk_feats/{tag}__{Path(args.data).stem}.npy")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    def _shape(t, B):
        if args.tokens:                                      # (B*64, C) tokens -> (B, C, 8, 8)
            C = t.shape[-1]
            return t.reshape(B, 64, C).permute(0, 2, 1).reshape(B, C, 8, 8)
        return t
    # probe feature shape
    with torch.no_grad():
        m(torch.from_numpy(planes[:2].astype(np.float32)).to(args.device))
    fshape = tuple(_shape(feats["t"], 2).shape[1:])
    print(f"[precompute] {args.onnx} hook={hook_name} feat{fshape} -> {out} | N={N:,} batch={args.batch}", flush=True)
    mm_out = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(N, *fshape))

    done = 0
    with torch.no_grad():
        for i in range(0, N, args.batch):
            x = torch.from_numpy(planes[i:i + args.batch].astype(np.float32)).to(args.device)
            m(x)
            mm_out[i:i + len(x)] = _shape(feats["t"], len(x)).to(torch.float16).cpu().numpy()
            done += len(x)
            if (i // args.batch) % 20 == 0:
                print(f"  {done:,}/{N:,} [{time.time()-t0:.0f}s, {done/max(time.time()-t0,1e-9):,.0f} pos/s]", flush=True)
    mm_out.flush()
    np.save(Path(out).with_suffix(".rows.npy"), rows)
    meta = dict(onnx=args.onnx, hook=hook_name, data=args.data, n=N, shape=list(fshape),
                dtype="float16", rows_mod=args.rows_mod, elapsed_s=round(time.time() - t0, 1))
    Path(out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"VERDICT precompute: {N:,} x {fshape} fp16 -> {out} "
          f"({Path(out).stat().st_size/1e9:.1f}GB) [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
