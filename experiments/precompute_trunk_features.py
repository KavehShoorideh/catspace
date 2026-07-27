#!/usr/bin/env python
"""experiments/precompute_trunk_features.py -- M1: precompute FROZEN Leela-trunk features for the
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", default="data/engines/maia/maia-1500.onnx")
    ap.add_argument("--data", default="data/derived/field_std_v1.npz")
    ap.add_argument("--out", default="")
    ap.add_argument("--hook", default="", help="module name to hook (default: last trunk relu)")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    t0 = time.time()
    from lczerolens import LczeroModel

    m = LczeroModel.from_onnx_path(args.onnx).float().to(args.device).eval()
    names = [n for n, _ in m.named_modules() if n]
    trunk = [n for n in names if all(k not in n for k in ("policy", "value", "wdl", "output"))]
    hook_name = args.hook or trunk[-1]
    feats = {}
    dict(m.named_modules())[hook_name].register_forward_hook(lambda mo, i, o: feats.__setitem__("t", o))

    z = np.load(args.data)
    planes = z["planes"]                                     # (N,112,8,8) uint8
    N = len(planes)
    tag = Path(args.onnx).stem
    out = args.out or f"data/derived/trunk_feats/{tag}__{Path(args.data).stem}.npy"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # probe feature shape
    with torch.no_grad():
        m(torch.from_numpy(planes[:2].astype(np.float32)).to(args.device))
    fshape = tuple(feats["t"].shape[1:])
    print(f"[precompute] {args.onnx} hook={hook_name} feat{fshape} -> {out} | N={N:,} batch={args.batch}", flush=True)
    mm_out = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(N, *fshape))

    done = 0
    with torch.no_grad():
        for i in range(0, N, args.batch):
            x = torch.from_numpy(planes[i:i + args.batch].astype(np.float32)).to(args.device)
            m(x)
            mm_out[i:i + len(x)] = feats["t"].to(torch.float16).cpu().numpy()
            done += len(x)
            if (i // args.batch) % 20 == 0:
                print(f"  {done:,}/{N:,} [{time.time()-t0:.0f}s, {done/max(time.time()-t0,1e-9):,.0f} pos/s]", flush=True)
    mm_out.flush()
    meta = dict(onnx=args.onnx, hook=hook_name, data=args.data, n=N, shape=list(fshape),
                dtype="float16", elapsed_s=round(time.time() - t0, 1))
    Path(out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"VERDICT precompute: {N:,} x {fshape} fp16 -> {out} "
          f"({Path(out).stat().st_size/1e9:.1f}GB) [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
