#!/usr/bin/env python
"""experiments/m3b_sae_concepts.py -- M3b concept mining, NO HAND-CODED FEATURES (Kaveh
2026-07-29). Concepts are LEARNED: a TopK sparse dictionary (dictionary_learning.AutoEncoderTopK)
over the frozen trunk's per-square tokens; a position's atom score = max activation over its 64
squares; then the SAME matched case-control harness as before (sharpness x phase x band strata,
game-clustered CIs) decides which atoms afford errors vs protect.

Multiple-comparison discipline: atoms are SELECTED on even-hash games (top |effect|) and
CONFIRMED on odd-hash games with clustered CIs -- an atom counts only if it survives
confirmation. Naming is post-hoc: each confirmed atom ships with its top-activating FENs
(the catalog artifact); we describe atoms AFTER the data picks them, never before.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.m3b_factors import build_strata, stratified_effect     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="data/derived/transition_data_labeled.npz")
    ap.add_argument("--tokens-cache", default="data/derived/m3b_trunk_tokens.npz")
    ap.add_argument("--dict-size", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--sae-steps", type=int, default=3000)
    ap.add_argument("--n-cand", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--case-thr", type=float, default=0.2)
    ap.add_argument("--ctrl-thr", type=float, default=0.05)
    args = ap.parse_args()
    t0 = time.time()
    from catspace.train.scaffold import resolve_device
    dev = resolve_device("auto")

    d = dict(np.load(args.labeled, allow_pickle=True))
    ok = ~np.isnan(d["mover_loss"])
    fen = d["fen"][ok]; y = d["mover_loss"][ok]; cb = d["committor_before"][ok]
    elo = d["elo_mover"][ok]; game = d["game"][ok].astype(np.int64)
    case = y >= args.case_thr; ctrl = y < args.ctrl_thr; keep = case | ctrl

    # ---- trunk per-square tokens (materialized once; TESTING §2.16) ----
    tc = Path(args.tokens_cache)
    if tc.exists():
        tok = np.load(tc)["tok"]
    else:
        from catspace.field import ReachabilityField
        from lczerolens import LczeroBoard
        rf = ReachabilityField(device=str(dev))
        outs = []
        for i in range(0, len(fen), 512):
            boards = [LczeroBoard(f) for f in fen[i:i + 512]]
            x = torch.stack([b.to_input_tensor() for b in boards]).float().to(rf.dev)
            with torch.no_grad():
                rf.trunk(x)
            t = rf._f["t"].detach()                      # (B, 64, C) tokens
            outs.append(t.reshape(len(boards), 64, -1).to(torch.float16).cpu().numpy())
            if i % 10240 == 0:
                print(f"  tokens {i:,}/{len(fen):,} [{time.time()-t0:.0f}s]", flush=True)
        tok = np.concatenate(outs)
        np.savez_compressed(tc, tok=tok)
        print(f"cached tokens {tok.shape} -> {tc}")
    C = tok.shape[-1]

    # ---- TopK SAE on square tokens (their model class; plain MSE loop) ----
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    rng = np.random.default_rng(0); torch.manual_seed(0)
    flat = tok.reshape(-1, C)
    sub = flat[rng.choice(len(flat), min(1_500_000, len(flat)), replace=False)]
    X = torch.from_numpy(sub.astype(np.float32)).to(dev)
    sae = AutoEncoderTopK(C, args.dict_size, args.k).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-4)
    for s in range(args.sae_steps):
        b = X[torch.randint(0, len(X), (4096,), device=dev)]
        xhat = sae(b)
        loss = ((xhat - b) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        r2 = 1 - ((sae(X[:65536]) - X[:65536]) ** 2).mean() / X[:65536].var()
    print(f"SAE trained: dict {args.dict_size} k={args.k} | recon R^2 {r2:.3f} "
          f"[{time.time()-t0:.0f}s]")

    # ---- position-level atom scores: max over squares, chunked ----
    acts = np.empty((len(fen), args.dict_size), np.float16)
    with torch.no_grad():
        for i in range(0, len(fen), 512):
            t = torch.from_numpy(tok[i:i + 512].astype(np.float32)).to(dev).reshape(-1, C)
            a = sae.encode(t).reshape(-1, 64, args.dict_size)
            acts[i:i + 512] = a.max(1).values.to(torch.float16).cpu().numpy()
    dead = (acts.max(0) == 0).sum()
    print(f"position atom scores {acts.shape} | dead atoms {dead}/{args.dict_size}")

    # ---- matched harness: SELECT on even games, CONFIRM on odd ----
    pieces = np.array([sum(ch.isalpha() for ch in f.split()[0]) for f in fen])
    strat = build_strata(cb, pieces, elo)
    even = keep & (game % 2 == 0); odd = keep & (game % 2 == 1)
    A = acts.astype(np.float32)
    eff_sel = np.array([stratified_effect(A[even, j], case[even], strat[even])
                        for j in range(args.dict_size)])
    scale = np.array([max(A[even, j].std(), 1e-6) for j in range(args.dict_size)])
    # DIRECTIONAL selection (2026-07-29): |effect| selection alone returned 0 attacking atoms --
    # a selection artifact (protective magnitudes dominate: -0.47 vs +0.20 SD). Select each
    # direction separately so the weaker-but-real attacking tail is represented.
    zsel = eff_sel / scale
    cand = np.concatenate([np.argsort(-zsel)[: args.n_cand // 2],
                           np.argsort(zsel)[: args.n_cand // 2]])

    games_o = np.unique(game[odd]); gidx = {g: np.flatnonzero(game[odd] == g) for g in games_o}
    Ao, co, so = A[odd], case[odd], strat[odd]
    confirmed = []
    for j in cand:
        obs = stratified_effect(Ao[:, j], co, so)
        boots = np.empty(args.n_boot)
        for bi in range(args.n_boot):
            rows = np.concatenate([gidx[games_o[p]] for p in
                                   np.random.default_rng(bi).integers(0, len(games_o), len(games_o))])
            boots[bi] = stratified_effect(Ao[rows, j], co[rows], so[rows])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        if lo > 0 or hi < 0:
            confirmed.append((int(j), obs / scale[j], lo / scale[j], hi / scale[j]))

    atk = sorted([c for c in confirmed if c[1] > 0], key=lambda c: -c[1])
    prot = sorted([c for c in confirmed if c[1] < 0], key=lambda c: c[1])
    print(f"\nVERDICT M3b-SAE (learned atoms; select even n={args.n_cand}, confirm odd, "
          f"clustered 95% CI, effects in atom-SD units):")
    catalog = {}
    for name, group in (("ATTACKING", atk), ("PROTECTIVE", prot)):
        for j, e, lo, hi in group:
            top_idx = np.argsort(-A[:, j])[:5]
            catalog[j] = [str(fen[i]) for i in top_idx]
            print(f"  {name} atom_{j:04d}  {e:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                  f"top-FEN: {fen[top_idx[0]][:52]}")
    print(f"VERDICT M3b-SAE GATE: {len(atk)} attacking + {len(prot)} protective confirmed "
          f"(need >=5 + >=5) -- {'PASS' if len(atk) >= 5 and len(prot) >= 5 else 'FAIL'}")
    out = Path("artifacts/experiments/m3b_atom_catalog.npz")
    np.savez_compressed(out, atoms=np.array(list(catalog.keys())),
                        fens=np.array([catalog[j] for j in catalog]),
                        effects=np.array([[j, e, lo, hi] for j, e, lo, hi in atk + prot]))
    torch.save({"state_dict": sae.state_dict(), "dict_size": args.dict_size, "k": args.k,
                "C": C}, "artifacts/experiments/m3b_sae_latest.pt")
    print(f"catalog -> {out} | SAE -> artifacts/experiments/m3b_sae_latest.pt "
          f"[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
