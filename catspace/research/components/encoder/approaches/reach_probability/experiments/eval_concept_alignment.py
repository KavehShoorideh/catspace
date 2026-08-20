#!/usr/bin/env python
"""eval_concept_alignment.py -- do the learned concepts align with the NAMED Rust basis?

The interpretability claim, finally measured instead of promised: encode positions with a
trained graph-concept model, extract the 1,892-dim named basis for the same positions with the
Rust crate (zero-mismatch-verified against python-chess), and ask how much of each named feature
the discrete codes explain.

HONESTY SPLIT, enforced in the output. Some named features are INPUTS to the graph encoder
(attacker/defender counts, pawn-attack, king-ring flags, piece identity) -- alignment there is
partly an echo of the featurization. The features that are NOT inputs -- SEE, pinned/pinning,
x-ray, least-attacker, mobility, pawn structure (passed/isolated/doubled/backward), king
shield/escape/open-files, tension, legal-move count, checks -- are computed by the basis and
never shown to the model. Alignment on THOSE is learned structure, and that is where the
interpretability claim lives.

Statistic: eta^2 -- the fraction of a named feature's variance explained by the discrete code
(1 - E[within-code variance]/variance). Works uniformly for binary and numeric features. For
marquee tactical bits (hanging, pinned, SEE<0) the per-code precision/lift table names the
specific codes.

    .venv/bin/python -m ...eval_concept_alignment --ckpt artifacts/experiments/gcf1_latest.pt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch


INPUT_ECHO_SQ = {"occ_", "att_w", "att_b", "pawn_att_w", "pawn_att_b", "king_ring_w",
                 "king_ring_b", "defenders"}
INPUT_ECHO_GL = {"n_", "stm_white", "castle_", "ep_file", "piece_count"}


def is_echo(name, table):
    return any(name.startswith(p) or name == p.rstrip("_") for p in table)


def eta2(codes, feats, k):
    """codes (N,) int, feats (N,F) -> (F,) fraction of variance explained by code identity."""
    N, F = feats.shape
    tot = feats.var(0)
    dead = tot < 1e-8            # constant features: eta^2 is 0/0, NOT 1.0 -- mask them
    tot = tot + 1e-12
    sums = np.zeros((k, F)); sqs = np.zeros((k, F)); cnt = np.bincount(codes, minlength=k)
    np.add.at(sums, codes, feats)
    np.add.at(sqs, codes, feats ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        wvar = sqs / np.maximum(cnt[:, None], 1) - (sums / np.maximum(cnt[:, None], 1)) ** 2
    within = (wvar * cnt[:, None]).sum(0) / N
    out = 1.0 - within / tot
    out[dead] = -1.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/gcf1_latest.pt")
    ap.add_argument("--graphs", default="data/derived/move_graphs_v2_300000_6545146.npz")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import catspace_concepts as cc
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_concept_field import GraphConceptField
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize, MV_CAP, N_NODES
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import factor_vocab
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    dev = args.device
    t0 = time.time()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    z = np.load(args.graphs)
    D = {k: torch.from_numpy(np.ascontiguousarray(z[k])) for k in
         ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "n_mv", "n_att", "p_row")}
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n = len(tr.tok)
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    cr, ce, co, egs = factor_vocab(vocab)
    net = GraphConceptField(d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"],
                            n_res=len(vocab), n_eg=len(egs),
                            cls_reach=cr, cls_eg=ce, cls_out=co,
                            edge_codes=A.get("edge_codes", 0), move_codes=A.get("move_codes", 0),
                            gps_parallel=bool(A.get("gps_parallel", 0)),
                            attn_bias=bool(A.get("attn_bias", 0)),
                            agg_max=bool(A.get("agg_max", 0))).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(D["p_row"]), min(args.n, len(D["p_row"])), replace=False))
    rows = D["p_row"][idx].numpy()

    def parent_batch(ii):
        b = {k: D[k][ii].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
        sqf, mvf, edges, mask = featurize(b, dev)
        off = (torch.arange(len(ii), device=dev) * N_NODES)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for name in ("vac", "dsc"):
            a = D[name][ii].to(dev).long()
            val = (a != 255) & mask[:, :, None]
            edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a))[val],
                           (a + off[:, :, None])[val])
        return sqf, mvf, edges, mask

    G, S = [], []
    with torch.no_grad():
        for a in range(0, len(idx), 512):
            ii = torch.from_numpy(idx[a:a + 512])
            sqf, mvf, edges, mask = parent_batch(ii)
            _, g_ids, sq_ids, _, _, _ = net.encode(sqf, mvf, edges, mask)
            G.append(g_ids.cpu().numpy()); S.append(sq_ids.cpu().numpy())
    G = np.concatenate(G); S = np.concatenate(S)
    print(f"[align] encoded {len(G):,} positions [{time.time()-t0:.0f}s]", flush=True)

    sq_feat, gl_feat, failed = cc.extract(np.ascontiguousarray(tr.tok[rows]),
                                          np.ascontiguousarray(tr.glob[rows]))
    assert failed == 0
    SQ_NAMES, GLOB_NAMES = cc.feature_names()
    print(f"[align] named basis extracted [{time.time()-t0:.0f}s]\n", flush=True)

    H, C, SC = A["heads"] if "heads" in A else 16, 32, 256
    H = G.shape[1]
    slow_k = A.get("slow_heads", 8)

    # ---- GLOBAL heads vs global named features ----------------------------------------
    print(f"  GLOBAL heads ({H} x {C}; heads 0-{slow_k-1} SLOW via VAMP-2, rest fast)")
    print(f"  {'head':>5}  {'top named features by eta^2 (fraction of variance explained)'}")
    best_by_feat = {}
    for h in range(H):
        e = eta2(G[:, h], gl_feat, C)
        order = np.argsort(-e)
        tags = []
        for j in order[:3]:
            nm = GLOB_NAMES[j]
            mark = "~" if is_echo(nm, INPUT_ECHO_GL) else ""
            tags.append(f"{mark}{nm} {e[j]:.2f}")
        kind = "SLOW" if h < slow_k else "fast"
        print(f"  {h:>3} {kind:>5}  " + " | ".join(tags))
        for j in range(len(GLOB_NAMES)):
            if e[j] > best_by_feat.get(GLOB_NAMES[j], (0, -1))[0]:
                best_by_feat[GLOB_NAMES[j]] = (e[j], h)

    marquee_gl = ["material_diff", "phase_nonpawn", "tension", "legal_moves",
                  "checks_available", "in_check", "king_shield_w", "king_escape_w",
                  "king_open_files_w", "pawn_passed_w_file4", "pawn_isolated_w_file3",
                  "pawn_doubled_b_file2"]
    print(f"\n  LEARNED (non-input) global features -- best head and eta^2:")
    for nm in marquee_gl:
        if nm in best_by_feat:
            e, h = best_by_feat[nm]
            kind = "SLOW" if h < slow_k else "fast"
            print(f"    {nm:<26} head {h:>2} ({kind})  eta2 {e:.3f}")

    # ---- SQUARE codes vs per-square named features -------------------------------------
    flat_codes = S.reshape(-1)
    flat_feats = sq_feat.reshape(-1, sq_feat.shape[-1])
    e_sq = eta2(flat_codes, flat_feats, SC)
    print(f"\n  SQUARE codes (256, shared) vs per-square named features")
    order = np.argsort(-e_sq)
    for j in order[:12]:
        nm = SQ_NAMES[j]
        mark = "~input" if is_echo(nm, INPUT_ECHO_SQ) else "LEARNED"
        print(f"    {nm:<18} eta2 {e_sq[j]:.3f}   {mark}")

    # marquee tactical bits: which CODES fire on them, precision and lift
    print(f"\n  tactical bits (never inputs) -- top codes by precision:")
    for nm, cond in (("hanging", flat_feats[:, SQ_NAMES.index("hanging")] > 0.5),
                     ("pinned", flat_feats[:, SQ_NAMES.index("pinned")] > 0.5),
                     ("SEE<0 on occupied", flat_feats[:, SQ_NAMES.index("see")] < 0)):
        base = cond.mean()
        prec = np.zeros(SC); cnt = np.bincount(flat_codes, minlength=SC)
        np.add.at(prec, flat_codes, cond.astype(float))
        with np.errstate(invalid="ignore"):
            prec = prec / np.maximum(cnt, 1)
        top = np.argsort(-np.where(cnt > 2000, prec, -1))[:3]
        s = "  ".join(f"code {c}: P={prec[c]:.2f} lift {prec[c]/max(base,1e-9):.0f}x "
                      f"(n={cnt[c]:,})" for c in top)
        print(f"    {nm:<18} base {base:.3f} | {s}")


if __name__ == "__main__":
    main()
