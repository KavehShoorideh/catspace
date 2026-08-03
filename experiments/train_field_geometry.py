#!/usr/bin/env python
"""experiments/train_field_geometry.py -- the SIMPLIFIED geometry objective
(Kaveh 2026-07-20). The quasimetric field = LEGAL reachability (what COULD be played),
learned from the directed move graph. The one-way / strata structure EMERGES; it is
not hand-built.

  L_pos   successor pin: d(s -> s') ~ 1 for a legal ply s->s'  (positives = edges)
  L_neg   triplet negatives: d(s->s'') >= d(s->s') + margin for random non-successors
  L_hard  HARD negative: d(s' -> s) >> floor for IRREVERSIBLE edges only. Random
          negatives never touch the specific reverse pair, so the one-way structure
          does NOT emerge on its own (verified: 200-step smoke gave 1.07x for
          irreversible vs 0.91x reversible -- both symmetric). Irreversible = the
          rule-defined set (chess.Board.is_irreversible): pawn moves, captures,
          castling, AND any move that reduces castling rights / gives up en passant.
          Reversible reverses are left symmetric.
  L_rank  within-material DTM ranking on the near-mate set (the goal direction)
  L_sym   horizontal-mirror invariance
  L_sep   material separation, PAWNLESS-ONLY (with pawns, promotion bridges classes)

The one-way is a TARGETED hard negative on rule-defined irreversibility, not a
pawn-specific L_inf hinge and not left to random negatives. Pawn-death asym is an
EVAL PROBE. Temporarily-illegal moves (e.g. castling blocked by a bishop) simply have
no edge, so nothing special is needed -- the path routes through the unblocking move.

BOARD-ONLY geometry: the halfmove clock (plane 18) + repetition (plane 19) are zeroed
into the distance tower -- they are separate monotone potentials for the planner, and
zeroing keeps shuffle-equivalent positions identifiable (clusterable).

Per-loss EMA scale-normalization so no term dominates the gradient by raw magnitude.

Usage:
  .venv/bin/python experiments/train_field_geometry.py --steps 3000 \
    --ckpt data/derived/sep/iqe_nucleus_gn.pt --out data/derived/sep/iqe_geom.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device, save_ckpt
from scipy.stats import spearmanr

BOARD_ONLY_ZERO = (18, 19)   # halfmove clock, repetition -> excluded from the geometry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/iqe_nucleus_gn.pt")
    ap.add_argument("--nearmate", default="data/derived/lichess_nearmate.npz")
    ap.add_argument("--edges", default="data/derived/successor_edges.npz")
    ap.add_argument("--pawndeath", default="data/derived/pawndeath_pairs.npz", help="EVAL probe only")
    ap.add_argument("--out", default="data/derived/sep/iqe_geom.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--neg-margin", type=float, default=3.0)
    ap.add_argument("--irrev-margin", type=float, default=6.0,
                    help="RELATIVE: d(child->parent) >= d(parent->child) + this for irreversible edges")
    ap.add_argument("--repel-floor", type=float, default=40.0,
                    help="UNREACHABLE-material pairs pushed >> this (huge). Fixes cross-material mate collapse.")
    ap.add_argument("--w-repel", type=float, default=1.0)
    ap.add_argument("--sep-margin", type=float, default=10.0)
    ap.add_argument("--w-pos", type=float, default=1.0)
    ap.add_argument("--w-neg", type=float, default=1.0)
    ap.add_argument("--w-hard", type=float, default=1.0)
    ap.add_argument("--w-rank", type=float, default=1.0)
    ap.add_argument("--w-grank", type=float, default=0.5, help="cross-material (global) DTM anchor")
    ap.add_argument("--w-sym", type=float, default=1.0)
    ap.add_argument("--w-sep", type=float, default=0.3)
    ap.add_argument("--ckpt-every", type=int, default=500, help="periodic checkpoint so a kill/relaunch resumes")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device)
    torch.manual_seed(args.seed)
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.train()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    opt = torch.optim.Adam(fb.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    # near-mate (goal anchoring) -- won positions + material key + pawnless flag
    nz = np.load(args.nearmate)
    won = nz["dtm"] > 0
    nmp, nmm, nmd = nz["packed"][won], nz["meta"][won], nz["dtm"][won].astype(np.float32)
    matkey = np.empty(len(nmd), dtype=object); haspawn = np.zeros(len(nmd), dtype=bool)
    for i in range(len(nmd)):
        ps = list(board_from_packed(nmp[i], nmm[i]).piece_map().values())
        matkey[i] = " ".join(sorted(p.symbol() for p in ps))
        haspawn[i] = any(p.piece_type == chess.PAWN for p in ps)
    matkey = matkey.astype(str)
    uniq = {m: k for k, m in enumerate(sorted(set(matkey)))}
    mat = np.array([uniq[m] for m in matkey], dtype=np.int64)
    low = np.argsort(nmd)[:512]
    ez = np.load(args.edges)                                          # successor edges
    irr_idx = np.flatnonzero(ez["irrev"])                            # irreversible edges (~5%): dedicated
    print(f"[stage] {len(ez['p_packed'])} edges, {len(irr_idx)} irreversible ({100*len(irr_idx)/len(ez['p_packed']):.1f}%)", flush=True)
    pz = np.load(args.pawndeath)                                      # eval probe only

    def bplanes(pk, mt):
        pl = feature_planes(pk, mt)
        pl[:, BOARD_ONLY_ZERO, :, :] = 0.0                           # board-only geometry
        return torch.from_numpy(pl).to(dev)

    def eF(pk, mt):
        return fb.embed_F(bplanes(pk, mt), torch.from_numpy(np.tile(om, (len(pk), 1))).to(dev))

    def eB(pk, mt):
        return fb.embed_B(bplanes(pk, mt))

    def probe_asym():                                                # eval-mode pawn-death one-way
        i = rng.choice(len(pz["p_packed"]), 300, replace=False)
        with torch.no_grad():
            fwd = fb.distance_matrix(eF(pz["p_packed"][i], pz["p_meta"][i]),
                                     eB(pz["c_packed"][i], pz["c_meta"][i])).diagonal().cpu().numpy()
            bwd = fb.distance_matrix(eF(pz["c_packed"][i], pz["c_meta"][i]),
                                     eB(pz["p_packed"][i], pz["p_meta"][i])).diagonal().cpu().numpy()
        return float(np.median(bwd / np.maximum(fwd, 1e-6)))

    print(f"[before] pawn-death one-way asym (EVAL probe) = {probe_asym():.2f}x", flush=True)
    t0 = time.time()
    for step in range(args.steps):
        if step % 500 == 0:
            with torch.no_grad():
                zmate = eB(nmp[low], nmm[low]).mean(0, keepdim=True).detach()
        # -- successor pins + in-batch triplet negatives + irreversible hard negative --
        ei = rng.integers(0, len(ez["p_packed"]), size=args.batch)
        Fp = eF(ez["p_packed"][ei], ez["p_meta"][ei])
        Bc = eB(ez["c_packed"][ei], ez["c_meta"][ei])
        D = fb.distance_matrix(Fp, Bc)                               # (B,B): d(p_i -> c_j)
        d_pos = D.diagonal()
        L_pos = ((d_pos - 1.0) ** 2).mean()                         # legal ply ~ 1 step
        off = ~torch.eye(len(ei), dtype=torch.bool, device=dev)
        L_neg = torch.relu(d_pos[:, None] + args.neg_margin - D)[off].mean()  # non-edges pushed larger
        # HARD negative on a DEDICATED irreversible batch (only ~5% of edges are
        # irreversible, too sparse to mask). Forward d(parent->child)~1 AND backward
        # d(child->parent)>>floor -> the one-way appears on exactly the irreversible edges.
        hi = rng.choice(irr_idx, size=min(args.batch, len(irr_idx)), replace=len(irr_idx) < args.batch)
        d_fwd_i = fb.distance_matrix(eF(ez["p_packed"][hi], ez["p_meta"][hi]),
                                     eB(ez["c_packed"][hi], ez["c_meta"][hi])).diagonal()
        d_bwd = fb.distance_matrix(eF(ez["c_packed"][hi], ez["c_meta"][hi]),
                                   eB(ez["p_packed"][hi], ez["p_meta"][hi])).diagonal()
        L_hard = ((d_fwd_i - 1.0) ** 2).mean() + torch.relu(d_fwd_i.detach() + args.irrev_margin - d_bwd).pow(2).mean()
        # -- goal anchoring: within-material DTM ranking on the near-mate set --
        ni = rng.integers(0, len(nmd), size=args.batch)
        f = eF(nmp[ni], nmm[ni])
        dm = fb.distance_matrix(f, zmate)[:, 0]
        dtm_b = torch.from_numpy(nmd[ni]).to(dev); mat_b = torch.from_numpy(mat[ni]).to(dev)
        same = mat_b[:, None] == mat_b[None, :]
        closer = dtm_b[:, None] < dtm_b[None, :]
        mask = same & closer
        L_rank = torch.relu(1.0 - (dm[None, :] - dm[:, None]))[mask].mean() if mask.any() else torch.zeros((), device=dev)
        # cross-material (global) DTM anchor: higher DTM => larger d-to-mate ACROSS materials
        # too, so the global scale isn't inverted while within-material order forms.
        L_grank = torch.relu(0.5 - (dm[None, :] - dm[:, None]))[closer].mean() if closer.any() else torch.zeros((), device=dev)
        # -- symmetry (mirror) on a sub-batch --
        nsym = ni[:32]
        mir = [board_from_packed(nmp[i], nmm[i]).transform(chess.flip_horizontal) for i in nsym]
        fm = eF(np.stack([encode_packed(b) for b in mir]), np.stack([encode_meta(b) for b in mir]))
        L_sym = ((f[:32] - fm) ** 2).sum(1).mean()
        # -- material separation, PAWNLESS ONLY --
        pl = torch.from_numpy(~haspawn[ni]).to(dev)
        sepm = (~same) & pl[:, None] & pl[None, :]
        L_sep = torch.relu(args.sep_margin - torch.cdist(f, f)[sepm]).pow(2).mean() if sepm.any() else torch.zeros((), device=dev)
        loss = (args.w_pos * L_pos + args.w_neg * L_neg
                + args.w_hard * L_hard + args.w_rank * L_rank + args.w_grank * L_grank
                + args.w_sym * L_sym + args.w_sep * L_sep)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            with torch.no_grad():
                sp = spearmanr(dm.detach().cpu().numpy(), nmd[ni]).correlation
            print(f"  step {step:4d}  L_pos {float(L_pos):.3f} L_neg {float(L_neg):.3f} "
                  f"L_hard {float(L_hard):.3f} L_rank {float(L_rank):.3f} L_grank {float(L_grank):.3f} "
                  f"L_sym {float(L_sym):.3f} L_sep {float(L_sep):.3f}  d_pos {float(d_pos.median()):.2f} "
                  f"d_bwd_irr {float(d_bwd.median()):.2f}  sp(d,DTM) {sp:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
            fb.eval(); save_ckpt(fb, Path(args.out), step=pay.get("step", 0), zgoals=pay.get("zgoals")); fb.train()

    fb.eval()
    a1 = probe_asym()
    save_ckpt(fb, Path(args.out), step=pay.get("step", 0), zgoals=pay.get("zgoals"))
    print(f"saved {args.out}")
    print(f"VERDICT GEOM d_pos~1 pawn-death_asym(EVAL,emergent)={a1:.1f}x "
          f"(one-way should EMERGE from the graph, not a loss term)")


if __name__ == "__main__":
    main()
