#!/usr/bin/env python
"""catspace/research/tools/embeddings/basin_hazard_field.py -- HAZARDS FOR HUMANS as a scalar
field, from two dynamics-conditioned committors.

THE PROBLEM THIS SOLVES. We want the regions where human play leaks outcome relative to engine
play. The obvious route -- embed the basins and look at where the two populations sit differently
-- is blocked: the field is a QUASIMETRIC, d(a->b) != d(b->a), and no 2-D point embedding can be
faithful to one. Measured, not assumed: basin_trilateration.py has median rms misfit 0.358 against
an anchor radius of 0.65 (55%), basin_trilat3d.py cuts that by 48% and is still a projection, and
basin_perply_umap.py showed outcome does not live in neighbourhood structure at all (54% of phi
variance is linear in ply, 1.8% in outcome).

THE WAY AROUND IT. A committor is defined WITH RESPECT TO A DYNAMICS -- the argument already made
in build_combined_field_data.py's docstring, there as a reason not to share a TB-corrected label.
Take it literally and train TWO fields, identical in every way except which population's games
they see (train_iqe_head.py --source human / --source sf). Then

    q_X(s) = P(White wins | s, X plays on) - P(Black wins | s, X plays on)      X in {H, SF}
    h(s)   = q_SF(s) - q_H(s)

h is a SCALAR. Scalars need no metric-faithful embedding, so the quasimetric obstruction simply
does not apply. Its sign is readable: h(s) > 0 means the position is worth more under engine play
than it is worth when humans play it out, i.e. humans give something away from here. That is a
hazard. h(s) < 0 means humans do BETTER than engines from here -- a swindle region, where the
practical chances a human generates exceed what perfect play would concede.

The natural chart is then the OFF-DIAGONAL PLOT: every position placed at (q_SF, q_H). Two
committor axes, both plain probabilities. On the diagonal the two dynamics agree; distance off it
IS the hazard, drawn without any dimensionality reduction, reflection artifact or symmetrisation.

WHAT IS AND IS NOT COMPARABLE. The two fields' phi spaces are NOT aligned and nothing here assumes
they are -- only the probabilities are compared, and those are commensurable exactly to the extent
both fields are calibrated, which their ECE gate measures. Where a THIRD, shared coordinate is
needed (clustering the hazard regions), this uses the incumbent field trained on BOTH populations
as the common chart, and saves phi from it alongside. See basin_hazard_areas.py.

TWO CONFOUNDS, both handled rather than hidden:
  * SUPPORT. h at a state only one population ever reaches is an extrapolation by the other's
    field. Cells are reported only where both populations have >= --min-count observed positions,
    and the mass that gates out is printed, not silently dropped.
  * OPENING COMPOSITION. SF games start from the top-100k most frequent human ply-8 prefixes, so
    below ply 8 the populations differ in which openings appear, not in how they are played.
    --min-ply defaults to 8 for that reason.

Positions come from FULL-GAME REPLAY (basin_tent_fullgames.replay), never the stored row subsample,
so the ply axis is complete. Both heads read the SAME frozen trunk features -- ReachabilityField
.trunk_feats -- so the trunk is paid for once and the two committors see bit-identical inputs.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from catspace.research.tools.training_infra.losses import basin_logp, WIN, DRAW, LOSS
from catspace.research.tools.embeddings.basin_tent import white_pov_x
from catspace.research.tools.embeddings.basin_simplex_chart import INK, MUTED
from catspace.research.tools.embeddings.basin_tent_fullgames import (
    replay, population_games_human, population_games_sf, parity_smooth)

COLOR_HUMAN, COLOR_SF = "#2a78d6", "#e34948"     # the fixed pairing used across the basin figures
CMAP_HAZARD = "coolwarm"                          # diverging, neutral midpoint, symmetric about 0
CMAP_DENSITY = "Blues"                            # sequential, single hue


def load_head(ckpt, device):
    """A second/third IQEHead on an already-loaded trunk. Mirrors ReachabilityField's loader."""
    from catspace.research.components.encoder.approaches.reachability_field.src.iqe_head import IQEHead
    p = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = p["cfg"]
    head = IQEHead(in_ch=cfg["in_ch"], d=cfg["d"], components=cfg["components"],
                   adapter_ch=cfg["adapter_ch"]).to(device)
    missing, _ = head.load_compat(p["state_dict"])
    if any(m.startswith(("poles", "log_T")) for m in missing):
        raise SystemExit(f"{ckpt} has no trained poles -- its basin readout would be noise")
    head.eval()
    return head


def fens_for(ucis, n):
    """First `n` plies of a game -> their FENs. Saved alongside phi so basin_hazard_areas.py can
    show the actual BOARDS of a hazard cluster without re-replaying anything: a cluster you cannot
    look at is not an explanation."""
    import chess
    b = chess.Board()
    out = []
    for u in ucis[:n]:
        try:
            b.push(chess.Move.from_uci(u))
        except Exception:
            break
        out.append(b.fen())
    return out


@torch.no_grad()
def evaluate(field, heads, pools, max_ply, batch, keep_phi="shared"):
    """Replay every game in `pools` and score each ply with EVERY head.

    Returns a flat dict of column arrays: one row per replayed position, so downstream binning is
    plain numpy. The trunk runs ONCE per batch and its features are handed to each head.
    """
    cols = {k: [] for k in ("source", "gid", "ply", "result", "phi", "fen")}
    cols.update({f"q_{n}": [] for n in heads})
    for sname, pool in pools.items():
        for gid, res, ucis, _tm in pool:
            planes, _board, _trunc = replay(ucis, max_ply)
            if planes is None or len(planes) < 6:
                continue
            fens = fens_for(ucis, len(planes))
            if len(fens) != len(planes):
                continue                                   # replay disagreed; drop rather than mis-key
            per_head = {n: [] for n in heads}
            phis = []
            for i in range(0, len(planes), batch):
                t = field.trunk_feats(list(planes[i:i + batch].astype(np.float32)))
                for n, hd in heads.items():
                    e = hd.phi(t)
                    per_head[n].append(basin_logp(hd.d_poles(e), hd.temperature).exp().cpu().numpy())
                    if n == keep_phi:
                        phis.append(e.cpu().numpy().astype(np.float16))
            ply = np.arange(len(planes))
            for n in heads:
                cols[f"q_{n}"].append(white_pov_x(np.concatenate(per_head[n]), ply))
            cols["phi"].append(np.concatenate(phis))
            cols["fen"].append(np.asarray(fens, dtype="U100"))   # fixed width -> no pickle on load
            cols["ply"].append(ply)
            cols["gid"].append(np.full(len(ply), gid, np.int64))
            cols["result"].append(np.full(len(ply), res, np.int8))
            cols["source"].append(np.full(len(ply), 0 if sname == "human" else 1, np.int8))
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}


def gated_cells(x, y, w, xb, yb, min_count):
    """Mean of `w` per (x,y) cell, NaN where the cell holds fewer than `min_count` samples."""
    n, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    s, _, _ = np.histogram2d(x, y, bins=[xb, yb], weights=w)
    out = np.divide(s, n, out=np.full_like(s, np.nan), where=n >= min_count)
    return out, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human-ckpt", default="artifacts/experiments/iqe_poles_human_v2_latest.pt")
    ap.add_argument("--sf-ckpt", default="artifacts/experiments/iqe_poles_sf_v2_latest.pt")
    ap.add_argument("--shared-ckpt", default="artifacts/experiments/movie4/iqe_4pole_30k_latest.pt",
                    help="field trained on BOTH populations; supplies the common phi coordinate "
                         "for basin_hazard_areas.py and a third, control committor")
    ap.add_argument("--onnx", default="assets/engines/lc0/t1-256x10.onnx")
    ap.add_argument("--sf-moves", default="data/derived/opening_pool_sfsf_moves.tsv")
    ap.add_argument("--human-records", default="data/records/lichess_2019-01")
    ap.add_argument("--n-games", type=int, default=1500,
                    help="games per source, sampled UNIFORMLY (population-representative: a "
                         "stratified sample would misstate how much hazard mass there actually is)")
    ap.add_argument("--max-ply", type=int, default=200)
    ap.add_argument("--min-ply", type=int, default=8,
                    help="below this the two populations differ in opening COMPOSITION, not "
                         "dynamics (SF starts from the head of the human opening distribution)")
    ap.add_argument("--min-count", type=int, default=25)
    ap.add_argument("--smooth", type=int, default=2,
                    help="parity box filter width; the field is turn-dependent (lag-1 autocorr "
                         "BELOW lag-2) and differencing raw per-ply values amplifies that")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_hazard")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField
    field = ReachabilityField(onnx=args.onnx, head=args.shared_ckpt)
    if not field.has_poles:
        raise SystemExit(f"{args.shared_ckpt} has no trained poles")
    heads = {"shared": field.head,
             "human": load_head(args.human_ckpt, field.dev),
             "sf": load_head(args.sf_ckpt, field.dev)}
    print(f"trunk + 3 heads loaded [{time.time()-t0:.0f}s]", flush=True)

    rng = np.random.default_rng(args.seed)
    # POPULATION proportions, not the stratified loaders: the headline is an occupancy-weighted
    # expected leaked score, so over-representing draws (the stratified pool gives humans 36%
    # against a true 4.5%) would put the weight on the wrong positions.
    pools = {"human": population_games_human(args.human_records, args.n_games, rng),
             "SF-vs-SF": population_games_sf(args.sf_moves, args.n_games, rng)}
    for k, v in pools.items():
        print(f"  {k}: {len(v)} games", flush=True)

    d = evaluate(field, heads, pools, args.max_ply, args.batch)
    print(f"  scored {len(d['ply']):,} positions [{time.time()-t0:.0f}s]", flush=True)

    # Parity-smooth q WITHIN each game before differencing (the field has no temporal-smoothness
    # term, so adjacent plies disagree more than plies two apart; differencing raw values would
    # report that jitter as hazard).
    if args.smooth > 1:
        # Split on ply == 0, not on gid changing: every replay starts at ply 0, whereas human
        # lichess ids and SF pool ids are independent counters and can collide, which would silently
        # smooth across a game boundary.
        bounds = np.flatnonzero(d["ply"] == 0)[1:]
        for key in ("q_human", "q_sf", "q_shared"):
            d[key] = np.concatenate([parity_smooth(seg, args.smooth)
                                     for seg in np.split(d[key], bounds)])
    d["h"] = d["q_sf"] - d["q_human"]

    keep = d["ply"] >= args.min_ply
    h, qh, qs = d["h"][keep], d["q_human"][keep], d["q_sf"][keep]
    ply, src = d["ply"][keep], d["source"][keep]
    hum, sf = src == 0, src == 1
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- printed verdict ----------
    print(f"\nHAZARD h = q_SF - q_human  (ply >= {args.min_ply}; + = humans give it away, "
          f"- = humans do better than perfect play would)")
    print(f"  {'population':>12s} {'n':>9s} {'median h':>9s} {'mean|h|':>8s} "
          f"{'h>+0.25':>8s} {'h<-0.25':>8s}")
    for name, m in [("human", hum), ("SF-vs-SF", sf)]:
        print(f"  {name:>12s} {int(m.sum()):>9,} {np.median(h[m]):>+9.3f} "
              f"{np.abs(h[m]).mean():>8.3f} {100*(h[m] > 0.25).mean():>7.1f}% "
              f"{100*(h[m] < -0.25).mean():>7.1f}%")
    print(f"\n  {'ply band':>10s} {'human med h':>12s} {'SF med h':>10s} {'human mean|h|':>14s}")
    for lo, hi in [(8, 20), (20, 40), (40, 60), (60, 90), (90, args.max_ply)]:
        b = (ply >= lo) & (ply < hi)
        if (b & hum).sum() < args.min_count:
            continue
        print(f"  {lo:>4d}-{hi:<5d} {np.median(h[b & hum]):>+12.3f} "
              f"{(np.median(h[b & sf]) if (b & sf).sum() >= args.min_count else float('nan')):>+10.3f} "
              f"{np.abs(h[b & hum]).mean():>14.3f}")

    # ---- Figure 1: the off-diagonal plot ------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    e = np.linspace(-1, 1, 61)
    for ax, (name, m, c) in zip(axes[:2], [("human positions", hum, COLOR_HUMAN),
                                           ("SF-vs-SF positions", sf, COLOR_SF)]):
        H, _, _ = np.histogram2d(qs[m], qh[m], bins=[e, e])
        ax.pcolormesh(e, e, np.log1p(H).T, cmap=CMAP_DENSITY, shading="flat")
        ax.plot([-1, 1], [-1, 1], "-", color=INK, lw=1.2, alpha=0.8)
        ax.set_xlabel("q under SF dynamics"); ax.set_aspect("equal")
        ax.set_title(f"{name}   n={int(m.sum()):,}", color=c)
        ax.text(0.62, -0.9, "humans\ngive it away", fontsize=8, color=MUTED, ha="center")
        ax.text(-0.62, 0.9, "humans do\nbetter", fontsize=8, color=MUTED, ha="center")
    axes[0].set_ylabel("q under human dynamics")
    ax = axes[2]
    bins = np.linspace(-1, 1, 81)
    for name, m, c in [("human", hum, COLOR_HUMAN), ("SF-vs-SF", sf, COLOR_SF)]:
        ax.hist(h[m], bins=bins, histtype="step", lw=2, color=c, density=True, label=name)
    ax.axvline(0, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("h = q_SF - q_human"); ax.set_ylabel("density")
    ax.set_title("where each population sits", color=INK)
    ax.legend(fontsize=9, frameon=False)
    fig.suptitle("Two dynamics, two committors: distance off the diagonal IS the hazard\n"
                 "no embedding, no symmetrisation -- both axes are probabilities")
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}_offdiag.png", dpi=140)

    # ---- Figure 2: hazard on the tent ---------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True, sharex=True)
    xb = np.linspace(-1, 1, 41); yb = np.arange(args.min_ply, args.max_ply + 1, 4)
    vmax = 0.0
    grids = {}
    for name, m in [("human", hum), ("SF-vs-SF", sf)]:
        g, n = gated_cells(qh[m], ply[m], h[m], xb, yb, args.min_count)
        grids[name] = (g, n)
        if np.isfinite(g).any():
            vmax = max(vmax, float(np.nanmax(np.abs(g))))
    vmax = max(vmax, 1e-3)
    for ax, (name, (g, n)) in zip(axes2, grids.items()):
        pc = ax.pcolormesh(xb, yb, g.T, cmap=CMAP_HAZARD, vmin=-vmax, vmax=vmax, shading="flat")
        ax.axvline(0, color=MUTED, lw=0.7, ls=":")
        ax.set_xlabel("q under human dynamics")
        ax.set_title(f"{name}   ({int((n >= args.min_count).sum())} of {n.size} cells supported)",
                     color=INK)
    axes2[0].set_ylabel("ply  (start at the top)")
    axes2[0].set_ylim(args.max_ply, args.min_ply)
    fig2.colorbar(pc, ax=axes2, label="mean h = q_SF - q_human  (red = hazard)", shrink=0.85)
    fig2.suptitle("Hazard on the tent -- blank cells are UNSUPPORTED "
                  f"(< {args.min_count} positions), not zero")
    fig2.savefig(f"{args.out_prefix}_tent.png", dpi=140, bbox_inches="tight")

    np.savez(f"{args.out_prefix}_data.npz", **{k: v for k, v in d.items()},
             min_ply=args.min_ply, smooth=args.smooth)
    print(f"\nwrote {args.out_prefix}_{{offdiag,tent}}.png + _data.npz "
          f"({len(d['ply']):,} rows) [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
