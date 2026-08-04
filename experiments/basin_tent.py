#!/usr/bin/env python
"""experiments/basin_tent.py -- Kaveh 2026-08-03: the TENT view of the W/D/L basins.

The picture: the apex at the TOP is the starting position, where nothing is decided. As plies
accumulate, points descend. The LEFT side is a White win, the RIGHT side is a Black win, and a
trajectory that comes to rest before reaching either side is a DRAW. The tent shape is not a
frame drawn around the data -- it emerges, because every game starts at x=0 and can only fan
outward as the game resolves.

  x = P(white wins) - P(black wins)   in [-1, +1]
  y = ply, increasing DOWNWARD from the apex

CRITICAL detail: the three poles are MOVER-POV, and the mover alternates every single ply. Plotted
raw, one game would zigzag left-right-left-right forever and the tent would be pure noise. Every
probability here is converted to WHITE-POV first (stm_white = ply % 2 == 1, the generator's own
convention), which is what makes "left = White, right = Black" hold along a trajectory.

Draw is NOT a third axis here: it is the CENTRE. |x| ~ 0 means the two win-probabilities cancel,
which is exactly the draw basin, so the same three basins appear as left edge / right edge /
centre column. That is why this works in 2-d without losing the third pole.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.losses import basin_logp, WIN, DRAW, LOSS
from experiments.basin_simplex_chart import load_head, COLOR_WIN, COLOR_DRAW, COLOR_LOSS, INK, MUTED

COLOR_WHITE_WIN, COLOR_BLACK_WIN = "#0ca30c", "#d03b3b"


def white_pov_x(p, ply):
    """(N,3) mover-POV probs + ply -> (N,) signed White advantage in [-1,+1].

    stm_white = (ply % 2 == 1) is the generator's convention (gen_field_data_fullgame.py); the
    2026-08-02 basin work had it backwards once and it was a real, corrected bug."""
    mover_is_white = (ply % 2 == 1)
    p_white = np.where(mover_is_white, p[:, WIN], p[:, LOSS])
    p_black = np.where(mover_is_white, p[:, LOSS], p[:, WIN])
    return p_white - p_black


@torch.no_grad()
def probs_for_rows(net, mm, local_rows, device, batch=8192):
    out = np.empty((len(local_rows), 3), np.float32)
    for i in range(0, len(local_rows), batch):
        sl = slice(i, i + batch)
        x = torch.from_numpy(np.asarray(mm[local_rows[sl]], dtype=np.float32)).to(device)
        out[sl] = basin_logp(net.d_poles(net.phi(x)), net.temperature).exp().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/iqe_poles_both_latest.pt")
    ap.add_argument("--combined", default="data/derived/field_combined_sub600k.npz")
    ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--max-ply", type=int, default=42,
                    help="HARD CAP at 42 by default, and this is not cosmetic. "
                         "gen_field_data_fullgame.py runs --stride 6 --per-game 8, so mid-game "
                         "stride samples cannot reach beyond ply (8-1)*6 = 42. Past ply ~54 the "
                         "dataset is 100%% TAIL rows -- positions 1-2 plies from the end. Plotting "
                         "a ply axis past 42 silently swaps the population from 'mid-game' to "
                         "'game endings' and the apparent fanning-out is that swap, not chess.")
    ap.add_argument("--max-to-end", type=int, default=40, help="axis cap for the endgame funnel")
    ap.add_argument("--n-traj", type=int, default=60, help="sample trajectories drawn per panel")
    ap.add_argument("--ply-stride", type=int, default=6,
                    help="ply rows are binned to THIS period. gen_field_data_fullgame.py samples "
                         "every --stride=6 plies (plus a 4-ply tail), so the ply axis is a COMB; "
                         "any finer binning aliases against it and paints horizontal stripes that "
                         "look like structure and are not.")
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_tent")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, LinearSegmentedColormap

    net = load_head(args.ckpt, args.device)
    z = np.load(args.combined, allow_pickle=True)
    meta = eval(str(z["_meta"][0]))
    mm = np.load(meta["feats"][0], mmap_mode="r")
    split = z["orig_source"] if "orig_source" in z.files else z["source"]
    rng = np.random.default_rng(args.seed)

    data = {}
    for name, s in [("human", 0), ("SF-vs-SF", 1)]:
        idx = np.flatnonzero((split == s) & (z["ply"] <= args.max_ply))
        take = np.sort(rng.choice(idx, min(args.n, len(idx)), replace=False))
        p = probs_for_rows(net, mm, z["local_row"][take], args.device)
        ply = z["ply"][take]
        data[name] = dict(x=white_pov_x(p, ply), ply=ply, p=p, game=z["game"][take],
                          n_to_end=z["n_to_end"][take],
                          y=z["y"][take], term=z["is_terminal"][take],
                          result=z["result"][take])
        print(f"  {name}: n={len(take):,}  mean|x| {np.abs(data[name]['x']).mean():.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: the tent, as CONDITIONAL density P(x | ply) ------------------------------
    # Aliasing, and why row-normalizing is the fix rather than a nicer bin size.
    # gen_field_data_fullgame.py samples `(ply - skip_open) % stride == 0` with stride=6 and the
    # SAME phase in every game, so plies at multiples of 6 hold ~all the mass and the plies
    # between them hold only the 4-ply tail. The JOINT density therefore has hard horizontal
    # ridges that are an artifact of the sampler, not of chess, and no choice of bin size removes
    # them -- a bin either straddles teeth or lands on one.
    # Normalizing each ply row to sum to 1 removes it BY CONSTRUCTION: row counts cancel, and what
    # is left is P(x | ply), which is the quantity of interest anyway ("given we are at ply k,
    # where is the game?"). Rows with too few samples are MASKED rather than shown as noise.
    def conditional_tent(x, ply, nx=61, stride=None, min_row=200):
        stride = stride or args.ply_stride
        edges_y = np.arange(-stride / 2, args.max_ply + stride, stride)   # one comb tooth per row
        edges_x = np.linspace(-1.0, 1.0, nx + 1)
        H, _, _ = np.histogram2d(ply, x, bins=[edges_y, edges_x])
        n = H.sum(1, keepdims=True)
        P = np.divide(H, n, out=np.full_like(H, np.nan), where=n >= min_row)
        return np.ma.masked_invalid(P), edges_x, edges_y, n.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)
    prep = {}
    for name, d in data.items():
        prep[name] = conditional_tent(d["x"], d["ply"])
    # LOG colour scale, shared across panels. Linear was unreadable: the SF-vs-SF draw column
    # holds ~80% of its row's mass while the human tent spreads its rows below ~10% per cell, so
    # a linear scale set by the spike washes the human panel to nothing. Log keeps both legible
    # AND keeps the two panels on ONE scale, which is required for them to be comparable at all.
    # Floor at 1e-3 = 0.1% of a row; cells below that are empty-ish and masked rather than shown.
    norm = LogNorm(vmin=1e-3, vmax=1.0)
    for ax, (name, d) in zip(axes, data.items()):
        P, ex, ey, n = prep[name]
        Pm = np.ma.masked_less(P, 1e-3)
        pc = ax.pcolormesh(ex, ey, Pm, cmap="Blues" if name == "human" else "Reds",
                           norm=norm, shading="flat")
        ax.invert_yaxis()
        ax.axvline(0, color=MUTED, lw=0.7, ls=":")
        ax.set_xlim(-1.02, 1.02); ax.set_ylim(args.max_ply, 0)
        ax.set_xlabel("P(White wins) - P(Black wins)")
        ax.set_title(f"{name}   ({int(np.nansum(n)):,} positions)", color=INK)
        fig.colorbar(pc, ax=ax, shrink=0.75, label="P(x | ply)   [rows sum to 1]")
    axes[0].set_ylabel("ply  (start at the top, game descends)")
    for ax in axes:
        ax.text(-0.99, args.max_ply * 0.04, "White wins", fontsize=9, color=COLOR_WHITE_WIN, ha="left")
        ax.text(0.99, args.max_ply * 0.04, "Black wins", fontsize=9, color=COLOR_BLACK_WIN, ha="right")
        ax.text(0, args.max_ply * 0.04, "draw", fontsize=9, color=MUTED, ha="center")
    fig.suptitle("The tent: P(result lean | ply). Games start undecided at the apex and fan "
                 "toward a result as plies accumulate\n(each row normalized, so the sampler's "
                 "6-ply comb cannot appear as banding)")
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}_1_tent.png", dpi=140)

    # ---- Figure 2: trajectories descending the tent -----------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)
    for ax, (name, d) in zip(axes2, data.items()):
        P, ex, ey, _ = prep[name]
        ax.pcolormesh(ex, ey, np.ma.masked_less(P, 1e-3), cmap="Greys",
                      norm=LogNorm(vmin=1e-3, vmax=1.0), alpha=0.35, shading="flat")
        games, cnt = np.unique(d["game"], return_counts=True)
        pick = games[cnt >= 6]
        pick = rng.choice(pick, min(args.n_traj, len(pick)), replace=False)
        for g in pick:
            m = np.flatnonzero(d["game"] == g)
            m = m[np.argsort(d["ply"][m])]
            res = int(d["result"][m[0]])                    # white-POV +1/0/-1
            c = COLOR_WHITE_WIN if res == 1 else (COLOR_BLACK_WIN if res == -1 else COLOR_DRAW)
            ax.plot(d["x"][m], d["ply"][m], "-", color=c, lw=0.9, alpha=0.55)
            ax.plot(d["x"][m[-1]], d["ply"][m[-1]], "o", color=c, ms=3.5, alpha=0.9)
        ax.invert_yaxis(); ax.set_xlim(-1.05, 1.05); ax.set_ylim(args.max_ply, 0)
        ax.axvline(0, color=MUTED, lw=0.7, ls=":")
        ax.set_xlabel("P(White wins) - P(Black wins)"); ax.set_title(name, color=INK)
    axes2[0].set_ylabel("ply")
    for c, lab in [(COLOR_WHITE_WIN, "White won"), (COLOR_DRAW, "drawn"), (COLOR_BLACK_WIN, "Black won")]:
        axes2[1].plot([], [], "-", color=c, label=lab)
    axes2[1].legend(fontsize=8, frameon=False, loc="lower right")
    fig2.suptitle("Individual games descending the tent -- dots mark the final position\n"
                  "a game that comes to rest in the middle is a draw")
    fig2.tight_layout(); fig2.savefig(f"{args.out_prefix}_2_trajectories.png", dpi=140)

    # ---- Figure 3: how wide is the tent at each ply? ----------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 5.4))
    bins = np.arange(0, args.max_ply + 1, 4)
    for name, d in data.items():
        c = "#2a78d6" if name == "human" else "#e34948"
        mid = 0.5 * (bins[:-1] + bins[1:])
        q = [np.quantile(np.abs(d["x"][(d["ply"] >= a) & (d["ply"] < b)]), [0.5, 0.9])
             if ((d["ply"] >= a) & (d["ply"] < b)).sum() > 30 else [np.nan, np.nan]
             for a, b in zip(bins[:-1], bins[1:])]
        q = np.array(q)
        ax3.plot(mid, q[:, 0], "-", color=c, lw=2, label=f"{name} median |x|")
        ax3.plot(mid, q[:, 1], "--", color=c, lw=1.2, label=f"{name} 90th pct")
    ax3.set_xlabel("ply"); ax3.set_ylabel("|P(White wins) - P(Black wins)|")
    ax3.set_title("Tent width vs ply -- how fast does each population commit to a result?", color=INK)
    ax3.legend(fontsize=8, frameon=False); ax3.set_ylim(0, 1)
    fig3.tight_layout(); fig3.savefig(f"{args.out_prefix}_3_width_vs_ply.png", dpi=140)

    # ---- Figure 4: the ENDGAME FUNNEL -- the honest use of the tail rows -------------------
    # The tail rows cannot support a ply axis, but they support a plies-TO-END axis perfectly:
    # every game contributes its last 4 plies, so coverage near the end is complete and unbiased.
    # Read it the same way as the tent, but time runs toward the END rather than from the start.
    fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, (name, d) in zip(axes4, data.items()):
        n2e = d["n_to_end"]
        m = n2e <= args.max_to_end
        edges_y = np.arange(-0.5, args.max_to_end + 1.5, 1.0)
        edges_x = np.linspace(-1.0, 1.0, 62)
        H, _, _ = np.histogram2d(n2e[m], d["x"][m], bins=[edges_y, edges_x])
        nn = H.sum(1, keepdims=True)
        P = np.divide(H, nn, out=np.full_like(H, np.nan), where=nn >= 200)
        ax.pcolormesh(edges_x, edges_y, np.ma.masked_less(np.ma.masked_invalid(P), 1e-3),
                      cmap="Blues" if name == "human" else "Reds",
                      norm=LogNorm(vmin=1e-3, vmax=1.0), shading="flat")
        ax.invert_yaxis(); ax.set_xlim(-1.02, 1.02)
        ax.axvline(0, color=MUTED, lw=0.7, ls=":")
        ax.set_xlabel("P(White wins) - P(Black wins)"); ax.set_title(name, color=INK)
    axes4[0].set_ylabel("plies REMAINING (0 = final position, at the bottom)")
    fig4.suptitle("The endgame funnel: the same view but indexed from the END of the game\n"
                  "(tail rows cover this completely; they cannot support a ply axis)")
    fig4.tight_layout(); fig4.savefig(f"{args.out_prefix}_4_endgame_funnel.png", dpi=140)

    print("\nENDGAME FUNNEL (median |x|) by plies remaining")
    print(f"  {'to end':>8s} {'human':>9s} {'SF-vs-SF':>9s}")
    for a, b in [(0, 1), (1, 2), (2, 4), (4, 10), (10, 25)]:
        row = []
        for name, d in data.items():
            m = (d["n_to_end"] >= a) & (d["n_to_end"] < b)
            row.append(np.median(np.abs(d["x"][m])) if m.sum() > 30 else np.nan)
        print(f"  {a:>3d}-{b:<4d} {row[0]:>9.3f} {row[1]:>9.3f}")

    print("\nTENT WIDTH (median |x|) by ply band")
    print(f"  {'ply':>10s} {'human':>9s} {'SF-vs-SF':>9s}")
    for a, b in [(0, 8), (8, 16), (16, 24), (24, 32), (32, args.max_ply)]:
        row = []
        for name, d in data.items():
            m = (d["ply"] >= a) & (d["ply"] < b)
            row.append(np.median(np.abs(d["x"][m])) if m.sum() > 30 else np.nan)
        print(f"  {a:>4d}-{b:<5d} {row[0]:>9.3f} {row[1]:>9.3f}")
    np.savez(f"{args.out_prefix}_data.npz",
             **{f"{n}_{k}": v for n, d in data.items() for k, v in d.items()})
    print(f"wrote {args.out_prefix}_{{1_tent,2_trajectories,3_width_vs_ply}}.png [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
