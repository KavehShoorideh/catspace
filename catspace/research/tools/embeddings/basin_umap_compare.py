#!/usr/bin/env python
"""catspace/research/tools/embeddings/basin_umap_compare.py -- Kaveh's ask (2026-08-02): embed both the
human-Lichess field data and the SF-vs-SF opening-pool data through the SAME
(original, human-trained) IQE field, UMAP them jointly so they share one
coordinate system, and visualize where the two datasets' BASINS (colored by
"destiny" -- the position's eventual game outcome, mover POV, not any immediate
feature) agree vs. disagree. Kaveh's framing: disagreement = human-only basin-
crossing transitions = the exploitable leak (M0's central finding, visualized
directly rather than just measured statistically).

"b" not "f": positions are labeled by where the game EVENTUALLY ends up
(win/draw/loss for whoever was to move there), not by any forward/immediate
feature -- "we want to look at the basins from the point of view of whether or
not they reach ... what endgame state they reach."

Three main figures + one bonus trajectory overlay:
  1. Human data, UMAP'd, colored by destiny.
  2. SF-vs-SF data, UMAP'd (same projection), colored by destiny.
  3. Density-DIFFERENCE map on a shared grid: where human density exceeds SF
     density (and vice versa) -- a diverging map, not two separate density plots,
     so the "XOR" Kaveh asked for is literally the figure, not something the
     reader has to compute by eye across two panels.
  4. Bonus: a sample of real game trajectories (line paths through the same UMAP
     space) from both datasets, so basin-crossing paths -- present in human
     games, essentially absent in near-perfect SF-vs-SF play -- are visible
     directly, not just implied by the density difference.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from catspace.io import paths


# Colors from the dataviz skill's validated reference palette (references/palette.md):
# status good/critical for win/loss, a neutral gray (not a status color) for draw,
# and the documented blue<->red diverging pair with its gray midpoint.
COLOR_WIN = "#0ca30c"
COLOR_DRAW = "#8a8985"
COLOR_LOSS = "#d03b3b"
DIVERGE_BLUE = "#2a78d6"     # human-only-heavy
DIVERGE_GRAY = "#f0efec"     # equal coverage
DIVERGE_RED = "#e34948"      # SF-only-heavy


def select_n_sf_games_full(moves_tsv, n, seed, rng_offset=0):
    """n random SF-vs-SF games (any outcome), full move lists."""
    rng = np.random.default_rng(seed + 400 + rng_offset)
    lines = []
    with open(moves_tsv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            lines.append((int(parts[0]), parts[2].split()))
    idx = rng.choice(len(lines), min(n, len(lines)), replace=False)
    return [lines[i] for i in idx]


def select_n_human_games_full(records_dir, field_data_path, n, seed, rng_offset=0,
                               games_per_shard=200000):
    """n random human games (any outcome), full move lists via the original
    Lichess parquet shards (field_data_path's own npz only has a per-game
    subsample -- see select_human_game_full's docstring)."""
    import pyarrow.parquet as pq
    d = np.load(field_data_path, allow_pickle=True)
    uniq_games = np.unique(d["game"])
    rng = np.random.default_rng(seed + 400 + rng_offset)
    picked = rng.choice(uniq_games, min(n, len(uniq_games)), replace=False)
    by_shard = {}
    for gid in picked:
        by_shard.setdefault(int(gid) // games_per_shard, []).append(int(gid))
    out = []
    for shard_idx, gids in by_shard.items():
        shard_path = f"{records_dir}/records_{shard_idx:05d}.parquet"
        t = pq.read_table(shard_path, columns=["game_id", "moves"])
        gid_arr = t["game_id"].to_numpy()
        for gid in gids:
            hits = np.flatnonzero(gid_arr == gid)
            if len(hits) == 0:
                continue
            moves = t["moves"][int(hits[0])].as_py().split()
            out.append((gid, moves))
    return out


def build_traj_source(games_moves_list):
    """[(gid, moves_uci), ...] -> concatenated full-replay planes/game/ply,
    for the density-overlay trajectory figures."""
    planes_list, game_list, ply_list = [], [], []
    for gid, moves in games_moves_list:
        p = replay_full_game(moves)
        planes_list.append(p)
        game_list.append(np.full(len(p), gid))
        ply_list.append(np.arange(len(p)))
    return dict(planes=np.concatenate(planes_list, 0), game=np.concatenate(game_list),
                ply=np.concatenate(ply_list))


def replay_full_game(moves_uci):
    """Replay every ply of a real game via LczeroBoard (correct 8-position
    history at each step, matching how the training data itself was generated)
    -> (n_ply+1, 112, 8, 8) uint8 planes, one per position INCLUDING the
    starting position before move 1."""
    from lczerolens import LczeroBoard
    import chess as _chess
    board = LczeroBoard()
    planes = [board.to_input_tensor().numpy()]
    for uci in moves_uci:
        board.push(_chess.Move.from_uci(uci))
        planes.append(board.to_input_tensor().numpy())
    return np.stack(planes)


def select_sf_game_full(moves_tsv, result_value, seed, rng_offset=0):
    """Full move-by-move SF-vs-SF game (not the subsampled derived npz) --
    the STARTING position's outcome (white-POV) drives selection, since ply=0
    is always White to move."""
    rng = np.random.default_rng(seed + 300 + rng_offset)
    candidates = []
    with open(moves_tsv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gid, res, moves = parts[0], int(parts[1]), parts[2].split()
            if res == result_value:
                candidates.append((int(gid), moves))
    gid, moves = candidates[rng.integers(len(candidates))]
    planes = replay_full_game(moves)
    return dict(planes=planes, ply=np.arange(len(planes)), game=gid, result=int(result_value))


def select_human_game_full(records_dir, field_data_path, result_value, seed, rng_offset=0,
                            games_per_shard=200000):
    """Full move-by-move human game -- field_data_path (field_std_v1.npz) only
    stores a per-game SUBSAMPLE (~12 positions/game), so this looks up the
    game's full move list from the original Lichess parquet shard instead."""
    import pyarrow.parquet as pq
    d = np.load(field_data_path, allow_pickle=True)
    rows = np.flatnonzero(d["result"] == result_value)
    rng = np.random.default_rng(seed + 300 + rng_offset)
    row = int(rows[rng.integers(len(rows))])
    gid = int(d["game"][row])
    shard_idx = gid // games_per_shard
    shard_path = f"{records_dir}/records_{shard_idx:05d}.parquet"
    t = pq.read_table(shard_path, columns=["game_id", "moves"])
    gids = t["game_id"].to_numpy()
    idx = int(np.flatnonzero(gids == gid)[0])
    moves = t["moves"][idx].as_py().split()
    planes = replay_full_game(moves)
    return dict(planes=planes, ply=np.arange(len(planes)), game=gid, result=int(result_value))


def load_sample(path, n, seed, rng_offset=0):
    d = np.load(path, allow_pickle=True)
    N = len(d["planes"])
    rng = np.random.default_rng(seed + rng_offset)
    idx = rng.choice(N, min(n, N), replace=False)
    idx.sort()   # sequential reads friendlier on the memmap-backed npz
    planes = d["planes"][idx]
    result = d["result"][idx]           # white-POV +1/0/-1
    ply = d["ply"][idx]
    game = d["game"][idx]
    # side-to-move convention per gen_field_data_fullgame.py: `stm_white = (ply % 2
    # == 1)` -- White to move on ODD ply. (Caught a real bug here: this script
    # originally had it backwards, ply%2==0. Since flipping parity for every row
    # uniformly swaps the win/loss LABELS but not their counts, that bug alone
    # wouldn't explain a real imbalance -- see the mover_is_white note below for
    # what does.)
    mover_is_white = (ply % 2 == 1)
    destiny = np.where(result == 0, 0, np.where(
        (result == 1) == mover_is_white, 1, -1))   # mover POV: +1 win, 0 draw, -1 loss
    return dict(planes=planes, destiny=destiny, ply=ply, game=game, idx=idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default=paths.derived("field_std_v1.npz"))
    ap.add_argument("--sf", default=paths.derived("opening_pool_sfsf.npz"))
    ap.add_argument("--sf-moves", default=paths.derived("opening_pool_sfsf_moves.tsv"))
    ap.add_argument("--human-records-dir", default=paths.records("lichess_2019-01"))
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--n-trajectories", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default=paths.experiment("basin_umap"))
    args = ap.parse_args()
    t0 = time.time()

    print("loading + sampling both datasets ...", flush=True)
    human = load_sample(args.human, args.n, args.seed, 0)
    sf = load_sample(args.sf, args.n, args.seed, 1)
    print(f"  human n={len(human['planes'])} | sf n={len(sf['planes'])} [{time.time()-t0:.0f}s]", flush=True)

    print("computing phi via the ORIGINAL (human-trained) field, for BOTH datasets ...", flush=True)
    from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField
    field = ReachabilityField()   # defaults: t1-256x10 trunk + field_iqe_t1_final.pt (human-trained)
    phi_human = field.phi_from_planes(list(human["planes"].astype(np.float32))).cpu().numpy()
    phi_sf = field.phi_from_planes(list(sf["planes"].astype(np.float32))).cpu().numpy()
    print(f"  phi_human {phi_human.shape} phi_sf {phi_sf.shape} [{time.time()-t0:.0f}s]", flush=True)

    print("fitting UMAP jointly (shared coordinate system for a valid overlay) ...", flush=True)
    import umap
    combined = np.concatenate([phi_human, phi_sf], axis=0)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=args.seed)
    emb2d = reducer.fit_transform(combined)
    human2d = emb2d[:len(phi_human)]
    sf2d = emb2d[len(phi_human):]
    print(f"  done [{time.time()-t0:.0f}s]", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    xlim = (min(human2d[:, 0].min(), sf2d[:, 0].min()), max(human2d[:, 0].max(), sf2d[:, 0].max()))
    ylim = (min(human2d[:, 1].min(), sf2d[:, 1].min()), max(human2d[:, 1].max(), sf2d[:, 1].max()))

    def scatter_by_destiny(ax, pts, destiny, title, seed=0):
        # plot in RANDOMIZED order, not class-by-class -- plotting one class last
        # (e.g. "loses" always painted over "wins"/"draw" in every dense region)
        # visually exaggerates that class regardless of its true relative density.
        # Caught by inspection: this was drawing win/draw/loss sequentially, so
        # "loses" always sat on top -- fixed here, not just in the destiny label.
        color_map = {1: COLOR_WIN, 0: COLOR_DRAW, -1: COLOR_LOSS}
        colors = np.array([color_map[d] for d in destiny])
        order = np.random.default_rng(seed).permutation(len(pts))
        ax.scatter(pts[order, 0], pts[order, 1], s=3, c=colors[order], alpha=0.5, linewidths=0)
        for val, color, label in [(1, COLOR_WIN, "mover wins"), (0, COLOR_DRAW, "draw"), (-1, COLOR_LOSS, "mover loses")]:
            ax.scatter([], [], s=12, c=color, label=label)   # legend-only proxies, don't affect z-order
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

    # ---- Figures 1 & 2: human and SF, colored by destiny ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    scatter_by_destiny(axes[0], human2d, human["destiny"], f"Human Lichess games (n={len(human2d):,})")
    scatter_by_destiny(axes[1], sf2d, sf["destiny"], f"Stockfish-vs-Stockfish (n={len(sf2d):,})")
    axes[1].legend(loc="upper right", fontsize=8, markerscale=3, frameon=False)
    fig.suptitle("Reachability-field UMAP, colored by eventual outcome (mover POV) -- same field, same projection")
    fig.tight_layout()
    Path(str(paths.experiments_dir())).mkdir(exist_ok=True, parents=True)
    Path(str(paths.figures_dir())).mkdir(exist_ok=True, parents=True)
    fig.savefig(f"{args.out_prefix}_1_2_side_by_side.png", dpi=140)
    fig.savefig(paths.figure("basin_umap_1_2_side_by_side.png"), dpi=140)

    # ---- Figure 3: density DIFFERENCE (the "XOR") on a shared grid ----
    from scipy.ndimage import gaussian_filter
    bins = 80
    h_hist, xedges, yedges = np.histogram2d(human2d[:, 0], human2d[:, 1], bins=bins, range=[xlim, ylim])
    s_hist, _, _ = np.histogram2d(sf2d[:, 0], sf2d[:, 1], bins=bins, range=[xlim, ylim])
    # light smoothing -- raw per-cell counts are noisy at any sample size finite
    # enough to run interactively; this is a display smoother, not a claim that
    # any single cell's exact value is meaningful.
    h_smooth = gaussian_filter(h_hist, sigma=1.2)
    s_smooth = gaussian_filter(s_hist, sigma=1.2)
    h_norm = h_smooth / h_smooth.sum()
    s_norm = s_smooth / s_smooth.sum()
    diff = h_norm - s_norm
    vmax = np.abs(diff).max()

    diverge_cmap = LinearSegmentedColormap.from_list(
        "human_sf_diverge", [DIVERGE_BLUE, DIVERGE_GRAY, DIVERGE_RED], N=256)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    im = ax2.imshow(diff.T, origin="lower", extent=[*xlim, *ylim], cmap=diverge_cmap,
                     vmin=-vmax, vmax=vmax, aspect="auto")
    cb = fig2.colorbar(im, ax=ax2, shrink=0.8)
    cb.set_label("density(human) - density(SF-vs-SF), normalized")
    ax2.set_title("Where the two datasets' basins DISAGREE\nblue = human-only coverage | red = SF-vs-SF-only coverage")
    ax2.set_xticks([]); ax2.set_yticks([])
    fig2.tight_layout()
    fig2.savefig(f"{args.out_prefix}_3_difference.png", dpi=140)
    fig2.savefig(paths.figure("basin_umap_3_difference.png"), dpi=140)

    # ---- Bonus figure 4: sample FULL game trajectories overlaid on the difference
    # map -- a random position sample rarely has >=3 points from the same game,
    # so these are separately sampled whole games, phi'd, and projected through
    # the ALREADY-FITTED UMAP reducer (out-of-sample .transform(), approximate
    # but fine for illustration) rather than drawn from the density sample.
    print("sampling whole games for the trajectory overlay ...", flush=True)
    human_traj_src = build_traj_source(
        select_n_human_games_full(args.human_records_dir, args.human, args.n_trajectories, args.seed, 0))
    sf_traj_src = build_traj_source(
        select_n_sf_games_full(args.sf_moves, args.n_trajectories, args.seed, 1))
    phi_human_traj = field.phi_from_planes(
        list(human_traj_src["planes"].astype(np.float32))).cpu().numpy()
    phi_sf_traj = field.phi_from_planes(
        list(sf_traj_src["planes"].astype(np.float32))).cpu().numpy()
    human_traj_2d = reducer.transform(phi_human_traj)
    sf_traj_2d = reducer.transform(phi_sf_traj)

    fig4, ax4 = plt.subplots(figsize=(7.5, 6.5))
    ax4.imshow(diff.T, origin="lower", extent=[*xlim, *ylim], cmap=diverge_cmap,
               vmin=-vmax, vmax=vmax, aspect="auto", alpha=0.55)

    # Openings are dropped from the multi-game overlay (fig 4/5 only): sklearn's
    # UMAP fit warned "Graph is not fully connected" -- with only ~15k background
    # points, true opening positions are so sparse they land as disconnected
    # outliers far from the main manifold, so every full-replay trajectory's first
    # few plies is one giant jump from that periphery into the real basin
    # structure. With >1 trajectory overlaid this turns into a starburst that
    # buries the thing being measured (does the MIDGAME/ENDGAME path stay in its
    # basin). Single-game figures 6/7 keep the opening (that's the point there).
    SKIP_OPENING_PLY = 10

    def plot_trajectories(pts2d, games, ply, color, label):
        first = True
        for g in np.unique(games):
            m = games == g
            if m.sum() < 3:
                continue
            order = np.argsort(ply[m])
            path = pts2d[m][order][SKIP_OPENING_PLY:]
            if len(path) < 2:
                continue
            ax4.plot(path[:, 0], path[:, 1], "-", color=color, alpha=0.6, lw=0.9,
                      label=(label if first else None))
            first = False

    plot_trajectories(human_traj_2d, human_traj_src["game"], human_traj_src["ply"],
                       DIVERGE_BLUE, "human game trajectory")
    plot_trajectories(sf_traj_2d, sf_traj_src["game"], sf_traj_src["ply"],
                       DIVERGE_RED, "SF-vs-SF game trajectory")
    ax4.legend(loc="upper right", fontsize=8, frameon=False)
    ax4.set_title(f"Sample game trajectories over the difference map (n={args.n_trajectories}/dataset)")
    ax4.set_xticks([]); ax4.set_yticks([])
    fig4.tight_layout()
    fig4.savefig(f"{args.out_prefix}_4_trajectories.png", dpi=140)
    fig4.savefig(paths.figure("basin_umap_4_trajectories.png"), dpi=140)

    # ---- Figure 5: trajectories over the GROUND-TRUTH basin regions (Kaveh's
    # follow-up ask) -- background = majority-vote win/draw/loss per spatial bin,
    # built from the SF-vs-SF destiny distribution specifically (near-perfect play
    # is the closest thing this project has to the TRUE basin structure; using it
    # as the reference means "does a trajectory stay inside its basin" has an
    # actual ground truth to be measured against, not just "does it match some
    # blended average"). Both human and SF trajectories are overlaid on the SAME
    # SF-derived background, so a human path crossing from a green region into a
    # red one is a visible basin leak, not a hypothesis.
    print("building the SF-derived basin-region background ...", flush=True)
    # Continuous (win_fraction - loss_fraction) per bin, NOT a majority vote --
    # SF-vs-SF is 76% draws overall, so a 3-way argmax vote gets swamped to
    # "draw" almost everywhere and win/loss basins nearly vanish (this was
    # tried first and looked wrong -- caught by inspection, not assumed).
    # A continuous diverging gradient (a standard eval-heatmap convention)
    # shows win/loss LEAN even in bins where draw is still the plurality.
    min_count = 5
    h_win, _, _ = np.histogram2d(sf2d[sf["destiny"] == 1, 0], sf2d[sf["destiny"] == 1, 1], bins=bins, range=[xlim, ylim])
    h_draw, _, _ = np.histogram2d(sf2d[sf["destiny"] == 0, 0], sf2d[sf["destiny"] == 0, 1], bins=bins, range=[xlim, ylim])
    h_loss, _, _ = np.histogram2d(sf2d[sf["destiny"] == -1, 0], sf2d[sf["destiny"] == -1, 1], bins=bins, range=[xlim, ylim])
    h_win = gaussian_filter(h_win, sigma=1.2)
    h_draw = gaussian_filter(h_draw, sigma=1.2)
    h_loss = gaussian_filter(h_loss, sigma=1.2)
    total = h_win + h_draw + h_loss
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(total > 0, (h_win - h_loss) / np.maximum(total, 1e-9), 0.0)   # [-1, +1]
    basin_cmap = LinearSegmentedColormap.from_list(
        "basin_value", [COLOR_LOSS, COLOR_DRAW, COLOR_WIN], N=256)
    bg = basin_cmap((value + 1) / 2)[..., :3]               # (bins,bins,3)
    alpha_bg = np.clip(total / max(1, np.percentile(total[total > 0], 60)), 0, 1) * 0.6
    alpha_bg[total < min_count] = 0.0
    bg_rgba = np.concatenate([bg, alpha_bg[..., None]], axis=-1)

    fig5, ax5 = plt.subplots(figsize=(7.5, 6.5))
    ax5.imshow(bg_rgba.transpose(1, 0, 2), origin="lower", extent=[*xlim, *ylim], aspect="auto")
    def plot_trajectories_on(ax, pts2d, games, ply, color, label):
        first = True
        for g in np.unique(games):
            m = games == g
            if m.sum() < 3:
                continue
            order = np.argsort(ply[m])
            path = pts2d[m][order][SKIP_OPENING_PLY:]
            if len(path) < 2:
                continue
            ax.plot(path[:, 0], path[:, 1], "-", color=color, alpha=0.85, lw=1.1,
                    label=(label if first else None))
            first = False
    TRAJ_HUMAN = "#2a78d6"    # categorical slot 1 (blue) -- distinct from win/draw/loss fills
    TRAJ_SF = "#4a3aa7"       # categorical slot 7 (violet) -- distinct from blue and from the fills
    plot_trajectories_on(ax5, human_traj_2d, human_traj_src["game"], human_traj_src["ply"],
                          TRAJ_HUMAN, "human game trajectory")
    plot_trajectories_on(ax5, sf_traj_2d, sf_traj_src["game"], sf_traj_src["ply"],
                          TRAJ_SF, "SF-vs-SF game trajectory")
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    legend_elems = [
        Line2D([0], [0], color=TRAJ_HUMAN, lw=1.5, label="human game trajectory"),
        Line2D([0], [0], color=TRAJ_SF, lw=1.5, label="SF-vs-SF game trajectory"),
    ]
    ax5.legend(handles=legend_elems, loc="upper right", fontsize=8, frameon=True, facecolor="white")
    sm = ScalarMappable(norm=Normalize(-1, 1), cmap=basin_cmap)
    cb5 = fig5.colorbar(sm, ax=ax5, shrink=0.8)
    cb5.set_label("SF-derived mover win-fraction minus loss-fraction (per bin)")
    ax5.set_title("Game trajectories over SF-derived (ground-truth) basin regions\n"
                   "does the path stay inside its basin, or leak across?")
    ax5.set_xticks([]); ax5.set_yticks([])
    fig5.tight_layout()
    fig5.savefig(f"{args.out_prefix}_5_trajectories_on_basins.png", dpi=140)
    fig5.savefig(paths.figure("basin_umap_5_trajectories_on_basins.png"), dpi=140)

    # ---- Figures 6 & 7: Kaveh's follow-up -- one lost/drawn/won game each from
    # SF-vs-SF and from human play (same GAMES aren't available across the two
    # pools, as Kaveh expected -- matched by outcome category instead), tracking
    # each game's basin value across every ply: does Stockfish stay in its
    # basin, and do humans leak across theirs?
    print("selecting 3 SF + 3 human single games (won/drawn/lost) ...", flush=True)
    outcome_names = {1: "White wins", 0: "draw", -1: "Black wins"}
    sf_games = {r: select_sf_game_full(args.sf_moves, r, args.seed, 0) for r in (1, 0, -1)}
    human_games = {r: select_human_game_full(args.human_records_dir, args.human, r, args.seed, 1)
                    for r in (1, 0, -1)}

    def grid_value_at(pts, grid, xlim, ylim, nbins):
        xi = np.clip(((pts[:, 0] - xlim[0]) / (xlim[1] - xlim[0]) * nbins).astype(int), 0, nbins - 1)
        yi = np.clip(((pts[:, 1] - ylim[0]) / (ylim[1] - ylim[0]) * nbins).astype(int), 0, nbins - 1)
        return grid[xi, yi]

    def project_and_value(game_dict):
        phi = field.phi_from_planes(list(game_dict["planes"].astype(np.float32))).cpu().numpy()
        pts2d = reducer.transform(phi)
        val = grid_value_at(pts2d, value, xlim, ylim, bins)
        return pts2d, val

    for r in (1, 0, -1):
        sf_games[r]["pts2d"], sf_games[r]["value"] = project_and_value(sf_games[r])
        human_games[r]["pts2d"], human_games[r]["value"] = project_and_value(human_games[r])

    # Figure 6: 2x3 small multiples, one panel per game, over the same basin background.
    fig6, axes6 = plt.subplots(2, 3, figsize=(15, 9.5))
    cols = [1, 0, -1]
    for c, r in enumerate(cols):
        for row_i, (games_dict, src_label, traj_color) in enumerate([
                (sf_games, "SF-vs-SF", TRAJ_SF), (human_games, "Human", TRAJ_HUMAN)]):
            ax = axes6[row_i, c]
            ax.imshow(bg_rgba.transpose(1, 0, 2), origin="lower", extent=[*xlim, *ylim], aspect="auto")
            g = games_dict[r]
            ax.plot(g["pts2d"][:, 0], g["pts2d"][:, 1], "-", color=traj_color, lw=1.4, alpha=0.9)
            ax.plot(g["pts2d"][0, 0], g["pts2d"][0, 1], "o", color=traj_color, ms=7, mec="black", mew=0.8)   # start
            ax.plot(g["pts2d"][-1, 0], g["pts2d"][-1, 1], "s", color=traj_color, ms=7, mec="black", mew=0.8)  # end
            ax.set_title(f"{src_label}: starting position -> {outcome_names[r]}\n"
                          f"(n_ply={len(g['ply'])}, game={g['game']})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig6.suptitle("One game per outcome, SF-vs-SF (top) vs human (bottom) -- circle=start, square=end")
    fig6.tight_layout()
    fig6.savefig(f"{args.out_prefix}_6_single_games_spatial.png", dpi=140)
    fig6.savefig(paths.figure("basin_umap_6_single_games_spatial.png"), dpi=140)

    # Figure 7: quantitative -- basin value (win_frac - loss_frac at the trajectory's
    # local UMAP position) vs normalized ply, one line per game -- does it hold
    # steady/consistent with the label, or swing/cross zero along the way?
    fig7, ax7 = plt.subplots(figsize=(8, 5.5))
    style_by_outcome = {1: "-", 0: "--", -1: ":"}
    for r in (1, 0, -1):
        g = sf_games[r]
        x = np.linspace(0, 1, len(g["value"]))
        ax7.plot(x, g["value"], style_by_outcome[r], color=TRAJ_SF, lw=1.8,
                  label=f"SF-vs-SF, {outcome_names[r]}")
        g = human_games[r]
        x = np.linspace(0, 1, len(g["value"]))
        ax7.plot(x, g["value"], style_by_outcome[r], color=TRAJ_HUMAN, lw=1.8,
                  label=f"Human, {outcome_names[r]}")
    ax7.axhline(0, color="#8a8985", lw=0.8)
    ax7.set_xlabel("game progress (ply, normalized 0-1)")
    ax7.set_ylabel("local basin value (SF-derived win-fraction minus loss-fraction)")
    ax7.set_ylim(-1.05, 1.05)
    ax7.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax7.set_title("Does each game's trajectory stay in its basin?\n"
                   "solid=White wins, dashed=draw, dotted=Black wins")
    fig7.tight_layout()
    fig7.savefig(f"{args.out_prefix}_7_basin_value_vs_ply.png", dpi=140)
    fig7.savefig(paths.figure("basin_umap_7_basin_value_vs_ply.png"), dpi=140)

    np.savez(f"{args.out_prefix}_data.npz", human2d=human2d, sf2d=sf2d,
             human_destiny=human["destiny"], sf_destiny=sf["destiny"],
             human_game=human["game"], sf_game=sf["game"],
             human_ply=human["ply"], sf_ply=sf["ply"], diff=diff, xlim=xlim, ylim=ylim)
    print(f"wrote {args.out_prefix}_1_2_side_by_side.png, _3_difference.png, "
          f"_4_trajectories.png (+ docs/figures copies) + _data.npz")
    print(f"DONE basin_umap_compare [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
