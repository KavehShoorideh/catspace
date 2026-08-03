#!/usr/bin/env python
"""experiments/basin_metastability.py -- Kaveh's ask (2026-08-02): "do all four
of the suggestions" from the physics/MSM-literature search (see JOURNAL.md):

  1. TICA instead of UMAP for the trajectory embedding -- TICA picks axes that
     maximize time-lagged autocorrelation (i.e. "slow" directions along which
     consecutive plies of the same game stay close), unlike UMAP/t-SNE which
     have no notion of dynamics at all. This is a from-scratch generalized
     eigenvalue solve (`_fit_tica`), NOT a package -- `deeptime` (the standard
     tool) has no wheel for this repo's Python 3.14 build and fails to compile
     from source (checked: no sdist build, no binary wheel on any index).
  2. Isocommittor contour lines on the SF-derived basin-value background
     (value=0 is the literal analog of the committor=0.5 transition-state
     surface).
  3. PCCA+-STYLE macrostate clustering: spectral clustering (KMeans on the
     leading non-trivial eigenvectors of the SF-derived microstate transition
     matrix) grouping microstates into macrostates. This is the "spectral
     clustering" family PCCA+ belongs to, but it is NOT the literal Weber
     inner-simplex PCCA+ algorithm (which needs the full package) -- labeled
     honestly as an approximation throughout.
  4. A disconnectivity-graph-STYLE dendrogram: hierarchical (average-linkage)
     clustering of microstates using a transition-probability-derived distance,
     rendered as a scipy dendrogram. This is NOT the literal Wales-style
     energy-threshold disconnectivity graph (which needs actual local minima +
     saddle points of an energy landscape, a concept chess positions don't
     have) -- it is a practical stand-in that shows the same qualitative thing
     (how basins nest/merge), built from the microstate transition matrix.
  5. An MSM-flux-diagram-STYLE 3-node network (win/draw/loss macrostates,
     edge width = observed macrostate-to-macrostate transition probability),
     built separately for SF-vs-SF and human data on the SAME (SF-derived)
     macrostate partition -- this is the direct, quantitative version of "do
     human trajectories leak between basins more than SF." It is empirical
     transition counting, NOT the full committor-weighted reactive-flux
     decomposition of Transition Path Theory (which requires the committor,
     which requires a converged MSM at a scale this dataset doesn't have).

Reuses field loading / game selection from basin_umap_compare.py rather than
duplicating it (that module's functions are already validated).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.basin_umap_compare import (   # noqa: E402
    COLOR_WIN, COLOR_DRAW, COLOR_LOSS,
    load_sample, select_n_human_games_full, select_n_sf_games_full,
    replay_full_game,
)

TRAJ_HUMAN = "#2a78d6"
TRAJ_SF = "#4a3aa7"


def games_to_phi_list(field, games_moves_list):
    """[(gid, moves_uci), ...] -> list of per-game (n_ply+1, 64) phi arrays,
    kept SEPARATE (not concatenated) so lagged pairs never cross a game
    boundary."""
    out = []
    for gid, moves in games_moves_list:
        planes = replay_full_game(moves)
        phi = field.phi_from_planes(list(planes.astype(np.float32))).cpu().numpy()
        out.append((gid, phi))
    return out


def lagged_pairs(phi_games, lag=1):
    """[(gid, phi (T,64)), ...] -> (X_t, X_t+lag), each (sum(T_g - lag), 64)."""
    xs, ys = [], []
    for _, phi in phi_games:
        if len(phi) <= lag:
            continue
        xs.append(phi[:-lag])
        ys.append(phi[lag:])
    return np.concatenate(xs, 0), np.concatenate(ys, 0)


def fit_tica(X_t, X_lag, n_components=2):
    """Textbook TICA: solve the generalized eigenproblem C_tau_sym v = lam C0 v,
    where C0 is the (mean-centered, pooled) instantaneous covariance and
    C_tau_sym is the symmetrized time-lagged covariance. Eigenvalues are
    autocorrelations in [-1, 1]; the top |n_components| eigenvectors (largest
    eigenvalue = slowest-decorrelating direction) are the TICA axes. Returns
    (mean, components) so any phi can be projected via (phi - mean) @ components.
    """
    mean = np.concatenate([X_t, X_lag], 0).mean(0)
    Xt = X_t - mean
    Xl = X_lag - mean
    n = len(Xt)
    C0 = (Xt.T @ Xt + Xl.T @ Xl) / (2 * n)          # symmetric instantaneous cov
    Ctau = (Xt.T @ Xl + Xl.T @ Xt) / (2 * n)         # symmetrized lagged cov
    C0 += 1e-6 * np.eye(C0.shape[0])                 # numerical floor
    from scipy.linalg import eigh
    eigvals, eigvecs = eigh(Ctau, C0)                # generalized eigenproblem
    order = np.argsort(-eigvals)
    top = order[:n_components]
    print(f"    TICA eigenvalues (autocorrelation, top {n_components}): "
          f"{eigvals[top].round(3).tolist()}")
    return mean, eigvecs[:, top]


def project(phi, mean, components):
    return (phi - mean) @ components


def basin_value_grid(pts2d, destiny, xlim, ylim, bins, sigma=1.2):
    from scipy.ndimage import gaussian_filter
    h_win, xedges, yedges = np.histogram2d(pts2d[destiny == 1, 0], pts2d[destiny == 1, 1], bins=bins, range=[xlim, ylim])
    h_draw, _, _ = np.histogram2d(pts2d[destiny == 0, 0], pts2d[destiny == 0, 1], bins=bins, range=[xlim, ylim])
    h_loss, _, _ = np.histogram2d(pts2d[destiny == -1, 0], pts2d[destiny == -1, 1], bins=bins, range=[xlim, ylim])
    h_win, h_draw, h_loss = (gaussian_filter(h, sigma=sigma) for h in (h_win, h_draw, h_loss))
    total = h_win + h_draw + h_loss
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(total > 0, (h_win - h_loss) / np.maximum(total, 1e-9), 0.0)
    return value, total, xedges, yedges


def main():
    t0 = time.time()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from sklearn.cluster import KMeans

    seed = 0
    n_density = 15000
    n_games = 100     # per source, for fitting TICA + the microstate transition matrix

    print("loading density samples + field ...", flush=True)
    human = load_sample("data/derived/field_std_v1.npz", n_density, seed, 0)
    sf = load_sample("data/derived/opening_pool_sfsf.npz", n_density, seed, 1)
    from catspace.encoder.field import ReachabilityField
    field = ReachabilityField()
    phi_human = field.phi_from_planes(list(human["planes"].astype(np.float32))).cpu().numpy()
    phi_sf = field.phi_from_planes(list(sf["planes"].astype(np.float32))).cpu().numpy()
    print(f"  done [{time.time()-t0:.0f}s]", flush=True)

    print(f"sampling {n_games}+{n_games} full games for TICA + the microstate MSM ...", flush=True)
    human_games = select_n_human_games_full("data/records/lichess_2019-01", "data/derived/field_std_v1.npz", n_games, seed, 10)
    sf_games = select_n_sf_games_full("data/derived/opening_pool_sfsf_moves.tsv", n_games, seed, 11)
    human_phi_games = games_to_phi_list(field, human_games)
    sf_phi_games = games_to_phi_list(field, sf_games)
    print(f"  human games: {len(human_phi_games)}, avg len {np.mean([len(p) for _,p in human_phi_games]):.0f} ply | "
          f"sf games: {len(sf_phi_games)}, avg len {np.mean([len(p) for _,p in sf_phi_games]):.0f} ply [{time.time()-t0:.0f}s]", flush=True)

    # ==== 1. TICA ====================================================
    print("[1/5] fitting TICA (lag=1 ply, pooled human+SF -- shared axes for a valid overlay) ...", flush=True)
    Xt, Xlag = lagged_pairs(human_phi_games + sf_phi_games, lag=1)
    tica_mean, tica_components = fit_tica(Xt, Xlag, n_components=2)

    human2d = project(phi_human, tica_mean, tica_components)
    sf2d = project(phi_sf, tica_mean, tica_components)
    xlim = (min(human2d[:, 0].min(), sf2d[:, 0].min()), max(human2d[:, 0].max(), sf2d[:, 0].max()))
    ylim = (min(human2d[:, 1].min(), sf2d[:, 1].min()), max(human2d[:, 1].max(), sf2d[:, 1].max()))
    bins = 70
    value, total, xedges, yedges = basin_value_grid(sf2d, sf["destiny"], xlim, ylim, bins)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    basin_cmap = LinearSegmentedColormap.from_list("basin_value", [COLOR_LOSS, COLOR_DRAW, COLOR_WIN], N=256)
    alpha_bg = np.clip(total / max(1, np.percentile(total[total > 0], 60)), 0, 1) * 0.6
    alpha_bg[total < 5] = 0.0
    bg = basin_cmap((value + 1) / 2)[..., :3]
    bg_rgba = np.concatenate([bg, alpha_bg[..., None]], axis=-1)

    def scatter_by_destiny(ax, pts, destiny, title, rng_seed=0):
        color_map = {1: COLOR_WIN, 0: COLOR_DRAW, -1: COLOR_LOSS}
        colors = np.array([color_map[d] for d in destiny])
        order = np.random.default_rng(rng_seed).permutation(len(pts))
        ax.scatter(pts[order, 0], pts[order, 1], s=3, c=colors[order], alpha=0.5, linewidths=0)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_title(title)
        ax.set_xlabel("TICA 1 (slowest)"); ax.set_ylabel("TICA 2")

    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5.5))
    scatter_by_destiny(axes1[0], human2d, human["destiny"], f"Human (TICA, n={len(human2d):,})")
    scatter_by_destiny(axes1[1], sf2d, sf["destiny"], f"SF-vs-SF (TICA, n={len(sf2d):,})")
    for val, color, label in [(1, COLOR_WIN, "mover wins"), (0, COLOR_DRAW, "draw"), (-1, COLOR_LOSS, "mover loses")]:
        axes1[1].scatter([], [], s=12, c=color, label=label)
    axes1[1].legend(loc="best", fontsize=8, markerscale=3, frameon=False)
    fig1.suptitle("TICA projection (slow axes from time-lagged autocorrelation), colored by eventual outcome")
    fig1.tight_layout()
    fig1.savefig("artifacts/experiments/meta_1_tica_side_by_side.png", dpi=140)
    Path("docs/figures").mkdir(exist_ok=True, parents=True)
    fig1.savefig("docs/figures/meta_1_tica_side_by_side.png", dpi=140)

    # trajectories on TICA + isocommittor contours (items 1+2 combined)
    print("[2/5] isocommittor-style contours over the TICA basin background ...", flush=True)
    fig2, ax2 = plt.subplots(figsize=(7.5, 6.5))
    ax2.imshow(bg_rgba.transpose(1, 0, 2), origin="lower", extent=[*xlim, *ylim], aspect="auto")
    Xg, Yg = np.meshgrid(xcenters, ycenters, indexing="ij")
    valid = total.T > 5
    Vc = np.where(valid, value.T, np.nan)
    cs = ax2.contour(Xg.T, Yg.T, Vc, levels=[-0.3, 0.0, 0.3], colors=["#7a1f1f", "#333333", "#1f5e1f"],
                      linewidths=1.3, linestyles=["--", "-", "--"])
    ax2.clabel(cs, fmt={-0.3: "loss-leaning", 0.0: "isocommittor (value=0)", 0.3: "win-leaning"}, fontsize=7)

    def plot_traj(ax, phi_games, color, label, skip=10):
        first = True
        for gid, phi in phi_games:
            if len(phi) < skip + 3:
                continue
            pts = project(phi, tica_mean, tica_components)[skip:]
            ax.plot(pts[:, 0], pts[:, 1], "-", color=color, alpha=0.5, lw=0.9, label=(label if first else None))
            first = False

    plot_traj(ax2, human_phi_games[:25], TRAJ_HUMAN, "human game trajectory")
    plot_traj(ax2, sf_phi_games[:25], TRAJ_SF, "SF-vs-SF game trajectory")
    ax2.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")
    ax2.set_title("TICA space: SF-derived basin background + isocommittor(-style) contours\n"
                   "+ sample trajectories (opening 10 ply dropped, as before)")
    ax2.set_xlabel("TICA 1 (slowest)"); ax2.set_ylabel("TICA 2")
    fig2.tight_layout()
    fig2.savefig("artifacts/experiments/meta_2_tica_contours_trajectories.png", dpi=140)
    fig2.savefig("docs/figures/meta_2_tica_contours_trajectories.png", dpi=140)

    # ==== 3. Microstate MSM + spectral (PCCA+-style) macrostates =====
    print("[3/5] building microstates (KMeans) + SF-derived transition matrix + spectral macrostates ...", flush=True)
    n_micro = 60
    pool2d = np.concatenate([human2d, sf2d], 0)
    km = KMeans(n_clusters=n_micro, n_init=10, random_state=seed).fit(pool2d)
    micro_centers = km.cluster_centers_

    def micro_assign(phi_games):
        labels_per_game = []
        for gid, phi in phi_games:
            pts = project(phi, tica_mean, tica_components)
            labels_per_game.append((gid, km.predict(pts)))
        return labels_per_game

    sf_micro_games = micro_assign(sf_phi_games)
    human_micro_games = micro_assign(human_phi_games)

    def count_matrix(micro_games, n_states, lag=1):
        C = np.zeros((n_states, n_states))
        for _, labs in micro_games:
            if len(labs) <= lag:
                continue
            for a, b in zip(labs[:-lag], labs[lag:]):
                C[a, b] += 1
        return C

    C_sf = count_matrix(sf_micro_games, n_micro)
    C_human = count_matrix(human_micro_games, n_micro)
    visited = C_sf.sum(1) > 0
    row_sums = C_sf.sum(1, keepdims=True)
    P_sf = np.divide(C_sf, row_sums, out=np.zeros_like(C_sf), where=row_sums > 0)
    for i in range(n_micro):
        if row_sums[i, 0] == 0:
            P_sf[i, i] = 1.0   # unvisited microstate: self-loop, stays out of any real flux

    # SF-derived destiny value per microstate (from the density sample, not just
    # the sparse trajectory visits -- more stable), for labeling macrostates.
    sf_micro_of_sample = km.predict(sf2d)
    micro_value = np.zeros(n_micro)
    for i in range(n_micro):
        m = sf_micro_of_sample == i
        micro_value[i] = sf["destiny"][m].mean() if m.sum() > 0 else 0.0

    from scipy.linalg import eig
    eigvals, eigvecs = eig(P_sf.T)   # left eigenvectors via eig on the transpose
    order = np.argsort(-eigvals.real)
    eigvals = eigvals.real[order]
    eigvecs = eigvecs.real[:, order]
    print(f"    P_sf leading eigenvalues: {eigvals[:5].round(3).tolist()}")
    n_macro = 3   # fixed at 3 to compare directly against win/draw/loss
    spectral_coords = eigvecs[:, 1:n_macro]   # drop the trivial eigval~1 (stationary) eigenvector
    km_macro = KMeans(n_clusters=n_macro, n_init=10, random_state=seed).fit(spectral_coords)
    macro_of_micro = km_macro.labels_
    # relabel macrostates 0,1,2 by ascending mean destiny value, so 0=loss-leaning,
    # 2=win-leaning macrostate, matching the win/draw/loss reading throughout.
    macro_mean_value = [micro_value[macro_of_micro == k].mean() if (macro_of_micro == k).any() else 0.0
                         for k in range(n_macro)]
    relabel = {old: new for new, old in enumerate(np.argsort(macro_mean_value))}
    macro_of_micro = np.array([relabel[m] for m in macro_of_micro])
    macro_names = {0: "loss-leaning macrostate", 1: "mid macrostate", 2: "win-leaning macrostate"}
    for k in range(n_macro):
        print(f"    macrostate {k} ({macro_names[k]}): {int((macro_of_micro==k).sum())} microstates, "
              f"mean SF destiny value {np.mean([micro_value[i] for i in range(n_micro) if macro_of_micro[i]==k]):.2f}")

    macro_cmap_colors = [COLOR_LOSS, COLOR_DRAW, COLOR_WIN]
    fig3, ax3 = plt.subplots(figsize=(7.5, 6.5))
    ax3.imshow(bg_rgba.transpose(1, 0, 2), origin="lower", extent=[*xlim, *ylim], aspect="auto", alpha=0.35)
    for k in range(n_macro):
        idx = macro_of_micro == k
        ax3.scatter(micro_centers[idx, 0], micro_centers[idx, 1], s=140, c=macro_cmap_colors[k],
                    edgecolors="black", linewidths=0.8, marker="s", alpha=0.85,
                    label=f"{macro_names[k]} (n={idx.sum()})")
    ax3.legend(loc="best", fontsize=8, frameon=True, facecolor="white")
    ax3.set_title(f"PCCA+-STYLE macrostates (spectral clustering of {n_micro} microstates,\n"
                   "SF-derived transition matrix -- simplified stand-in, not literal Weber PCCA+)")
    ax3.set_xlabel("TICA 1"); ax3.set_ylabel("TICA 2")
    fig3.tight_layout()
    fig3.savefig("artifacts/experiments/meta_3_pcca_macrostates.png", dpi=140)
    fig3.savefig("docs/figures/meta_3_pcca_macrostates.png", dpi=140)

    # ==== 4. Disconnectivity-graph-style dendrogram ===================
    print("[4/5] disconnectivity-graph-style dendrogram over the microstate transition matrix ...", flush=True)
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    P_sym = (P_sf + P_sf.T) / 2
    dist = 1.0 - P_sym / max(P_sym.max(), 1e-9)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method="average")

    fig4, ax4 = plt.subplots(figsize=(13, 5.5))
    leaf_colors = [macro_cmap_colors[c] for c in [0, 1, 2]]  # unused directly; matplotlib link_color via leaves below

    def leaf_color_func(leaf_id):
        v = micro_value[leaf_id]
        return COLOR_WIN if v > 0.15 else (COLOR_LOSS if v < -0.15 else COLOR_DRAW)

    dn = dendrogram(Z, ax=ax4, leaf_rotation=90, leaf_font_size=6,
                     link_color_func=lambda k: "#8a8985")
    # recolor leaf tick labels by that microstate's SF destiny value
    for lbl, leaf_id in zip(ax4.get_xticklabels(), dn["leaves"]):
        lbl.set_color(leaf_color_func(leaf_id))
    ax4.set_title("Disconnectivity-graph-STYLE dendrogram: microstates merged by transition-probability\n"
                   "distance (average linkage) -- leaf tick color = that microstate's SF win/draw/loss lean\n"
                   "(NOT the literal Wales energy-threshold construction -- no potential-energy minima here)")
    ax4.set_ylabel("1 - normalized transition probability (merge distance)")
    ax4.set_xlabel("microstate id (leaf color: green=win-leaning, gray=draw-leaning, red=loss-leaning)")
    fig4.tight_layout()
    fig4.savefig("artifacts/experiments/meta_4_disconnectivity_dendrogram.png", dpi=140)
    fig4.savefig("docs/figures/meta_4_disconnectivity_dendrogram.png", dpi=140)

    # ==== 5. MSM-flux-diagram-STYLE 3-node network =====================
    print("[5/5] macrostate flux diagrams (SF vs human, same SF-derived macrostate partition) ...", flush=True)
    import networkx as nx

    def macro_transition_matrix(C_micro):
        n = n_macro
        M = np.zeros((n, n))
        for i in range(n_micro):
            for j in range(n_micro):
                M[macro_of_micro[i], macro_of_micro[j]] += C_micro[i, j]
        row_sums = M.sum(1, keepdims=True)
        return np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)

    M_sf = macro_transition_matrix(C_sf)
    M_human = macro_transition_matrix(C_human)
    print(f"    SF macrostate transition matrix:\n{np.round(M_sf, 3)}")
    print(f"    Human macrostate transition matrix:\n{np.round(M_human, 3)}")
    off_diag_sf = M_sf.sum() - np.trace(M_sf)
    off_diag_human = M_human.sum() - np.trace(M_human)
    print(f"    total off-diagonal (basin-CROSSING) mass: SF={off_diag_sf:.3f} human={off_diag_human:.3f} "
          f"(higher = more leaking between basins)")

    fig5, axes5 = plt.subplots(1, 2, figsize=(13, 6))
    pos = {0: (-1, -0.6), 1: (0, 1), 2: (1, -0.6)}
    for ax, M, title, off in [(axes5[0], M_sf, "SF-vs-SF", off_diag_sf),
                               (axes5[1], M_human, "Human", off_diag_human)]:
        G = nx.DiGraph()
        for k in range(n_macro):
            G.add_node(k)
        for i in range(n_macro):
            for j in range(n_macro):
                if i != j and M[i, j] > 0.001:
                    G.add_edge(i, j, weight=M[i, j])
        node_colors = [macro_cmap_colors[k] for k in range(n_macro)]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=2200,
                                edgecolors="black", linewidths=1.0)
        nx.draw_networkx_labels(G, pos, ax=ax, labels={k: f"{macro_names[k].split()[0]}\n{M[k,k]:.2f} stay" for k in range(n_macro)},
                                 font_size=7)
        for (u, v, d) in G.edges(data=True):
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], width=max(0.5, d["weight"] * 25),
                                    edge_color="#333333", alpha=0.7, arrowsize=15,
                                    connectionstyle="arc3,rad=0.15")
            xm = 0.55 * pos[u][0] + 0.45 * pos[v][0]
            ym = 0.55 * pos[u][1] + 0.45 * pos[v][1]
            ax.text(xm, ym, f"{d['weight']:.3f}", fontsize=7, color="#333333")
        ax.set_title(f"{title}: macrostate transition probabilities\n(off-diagonal / basin-crossing mass = {off:.3f})")
        ax.axis("off")
    fig5.suptitle("MSM-flux-diagram-STYLE macrostate transitions, SAME SF-derived macrostate partition for both\n"
                   "(empirical transition counting -- not full committor-weighted TPT reactive flux)")
    fig5.tight_layout()
    fig5.savefig("artifacts/experiments/meta_5_flux_diagram.png", dpi=140)
    fig5.savefig("docs/figures/meta_5_flux_diagram.png", dpi=140)

    np.savez("artifacts/experiments/meta_data.npz",
             tica_mean=tica_mean, tica_components=tica_components,
             human2d=human2d, sf2d=sf2d, human_destiny=human["destiny"], sf_destiny=sf["destiny"],
             micro_centers=micro_centers, macro_of_micro=macro_of_micro, micro_value=micro_value,
             C_sf=C_sf, C_human=C_human, M_sf=M_sf, M_human=M_human)
    print(f"wrote artifacts/experiments/meta_{{1_tica_side_by_side,2_tica_contours_trajectories,"
          f"3_pcca_macrostates,4_disconnectivity_dendrogram,5_flux_diagram}}.png (+ docs/figures copies) + meta_data.npz")
    print(f"DONE basin_metastability [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
