#!/usr/bin/env python
"""catspace/research/components/planner/approaches/subgoal_cascade/experiments/planner_landmark.py -- PHASE 1 of the subgoal planner (Kaveh 2026-07-20):
high-level plan over natural clusters, grounded in Search-on-the-Replay-Buffer (SoRB,
Eysenbach 2019) / Projective Quasimetric Planning (ProQ 2025), adapted to our directed
quasimetric field.

  1. Embed nucleus positions (board-only F/B) with iqe_geom.
  2. Find NATURAL clusters (HDBSCAN, min_cluster_size>=3); each cluster's MEDOID (the
     real position nearest the cluster mean) is a LANDMARK.
  3. Directed landmark graph: edge i->j weight = field quasimetric d(F(i), B(j)) ~ plies;
     each landmark's cost-to-mate = its tablebase DTM (ground truth, ~plies -- same units
     as the field's 1-per-ply successor pin).
  4. For a query position: Dijkstra START->MATE through landmarks = the shortest plan;
     Yen's K-shortest-paths = ranked candidate plans. First landmark >=~5-6 plies out on
     the best plan = the immediate SUBGOAL.
NOTE the field distance is the COOPERATIVE/optimistic heuristic; Phase 2 (adversarial
search) finds the REAL cost to the subgoal. Logs the plans; draws them over the UMAP.

Usage:
  .venv/bin/python catspace/research/components/planner/approaches/subgoal_cascade/experiments/planner_landmark.py --query-dtm 24 --k-plans 4
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch, chess, networkx as nx, matplotlib
from catspace.io import paths
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from sklearn.cluster import HDBSCAN, KMeans

BOARD_ONLY = (18, 19)


def matkey(pk, mt):
    return "".join(sorted(p.symbol() for p in board_from_packed(pk, mt).piece_map().values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.sep("iqe_geom.pt"))
    ap.add_argument("--data", default=paths.derived("lichess_nearmate.npz"))
    ap.add_argument("--n", type=int, default=6000, help="nucleus positions sampled for landmarks+manifold")
    ap.add_argument("--min-cluster", type=int, default=25, help="HDBSCAN min_cluster_size (natural clusters)")
    ap.add_argument("--kmeans", type=int, default=0, help="use KMeans with this k instead of HDBSCAN")
    ap.add_argument("--knn", type=int, default=8, help="each landmark connects to its knn nearest landmarks")
    ap.add_argument("--subgoal-plies", type=float, default=5.0, help="immediate subgoal is first landmark >= this far")
    ap.add_argument("--query-dtm", type=int, default=24, help="pick a query position near this DTM")
    ap.add_argument("--k-plans", type=int, default=4)
    ap.add_argument("--out", default=paths.experiment("planner_landmark.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device("auto")
    fb, _ = load_ckpt(Path(args.ckpt), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.data); won = np.flatnonzero(nz["dtm"] > 0)
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(won, min(args.n, len(won)), replace=False)
    pk, mt, dtm = nz["packed"][sel], nz["meta"][sel], nz["dtm"][sel].astype(np.float32)
    stm = mt[:, 0]                                                    # side to move (0=W,1=B)

    def emb(pk, mt, side):
        out = []
        for i in range(0, len(pk), 2048):
            pl = feature_planes(pk[i:i+2048], mt[i:i+2048]); pl[:, BOARD_ONLY] = 0.0
            t = torch.from_numpy(pl).to(dev)
            with torch.no_grad():
                if side == "F":
                    e = fb.embed_F(t, torch.from_numpy(np.tile(om, (len(pl), 1))).to(dev))
                else:
                    e = fb.embed_B(t)
            out.append(e.cpu())
        return torch.cat(out)

    print(f"[stage] embedding {len(sel)} nucleus positions (board-only F/B)...", flush=True)
    F = emb(pk, mt, "F"); B = emb(pk, mt, "B")
    Fn = torch.nn.functional.normalize(F, dim=1).numpy()

    # -- natural clusters -> landmark medoids --
    if args.kmeans:
        lab = KMeans(n_clusters=args.kmeans, n_init=4, random_state=args.seed).fit_predict(Fn)
    else:
        lab = HDBSCAN(min_cluster_size=args.min_cluster).fit_predict(Fn)
    clusters = [c for c in sorted(set(lab)) if c >= 0]               # drop HDBSCAN noise (-1)
    lm = []                                                          # landmark member-indices (medoids)
    for c in clusters:
        idx = np.flatnonzero(lab == c)
        centroid = Fn[idx].mean(0)
        medoid = idx[np.argmax(Fn[idx] @ centroid)]                 # member nearest cluster mean (cosine)
        lm.append(medoid)
    lm = np.array(lm)
    print(f"[stage] {len(clusters)} natural clusters (n>= {args.min_cluster}); "
          f"{int((lab<0).mean()*100)}% noise; {len(lm)} landmark medoids", flush=True)

    # -- directed landmark graph: edge i->j = field d(F_i, B_j); cost-to-mate = DTM --
    with torch.no_grad():
        Dll = fb.distance_matrix(F[lm].to(dev), B[lm].to(dev)).cpu().numpy()   # (L,L)
    G = nx.DiGraph()
    for i in range(len(lm)):
        G.add_node(int(i), dtm=float(dtm[lm[i]]), mat=matkey(pk[lm[i]], mt[lm[i]]), stm=int(stm[lm[i]]))
        G.add_edge(int(i), "MATE", weight=float(dtm[lm[i]]))          # landmark -> mate = tablebase DTM
    for i in range(len(lm)):
        near = np.argsort(Dll[i])[:args.knn + 1]
        for j in near:
            if j != i:
                G.add_edge(int(i), int(j), weight=float(Dll[i, j]))

    # -- query position (near a target DTM), plan START->MATE --
    q = sel[np.argmin(np.abs(dtm - args.query_dtm))]                 # a real nucleus position
    qpk, qmt = nz["packed"][q][None], nz["meta"][q][None]
    with torch.no_grad():
        pl = feature_planes(qpk, qmt); pl[:, BOARD_ONLY] = 0.0
        Fq = fb.embed_F(torch.from_numpy(pl).to(dev), torch.from_numpy(om[None]).to(dev))
        dq_lm = fb.distance_matrix(Fq, B[lm].to(dev)).cpu().numpy()[0]           # query -> each landmark
    Gq = G.copy()
    Gq.add_node("START", dtm=float(nz["dtm"][q]), mat=matkey(nz["packed"][q], nz["meta"][q]), stm=int(nz["meta"][q][0]))
    for j in range(len(lm)):
        Gq.add_edge("START", int(j), weight=float(dq_lm[j]))
    Gq.add_edge("START", "MATE", weight=float(nz["dtm"][q]))          # direct (no subgoal) baseline

    paths = []
    for p in nx.shortest_simple_paths(Gq, "START", "MATE", weight="weight"):
        cost = nx.path_weight(Gq, p, "weight")
        paths.append((p, cost))
        if len(paths) >= args.k_plans:
            break

    print(f"\n=== QUERY position: DTM={int(nz['dtm'][q])} material={matkey(nz['packed'][q], nz['meta'][q])} "
          f"{'W' if nz['meta'][q][0]==0 else 'B'}-to-move ===")
    print(f"    direct field->mate estimate (no subgoal): {float(nz['dtm'][q]):.1f} (tablebase DTM)")
    for r, (p, cost) in enumerate(paths):
        hop = " -> ".join("MATE" if x == "MATE" else "START" if x == "START"
                          else f"L{x}[{G.nodes[x]['mat']},dtm{int(G.nodes[x]['dtm'])}]" for x in p)
        print(f"  PLAN {r+1}  total~{cost:.1f} plies, {len(p)-2} subgoals:  {hop}")
    # immediate subgoal on the best plan: first landmark >= subgoal-plies field-hop away
    best = paths[0][0]
    subgoal = None; acc = 0.0; prev = "START"
    for x in best[1:]:
        acc += Gq[prev][x]["weight"]; prev = x
        if x != "MATE" and acc >= args.subgoal_plies:
            subgoal = x; break
    if subgoal is None and len(best) > 2:
        subgoal = best[1]
    print(f"  => IMMEDIATE SUBGOAL: "
          f"{'(mate directly)' if subgoal is None else f'L{subgoal} [{G.nodes[subgoal]['mat']}, dtm {int(G.nodes[subgoal]['dtm'])}], ~{acc:.1f} plies out'}")

    # -- visualize over UMAP: manifold + plan hops (nodes white/black by side-to-move) --
    print("[stage] UMAP + plan viz...", flush=True)
    import umap
    allF = np.vstack([Fn, torch.nn.functional.normalize(Fq.cpu(), dim=1).numpy()])
    XY = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0).fit_transform(allF)
    xy_pos, xy_q = XY[:-1], XY[-1]
    xy_lm = xy_pos[lm]
    xy_mate = xy_pos[np.argsort(dtm)[:200]].mean(0)                  # mate "pole" = mean of lowest-DTM
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(xy_pos[:, 0], xy_pos[:, 1], s=3, c=np.minimum(dtm, 30), cmap="viridis_r", alpha=0.25)
    node_xy = {"START": xy_q, "MATE": xy_mate, **{int(i): xy_lm[i] for i in range(len(lm))}}
    node_stm = {"START": int(nz["meta"][q][0]), "MATE": 0, **{int(i): int(stm[lm[i]]) for i in range(len(lm))}}
    ax.scatter(xy_lm[:, 0], xy_lm[:, 1], s=45, facecolors="none", edgecolors="0.4", linewidths=0.8, zorder=3)
    # best plan hops as black arrows; nodes white(=W to move)/black(=B to move)
    p = paths[0][0]
    for a, b in zip(p[:-1], p[1:]):
        xa, xb = node_xy[a], node_xy[b]
        ax.annotate("", xy=xb, xytext=xa, zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2, shrinkA=8, shrinkB=8))
    for name in p:
        c = "white" if node_stm[name] == 0 else "black"
        ax.scatter(*node_xy[name], s=180 if name in ("START", "MATE") else 110,
                   c=c, edgecolors="black", linewidths=1.5, zorder=5)
    ax.annotate("START", xy_q, textcoords="offset points", xytext=(8, 8), fontweight="bold", zorder=6)
    ax.annotate("MATE", xy_mate, textcoords="offset points", xytext=(8, 8), fontweight="bold", zorder=6)
    ax.set_title(f"Phase-1 subgoal plan over the field (SoRB/ProQ-style). Query DTM {int(nz['dtm'][q])}, "
                 f"best plan {paths[0][1]:.0f} plies, {len(p)-2} subgoals.\n"
                 f"White node = White to move, black = Black to move; black arrows = subgoal hops.")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"VERDICT PLANNER landmarks={len(lm)} best_plan_plies={paths[0][1]:.1f} "
          f"n_subgoals={len(p)-2} -> {args.out}")


if __name__ == "__main__":
    main()
