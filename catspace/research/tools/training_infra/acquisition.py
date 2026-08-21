#!/usr/bin/env python
"""acquisition.py -- the flywheel's brain: score what a label would teach, and GROW the
pool where the model is uncertain (Kaveh 2026-08-21: "wire it in so it kicks in
immediately").

Score(position) = z(interval width) + z(code rarity) + z(policy entropy)
    width    gciv1's concepts->alpha/beta head: mean over legal moves of (hi - lo)
    rarity   1/sqrt(count) of the position's active codes (gcf4h corpus stats)
    entropy  the current move head's predicted-distribution entropy (elo 1500 view)
Components are z-scored over the scored batch and SUMMED: a flat component leaves the
ranking unchanged (the U15 invariant class). Top-K scored positions also seed EXPANSION
WALKS (2-4 random plies) whose endpoints join the pool -- generation as exploration.

Ranked fens are enqueued with their score; producers claim score-DESC, so ranked rows
preempt the random-refill fallback with no producer changes.

    .venv/bin/python -m ...acquisition loop --sample 12000 --top 6000
"""
from __future__ import annotations

import argparse
import sqlite3
import time

import numpy as np
import torch

DB = "data/derived/gko1_oracle.sqlite"
MV_CAP = 48


def load_models(gko_ckpt, gciv_ckpt, stats_path, dev):
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_concept_field import GraphConceptField
    import torch.nn as nn
    ck = torch.load(gko_ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    A = torch.load("artifacts/experiments/gcf4h_latest.pt",
                   map_location="cpu", weights_only=False)["args"]
    field = GraphConceptField(
        d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"], n_res=3, n_eg=1,
        cls_reach=np.zeros(3, np.int64), cls_eg=np.zeros(3, np.int64),
        cls_out=np.arange(3, dtype=np.int64), edge_codes=A["edge_codes"],
        move_codes=A["move_codes"], gps_parallel=bool(A["gps_parallel"]),
        attn_bias=bool(A["attn_bias"]), agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
        file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
        threat_head=bool(A["threat_head"]), elo_cond=True,
        g_heads=A["g_heads"], g_codes=A["g_codes"], d_code=A["d_code"]).to(dev)
    fkeys = {k[len("field."):]: v for k, v in sd.items() if k.startswith("field.")}
    field.load_state_dict(fkeys if fkeys else sd)
    field.eval()
    gc = torch.load(gciv_ckpt, map_location="cpu", weights_only=False)
    head = nn.Sequential(nn.Linear(field.d_state + 32, 256), nn.GELU(),
                         nn.Linear(256, 3)).to(dev)
    head.load_state_dict(gc["head"]); head.eval()
    sz = np.load(stats_path, allow_pickle=True)
    counts = sz["count"]
    return field, head, counts, A


@torch.no_grad()
def score_fens(fens, field, head, counts, dev, elo=1500.0):
    import chess
    import torch.nn.functional as F
    from catspace.research.tools.training_infra.gen_move_graphs import (
        parent_graph, glob_of_board)
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize
    from catspace.research.components.encoder.approaches.reach_probability.src \
        .trajectories import normalise_elo
    widths, rars, ents, sigs = [], [], [], []
    ev = float(normalise_elo(np.float32(elo)))
    B = 96
    for a in range(0, len(fens), B):
        chunk = fens[a:a + B]
        gs, ok = [], []
        for f in chunk:
            try:
                b = chess.Board(f)
                gs.append((parent_graph(b)[0], glob_of_board(b)))
                ok.append(True)
            except Exception:
                ok.append(False)
        if not gs:
            widths += [np.nan] * len(chunk); rars += [np.nan] * len(chunk)
            ents += [np.nan] * len(chunk); sigs += [None] * len(chunk)
            continue
        D = {k: torch.from_numpy(np.stack([g[0][k] for g in gs]))
             for k in ("sq", "mv", "mut", "vac", "dsc", "att")}
        D["gl"] = torch.from_numpy(np.stack([g[1] for g in gs]))
        D["n_mv"] = torch.tensor([g[0]["n_mv"] for g in gs], dtype=torch.uint8)
        D["n_att"] = torch.tensor([g[0]["n_att"] for g in gs], dtype=torch.uint16)
        bb = {k: D[k].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
        sqf, mvf, edges, mask = featurize(bb, dev, n_nodes=field.n_nodes)
        off = (torch.arange(len(gs), device=dev) * field.n_nodes)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for name in ("vac", "dsc"):
            arr = D[name].to(dev).long()
            val = (arr != 255) & mask[:, :, None]
            edges[name] = (((jj + 64 + off)[:, :, None].expand_as(arr))[val],
                           (arr + off[:, :, None])[val])
        elo_t = torch.full((len(gs), 2), ev, device=dev)
        zq, g_ids, _, _, h_mv, _ = field.encode(sqf, mvf, edges, mask, elo=elo_t)
        lg = field.move_logits(zq, h_mv, mask)
        p = torch.softmax(lg.masked_fill(~mask, -1e9), -1)
        ent = -(p * torch.log(p.clamp(min=1e-12))).sum(-1)
        q_mv, _, _ = field.mv_vq(field.mv_vq_proj(h_mv))
        x = torch.cat([zq[:, None, :].expand(-1, MV_CAP, -1), q_mv], -1)
        o = head(x)
        wid = (torch.nn.functional.softplus(o[..., 1])
               + torch.nn.functional.softplus(o[..., 2]))
        wid = (wid * mask.float()).sum(-1) / mask.float().sum(-1).clamp(min=1)
        gi = g_ids.cpu().numpy()
        r = (1.0 / np.sqrt(np.maximum(
            counts[np.arange(counts.shape[0])[None, :], gi], 1))).sum(1)
        it = iter(range(len(gs)))
        for okj in ok:
            if okj:
                j = next(it)
                widths.append(float(wid[j])); rars.append(float(r[j]))
                ents.append(float(ent[j]))
                sigs.append(tuple(gi[j][:8]))
            else:
                widths.append(np.nan); rars.append(np.nan)
                ents.append(np.nan); sigs.append(None)
    def z(v):
        v = np.asarray(v, np.float64)
        m = np.nanmean(v); s = np.nanstd(v)
        return (v - m) / (s if s > 1e-9 else 1.0)
    score = z(widths) + z(rars) + z(ents)
    return score, sigs


def expansion_walks(fens, n_walks=2, max_plies=4, seed=0):
    import chess
    rng = np.random.default_rng(seed)
    out = []
    for f in fens:
        for _ in range(n_walks):
            try:
                b = chess.Board(f)
                for _2 in range(int(rng.integers(2, max_plies + 1))):
                    mv = list(b.legal_moves)
                    if not mv or b.is_game_over():
                        break
                    b.push(mv[rng.integers(0, len(mv))])
                if not b.is_game_over():
                    out.append(b.fen())
            except Exception:
                pass
    return out


def round_once(a, dev="cpu"):
    field, head, counts, A = load_models(a.gko, a.gciv, a.stats, dev)
    c = sqlite3.connect(DB, timeout=60)
    c.execute("PRAGMA journal_mode=DELETE")
    fens = [r[0] for r in c.execute(
        "SELECT fen FROM pool ORDER BY RANDOM() LIMIT ?", (a.sample,))]
    t0 = time.time()
    score, sigs = score_fens(fens, field, head, counts, dev)
    order = np.argsort(-np.where(np.isfinite(score), score, -1e9))
    seen_sig, ranked = set(), []
    for i in order:
        if len(ranked) >= a.top:
            break
        s = sigs[i]
        if s is not None and s in seen_sig:
            continue
        if s is not None:
            seen_sig.add(s)
        ranked.append((fens[i], float(score[i])))
    with c:
        for oracle in a.oracles.split(","):
            c.executemany(
                "INSERT OR IGNORE INTO queue(fen, oracle, score) VALUES (?,?,?)",
                [(f, oracle, sc) for f, sc in ranked])
    grow = expansion_walks([f for f, _ in ranked[:a.grow_from]],
                           n_walks=a.walks, seed=int(time.time()) % 99999)
    # U11 hygiene: never grow the pool into a ChessBench test position
    try:
        import numpy as _np
        _tk = set(_np.load("data/derived/chessbench_test_keys.npz")["keys"].tolist())
        pre = len(grow)
        grow = [f for f in grow if " ".join(f.split()[:4]) not in _tk]
        if pre - len(grow):
            print(f"[acq] U11 filter dropped {pre - len(grow)} test-set walk fens", flush=True)
    except FileNotFoundError:
        pass
    with c:
        c.executemany("INSERT OR IGNORE INTO pool VALUES (?, 'expand')",
                      [(f,) for f in grow])
    c.close()
    print(f"[acq] scored {len(fens):,} -> enqueued {len(ranked):,}/oracle "
          f"(dedup {len(order)-len(ranked)}) | pool +{len(grow):,} expansion fens "
          f"[{(time.time()-t0)/60:.1f}m]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("once", "loop"):
        p = sub.add_parser(nm)
        p.add_argument("--gko", default="artifacts/experiments/gko0_latest.pt")
        p.add_argument("--gciv", default="artifacts/experiments/gciv1_latest.pt")
        p.add_argument("--stats", default="data/derived/concept_stats_gcf4h.npz")
        p.add_argument("--sample", type=int, default=12000)
        p.add_argument("--top", type=int, default=6000)
        p.add_argument("--grow-from", type=int, default=1500)
        p.add_argument("--walks", type=int, default=2)
        p.add_argument("--oracles",
                       default="sf,maia:1100,maia:1300,maia:1500,maia:1700,maia:1900")
        p.add_argument("--every", type=int, default=3600)
    a = ap.parse_args()
    while True:
        try:
            round_once(a)
        except Exception as e:
            print(f"[acq] round failed: {type(e).__name__}: {str(e)[:140]}", flush=True)
        if a.cmd == "once":
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
