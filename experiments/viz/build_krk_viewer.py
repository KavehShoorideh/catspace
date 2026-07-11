#!/usr/bin/env python
"""
experiments/viz/build_krk_viewer.py — the rung-1 (KRk) interactive viewer:
per-ply board position, learned candidate-move scores, plan token, and
ground-truth concept values, rendered into catspace/viz/templates/
krk_viewer.html via viz.build_html. Port of gen_ui_data.py + viewer_template
.html (this is the piece that was previously assembled by an uncommitted,
manual injection step).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from catspace.chain import empirical_P
from catspace.concepts import KMeansVQ
from catspace.domains import krk
from catspace.domains.krk import white_moves
from catspace.game import rollout_transitions
from catspace.io.paths import REPO_ROOT, generated_dir
from catspace.opponents import RandomOpponent
from catspace.planner.policy import RandomPolicy
from catspace.cone.tabular import fb_from_svd, randomized_svd_sm
from catspace.viz.build_html import build_html
from catspace.viz.payload import json_default

GAMMA = 0.92
TEMPLATE = REPO_ROOT / "catspace/viz/templates/krk_viewer.html"


@dataclass
class GameContext:
    chain: object
    W: list
    dtm_w: np.ndarray
    feats: dict
    F: np.ndarray
    zg: np.ndarray
    visited: np.ndarray
    assign: np.ndarray


def mv_fromto(st, bnode):
    (wk, wr, bk), (wk2, wr2, _) = st, bnode
    return [wk, wk2] if wk2 != wk else [wr, wr2]


def score_moves(ctx: GameContext, si: int):
    st = ctx.W[si]
    bnodes = white_moves(*st)
    chain = ctx.chain
    lo, hi = int(chain.move_ptr[si]), int(chain.move_ptr[si + 1])
    out = []
    for mi, mid in enumerate(range(lo, hi)):
        outs = chain.outs_of(mid)
        v, has_mate = 0.0, False
        for o in outs:
            o = int(o)
            if o == chain.terminals.mate:
                v += float(ctx.F[o] @ ctx.zg); has_mate = True
            elif o == chain.terminals.draw:
                v += 0.0
            else:
                v += float(ctx.F[o] @ ctx.zg) if ctx.visited[o] else 0.0
        out.append((v / len(outs), mi, has_mate, bnodes[mi]))
    out.sort(key=lambda t: -t[0])
    return out, st


def play_game(ctx: GameContext, start: int, cap: int = 60, seed: int = 0):
    chain = ctx.chain
    rng = np.random.default_rng(seed)
    s = int(start)
    plies = []
    result = "unfinished"
    for _ in range(cap):
        scored, st = score_moves(ctx, s)
        best_v, best_mi, _, best_bnode = scored[0]
        cands = [dict(name=chain.move_names[int(chain.move_ptr[s]) + mi], score=float(v),
                      fromTo=mv_fromto(st, bnode), mates=bool(hm))
                 for v, mi, hm, bnode in scored[:6]]
        mid = int(chain.move_ptr[s]) + best_mi
        outs = chain.outs_of(mid)
        nxt = int(outs[rng.integers(0, len(outs))])
        chosen_ft = mv_fromto(st, best_bnode)
        black_ft = None
        if nxt < chain.n_live:
            black_ft = [best_bnode[2], ctx.W[nxt][2]]
        plies.append(dict(
            wk=st[0], wr=st[1], bk=st[2],
            move=chain.move_names[mid], fromTo=chosen_ft, blackReply=black_ft,
            token=int(ctx.assign[s]), reach=float(ctx.F[s] @ ctx.zg),
            concepts=dict(dtm=float(min(ctx.dtm_w[s], 60)), box=float(ctx.feats["box_area"][s]),
                          kk=float(ctx.feats["kk_dist"][s]), rookbk=float(ctx.feats["rook_bk_dist"][s]),
                          bkedge=float(ctx.feats["bk_edge"][s])),
            candidates=cands))
        if nxt == chain.terminals.mate:
            result = "mate"; break
        if nxt == chain.terminals.draw:
            result = "draw"; break
        s = nxt
    return dict(result=result, plies=plies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=32000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    chain = krk.build_chain()
    W, B = krk.enumerate_states()
    dtm_w, _ = krk.compute_dtm(W, B)
    feats = krk.concept_features(W, dtm_w)
    region = np.concatenate([np.where(dtm_w <= 3)[0], [chain.terminals.mate]])

    rng = np.random.default_rng(11)
    starts = rng.integers(0, chain.n_live, size=args.n_games)
    rows, cols, _ = rollout_transitions(chain, RandomPolicy(), RandomOpponent(), starts, cap=200, rng=rng)
    Phat, visited = empirical_P(rows, cols, chain.n, chain.terminals)
    U, S, V = randomized_svd_sm(Phat, GAMMA, d=64, seed=0)
    F, Bm = fb_from_svd(U, S, V)
    zg = Bm[region].sum(0)

    Fn = F[:chain.n_live]
    Xf = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-9)
    K = 32
    vq = KMeansVQ(n_tokens=K, seed=5).fit(Xf[:, :16])
    assign = vq.tokens(Xf[:, :16])

    ctx = GameContext(chain=chain, W=W, dtm_w=dtm_w, feats=feats, F=F, zg=zg, visited=visited, assign=assign)

    token_legend = []
    for k in range(K):
        m = assign == k
        token_legend.append(dict(id=k, size=int(m.sum()),
                                  meanDtm=float(feats["dtm"][m].mean()) if m.any() else None,
                                  meanBox=float(feats["box_area"][m].mean()) if m.any() else None))

    games = []
    game_rng = np.random.default_rng(4)
    for lo, hi, seed in ((15, 20, 1), (13, 20, 2), (11, 14, 3), (9, 12, 4), (7, 10, 5), (16, 20, 6)):
        cand = np.where((dtm_w >= lo) & (dtm_w <= hi))[0]
        start = int(cand[game_rng.integers(0, len(cand))])
        g = play_game(ctx, start, seed=seed)
        g["startDtm"] = float(dtm_w[start])
        games.append(g)
        print(f"game: start DTM {dtm_w[start]:.0f} -> {g['result']} in {len(g['plies'])} plies")

    data = dict(N=5, gamma=GAMMA, K=K, tokens=token_legend, games=games,
                meta=dict(trainedGames=args.n_games, rank=64, regionDef="DTM<=3 ∪ {mate}",
                          engine="greedy cone-steering, learned from random play"))

    out_dir = generated_dir()
    json_path = out_dir / "ui_data.json"
    json_path.write_text(json.dumps(data, default=json_default))
    print(f"wrote {json_path}")

    out = args.out or (out_dir / "krk-viewer.html")
    build_html(TEMPLATE, data, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
