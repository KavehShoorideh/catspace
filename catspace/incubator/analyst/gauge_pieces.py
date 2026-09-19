"""GAUGES FOR THE PER-PIECE FUTURES (plan 2026-09-18, step F). Every number beside its degenerate baseline.

  heads      the test split, per piece kind: 'no free moves 20 plies on' AUC of the head vs PERSISTENCE (inactive now);
             'exchanged within 20' AUC vs the base rate; 'hits the enemy king zone 20 plies on' AUC vs persistence.
  engine     test positions: for every piece with a priced trade (the piece-value table), the moves that carry the
             trade out (rank.piece_serving_moves with the operator forced: trade ours / remove theirs); the engine's
             evaluation of the best such move minus its best move (depth 12, cached). Spearman of the outcome value
             against that gap, beside zero; the mean gap by the value's sign.
  repertoire Kaveh's games grouped by opening (ECO): at his move near ply 16, the per-piece futures for his pieces
             and the opponent's, averaged by kind and attribute ('our light-squared bishop'), against the corpus rate
             for the kind in the phase -- the prep view: 'in the French your light-squared bishop tends to be stuck'.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

from . import knowledge_net as K

OUT = Path("data/derived/pieces")
KIND = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}


@torch.no_grad()
def heads(ckpt: str, data_dir: str = "data/derived/knowledge", log=print):
    from sklearn.metrics import roc_auc_score
    model, cfg, device = K.load_model(ckpt)
    meta = json.load(open(Path(ckpt).parent / "meta.json"))
    data = K.Data(data_dir, buckets=K.Buckets.from_state(cfg.buckets, meta["buckets"]), n_buckets=cfg.buckets)
    ids = data.ids("test")
    FMq, KZq = K.PIECE_FUTURE_QUESTIONS.index("free_moves"), K.PIECE_FUTURE_QUESTIONS.index("king_zone_hits")
    b_fm = int(np.searchsorted(data.buckets.edges[K.NODE_QUESTIONS.index("free_moves")], 0.0, side="right"))
    b_kz = int(np.searchsorted(data.buckets.edges[K.NODE_QUESTIONS.index("king_zone_hits")], 0.0, side="right"))
    acc = defaultdict(lambda: defaultdict(list))
    for s in range(0, len(ids), 256):
        chunk = ids[s:s + 256]
        b = data.batch(chunk, device); o = model(b)
        pfl, fl = model.piece_future(o["h"])
        P = torch.softmax(pfl[:, :, 1], -1).cpu().numpy(); Fp = torch.softmax(fl[:, :, 1], -1).cpu().numpy()
        types = b["types"].cpu().numpy(); has = b["pf_has20"].cpu().numpy(); fate = b["fate20"].cpu().numpy(); mask = b["mask"].cpu().numpy()
        now = b["buckets"].cpu().numpy()[..., data.pf_q]; fut = b["pf20"].cpu().numpy()
        for i in range(len(chunk)):
            for j in range(32):
                if not mask[i, j]:
                    continue
                t = int(types[i, j]); kind = KIND[t if t <= 6 else t - 6]; side = "our" if t <= 6 else "their"
                for key in (f"{side} {kind}", f"any {kind}"):
                    a = acc[key]
                    if has[i, j]:
                        a["in_y"].append(fut[i, j, FMq] == b_fm); a["in_p"].append(P[i, j, FMq, b_fm]); a["in_now"].append(now[i, j, FMq] == b_fm)
                        a["kz_y"].append(fut[i, j, KZq] != b_kz); a["kz_p"].append(1 - P[i, j, KZq, b_kz]); a["kz_now"].append(now[i, j, KZq] != b_kz)
                    if fate[i, j] >= 0:
                        a["ex_y"].append(fate[i, j] == 1); a["ex_p"].append(Fp[i, j, 1])
    def auc(y, p):
        y = np.asarray(y, bool)
        return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    log(f"[gauge-pieces] heads, {ckpt}, test split ({len(ids)} positions); AUC of the head (persistence) [base rate]")
    rows = {}
    for key in sorted(acc, key=lambda k: (k.split()[0] != "any", k)):
        a = acc[key]
        if len(a["in_y"]) < 200 or key.endswith("king"):
            continue
        r = dict(n=len(a["in_y"]), inactive=auc(a["in_y"], a["in_p"]), inactive_persist=auc(a["in_y"], np.asarray(a["in_now"], float)), inactive_base=float(np.mean(a["in_y"])),
                 kz=auc(a["kz_y"], a["kz_p"]), kz_persist=auc(a["kz_y"], np.asarray(a["kz_now"], float)), kz_base=float(np.mean(a["kz_y"])),
                 exchanged=auc(a["ex_y"], a["ex_p"]), exchanged_base=float(np.mean(a["ex_y"])) if a["ex_y"] else float("nan"))
        rows[key] = r
        log(f"  {key:<14} n={r['n']:6d} | no free moves @20 {r['inactive']:.3f} ({r['inactive_persist']:.3f}) [{r['inactive_base']:.2f}] | "
            f"hits the king zone @20 {r['kz']:.3f} ({r['kz_persist']:.3f}) [{r['kz_base']:.2f}] | exchanged within 20 {r['exchanged']:.3f} [{r['exchanged_base']:.3f}]")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT / "heads.json", "w"), indent=1)
    return rows


def engine_gauge(ckpt: str, data_dir: str = "data/derived/knowledge", n: int = 300, seed: int = 0, depth: int = 12, log=print):
    from scipy.stats import spearmanr
    from .planner import Planner
    from .engine import Engine
    from .memo import Memo
    from .rank import piece_serving_moves
    OUT.mkdir(parents=True, exist_ok=True)
    pl = Planner(ckpt)
    D = K.Data(data_dir, meta_only=True, log=lambda *a, **k: None)
    ids = [i for i in D.ids("test") if D.meta[i]["ply"] >= 12]
    random.Random(seed).shuffle(ids)
    eng = Engine(depth=depth, cache_path=str(OUT / "engine.sqlite"))
    pts = []
    t0 = time.perf_counter()
    for k, i in enumerate(ids[:n]):
        m = D.meta[i]; b = chess.Board(m["fen"]); memo = Memo()
        try:
            plan = pl.plan(b, memo=memo, cp=m["best_cp"])
            best = eng.best(b)
            if best is None:
                continue
            for idea in plan.pieces:
                if idea.value is None:
                    continue
                forced = type(idea)(**{**idea.__dict__, "op": "trade" if idea.ours else "remove"})
                class P_: pass
                p_ = P_(); p_.pieces = [forced]
                moves = piece_serving_moves(b, p_, list(b.legal_moves), memo)
                if not moves:
                    continue
                evs = [eng.eval_move(b, mv) for mv in moves]
                evs = [e.cp for e in evs if e is not None]
                if not evs:
                    continue
                gap = max(evs) - best.cp
                pts.append(dict(value=idea.value, se=idea.se, gap=gap, ours=idea.ours, stands_out=bool(idea.line), kind=idea.text.split(" on ")[0], fen=m["fen"], cp=m["best_cp"]))
        except Exception as ex:
            log(f"[gauge-pieces] {i}: {type(ex).__name__} {str(ex)[:80]}")
        if (k + 1) % 25 == 0:
            log(f"[gauge-pieces] engine {k + 1}/{n} positions, {len(pts)} priced trades, {time.perf_counter() - t0:.0f}s", flush=True)
    eng.close()
    json.dump(pts, open(OUT / "engine.json", "w"))
    report_engine(pts, log)
    return pts


def report_engine(pts, log=print):
    from scipy.stats import spearmanr
    if len(pts) < 10:
        log(f"[gauge-pieces] too few priced trades ({len(pts)})"); return
    v = np.array([p["value"] for p in pts]); g = np.clip(np.array([p["gap"] for p in pts], float), -1000, 0)
    rho, pv = spearmanr(v, g)
    log(f"[gauge-pieces] engine: {len(pts)} priced trades; Spearman(outcome value, engine gap of the best move that carries it out) {rho:+.3f} (p {pv:.3f}; zero is the baseline)")
    for name, sel in (("value > +0.02", v > 0.02), ("|value| <= 0.02", np.abs(v) <= 0.02), ("value < -0.02", v < -0.02)):
        if sel.sum():
            log(f"  {name:<16} n={int(sel.sum()):4d}: median gap {np.median(g[sel]):+.0f} cp, within 30 cp of the best {100 * float((g[sel] >= -30).mean()):.0f}%")
    for who in (True, False):
        sel = np.array([p["ours"] == who for p in pts])
        if sel.sum() >= 10:
            r_, p_ = spearmanr(v[sel], g[sel])
            log(f"  {'our pieces (trade)' if who else 'their pieces (remove)':<22} n={int(sel.sum())}: Spearman {r_:+.3f} (p {p_:.3f})")


def repertoire(ckpt: str, pgn: str = "data/lichess/huvaxshtra_all.pgn", player: str = "huvaxshtra", ply: int = 16, min_games: int = 8, max_games: int = 3000, log=print):
    from .planner import Planner
    from .memo import Memo
    pl = Planner(ckpt)
    groups = defaultdict(list)
    with open(pgn) as f:
        for _ in range(max_games):
            g = chess.pgn.read_game(f)
            if g is None:
                break
            h = g.headers
            me = chess.WHITE if h.get("White", "").lower() == player else chess.BLACK if h.get("Black", "").lower() == player else None
            if me is None:
                continue
            fam = re.split(r"[:,]", h.get("Opening", "?"))[0].strip()
            groups[(fam, me)].append(g)
    t0 = time.perf_counter()
    out = {}
    for (fam, me), games in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(games) < min_games:
            continue
        agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])     # p_inactive, base, p_kz, base_kz, n
        for g in games:
            b = g.board(); k = 0
            for mv in g.mainline_moves():
                if k >= ply and b.turn == me:
                    break
                b.push(mv); k += 1
            if k < ply or b.is_game_over():
                continue
            try:
                plan = pl.plan(b, memo=Memo())
            except Exception:
                continue
            for i in plan.pieces:
                label = re.sub(r" on [a-h][1-8]$", "", i.text).replace("our ", "your ").replace("their ", "their ")
                a = agg[label]; a[0] += i.p_inactive20; a[1] += i.base_inactive20; a[2] += i.p_kz20; a[3] += i.base_kz20; a[4] += 1
        rows = []
        for label, (pi, bi, pk, bk, n_) in agg.items():
            if n_ >= max(4, len(games) // 3):
                rows.append((label, pi / n_, bi / n_, pk / n_, bk / n_, n_))
        if not rows:
            continue
        colour = "White" if me == chess.WHITE else "Black"
        stuck = sorted([r for r in rows if r[0].startswith("your")], key=lambda r: -(r[1] - r[2]))[:2]
        active = sorted([r for r in rows if r[0].startswith("their")], key=lambda r: -(r[3] - r[4]))[:2]
        log(f"[repertoire] {fam} as {colour} ({len(games)} games, the position at ply {ply}):")
        for r in stuck:
            log(f"    {r[0]:<34} no free moves 20 plies on {r[1]:.0%} (its kind in the phase {r[2]:.0%}; n={r[5]})")
        for r in active:
            log(f"    {r[0]:<34} hits your king zone 20 plies on {r[3]:.0%} (its kind {r[4]:.0%}; n={r[5]})")
        out[f"{fam} | {colour}"] = rows
    json.dump(out, open(OUT / "repertoire.json", "w"), indent=1)
    log(f"[repertoire] {len(out)} opening groups in {time.perf_counter() - t0:.0f}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["heads", "engine", "repertoire", "report-engine"])
    ap.add_argument("--ckpt", default="data/derived/checkpoints/kn_pieces/best.pt")
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ply", type=int, default=16); ap.add_argument("--min-games", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "heads":
        heads(a.ckpt)
    elif a.cmd == "engine":
        engine_gauge(a.ckpt, n=a.n, seed=a.seed)
    elif a.cmd == "report-engine":
        report_engine(json.load(open(OUT / "engine.json")))
    else:
        repertoire(a.ckpt, ply=a.ply, min_games=a.min_games)


if __name__ == "__main__":
    main()
