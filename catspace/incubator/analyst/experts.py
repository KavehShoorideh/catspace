"""experts.py -- three experts and a gate that is blind to the board (Kaveh 2026-09-09: "train A, train B, keep the
rules-based score on the side, and let a small learned combiner decide; then we can see which one is used where").

Per position, every legal move gets three scores: A, the board-only network's policy (identity and square tokens,
no rule answers); B, the knowledge network's policy (the rules' answers as tokens); C, the reasoning's ranking
(rank.score_moves, no learning). Optionally the DISTANCE score of a network: the child position's distance to the
loss pole minus its distance to the win pole (from the mover's view), which needs one graph per legal move.

The gate sees only NAMED CONTEXT -- the position indicators (phase, material, queens, passed pawns, races, king
attackers, the ledgers' biggest threats, zones, the hottest temperature) plus whether the reasoning has anything
to say and how many moves it names -- never the board, so its three weights per position mean 'trust A / B / C
here'. Final score = sum_e w_e(context) * (a_e * s_e + b_e), softmax over legal moves, cross-entropy against the
engine's best move. Printed beside it: each expert alone, the per-position ORACLE (any expert right = the bound on
what any gate can reach), a fixed blend with no context, and the gate's mean weights by indicator.

    python -m catspace.incubator.analyst.experts collect --data data/derived/knowledge --a ckpt_a/best.pt --b ckpt_b/best.pt --out data/derived/experts --shard 0/4
    python -m catspace.incubator.analyst.experts fit --dir data/derived/experts
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .memo import Memo
from .graph import build, POSITION_QUESTIONS, NODE_QUESTIONS, EDGE_RELATIONS
from .rank import score_moves, move_kind
from . import knowledge_net as K

CONTEXT_KEYS = tuple(POSITION_QUESTIONS) + ("rules_silent", "rules_named", "rules_top", "n_legal", "best_is_quiet_unknown")


# ---------------------------------------------------------------- scoring one position by the networks
def net_sample(g, board: chess.Board, buckets: K.Buckets, pos_buckets: K.Buckets):
    nm = g.node_matrix(); n = nm.shape[0]
    nodes = np.zeros((K.MAX_PIECES, K.Q), np.float32); nodes[:n] = nm
    mask = np.zeros(K.MAX_PIECES, bool); mask[:n] = True
    et = g.edge_tensor(); dense = np.zeros((K.MAX_PIECES, K.MAX_PIECES, K.R), np.float32); dense[:n, :n] = et
    related = np.zeros((K.MAX_PIECES, K.MAX_PIECES), bool)
    for (i, j) in g.edges:
        related[i, j] = True
    squares = np.array([nd.square for nd in g.nodes] + [64] * (K.MAX_PIECES - n))
    types = np.array([nd.piece.piece_type if nd.piece.color == board.turn else 6 + nd.piece.piece_type for nd in g.nodes] + [0] * (K.MAX_PIECES - n))
    legal = np.zeros((K.MAX_PIECES, 64), bool); idx = {nd.square: i for i, nd in enumerate(g.nodes)}
    for mv in board.legal_moves:
        legal[idx[mv.from_square], mv.to_square] = True
    z = np.zeros(K.K_REG, np.float32)
    return dict(buckets=buckets.apply(nodes), mask=mask, edges=dense, related=related, squares=squares, types=types,
                pos_buckets=pos_buckets.apply(g.position_vector()[None])[0], legal=legal, best=-1, cp=0.0, result=-1,
                regime_now=z, ahead10=z, ahead20=z, ended10=0, ended20=0, game=0, ply=0, i=0), idx


class NetScorer:
    def __init__(self, ckpt: str):
        self.model, self.cfg, self.device = K.load_model(ckpt)
        meta = json.load(open(Path(ckpt).parent / "meta.json"))
        self.buckets = K.Buckets.from_state(self.cfg.buckets, meta["buckets"]); self.pos_buckets = K.Buckets.from_state(self.cfg.buckets, meta["pos_buckets"])

    @torch.no_grad()
    def policy(self, board: chess.Board, g) -> dict[chess.Move, float]:
        s, idx = net_sample(g, board, self.buckets, self.pos_buckets)
        b = K.collate([s], self.device)
        o = self.model(b); pl = self.model.policy_logits(o["h"], b)[0]
        lp = F.log_softmax(pl.view(-1), dim=0).view(K.MAX_PIECES, 64)
        return {mv: float(lp[idx[mv.from_square], mv.to_square]) for mv in board.legal_moves}

    @torch.no_grad()
    def basin(self, board: chess.Board, g) -> tuple[float, float, float]:
        """Distances from phi to the win / draw / loss poles, side to move's view."""
        s, _ = net_sample(g, board, self.buckets, self.pos_buckets)
        b = K.collate([s], self.device)
        o = self.model(b); _, d = self.model.basin_logits(o["phi"])
        return tuple(float(x) for x in d[0])

    def distance_scores(self, board: chess.Board, memo: Memo) -> dict[chess.Move, float]:
        """One child graph per legal move: the opponent's distance to their loss pole minus to their win pole."""
        out = {}
        for mv in board.legal_moves:
            nb = board.copy(stack=False); nb.push(mv)
            if nb.is_game_over():
                oc = nb.outcome(); out[mv] = 100.0 if oc and oc.winner == board.turn else (0.0 if oc and oc.winner is None else -100.0); continue
            g = build(nb, memo)
            w, d, l = self.basin(nb, g)
            out[mv] = w - l                                  # far from their win, near their loss: good for us
        return out


# ---------------------------------------------------------------- collection
def collect(data_dir: str, ckpt_a: str, ckpt_b: str, out_dir: str, shard: str = "0/1", split: str = "test", max_n: int = 2000,
            distances: bool = False, budget: int = 120, judge: int = 6, log=print):
    from .engine import Engine
    i, k = (int(x) for x in shard.split("/"))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    data = K.Data(data_dir)
    ids = [int(x) for x in data.ids(split)][:max_n][i::k]
    A, B = NetScorer(ckpt_a), NetScorer(ckpt_b)
    eng = Engine(depth=12, cache_path=str(out / f"engine_{i}.sqlite")); memo = Memo()
    rows = []; t0 = time.perf_counter()
    for n, idx in enumerate(ids, 1):
        m = data.meta[idx]
        board = chess.Board(m["fen"])
        if not m["best"]:
            continue
        try:
            g = build(board, memo)
            sa, sb = A.policy(board, g), B.policy(board, g)
            sc = score_moves(board, memo, eng, budget=budget, engine_depth=judge)
        except Exception as ex:
            log(f"[experts] {idx} error {ex}", flush=True); continue
        legal = list(board.legal_moves)
        row = dict(i=idx, fen=m["fen"], best=m["best"], best_kind=m["best_kind"], legal=[mv.uci() for mv in legal],
                   a=[sa[mv] for mv in legal], b=[sb[mv] for mv in legal], c=[float(sc.get(mv, (0.0, ""))[0]) for mv in legal],
                   c_why=[sc.get(mv, (0.0, ""))[1] for mv in legal], position={q: float(v) for q, v in g.position.items()})
        if distances:
            da, db = A.distance_scores(board, memo), B.distance_scores(board, memo)
            row["a_dist"] = [da[mv] for mv in legal]; row["b_dist"] = [db[mv] for mv in legal]
        rows.append(row)
        if n % 10 == 0:
            log(f"[experts {shard}] {n}/{len(ids)} positions, {time.perf_counter() - t0:.0f}s", flush=True)
    eng.close()
    with open(out / f"rows_{i}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"[experts {shard}] done {len(rows)} positions in {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- the gate
def context_of(row: dict) -> np.ndarray:
    c = [row["position"].get(q, 0.0) for q in POSITION_QUESTIONS]
    cs = np.array(row["c"]); named = int((cs > 0).sum())
    c += [float(named == 0), float(named), float(cs.max()) if len(cs) else 0.0, float(len(row["legal"])), 0.0]
    return np.array(c, dtype=np.float32)


class Gate(nn.Module):
    def __init__(self, n_ctx: int, n_experts: int = 3, hidden: int = 32, fixed: bool = False):
        super().__init__()
        self.fixed = fixed
        self.calib = nn.Parameter(torch.tensor([[1.0, 0.0]] * n_experts))          # a_e, b_e
        self.w0 = nn.Parameter(torch.zeros(n_experts))
        self.mlp = nn.Sequential(nn.Linear(n_ctx, hidden), nn.ReLU(), nn.Linear(hidden, n_experts)) if not fixed else None

    def weights(self, ctx):
        if self.fixed:
            return F.softmax(self.w0, dim=0)[None].expand(ctx.shape[0], -1)
        return F.softmax(self.w0[None] + self.mlp(ctx), dim=-1)

    def forward(self, ctx, scores, mask):
        """ctx (B, C); scores (B, E, M) per expert per move; mask (B, M) legal. -> logits (B, M)"""
        w = self.weights(ctx)                                                       # (B, E)
        cal = self.calib[None, :, 0, None] * scores + self.calib[None, :, 1, None]
        logits = (w[..., None] * cal).sum(1)
        return logits.masked_fill(~mask, float("-inf")), w


def _tensors(rows, experts=("a", "b", "c"), ctx_mean=None, ctx_std=None):
    M = max(len(r["legal"]) for r in rows)
    S = np.zeros((len(rows), len(experts), M), np.float32); mask = np.zeros((len(rows), M), bool); y = np.zeros(len(rows), np.int64)
    ctx = np.stack([context_of(r) for r in rows])
    for n, r in enumerate(rows):
        for e, key in enumerate(experts):
            v = np.array(r[key], np.float32)
            if key == "c":
                v = v / (np.abs(v).max() + 1.0)                                    # the reasoning's scale is pawns (mates 1000): squash per position
            S[n, e, :len(v)] = v
        mask[n, :len(r["legal"])] = True
        y[n] = r["legal"].index(r["best"])
    if ctx_mean is None:
        ctx_mean, ctx_std = ctx.mean(0), ctx.std(0) + 1e-6
    ctx = (ctx - ctx_mean) / ctx_std
    return torch.tensor(S), torch.tensor(mask), torch.tensor(y), torch.tensor(ctx), ctx_mean, ctx_std


def fit(dir_: str, experts=("a", "b", "c"), epochs: int = 300, seed: int = 0, log=print):
    rows = []
    for f in sorted(Path(dir_).glob("rows_*.jsonl")):
        rows.extend(json.loads(l) for l in open(f))
    rng = random.Random(seed); rng.shuffle(rows)
    n = len(rows); cut = int(n * 0.7)
    train, test = rows[:cut], rows[cut:]
    S, mask, y, ctx, mu, sd = _tensors(train, experts)
    St, maskt, yt, ctxt, _, _ = _tensors(test, experts, mu, sd)
    out = [f"[experts] {n} positions ({len(train)} fit, {len(test)} held out); experts {experts}"]
    # single experts and the oracle
    def top1(scores, mask_, y_):
        s = scores.masked_fill(~mask_, float("-inf")); return (s.argmax(-1) == y_)
    hits = {e: top1(St[:, k], maskt, yt) for k, e in enumerate(experts)}
    quiet = torch.tensor([r["best_kind"] == "quiet" for r in test]); capt = torch.tensor([r["best_kind"] == "capture" for r in test])
    def pct(v, sel=None):
        v = v if sel is None else v[sel]; return f"{v.float().mean():.1%} ({len(v)})"
    for e in experts:
        out.append(f"  {e} alone: all {pct(hits[e])} | quiet {pct(hits[e], quiet)} | capture {pct(hits[e], capt)}")
    oracle = torch.stack([hits[e] for e in experts]).any(0)
    out.append(f"  ORACLE (any expert right): all {pct(oracle)} | quiet {pct(oracle, quiet)} | capture {pct(oracle, capt)}")
    agree = torch.stack([St[:, k].masked_fill(~maskt, float('-inf')).argmax(-1) for k in range(len(experts))])
    out.append(f"  all experts agree on the move: {(agree == agree[0]).all(0).float().mean():.1%}; when they agree it is right {hits[experts[0]][(agree == agree[0]).all(0)].float().mean():.1%}")
    results = {}
    for name, fixed in (("fixed blend", True), ("context gate", False)):
        torch.manual_seed(seed)
        gate = Gate(ctx.shape[1], len(experts), fixed=fixed)
        opt = torch.optim.Adam(gate.parameters(), lr=1e-2, weight_decay=1e-3)
        for ep in range(epochs):
            logits, w = gate(ctx, S, mask); loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            logits, w = gate(ctxt, St, maskt); h = logits.argmax(-1) == yt
            results[name] = (h, w, gate)
        out.append(f"  {name}: all {pct(h)} | quiet {pct(h, quiet)} | capture {pct(h, capt)}" + (f"; weights {[round(float(x), 2) for x in gate.weights(ctxt)[0]]}" if fixed else ""))
    # the scenario table: mean gate weights by indicator
    h, w, gate = results["context gate"]
    out.append(f"  gate mean weights on held-out: " + ", ".join(f"{e} {float(w[:, k].mean()):.2f}" for k, e in enumerate(experts)))
    out.append("  scenario table (mean weight per expert, held-out positions grouped by a named indicator):")
    groups = {
        "reasoning silent": lambda r: (np.array(r["c"]) > 0).sum() == 0, "reasoning names a move": lambda r: (np.array(r["c"]) > 0).sum() > 0,
        "in check": lambda r: r["position"]["in_check"] > 0, "queens off": lambda r: r["position"]["queens_on"] == 0, "queens on": lambda r: r["position"]["queens_on"] > 0,
        "phase >= 40": lambda r: r["position"]["phase"] >= 40, "phase < 20 (ending)": lambda r: r["position"]["phase"] < 20,
        "a passed pawn (either side)": lambda r: r["position"]["passed_ours"] + r["position"]["passed_theirs"] > 0,
        "a piece falls (either side)": lambda r: r["position"]["falls_ours"] + r["position"]["falls_theirs"] > 0,
        "hottest zone > 0": lambda r: r["position"]["hottest"] > 0, "no zone hot": lambda r: r["position"]["hottest"] == 0,
        "best move is quiet": lambda r: r["best_kind"] == "quiet", "best move is a capture": lambda r: r["best_kind"] == "capture",
    }
    for gname, fn in groups.items():
        sel = torch.tensor([bool(fn(r)) for r in test])
        if sel.sum() < 10:
            continue
        out.append(f"    {gname:<30} n={int(sel.sum()):<5} " + " ".join(f"{e} {float(w[sel, k].mean()):.2f}" for k, e in enumerate(experts)) + f"  | gate right {h[sel].float().mean():.1%}, oracle {oracle[sel].float().mean():.1%}")
    text = "\n".join(out); log(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "fit"]); ap.add_argument("--data", default="data/derived/knowledge"); ap.add_argument("--a", default=""); ap.add_argument("--b", default="")
    ap.add_argument("--out", default="data/derived/experts"); ap.add_argument("--dir", default="data/derived/experts"); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--split", default="test"); ap.add_argument("--max-n", type=int, default=2000); ap.add_argument("--distances", action="store_true")
    ap.add_argument("--experts", default="a,b,c")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(a.data, a.a, a.b, a.out, a.shard, a.split, a.max_n, a.distances)
    else:
        fit(a.dir, tuple(a.experts.split(",")))


if __name__ == "__main__":
    main()
