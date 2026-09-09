"""knowledge_net.py -- a transformer over the interaction graph whose inputs are the rules' ANSWERS and whose output
is a sum of NAMED concept vectors (Kaveh 2026-09-09: "the questions and answers come from our rules-based system;
the token for the bishop came out at this angle, and that angle is the sum of these three concepts").

Tokens are pieces (plus one board token). A piece token is the SUM of one learned embedding per answer bucket (43
questions, quantile buckets fixed from the training split) plus its identity and square -- so the input side is
exactly decomposable by question. Pair relations enter as an additive per-head attention bias (Graphormer's edge
encoding; what Leela's nets learn as 'smolgen' is computed here from the rules). The output is built the Backpack
way: a weight per (piece, question) from the transformer, times a SENSE vector per answer bucket, summed -- the
representation phi is linear in named senses, so 'why is this position where it is' reads off as a list of
(piece, question, answer, weight). Heads on phi: the W/D/L basin as distances to three poles under an interval
quasimetric (the geometry is the committor), the engine's evaluation, the regime 10 and 20 plies ahead; the best
move as a from-piece x to-square head on the tokens. The trajectory loss calibrates d(phi_t -> phi_{t+k}) to k plies
against cross-game negatives; NOTHING in the loss says pawn moves are special (the asymmetry test is honest).

Variants (the same code path, a switch): full = answers + identity + square; board = identity + square only (the
board-only control, approach A); rules = answers only (are the squares needed at all?).

    python -m catspace.incubator.analyst.knowledge_net train --data data/derived/knowledge --out data/derived/checkpoints/kn_full --variant full --epochs 20
    python -m catspace.incubator.analyst.knowledge_net eval  --ckpt data/derived/checkpoints/kn_full/last.pt --data data/derived/knowledge
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import NODE_QUESTIONS, EDGE_RELATIONS, POSITION_QUESTIONS
from .dataset import load as load_dataset, REGIME_KEYS, MAX_PIECES, regime_of
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE

Q = len(NODE_QUESTIONS); R = len(EDGE_RELATIONS); P = len(POSITION_QUESTIONS); K_REG = len(REGIME_KEYS)
N_TOK = MAX_PIECES + 1                      # pieces + the board token (index 0)
DIST_COL = EDGE_RELATIONS.index("distance")


@dataclass
class Config:
    d: int = 128
    layers: int = 6
    heads: int = 8
    buckets: int = 12
    variant: str = "full"                   # full | board | rules
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.05
    batch: int = 128                        # anchors per batch (each brings a partner k plies ahead: 2x tokens)
    epochs: int = 20
    k_max: int = 20
    neg_margin: float = 30.0
    w_cp: float = 1.0
    w_regime: float = 1.0
    w_policy: float = 1.0
    w_traj: float = 0.5
    seed: int = 1


# ---------------------------------------------------------------- buckets: quantile edges per question, fixed from the training split
class Buckets:
    def __init__(self, n: int):
        self.n = n; self.edges = None      # (Q, n-1)

    def fit(self, values: np.ndarray, mask: np.ndarray):
        """values (N, 32, Q) with mask (N, 32): per-question quantile edges over real pieces."""
        v = values[mask]                   # (M, Q)
        qs = np.linspace(0, 1, self.n + 1)[1:-1]
        self.edges = [np.unique(np.quantile(v[:, q], qs)) for q in range(v.shape[1])]
        return self

    def apply(self, values: np.ndarray) -> np.ndarray:
        """(..., Q) -> (..., Q) bucket ids in [0, n)."""
        out = np.zeros(values.shape, dtype=np.int64)
        for q, e in enumerate(self.edges):
            out[..., q] = np.searchsorted(e, values[..., q], side="right")
        return out

    def state(self):
        return [e.tolist() for e in self.edges]

    @classmethod
    def from_state(cls, n, st):
        b = cls(n); b.edges = [np.array(e) for e in st]; return b


# ---------------------------------------------------------------- the model
class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.heads = heads; self.drop = nn.Dropout(dropout)

    def forward(self, x, bias):
        """x (B, N, d); bias (B, H, N, N) additive, -inf on padding keys."""
        B, N, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, N, self.heads, -1).transpose(1, 2); k = k.view(B, N, self.heads, -1).transpose(1, 2); v = v.view(B, N, self.heads, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, N, d)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class KnowledgeNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg; d = cfg.d
        self.answer_emb = nn.Embedding(Q * cfg.buckets, d)            # input side: one embedding per (question, bucket)
        self.sense = nn.Embedding(Q * cfg.buckets, d)                 # output side: one sense vector per (question, bucket)
        self.type_emb = nn.Embedding(13, d); self.square_emb = nn.Embedding(65, d)   # identity: signed piece type (0 = pad), square (64 = pad)
        self.type_sense = nn.Embedding(13, d); self.square_sense = nn.Embedding(65, d)
        self.pos_emb = nn.Embedding(P * cfg.buckets, d)               # the board token: one embedding per (indicator, bucket)
        self.rel_bias = nn.Linear(R, cfg.heads, bias=False)           # pair relations -> per-head bias
        self.dist_bias = nn.Embedding(9, cfg.heads)                   # distance 0..7 (8 = unrelated)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, cfg.dropout) for _ in range(cfg.layers)])
        self.ln = nn.LayerNorm(d)
        self.alpha = nn.Linear(d, Q + 2)                              # per piece: a weight per question sense, plus identity and square senses
        self.iqe = IQE(d, components=16)
        self.poles = nn.Parameter(torch.randn(3, d) * 0.05)           # win / draw / loss, side-to-move view
        self.log_T = nn.Parameter(torch.zeros(()))
        self.cp_head = nn.Linear(d, 1)
        self.regime_head = nn.Linear(d, 2 * K_REG)
        self.pol_from = nn.Linear(d, d); self.pol_to = nn.Embedding(64, d)
        self.register_buffer("q_offsets", torch.arange(Q) * cfg.buckets)
        self.register_buffer("p_offsets", torch.arange(P) * cfg.buckets)

    def tokens(self, b):
        cfg = self.cfg
        answers = self.answer_emb(b["buckets"] + self.q_offsets).sum(2)                 # (B, 32, d)
        ident = self.type_emb(b["types"]) + self.square_emb(b["squares"])
        if cfg.variant == "board":
            x = ident
        elif cfg.variant == "rules":
            x = answers + self.type_emb(b["types"])                                     # identity without the square
        else:
            x = answers + ident
        board = self.pos_emb(b["pos_buckets"] + self.p_offsets).sum(1, keepdim=True)    # (B, 1, d)
        if cfg.variant == "board":
            board = torch.zeros_like(board)
        return torch.cat([board, x], dim=1)                                             # (B, 33, d)

    def bias(self, b):
        B = b["edges"].shape[0]
        rel = self.rel_bias(b["edges"])                                                 # (B, 32, 32, H)
        dist = b["edges"][..., DIST_COL].long().clamp(0, 8)
        dist = torch.where(b["related"], dist, torch.full_like(dist, 8))
        rel = rel + self.dist_bias(dist)
        rel = rel.permute(0, 3, 1, 2)                                                   # (B, H, 32, 32)
        if self.cfg.variant == "board":
            rel = torch.zeros_like(rel)
        full = torch.zeros(B, self.cfg.heads, N_TOK, N_TOK, device=rel.device, dtype=rel.dtype)
        full[:, :, 1:, 1:] = rel
        pad = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=rel.device), b["mask"]], dim=1)   # keys that exist
        full = full.masked_fill(~pad[:, None, None, :], float("-inf"))
        return full

    def forward(self, b):
        x = self.tokens(b); bias = self.bias(b)
        for blk in self.blocks:
            x = blk(x, bias)
        h = self.ln(x)                                                                  # (B, 33, d)
        pieces = h[:, 1:]
        a = F.softplus(self.alpha(pieces)) * b["mask"][..., None]                       # (B, 32, Q+2) >= 0, zero on padding
        senses = self.sense(b["buckets"] + self.q_offsets)                              # (B, 32, Q, d)
        ts, ss = self.type_sense(b["types"]), self.square_sense(b["squares"])           # (B, 32, d)
        if self.cfg.variant == "board":
            contrib = a[..., Q:Q + 1] * ts + a[..., Q + 1:Q + 2] * ss
        elif self.cfg.variant == "rules":
            contrib = (a[..., :Q, None] * senses).sum(2) + a[..., Q:Q + 1] * ts
        else:
            contrib = (a[..., :Q, None] * senses).sum(2) + a[..., Q:Q + 1] * ts + a[..., Q + 1:Q + 2] * ss
        norm = a.sum((1, 2))[:, None] + 1.0
        phi = contrib.sum(1) / norm                                                     # (B, d): linear in the senses
        return dict(h=h, phi=phi, alpha=a, norm=norm)

    def basin_logits(self, phi):
        B = phi.shape[0]
        u = phi[:, None, :].expand(B, 3, -1).reshape(B * 3, -1)
        v = self.poles[None].expand(B, 3, -1).reshape(B * 3, -1)
        d = self.iqe(u, v).reshape(B, 3)
        return -d / torch.exp(self.log_T), d

    def policy_logits(self, h, b):
        """(B, 32, 64): from-piece token x to-square, -inf where not legal."""
        f = self.pol_from(h[:, 1:])                                                     # (B, 32, d)
        logits = torch.einsum("bid,sd->bis", f, self.pol_to.weight) / math.sqrt(f.shape[-1])
        return logits.masked_fill(~b["legal"], float("-inf"))


# ---------------------------------------------------------------- data
class Data:
    """The dataset in memory, every per-sample array precomputed once (dense int8 relations, legal masks), so a batch
    is fancy indexing; partner lookup by (game, ply) for the trajectory pairs."""
    def __init__(self, path: str, buckets: Buckets | None = None, n_buckets: int = 12, split_seed: int = 0, log=print, meta_only: bool = False):
        t0 = time.perf_counter()
        d = load_dataset(path, meta_only=meta_only)
        self.meta = d["meta"]
        N = len(self.meta)
        games = sorted({m["game"] for m in self.meta})
        rng = random.Random(split_seed); rng.shuffle(games)
        n = len(games); test_games = set(games[: n // 10]); val_games = set(games[n // 10: n // 5])
        self.split = np.array(["test" if m["game"] in test_games else "val" if m["game"] in val_games else "train" for m in self.meta])
        self.index = {(m["game"], m["ply"]): i for i, m in enumerate(self.meta)}
        if meta_only:
            return
        self.nodes, self.mask, self.position, edges = d["nodes"], d["mask"], d["position"], d["edges"]
        tr = self.split == "train"
        self.buckets = buckets or Buckets(n_buckets).fit(self.nodes[tr], self.mask[tr])
        self.node_b = self.buckets.apply(self.nodes).astype(np.int8)                     # (N, 32, Q)
        pb = Buckets(n_buckets); pb.fit(self.position[tr][:, None, :], np.ones((int(tr.sum()), 1), bool))
        self.pos_buckets_fn = pb; self.pos_b = pb.apply(self.position).astype(np.int8)
        # dense relations as int8 (values are small integers: flags, distances 0-7, a value beyond a line clipped)
        self.edges = np.zeros((N, MAX_PIECES, MAX_PIECES, R), dtype=np.int8)
        self.related = np.zeros((N, MAX_PIECES, MAX_PIECES), dtype=bool)
        for i, e in enumerate(edges):
            if len(e):
                a, b_ = e[:, 0].astype(int), e[:, 1].astype(int)
                self.edges[i, a, b_] = np.clip(e[:, 2:].astype(np.float32), -127, 127).astype(np.int8); self.related[i, a, b_] = True
        self.squares = np.array([m["squares"] for m in self.meta], dtype=np.int64)
        self.types = np.array([[t if t > 0 else (6 - t if t < 0 else 0) for t in m["types"]] for m in self.meta], dtype=np.int64)   # 0 pad, 1-6 ours, 7-12 theirs
        self.legal = np.zeros((N, MAX_PIECES, 64), dtype=bool); self.best = np.full(N, -1, dtype=np.int64)
        for i, m in enumerate(self.meta):
            board = chess.Board(m["fen"]); sq_index = {s: j for j, s in enumerate(m["squares"]) if s < 64}
            for mv in board.legal_moves:
                self.legal[i, sq_index[mv.from_square], mv.to_square] = True
            if m["best"]:
                bm = chess.Move.from_uci(m["best"]); self.best[i] = sq_index[bm.from_square] * 64 + bm.to_square
        reg = np.array([regime_of(chess.Board(m["fen"]), bool(m["side"])) for m in self.meta], dtype=np.float32)
        self.reg_mean = reg[tr].mean(0); self.reg_std = reg[tr].std(0) + 1e-6
        self.regime_now = (reg - self.reg_mean) / self.reg_std
        self.ahead = {h: (np.array([m["ahead"][str(h)] for m in self.meta], dtype=np.float32) - self.reg_mean) / self.reg_std for h in (10, 20)}
        cps = np.array([m["best_cp"] if m["best_cp"] is not None else 0 for m in self.meta], dtype=np.float32)
        self.cp = np.tanh(cps / 400.0)
        self.result = np.array([-1 if m["result"] is None else int(round(m["result"] * 2)) for m in self.meta])     # 0 loss, 1 draw, 2 win, -1 unknown
        self.game = np.array([m["game"] for m in self.meta]); self.ply = np.array([m["ply"] for m in self.meta])
        self.ended = {h: np.array([m["ended_within"][str(h)] for m in self.meta]) for h in (10, 20)}
        log(f"[kn] data ready in {time.perf_counter() - t0:.0f}s: {N} positions, relations {self.edges.nbytes / 1e9:.2f} GB", flush=True)

    def ids(self, split):
        return np.nonzero(self.split == split)[0]

    def batch(self, idx, device) -> dict:
        idx = np.asarray(idx)
        out = dict(buckets=torch.as_tensor(self.node_b[idx]).long(), mask=torch.as_tensor(self.mask[idx]),
                   edges=torch.as_tensor(self.edges[idx]).float(), related=torch.as_tensor(self.related[idx]),
                   squares=torch.as_tensor(self.squares[idx]).clamp(0, 64), types=torch.as_tensor(self.types[idx]),
                   pos_buckets=torch.as_tensor(self.pos_b[idx]).long(), legal=torch.as_tensor(self.legal[idx]), best=torch.as_tensor(self.best[idx]),
                   cp=torch.as_tensor(self.cp[idx]), result=torch.as_tensor(self.result[idx]), regime_now=torch.as_tensor(self.regime_now[idx]),
                   ahead10=torch.as_tensor(self.ahead[10][idx]), ahead20=torch.as_tensor(self.ahead[20][idx]),
                   ended10=torch.as_tensor(self.ended[10][idx]), ended20=torch.as_tensor(self.ended[20][idx]),
                   game=torch.as_tensor(self.game[idx]), ply=torch.as_tensor(self.ply[idx]), i=torch.as_tensor(idx))
        return {k: v.to(device) for k, v in out.items()}

    def partner(self, i: int, k_max: int, rng: random.Random):
        """A later position of the same game, k plies ahead (k in 1..k_max), when present."""
        g, p = int(self.game[i]), int(self.ply[i])
        ks = list(range(1, k_max + 1)); rng.shuffle(ks)
        for k in ks:
            j = self.index.get((g, p + k))
            if j is not None:
                return j, k
        return None, 0


def collate(samples: list[dict], device) -> dict:
    out = {}
    for key in ("buckets", "mask", "edges", "related", "squares", "types", "pos_buckets", "legal", "regime_now", "ahead10", "ahead20"):
        arr = np.stack([s[key] for s in samples])
        out[key] = torch.as_tensor(arr).to(device)
    out["types"] = out["types"].long(); out["squares"] = out["squares"].long().clamp(0, 64); out["buckets"] = out["buckets"].long(); out["pos_buckets"] = out["pos_buckets"].long()
    for key in ("best", "result", "ended10", "ended20", "game", "ply", "i"):
        out[key] = torch.as_tensor(np.array([s[key] for s in samples])).to(device)
    out["cp"] = torch.as_tensor(np.array([s["cp"] for s in samples], dtype=np.float32)).to(device)
    return out


# ---------------------------------------------------------------- losses (masked, tested)
def losses(model: KnowledgeNet, b: dict, cfg: Config, k: torch.Tensor | None = None, n_anchor: int = 0) -> dict:
    o = model(b)
    phi, h = o["phi"], o["h"]
    out = {}
    logits, dpoles = model.basin_logits(phi)
    known = b["result"] >= 0
    out["wdl"] = F.cross_entropy(logits[known], b["result"][known]) if known.any() else phi.sum() * 0
    out["cp"] = F.mse_loss(model.cp_head(phi).squeeze(-1), b["cp"])
    reg = model.regime_head(phi).view(-1, 2, K_REG)
    out["regime"] = 0.5 * (F.mse_loss(reg[:, 0], b["ahead10"]) + F.mse_loss(reg[:, 1], b["ahead20"]))
    pl = model.policy_logits(h, b).view(phi.shape[0], -1)
    has = b["best"] >= 0
    out["policy"] = F.cross_entropy(pl[has], b["best"][has]) if has.any() else phi.sum() * 0
    if k is not None and n_anchor > 0:
        a, p = phi[:n_anchor], phi[n_anchor:2 * n_anchor]
        d_fwd = model.iqe(a, p)
        out["traj_fwd"] = F.smooth_l1_loss(d_fwd, k.float())
        dm = model.iqe.pairwise(a, phi)                                                  # (n, B)
        games = b["game"]
        other = games[:n_anchor, None] != games[None, :]
        out["traj_neg"] = (F.relu(cfg.neg_margin - dm) ** 2)[other].mean() / cfg.neg_margin if other.any() else phi.sum() * 0
    total = out["wdl"] + cfg.w_cp * out["cp"] + cfg.w_regime * out["regime"] + cfg.w_policy * out["policy"]
    if "traj_fwd" in out:
        total = total + cfg.w_traj * (out["traj_fwd"] + out["traj_neg"])
    out["total"] = total
    out["_phi"] = phi; out["_logits"] = logits; out["_policy"] = pl; out["_regime"] = reg
    return out


# ---------------------------------------------------------------- evaluation with degenerate baselines
@torch.no_grad()
def evaluate(model: KnowledgeNet, data: Data, split: str, cfg: Config, device, max_n: int = 4000) -> dict:
    model.eval()
    ids = data.ids(split)[:max_n]
    n = len(ids); stats = dict(n=n)
    wdl_ok = wdl_n = 0; maj = np.bincount(data.result[data.ids("train")][data.result[data.ids("train")] >= 0], minlength=3).argmax()
    wdl_maj = 0
    pol_ok = pol_n = 0; quiet_ok = quiet_n = 0; cap_ok = cap_n = 0
    cp_pred, cp_true = [], []
    reg_pred = {10: [], 20: []}; reg_true = {10: [], 20: []}; reg_now = []
    phis = []
    for s in range(0, n, 256):
        chunk = ids[s:s + 256]
        b = data.batch(chunk, device)
        o = model(b); logits, _ = model.basin_logits(o["phi"])
        known = b["result"] >= 0
        wdl_ok += (logits.argmax(-1)[known] == b["result"][known]).sum().item(); wdl_n += known.sum().item()
        wdl_maj += (b["result"][known] == maj).sum().item()
        pl = model.policy_logits(o["h"], b).view(len(chunk), -1)
        pred = pl.argmax(-1); has = b["best"] >= 0
        hit = (pred == b["best"]) & has
        pol_ok += hit.sum().item(); pol_n += has.sum().item()
        for j, ii in enumerate(chunk):
            kind = data.meta[int(ii)]["best_kind"]
            if kind == "quiet":
                quiet_n += 1; quiet_ok += int(hit[j].item())
            elif kind == "capture":
                cap_n += 1; cap_ok += int(hit[j].item())
        cp_pred.append(model.cp_head(o["phi"]).squeeze(-1).cpu().numpy()); cp_true.append(b["cp"].cpu().numpy())
        reg = model.regime_head(o["phi"]).view(len(chunk), 2, K_REG).cpu().numpy()
        reg_pred[10].append(reg[:, 0]); reg_pred[20].append(reg[:, 1])
        reg_true[10].append(b["ahead10"].cpu().numpy()); reg_true[20].append(b["ahead20"].cpu().numpy()); reg_now.append(b["regime_now"].cpu().numpy())
        phis.append(o["phi"].cpu().numpy())
    stats["wdl_acc"] = wdl_ok / max(1, wdl_n); stats["wdl_majority"] = wdl_maj / max(1, wdl_n); stats["wdl_n"] = wdl_n
    stats["policy_top1"] = pol_ok / max(1, pol_n); stats["policy_quiet"] = quiet_ok / max(1, quiet_n); stats["policy_capture"] = cap_ok / max(1, cap_n); stats["quiet_n"] = quiet_n
    cp_pred, cp_true = np.concatenate(cp_pred), np.concatenate(cp_true)
    stats["cp_r2"] = float(1 - ((cp_pred - cp_true) ** 2).mean() / max(1e-9, cp_true.var()))
    now = np.concatenate(reg_now)
    for hz in (10, 20):
        pr, tr = np.concatenate(reg_pred[hz]), np.concatenate(reg_true[hz])
        var = tr.var(0); live = var > 1e-6                                              # keys with no variance in this split are not scored
        var = np.where(live, var, 1.0)
        r2 = 1 - ((pr - tr) ** 2).mean(0) / var; pr2 = 1 - ((now - tr) ** 2).mean(0) / var
        stats[f"regime{hz}_r2"] = float(r2[live].mean()) if live.any() else float("nan")
        stats[f"regime{hz}_persist_r2"] = float(pr2[live].mean()) if live.any() else float("nan")
        stats[f"regime{hz}_r2_by_key"] = {k: (round(float(r2[j]), 3) if live[j] else None) for j, k in enumerate(REGIME_KEYS)}
        stats[f"regime{hz}_persist_by_key"] = {k: (round(float(pr2[j]), 3) if live[j] else None) for j, k in enumerate(REGIME_KEYS)}
    phi = np.concatenate(phis)
    stats["effective_rank"] = effective_rank(phi)
    stats["phi_std"] = float(phi.std())
    model.train()
    return stats


def effective_rank(x: np.ndarray, boots: int = 5, seed: int = 0) -> float:
    rng = np.random.default_rng(seed); out = []
    for _ in range(boots):
        s = x[rng.choice(len(x), min(len(x), 2000), replace=False)]
        s = s - s.mean(0)
        sv = np.linalg.svd(s, compute_uv=False); p = sv ** 2 / (sv ** 2).sum()
        out.append(float(np.exp(-(p * np.log(p + 1e-12)).sum())))
    return float(np.mean(out))


@torch.no_grad()
def asymmetry(model: KnowledgeNet, data: Data, split: str, device, max_n: int = 3000) -> dict:
    """The pawn test: over consecutive positions (t, t+1) of held-out games, d(phi_t -> phi_t+1) against
    d(phi_t+1 -> phi_t), split by what the move was: a pawn move, a capture, a piece move. Nothing in training named
    pawns; if the reverse distance is larger after pawn moves and not after piece moves, the geometry found the arrow."""
    model.eval()
    ids = data.ids(split)
    pairs = []
    for i in ids:
        m = data.meta[i]; j = data.index.get((m["game"], m["ply"] + 1))
        if j is not None:
            pairs.append((i, j))
        if len(pairs) >= max_n:
            break
    out = {}
    rows = {"pawn": [], "capture": [], "piece": []}
    for s in range(0, len(pairs), 128):
        chunk = pairs[s:s + 128]
        ba = data.batch([i for i, _ in chunk], device); bb = data.batch([j for _, j in chunk], device)
        pa, pb = model(ba)["phi"], model(bb)["phi"]
        fwd = model.iqe(pa, pb).cpu().numpy(); rev = model.iqe(pb, pa).cpu().numpy()
        for (i, j), f, r in zip(chunk, fwd, rev):
            m = data.meta[i]; board = chess.Board(m["fen"]); mv = chess.Move.from_uci(m["played"])
            kind = "capture" if board.is_capture(mv) else "pawn" if board.piece_at(mv.from_square).piece_type == chess.PAWN else "piece"
            rows[kind].append((f, r))
    for kind, rs in rows.items():
        if rs:
            f = np.array([x[0] for x in rs]); r = np.array([x[1] for x in rs])
            out[kind] = dict(n=len(rs), fwd_median=float(np.median(f)), rev_median=float(np.median(r)), rev_gt_fwd=float((r > f).mean()), ratio_median=float(np.median((r + 1e-6) / (f + 1e-6))))
    model.train()
    return out


# ---------------------------------------------------------------- training
def train(args):
    cfg = Config(**{k: v for k, v in vars(args).items() if k in Config.__dataclass_fields__})
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    data = Data(args.data, n_buckets=cfg.buckets)
    print(f"[kn] data: {len(data.meta)} positions; train {len(data.ids('train'))} val {len(data.ids('val'))} test {len(data.ids('test'))}; loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    model = KnowledgeNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[kn] model {cfg.variant}: {n_params / 1e6:.2f}M params, d={cfg.d} layers={cfg.layers} heads={cfg.heads} buckets={cfg.buckets}; device {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_ids = data.ids("train"); rng = random.Random(cfg.seed)
    steps_per_epoch = len(train_ids) // cfg.batch
    total_steps = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(1, total_steps), pct_start=0.05)
    log = open(out / "steps.jsonl", "a")
    json.dump(dict(config=asdict(cfg), buckets=data.buckets.state(), pos_buckets=data.pos_buckets_fn.state(), reg_mean=data.reg_mean.tolist(), reg_std=data.reg_std.tolist(), n_params=n_params), open(out / "meta.json", "w"))
    step = 0; best_val = None
    for epoch in range(cfg.epochs):
        ids = list(train_ids); rng.shuffle(ids)
        acc = {}; te = time.perf_counter()
        for s in range(0, len(ids) - cfg.batch + 1, cfg.batch):
            anchors = ids[s:s + cfg.batch]
            # anchors that have a partner come first, aligned with their partners; the rest follow
            a_ids = [int(i) for i in anchors]
            pairs = [(int(i), data.partner(int(i), cfg.k_max, rng)) for i in anchors]
            pairs = [(i, j, k) for i, (j, k) in pairs if j is not None]
            n_anchor = len(pairs)
            order = [i for i, _, _ in pairs] + [j for _, j, _ in pairs] + [i for i in a_ids if i not in {p[0] for p in pairs}]
            b = data.batch(order, device)
            k_t = torch.tensor([k for _, _, k in pairs], device=device)
            L = losses(model, b, cfg, k_t, n_anchor)
            opt.zero_grad(set_to_none=True); L["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            for key, v in L.items():
                if not key.startswith("_"):
                    acc[key] = acc.get(key, 0.0) + float(v)
            if step % 50 == 0:
                row = dict(step=step, epoch=epoch, **{k: v / 50 for k, v in acc.items()}, lr=sched.get_last_lr()[0], t=time.perf_counter() - t0)
                log.write(json.dumps(row) + "\n"); log.flush()
                print(f"[kn] step {step}/{total_steps} " + " ".join(f"{k} {v / 50:.3f}" for k, v in acc.items()) + f" ({time.perf_counter() - te:.0f}s this epoch)", flush=True)
                acc = {}
        ev = evaluate(model, data, "val", cfg, device, max_n=2000)
        line = dict(step=step, epoch=epoch, split="val", **{k: v for k, v in ev.items() if not isinstance(v, dict)})
        log.write(json.dumps(line) + "\n"); log.flush()
        print(f"[kn] epoch {epoch} val: wdl {ev['wdl_acc']:.3f} (majority {ev['wdl_majority']:.3f}) policy {ev['policy_top1']:.3f} quiet {ev['policy_quiet']:.3f} capture {ev['policy_capture']:.3f} "
              f"cp_r2 {ev['cp_r2']:.3f} regime10 {ev['regime10_r2']:.3f} (persist {ev['regime10_persist_r2']:.3f}) regime20 {ev['regime20_r2']:.3f} (persist {ev['regime20_persist_r2']:.3f}) "
              f"eff_rank {ev['effective_rank']:.1f} phi_std {ev['phi_std']:.3f} | {time.perf_counter() - t0:.0f}s", flush=True)
        torch.save(dict(model=model.state_dict(), config=asdict(cfg), epoch=epoch, step=step), out / f"epoch_{epoch:03d}.pt")
        torch.save(dict(model=model.state_dict(), config=asdict(cfg), epoch=epoch, step=step), out / "last.pt")
        if best_val is None or ev["policy_top1"] > best_val:
            best_val = ev["policy_top1"]; torch.save(dict(model=model.state_dict(), config=asdict(cfg), epoch=epoch, step=step), out / "best.pt")
    ev = evaluate(model, data, "test", cfg, device)
    asym = asymmetry(model, data, "test", device)
    json.dump(dict(test=ev, asymmetry=asym), open(out / "final.json", "w"), indent=1)
    print("[kn] TEST " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items() if not isinstance(v, dict)}), flush=True)
    print("[kn] regime10 by key: " + json.dumps(ev["regime10_r2_by_key"]) + "\n[kn] persistence10 by key: " + json.dumps(ev["regime10_persist_by_key"]), flush=True)
    print("[kn] regime20 by key: " + json.dumps(ev["regime20_r2_by_key"]) + "\n[kn] persistence20 by key: " + json.dumps(ev["regime20_persist_by_key"]), flush=True)
    print("[kn] ASYMMETRY " + json.dumps(asym), flush=True)
    return model, data


# ---------------------------------------------------------------- the readout: why is this position where it is
def explain(model: KnowledgeNet, b: dict, meta_row: dict | None = None, top: int = 10, pole: int = 0) -> list[tuple[str, str, float, float]]:
    """The Backpack readout for the first sample of the batch: phi is a sum of (piece, question) contributions
    alpha * sense / norm, and the distance to a pole is piecewise linear in phi, so the gradient of the distance
    dotted with each contribution is that contribution's exact local share. Returns (piece, question=answer,
    weight alpha, share of the distance in plies), largest shares first (negative = pulls the position toward the
    pole: toward the win for pole 0)."""
    model.eval()
    o = model(b)
    phi = o["phi"][:1].detach().requires_grad_(True)
    _, d = model.basin_logits(phi)
    d[0, pole].backward()
    grad = phi.grad[0].detach()                                                         # (d,)
    a = o["alpha"][0].detach(); senses = model.sense(b["buckets"][:1] + model.q_offsets)[0].detach()     # (32, Q+2), (32, Q, d)
    norm = o["norm"][0, 0].detach()
    rows = []
    n_pieces = int(b["mask"][0].sum())
    for i in range(n_pieces):
        sq = int(b["squares"][0, i]); t = int(b["types"][0, i])
        name = f"{'our' if t <= 6 else 'their'} {chess.piece_name(t if t <= 6 else t - 6)} on {chess.square_name(sq)}"
        for q in range(Q):
            share = float((a[i, q] * senses[i, q] / norm) @ grad)
            if a[i, q] > 0:
                rows.append((name, f"{NODE_QUESTIONS[q]}={b['buckets'][0, i, q].item()}", float(a[i, q]), share))
        rows.append((name, "identity", float(a[i, Q]), float((a[i, Q] * model.type_sense(b["types"][:1, i])[0].detach() / norm) @ grad)))
        rows.append((name, "square", float(a[i, Q + 1]), float((a[i, Q + 1] * model.square_sense(b["squares"][:1, i])[0].detach() / norm) @ grad)))
    rows.sort(key=lambda r: -abs(r[3]))
    return rows[:top]


def load_model(ckpt_path: str, device=None):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = Config(**ck["config"]); model = KnowledgeNet(cfg); model.load_state_dict(ck["model"])
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return model.to(device).eval(), cfg, device


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"]); ap.add_argument("--data", default="data/derived/knowledge"); ap.add_argument("--out", default="data/derived/checkpoints/kn")
    ap.add_argument("--ckpt", default="")
    for f in Config.__dataclass_fields__.values():
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        train(a)
    else:
        model, cfg, device = load_model(a.ckpt)
        meta = json.load(open(Path(a.ckpt).parent / "meta.json"))
        data = Data(a.data, buckets=Buckets.from_state(cfg.buckets, meta["buckets"]), n_buckets=cfg.buckets)
        ev = evaluate(model, data, "test", cfg, device); print(json.dumps({k: v for k, v in ev.items()}, indent=1))
        print(json.dumps(asymmetry(model, data, "test", device), indent=1))


if __name__ == "__main__":
    main()
