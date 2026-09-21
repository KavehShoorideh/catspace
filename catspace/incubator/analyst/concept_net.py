"""THE CONCEPT-TOKEN NETWORK (Kaveh 2026-09-20): a shallow transformer whose tokens are the board's pieces AND the
instances of the registry's concepts (concepts.py), whose relations between tokens are the rules' (a piece is an
argument of an instance; two instances share a piece; a piece attacks / defends another), and whose decision is a
SUM over tokens: the score of a move is the sum of per-token contributions, so the attribution of a decision to the
concepts is exact. The board is seen (the piece tokens); the network is shallow, so the board must spend layers
to rebuild a concept the registry lacks -- the residual the proposal loop feeds on.
Target: the PARTITION of legal moves into result-keeping and result-losing (the tablebase), uniform over the
keepers -- which losing move is more losing is not asked. Gauges: the chosen move keeps the result (beside random
legal and beside the same network without concept tokens), the explanation's length (tokens carrying 80% of the
score), faithfulness under intervention (drop the top concept token: does the move change as the attribution
said), and the residual by material and by the concepts present.
    prepare  -- the instances per position (parallel shards) -> data/derived/concept_net/shard_i.npz
    train    -- [--no-concepts] the network -> data/derived/checkpoints/concept_net*/
    gauge    -- the gauges on the test and held splits"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import os

import chess
import chess.syzygy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .concepts import Registry, DIRECTIONS
from .endgames import load as load_rows, split_of
from .memo import Memo

DATA = Path("data/derived/endgames_v2")
OUT = Path("data/derived/checkpoints/concept_net")
HELD = {"KRPvKP", "KQvKRP", "KBNvK"}
MAX_INST = 256                           # ~211 instances per position in round zero
KIND_PRIORITY = {"position": 0, "line": 1, "region_pair": 2, "region_unary": 3, "pair": 4, "node": 5}   # what survives the cap: the relational kinds first, the node answers last (the pieces carry most of them)
MAX_PIECES = 6
N_VALUE_BUCKETS = 8
VALUE_EDGES = np.array([0.5, 1.5, 2.5, 3.5, 5.5, 8.5, 20.5])
PIECE_ID = {(chess.WHITE, chess.KING): 1, (chess.WHITE, chess.QUEEN): 2, (chess.WHITE, chess.ROOK): 3, (chess.WHITE, chess.BISHOP): 4, (chess.WHITE, chess.KNIGHT): 5, (chess.WHITE, chess.PAWN): 6,
            (chess.BLACK, chess.KING): 7, (chess.BLACK, chess.QUEEN): 8, (chess.BLACK, chess.ROOK): 9, (chess.BLACK, chess.BISHOP): 10, (chess.BLACK, chess.KNIGHT): 11, (chess.BLACK, chess.PAWN): 12}
REL = ("arg_of", "shares_piece", "attacks", "defends")
# MOVE CONSEQUENCES (proposal round one, 2026-09-20, from the residual: the network chose moves whose child it could not
# see -- a king step into the enemy queen's safe checks, a hanging knight left hanging, a pawn's escort lost). Per legal
# move, cheap facts about ITS CHILD from the rules: rules-only, canonical, and attributed as their own terms.
MOVE_FEATURES_V1 = ("gives_check", "our_pieces_falling", "enemy_safe_checks", "our_king_meets_our_pawn_path", "enemy_king_meets_our_pawn_path",
                    "our_pawn_promotion_safe", "enemy_pawn_promotion_safe", "captures", "enemy_king_safe_size", "our_king_safe_size")
MOVE_FEATURES_V2 = MOVE_FEATURES_V1 + ("enemy_discovered_checks",)          # round two: the enemy's discovered checks available after the move (registry v1's discovery)
MOVE_FEATURES = MOVE_FEATURES_V1
REGISTRY_VERSION = os.environ.get("ANALYST_REGISTRY", "0")
if REGISTRY_VERSION == "1":
    MOVE_FEATURES = MOVE_FEATURES_V2
    PREP_SUFFIX = "_v3"
else:
    PREP_SUFFIX = "_v2"
PREP = Path("data/derived/concept_net" + PREP_SUFFIX)       # v2: with the move consequences; v3: registry v1 + the discovered-check consequence


def value_bucket(v: float) -> int:
    return int(np.searchsorted(VALUE_EDGES, abs(v), side="right")) if v >= 0 else min(N_VALUE_BUCKETS - 1, int(np.searchsorted(VALUE_EDGES, abs(v), side="right")))


def move_consequences(board: chess.Board, m: chess.Move) -> list[float]:
    """The child's cheap facts, in the MOVER's view (the side that played m)."""
    from .regions import region
    from .defense import defense
    us = board.turn
    nb = board.copy(stack=False); nb.push(m)
    if nb.is_game_over():
        return [float(nb.is_check()), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(board.is_capture(m)), 0.0, 0.0] + ([0.0] if REGISTRY_VERSION == "1" else [])
    regs = {sq: region(nb, sq) for sq in nb.piece_map()}
    ok, ek = nb.king(us), nb.king(not us)
    falling = sum(1 for sq, p in nb.piece_map().items() if p.color == us and p.piece_type != chess.KING and (d := defense(nb, sq)) is not None and d.falls)
    safe_checks = sum(1 for sq, p in nb.piece_map().items() if p.color != us and p.piece_type != chess.KING and any(regs[sq].attacks_from[t] & chess.BB_SQUARES[ok] for t in regs[sq].steps))
    our_pawns = [sq for sq, p in nb.piece_map().items() if p.color == us and p.piece_type == chess.PAWN]
    their_pawns = [sq for sq, p in nb.piece_map().items() if p.color != us and p.piece_type == chess.PAWN]
    ok_meets = float(any(regs[ok].safe & regs[pw].safe for pw in our_pawns)) if ok is not None else 0.0
    ek_meets = float(any(regs[ek].safe & regs[pw].safe for pw in our_pawns)) if ek is not None else 0.0
    our_prom = float(any(regs[pw].safe & regs[pw].promotion_squares() for pw in our_pawns))
    their_prom = float(any(regs[pw].safe & regs[pw].promotion_squares() for pw in their_pawns))
    out = [float(nb.is_check()), float(falling), float(safe_checks), ok_meets, ek_meets, our_prom, their_prom, float(board.is_capture(m)),
           float(chess.popcount(regs[ek].safe)) / 10.0 if ek is not None else 0.0, float(chess.popcount(regs[ok].safe)) / 10.0 if ok is not None else 0.0]
    if REGISTRY_VERSION == "1":
        from .concepts import _discovery_program, Context
        from .graph import Graph
        # the enemy's discovered checks available after the move: instances of discovery(A, B) for the side to move in the child
        class _Ctx:  # a light context: the discovery program reads only the board
            pass
        ctx = _Ctx(); ctx.board = nb
        out.append(float(sum(i.value for i in _discovery_program(ctx) if nb.piece_at(i.args[0]).color == nb.turn)))
    return out


def encode_row(row: dict, reg: Registry, cid: dict) -> dict:
    """Pieces, instances, relations, legal moves and their labels for one position."""
    b = chess.Board(row["fen"]); memo = Memo()
    inst = reg.instances(b, memo)
    kind = {c.name: KIND_PRIORITY[c.kind] for c in reg.concepts}
    inst.sort(key=lambda i: (kind[i.concept], cid[i.concept], i.args, i.line))
    inst = inst[:MAX_INST]
    pieces = sorted(b.piece_map().items())
    psq = np.full(MAX_PIECES, 64, np.int64); pty = np.zeros(MAX_PIECES, np.int64)
    for k, (sq, p) in enumerate(pieces[:MAX_PIECES]):
        psq[k] = sq; pty[k] = PIECE_ID[(p.color, p.piece_type)]
    pidx = {sq: k for k, (sq, p) in enumerate(pieces[:MAX_PIECES])}
    ic = np.zeros(MAX_INST, np.int64); ia = np.full((MAX_INST, 2), -1, np.int64); iv = np.zeros(MAX_INST, np.int64); il = np.full(MAX_INST, -1, np.int64); im = np.zeros(MAX_INST, bool)
    for k, i in enumerate(inst):
        ic[k] = cid[i.concept]; im[k] = True; iv[k] = value_bucket(i.value); il[k] = i.line
        for j, a in enumerate(i.args[:2]):
            ia[k, j] = pidx.get(a, -1)
    # relations among PIECES (attacks / defends); the instance relations (arg_of, shares_piece) are derived from ia in the model
    rel = np.zeros((MAX_PIECES, MAX_PIECES, 2), np.int8)
    for sq, p in pieces[:MAX_PIECES]:
        for t in b.attacks(sq):
            q = b.piece_at(t)
            if q is not None and t in pidx:
                rel[pidx[sq], pidx[t], 0 if q.color != p.color else 1] = 1
    # legal moves and the partition
    moves = row.get("moves", {})
    legal = list(b.legal_moves)
    best = max((np.sign(v) for v in moves.values()), default=0)
    mf = np.full(64, -1, np.int64); mt = np.full(64, -1, np.int64); keep = np.zeros(64, bool); lm = np.zeros(64, bool); mvf = np.zeros((64, len(MOVE_FEATURES)), np.float32)
    for k, m in enumerate(legal[:64]):
        if m.uci() not in moves:
            continue
        mf[k] = pidx.get(m.from_square, -1); mt[k] = m.to_square; lm[k] = mf[k] >= 0; keep[k] = np.sign(moves[m.uci()]) == best
        if lm[k]:
            mvf[k] = move_consequences(b, m)
    return dict(psq=psq, pty=pty, side=int(b.turn), ic=ic, ia=ia, iv=iv, il=il, im=im, rel=rel, mf=mf, mt=mt, lm=lm, keep=keep, mvf=mvf, n_inst=len(inst))


def prepare(shard: str, log=print):
    i, n = (int(x) for x in shard.split("/"))
    rows = load_rows(DATA); reg = Registry(version=REGISTRY_VERSION); cid = {c.name: k for k, c in enumerate(reg.concepts)}
    PREP.mkdir(parents=True, exist_ok=True)
    idx = list(range(i, len(rows), n)); enc = []; t0 = time.perf_counter()
    for k, r in enumerate(idx):
        enc.append(encode_row(rows[r], reg, cid))
        if (k + 1) % 2000 == 0:
            log(f"[concept_net {shard}] {k + 1}/{len(idx)} in {time.perf_counter() - t0:.0f}s", flush=True)
    keys = enc[0].keys()
    np.savez_compressed(PREP / f"shard_{i}.npz", idx=np.array(idx), **{k: np.stack([e[k] for e in enc]) for k in keys})
    log(f"[concept_net {shard}] done {len(idx)} in {time.perf_counter() - t0:.0f}s; mean instances {np.mean([e['n_inst'] for e in enc]):.1f}", flush=True)


class Data:
    def __init__(self, log=print):
        rows = load_rows(DATA); self.rows = rows
        parts = [np.load(p) for p in sorted(PREP.glob("shard_*.npz"))]
        order = np.concatenate([p["idx"] for p in parts]); inv = np.argsort(order)
        self.arr = {k: np.concatenate([p[k] for p in parts])[inv] for k in parts[0].files if k != "idx"}
        N = len(order); assert N == len(rows), (N, len(rows))
        self.split = np.array([split_of(r, HELD) for r in rows]); self.material = np.array([r["material"] for r in rows])
        self.wdl = np.array([(1 if r["wdl"] > 0 else -1 if r["wdl"] < 0 else 0) + 1 for r in rows])        # the mover's view
        log(f"[concept_net] {N} positions; instances/position {self.arr['n_inst'].mean():.1f}; keepers/legal {self.arr['keep'].sum() / max(1, self.arr['lm'].sum()):.3f}; splits {dict(zip(*np.unique(self.split, return_counts=True)))}", flush=True)

    def ids(self, split):
        return np.nonzero(self.split == split)[0]

    def batch(self, idx, device):
        out = {k: torch.as_tensor(self.arr[k][idx]) for k in ("psq", "pty", "side", "ic", "ia", "iv", "il", "im", "mf", "mt", "lm", "keep")}
        if "mvf" in self.arr:
            out["mvf"] = torch.as_tensor(self.arr["mvf"][idx])
        out["rel"] = torch.as_tensor(self.arr["rel"][idx]).float(); out["wdl"] = torch.as_tensor(self.wdl[idx])
        return {k: v.to(device) for k, v in out.items()}


@dataclass
class Config:
    d: int = 96
    layers: int = 2
    heads: int = 4
    lr: float = 1e-3
    batch: int = 128
    epochs: int = 12
    concepts: bool = True
    w_value: float = 0.5
    consequences: bool = False       # the move's child facts as their own attributed terms (proposal round one)
    decision_from_concepts: bool = False   # the board and piece tokens are CONTEXT (attention) only: no contribution term of their own (the third loophole, 2026-09-20: the whole contrast sat on the board token)
    w_attr_sparse: float = 0.1       # the attribution should rest on few tokens: entropy of the normalized |contributions| over the concept tokens (the top-token intervention was satisfied by DIFFUSING the attribution over 69 tokens, 2026-09-20)
    int_random: bool = True          # intervene on a RANDOM concept token, not the top one: consistency must hold for any token, so diffusion does not help
    w_int: float = 0.0               # INTERVENTION CONSISTENCY: removing the top-attributed concept token must change the chosen move's score by what the attribution said (2026-09-20: without it, the top concept was causal in a third of decisions)
    seed: int = 1


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.heads = heads
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d); self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, bias, mask):
        B, T, d = x.shape; h = self.heads
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, h, d // h).unbind(2)
        att = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(d // h) + bias
        att = att.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
        x = x + self.o(torch.einsum("bhts,bshd->bthd", att, v).reshape(B, T, d))
        return x + self.ff(self.ln2(x))


class ConceptNet(nn.Module):
    def __init__(self, cfg: Config, n_concepts: int):
        super().__init__()
        self.cfg = cfg; d = cfg.d
        self.piece = nn.Embedding(13, d); self.square = nn.Embedding(65, d); self.side = nn.Embedding(2, d); self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.concept = nn.Embedding(n_concepts, d); self.role = nn.Embedding(2, d); self.value = nn.Embedding(N_VALUE_BUCKETS, d); self.line = nn.Embedding(len(DIRECTIONS) + 1, d)
        self.rel_bias = nn.Linear(len(REL), cfg.heads, bias=False)
        self.blocks = nn.ModuleList([Block(d, cfg.heads) for _ in range(cfg.layers)])
        self.ln = nn.LayerNorm(d)
        self.move_q = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.contrib = nn.Linear(d, d)                                        # per token: a vector whose dot with the move query is its contribution
        self.gate = nn.Linear(d, d)                                           # per token: attention logits against the move query
        self.value_head = nn.Linear(d, 3)
        self.move_feat = nn.Linear(len(MOVE_FEATURES), 1, bias=False)     # one weight per consequence: exact, readable terms

    def tokens(self, b):
        B = b["psq"].shape[0]; d = self.cfg.d
        piece_tok = self.piece(b["pty"]) + self.square(b["psq"])                                              # (B, P, d)
        pmask = b["pty"] > 0
        cls = self.cls.expand(B, 1, -1) + self.side(b["side"])[:, None]
        P = MAX_PIECES
        if self.cfg.concepts:
            ia = b["ia"].clamp(min=0)
            arg_tok = torch.gather(piece_tok, 1, ia.reshape(B, -1, 1).expand(-1, -1, d)).view(B, MAX_INST, 2, d)
            arg_tok = (arg_tok + self.role.weight[None, None]) * (b["ia"] >= 0)[..., None]
            inst_tok = self.concept(b["ic"]) + arg_tok.sum(2) + self.value(b["iv"]) + self.line(b["il"] + 1)
            x = torch.cat([cls, piece_tok, inst_tok], 1); mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), pmask, b["im"]], 1)
            T = P + MAX_INST
            A = torch.zeros(B, MAX_INST, P, device=x.device)                                                  # instance k has piece p as an argument
            A.scatter_(2, ia, (b["ia"] >= 0).float())
            rel = torch.zeros(B, T, T, len(REL), device=x.device)
            rel[:, P:, :P, 0] = A; rel[:, :P, P:, 0] = A.transpose(1, 2)
            S = (A @ A.transpose(1, 2) > 0).float(); S = S * (1 - torch.eye(MAX_INST, device=x.device))[None]
            rel[:, P:, P:, 1] = S
            rel[:, :P, :P, 2:] = b["rel"]
            bias = torch.zeros(B, self.cfg.heads, 1 + T, 1 + T, device=x.device)
            bias[:, :, 1:, 1:] = self.rel_bias(rel).permute(0, 3, 1, 2)
        else:
            x = torch.cat([cls, piece_tok], 1); mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), pmask], 1)
            rel = torch.zeros(B, P, P, len(REL), device=x.device); rel[..., 2:] = b["rel"]
            bias = torch.zeros(B, self.cfg.heads, 1 + P, 1 + P, device=x.device)
            bias[:, :, 1:, 1:] = self.rel_bias(rel).permute(0, 3, 1, 2)
        return x, mask, bias, piece_tok

    def forward(self, b):
        x, mask, bias, piece_tok = self.tokens(b)
        for blk in self.blocks:
            x = blk(x, bias, mask)
        h = self.ln(x)                                                                                        # (B, T, d)
        B = h.shape[0]
        hp = h[:, 1:1 + MAX_PIECES]
        mf = b["mf"].clamp(min=0)
        q = self.move_q(torch.cat([torch.gather(hp, 1, mf[..., None].expand(-1, -1, self.cfg.d)), self.square(b["mt"].clamp(min=0))], -1))   # (B, 64, d)
        c = self.contrib(h); g = self.gate(h)                                                                 # (B, T, d)
        logits_g = torch.einsum("bmd,btd->bmt", q, g) / math.sqrt(self.cfg.d)
        logits_g = logits_g.masked_fill(~mask[:, None, :], float("-inf"))
        alpha = logits_g.softmax(-1)                                                                          # (B, 64, T): which tokens the move listens to
        per_tok = alpha * torch.einsum("bmd,btd->bmt", q, c) / math.sqrt(self.cfg.d)                        # the exact per-token contribution
        if self.cfg.decision_from_concepts and self.cfg.concepts:
            per_tok = per_tok * torch.cat([torch.zeros(1 + MAX_PIECES, device=per_tok.device), torch.ones(MAX_INST, device=per_tok.device)])[None, None]
        score = per_tok.sum(-1)                                                                               # (B, 64)
        per_feat = None
        if self.cfg.consequences and "mvf" in b:
            per_feat = b["mvf"] * self.move_feat.weight[0][None, None]                                        # (B, 64, F): each consequence's own term
            score = score + per_feat.sum(-1)
        score = score.masked_fill(~b["lm"], float("-inf"))
        return dict(score=score, per_tok=per_tok, per_feat=per_feat, alpha=alpha, value=self.value_head(h[:, 0]), mask=mask)


def contrast(o, b):
    """The decision's attribution: per-token and per-consequence contributions to score(chosen) - score(runner-up).
    A constant offset cancels; a diffuse attribution still has to be consistent under intervention."""
    sc = o["score"]
    pick = sc.argmax(-1)
    sc2 = sc.scatter(1, pick[:, None], float("-inf")); second = sc2.argmax(-1)
    T = o["per_tok"].shape[-1]
    pt = o["per_tok"].gather(1, pick[:, None, None].expand(-1, 1, T)).squeeze(1) - o["per_tok"].gather(1, second[:, None, None].expand(-1, 1, T)).squeeze(1)
    pf = None
    if o.get("per_feat") is not None:
        Fn = o["per_feat"].shape[-1]
        pf = o["per_feat"].gather(1, pick[:, None, None].expand(-1, 1, Fn)).squeeze(1) - o["per_feat"].gather(1, second[:, None, None].expand(-1, 1, Fn)).squeeze(1)
    margin = sc.gather(1, pick[:, None]).squeeze(1) - sc.gather(1, second[:, None]).squeeze(1)
    return pick, second, pt, pf, margin


def losses(model, b, cfg):
    o = model(b)
    keep = b["keep"] & b["lm"]
    logp = o["score"].log_softmax(-1)
    tgt = keep.float() / keep.float().sum(-1, keepdim=True).clamp(min=1)
    has = keep.any(-1)
    out = dict(policy=-(tgt * logp.masked_fill(~b["lm"], 0.0)).sum(-1)[has].mean() if has.any() else o["score"].new_zeros(()),
               value=F.cross_entropy(o["value"], b["wdl"]))
    if cfg.w_int > 0 and cfg.concepts:
        # ON THE CONTRAST: removing a concept token must change (score(chosen) - score(runner-up)) by the token's attributed
        # contribution to that contrast; a token contributing equally to both moves is attributed nothing and must change nothing
        pick, second, pt, pf, margin = contrast(o, b)
        pt_inst = pt[:, 1 + MAX_PIECES:].masked_fill(~b["im"], 0.0)
        if cfg.int_random:
            r = torch.rand_like(pt_inst).masked_fill(~b["im"], -1.0); top = r.argmax(-1)
        else:
            top = pt_inst.abs().argmax(-1)
        attributed = pt_inst.gather(1, top[:, None]).squeeze(1)
        b2 = dict(b); b2["im"] = b["im"].clone(); b2["im"][torch.arange(len(top)), top] = False
        o2 = model(b2)
        m2 = o2["score"].gather(1, pick[:, None]).squeeze(1) - o2["score"].gather(1, second[:, None]).squeeze(1)
        valid = b["im"].any(-1) & torch.isfinite(margin) & torch.isfinite(m2)
        out["int"] = ((margin - m2) - attributed).abs()[valid].mean() if valid.any() else o["score"].sum() * 0
    else:
        out["int"] = o["score"].new_zeros(())
    if cfg.concepts and cfg.w_attr_sparse > 0:
        pick, second, pt, pf, margin = contrast(o, b)
        a = pt[:, 1 + MAX_PIECES:].abs().masked_fill(~b["im"], 0.0)
        if pf is not None:
            a = torch.cat([a, pf.abs()], -1)                                                                  # the consequences count as explanation terms too
        a = a / a.sum(-1, keepdim=True).clamp(min=1e-9)
        out["attr_sparse"] = -(a * (a + 1e-9).log()).sum(-1).mean()
    else:
        out["attr_sparse"] = o["score"].new_zeros(())
    out["total"] = out["policy"] + cfg.w_value * out["value"] + cfg.w_int * out["int"] + cfg.w_attr_sparse * out["attr_sparse"]
    return out, o


@torch.no_grad()
def evaluate(model, data, split, cfg, device, max_n=200000):
    model.eval(); ids = data.ids(split)[:max_n]
    if len(ids) == 0:
        model.train(); return {}
    ok = n = 0; rand = 0.0; mats = []; oks = []; lens = []
    for s in range(0, len(ids), 512):
        b = data.batch(ids[s:s + 512], device); o = model(b)
        keep = b["keep"] & b["lm"]; has = keep.any(-1)
        pick = o["score"].argmax(-1)
        hit = keep.gather(1, pick[:, None]).squeeze(1)
        ok += int(hit[has].sum()); n += int(has.sum()); rand += float((keep.float().sum(-1) / b["lm"].float().sum(-1).clamp(min=1))[has].sum())
        mats += list(data.material[ids[s:s + 512]][has.cpu().numpy()]); oks += hit[has].cpu().numpy().tolist()
        _, _, ptc, pfc, _ = contrast(o, b)
        a = ptc.abs()
        if pfc is not None:
            a = torch.cat([a, pfc.abs()], -1)
        srt = a.sort(-1, descending=True).values; tot = srt.sum(-1, keepdim=True); cum = srt.cumsum(-1) / tot.clamp(min=1e-9)
        ok_ = (tot.squeeze(-1) > 1e-4)                                                                        # a tie (no contrast) has no explanation length
        lens += (cum < 0.8).sum(-1).add(1)[ok_].cpu().numpy().tolist()
    st = dict(n=n, keeps=ok / max(1, n), random_legal=rand / max(1, n), explanation_len=float(np.mean(lens)))
    by = defaultdict(list)
    for m, h in zip(mats, oks):
        by[m].append(h)
    st["by_material"] = {m: float(np.mean(v)) for m, v in by.items()}
    model.train(); return st


def train(a):
    cfg = Config(**{k: v for k, v in vars(a).items() if k in Config.__dataclass_fields__})
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    data = Data(); reg = Registry(version=REGISTRY_VERSION)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ConceptNet(cfg, len(reg.concepts)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr = data.ids("train"); steps = len(tr) // cfg.batch; t0 = time.perf_counter()
    print(f"[concept_net] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, concepts={cfg.concepts}; train {len(tr)}; {steps} steps/epoch", flush=True)
    for epoch in range(cfg.epochs):
        perm = np.random.default_rng(cfg.seed + epoch).permutation(tr); acc = {}
        for s in range(steps):
            b = data.batch(perm[s * cfg.batch:(s + 1) * cfg.batch], device)
            L, _ = losses(model, b, cfg)
            opt.zero_grad(); L["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            for k, v in L.items():
                acc[k] = acc.get(k, 0.0) + float(v)
        ev = evaluate(model, data, "val", cfg, device)
        print(f"[concept_net] epoch {epoch} " + " ".join(f"{k} {v / steps:.3f}" for k, v in acc.items()) + f" | val: the chosen move keeps the result {ev['keeps']:.3f} (random legal {ev['random_legal']:.3f}); explanation {ev['explanation_len']:.1f} tokens | {time.perf_counter() - t0:.0f}s", flush=True)
        torch.save(dict(model=model.state_dict(), config=asdict(cfg), n_concepts=len(reg.concepts), registry_version=reg.version, move_features=list(MOVE_FEATURES)), out / "last.pt")
    final = {s: evaluate(model, data, s, cfg, device) for s in ("test", "held")}
    json.dump(final, open(out / "final.json", "w"), indent=1)
    for s, ev in final.items():
        print(f"[concept_net] {s}: keeps {ev['keeps']:.3f} (random legal {ev['random_legal']:.3f}) n {ev['n']} explanation {ev['explanation_len']:.1f}; by material: " + "; ".join(f"{m} {v:.2f}" for m, v in sorted(ev["by_material"].items())), flush=True)


def load_model(path: str):
    ck = torch.load(path, map_location="cpu"); cfg = Config(**ck["config"]); m = ConceptNet(cfg, ck["n_concepts"]); m.load_state_dict(ck["model"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return m.to(device).eval(), cfg, device


@torch.no_grad()
def gauge(ckpt: str, log=print, max_n: int = 4000):
    """Faithfulness under intervention (drop the top concept token of the chosen move: does the choice change?),
    the residual by material and by the concepts present, and the explanation on Kaveh's position."""
    model, cfg, device = load_model(ckpt); data = Data(log=lambda *a, **k: None); reg = Registry(version=REGISTRY_VERSION); names = [c.name for c in reg.concepts]
    for split in ("test", "held"):
        ids = data.ids(split)[:max_n]; changed = same_dir = n = 0; wrong_by = Counter(); present_by = Counter(); present_all = Counter(); n_wrong = 0; n_all = 0
        for s in range(0, len(ids), 256):
            b = data.batch(ids[s:s + 256], device); o = model(b)
            keep = b["keep"] & b["lm"]; has = keep.any(-1); pick = o["score"].argmax(-1)
            _, _, ptc, _, _ = contrast(o, b)
            pt_inst = ptc[:, 1 + MAX_PIECES:].masked_fill(~b["im"], 0.0)
            top = pt_inst.abs().argmax(-1)                                                                    # the concept token contributing most to the CHOICE (chosen over the runner-up)
            b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
            b2["im"][torch.arange(len(top)), top] = False
            o2 = model(b2); pick2 = o2["score"].argmax(-1)
            valid = has & b["im"].any(-1)
            changed += int((pick2 != pick)[valid].sum()); n += int(valid.sum())
            hit = keep.gather(1, pick[:, None]).squeeze(1)
            for i in torch.nonzero(has).squeeze(-1).tolist():
                n_all += 1
                cs = b["ic"][i][b["im"][i]].unique().tolist()
                for c in cs:
                    present_all[names[c]] += 1
                if not hit[i]:
                    n_wrong += 1; wrong_by[data.material[ids[s + i]]] += 1
                    for c in cs:
                        present_by[names[c]] += 1
        ev = evaluate(model, data, split, cfg, device, max_n=max_n)
        log(f"[concept_net gauge] {split}: keeps {ev['keeps']:.3f}; explanation length (ties excluded) {ev['explanation_len']:.1f}; dropping the top concept token changes the chosen move in {100 * changed / max(1, n):.1f}% of {n} positions (faithfulness under intervention)")
        lift = {c: (present_by[c] / max(1, n_wrong)) / (present_all[c] / max(1, n_all)) for c in present_by if present_all[c] >= 50}
        log(f"   residual: {n_wrong} wrong decisions of {n_all}; by material " + "; ".join(f"{m} {c}" for m, c in wrong_by.most_common(6)) +
            "; concepts OVER-represented in the wrong ones (presence in wrong / presence overall): " + "; ".join(f"{c} x{l:.1f}" for c, l in sorted(lift.items(), key=lambda kv: -kv[1])[:8]))
    explain(model, cfg, device, reg, "4k3/1R6/4K3/4P3/8/8/8/5r2 b - - 0 58", log)


@torch.no_grad()
def explain(model, cfg, device, reg, fen: str, log=print):
    cid = {c.name: k for k, c in enumerate(reg.concepts)}
    row = dict(fen=fen, moves={})
    b = chess.Board(fen)
    tb = chess.syzygy.open_tablebase("assets/tablebases/syzygy")
    for m in b.legal_moves:
        nb = b.copy(stack=False); nb.push(m); row["moves"][m.uci()] = -tb.probe_wdl(nb) if not nb.is_game_over() else 0
    tb.close()
    e = encode_row(row, reg, cid)
    bt = {k: torch.as_tensor(np.asarray(v)[None]).to(device) for k, v in e.items() if k != "n_inst"}; bt["rel"] = bt["rel"].float(); bt["wdl"] = torch.zeros(1, dtype=torch.long, device=device)
    if not cfg.consequences:
        bt.pop("mvf", None)
    o = model(bt); legal = list(b.legal_moves)
    sc = o["score"][0].cpu().numpy()
    order = np.argsort(-sc)
    log(f"[explain] {fen}: " + ", ".join(f"{b.san(legal[k])} {sc[k]:+.2f} ({'keeps' if np.sign(row['moves'][legal[k].uci()]) == max(np.sign(v) for v in row['moves'].values()) else 'loses'})" for k in order[:5] if k < len(legal)))
    inst = reg.instances(b); inst.sort(key=lambda i: (cid[i.concept], i.args, i.line)); inst = inst[:MAX_INST]
    if len(order) >= 2 and order[1] < len(legal):
        k1, k2 = int(order[0]), int(order[1])
        d = (o["per_tok"][0, k1] - o["per_tok"][0, k2]).cpu().numpy()
        parts = []
        for t in np.argsort(-np.abs(d))[:6]:
            if abs(d[t]) < 1e-3:
                continue
            if t == 0:
                parts.append(f"board {d[t]:+.2f}")
            elif t < 1 + MAX_PIECES:
                sq = int(e["psq"][t - 1]); parts.append(f"{chess.piece_name(b.piece_at(sq).piece_type)} {chess.square_name(sq)} {d[t]:+.2f}" if sq < 64 else "")
            else:
                j = t - 1 - MAX_PIECES
                if j < len(inst):
                    parts.append(f"{inst[j].text(b)} {d[t]:+.2f}")
        if o.get("per_feat") is not None:
            df = (o["per_feat"][0, k1] - o["per_feat"][0, k2]).cpu().numpy()
            parts += [f"{MOVE_FEATURES[j]} {df[j]:+.2f}" for j in np.argsort(-np.abs(df))[:3] if abs(df[j]) > 1e-3]
        log(f"   why {b.san(legal[k1])} over {b.san(legal[k2])}: " + "; ".join(p for p in parts if p))
    for k in order[:2]:
        if k >= len(legal):
            continue
        if o.get("per_feat") is not None:
            pf = o["per_feat"][0, k].cpu().numpy(); log(f"   {b.san(legal[k])} consequences: " + "; ".join(f"{MOVE_FEATURES[j]} {pf[j]:+.2f}" for j in np.argsort(-np.abs(pf))[:4] if abs(pf[j]) > 1e-3))
        pt = o["per_tok"][0, k].cpu().numpy()
        top = np.argsort(-np.abs(pt))[:6]
        parts = []
        for t in top:
            if t == 0:
                parts.append(f"board {pt[t]:+.2f}")
            elif t < 1 + MAX_PIECES:
                sq = int(e["psq"][t - 1]); parts.append(f"{chess.piece_name(b.piece_at(sq).piece_type)} {chess.square_name(sq)} {pt[t]:+.2f}" if sq < 64 else "")
            else:
                j = t - 1 - MAX_PIECES
                if j < len(inst):
                    parts.append(f"{inst[j].text(b)} {pt[t]:+.2f}")
        log(f"   {b.san(legal[k])}: " + "; ".join(p for p in parts if p))


@torch.no_grad()
def residual(ckpt: str, materials: str, per: int = 4, log=print):
    """The proposal prompt: wrong decisions in the named materials -- the position, what was chosen, what kept the
    result, and the concept instances the chosen move listened to most."""
    model, cfg, device = load_model(ckpt); data = Data(log=lambda *a, **k: None); reg = Registry(version=REGISTRY_VERSION)
    cid = {c.name: k for k, c in enumerate(reg.concepts)}
    for mat in materials.split(","):
        ids = [i for i in np.nonzero(data.material == mat)[0] if data.split[i] in ("test", "held")]
        shown = 0
        for s in range(0, len(ids), 256):
            if shown >= per:
                break
            b = data.batch(ids[s:s + 256], device); o = model(b)
            keep = b["keep"] & b["lm"]; has = keep.any(-1); pick = o["score"].argmax(-1); hit = keep.gather(1, pick[:, None]).squeeze(1)
            for i in torch.nonzero(has & ~hit).squeeze(-1).tolist():
                if shown >= per:
                    break
                r = data.rows[ids[s + i]]; bd = chess.Board(r["fen"]); legal = list(bd.legal_moves)
                best = max(np.sign(v) for v in r["moves"].values())
                keepers = [chess.Move.from_uci(u) for u, v in r["moves"].items() if np.sign(v) == best]
                chosen = legal[int(pick[i])]
                inst = reg.instances(bd); kind = {c.name: {"position": 0, "line": 1, "region_pair": 2, "region_unary": 3, "pair": 4, "node": 5}[c.kind] for c in reg.concepts}
                inst.sort(key=lambda x: (kind[x.concept], cid[x.concept], x.args, x.line)); inst = inst[:MAX_INST]
                pt = o["per_tok"][i, int(pick[i]), 1 + MAX_PIECES:].cpu().numpy()
                top = [j for j in np.argsort(-np.abs(pt))[:5] if j < len(inst)]
                log(f"[residual {mat}] {r['fen']} | result {r['wdl']:+d} dtz {r['dtz']} | chose {bd.san(chosen)} ({'keeps' if np.sign(r['moves'][chosen.uci()]) == best else 'loses'}); keepers: {', '.join(bd.san(m) for m in keepers[:6])}")
                log("    listened to: " + "; ".join(f"{inst[j].text(bd)} {pt[j]:+.2f}" for j in top))
                shown += 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "train", "gauge", "residual"]); ap.add_argument("--materials", default="KQPvKQ,KPvKP,KBNvK"); ap.add_argument("--per", type=int, default=4); ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", default=str(OUT)); ap.add_argument("--ckpt", default=str(OUT / "last.pt"))
    ap.add_argument("--no-concepts", action="store_true"); ap.add_argument("--consequences", action="store_true"); ap.add_argument("--decision-from-concepts", action="store_true")
    for k, f in Config.__dataclass_fields__.items():
        if k in ("concepts", "consequences", "int_random", "decision_from_concepts"):
            continue
        ap.add_argument(f"--{k}", type=type(f.default), default=f.default)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a.shard)
    elif a.cmd == "train":
        a.concepts = not a.no_concepts; a.decision_from_concepts = a.decision_from_concepts; train(a)
    elif a.cmd == "residual":
        residual(a.ckpt, a.materials, a.per)
    else:
        gauge(a.ckpt)


if __name__ == "__main__":
    main()
