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
from .regimes import Codebook, fit as fit_codebook
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE

Q = len(NODE_QUESTIONS); R = len(EDGE_RELATIONS); P = len(POSITION_QUESTIONS); K_REG = len(REGIME_KEYS)
N_TOK = MAX_PIECES + 1                      # pieces + the board token (index 0)
HIDE_LABELS: set = set()                    # acquire.py: positions whose move labels are withheld in the next train()
DIST_COL = EDGE_RELATIONS.index("distance")


PIECE_FUTURE_QUESTIONS = ("free_moves", "safe_moves", "stuck", "trapped", "attacks_up", "king_zone_hits", "exchange", "falls", "pinned", "holds_value")


@dataclass
class Config:
    d: int = 128
    d_geo: int = 64                         # the geometry's dimension (psi = W phi)
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
    geo_lr_mult: float = 30.0               # the quasimetric's scale/mix and the basin temperature learn 30x faster
    neg_mode: str = "qrl"                   # qrl: push apart every pair not known reachable (other games AND earlier positions of the same game); cross: other games only
    w_cp: float = 1.0
    w_regime: float = 1.0
    w_policy: float = 1.0
    w_traj: float = 0.05
    w_delta: float = 1.0                    # the move-conditioned readout
    w_future: float = 1.0                   # the quantized future: cross-entropy over regime codes at +10 and +20
    w_value: float = 1.0                    # the engine's evaluation at +10 and +20
    w_sharp: float = 1.0                    # the window sharpness labels
    w_q: float = 1.0                        # the action-value head (soft labels)
    w_piece_future: float = 0.5             # the per-node future: each piece's answer buckets at +10/+20 and its fate (kept/exchanged/lost/promoted)
    q_temp: float = 0.05                    # the policy's soft target: softmax(win probability / q_temp) over legal moves
    n_codes: int = 48
    head_input: str = "phi"                 # phi: the heads read the Backpack sum (interpretable by construction); board: the board token (free)
    w_vic: float = 2.0                      # variance-covariance regularizer on phi (VICReg's two terms): repulsion against linear collapse
    seed: int = 1


# ---------------------------------------------------------------- buckets: quantile edges per question, fixed from the training split
class Buckets:
    def __init__(self, n: int):
        self.n = n; self.edges = None      # (Q, n-1)

    def fit(self, values: np.ndarray, mask: np.ndarray):
        """values (N, 32, Q) with mask (N, 32): per-question quantile edges over real pieces."""
        v = values[mask]                   # (M, Q)
        qs = np.linspace(0, 1, self.n + 1)[1:-1]
        self.edges = []
        for q in range(v.shape[1]):
            distinct = np.unique(v[:, q])
            if len(distinct) <= self.n:
                # a sparse or small-integer answer (pinned, falls, trapped, attacks_up, king_zone_hits ...): one bucket per
                # distinct value. Quantile edges collapse anything nonzero in fewer than 1/n of the pieces into ONE bucket,
                # which left the network blind to exactly the tactical answers (found 2026-09-18 by the persistence readout)
                self.edges.append(distinct[1:] if len(distinct) > 1 else distinct)      # searchsorted right: a value equal to an edge goes up
            else:
                self.edges.append(np.unique(np.quantile(v[:, q], qs)))
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
        self.sense = nn.Embedding(13 * Q * cfg.buckets, d)            # output side: one sense vector per (piece type, question, bucket)
        self.type_emb = nn.Embedding(13, d); self.square_emb = nn.Embedding(65, d)   # identity: signed piece type (0 = pad), square (64 = pad)
        self.type_sense = nn.Embedding(13, d); self.square_sense = nn.Embedding(65, d)
        self.pos_emb = nn.Embedding(P * cfg.buckets, d)               # the board token: one embedding per (indicator, bucket)
        self.rel_bias = nn.Linear(R, cfg.heads, bias=False)           # pair relations -> per-head bias
        self.dist_bias = nn.Embedding(9, cfg.heads)                   # distance 0..7 (8 = unrelated)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, cfg.dropout) for _ in range(cfg.layers)])
        self.ln = nn.LayerNorm(d)
        self.alpha = nn.Linear(d, Q + 2)                              # per piece: a weight per question sense, plus identity and square senses
        self.iqe = IQE(cfg.d_geo, components=16)
        self.geo = nn.Linear(d, cfg.d_geo)                            # the geometry's own projection: the trajectory shapes psi, the heads keep phi
        self.poles = nn.Parameter(torch.randn(3, cfg.d_geo) * 0.05)   # win / draw / loss, side-to-move view (in psi)
        self.log_T = nn.Parameter(torch.zeros(()))
        self.cp_head = nn.Linear(d, 1)
        self.regime_head = nn.Linear(d, 2 * K_REG)
        self.future_head = nn.Linear(d, 2 * cfg.n_codes)             # the distribution over regime codes at +10 and +20
        self.sharp_head = nn.Linear(d, 2)                             # logits: a big threat within 10 plies; the engine's best moves forcing over the window
        self.value_head = nn.Sequential(nn.Linear(d + 1, d), nn.GELU(), nn.Linear(d, 2))   # the DRIFT of tanh(cp/400) by +10 and +20 from phi and the evaluation now
        nn.init.zeros_(self.value_head[2].weight); nn.init.zeros_(self.value_head[2].bias)   # starts at 'no change': persistence is the floor by construction
        self.pol_from = nn.Linear(d, d); self.pol_to = nn.Embedding(64, d)
        # the move-conditioned readout (Kaveh 2026-08-11, concept dynamics): from the mover's token, the destination
        # and the board token, predict the CHANGE of the geometry after the move as the opponent will see it. Scoring
        # a move then needs no child graph: value(move) = the predicted psi', read against the poles.
        self.delta_to = nn.Embedding(64, d)
        self.delta = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, cfg.d_geo))
        # the ACTION-VALUE head: from the mover's token, the destination and the board token, the teacher's win
        # probability of that move (ChessBench's target) -- a move score with no child graph
        self.q_to = nn.Embedding(64, d)
        self.q_head = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1))
        # THE PER-PIECE FUTURE (Kaveh 2026-09-18): from each piece's token, its answer buckets at +10 and +20 for the
        # activity questions, and its fate (kept / exchanged / lost / promoted) -- "this bishop tends to be inactive"
        self.n_pfq = len(PIECE_FUTURE_QUESTIONS)
        self.piece_future_head = nn.Linear(d, 2 * self.n_pfq * cfg.buckets)
        self.fate_head = nn.Linear(d, 2 * 4)
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
        senses = self.sense(self.sense_index(b))                                        # (B, 32, Q, d)
        ts, ss = self.type_sense(b["types"]), self.square_sense(b["squares"])           # (B, 32, d)
        if self.cfg.variant == "board":
            contrib = a[..., Q:Q + 1] * ts + a[..., Q + 1:Q + 2] * ss
        elif self.cfg.variant == "rules":
            contrib = (a[..., :Q, None] * senses).sum(2) + a[..., Q:Q + 1] * ts
        else:
            contrib = (a[..., :Q, None] * senses).sum(2) + a[..., Q:Q + 1] * ts + a[..., Q + 1:Q + 2] * ss
        norm = a.sum((1, 2))[:, None] + 1.0
        phi = contrib.sum(1) / norm                                                     # (B, d): linear in the senses
        if self.cfg.head_input == "board":
            phi = h[:, 0]                                                               # the free path: the heads read the board token
        return dict(h=h, phi=phi, alpha=a, norm=norm)

    def sense_index(self, b):
        """(B, 32, Q): the sense of (piece type, question, answer bucket) -- 'a queen attacked twice' is its own vector."""
        return b["types"][..., None] * (Q * self.cfg.buckets) + b["buckets"] + self.q_offsets

    def piece_future(self, h):
        """(B, 32, 2, Qf, buckets) logits and (B, 32, 2, 4) fate logits from the piece tokens h[:, 1:]."""
        pieces = h[:, 1:]
        B, P, _ = pieces.shape
        return self.piece_future_head(pieces).view(B, P, 2, self.n_pfq, self.cfg.buckets), self.fate_head(pieces).view(B, P, 2, 4)

    def psi(self, phi):
        """The geometry's coordinates: a linear map of phi, so the readout by senses still holds (psi is linear in them too)."""
        return self.geo(phi)

    def basin_logits(self, phi):
        B = phi.shape[0]
        psi = self.psi(phi)
        u = psi[:, None, :].expand(B, 3, -1).reshape(B * 3, -1)
        v = self.poles[None].expand(B, 3, -1).reshape(B * 3, -1)
        d = self.iqe(u, v).reshape(B, 3)
        return -d / torch.exp(self.log_T), d

    def predicted_psi(self, h, phi, from_idx, to_sq):
        """(B,) mover token index and destination square -> the predicted psi of the position after the move, from
        the SIDE THEN TO MOVE (the opponent): psi_next = psi(phi) + delta(mover, destination, board)."""
        B = h.shape[0]
        mover = h[torch.arange(B, device=h.device), 1 + from_idx]                        # (B, d)
        x = torch.cat([mover, self.delta_to(to_sq), h[:, 0]], dim=-1)
        return self.psi(phi) + self.delta(x)

    def move_values(self, h, phi, b):
        """(B, 32, 64): for every legal move, the mover's predicted standing after it -- the opponent's distance to
        THEIR win pole minus to their loss pole (far from their win, near their loss = good for us)."""
        B = h.shape[0]
        legal = b["legal"]
        idx = legal.nonzero(as_tuple=False)                                              # (M, 3): batch, from, to
        if len(idx) == 0:
            return torch.full_like(legal, float("-inf"), dtype=h.dtype)
        mover = h[idx[:, 0], 1 + idx[:, 1]]
        x = torch.cat([mover, self.delta_to(idx[:, 2]), h[idx[:, 0], 0]], dim=-1)
        psi_next = self.psi(phi)[idx[:, 0]] + self.delta(x)                              # (M, d_geo)
        M = psi_next.shape[0]
        u = psi_next[:, None, :].expand(M, 3, -1).reshape(M * 3, -1)
        v = self.poles[None].expand(M, 3, -1).reshape(M * 3, -1)
        d = self.iqe(u, v).reshape(M, 3)                                                 # to win / draw / loss of the side then to move
        val = d[:, 0] - d[:, 2]
        out = torch.full(legal.shape, float("-inf"), dtype=h.dtype, device=h.device)
        out[idx[:, 0], idx[:, 1], idx[:, 2]] = val
        return out

    def value_ahead(self, phi, cp_now=None):
        """(B, 2): the evaluation at +10 and +20 given phi and the evaluation NOW (the engine's number when the
        product has it; the network's own estimate otherwise), so 'no change' is the floor and the head learns the drift."""
        if cp_now is None:
            cp_now = self.cp_head(phi).squeeze(-1).detach()
        return cp_now[:, None] + self.value_head(torch.cat([phi, cp_now[:, None]], dim=-1))

    def q_logits(self, h, b):
        """(B, 32, 64): the action-value logit of every legal move (sigmoid = win probability), -inf elsewhere."""
        legal = b["legal"]
        idx = legal.nonzero(as_tuple=False)
        out = torch.full(legal.shape, float("-inf"), dtype=h.dtype, device=h.device)
        if len(idx) == 0:
            return out
        mover = h[idx[:, 0], 1 + idx[:, 1]]
        x = torch.cat([mover, self.q_to(idx[:, 2]), h[idx[:, 0], 0]], dim=-1)
        out[idx[:, 0], idx[:, 1], idx[:, 2]] = self.q_head(x).squeeze(-1)
        return out

    def policy_logits(self, h, b):
        """(B, 32, 64): from-piece token x to-square, -inf where not legal."""
        f = self.pol_from(h[:, 1:])                                                     # (B, 32, d)
        logits = torch.einsum("bid,sd->bis", f, self.pol_to.weight) / math.sqrt(f.shape[-1])
        return logits.masked_fill(~b["legal"], float("-inf"))


# ---------------------------------------------------------------- data
class Data:
    """The dataset in memory, every per-sample array precomputed once (dense int8 relations, legal masks), so a batch
    is fancy indexing; partner lookup by (game, ply) for the trajectory pairs."""
    def __init__(self, path: str, buckets: Buckets | None = None, n_buckets: int = 12, split_seed: int = 0, log=print, meta_only: bool = False, n_codes: int = 48, hide_labels=None, pos_buckets: Buckets | None = None):
        """`path`: one directory, or several joined by ','. `hide_labels`: a set of position indices (into the joined
        order) whose move labels are withheld -- the unlabeled pool of an acquisition experiment."""
        t0 = time.perf_counter()
        parts = [load_dataset(p_, meta_only=meta_only) for p_ in path.split(",")]
        d = parts[0] if len(parts) == 1 else dict(meta=[m for p_ in parts for m in p_["meta"]], **({} if meta_only else dict(
            nodes=np.concatenate([p_["nodes"] for p_ in parts]), mask=np.concatenate([p_["mask"] for p_ in parts]),
            position=np.concatenate([p_["position"] for p_ in parts]), edges=[e for p_ in parts for e in p_["edges"]])))
        self.meta = d["meta"]
        self.hidden = set(hide_labels or ())
        N = len(self.meta)
        # the split is PER CORPUS: each corpus's games are shuffled on their own, exactly as when that corpus is
        # loaded alone, so joining corpora never moves a game between train and test (2026-09-13: joining ChessBench
        # had reshuffled Kaveh's games and put 54 of the 67 expert-row games into training)
        self.split = np.empty(N, dtype="<U5")
        for part in parts:
            ids = {m["game"] for m in part["meta"]}
            games = sorted(ids); rng = random.Random(split_seed); rng.shuffle(games)
            n = len(games); test_games = set(games[: n // 10]); val_games = set(games[n // 10: n // 5])
            for i, m in enumerate(self.meta):
                if m["game"] in ids:
                    self.split[i] = "test" if m["game"] in test_games else "val" if m["game"] in val_games else "train"
        self.index = {(m["game"], m["ply"]): i for i, m in enumerate(self.meta)}
        if meta_only:
            return
        self.nodes, self.mask, self.position, edges = d["nodes"], d["mask"], d["position"], d["edges"]
        tr = self.split == "train"
        self.buckets = buckets or Buckets(n_buckets).fit(self.nodes[tr], self.mask[tr])
        self.node_b = self.buckets.apply(self.nodes).astype(np.int8)                     # (N, 32, Q)
        # the per-piece futures (dataset.piece_futures), per corpus; a corpus without them is masked out
        from .dataset import load_piece_futures
        PFQ = [NODE_QUESTIONS.index(q) for q in PIECE_FUTURE_QUESTIONS]
        self.pf_b, self.pf_has, self.fate = {}, {}, {}
        for h in (10, 20):
            fb, fh, ft = [], [], []
            for p_, part in zip(path.split(","), parts):
                n_ = len(part["meta"]); pf = load_piece_futures(p_)
                if pf is None or len(pf[f"fut_has_{h}"]) != n_:
                    fb.append(np.zeros((n_, MAX_PIECES, len(PFQ)), np.int8)); fh.append(np.zeros((n_, MAX_PIECES), bool)); ft.append(np.full((n_, MAX_PIECES), -1, np.int8))
                else:
                    fb.append(self.buckets.apply(pf[f"fut_nodes_{h}"])[..., PFQ].astype(np.int8)); fh.append(pf[f"fut_has_{h}"]); ft.append(pf[f"fate_{h}"])
            self.pf_b[h] = np.concatenate(fb); self.pf_has[h] = np.concatenate(fh) & self.mask; self.fate[h] = np.concatenate(ft)
        self.pf_q = np.array(PFQ)
        # a checkpoint's own position buckets when evaluating it: refitting them changes the board token's inputs
        # whenever the fitting rule changes (2026-09-19: the sparse-answer buckets silently broke old checkpoints here)
        pb = pos_buckets if pos_buckets is not None else Buckets(n_buckets).fit(self.position[tr][:, None, :], np.ones((int(tr.sum()), 1), bool))
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
        self.played_from = np.zeros(N, dtype=np.int64); self.played_to = np.zeros(N, dtype=np.int64); self.next = np.full(N, -1, dtype=np.int64)
        # soft labels: the teacher's win probability for every legal move where the corpus has them (ChessBench)
        self.q = np.zeros((N, MAX_PIECES, 64), dtype=np.float16); self.has_q = np.zeros(N, dtype=bool)
        for i, m in enumerate(self.meta):
            board = chess.Board(m["fen"]); sq_index = {s: j for j, s in enumerate(m["squares"]) if s < 64}
            for mv in board.legal_moves:
                self.legal[i, sq_index[mv.from_square], mv.to_square] = True
            if i in self.hidden:
                continue                                                               # the pool: legal moves known, labels withheld
            if m["best"]:
                bm = chess.Move.from_uci(m["best"]); self.best[i] = sq_index[bm.from_square] * 64 + bm.to_square
            labels = m.get("labels")
            if labels:
                self.has_q[i] = True
                for u, wp in labels.items():
                    mv = chess.Move.from_uci(u)
                    if mv.from_square in sq_index:
                        self.q[i, sq_index[mv.from_square], mv.to_square] = wp
            pm = chess.Move.from_uci(m["played"]); self.played_from[i] = sq_index[pm.from_square]; self.played_to[i] = pm.to_square
            self.next[i] = self.index.get((m["game"], m["ply"] + 1), -1)
        reg = np.array([regime_of(chess.Board(m["fen"]), bool(m["side"])) for m in self.meta], dtype=np.float32)
        self.reg_mean = reg[tr].mean(0); self.reg_std = reg[tr].std(0) + 1e-6
        self.regime_now = (reg - self.reg_mean) / self.reg_std
        self.ahead = {h: (np.array([m["ahead"][str(h)] for m in self.meta], dtype=np.float32) - self.reg_mean) / self.reg_std for h in (10, 20)}
        cps = np.array([m["best_cp"] if m["best_cp"] is not None else 0 for m in self.meta], dtype=np.float32)
        self.cp = np.tanh(cps / 400.0)
        # the quantized future: the codebook (fit on the training games when absent), the code now and at +10 / +20
        # (the game's last position when it ended before), the engine's evaluation there (the same side to move)
        res_f = np.array([-1.0 if m["result"] is None else m["result"] for m in self.meta])
        cb_path = Path(path.split(",")[0]) / "codebook.npz"                                  # the first corpus owns the codebook
        self.codebook = Codebook.load(str(cb_path)) if cb_path.exists() else fit_codebook(self.position, tr, res_f, n_codes)
        if not cb_path.exists():
            self.codebook.save(str(cb_path))
        self.code_now = self.codebook.assign(self.position)
        last = {}
        for i, m in enumerate(self.meta):
            if m["game"] not in last or m["ply"] > self.meta[last[m["game"]]]["ply"]:
                last[m["game"]] = i
        self.fut = {}; self.has_fut = {}
        for h in (10, 20):
            f = np.array([self.index.get((m["game"], m["ply"] + h), -1) for m in self.meta])
            self.has_fut[h] = f >= 0
            self.fut[h] = np.where(f >= 0, f, np.array([last[m["game"]] for m in self.meta]))
        self.code_ahead = {h: self.code_now[self.fut[h]] for h in (10, 20)}
        self.cp_ahead = {h: self.cp[self.fut[h]] for h in (10, 20)}
        # the window labels: a big threat in either ledger anywhere in the next 10 plies; the engine's best move
        # forcing in at least half of them (the sharpness the planner budgets by)
        PQ = {q: i for i, q in enumerate(POSITION_QUESTIONS)}
        big = np.maximum(self.position[:, PQ["threat_ours"]], self.position[:, PQ["threat_theirs"]]) >= 3.0
        forcing = np.array([m["best_kind"] != "quiet" for m in self.meta])
        self.sharp_window = np.zeros(N, dtype=np.float32); self.forcing_window = np.zeros(N, dtype=np.float32); self.has_window = np.zeros(N, dtype=bool)
        for i, m in enumerate(self.meta):
            ids = [self.index.get((m["game"], m["ply"] + j), -1) for j in range(1, 11)]
            ids = [j for j in ids if j >= 0]
            if ids:
                self.has_window[i] = True; self.sharp_window[i] = float(any(big[j] for j in ids)); self.forcing_window[i] = float(np.mean([forcing[j] for j in ids]) >= 0.5)
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
                   game=torch.as_tensor(self.game[idx]), ply=torch.as_tensor(self.ply[idx]), i=torch.as_tensor(idx),
                   played_from=torch.as_tensor(self.played_from[idx]), played_to=torch.as_tensor(self.played_to[idx]), next=torch.as_tensor(self.next[idx]),
                   code_now=torch.as_tensor(self.code_now[idx]), code10=torch.as_tensor(self.code_ahead[10][idx]), code20=torch.as_tensor(self.code_ahead[20][idx]),
                   cp10=torch.as_tensor(self.cp_ahead[10][idx]), cp20=torch.as_tensor(self.cp_ahead[20][idx]),
                   has10=torch.as_tensor(self.has_fut[10][idx]), has20=torch.as_tensor(self.has_fut[20][idx]),
                   sharp_window=torch.as_tensor(self.sharp_window[idx]), forcing_window=torch.as_tensor(self.forcing_window[idx]), has_window=torch.as_tensor(self.has_window[idx]),
                   q=torch.as_tensor(self.q[idx]).float(), has_q=torch.as_tensor(self.has_q[idx]),
                   pf10=torch.as_tensor(self.pf_b[10][idx]).long(), pf20=torch.as_tensor(self.pf_b[20][idx]).long(),
                   pf_has10=torch.as_tensor(self.pf_has[10][idx]), pf_has20=torch.as_tensor(self.pf_has[20][idx]),
                   fate10=torch.as_tensor(self.fate[10][idx]).long(), fate20=torch.as_tensor(self.fate[20][idx]).long())
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
    if "has10" in b:
        h10, h20 = b["has10"], b["has20"]
        r10 = F.mse_loss(reg[h10, 0], b["ahead10"][h10]) if h10.any() else phi.sum() * 0
        r20 = F.mse_loss(reg[h20, 1], b["ahead20"][h20]) if h20.any() else phi.sum() * 0
        out["regime"] = 0.5 * (r10 + r20)
    else:
        out["regime"] = 0.5 * (F.mse_loss(reg[:, 0], b["ahead10"]) + F.mse_loss(reg[:, 1], b["ahead20"]))
    pl = model.policy_logits(h, b).view(phi.shape[0], -1)
    has = b["best"] >= 0
    hq = b["has_q"] if "has_q" in b else torch.zeros_like(has)
    hard = has & ~hq
    out["policy"] = F.cross_entropy(pl[hard], b["best"][hard]) if hard.any() else phi.sum() * 0
    if hq.any():
        # the soft target: the teacher's win probabilities over the legal moves, sharpened; the same head, so the
        # two corpora train one policy
        qt = b["q"].view(phi.shape[0], -1)[hq] / cfg.q_temp
        qt = qt.masked_fill(~b["legal"].view(phi.shape[0], -1)[hq], float("-inf"))
        target = F.softmax(qt, dim=-1)
        lsm = F.log_softmax(pl[hq], dim=-1).masked_fill(target <= 0, 0.0)               # 0 x -inf would print NaN (the gradient was finite)
        out["policy_soft"] = -(target * lsm).sum(-1).mean()
        ql = model.q_logits(h, b).view(phi.shape[0], -1)[hq]
        lg = b["legal"].view(phi.shape[0], -1)[hq]
        out["q"] = F.binary_cross_entropy_with_logits(ql[lg], b["q"].view(phi.shape[0], -1)[hq][lg])
    else:
        out["policy_soft"] = phi.sum() * 0; out["q"] = phi.sum() * 0
    if "code10" in b:
        fl = model.future_head(phi).view(-1, 2, cfg.n_codes)
        h10, h20 = b["has10"], b["has20"]
        f10 = F.cross_entropy(fl[h10, 0], b["code10"][h10]) if h10.any() else phi.sum() * 0
        f20 = F.cross_entropy(fl[h20, 1], b["code20"][h20]) if h20.any() else phi.sum() * 0
        out["future"] = 0.5 * (f10 + f20)
        va = model.value_ahead(phi, b["cp"])
        h10, h20 = b["has10"], b["has20"]                                                # only positions with a real future: the game's last
        v10 = F.mse_loss(va[h10, 0], b["cp10"][h10]) if h10.any() else phi.sum() * 0    # evaluation is often a mate score and would teach
        v20 = F.mse_loss(va[h20, 1], b["cp20"][h20]) if h20.any() else phi.sum() * 0    # 'late = extreme' to positions that go on
        out["value"] = 0.5 * (v10 + v20)
        hw = b["has_window"]
        if hw.any():
            sl = model.sharp_head(phi)[hw]
            out["sharp"] = 0.5 * (F.binary_cross_entropy_with_logits(sl[:, 0], b["sharp_window"][hw]) + F.binary_cross_entropy_with_logits(sl[:, 1], b["forcing_window"][hw]))
        else:
            out["sharp"] = phi.sum() * 0
    if k is not None and n_anchor > 0:
        psi = model.psi(phi)
        a, p = psi[:n_anchor], psi[n_anchor:2 * n_anchor]
        d_fwd = model.iqe(a, p)
        # the quasimetric objective (Wang & Isola) over the WHOLE batch, both directions: d(x -> y) for every ordered
        # pair; pairs known reachable (same game, y later than x) are bounded by their ply gap; every other pair --
        # other games AND the reverse of time within a game -- is pushed apart. A knight's hop back is pulled close
        # again wherever the same hop appears forward in another game; a pawn's never is. Nothing names pawns.
        D = model.iqe.pairwise(psi, psi)                                                 # (B, B): d(x -> y)
        games, plies = b["game"], b["ply"]
        same = games[:, None] == games[None, :]
        later = same & (plies[None, :] > plies[:, None])
        gap = (plies[None, :] - plies[:, None]).float().clamp(min=1.0)
        eye = torch.eye(phi.shape[0], dtype=torch.bool, device=phi.device)
        pushed = (~later) & ~eye if cfg.neg_mode == "qrl" else (~same)
        out["traj_neg"] = F.relu(cfg.neg_margin - D)[pushed].mean() if pushed.any() else phi.sum() * 0        # a linear hinge in plies: bounded gradient
        out["traj_bound"] = F.relu(D - gap)[later].mean() if later.any() else phi.sum() * 0                     # reachable in `gap` plies: no farther than that
        out["traj_fwd"] = F.smooth_l1_loss(d_fwd, k.float())                                                     # calibrated in plies
        # the geometry's own spread: VICReg's variance and covariance on psi, so the repulsion cannot be met by a
        # ply clock and a game id alone (the collapse of the first run, now in psi's coordinates)
        if psi.shape[0] >= 2:
            zp = psi - psi.mean(0, keepdim=True)
            sp = torch.sqrt(zp.var(0, unbiased=False) + 1e-4)
            covp = (zp.T @ zp) / max(1, psi.shape[0] - 1)
            out["psi_vic"] = F.relu(1.0 - sp).mean() + (covp - torch.diag(torch.diag(covp))).pow(2).sum() / psi.shape[1]
        else:
            out["psi_vic"] = phi.sum() * 0
    # the move-conditioned readout: for anchors whose NEXT position is in the batch (the k=1 partners), the predicted
    # psi after the played move against the next position's psi (a fixed target: the geometry is shaped elsewhere)
    if "next" in b and "played_from" in b:
        pos = {int(i): n for n, i in enumerate(b["i"].tolist())}
        src, dst = [], []
        for n, j in enumerate(b["next"].tolist()):
            if j in pos:
                src.append(n); dst.append(pos[j])
        if src:
            src_t = torch.tensor(src, device=phi.device); dst_t = torch.tensor(dst, device=phi.device)
            pred = model.predicted_psi(h[src_t], phi[src_t], b["played_from"][src_t], b["played_to"][src_t])
            target = model.psi(phi[dst_t]).detach()
            out["delta"] = F.smooth_l1_loss(pred, target)
        else:
            out["delta"] = phi.sum() * 0
    # VICReg's variance and covariance terms on phi: every dimension keeps a spread of one, dimensions decorrelate
    if phi.shape[0] >= 2:
        z = phi - phi.mean(0, keepdim=True)
        std = torch.sqrt(z.var(0, unbiased=False) + 1e-4)
        out["vic_var"] = F.relu(1.0 - std).mean()
        cov = (z.T @ z) / max(1, phi.shape[0] - 1)
        out["vic_cov"] = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / phi.shape[1]
    else:
        out["vic_var"] = phi.sum() * 0; out["vic_cov"] = phi.sum() * 0
    if cfg.w_piece_future > 0 and "pf_has10" in b:                                     # a batch built by hand (tests, the app) may carry no piece futures
        pfl, fl_ = model.piece_future(o["h"])
        terms = []
        for j, hz in enumerate((10, 20)):
            m_ = b[f"pf_has{hz}"]
            if m_.any():
                terms.append(F.cross_entropy(pfl[:, :, j][m_].reshape(-1, cfg.buckets), b[f"pf{hz}"][m_].reshape(-1)))
            fm = b["mask"] & (b[f"fate{hz}"] >= 0)
            if fm.any():
                terms.append(F.cross_entropy(fl_[:, :, j][fm], b[f"fate{hz}"][fm]))
        out["piece_future"] = sum(terms) / max(1, len(terms)) if terms else phi.sum() * 0
    total = out["wdl"] + cfg.w_cp * out["cp"] + cfg.w_regime * out["regime"] + cfg.w_policy * (out["policy"] + out["policy_soft"]) + cfg.w_q * out["q"] + cfg.w_vic * (out["vic_var"] + out["vic_cov"]) + cfg.w_delta * out.get("delta", 0.0) \
        + cfg.w_future * out.get("future", 0.0) + cfg.w_value * out.get("value", 0.0) + cfg.w_sharp * out.get("sharp", 0.0) + cfg.w_piece_future * out.get("piece_future", 0.0)
    if "traj_fwd" in out:
        total = total + cfg.w_traj * (out["traj_fwd"] + out["traj_neg"] + out["traj_bound"]) + cfg.w_vic * out["psi_vic"]
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
    pol_ok = pol_n = 0; quiet_ok = quiet_n = 0; cap_ok = cap_n = 0; d_ok = dquiet_ok = 0
    from collections import defaultdict
    q_stats = defaultdict(list)
    cp_pred, cp_true = [], []
    reg_pred = {10: [], 20: []}; reg_true = {10: [], 20: []}; reg_now = []
    phis = []
    fut_logits = {10: [], 20: []}; fut_true = {10: [], 20: []}; fut_now = []; fut_has = {10: [], 20: []}
    val_pred = {10: [], 20: []}; val_true = {10: [], 20: []}; val_now = []
    sh_logit = []; sh_true = []; sh_has = []; sh_now = []
    pf_pred = {10: [], 20: []}; pf_prob0 = {10: [], 20: []}; pf_true = {10: [], 20: []}; pf_now = {10: [], 20: []}; fate_prob = {10: [], 20: []}; fate_true = {10: [], 20: []}
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
        mv = model.move_values(o["h"], o["phi"], b).view(len(chunk), -1)
        hit_d = (mv.argmax(-1) == b["best"]) & has
        if b["has_q"].any():
            hq = b["has_q"]; qv = b["q"].view(len(chunk), -1)
            ql = model.q_logits(o["h"], b).view(len(chunk), -1)
            best_wp = qv.max(-1).values
            for name, pick in (("policy", pl.argmax(-1)), ("qhead", ql.argmax(-1))):
                chosen = qv[torch.arange(len(chunk), device=qv.device), pick]
                q_stats[name + "_regret"].extend((best_wp - chosen)[hq].tolist()); q_stats[name + "_top1"].extend((pick == b["best"])[hq].tolist())
            q_stats["q_mae"].extend((torch.sigmoid(ql) - qv)[b["legal"].view(len(chunk), -1) & hq[:, None]].abs().tolist())
        for j, ii in enumerate(chunk):
            kind = data.meta[int(ii)]["best_kind"]
            if kind == "quiet":
                quiet_n += 1; quiet_ok += int(hit[j].item()); dquiet_ok += int(hit_d[j].item())
            elif kind == "capture":
                cap_n += 1; cap_ok += int(hit[j].item())
        d_ok += hit_d.sum().item()
        cp_pred.append(model.cp_head(o["phi"]).squeeze(-1).cpu().numpy()); cp_true.append(b["cp"].cpu().numpy())
        reg = model.regime_head(o["phi"]).view(len(chunk), 2, K_REG).cpu().numpy()
        fl = model.future_head(o["phi"]).view(len(chunk), 2, cfg.n_codes).cpu().numpy(); va = model.value_ahead(o["phi"], b["cp"]).cpu().numpy()
        for j, hz in enumerate((10, 20)):
            fut_logits[hz].append(fl[:, j]); fut_true[hz].append(b[f"code{hz}"].cpu().numpy()); fut_has[hz].append(b[f"has{hz}"].cpu().numpy())
            val_pred[hz].append(va[:, j]); val_true[hz].append(b[f"cp{hz}"].cpu().numpy())
        fut_now.append(b["code_now"].cpu().numpy()); val_now.append(b["cp"].cpu().numpy())
        sh_logit.append(model.sharp_head(o["phi"]).cpu().numpy()); sh_true.append(np.stack([b["sharp_window"].cpu().numpy(), b["forcing_window"].cpu().numpy()], 1)); sh_has.append(b["has_window"].cpu().numpy())
        sh_now.append(np.maximum(data.position[chunk, POSITION_QUESTIONS.index("threat_ours")], data.position[chunk, POSITION_QUESTIONS.index("threat_theirs")]) >= 3.0)
        pfl, fl_ = model.piece_future(o["h"])
        for j, hz in enumerate((10, 20)):
            if f"pf_has{hz}" not in b:
                continue
            m_ = b[f"pf_has{hz}"].cpu().numpy()
            fm_q = PIECE_FUTURE_QUESTIONS.index("free_moves"); b0 = int(np.searchsorted(data.buckets.edges[data.pf_q[fm_q]], 0.0, side="right"))   # the bucket of 'no free moves'
            pf_pred[hz].append(pfl[:, :, j].argmax(-1).cpu().numpy()[m_]); pf_prob0[hz].append(torch.softmax(pfl[:, :, j, fm_q], -1)[..., b0].cpu().numpy()[m_])
            pf_true[hz].append(b[f"pf{hz}"].cpu().numpy()[m_]); pf_now[hz].append(b["buckets"].cpu().numpy()[..., data.pf_q][m_])
            fm = (b["mask"] & (b[f"fate{hz}"] >= 0)).cpu().numpy()
            fate_prob[hz].append(torch.softmax(fl_[:, :, j], -1).cpu().numpy()[fm]); fate_true[hz].append(b[f"fate{hz}"].cpu().numpy()[fm])
        reg_pred[10].append(reg[:, 0]); reg_pred[20].append(reg[:, 1])
        reg_true[10].append(b["ahead10"].cpu().numpy()); reg_true[20].append(b["ahead20"].cpu().numpy()); reg_now.append(b["regime_now"].cpu().numpy())
        phis.append(o["phi"].cpu().numpy())
    for hz in (10, 20):
        if not pf_true[hz] or sum(len(x) for x in pf_true[hz]) == 0:
            continue
        P_, T_, N_, P0 = (np.concatenate(pf_pred[hz]), np.concatenate(pf_true[hz]), np.concatenate(pf_now[hz]), np.concatenate(pf_prob0[hz]))
        for qi, name in enumerate(PIECE_FUTURE_QUESTIONS):
            stats[f"pf{hz}_{name}_acc"] = float((P_[:, qi] == T_[:, qi]).mean()); stats[f"pf{hz}_{name}_persist"] = float((N_[:, qi] == T_[:, qi]).mean())
            stats[f"pf{hz}_{name}_marginal"] = float(np.bincount(T_[:, qi]).max() / len(T_))
        # the headline: INACTIVE at +h = free_moves in the lowest bucket; AUC of the head's probability vs persistence (now inactive)
        qi = PIECE_FUTURE_QUESTIONS.index("free_moves"); b0 = int(np.searchsorted(data.buckets.edges[data.pf_q[qi]], 0.0, side="right")); y = (T_[:, qi] == b0)
        if 0 < y.sum() < len(y):
            from sklearn.metrics import roc_auc_score
            stats[f"pf{hz}_inactive_auc"] = float(roc_auc_score(y, P0)); stats[f"pf{hz}_inactive_persist_auc"] = float(roc_auc_score(y, (N_[:, qi] == b0).astype(float)))
            stats[f"pf{hz}_inactive_base"] = float(y.mean())
        FP, FT = np.concatenate(fate_prob[hz]), np.concatenate(fate_true[hz])
        y = (FT == 1)
        if 0 < y.sum() < len(y):
            from sklearn.metrics import roc_auc_score
            stats[f"fate{hz}_exchanged_auc"] = float(roc_auc_score(y, FP[:, 1])); stats[f"fate{hz}_exchanged_base"] = float(y.mean())
            stats[f"fate{hz}_gone_auc"] = float(roc_auc_score(FT >= 1, 1.0 - FP[:, 0])); stats[f"fate{hz}_gone_base"] = float((FT >= 1).mean())
        stats[f"fate{hz}_acc"] = float((FP.argmax(1) == FT).mean()); stats[f"fate{hz}_majority"] = float(np.bincount(FT).max() / len(FT))
    stats["wdl_acc"] = wdl_ok / max(1, wdl_n); stats["wdl_majority"] = wdl_maj / max(1, wdl_n); stats["wdl_n"] = wdl_n
    stats["policy_top1"] = pol_ok / max(1, pol_n); stats["policy_quiet"] = quiet_ok / max(1, quiet_n); stats["policy_capture"] = cap_ok / max(1, cap_n); stats["quiet_n"] = quiet_n
    stats["delta_top1"] = d_ok / max(1, pol_n); stats["delta_quiet"] = dquiet_ok / max(1, quiet_n)
    if q_stats:
        stats["labeled_n"] = len(q_stats["policy_top1"])
        for k_, v in q_stats.items():
            stats["cb_" + k_] = float(np.mean(v)) if v else float("nan")
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
    now_c = np.concatenate(fut_now); now_v = np.concatenate(val_now)
    train_codes = data.code_now[data.ids("train")]; marginal = np.bincount(train_codes, minlength=cfg.n_codes) / len(train_codes)
    sharp = data.codebook.sharp()
    for hz in (10, 20):
        L = np.concatenate(fut_logits[hz]); T = np.concatenate(fut_true[hz]); H = np.concatenate(fut_has[hz]).astype(bool)
        L, T, nc = L[H], T[H], now_c[H]
        pr = np.exp(L - L.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        stats[f"code{hz}_top1"] = float((pr.argmax(1) == T).mean()); stats[f"code{hz}_top3"] = float(np.mean([t in np.argsort(-p)[:3] for p, t in zip(pr, T)]))
        stats[f"code{hz}_nll"] = float(-np.log(pr[np.arange(len(T)), T] + 1e-9).mean())
        stats[f"code{hz}_persist_top1"] = float((nc == T).mean()); stats[f"code{hz}_marginal_top1"] = float((T == marginal.argmax()).mean())
        stats[f"code{hz}_marginal_nll"] = float(-np.log(marginal[T] + 1e-9).mean())
        stats[f"code{hz}_entropy"] = float(-(pr * np.log(pr + 1e-9)).sum(1).mean())
        # the sharpness read off the distribution: the mass on sharp codes against the actual sharpness at t+hz
        y = sharp[T]
        if 0 < y.sum() < len(y):
            from sklearn.metrics import roc_auc_score
            stats[f"sharp{hz}_auc"] = float(roc_auc_score(y, pr[:, sharp].sum(1))); stats[f"sharp{hz}_persist_auc"] = float(roc_auc_score(y, sharp[nc].astype(float)))
        # the value read off the distribution (the codes' mean results) against the engine's evaluation there
        ev = pr @ data.codebook.values
        vt = np.concatenate(val_true[hz])[H]; vp = np.concatenate(val_pred[hz])[H]; vn = now_v[H]
        var = vt.var() + 1e-9
        stats[f"value{hz}_r2"] = float(1 - ((vp - vt) ** 2).mean() / var); stats[f"value{hz}_persist_r2"] = float(1 - ((vn - vt) ** 2).mean() / var)
        stats[f"value{hz}_from_codes_corr"] = float(np.corrcoef(ev, vt)[0, 1])
    SL = np.concatenate(sh_logit); ST = np.concatenate(sh_true); SH = np.concatenate(sh_has).astype(bool); SN = np.concatenate(sh_now)
    if SH.any():
        from sklearn.metrics import roc_auc_score
        for j, name in enumerate(("sharp_window", "forcing_window")):
            y = ST[SH, j]
            if 0 < y.sum() < len(y):
                stats[f"{name}_auc"] = float(roc_auc_score(y, SL[SH, j])); stats[f"{name}_base"] = float(y.mean())
        y = ST[SH, 0]
        if 0 < y.sum() < len(y):
            stats["sharp_window_persist_auc"] = float(roc_auc_score(y, SN[SH].astype(float)))
    phi = np.concatenate(phis)
    stats["effective_rank"] = effective_rank(phi)
    stats["phi_std"] = float(phi.std())
    with torch.no_grad():
        psi = model.psi(torch.as_tensor(phi, device=device)).cpu().numpy()
    stats["psi_rank"] = effective_rank(psi)
    stats["psi_std"] = float(psi.std())
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
        pa, pb = model.psi(model(ba)["phi"]), model.psi(model(bb)["phi"])
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
    data = Data(args.data, n_buckets=cfg.buckets, n_codes=cfg.n_codes, hide_labels=HIDE_LABELS)
    print(f"[kn] data: {len(data.meta)} positions; train {len(data.ids('train'))} val {len(data.ids('val'))} test {len(data.ids('test'))}; loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    model = KnowledgeNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[kn] model {cfg.variant}: {n_params / 1e6:.2f}M params, d={cfg.d} layers={cfg.layers} heads={cfg.heads} buckets={cfg.buckets}; device {device}", flush=True)
    # the geometry's scalars (the quasimetric's scale and mix, the basin temperature) move at their own pace: Adam
    # steps a scalar by ~lr per step, and a log-scale that must travel by 3 would need ten thousand steps at 3e-4
    geo = [model.iqe.log_scale, model.iqe.alpha_logit, model.log_T]
    geo_ids = {id(p) for p in geo}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in geo_ids]},
                             {"params": geo, "lr": cfg.lr * cfg.geo_lr_mult, "weight_decay": 0.0}], lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_ids = data.ids("train"); rng = random.Random(cfg.seed)
    steps_per_epoch = len(train_ids) // cfg.batch
    total_steps = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg.lr, cfg.lr * cfg.geo_lr_mult], total_steps=max(1, total_steps), pct_start=0.05)
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
        cb = f" | LABELED {ev['labeled_n']}: policy top1 {ev['cb_policy_top1']:.3f} regret {ev['cb_policy_regret']:.4f}; q-head top1 {ev['cb_qhead_top1']:.3f} regret {ev['cb_qhead_regret']:.4f} mae {ev['cb_q_mae']:.4f}" if "labeled_n" in ev else ""
        print(f"[kn] epoch {epoch} val: wdl {ev['wdl_acc']:.3f} (majority {ev['wdl_majority']:.3f}) policy {ev['policy_top1']:.3f} quiet {ev['policy_quiet']:.3f} capture {ev['policy_capture']:.3f} delta-pick {ev['delta_top1']:.3f} (quiet {ev['delta_quiet']:.3f}){cb} "
              f"cp_r2 {ev['cp_r2']:.3f} regime10 {ev['regime10_r2']:.3f} (persist {ev['regime10_persist_r2']:.3f}) regime20 {ev['regime20_r2']:.3f} (persist {ev['regime20_persist_r2']:.3f}) "
              f"eff_rank {ev['effective_rank']:.1f} psi_rank {ev['psi_rank']:.1f} iqe_scale {math.exp(float(model.iqe.log_scale)):.2f} | {time.perf_counter() - t0:.0f}s", flush=True)
        print(f"[kn]   future: code10 top1 {ev['code10_top1']:.3f} (persist {ev['code10_persist_top1']:.3f}, marginal {ev['code10_marginal_top1']:.3f}) top3 {ev['code10_top3']:.3f} nll {ev['code10_nll']:.3f} (marginal {ev['code10_marginal_nll']:.3f}) "
              f"| code20 top1 {ev['code20_top1']:.3f} (persist {ev['code20_persist_top1']:.3f}) nll {ev['code20_nll']:.3f} (marginal {ev['code20_marginal_nll']:.3f}) "
              f"| sharp10 auc {ev.get('sharp10_auc', float('nan')):.3f} (persist {ev.get('sharp10_persist_auc', float('nan')):.3f}) sharp20 {ev.get('sharp20_auc', float('nan')):.3f} "
              f"| value10 r2 {ev['value10_r2']:.3f} (persist {ev['value10_persist_r2']:.3f}) value20 {ev['value20_r2']:.3f} (persist {ev['value20_persist_r2']:.3f}) from-codes corr {ev['value10_from_codes_corr']:.3f}/{ev['value20_from_codes_corr']:.3f} "
              f"| sharp-window auc {ev.get('sharp_window_auc', float('nan')):.3f} (persist {ev.get('sharp_window_persist_auc', float('nan')):.3f}, base {ev.get('sharp_window_base', float('nan')):.2f}) forcing-window auc {ev.get('forcing_window_auc', float('nan')):.3f}", flush=True)
        if "pf20_inactive_auc" in ev:
            print(f"[kn]   pieces: inactive@20 auc {ev['pf20_inactive_auc']:.3f} (persist {ev['pf20_inactive_persist_auc']:.3f}, base {ev['pf20_inactive_base']:.2f}) inactive@10 auc {ev.get('pf10_inactive_auc', float('nan')):.3f} (persist {ev.get('pf10_inactive_persist_auc', float('nan')):.3f}) "
                  f"| free_moves@20 acc {ev['pf20_free_moves_acc']:.3f} (persist {ev['pf20_free_moves_persist']:.3f}, marginal {ev['pf20_free_moves_marginal']:.3f}) king_zone_hits@20 acc {ev['pf20_king_zone_hits_acc']:.3f} (persist {ev['pf20_king_zone_hits_persist']:.3f}) "
                  f"| exchanged@20 auc {ev.get('fate20_exchanged_auc', float('nan')):.3f} (base {ev.get('fate20_exchanged_base', float('nan')):.3f}) gone@20 auc {ev.get('fate20_gone_auc', float('nan')):.3f} (base {ev.get('fate20_gone_base', float('nan')):.2f}) fate acc {ev['fate20_acc']:.3f} (majority {ev['fate20_majority']:.3f})", flush=True)
        torch.save(dict(model=model.state_dict(), config=asdict(cfg), epoch=epoch, step=step), out / f"epoch_{epoch:03d}.pt")
        torch.save(dict(model=model.state_dict(), config=asdict(cfg), epoch=epoch, step=step), out / "last.pt")
        if best_val is None or ev["policy_top1"] > best_val:
            best_val = ev["policy_top1"]; torch.save(dict(model=model.state_dict(), config=asdict(cfg), epoch=epoch, step=step), out / "best.pt")
    ev = evaluate(model, data, "test", cfg, device, max_n=200000)                      # the whole test split: several corpora may be joined
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
    _, d = model.basin_logits(phi)                                                       # through psi = W phi: still linear in the contributions
    d[0, pole].backward()
    grad = phi.grad[0].detach()                                                         # (d,)
    a = o["alpha"][0].detach(); senses = model.sense(model.sense_index({k: v[:1] for k, v in b.items()}))[0].detach()     # (32, Q+2), (32, Q, d)
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
    """Older checkpoints (before the psi projection and the delta head) load with those modules at their init and
    the poles in phi's space: their geometry read through psi is meaningless, their policy and phi heads are intact."""
    ck = torch.load(ckpt_path, map_location="cpu")
    conf = dict(ck["config"])
    if "d_geo" not in conf:                                    # a pre-projection checkpoint: its poles and IQE live on phi
        conf["d_geo"] = conf["d"]
    cfg = Config(**{k: v for k, v in conf.items() if k in Config.__dataclass_fields__}); model = KnowledgeNet(cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    model.loaded_modules = {k.split(".")[0] for k in ck["model"]}                # heads a checkpoint never trained stay at init: callers must not read them
    if missing:
        print(f"[kn] {ckpt_path}: {len(missing)} modules at init (pre-projection checkpoint): {sorted({m.split('.')[0] for m in missing})}", flush=True)
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
        data = Data(a.data, buckets=Buckets.from_state(cfg.buckets, meta["buckets"]), n_buckets=cfg.buckets, pos_buckets=Buckets.from_state(cfg.buckets, meta["pos_buckets"]))
        ev = evaluate(model, data, "test", cfg, device); print(json.dumps({k: v for k, v in ev.items()}, indent=1))
        print(json.dumps(asymmetry(model, data, "test", device), indent=1))


if __name__ == "__main__":
    main()
