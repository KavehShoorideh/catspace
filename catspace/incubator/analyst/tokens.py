"""INVARIANT TOKENS FROM THE TABLEBASE (Kaveh 2026-09-19: "learn invariant tokens that help predict position value,
then root them in the graph operations").

Input: the 64 squares as tokens (the piece on each, White's view, the side to move as a token), with the RULES as
the only relations between them -- for each ordered pair of squares: the piece on i attacks j; the piece on i
can move to j; i and j are on a line (rank, file, diagonal) -- as attention biases. Nothing else is given.
Bottleneck: K binary tokens per position (straight-through sigmoid), few on. The result (win / draw / loss, White's
view) is predicted FROM THE TOKENS ALONE, so the tokens must carry it. COHERENCE across a reply: when the result
after the reply is the same, the tokens must not change; when it differs, some token must. SPARSITY: few tokens
on. Gates: code usage (a code that never fires or always fires is dead), the effective rank of the soft tokens,
accuracy on held-out positions AND on held-out materials, beside the majority per material and a plain net
without the bottleneck (--plain)."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .endgames import load as load_rows, split_of

OUT = Path("data/derived/checkpoints/tokens")
PIECE_ID = {(chess.WHITE, chess.KING): 1, (chess.WHITE, chess.QUEEN): 2, (chess.WHITE, chess.ROOK): 3, (chess.WHITE, chess.BISHOP): 4, (chess.WHITE, chess.KNIGHT): 5, (chess.WHITE, chess.PAWN): 6,
            (chess.BLACK, chess.KING): 7, (chess.BLACK, chess.QUEEN): 8, (chess.BLACK, chess.ROOK): 9, (chess.BLACK, chess.BISHOP): 10, (chess.BLACK, chess.KNIGHT): 11, (chess.BLACK, chess.PAWN): 12}
N_REL = 3


def encode(board: chess.Board):
    """(64,) piece ids, side, (64, 64, 3) relations from the rules: attacks, can move to, same line."""
    sq = np.zeros(64, np.int64)
    rel = np.zeros((64, 64, N_REL), np.int8)
    for s, p in board.piece_map().items():
        sq[s] = PIECE_ID[(p.color, p.piece_type)]
        att = board.attacks(s)
        for t in att:
            rel[s, t, 0] = 1
        b = board.copy(stack=False); b.turn = p.color
        if b.is_valid():
            for m in b.legal_moves:
                if m.from_square == s:
                    rel[s, m.to_square, 1] = 1
    for s in range(64):
        rel[s, :, 2] = [(chess.square_rank(s) == chess.square_rank(t) or chess.square_file(s) == chess.square_file(t) or abs(chess.square_rank(s) - chess.square_rank(t)) == abs(chess.square_file(s) - chess.square_file(t))) for t in range(64)]
    return sq, int(board.turn), rel


def white_wdl(wdl_mover: int, white_to_move: bool) -> int:
    v = 1 if wdl_mover > 0 else -1 if wdl_mover < 0 else 0
    return v if white_to_move else -v


class Data:
    def __init__(self, rows: list[dict], held_materials: set, log=print):
        t0 = time.perf_counter()
        self.rows = rows
        N = len(rows)
        self.sq = np.zeros((N, 64), np.int64); self.side = np.zeros(N, np.int64); self.rel = np.zeros((N, 64, 64, N_REL), np.int8)
        self.wdl = np.zeros(N, np.int64)
        self.csq = np.zeros((N, 64), np.int64); self.cside = np.zeros(N, np.int64); self.crel = np.zeros((N, 64, 64, N_REL), np.int8); self.cwdl = np.zeros(N, np.int64); self.has_child = np.zeros(N, bool)
        self.split = np.empty(N, dtype="<U5"); self.material = np.array([r["material"] for r in rows])
        for i, r in enumerate(rows):
            b = chess.Board(r["fen"])
            self.sq[i], self.side[i], self.rel[i] = encode(b)
            self.wdl[i] = white_wdl(r["wdl"], b.turn) + 1                                      # 0 loss, 1 draw, 2 win (White)
            self.split[i] = split_of(r, held_materials)
            if r.get("child"):
                cb = chess.Board(r["child"]["fen"])
                self.csq[i], self.cside[i], self.crel[i] = encode(cb)
                self.cwdl[i] = white_wdl(r["child"]["wdl"], b.turn) + 1                          # the child's wdl is stored in the PARENT mover's view
                self.has_child[i] = True
        log(f"[tokens] {N} positions encoded in {time.perf_counter() - t0:.0f}s; splits {dict(zip(*np.unique(self.split, return_counts=True)))}", flush=True)

    def ids(self, split):
        return np.nonzero(self.split == split)[0]

    def batch(self, idx, device):
        idx = np.asarray(idx)
        out = dict(sq=torch.as_tensor(self.sq[idx]), side=torch.as_tensor(self.side[idx]), rel=torch.as_tensor(self.rel[idx]).float(), wdl=torch.as_tensor(self.wdl[idx]),
                   csq=torch.as_tensor(self.csq[idx]), cside=torch.as_tensor(self.cside[idx]), crel=torch.as_tensor(self.crel[idx]).float(), cwdl=torch.as_tensor(self.cwdl[idx]), has_child=torch.as_tensor(self.has_child[idx]))
        return {k: v.to(device) for k, v in out.items()}


@dataclass
class Config:
    d: int = 64
    layers: int = 4
    heads: int = 4
    k: int = 32                      # binary tokens
    lr: float = 1e-3
    batch: int = 256
    epochs: int = 10
    w_coh: float = 1.0               # coherence across a reply
    w_sparse: float = 0.05           # mean activation
    w_use: float = 0.5               # every code used (batch entropy)
    plain: bool = False              # no bottleneck: the result from the board token (the ceiling / control)
    coh_margin: float = 1.0
    seed: int = 1


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.heads = heads
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, bias):
        B, T, d = x.shape; h = self.heads
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, h, d // h).unbind(2)
        att = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(d // h) + bias
        x = x + self.o(torch.einsum("bhts,bshd->bthd", att.softmax(-1), v).reshape(B, T, d))
        return x + self.ff(self.ln2(x))


class TokenNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg; d = cfg.d
        self.piece = nn.Embedding(13, d); self.square = nn.Embedding(64, d); self.side = nn.Embedding(2, d); self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.rel_bias = nn.Linear(N_REL, cfg.heads, bias=False)
        self.blocks = nn.ModuleList([Block(d, cfg.heads) for _ in range(cfg.layers)])
        self.ln = nn.LayerNorm(d)
        self.to_tokens = nn.Linear(d, cfg.k)
        self.wdl_from_tokens = nn.Sequential(nn.Linear(cfg.k, 2 * d), nn.GELU(), nn.Linear(2 * d, 3))
        self.wdl_plain = nn.Linear(d, 3)

    def encode(self, sq, side, rel):
        B = sq.shape[0]
        x = self.piece(sq) + self.square.weight[None] + self.side(side)[:, None]
        x = torch.cat([self.cls.expand(B, 1, -1) + self.side(side)[:, None], x], 1)              # (B, 65, d)
        bias = torch.zeros(B, self.cfg.heads, 65, 65, device=sq.device)
        bias[:, :, 1:, 1:] = self.rel_bias(rel).permute(0, 3, 1, 2)
        for blk in self.blocks:
            x = blk(x, bias)
        return self.ln(x[:, 0])

    def tokens(self, h):
        """Soft probabilities and hard binary tokens (straight-through)."""
        p = torch.sigmoid(self.to_tokens(h))
        hard = (p > 0.5).float()
        return p, hard + p - p.detach()

    def forward(self, sq, side, rel):
        h = self.encode(sq, side, rel)
        if self.cfg.plain:
            return dict(h=h, logits=self.wdl_plain(h))
        p, t = self.tokens(h)
        return dict(h=h, p=p, t=t, logits=self.wdl_from_tokens(t))


def losses(model: TokenNet, b: dict, cfg: Config) -> dict:
    o = model(b["sq"], b["side"], b["rel"])
    out = dict(wdl=F.cross_entropy(o["logits"], b["wdl"]))
    if cfg.plain:
        out["total"] = out["wdl"]; return out
    p = o["p"]
    out["sparse"] = p.mean()
    u = p.mean(0).clamp(1e-4, 1 - 1e-4)                                                             # the batch usage per code
    out["use"] = -(u * u.log() + (1 - u) * (1 - u).log()).mean()                                   # entropy: dead or always-on codes cost
    hc = b["has_child"]
    if hc.any():
        oc = model(b["csq"][hc], b["cside"][hc], b["crel"][hc])
        d = (p[hc] - oc["p"]).abs().sum(-1)                                                         # L1 between soft token vectors
        same = (b["wdl"][hc] == b["cwdl"][hc]).float()
        out["coh"] = (same * d).sum() / same.sum().clamp(min=1) + ((1 - same) * F.relu(cfg.coh_margin - d)).sum() / (1 - same).sum().clamp(min=1)
        out["coh_same_d"] = (same * d).sum() / same.sum().clamp(min=1); out["coh_diff_d"] = ((1 - same) * d).sum() / (1 - same).sum().clamp(min=1)
    else:
        out["coh"] = p.sum() * 0
    out["total"] = out["wdl"] + cfg.w_coh * out["coh"] + cfg.w_sparse * out["sparse"] + cfg.w_use * out["use"]
    return out


@torch.no_grad()
def evaluate(model: TokenNet, data: Data, split: str, cfg: Config, device) -> dict:
    model.eval()
    ids = data.ids(split)
    if len(ids) == 0:
        model.train(); return {}
    pred, true, mats, P, H = [], [], [], [], []
    for s in range(0, len(ids), 512):
        b = data.batch(ids[s:s + 512], device); o = model(b["sq"], b["side"], b["rel"])
        pred.append(o["logits"].argmax(-1).cpu().numpy()); true.append(b["wdl"].cpu().numpy()); mats.append(data.material[ids[s:s + 512]])
        if not cfg.plain:
            P.append(o["p"].cpu().numpy())
        H.append(o["h"].cpu().numpy())
    pred, true, mats = np.concatenate(pred), np.concatenate(true), np.concatenate(mats)
    st = dict(n=len(ids), acc=float((pred == true).mean()))
    # the majority per material, fitted on the train split
    tr = data.ids("train"); maj = {}
    for m in np.unique(data.material[tr]):
        maj[m] = int(np.bincount(data.wdl[tr][data.material[tr] == m], minlength=3).argmax())
    st["majority"] = float(np.mean([maj.get(m, 1) == t for m, t in zip(mats, true)]))
    st["by_material"] = {m: dict(n=int((mats == m).sum()), acc=float((pred[mats == m] == true[mats == m]).mean()), majority=float(np.mean([maj.get(m, 1) == t for t in true[mats == m]]))) for m in np.unique(mats)}
    if P:
        P = np.concatenate(P); hard = P > 0.5
        st["tokens_on_mean"] = float(hard.sum(1).mean()); use = hard.mean(0)
        st["codes_alive"] = int(((use > 0.02) & (use < 0.98)).sum()); st["codes_dead"] = int((use <= 0.02).sum()); st["codes_always"] = int((use >= 0.98).sum())
        st["distinct_codes"] = int(len({tuple(r) for r in hard.astype(int).tolist()}))
        c = P - P.mean(0); sv = np.linalg.svd(c[:4000], compute_uv=False); pr = sv ** 2 / (sv ** 2).sum(); st["eff_rank"] = float(np.exp(-(pr * np.log(pr + 1e-12)).sum()))
    model.train()
    return st


def train(a):
    cfg = Config(**{k: v for k, v in vars(a).items() if k in Config.__dataclass_fields__})
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    rows = load_rows()
    held = set(a.held.split(",")) if a.held else set()
    data = Data(rows, held)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TokenNet(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr = data.ids("train"); steps = len(tr) // cfg.batch
    print(f"[tokens] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params; train {len(tr)} val {len(data.ids('val'))} test {len(data.ids('test'))} held {len(data.ids('held'))}; {steps} steps/epoch", flush=True)
    t0 = time.perf_counter(); log = open(out / "steps.jsonl", "w")
    for epoch in range(cfg.epochs):
        rng = np.random.default_rng(cfg.seed + epoch); perm = rng.permutation(tr); acc = {}
        for s in range(steps):
            b = data.batch(perm[s * cfg.batch:(s + 1) * cfg.batch], device)
            L = losses(model, b, cfg)
            opt.zero_grad(); L["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            for k, v in L.items():
                acc[k] = acc.get(k, 0.0) + float(v)
        line = {k: v / steps for k, v in acc.items()}
        ev = evaluate(model, data, "val", cfg, device)
        print(f"[tokens] epoch {epoch} " + " ".join(f"{k} {v:.3f}" for k, v in line.items()) + f" | val acc {ev['acc']:.3f} (majority {ev['majority']:.3f})" +
              (f" tokens on {ev['tokens_on_mean']:.1f}/{cfg.k} alive {ev['codes_alive']} dead {ev['codes_dead']} always {ev['codes_always']} distinct {ev['distinct_codes']} eff_rank {ev['eff_rank']:.1f}" if not cfg.plain else "") + f" | {time.perf_counter() - t0:.0f}s", flush=True)
        log.write(json.dumps(dict(epoch=epoch, **line, **{k: v for k, v in ev.items() if not isinstance(v, dict)})) + "\n"); log.flush()
        torch.save(dict(model=model.state_dict(), config=asdict(cfg), held=sorted(held)), out / "last.pt")
    final = {s: evaluate(model, data, s, cfg, device) for s in ("test", "held")}
    json.dump(final, open(out / "final.json", "w"), indent=1)
    for s, ev in final.items():
        if ev:
            print(f"[tokens] {s}: acc {ev['acc']:.3f} (majority {ev['majority']:.3f}) n {ev['n']}; by material: " + "; ".join(f"{m} {v['acc']:.2f}/{v['majority']:.2f}" for m, v in ev["by_material"].items()), flush=True)


def load_model(path: str):
    ck = torch.load(path, map_location="cpu")
    cfg = Config(**ck["config"]); m = TokenNet(cfg); m.load_state_dict(ck["model"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return m.to(device).eval(), cfg, device, set(ck.get("held", []))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"]); ap.add_argument("--out", default=str(OUT)); ap.add_argument("--held", default="KRPvKP,KQvKRP,KBNvK")
    for k, f in Config.__dataclass_fields__.items():
        if f.type == "bool" or isinstance(f.default, bool):
            ap.add_argument(f"--{k}", action="store_true")
        else:
            ap.add_argument(f"--{k}", type=type(f.default), default=f.default)
    a = ap.parse_args(argv)
    train(a)


if __name__ == "__main__":
    main()
