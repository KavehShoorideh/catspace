"""THE GRAPH-TOKEN GUIDE (Kaveh 2026-09-23: "a transformer-based network to decide the next piece to add to the
frame; the graph structure makes it easy: pieces with line of sight, one hop, two hops, one move away, two moves
away"). Tokens = the pieces of the position; attention bias = one learned weight per head per RELATION between two
pieces, the relations being the domain's binary rules relations (attacks, defends, aligned ...) plus the move
graph (i can reach j's square in one move; in two), plus "in the frame already" and "is the mover". Labels = the
Möbius share of each piece in the move's outcome (shap_bodies), conditioned on a random prefix of the relevant set
already in the frame, so the head learns "which piece next". Gauges beside the MLP guide and the heuristic.

    python -m catspace.incubator.analyst.body_guide_tf train
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

OUT = Path("data/derived/lemmas")
CKPT = OUT / "body_guide_tf.pt"
MAXP = 32


def move_graph(board: chess.Board):
    """reach1[i][j]: piece i can move to piece j's square in one move (a capture or, for a friend, a defence line);
    reach2: in two moves through an empty square. From the rules only (attack sets)."""
    sqs = list(board.piece_map().keys())
    idx = {s: k for k, s in enumerate(sqs)}
    n = len(sqs)
    r1 = np.zeros((n, n), dtype=np.float32); r2 = np.zeros((n, n), dtype=np.float32)
    for s in sqs:
        att = board.attacks(s)
        for t in att:
            if t in idx:
                r1[idx[s], idx[t]] = 1.0
        # two moves: squares reachable from an empty attacked square
        for mid in att:
            if board.piece_at(mid) is None:
                b2 = board.copy(stack=False); p = b2.remove_piece_at(s); b2.set_piece_at(mid, p)
                for t in b2.attacks(mid):
                    if t in idx and t != s:
                        r2[idx[s], idx[t]] = 1.0
    return sqs, idx, r1, r2


def encode(dom, board: chess.Board, move: chess.Move, in_frame: set):
    """Token features per piece and the relation tensor (P, P, R)."""
    rels_bin = [r for r in dom.relations if not r.on_action and r.arity == 2]
    sqs, idx, r1, r2 = move_graph(board)
    P = len(sqs)
    types = dom.types
    feat = np.zeros((P, len(types) + 4), dtype=np.float32)
    for k, s in enumerate(sqs):
        p = board.piece_at(s)
        t = ("w" if p.color else "b") + p.symbol().upper()
        if t in types:
            feat[k, types.index(t)] = 1.0
        feat[k, len(types)] = 1.0 if s == move.from_square else 0.0
        feat[k, len(types) + 1] = 1.0 if s in in_frame else 0.0
        feat[k, len(types) + 2] = chess.square_file(s) / 7.0
        feat[k, len(types) + 3] = chess.square_rank(s) / 7.0
    R = len(rels_bin) + 4
    rel = np.zeros((P, P, R), dtype=np.float32)
    for i, si in enumerate(sqs):
        for j, sj in enumerate(sqs):
            if i == j:
                continue
            st = dom.state_of(board, [si, sj])
            for k, r in enumerate(rels_bin):
                try:
                    rel[i, j, k] = float(r.fn(st, si, sj))
                except Exception:
                    pass
            rel[i, j, len(rels_bin)] = r1[i, j]; rel[i, j, len(rels_bin) + 1] = r2[i, j]
            rel[i, j, len(rels_bin) + 2] = 1.0 if chess.square_distance(si, sj) == 1 else 0.0
            rel[i, j, len(rels_bin) + 3] = 1.0 if (si == move.to_square or sj == move.to_square) else 0.0   # touches the landing square
    return sqs, feat, rel


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h; self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d); self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, bias, mask):
        B, T, d = x.shape; h = self.h
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, h, d // h).unbind(2)
        att = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(d // h) + bias
        att = att.masked_fill(~mask[:, None, None, :], -1e9).softmax(-1)
        x = x + self.o(torch.einsum("bhts,bshd->bthd", att, v).reshape(B, T, d))
        return x + self.ff(self.n2(x))


class GraphGuide(nn.Module):
    def __init__(self, d_feat: int, n_rel: int, d: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.emb = nn.Linear(d_feat, d); self.rel_bias = nn.Linear(n_rel, heads, bias=False)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)]); self.head = nn.Linear(d, 1)

    def forward(self, feat, rel, mask):
        x = self.emb(feat)
        bias = self.rel_bias(rel).permute(0, 3, 1, 2)                          # B, heads, P, P
        for blk in self.blocks:
            x = blk(x, bias, mask)
        return self.head(x).squeeze(-1)


def build(dom, records, rng, prefix: bool = True, log=print):
    X = []; RL = []; Y = []; M = []; G = []
    t0 = time.perf_counter()
    for gi, rec in enumerate(records):
        b = chess.Board(rec["fen"]); mv = chess.Move.from_uci(rec["move"])
        share = {chess.parse_square(k): v for k, v in rec["share"].items()}
        relevant = sorted(share, key=lambda s: -share[s])
        # a random prefix of the relevant set is "already in the frame"; the label is the share of the rest
        k = rng.randrange(0, max(1, len(relevant))) if prefix else 0
        in_frame = set(relevant[:k]) | {mv.from_square}
        sqs, feat, rel = encode(dom, b, mv, in_frame)
        P = len(sqs)
        y = np.array([0.0 if (s in in_frame or b.piece_at(s).piece_type == chess.KING) else share.get(s, 0.0) for s in sqs], dtype=np.float32)
        pad = MAXP - P
        X.append(np.pad(feat, ((0, pad), (0, 0)))); RL.append(np.pad(rel, ((0, pad), (0, pad), (0, 0)))); Y.append(np.pad(y, (0, pad)))
        m = np.zeros(MAXP, dtype=bool); m[:P] = True; M.append(m); G.append(gi)
        if gi % 500 == 0 and gi:
            log(f"[guide-tf] encoded {gi}/{len(records)} ({time.perf_counter() - t0:.0f}s)")
    return torch.tensor(np.stack(X)), torch.tensor(np.stack(RL)), torch.tensor(np.stack(Y)), torch.tensor(np.stack(M)), np.array(G)


def auc(scores, Y, M, thresh=0.05):
    from sklearn.metrics import roc_auc_score
    out = []
    for s, y, m in zip(scores, Y, M):
        s = s[m].numpy(); y = y[m].numpy()
        lab = (y >= thresh).astype(int)
        cand = y >= 0  # exclude nothing; the in-frame/king tokens have y=0 and are negatives
        if lab.min() == lab.max():
            continue
        out.append(roc_auc_score(lab, s))
    return float(np.mean(out)) if out else float("nan"), len(out)


def train(labels_path: Path, epochs: int = 40, seed: int = 0, log=print):
    from .chess_domain import domain
    dom = domain()
    data = json.load(open(labels_path))
    recs = [r for r in data["records"] if r.get("share")]
    rng = random.Random(seed); rng.shuffle(recs)
    fens = sorted({r["fen"] for r in recs}); rng.shuffle(fens); test_f = set(fens[: len(fens) // 5])
    tr = [r for r in recs if r["fen"] not in test_f]; te = [r for r in recs if r["fen"] in test_f]
    log(f"[guide-tf] records train {len(tr)} test {len(te)} (split by position)")
    Xtr, Rtr, Ytr, Mtr, _ = build(dom, tr, rng, log=log); Xte, Rte, Yte, Mte, _ = build(dom, te, rng, log=log)
    torch.manual_seed(seed)
    model = GraphGuide(Xtr.shape[-1], Rtr.shape[-1]); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # the heuristic baseline on the same test tokens: rarity-weighted relation count to the mover (the MLP's own feature)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), 64):
            idx = perm[i: i + 64]
            s = model(Xtr[idx], Rtr[idx], Mtr[idx])
            loss = ((s - Ytr[idx]) ** 2 * Mtr[idx]).sum() / Mtr[idx].sum()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == epochs - 1:
            with torch.no_grad():
                se = model(Xte, Rte, Mte)
            a, n = auc(se, Yte, Mte)
            a_r, _ = auc(torch.rand_like(se), Yte, Mte)
            log(f"[guide-tf] epoch {ep} train mse {tot / len(Xtr):.4f} | test AUC next-body ranking: graph guide {a:.3f}  random {a_r:.3f}  (n {n})")
    torch.save(dict(state=model.state_dict(), d_feat=Xtr.shape[-1], n_rel=Rtr.shape[-1]), CKPT)
    log(f"[guide-tf] -> {CKPT}")


class GraphRanker:
    """Ranks candidate bodies for body_search, conditioned on the bodies already in the frame."""
    def __init__(self, dom, ckpt: Path = CKPT):
        d = torch.load(ckpt, weights_only=False)
        self.model = GraphGuide(d["d_feat"], d["n_rel"]); self.model.load_state_dict(d["state"]); self.model.eval(); self.dom = dom

    def rank(self, board, move, candidates, in_frame=None):
        sqs, feat, rel = encode(self.dom, board, move, set(in_frame or [move.from_square]))
        P = len(sqs); pad = MAXP - P
        X = torch.tensor(np.pad(feat, ((0, pad), (0, 0))))[None]; R = torch.tensor(np.pad(rel, ((0, pad), (0, pad), (0, 0))))[None]
        M = torch.zeros(1, MAXP, dtype=torch.bool); M[0, :P] = True
        with torch.no_grad():
            s = self.model(X, R, M)[0, :P].numpy()
        sc = {sq: float(s[k]) for k, sq in enumerate(sqs)}
        return sorted(((sc.get(c, -1e9), c) for c in candidates), key=lambda x: (-x[0], x[1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"]); ap.add_argument("--labels", default=str(OUT / "shap_bodies_relative_labels.json")); ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args(argv)
    train(Path(a.labels), a.epochs)


if __name__ == "__main__":
    main()
