"""THE LEARNED PRIORS (Kaveh 2026-10-12: "leave room to make it modular ... a graph network or a transformer or both").
Both score a candidate (a term over the base, a schema) for a hole; both are trained on the EXPERIENCE STORE -- every
verdict of the verifier, successes and failures -- so a failure lowers the terms that look like it. They only order the
search: meaning comes from the base, correctness from the verifier.

  GRAPH NETWORK  PyTorch Geometric GINE layers over the term's syntax tree (node = a base symbol, edge = an argument slot)
  TRANSFORMER    an encoder over the term's symbols in prefix order (formal symbols, not language)
  CONTEXT ONLY   the same head on the hole-aware features alone (the ablation)
Each is (structure encoder) + (context: the schema, the material, the hole's components, the term's values and
controllability on the hole's boards) -> a score in [0, 1].
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from . import base_vocabulary as V

NAMED = ["fn:wall", "fn:opposition", "fn:chase", "fn:kq_offedge", "fn:kq_approach", "fn:kq_region"]
SCHEMAS = ["append", "sum", "replace", "prepend"]


def context_features(hole: dict, term: str, schema: str, boards: list, cache: dict) -> np.ndarray:
    """Hole-aware, cheap: the schema, the material, which components the hole has; the term's values on the boards
    (spread, its correlation with each component) and who controls it (its change under our moves and their replies)."""
    import analyst_core as A
    key = (hole["name"], term)
    if key not in cache:
        vals = np.array(A.vocab_values(boards, term), dtype=np.float32)
        comps = np.array(A.component_values(boards, hole["base"]), dtype=np.float32) if hole["base"] else np.zeros((len(boards), 0), np.float32)
        dyn = np.array(A.term_dynamics(boards, term), dtype=np.float32)
        corr = []
        for j in range(2):
            if j < comps.shape[1] and vals.std() > 0 and comps[:, j].std() > 0:
                corr.append(float(np.corrcoef(vals, comps[:, j])[0, 1]))
            else:
                corr.append(0.0)
        ours, theirs = dyn[:, 0].mean(), dyn[:, 1].mean()
        cache[key] = np.array([vals.mean() / 64, vals.std() / 16, vals.min() / 64, vals.max() / 64, (vals == 0).mean(),
                               *corr, ours / 8, theirs / 8, (theirs + 1e-3) / (ours + theirs + 2e-3),
                               *contrast(hole, term, cache)], dtype=np.float32)
    sch = [1.0 if schema == s else 0.0 for s in SCHEMAS]
    mat = [1.0 if hole["material"] == m else 0.0 for m in ("KR", "KQ")]
    named = [1.0 if n in hole["base"] else 0.0 for n in NAMED]
    return np.concatenate([cache[key], np.array(sch + mat + named + [len(hole["base"]) / 3], dtype=np.float32)])


CTX = 12 + len(SCHEMAS) + 2 + len(NAMED) + 1


def contrast(hole: dict, term: str, cache: dict) -> tuple:
    """WHAT THE GAMES SAY ABOUT A TERM (Kaveh 2026-10-12: "if our proposer is not proposing the things it's supposed to
    propose, then the proposer needs to change"): the hole carries the round's positions shortly before a win and in the
    draws' tails (surprise_loop.mine); a measure is minimised, so a term LOWER before wins than in the draws points the
    way -> (the mean difference, scaled; the rank separation 2*P(before a win < in a draw) - 1, ties half). (0, 0)
    without the games (the labeled holes)."""
    c = hole.get("contrast")
    if not c or not c.get("win") or not c.get("draw"):
        return (0.0, 0.0)
    key = ("contrast", hole["name"], term)
    if key not in cache:
        import analyst_core as A
        w = np.array(A.vocab_values(c["win"], term), dtype=np.float32)
        d = np.array(A.vocab_values(c["draw"], term), dtype=np.float32)
        allv = np.concatenate([w, d])
        sep = float((d.mean() - w.mean()) / (allv.std() + 1.0))
        less = (w[:, None] < d[None, :]).mean() + 0.5 * (w[:, None] == d[None, :]).mean()
        cache[key] = (sep, float(2 * less - 1))
    return cache[key]


class Head(nn.Module):
    def __init__(self, d_struct: int, d: int = 64):
        super().__init__()
        self.ctx = nn.Sequential(nn.Linear(CTX, d), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(d + d_struct, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, s, c):
        return torch.sigmoid(self.out(torch.cat([s, self.ctx(c)], dim=-1))).squeeze(-1)


class TermGNN(nn.Module):
    """GINE over the syntax tree (both directions), pooled by sum and mean."""
    def __init__(self, d: int = 64, layers: int = 3):
        super().__init__()
        from torch_geometric.nn import GINEConv
        self.emb = nn.Embedding(len(V.SYMBOLS), d)
        self.edge = nn.Embedding(6, d)                       # argument slots 0-2, and their reverses 3-5
        self.convs = nn.ModuleList([GINEConv(nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)), edge_dim=d) for _ in range(layers)])
        self.head = Head(2 * d, d)

    def forward(self, batch, ctx):
        from torch_geometric.nn import global_add_pool, global_mean_pool
        x = self.emb(batch.x)
        e = self.edge(batch.edge_attr)
        for conv in self.convs:
            x = torch.relu(conv(x, batch.edge_index, e)) + x
        s = torch.cat([global_add_pool(x, batch.batch), global_mean_pool(x, batch.batch)], dim=-1)
        return self.head(s, ctx)


class TermTransformer(nn.Module):
    """An encoder over the term's symbols in prefix order."""
    def __init__(self, d: int = 64, layers: int = 2, max_len: int = 24):
        super().__init__()
        self.emb = nn.Embedding(len(V.SYMBOLS) + 1, d, padding_idx=len(V.SYMBOLS))
        self.pos = nn.Embedding(max_len, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, batch_first=True, dropout=0.0), layers)
        self.head = Head(d, d)
        self.max_len = max_len

    def forward(self, toks, ctx):
        pad = toks == len(V.SYMBOLS)
        h = self.emb(toks) + self.pos(torch.arange(toks.shape[1], device=toks.device).unsqueeze(0))
        h = self.enc(h, src_key_padding_mask=pad)
        h = (h * (~pad).unsqueeze(-1)).sum(1) / (~pad).sum(1, keepdim=True).clamp(min=1)
        return self.head(h, ctx)


class ContextOnly(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.head = Head(0, d)

    def forward(self, _x, ctx):
        return self.head(torch.zeros(ctx.shape[0], 0), ctx)


def graph_batch(terms):
    from torch_geometric.data import Batch, Data
    ds = []
    for t in terms:
        nodes, edges = V.graph(V.parse(t))
        src = [a for a, b, _ in edges] + [b for a, b, _ in edges]
        dst = [b for a, b, _ in edges] + [a for a, b, _ in edges]
        attr = [s for _, _, s in edges] + [3 + s for _, _, s in edges]
        ds.append(Data(x=torch.tensor(nodes), edge_index=torch.tensor([src, dst], dtype=torch.long).reshape(2, -1),
                       edge_attr=torch.tensor(attr, dtype=torch.long)))
    return Batch.from_data_list(ds)


def token_batch(terms, max_len: int = 24):
    pad = len(V.SYMBOLS)
    rows = [V.tokens(V.parse(t))[:max_len] for t in terms]
    return torch.tensor([r + [pad] * (max_len - len(r)) for r in rows], dtype=torch.long)


def make(kind: str):
    return {"gnn": TermGNN, "transformer": TermTransformer, "context": ContextOnly}[kind]()


def inputs(kind: str, terms):
    return graph_batch(terms) if kind == "gnn" else token_batch(terms) if kind == "transformer" else None


def train(kind: str, rows: list, ctx: np.ndarray, epochs: int = 40, seed: int = 0, lr: float = 2e-3, model=None):
    """Regress the verifier's score (both stages) on the experience rows; a listwise term per hole sharpens the ranking.
    `model`: continue training it (INCREMENTAL -- the new verdicts only, a few epochs) instead of a fresh one."""
    torch.manual_seed(seed)
    m = model if model is not None else make(kind)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    y = torch.tensor([r["score"] for r in rows], dtype=torch.float32)
    c = torch.tensor(ctx)
    terms = [r["term"] for r in rows]
    holes = np.array([r["hole"] for r in rows])
    idx = np.arange(len(rows))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(idx)
        for b in range(0, len(idx), 256):
            j = idx[b:b + 256]
            p = m(inputs(kind, [terms[i] for i in j]), c[j])
            loss = ((p - y[j]) ** 2).mean()
            # listwise: within each hole of the batch, the softmax of the predictions should follow the scores
            for h in set(holes[j]):
                k = torch.tensor(holes[j] == h)
                if k.sum() > 1:
                    t = torch.softmax(y[j][k] * 20, 0)
                    loss = loss - 0.1 * (t * torch.log_softmax(p[k] * 20, 0)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m


@torch.no_grad()
def score(m, kind: str, terms: list, ctx: np.ndarray) -> np.ndarray:
    out = []
    for b in range(0, len(terms), 512):
        out.append(m(inputs(kind, terms[b:b + 512]), torch.tensor(ctx[b:b + 512])).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def save(m, path) -> None:
    """Atomically (a reader never sees half a checkpoint)."""
    import os
    tmp = f"{path}.tmp"
    torch.save(m.state_dict(), tmp)
    os.replace(tmp, path)


def load(kind: str, path):
    m = make(kind)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m
