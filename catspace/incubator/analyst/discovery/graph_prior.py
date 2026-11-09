"""PLAN 78.20 THE GRAPH PRIOR: a small relational graph network over the typed fact graph (fact_graph.py) that SCORES THE LEGAL MOVES, trained in the training kit on the engine's
exact cards, used in the inference kit as a rank source (`GraphSource`: it ranks, it claims nothing). Generic: tokens, not a game.

  net      GraphNet: token embeddings (term, relation + kind + present), `layers` HeteroConv(SAGEConv per edge type) layers with a residual, an MLP on the move nodes -> one score a move.
  loss     cross-entropy of the softmax over a position's moves against the uniform distribution over the OPTIMAL set (value == the card's best).
  train    `train(cards_path, out_path, epochs=, seed=, dev_fn=, vocabulary=)`: dev groups train, held groups evaluate; prints `OBS gnn epoch ...` and the baselines on the held cards
           (uniform random: the mean of |optimal| / |moves|; fewest units lost, when the cards carry `lost`).
  source   `GraphSource(model_path)`: `rank(variant, moves) -> [(move, score)]` best first; the position's facts and each move's removed / added rows (the child's facts in the
           PARENT's perspective, restricted to the vocabulary's `delta` relations) come from the engine's fact layer. `source = "gnn"`, `training_only = False`.

  .venv/bin/python -m catspace.incubator.analyst.discovery.graph_prior build-cards chess --log <wake.jsonl> --depth 5 --stride 1 --max-cards 4000 --judge "material (mined)" --out <cards>
  .venv/bin/python -m catspace.incubator.analyst.discovery.graph_prior train --cards <cards> --out <model.pt> --epochs 30 --seed 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import softmax as seg_softmax

import cs_engine as CE

from catspace.incubator.analyst.discovery import fact_graph as FG
from catspace.io import paths

DIM, LAYERS = 64, 3


# ---------------------------------------------------------------------------------------------------------------- the net

class GraphNet(nn.Module):
    """Scores every `move` node of a (batched) HeteroData from `fact_graph.graph_of`."""

    def __init__(self, n_term: int, n_rel: int, dim: int = DIM, layers: int = LAYERS, max_arity: int = FG.MAX_ARITY):
        super().__init__()
        self.term, self.rel = nn.Embedding(n_term, dim), nn.Embedding(n_rel, dim)
        self.kind, self.present = nn.Embedding(2, dim), nn.Embedding(2, dim)
        self.prob = nn.Linear(1, dim)
        ets = FG.edge_types(max_arity)
        self.convs = nn.ModuleList([HeteroConv({et: SAGEConv(dim, dim) for et in ets}, aggr="sum") for _ in range(layers)])
        self.norms = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(dim) for t in ("term", "fact", "move", "claim")}) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, d: HeteroData) -> torch.Tensor:
        f = d["fact"].x
        x = {"term": self.term(d["term"].x[:, 0]), "fact": self.rel(f[:, 0]) + self.kind(f[:, 1]) + self.present(f[:, 2]),
             "move": self.rel(d["move"].x[:, 0]), "claim": self.rel(d["claim"].x[:, 0]) + self.prob(d["claim"].prob)}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(x, d.edge_index_dict)
            x = {t: x[t] + torch.relu(norm[t](out[t])) if t in out else x[t] for t in x}
        return self.head(x["move"]).squeeze(-1)


# ---------------------------------------------------------------------------------------------------------------- the corpus

def graphs_of_cards(cards, vocabulary: dict, dev_fn, max_arity: int = FG.MAX_ARITY) -> tuple:
    """-> (dev graphs, held graphs) with token strings (no ids yet), one per card with at least one move; `dev_fn(group) -> bool`."""
    dev, held = [], []
    for i in range(len(cards)):
        c = json.loads(cards.card(i))
        if not c["moves"]:
            continue
        g = FG.graph_of(c["facts"], c["moves"], vocabulary, max_arity=max_arity)
        g.cid = c["id"]
        (dev if dev_fn(c["group"]) else held).append(g)
    return dev, held


def build_vocabs(graphs: list) -> dict:
    tc, rc = {}, {}
    for g in graphs:
        for t in g.tokens["term"]:
            tc[t] = tc.get(t, 0) + 1
        for r in g.tokens["fact"]:
            rc[r] = rc.get(r, 0) + 1
    return {"term": FG.Vocab.build(tc), "rel": FG.Vocab.build(rc, always=(FG.MOVE, FG.CLAIM))}


def baselines(graphs: list) -> dict:
    """On graphs with targets: the expected top-1 of a uniformly random move (mean |optimal| / |moves|) and of 'fewest units lost' (ties uniform), and the same on the
    CONTESTED cards (some move is not optimal)."""
    uni, lost, cu, cl = [], [], [], []
    for g in graphs:
        m = g["move"]
        if not bool(m.has_target.all()):
            continue
        opt = m.opt > 0.5
        u = opt.float().mean().item()
        uni.append(u)
        if bool(m.has_lost.all()):
            tied = m.lost == m.lost.min()
            lost.append((tied & opt).float().sum().item() / tied.float().sum().item())
        if not bool(opt.all()):
            cu.append(u)
            if bool(m.has_lost.all()):
                cl.append(lost[-1])
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {"uniform": mean(uni), "fewest_lost": mean(lost) if lost else None, "contested_uniform": mean(cu), "contested_fewest_lost": mean(cl) if cl else None,
            "contested": len(cu), "n": len(uni)}


# ---------------------------------------------------------------------------------------------------------------- training

def loss_of(score: torch.Tensor, batch: HeteroData) -> torch.Tensor:
    idx = batch["move"].batch
    n = int(idx.max()) + 1
    logp = torch.log(seg_softmax(score, idx, num_nodes=n).clamp_min(1e-9))
    opt = batch["move"].opt
    tot = torch.zeros(n).index_add_(0, idx, opt)
    target = opt / tot[idx].clamp_min(1.0)
    return -(torch.zeros(n).index_add_(0, idx, target * logp)).mean()


@torch.no_grad()
def top1(net: GraphNet, graphs: list, batch_size: int = 128) -> dict:
    """The argmax move is in the optimal set: overall and on the contested cards."""
    net.eval()
    hit, chit, nc = 0, 0, 0
    for b in DataLoader(graphs, batch_size=batch_size):
        s = net(b)
        ptr = b["move"].ptr
        for k in range(len(ptr) - 1):
            lo, hi = int(ptr[k]), int(ptr[k + 1])
            ok = bool(b["move"].opt[lo + int(s[lo:hi].argmax())] > 0.5)
            hit += ok
            if not bool(b["move"].opt[lo:hi].min() > 0.5):
                nc += 1
                chit += ok
    return {"top1": hit / max(len(graphs), 1), "contested_top1": chit / nc if nc else float("nan")}


def train(cards_path: str, out_path: str, *, epochs: int = 30, seed: int = 1, dev_fn=None, vocabulary: "dict | None" = None, game: str = "chess", dim: int = DIM,
          layers: int = LAYERS, lr: float = 2e-3, batch_size: int = 32, say=print) -> dict:
    """Train on the dev cards of a saved CardSet, evaluate on the held; save {state, vocabs, config, vocabulary, game} to `out_path` (torch.save) -> the report."""
    from catspace.incubator.analyst.discovery import game_store as GS
    from catspace.incubator.analyst.discovery import sleep as SL
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    vocabulary = vocabulary if vocabulary is not None else GS.vocabulary(game)
    dev_fn = dev_fn or (lambda g: SL.is_dev(g, SL.Settings(seed=seed)))
    t0 = time.perf_counter()
    cards = CE.CardSet.load(cards_path)
    dev, held = graphs_of_cards(cards, vocabulary, dev_fn)
    vocabs = build_vocabs(dev)
    for g in dev + held:
        FG.encode(g, vocabs)
        del g.tokens
    say(f"OBS gnn corpus: cards {len(cards)}, dev {len(dev)}, held {len(held)}, terms {len(vocabs['term'])}, relations {len(vocabs['rel'])}, graph seconds {time.perf_counter() - t0:.1f}")
    base = baselines(held)
    say(f"OBS gnn baselines held ({base['n']} cards, {base['contested']} contested): uniform {base['uniform']:.4f}, fewest lost "
        f"{'-' if base['fewest_lost'] is None else format(base['fewest_lost'], '.4f')}; contested only: uniform {base['contested_uniform']:.4f}, fewest lost "
        f"{'-' if base['contested_fewest_lost'] is None else format(base['contested_fewest_lost'], '.4f')}")
    net = GraphNet(len(vocabs["term"]), len(vocabs["rel"]), dim, layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(dev, batch_size=batch_size, shuffle=True, generator=gen)
    t1, hist = time.perf_counter(), []
    for ep in range(1, epochs + 1):
        net.train()
        tot, n = 0.0, 0
        for b in loader:
            opt.zero_grad()
            loss = loss_of(net(b), b)
            loss.backward()
            opt.step()
            tot += loss.item() * b.num_graphs
            n += b.num_graphs
        r = top1(net, held)
        hist.append({"epoch": ep, "loss": tot / max(n, 1), **r})
        say(f"OBS gnn epoch {ep} loss {tot / max(n, 1):.4f} held top1 {r['top1']:.4f} (contested {r['contested_top1']:.4f}) elapsed {time.perf_counter() - t1:.0f}s")
    config = {"dim": dim, "layers": layers, "max_arity": FG.MAX_ARITY, "seed": seed, "epochs": epochs}
    torch.save({"state": net.state_dict(), "vocabs": {k: v.to_json() for k, v in vocabs.items()}, "config": config, "vocabulary": vocabulary, "game": game}, out_path)
    return {"history": hist, "baselines": base, "seconds": time.perf_counter() - t1, "out": out_path}


# ---------------------------------------------------------------------------------------------------------------- the rank source

def swap_perspective(rows: list) -> list:
    """A child's fact rows (the CHILD mover's us / them) in the PARENT's perspective: us <-> them in every argument."""
    sw = {"us": "them", "them": "us"}
    return [[r[0]] + [sw.get(a, a) for a in r[1:]] for r in rows]


def move_effects(variant: str, spec: str, vocab_json: str, vocabulary: dict, moves: list, parent: list, legal: list) -> list:
    """-> [{name, removed, added}] for every legal move: the difference of the child's and the parent's facts over the vocabulary's `delta` relations, in the parent's perspective."""
    delta = set(vocabulary.get("delta", []))
    base = {FG.row_key(r) for r in parent if r[0] in delta}
    out = []
    for m in legal:
        child = {FG.row_key(r) for r in swap_perspective(CE.facts(variant, spec, vocab_json, None, list(moves) + [m], True)) if r[0] in delta}
        out.append({"name": m, "removed": [list(r) for r in sorted(base - child)], "added": [list(r) for r in sorted(child - base)]})
    return out


class GraphSource:
    """A RANK SOURCE over the trained graph net: INFERENCE kit (it ranks from the rules' fact layer alone, no teacher, no search). `rank(variant, moves)` -> [(move, score)] best
    first. `specs` {variant: description text} defaults to the game store's for the model's game. `ms` is the mean milliseconds a ranked position so far."""
    source = "gnn"
    training_only = False

    def __init__(self, model_path: str, specs: "dict | None" = None):
        from catspace.incubator.analyst.discovery import game_store as GS
        ck = torch.load(model_path, weights_only=False)
        self.game, self.vocabulary, self.config = ck["game"], ck["vocabulary"], ck["config"]
        self.vocabs = {k: FG.Vocab.from_json(v) for k, v in ck["vocabs"].items()}
        self.net = GraphNet(len(self.vocabs["term"]), len(self.vocabs["rel"]), self.config["dim"], self.config["layers"], self.config["max_arity"])
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.specs = specs if specs is not None else GS.spec_texts(self.game)
        self.vocab_json = json.dumps(self.vocabulary)
        self.seconds, self.positions = 0.0, 0

    @property
    def ms(self) -> float:
        return 1000.0 * self.seconds / self.positions if self.positions else float("nan")

    def graph(self, variant: str, moves: list) -> tuple:
        """-> (graph, legal moves) of the position after `moves`."""
        spec = self.specs[variant]
        legal = sorted(CE.state(variant, None, list(moves))[2])
        facts = CE.facts(variant, spec, self.vocab_json, None, list(moves), True)
        effects = move_effects(variant, spec, self.vocab_json, self.vocabulary, moves, facts, legal)
        return FG.graph_of(facts, effects, self.vocabulary, vocabs=self.vocabs, max_arity=self.config["max_arity"]), legal

    def rank(self, variant: str, moves: list, cap: "int | None" = None) -> list:
        t0 = time.perf_counter()
        g, legal = self.graph(variant, moves)
        if not legal:
            return []
        with torch.no_grad():
            s = self.net(g).tolist()
        self.seconds += time.perf_counter() - t0
        self.positions += 1
        out = sorted(zip(legal, s), key=lambda x: (-x[1], x[0]))
        return out[:cap] if cap else out

    def __call__(self, variant: str, moves: list) -> "str | None":
        r = self.rank(variant, moves)
        return r[0][0] if r else None

    def close(self) -> None:
        pass


def default_path(game: str = "chess") -> str:
    return paths.derived(f"graph_prior_{game}.pt")


# ---------------------------------------------------------------------------------------------------------------- the CLI

def corpus_items(log: str, variant: str, stride: int, max_cards: int) -> list:
    """The decision positions of a wake log at a stride (sleep's corpus), capped at `max_cards` by an even subsample -> [(variant, None, moves, id, group)]."""
    from catspace.incubator.analyst.discovery import sleep as SL
    decisions, games = SL.read_log(log)
    items = []
    for (run, g, ply), d in sorted(decisions.items()):
        row = games.get((run, g))
        if row is None or ply % stride != 0:
            continue
        items.append((variant, None, list(row["moves"][:ply]), f"{run}:{g}:{ply}", f"{run}:{g}"))
    if len(items) > max_cards:
        items = [items[int(i * len(items) / max_cards)] for i in range(max_cards)]
    return items


def build_cards(game: str, log: str, *, depth: int, stride: int, max_cards: int, judge: "str | None", out: str, variant: "str | None" = None, threads: int = 0, fuse: int = 20_000_000) -> int:
    from catspace.incubator.analyst.discovery import game_store as GS
    j = GS.judge_of(game, judge) or {}
    items = corpus_items(log, variant or game, stride, max_cards)
    t0 = time.perf_counter()
    cards = CE.build_cards(items, GS.spec_texts(game), json.dumps(GS.vocabulary(game)), depth, fuse, threads, j.get("worth", {}), bool(j.get("ours_first", False)))
    cards.save(out)
    print(f"OBS gnn build-cards {game}: {len(cards)} cards from {len(items)} positions, depth {depth}, judge {j.get('name', 'count')}, {time.perf_counter() - t0:.0f} s -> {out}")
    return len(cards)


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-cards")
    b.add_argument("game")
    b.add_argument("--variant", default=None)
    b.add_argument("--log", required=True)
    b.add_argument("--depth", type=int, default=5)
    b.add_argument("--stride", type=int, default=1)
    b.add_argument("--max-cards", type=int, default=4000)
    b.add_argument("--judge", default=None)
    b.add_argument("--threads", type=int, default=0)
    b.add_argument("--out", default=None)
    t = sub.add_parser("train")
    t.add_argument("--game", default="chess")
    t.add_argument("--cards", required=True)
    t.add_argument("--out", default=None)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "build-cards":
        build_cards(a.game, a.log, depth=a.depth, stride=a.stride, max_cards=a.max_cards, judge=a.judge, variant=a.variant, threads=a.threads,
                    out=a.out or paths.derived(f"graph_cards_{a.game}.cards"))
    else:
        r = train(a.cards, a.out or default_path(a.game), epochs=a.epochs, seed=a.seed, game=a.game)
        print(f"OBS gnn trained: {r['seconds']:.0f} s, model {r['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
