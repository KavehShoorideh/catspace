"""PLAN 78.20 / 78.21 THE GRAPH PRIOR: a small relational graph network over the typed fact graph that SCORES THE LEGAL MOVES. TRAINING (torch) is the training kit; INFERENCE is
entirely in Rust (rust/engine/decision: graph.rs builds the graph, graph_net.rs runs the forward pass from the exported weights file, graph_prior.rs is the prior;
`CE.GraphModel` is its binding), so there is ONE graph builder (the engine's) and the rank source `gnn` needs no torch. Generic: tokens, not a game.

  net      GraphNet: token embeddings (term; relation + kind (state / derived / static) + present), `layers` HeteroConv(SAGEConv per edge type) layers with a residual, an MLP on the
           move nodes -> one score a move. Edge types: arg<i>, part<i> (the fused objects' components), removes, adds, binds, and the reverses.
  loss     cross-entropy of the softmax over a position's moves against the uniform distribution over the OPTIMAL set (value == the card's best).
  train    `train(cards_path, out_path, log=, epochs=, seed=, dev_fn=, vocabulary=, specs=)`: each training graph is built by `CE.graph` from the card's POSITION (the card's id
           "<run>:<game>:<ply>" joins the wake `log` to the moves; the card gives each move's value and units lost by name); the graph vocabulary and description come from
           game_store.graph_vocabulary / graph_spec_texts (parameters override); vocabularies are the engine's constant ones (`CE.graph_vocab`). Dev groups train, held groups
           evaluate; prints `OBS gnn epoch ...`, the baselines on the held cards, and `OBS gnn parity` (max abs difference between torch and Rust scores). Writes the torch
           checkpoint AND the Rust weights file (`export`).
  weights  the Rust weights file is JSON (analyst_core's threat Policy convention): flat row-major float lists, the vocabularies, one conv block per layer keyed
           "<src>__<relation>__<dst>"; the Rust side refuses a file whose vocabularies are not the description's.
  source   `GraphSource(weights_path)`: `rank(variant, moves) -> [(move, score)]` best first, entirely in Rust. `source = "gnn"`, `training_only = False`.

  .venv/bin/python -m catspace.incubator.analyst.discovery.graph_prior build-cards chess --log <wake.jsonl> --depth 5 --stride 1 --max-cards 4000 --judge "material (mined)" --out <cards>
  .venv/bin/python -m catspace.incubator.analyst.discovery.graph_prior train --cards <cards> --log <wake.jsonl> --out <model.pt> --epochs 30 --seed 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import cs_engine as CE

from catspace.io import paths

try:                                   # the training kit; the inference kit (GraphSource) runs without it
    import torch
    from torch import nn
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import HeteroConv, SAGEConv
    from torch_geometric.utils import softmax as seg_softmax

    from catspace.incubator.analyst.discovery import fact_graph as FG
except ImportError:                    # pragma: no cover
    torch = None
    nn = type("nn", (), {"Module": object})

DIM, LAYERS = 64, 3


# ---------------------------------------------------------------------------------------------------------------- the net

class GraphNet(nn.Module):
    """Scores every `move` node of a (batched) HeteroData from `fact_graph.from_engine` (or `graph_of`)."""

    def __init__(self, n_term: int, n_rel: int, dim: int = DIM, layers: int = LAYERS, max_arity: int = 6, max_parts: int = 0):
        super().__init__()
        self.term, self.rel = nn.Embedding(n_term, dim), nn.Embedding(n_rel, dim)
        self.kind, self.present = nn.Embedding(3, dim), nn.Embedding(2, dim)
        self.prob = nn.Linear(1, dim)
        ets = FG.edge_types(max_arity, max_parts)
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

def graphs_of_cards(cards, items: dict, spec: str, vocabulary_json: str, dev_fn) -> tuple:
    """-> (dev graphs, held graphs, history, skipped): one engine-built graph per card whose id is in `items` ({id: (variant, start, moves, id, group)}, `corpus_items`) and whose
    legal moves ALL carry a value; the move values / units lost come from the card by move name. `history` {id: (variant, moves, start)} lets a graph be rebuilt or scored in Rust;
    `skipped` counts the cards left out. `dev_fn(group) -> bool`."""
    dev, held, history = [], [], {}
    skipped = {"no_moves": 0, "unknown_id": 0, "partial": 0}
    for i in range(len(cards)):
        c = json.loads(cards.card(i))
        if not c["moves"]:
            skipped["no_moves"] += 1
            continue
        it = items.get(c["id"])
        if it is None:
            skipped["unknown_id"] += 1
            continue
        d = CE.graph(it[0], spec, vocabulary_json, it[1], it[2])
        g = FG.from_engine(d, {m["name"]: m["value"] for m in c["moves"]}, {m["name"]: m.get("lost", 0) for m in c["moves"]})
        if not bool(g["move"].has_target.all()):
            skipped["partial"] += 1
            continue
        del g.tokens
        g.cid = c["id"]
        history[c["id"]] = (it[0], it[2], it[1])
        (dev if dev_fn(c["group"]) else held).append(g)
    return dev, held, history, skipped


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


def train(cards_path: str, out_path: str, *, log: str, epochs: int = 30, seed: int = 1, dev_fn=None, vocabulary: "dict | None" = None, specs: "dict | None" = None,
          game: str = "chess", variant: "str | None" = None, dim: int = DIM, layers: int = LAYERS, lr: float = 2e-3, batch_size: int = 32, weights_path: "str | None" = None,
          parity_n: int = 50, say=print) -> dict:
    """Train on the dev cards of a saved CardSet (built from `log`'s positions), evaluate on the held; save the torch checkpoint to `out_path` and the Rust weights file to
    `weights_path` (default paths.derived("graph_prior_<game>.json")) -> the report."""
    from catspace.incubator.analyst.discovery import game_store as GS
    from catspace.incubator.analyst.discovery import sleep as SL
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    vocabulary = vocabulary if vocabulary is not None else GS.graph_vocabulary(game)
    specs = specs if specs is not None else GS.graph_spec_texts(game)
    variant = variant or game
    vj, spec = json.dumps(vocabulary), specs[variant]
    dev_fn = dev_fn or (lambda g: SL.is_dev(g, SL.Settings(seed=seed)))
    t0 = time.perf_counter()
    voc = CE.graph_vocab(spec, vj)
    vocabs = {"term": FG.Vocab(voc["term"]), "rel": FG.Vocab(voc["rel"])}
    items = {it[3]: it for it in corpus_items(log, variant, 1, 1 << 40)}
    cards = CE.CardSet.load(cards_path)
    dev, held, history, skipped = graphs_of_cards(cards, items, spec, vj, dev_fn)
    say(f"OBS gnn corpus: cards {len(cards)}, dev {len(dev)}, held {len(held)}, skipped {sum(skipped.values())} {skipped}, terms {len(vocabs['term'])}, relations {len(vocabs['rel'])}, "
        f"graph seconds {time.perf_counter() - t0:.1f}")
    base = baselines(held)
    say(f"OBS gnn baselines held ({base['n']} cards, {base['contested']} contested): uniform {base['uniform']:.4f}, fewest lost "
        f"{'-' if base['fewest_lost'] is None else format(base['fewest_lost'], '.4f')}; contested only: uniform {base['contested_uniform']:.4f}, fewest lost "
        f"{'-' if base['contested_fewest_lost'] is None else format(base['contested_fewest_lost'], '.4f')}")
    net = GraphNet(len(vocabs["term"]), len(vocabs["rel"]), dim, layers, voc["max_arity"], voc["max_parts"])
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
    config = {"dim": dim, "layers": layers, "max_arity": voc["max_arity"], "max_parts": voc["max_parts"], "seed": seed, "epochs": epochs}
    torch.save({"state": net.state_dict(), "vocabs": {k: v.to_json() for k, v in vocabs.items()}, "config": config, "vocabulary": vocabulary, "game": game, "variant": variant},
               out_path)
    weights_path = weights_path or default_path(game)
    export(out_path, weights_path)
    parity = parity_of(net, weights_path, {variant: spec}, vj, (held or dev)[:parity_n], history)
    say(f"OBS gnn parity: max abs difference torch vs Rust scores {parity['max_abs_diff']:.2e} over {parity['n']} {'held' if held else 'dev'} positions ({parity['moves']} moves)")
    return {"history": hist, "baselines": base, "seconds": time.perf_counter() - t1, "out": out_path, "weights": weights_path, "parity": parity, "skipped": skipped}


@torch.no_grad() if torch is not None else (lambda f: f)
def parity_of(net: "GraphNet", weights_path: str, specs: dict, vocabulary_json: str, graphs: list, history: dict) -> dict:
    """The largest absolute difference between the torch net's scores on the training graphs and `CE.GraphModel`'s scores of the same positions (built and run in Rust)."""
    net.eval()
    model = CE.GraphModel(weights_path, specs, vocabulary_json)
    worst, moves = 0.0, 0
    for g in graphs:
        variant, hist, *start = history[g.cid]
        names, rust = model.scores(variant, list(hist), *start)
        ts = net(g).tolist()
        assert len(names) == len(ts), (g.cid, len(names), len(ts))
        worst = max([worst] + [abs(a - b) for a, b in zip(ts, rust)])
        moves += len(ts)
    return {"max_abs_diff": worst, "n": len(graphs), "moves": moves, "timing": model.timing()}


# ---------------------------------------------------------------------------------------------------------------- the Rust weights file

def export(checkpoint_path: str, weights_path: str) -> str:
    """Write the Rust weights file (JSON; flat row-major lists; a torch Linear weight is [out, in]) of a torch checkpoint saved by `train`."""
    ck = torch.load(checkpoint_path, weights_only=False)
    st, cfg = ck["state"], ck["config"]

    def flat(t) -> list:
        return [float(f"{v:.9g}") for v in t.detach().float().reshape(-1).tolist()]    # 9 significant digits round-trip a float32

    conv, norm = [], []
    for l in range(cfg["layers"]):
        pre = f"convs.{l}.convs."
        keys = sorted({k[len(pre):].split(".lin_")[0] for k in st if k.startswith(pre)})
        # (this PyG names a block `<src___relation___dst>`; the file's key is `src__relation__dst`)
        conv.append({k.strip("<>").replace("___", "__"): {"lin_l_w": flat(st[f"{pre}{k}.lin_l.weight"]), "lin_l_b": flat(st[f"{pre}{k}.lin_l.bias"]),
                                                          "lin_r_w": flat(st[f"{pre}{k}.lin_r.weight"])} for k in keys})
        norm.append({t: {"w": flat(st[f"norms.{l}.{t}.weight"]), "b": flat(st[f"norms.{l}.{t}.bias"])} for t in ("term", "fact", "move", "claim")})
    out = {"format": "graph_net", "version": 1, "dim": cfg["dim"], "layers": cfg["layers"], "max_arity": cfg["max_arity"], "max_parts": cfg["max_parts"],
           "game": ck.get("game"), "vocab": ck["vocabs"],
           "emb": {"term": flat(st["term.weight"]), "rel": flat(st["rel.weight"]), "kind": flat(st["kind.weight"]), "present": flat(st["present.weight"]),
                   "prob_w": flat(st["prob.weight"]), "prob_b": flat(st["prob.bias"])},
           "conv": conv, "norm": norm,
           "head": {"w0": flat(st["head.0.weight"]), "b0": flat(st["head.0.bias"]), "w1": flat(st["head.2.weight"]), "head_bias": flat(st["head.2.bias"])}}
    with open(weights_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return weights_path


# ---------------------------------------------------------------------------------------------------------------- the rank source

class GraphSource:
    """A RANK SOURCE over the trained graph net: INFERENCE kit, ENTIRELY IN RUST (`CE.GraphModel`: the typed fact graph and the forward pass; no torch, no teacher, no search).
    `rank(variant, moves)` -> [(move, score)] best first. `specs` {variant: description text} and `vocabulary` (the graph vocabulary) default to the game store's for `game`.
    `ms` is the mean milliseconds a ranked position so far; `timing()` the Rust model's means (facts, graph, forward, total)."""
    source = "gnn"
    training_only = False

    def __init__(self, model_path: str, specs: "dict | None" = None, vocabulary: "dict | None" = None, game: str = "chess"):
        from catspace.incubator.analyst.discovery import game_store as GS
        if str(model_path).endswith(".pt"):
            raise ValueError(f"{model_path}: a torch checkpoint; the rank source reads the Rust weights file (`export`, default {default_path(game)})")
        self.game = game
        self.vocabulary = vocabulary if vocabulary is not None else GS.graph_vocabulary(game)
        self.specs = specs if specs is not None else GS.graph_spec_texts(game)
        self.model = CE.GraphModel(model_path, self.specs, json.dumps(self.vocabulary))

    def timing(self) -> dict:
        return self.model.timing()

    @property
    def ms(self) -> float:
        t = self.model.timing()
        return t["ms_total"] if t["positions"] else float("nan")

    def rank(self, variant: str, moves: list, cap: "int | None" = None, start: "str | None" = None) -> list:
        out = self.model.rank(variant, list(moves), start)
        return out[:cap] if cap else out

    def __call__(self, variant: str, moves: list, start: "str | None" = None) -> "str | None":
        r = self.rank(variant, moves, start=start)
        return r[0][0] if r else None

    def close(self) -> None:
        pass


def default_path(game: str = "chess") -> str:
    return paths.derived(f"graph_prior_{game}.json")


# ---------------------------------------------------------------------------------------------------------------- the CLI

def corpus_items(log: str, variant: str, stride: int, max_cards: int) -> list:
    """The decision positions of a wake log at a stride (sleep's corpus), capped at `max_cards` by an even subsample -> [(variant, start, moves, id, group)] (`start`: the game's own start position string, None for the game's initial position)."""
    from catspace.incubator.analyst.discovery import sleep as SL
    decisions, games = SL.read_log(log)
    items = []
    for (run, g, ply), d in sorted(decisions.items()):
        row = games.get((run, g))
        if row is None or ply % stride != 0:
            continue
        items.append((variant, row.get("start"), list(row["moves"][:ply]), f"{run}:{g}:{ply}", f"{run}:{g}"))
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
    t.add_argument("--log", required=True, help="the wake log the cards were built from (the ids join the cards to their positions)")
    t.add_argument("--out", default=None)
    t.add_argument("--weights", default=None, help="the Rust weights file (default paths.derived(graph_prior_<game>.json))")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "build-cards":
        build_cards(a.game, a.log, depth=a.depth, stride=a.stride, max_cards=a.max_cards, judge=a.judge, variant=a.variant, threads=a.threads,
                    out=a.out or paths.derived(f"graph_cards_{a.game}.cards"))
    else:
        r = train(a.cards, a.out or paths.derived(f"graph_prior_{a.game}.pt"), log=a.log, epochs=a.epochs, seed=a.seed, game=a.game, weights_path=a.weights)
        print(f"OBS gnn trained: {r['seconds']:.0f} s, model {r['out']}, rust weights {r['weights']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
