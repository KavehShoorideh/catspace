"""PLAN 78.20 THE TYPED FACT GRAPH: a position's fact layer (and the legal moves' effects on it) as a heterogeneous graph a relational network can read. GENERIC: relation names
and constants are TOKENS, nothing here knows a game.

  node types  term   one per distinct constant string (token = the string)
              fact   one per fact row (feature: relation token, kind ground/derived from the vocabulary, present = in the position's facts rather than only added by a move)
              move   one per legal move (token "move"); `y` its value and `opt` 1.0 if it is among the best, when the moves carry values
              claim  RESERVED for lemma bindings / conjectures (token "claim", a probability feature); `graph_of(bindings=[{lemma, prob, facts: [indices]}])` builds them.
                     NOTHING SUPPLIES BINDINGS YET: the type exists (empty) so a later learner can feed lemma matches without changing the net's shape.
  edge types  (fact, arg<i>, term)    argument position i of the fact row, i < max_arity (one type per position)
              (move, removes, fact)   the fact rows the move removes;  (move, adds, fact) the rows it adds (a row not in the position is added as a fact node, present = 0)
              (claim, binds, fact)    the facts a claim is about
              plus every reverse, `rev_<name>`, for message passing.
  tokens      `Vocab` (built from the training corpus, saved with the model; an unknown token maps to UNK = id 0). `graph_of(..., vocabs=None)` keeps the strings in
              `data.tokens`; with `vocabs` it also fills the integer features.

`graph_hash(facts, vocabulary)` is the abstraction key of plan 78.19 reading (ii): a Weisfeiler-Lehman hash of the term/fact graph, invariant to the order of the facts.
"""
from __future__ import annotations

import hashlib

import torch
from torch_geometric.data import HeteroData

UNK = "<unk>"
MOVE, CLAIM = "move", "claim"
MAX_ARITY = 6                       # argument positions with their own edge type (a longer row's tail shares the last position's)
PERSPECTIVES = ("us", "them")       # the engine's perspective constants (CE.facts with perspective=True)


class Vocab:
    """token <-> id, UNK = 0. `build` from the training corpus; `to_json` / `from_json` to save with a model."""

    def __init__(self, tokens=()):
        self.tokens = [UNK] + [t for t in tokens if t != UNK]
        self.index = {t: i for i, t in enumerate(self.tokens)}

    @classmethod
    def build(cls, counts, min_count: int = 1, always=()) -> "Vocab":
        keep = [t for t, n in sorted(counts.items()) if n >= min_count]
        return cls(list(always) + [t for t in keep if t not in always])

    def id(self, token: str) -> int:
        return self.index.get(token, 0)

    def __len__(self) -> int:
        return len(self.tokens)

    def to_json(self) -> list:
        return list(self.tokens[1:])

    @classmethod
    def from_json(cls, tokens) -> "Vocab":
        return cls(tokens)


def edge_types(max_arity: int = MAX_ARITY) -> list:
    """Every edge type of the graph, forward then reverse."""
    fwd = [("fact", f"arg{i}", "term") for i in range(max_arity)] + [("move", "removes", "fact"), ("move", "adds", "fact"), ("claim", "binds", "fact")]
    return fwd + [(d, f"rev_{r}", s) for s, r, d in fwd]


def kind_of(rel: str, vocabulary: dict) -> int:
    """0 ground (state), 1 derived (the vocabulary's `derived` relations)."""
    return 1 if rel in set(vocabulary.get("derived", [])) else 0


def row_key(row) -> tuple:
    return tuple(str(x) for x in row)


def graph_of(facts: list, moves: list, vocabulary: dict, bindings: "list | None" = None, *, vocabs: "dict | None" = None, max_arity: int = MAX_ARITY) -> HeteroData:
    """facts: [[relation, args...]] in the mover's perspective; moves: [{name, removed, added, value?}]; vocabulary: {derived, delta, ...}; bindings: [{lemma, prob, facts:
    [indices into `facts`]}] or None; vocabs: {"term": Vocab, "rel": Vocab} to fill integer features. -> HeteroData."""
    fact_rows, fact_ix = [], {}

    def fact_node(row, present: int) -> int:
        k = row_key(row)
        if k not in fact_ix:
            fact_ix[k] = len(fact_rows)
            fact_rows.append((k, present))
        return fact_ix[k]

    for row in facts:
        fact_node(row, 1)
    n_pos = len(facts)
    mv_edges = {"removes": [], "adds": []}
    for mi, m in enumerate(moves):
        for kind, key in (("removes", "removed"), ("adds", "added")):
            for row in m.get(key, []):
                mv_edges[kind].append((mi, fact_node(row, 0)))
    term_ix, arg_edges = {}, [[] for _ in range(max_arity)]
    for fi, (k, _p) in enumerate(fact_rows):
        for pos, c in enumerate(k[1:]):
            ti = term_ix.setdefault(c, len(term_ix))
            arg_edges[min(pos, max_arity - 1)].append((fi, ti))
    terms = sorted(term_ix, key=term_ix.get)
    claim_edges, probs = [], []
    for ci, b in enumerate(bindings or []):
        probs.append(float(b.get("prob", 0.0)))
        for fi in b.get("facts", []):
            claim_edges.append((ci, fi))

    d = HeteroData()
    rels = [k[0] for k, _ in fact_rows]
    d.tokens = {"term": terms, "fact": rels, "move": [MOVE] * len(moves), "claim": [CLAIM] * len(probs)}
    d["term"].x = torch.zeros((len(terms), 1), dtype=torch.long)
    d["fact"].x = torch.tensor([[0, kind_of(r, vocabulary), p] for r, (_k, p) in zip(rels, fact_rows)], dtype=torch.long).reshape(len(rels), 3)
    d["move"].x = torch.zeros((len(moves), 1), dtype=torch.long)
    d["claim"].x = torch.zeros((len(probs), 1), dtype=torch.long)
    d["claim"].prob = torch.tensor(probs, dtype=torch.float).reshape(len(probs), 1)
    vals = [m.get("value") for m in moves]
    has = bool(moves) and all(v is not None for v in vals)
    top = max(vals) if has else None
    d["move"].y = torch.tensor([float(v) if has else 0.0 for v in vals], dtype=torch.float)
    d["move"].opt = torch.tensor([1.0 if has and v == top else 0.0 for v in vals], dtype=torch.float)
    d["move"].has_target = torch.full((len(moves),), has, dtype=torch.bool)
    d["move"].lost = torch.tensor([float(m.get("lost", 0)) for m in moves], dtype=torch.float)
    d["move"].has_lost = torch.full((len(moves),), bool(moves) and all("lost" in m for m in moves), dtype=torch.bool)

    def put(et, pairs):
        ei = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()
        d[et].edge_index = ei
        s, r, t = et
        d[(t, f"rev_{r}", s)].edge_index = ei.flip(0)

    for i in range(max_arity):
        put(("fact", f"arg{i}", "term"), arg_edges[i])
    put(("move", "removes", "fact"), mv_edges["removes"])
    put(("move", "adds", "fact"), mv_edges["adds"])
    put(("claim", "binds", "fact"), claim_edges)
    d.n_present = n_pos
    return encode(d, vocabs) if vocabs else d


def encode(d: HeteroData, vocabs: dict) -> HeteroData:
    """Fill the integer token features of a graph from its strings (`vocabs` = {"term": Vocab, "rel": Vocab}); an unknown token is UNK."""
    t = d.tokens
    d["term"].x = torch.tensor([[vocabs["term"].id(c)] for c in t["term"]], dtype=torch.long).reshape(len(t["term"]), 1)
    d["fact"].x[:, 0] = torch.tensor([vocabs["rel"].id(r) for r in t["fact"]], dtype=torch.long)
    d["move"].x = torch.full((len(t["move"]), 1), vocabs["rel"].id(MOVE), dtype=torch.long)
    d["claim"].x = torch.full((len(t["claim"]), 1), vocabs["rel"].id(CLAIM), dtype=torch.long)
    return d


def tokens_of(d: HeteroData) -> dict:
    """The strings behind a graph (`graph_of` without vocabs)."""
    return d.tokens


# ---------------------------------------------------------------------------------------------------------------- the Weisfeiler-Lehman hash

def _h(*parts) -> str:
    return hashlib.sha1("\x1f".join(map(str, parts)).encode()).hexdigest()[:16]


def graph_hash(facts: list, vocabulary: "dict | None" = None, rounds: int = 3) -> str:
    """The abstraction key: a Weisfeiler-Lehman hash of the term/fact graph (a fact's colour reads its relation, its kind and its arguments' colours by position; a term's
    colour reads its constant and the (relation, position, fact colour) it appears in), `rounds` rounds, then the sorted multiset of colours. A deterministic function of the SET
    of facts: invariant to their order, sensitive to a changed fact."""
    vocabulary = vocabulary or {}
    rows = sorted({row_key(r) for r in facts})
    terms = sorted({c for r in rows for c in r[1:]})
    fcol = {r: _h("f", r[0], kind_of(r[0], vocabulary)) for r in rows}
    tcol = {c: _h("t", c) for c in terms}
    for _ in range(rounds):
        nf = {r: _h("f", r[0], kind_of(r[0], vocabulary), *[tcol[c] for c in r[1:]]) for r in rows}
        occ = {c: [] for c in terms}
        for r in rows:
            for pos, c in enumerate(r[1:]):
                occ[c].append(_h(r[0], pos, fcol[r]))
        nt = {c: _h("t", tcol[c], *sorted(occ[c])) for c in terms}
        fcol, tcol = nf, nt
    return _h(*sorted(fcol.values()), "|", *sorted(tcol.values()))
