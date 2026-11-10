"""Plan 78.20: the typed fact graph, its WL hash, the graph net (a toy overfit), the GraphSource's move effects against the engine's cards, and source_of("gnn:..."). tmp_path only."""
import json

import cs_engine as CE
import pytest
import torch

from catspace.incubator.analyst.discovery import fact_graph as FG
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import graph_prior as GP
from catspace.incubator.analyst.discovery import sleep as SL

VOC = {"derived": ["att"], "delta": ["cell"]}
FACTS = [["cell", "a", "1", "us", "rook"], ["cell", "b", "1", "them", "pawn"], ["att", "us", "a", "1", "b", "1"]]
MOVES = [{"name": "a1b1", "removed": [["cell", "a", "1", "us", "rook"], ["cell", "b", "1", "them", "pawn"]], "added": [["cell", "b", "1", "us", "rook"]], "value": 2},
         {"name": "a1a2", "removed": [["cell", "a", "1", "us", "rook"]], "added": [["cell", "a", "2", "us", "rook"]], "value": 0}]


def test_graph_counts():
    g = FG.graph_of(FACTS, MOVES, VOC)
    terms = {"a", "1", "us", "rook", "b", "them", "pawn", "2"}
    assert g["term"].num_nodes == len(terms) == 8      # "2" enters with the added row
    assert g["fact"].num_nodes == 3 + 2          # three facts + two added rows (b1 us rook, a2 us rook)
    assert g["move"].num_nodes == 2 and g["claim"].num_nodes == 0
    assert g["fact"].x[:, 1].tolist() == [0, 0, 1, 0, 0] and g["fact"].x[:, 2].tolist() == [1, 1, 1, 0, 0]
    assert g["move"].opt.tolist() == [1.0, 0.0]
    assert g["fact", "arg0", "term"].edge_index.shape[1] == 5 and g["fact", "arg4", "term"].edge_index.shape[1] == 1     # att has 5 arguments
    assert g["move", "removes", "fact"].edge_index.shape[1] == 3 and g["move", "adds", "fact"].edge_index.shape[1] == 2
    assert g["fact", "rev_removes", "move"].edge_index.shape[1] == 3
    assert g["fact", "arg0", "term"].edge_index.shape[1] == g["term", "rev_arg0", "fact"].edge_index.shape[1]


def test_claims_accepted():
    g = FG.graph_of(FACTS, MOVES, VOC, bindings=[{"lemma": "L", "prob": 0.7, "facts": [0, 2]}])
    assert g["claim"].num_nodes == 1 and g["claim"].prob.item() == pytest.approx(0.7)
    assert g["claim", "binds", "fact"].edge_index.shape[1] == 2


def test_unknown_token_is_unk():
    vocabs = {"term": FG.Vocab(["a", "1"]), "rel": FG.Vocab.build({"cell": 1}, always=("move", "claim"))}
    g = FG.graph_of(FACTS, MOVES, VOC, vocabs=vocabs)
    assert g["term"].x[:, 0].tolist().count(0) >= 1           # "rook", "us", ... are unknown
    assert g["fact"].x[0, 0].item() == vocabs["rel"].id("cell") != 0 and g["fact"].x[2, 0].item() == 0     # "att" is not in the vocabulary


def test_hash_permutation_invariant_and_sensitive():
    h = FG.graph_hash(FACTS, VOC)
    assert h == FG.graph_hash(list(reversed(FACTS)), VOC) == FG.graph_hash(FACTS + [FACTS[0]], VOC)
    changed = [FACTS[0], ["cell", "b", "1", "us", "pawn"], FACTS[2]]
    assert FG.graph_hash(changed, VOC) != h
    assert FG.graph_hash(FACTS[:2], VOC) != h


# PLAN 78.21 changed these three tests' interface (the graphs are built by the engine from the card's POSITION, the wake log joins the ids, the rank source reads the Rust
# weights file): toy_cards writes the log the train() call needs; the overfit test reads the checkpoint with the engine's vocabularies; the move-effect test compares the
# engine-built graph's removes / adds edges with the cards' (the old `move_effects` is gone: one graph builder).
GV = {"state": ["cell", "control", "castling_right", "ep_square", "halfmove"], "derived": ["attack"], "static": ["step"], "delta": ["cell"],
      "objects": {"square": ["file", "rank"]}, "scales": ["clock"], "units": None}
GSPECS = {"chess": GS.spec_texts("chess")["chess"]}
TOY = [("chess", None, ["e2e4", "e7e5", "g1f3"], "t:0:3", "t:0"), ("chess", None, ["e2e4", "d7d5", "e4d5"], "t:1:3", "t:1")]


def toy_cards(tmp_path):
    spec, voc = GS.spec_texts("chess"), json.dumps(GS.vocabulary("chess"))
    cards = CE.build_cards(TOY, spec, voc, 2, 20_000_000, 0, {"q": 14, "r": 8, "b": 5, "n": 4, "p": 2, "k": 0}, False)
    p = str(tmp_path / "toy.cards")
    cards.save(p)
    rows = [{"kind": "game", "run": "t", "game": 0, "moves": TOY[0][2] + ["b8c6"]}, {"kind": "game", "run": "t", "game": 1, "moves": TOY[1][2] + ["d8d5"]}]
    rows += [{"kind": "position", "label": f"t:{g}:3:0", "move": "x", "steps": [], "error": None} for g in (0, 1)]
    log = tmp_path / "toy.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows))
    return p, str(log)


def test_toy_training_overfits(tmp_path):
    p, log = toy_cards(tmp_path)
    out = str(tmp_path / "toy.pt")
    r = GP.train(p, out, log=log, epochs=60, seed=1, dev_fn=lambda g: True, game="chess", vocabulary=GV, specs=GSPECS, dim=32, layers=2, lr=5e-3,
                 weights_path=str(tmp_path / "toy.json"), say=lambda *_: None)
    assert r["history"][-1]["loss"] < r["history"][0]["loss"] * 0.7
    # held = dev for the toy (the held set is the dev set: the function above puts everything in dev, evaluate on it)
    cards = CE.CardSet.load(p)
    items = {it[3]: it for it in GP.corpus_items(log, "chess", 1, 100)}
    dev, _, _, _ = GP.graphs_of_cards(cards, items, GSPECS["chess"], json.dumps(GV), lambda g: True)
    ck = torch.load(out, weights_only=False)
    net = GP.GraphNet(len(ck["vocabs"]["term"]) + 1, len(ck["vocabs"]["rel"]) + 1, 32, 2, ck["config"]["max_arity"], ck["config"]["max_parts"])
    net.load_state_dict(ck["state"])
    assert GP.top1(net, dev)["top1"] == 1.0


POSITIONS = [["e2e4", "e7e5", "g1f3"], ["e2e4", "d7d5"], ["e2e4", "d7d5", "e4d5", "d8d5"],               # a black-to-move, a capture
             ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"],                                         # white castled available next
             ["e2e4", "a7a6", "e4e5", "d7d5"]]                                                                  # en passant is legal for white


def test_move_effects_match_cards(tmp_path):
    spec, voc = GS.spec_texts("chess"), json.dumps(GS.vocabulary("chess"))
    items = [("chess", None, mv, f"p{i}", f"g{i}") for i, mv in enumerate(POSITIONS)]
    cards = CE.build_cards(items, spec, voc, 1, 20_000_000, 0, {"q": 14, "r": 8, "b": 5, "n": 4, "p": 2, "k": 0}, False)
    gj, spec = json.dumps(GV), GSPECS["chess"]
    checked = 0
    for i, mv in enumerate(POSITIONS):
        card = json.loads(cards.card(i))
        d = CE.graph("chess", spec, gj, None, mv)

        def row(fi):                                   # a fact node back to the card's row (the squares unfused)
            args = {}
            for name, (src, dst) in d["edges"].items():
                if name.startswith("arg"):
                    args.update({int(name[3:]): d["terms"][t] for s, t in zip(src, dst) if s == fi})
            out = [d["facts"][fi][0]]
            for k in sorted(args):
                sort, _, c = args[k].partition(":")
                out += [c[0], c[1:]] if sort == "square" else [c]
            return tuple(out)

        truth = {m["name"]: m for m in card["moves"]}
        assert set(truth) == set(d["moves"])
        for mi, name in enumerate(d["moves"]):
            t = truth[name]
            assert sorted(row(f) for s, f in zip(*d["edges"]["removes"]) if s == mi) == sorted(map(tuple, t["removed"])), (mv, name)
            assert sorted(row(f) for s, f in zip(*d["edges"]["adds"]) if s == mi) == sorted(map(tuple, t["added"])), (mv, name)
            checked += 1
    assert checked > 100


def test_source_of_gnn_and_rank(tmp_path, monkeypatch):
    p, log = toy_cards(tmp_path)
    out, w = str(tmp_path / "toy.pt"), str(tmp_path / "toy.json")
    GP.train(p, out, log=log, epochs=1, seed=1, dev_fn=lambda g: True, game="chess", vocabulary=GV, specs=GSPECS, dim=16, layers=2, weights_path=w, say=lambda *_: None)
    monkeypatch.setattr(GS, "graph_spec_texts", lambda game: GSPECS)
    monkeypatch.setattr(GS, "graph_vocabulary", lambda game: GV)
    src = SL.source_of(f"gnn:{w}")
    assert src.source == "gnn" and src.training_only is False
    r = src.rank("chess", ["e2e4", "e7e5"])
    legal = CE.state("chess", None, ["e2e4", "e7e5"])[2]
    assert sorted(m for m, _ in r) == sorted(legal)
    assert all(r[i][1] >= r[i + 1][1] for i in range(len(r) - 1))
    assert src.ms > 0
    src.close()
