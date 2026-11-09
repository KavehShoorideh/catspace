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


def toy_cards(tmp_path):
    spec, voc = GS.spec_texts("chess"), json.dumps(GS.vocabulary("chess"))
    items = [("chess", None, ["e2e4", "e7e5", "g1f3"], "a", "ga"), ("chess", None, ["e2e4", "d7d5", "e4d5"], "b", "gb")]
    cards = CE.build_cards(items, spec, voc, 2, 20_000_000, 0, {"q": 14, "r": 8, "b": 5, "n": 4, "p": 2, "k": 0}, False)
    p = str(tmp_path / "toy.cards")
    cards.save(p)
    return p


def test_toy_training_overfits(tmp_path):
    p = toy_cards(tmp_path)
    out = str(tmp_path / "toy.pt")
    r = GP.train(p, out, epochs=60, seed=1, dev_fn=lambda g: True, game="chess", dim=32, layers=2, lr=5e-3, say=lambda *_: None)
    assert r["history"][-1]["loss"] < r["history"][0]["loss"] * 0.7
    # held = dev for the toy (the held set is the dev set: the function above puts everything in dev, evaluate on it)
    cards = CE.CardSet.load(p)
    dev, _ = GP.graphs_of_cards(cards, GS.vocabulary("chess"), lambda g: True)
    ck = torch.load(out, weights_only=False)
    vocabs = {k: FG.Vocab.from_json(v) for k, v in ck["vocabs"].items()}
    for g in dev:
        FG.encode(g, vocabs)
        del g.tokens
    net = GP.GraphNet(len(vocabs["term"]), len(vocabs["rel"]), 32, 2)
    net.load_state_dict(ck["state"])
    assert GP.top1(net, dev)["top1"] == 1.0


def effects_of(moves):
    game = "chess"
    spec, voc = GS.spec_texts(game), GS.vocabulary(game)
    vj = json.dumps(voc)
    parent = CE.facts("chess", spec["chess"], vj, None, moves, True)
    legal = sorted(CE.state("chess", None, moves)[2])
    return GP.move_effects("chess", spec["chess"], vj, voc, moves, parent, legal), spec, vj


POSITIONS = [["e2e4", "e7e5", "g1f3"], ["e2e4", "d7d5"], ["e2e4", "d7d5", "e4d5", "d8d5"],               # a black-to-move, a capture
             ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"],                                         # white castled available next
             ["e2e4", "a7a6", "e4e5", "d7d5"]]                                                                  # en passant is legal for white


def test_move_effects_match_cards(tmp_path):
    spec, voc = GS.spec_texts("chess"), json.dumps(GS.vocabulary("chess"))
    items = [("chess", None, mv, f"p{i}", f"g{i}") for i, mv in enumerate(POSITIONS)]
    cards = CE.build_cards(items, spec, voc, 1, 20_000_000, 0, {"q": 14, "r": 8, "b": 5, "n": 4, "p": 2, "k": 0}, False)
    checked = 0
    for i, mv in enumerate(POSITIONS):
        card = json.loads(cards.card(i))
        eff, _, _ = effects_of(mv)
        truth = {m["name"]: m for m in card["moves"]}
        assert set(truth) == {e["name"] for e in eff}
        for e in eff:
            t = truth[e["name"]]
            assert sorted(map(tuple, e["removed"])) == sorted(map(tuple, t["removed"])), (mv, e["name"])
            assert sorted(map(tuple, e["added"])) == sorted(map(tuple, t["added"])), (mv, e["name"])
            checked += 1
    assert checked > 100


def test_source_of_gnn_and_rank(tmp_path):
    p = toy_cards(tmp_path)
    out = str(tmp_path / "toy.pt")
    GP.train(p, out, epochs=1, seed=1, dev_fn=lambda g: True, game="chess", dim=16, layers=2, say=lambda *_: None)
    src = SL.source_of(f"gnn:{out}")
    assert src.source == "gnn" and src.training_only is False
    r = src.rank("chess", ["e2e4", "e7e5"])
    legal = CE.state("chess", None, ["e2e4", "e7e5"])[2]
    assert sorted(m for m, _ in r) == sorted(legal)
    assert all(r[i][1] >= r[i + 1][1] for i in range(len(r) - 1))
    assert src.ms > 0
    src.close()
