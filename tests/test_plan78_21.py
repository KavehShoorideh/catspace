"""Plan 78.21: the typed fact graph built in Rust (`CE.graph`), `from_engine`, the torch <-> Rust parity of the graph net, the all-Rust GraphSource, and a tiny training run
end to end (export + Rust load). A test-local graph vocabulary over rules/chess.gdl, so nothing here depends on the chess graph vocabulary file. tmp_path only."""
import json
import random

import cs_engine as CE
import pytest
import torch

from catspace.incubator.analyst.discovery import fact_graph as FG
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import graph_prior as GP
from catspace.incubator.analyst.discovery import sleep as SL

V = {"state": ["cell", "control", "castling_right", "ep_square", "halfmove"], "derived": ["attack"], "static": ["step"], "delta": ["cell"],
     "objects": {"square": ["file", "rank"]}, "scales": ["clock"], "units": None}
VJ = json.dumps(V)
SPEC = GS.spec_texts("chess")["chess"]
SPECS = {"chess": SPEC}
P1, P2 = ["e2e4", "e7e5", "g1f3"], ["e2e4", "d7d5", "e4d5", "d8d5"]


def graph(moves):
    return CE.graph("chess", SPEC, VJ, None, moves)


def fact_row(d, fi):
    """The row [relation, args...] of fact node `fi`, the squares unfused."""
    args = {}
    for name, (src, dst) in d["edges"].items():
        if name.startswith("arg"):
            for s, t in zip(src, dst):
                if s == fi:
                    args[int(name[3:])] = d["terms"][t]
    out = [d["facts"][fi][0]]
    for i in sorted(args):
        sort, _, c = args[i].partition(":")
        out += [c[0], c[1:]] if sort == "square" else [c]
    return out


def test_vocab_is_constant_and_typed():
    voc = CE.graph_vocab(SPEC, VJ)
    assert voc["rel"][:2] == ["move", "claim"] and set(voc["rel"][2:]) == {"cell", "control", "castling_right", "ep_square", "halfmove", "attack", "step"}
    assert "square:e4" in voc["term"] and "rank:4" in voc["term"] and "clock:4" in voc["term"] and "side:us" in voc["term"] and "side:white" in voc["term"]
    assert voc["term"] == sorted(voc["term"]) and voc["max_parts"] == 2
    assert CE.graph_vocab(SPEC, VJ) == voc


def test_graph_dict_constants_fusion_statics():
    d1, d2 = graph(P1), graph(P2)
    scale = lambda d: {t for t in d["terms"] if not t.startswith("clock:")}
    assert scale(d1) == scale(d2) and len(scale(d1)) == len(d1["terms"]) - sum(t.startswith("clock:") for t in d1["terms"])
    squares = {t for t in d1["terms"] if t.startswith("square:")}
    assert len(squares) == 64 and "square:e4" in squares
    # an object is linked to its components; a rank and a clock value are different nodes
    assert {t for t in d1["terms"] if t.startswith("rank:")} == {f"rank:{i}" for i in range(1, 9)}
    p0, p1 = d1["edges"]["part0"], d1["edges"]["part1"]
    e4 = d1["terms"].index("square:e4")
    assert d1["terms"][p0[1][p0[0].index(e4)]] == "file:e" and d1["terms"][p1[1][p1[0].index(e4)]] == "rank:4"
    assert [t for t in d1["terms"] if t.startswith("clock:")] == ["clock:1"] and [t for t in d2["terms"] if t.startswith("clock:")] == ["clock:0"] and "rank:0" not in d1["terms"]   # the scale occurs where it is read
    # static facts: the same prefix, kind 2, present 1
    n = d1["n_static_facts"]
    assert n == d2["n_static_facts"] > 0 and d1["facts"][:n] == d2["facts"][:n] and all(f[1] == 2 and f[2] == 1 for f in d1["facts"][:n])
    assert d1["terms"][:d1["n_static_terms"]] == d2["terms"][:d2["n_static_terms"]]
    assert [fact_row(d1, i) for i in range(n)] == [fact_row(d2, i) for i in range(n)]
    assert any(r[0] == "step" and len(r) == 6 for r in map(lambda i: fact_row(d1, i), range(n)))        # step direction (file rank) (file rank): fused squares unfused again
    # state facts are typed in the mover's perspective
    dyn = [fact_row(d1, i) for i in range(n, len(d1["facts"])) if d1["facts"][i][2] == 1]
    assert ["cell", "e", "4", "them", "pawn"] in dyn and ["cell", "e", "5", "us", "pawn"] in dyn and ["control", "us"] in dyn     # black to move: white is `them`
    assert d1["moves"] == sorted(CE.state("chess", None, P1)[2]) or sorted(d1["moves"]) == sorted(CE.state("chess", None, P1)[2])


def test_delta_edges_match_facts_differences():
    d = graph(P1)
    n = d["n_static_facts"]
    parent = {tuple(r) for r in CE.facts("chess", SPEC, VJ, None, P1, True) if r[0] == "cell"}
    sw = {"us": "them", "them": "us"}
    for name in ("g8f6", "e8e7", "f8b4", "e5e4"):
        if name not in d["moves"]:
            continue
        mi = d["moves"].index(name)
        child = {tuple([r[0]] + [sw.get(a, a) for a in r[1:]]) for r in CE.facts("chess", SPEC, VJ, None, P1 + [name], True) if r[0] == "cell"}
        removed = {tuple(fact_row(d, f)) for s, f in zip(*d["edges"]["removes"]) if s == mi}
        added = {tuple(fact_row(d, f)) for s, f in zip(*d["edges"]["adds"]) if s == mi}
        assert removed == parent - child and added == child - parent, name
        assert all(d["facts"][f][2] == 0 for s, f in zip(*d["edges"]["adds"]) if s == mi and tuple(fact_row(d, f)) not in parent)
    assert "g8f6" in d["moves"] and len(d["edges"]["removes"][0]) > 0 and all(f >= n for f in d["edges"]["adds"][1])


def test_from_engine_round_trip():
    d = graph(P2)
    vals = {m: float(i % 3) for i, m in enumerate(d["moves"])}
    g = FG.from_engine(d, vals, {m: i % 2 for i, m in enumerate(d["moves"])})
    assert g["term"].num_nodes == len(d["terms"]) and g["fact"].num_nodes == len(d["facts"]) and g["move"].num_nodes == len(d["moves"]) and g["claim"].num_nodes == 0
    assert g["fact"].x[:, 0].tolist() == d["rel_ids"] and g["term"].x[:, 0].tolist() == d["term_ids"]
    for name, (s, t) in d["edges"].items():
        src, dst = {"removes": ("move", "fact"), "adds": ("move", "fact"), "binds": ("claim", "fact")}.get(name, ("fact", "term") if name.startswith("arg") else ("term", "term"))
        assert g[(src, name, dst)].edge_index.tolist() == [s, t] or (not s and g[(src, name, dst)].edge_index.shape == (2, 0))
        assert g[(dst, f"rev_{name}", src)].edge_index.tolist() == [t, s] or not s
    assert set(g.edge_types) == set(FG.edge_types(6, 2))
    assert g["move"].y.tolist() == list(vals.values()) and bool(g["move"].has_target.all()) and g["move"].opt.sum() >= 1 and bool(g["move"].has_lost.all())
    assert not bool(FG.from_engine(d, {d["moves"][0]: 1.0})["move"].has_target.any())
    # every term of the dict is in the engine's vocabulary (no UNK)
    assert 0 not in g["term"].x[:, 0].tolist()


def random_positions(n, seed=3, plies=14):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        moves = []
        for _ in range(rng.randrange(1, plies)):
            legal = sorted(CE.state("chess", None, moves)[2])
            if not legal:
                break
            moves.append(rng.choice(legal))
        if CE.state("chess", None, moves)[2]:
            out.append(moves)
    return out


@pytest.fixture(scope="module")
def trained_random(tmp_path_factory):
    """A randomly initialised torch net (norm affine randomised), its checkpoint and its Rust weights file."""
    tmp = tmp_path_factory.mktemp("gnn")
    torch.manual_seed(5)
    voc = CE.graph_vocab(SPEC, VJ)
    vocabs = {"term": FG.Vocab(voc["term"]), "rel": FG.Vocab(voc["rel"])}
    net = GP.GraphNet(len(vocabs["term"]), len(vocabs["rel"]), 16, 2, voc["max_arity"], voc["max_parts"])
    for n, p in net.named_parameters():
        if "norms" in n:
            p.data = (1.0 if n.endswith("weight") else 0.0) + 0.3 * torch.randn_like(p)
    net.eval()
    ck, w = str(tmp / "r.pt"), str(tmp / "r.json")
    torch.save({"state": net.state_dict(), "vocabs": {k: v.to_json() for k, v in vocabs.items()}, "game": "chess",
                "config": {"dim": 16, "layers": 2, "max_arity": voc["max_arity"], "max_parts": voc["max_parts"]}}, ck)
    GP.export(ck, w)
    return net, w


def test_torch_rust_parity(trained_random):
    net, w = trained_random
    model = CE.GraphModel(w, SPECS, VJ)
    worst, n = 0.0, 0
    for moves in random_positions(24):
        names, rust = model.scores("chess", moves)
        g = FG.from_engine(graph(moves))
        with torch.no_grad():
            ts = net(g).tolist()
        assert len(names) == len(ts) == len(rust) > 0
        worst = max([worst] + [abs(a - b) for a, b in zip(ts, rust)])
        n += 1
    assert n >= 20 and worst <= 1e-4, worst
    assert model.timing()["positions"] == n


def test_graph_source_ranks_in_rust(trained_random):
    _net, w = trained_random
    src = GP.GraphSource(w, specs=SPECS, vocabulary=V)
    assert src.source == "gnn" and src.training_only is False
    r = src.rank("chess", P1)
    assert sorted(m for m, _ in r) == sorted(CE.state("chess", None, P1)[2])
    assert all(r[i][1] > r[i + 1][1] or (r[i][1] == r[i + 1][1] and r[i][0] < r[i + 1][0]) for i in range(len(r) - 1))
    assert src.rank("chess", P1) == r and src.ms > 0 and src.timing()["positions"] == 2 and len(src.rank("chess", P1, cap=3)) == 3
    assert src("chess", P1) == r[0][0]
    with pytest.raises(ValueError):
        GP.GraphSource(w.replace(".json", ".pt"), specs=SPECS, vocabulary=V)
    # a weights file made for another vocabulary is refused
    with pytest.raises(ValueError):
        GP.GraphSource(w, specs=SPECS, vocabulary={**V, "derived": []})


def test_source_of_gnn(trained_random, monkeypatch):
    _net, w = trained_random
    GS_texts = lambda game: SPECS
    monkeypatch.setattr(GS, "graph_spec_texts", GS_texts)
    monkeypatch.setattr(GS, "graph_vocabulary", lambda game: V)
    src = SL.source_of(f"gnn:{w}")
    assert src.source == "gnn" and len(src.rank("chess", ["e2e4"])) == 20


def test_tiny_training_end_to_end(tmp_path):
    rng = random.Random(11)
    log = tmp_path / "wake.jsonl"
    items, rows = [], []
    for gi in range(3):
        moves = []
        for _ in range(9):
            legal = sorted(CE.state("chess", None, moves)[2])
            moves.append(rng.choice(legal))
        rows.append({"kind": "game", "run": "t", "game": gi, "moves": moves})
        for ply in range(2, 9):
            rows.append({"kind": "position", "label": f"t:{gi}:{ply}:{ply % 2}", "move": moves[ply], "steps": [], "error": None})
            items.append(("chess", None, moves[:ply], f"t:{gi}:{ply}", f"t:{gi}"))
    log.write_text("\n".join(json.dumps(r) for r in rows))
    cards = CE.build_cards(items, SPECS, VJ, 1, 20_000_000, 0, {"q": 14, "r": 8, "b": 5, "n": 4, "p": 2, "k": 0}, False)
    cp = str(tmp_path / "t.cards")
    cards.save(cp)
    lines = []
    out, w = str(tmp_path / "t.pt"), str(tmp_path / "t.json")
    r = GP.train(cp, out, log=str(log), epochs=2, seed=1, dev_fn=lambda g: g != "t:2", vocabulary=V, specs=SPECS, game="chess", dim=16, layers=2, weights_path=w, say=lines.append)
    assert len(r["history"]) == 2 and r["parity"]["max_abs_diff"] <= 1e-4 and r["parity"]["n"] > 0
    assert any("OBS gnn parity" in x for x in lines) and sum("OBS gnn epoch" in x for x in lines) == 2
    src = GP.GraphSource(w, specs=SPECS, vocabulary=V)
    assert sorted(m for m, _ in src.rank("chess", ["e2e4", "e7e5"])) == sorted(CE.state("chess", None, ["e2e4", "e7e5"])[2])
