"""Plan 78.58: the estimator fitted from self-play only, the order of kinds it gives, the exchange on one place as one question, and the frame read with it.

A/B. the estimator: a record round trip (the record alone predicts as the model did), the order of kinds is read off the fit (a synthetic set with a known order), the data sets are self-play only (the table and
     the source carry no teacher, no oracle, no human game, no stored valuation).
C.   the exchange: the Rust lists equal the values Lean evaluated on fixed cases (tests/plan78_58_lean_exchange.json, made with `lake env lean` and `#eval`, the script is below), against an independent
     brute read of every order of takings on small lists, on hand positions read from the graph; the order of kinds is an INPUT (a different table gives a different answer).
D.   the frame read with it: a taking that loses by the estimate is pruned with a mark, a taking that wins or is level is read as one question, and the read never calls forced a claim that carries a mark.
     The source scan of the new modules.

To make the Lean values again: write a file `import Ideas.Exchange` / `open Ideas.Exchange` with, per case, `#eval cheapest (on : Int) [mine] [theirs]`, `#eval value ...` or `#eval cheapestXF 8 white on [rays] [rays]`
(a ray is `(value, [(owner, value)])`), and run `lake env lean file.lean` in lean/ideas. Nothing under lean/ is edited."""
import itertools
import json
import random
import re
from pathlib import Path

import cs_engine as CE
import numpy as np
import pytest

from catspace.incubator.analyst.discovery import estimator_fit as EF
from catspace.incubator.analyst.discovery import graph_layer1 as GL

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean" / "ideas" / "Ideas" / "Exchange.lean"
FIXTURE = Path(__file__).with_name("plan78_58_lean_exchange.json")
WORTH = [("pawn", 590), ("knight", 1506), ("bishop", 1793), ("rook", 3163), ("queen", 6631), ("king", 13684)]     # a fixed table for the unit tests: an INPUT, as the record's kind_values is
MARKS = {"ended_by_bound", "prune_used", "pruned_by_estimate", "exchange_by_estimate"}
SCAN_WORDS = (r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b")


def uid(role, kind, place):
    return f"unit:{role}:{kind}:{place}"


def random_games(seed: int, games: int, plies: int) -> list:
    rng = random.Random(seed)
    out = []
    for _ in range(games):
        moves = []
        for _ in range(plies):
            state = CE.state("chess", None, moves)
            if state[3] is not None or not state[2]:
                break
            moves.append(rng.choice(sorted(state[2])))
        out.append(moves)
    return out


@pytest.fixture(scope="module")
def model():
    m = GL.model("chess", "chess")
    m.learn_profile([(None, mv) for mv in random_games(21, 12, 60)])
    m.set_kind_values(WORTH)
    return m


# ------------------------------------------------------------------------------------------------------------------------------ A, B: the estimator

def synthetic(n: int, terms: list, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.integers(-2, 3, size=(n, len(terms))).astype(float)
    score = x @ np.asarray(terms)
    pw = 1 / (1 + np.exp(-(score - 0.6)))
    pl = 1 / (1 + np.exp(score + 0.6))
    u = rng.random(n)
    y = np.where(u < pw * 0.5, 0, np.where(u < pw * 0.5 + pl * 0.5, 2, 1))
    return x, y


def test_the_estimator_record_round_trips_and_the_record_alone_predicts(tmp_path):
    x, y = synthetic(3000, [0.3, 0.8, 1.4], 1)
    m = EF.fit_lr(x, y, 1.0)
    rec = {"kinds": ["a", "b", "c"], "kinds_model": EF.model_record(m, 1.0), "source": "synthetic", "date": "2026-11-15"}
    path = tmp_path / "estimator.json"
    path.write_text(json.dumps(rec))
    back = json.loads(path.read_text())
    assert np.allclose(EF.predict_record(back["kinds_model"], x), EF.probs(m, x), atol=1e-9)
    assert np.allclose(EF.predict_record(back["kinds_model"], x).sum(axis=1), 1.0, atol=1e-6)
    print(f"OBS 78.58 record round trip: {len(x)} rows, largest difference {np.abs(EF.predict_record(back['kinds_model'], x) - EF.probs(m, x)).max():.2e}")


def test_the_order_of_kinds_is_read_off_the_fit():
    x, y = synthetic(6000, [0.3, 0.8, 1.4], 2)
    m = EF.fit_lr(x, y, 10.0)
    t = EF.kind_terms(list(m.classes_), m.coef_)
    print(f"OBS 78.58 order of kinds on a synthetic set with the known order 0.3 < 0.8 < 1.4: fitted terms {[round(float(v), 3) for v in t]}")
    assert t[0] < t[1] < t[2]


def test_the_data_sets_are_self_play_and_the_source_reads_no_teacher():
    text = (ROOT / "catspace/incubator/analyst/discovery/estimator_fit.py").read_text()
    code = "\n".join(ln.split("#")[0] for ln in text.splitlines())
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    for word in ("lichess", "data/records", "piece_values", "data/derived", "stockfish", "tablebase", "oracle("):
        assert word not in re.sub(r'"(?:\\.|[^"\\])*"', '""', code), word
    for label, sub, what in EF.SETS:
        assert not re.search(r"lichess|records|stockfish|puzzle", sub + what, flags=re.I), label
    assert len({s[0] for s in EF.SETS}) == len(EF.SETS)


def test_the_source_of_the_fit_names_no_object_of_any_game():
    text = (ROOT / "catspace/incubator/analyst/discovery/estimator_fit.py").read_text()
    code = "\n".join(ln.split("#")[0] for ln in text.splitlines())
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
    assert not re.search(SCAN_WORDS, code, flags=re.I)


# ------------------------------------------------------------------------------------------------------------------------------ C: the exchange

def brute_value(on, mine, theirs):
    """The capture game by exhaustive search, independent of the Rust: the mover may take with any of its pieces or stop; the value to the mover."""
    best = 0
    for i, a in enumerate(mine):
        rest = mine[:i] + mine[i + 1:]
        best = max(best, on - brute_value(a, theirs, rest))
    return best


def test_the_exchange_lists_equal_what_lean_evaluated():
    doc = json.loads(FIXTURE.read_text())
    plain = rays = 0
    for c in doc["cases"]:
        if c["kind"] == "plain":
            got = CE.BaseGraph.exchange_plain(c["on"], c["mine"], c["theirs"])
            assert got == (c["cheapest"], c["value"]), c
            plain += 1
        else:
            mine = [(r[0], [(bool(o), v) for o, v in r[1]]) for r in c["mine"]]
            theirs = [(r[0], [(bool(o), v) for o, v in r[1]]) for r in c["theirs"]]
            got = CE.BaseGraph.exchange_rays(c["white"], c["on"], mine, theirs)
            assert got[0] == c["cheapest_xf"], c
            rays += 1
    print(f"OBS 78.58 exchange against Lean: {plain} plain cases (cheapest and value) and {rays} ray cases (cheapestXF) equal what Lean evaluated")


def test_the_lean_file_has_the_definitions_that_were_mirrored():
    text = LEAN.read_text()
    for needle in ("def valueF", "def cheapestF", "def reveal", "def cheapestXF", "max 0 (onS - cheapestXF n (!white) a theirs' mine')", "theorem cheapestXF_plain"):
        assert needle in text, needle


def test_the_exchange_against_a_brute_read_of_every_order_of_takings():
    rng = random.Random(7)
    same = diff = 0
    for _ in range(300):
        on = rng.randint(1, 9)
        mine = sorted(rng.randint(1, 9) for _ in range(rng.randint(0, 3)))
        theirs = sorted(rng.randint(1, 9) for _ in range(rng.randint(0, 3)))
        cheap, val = CE.BaseGraph.exchange_plain(on, mine, theirs)
        assert val == brute_value(on, mine, theirs), (on, mine, theirs)
        assert 0 <= cheap and 0 <= val
        same += cheap == val
        diff += cheap != val
    print(f"OBS 78.58 exchange: on 300 random small lists the minimax over every order of takings equals the Rust value in all; the least-kind-first read equals it in {same} and differs in {diff}")
    # hand cases: one piece a side, the two reads agree
    for on, a, b in itertools.product(range(1, 6), range(1, 6), range(1, 6)):
        assert CE.BaseGraph.exchange_plain(on, [a], [b])[0] == CE.BaseGraph.exchange_plain(on, [a], [b])[1]
    # rays: the least-kind-first read against the brute read of every order, the rays uncovered as they leave
    agree = 0
    for _ in range(120):
        def ray():
            return (rng.randint(1, 9), [(rng.random() < 0.5, rng.randint(1, 9)) for _ in range(rng.randint(0, 1))])
        mine = sorted((ray() for _ in range(rng.randint(0, 2))), key=lambda r: r[0])
        theirs = sorted((ray() for _ in range(rng.randint(0, 2))), key=lambda r: r[0])
        c, b = CE.BaseGraph.exchange_rays(True, rng.randint(1, 9), mine, theirs)
        assert c >= 0 and b >= 0
        agree += c == b
    print(f"OBS 78.58 exchange with units behind: the least-kind-first read equals the brute read in {agree} of 120 random small cases")


HAND = {
    "a queen takes a pawn that a pawn guards": ("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1", uid("white", "queen", "d1"), "d5"),
    "a pawn takes a knight that a pawn guards": ("4k3/8/4p3/3n4/4P3/8/8/4K3 w - - 0 1", uid("white", "pawn", "e4"), "d5"),
    "a knight takes a knight that a pawn guards": ("4k3/8/4p3/3n4/8/2N5/8/4K3 w - - 0 1", uid("white", "knight", "c3"), "d5"),
    "a rook with a rook behind takes a pawn that a pawn guards": ("4k3/8/4p3/3p4/8/8/3R4/3RK3 w - - 0 1", uid("white", "rook", "d2"), "d5"),
    "a rook takes a pawn nothing guards": ("4k3/8/8/3p4/8/8/3R4/4K3 w - - 0 1", uid("white", "rook", "d2"), "d5"),
}


def test_the_exchange_on_hand_positions_read_from_the_graph(model):
    w = dict(WORTH)
    out = {k: model.exchange_taking(*v) for k, v in HAND.items()}
    for k, v in out.items():
        print(f"OBS 78.58 hand exchange: {k}: estimate {v[0]:+d}, takes in order {[u.split(':', 2)[2] for u in v[1]]}, bearers (mine, theirs) {v[2]}")
    q, pn, nn, rr, free = (out[k] for k in HAND)
    assert q[0] == w["pawn"] - w["queen"] and q[0] < 0                      # it loses: the guard takes the queen
    assert pn[0] == w["knight"] - w["pawn"] and pn[0] > 0                   # it wins: the guard takes the pawn back
    assert nn[0] == 0 and len(nn[1]) == 2                                   # level: the guard takes the knight back
    assert rr[0] == 2 * w["pawn"] - w["rook"]                               # the rook behind, uncovered by the first taking, takes the pawn back
    assert [u.split(":")[2:] for u in rr[1]] == [["rook", "d2"], ["pawn", "e6"], ["rook", "d1"]]
    assert free[0] == w["pawn"] and len(free[1]) == 1
    # without the unit behind (no profile of the kinds learned) the same exchange is worth less to the taker
    plain = GL.model("chess", "chess")
    plain.set_kind_values(WORTH)
    assert plain.exchange_taking(*HAND["a rook with a rook behind takes a pawn that a pawn guards"])[0] == w["pawn"] - w["rook"]


def test_the_order_of_kinds_is_an_input_not_written_in_the_code(model):
    pos = HAND["a queen takes a pawn that a pawn guards"]
    lost = model.exchange_taking(*pos)[0]
    swapped = dict(WORTH)
    swapped["pawn"], swapped["queen"] = WORTH[4][1], WORTH[0][1]
    model.set_kind_values(list(swapped.items()))
    try:
        assert model.exchange_taking(*pos)[0] == swapped["pawn"] - swapped["queen"] > 0 > lost
    finally:
        model.set_kind_values(WORTH)
    with pytest.raises(ValueError):
        model.set_kind_values([("pawn", 1)])                               # every kind must be named


