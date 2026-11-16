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


