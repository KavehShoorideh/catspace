"""PLAN 60 / 69.1: the GRAPH QUESTION POLICY (graph_policy.py). It sees the board (squares, pieces, attacks) and the facts so far (the questions asked with their outcome, the status) and
ORDERS the open questions; nothing it does touches a claim, a kill or a stop. Numbers are directional (OBS lines)."""
import chess
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from catspace.incubator.analyst.discovery import graph_policy as GP

ESJQH = "8/1Rb3kp/p2q2p1/2p1pp2/pQ6/1bP4P/1P3PPK/8 w - - 0 35"


def _qs():
    return [dict(goal="mate", stake=100, depth=0, target=chess.C7, status="killed", word="tempo", value=0),
            dict(goal="win", stake=3, depth=0, target=chess.B3, status="open"),
            dict(goal="win", stake=1, depth=0, target=chess.A4, status="open")]


def test_60_the_graph_holds_the_board_the_questions_and_the_status():
    b = chess.Board(ESJQH)
    g = GP.build(b, True, _qs(), dict(balance=-3, threat=7, line=0), 0.5)
    assert g["square"].x.shape[0] == 64 and g["piece"].x.shape[0] == len(b.piece_map()) and g["question"].x.shape[0] == 3 and g["root"].x.shape == (1, 6)
    assert g["piece", "attacks", "square"].edge_index.shape[1] > 0 and g["question", "targets", "square"].edge_index.shape[1] == 3
    assert any(r.startswith("rev_") for _a, r, _b in g.edge_types)                       # messages flow both ways
    assert float(g["root"].x[0, 3]) == pytest.approx(0.3)                                 # the need (line - balance) / 10


def test_60_the_model_scores_every_question_and_the_order_is_all_it_touches():
    b = chess.Board(ESJQH)
    g = GP.build(b, True, _qs(), None)
    m = GP.make_model((g.node_types, g.edge_types))
    with torch.no_grad():
        s = m(g)
    assert s.shape == (3,)


def test_60_the_examples_reveal_only_what_was_asked_before_the_decision_point():
    rows = [dict(id="x", set="s", run="r", order=0, fen=ESJQH, goal="mate", stake=100, status="killed", word="tempo", decided=False, decision="b4a5", sf="RIGHT", answer="b4a5", target_sq=chess.C7, move=None),
            dict(id="x", set="s", run="r", order=1, fen=ESJQH, goal="win", stake=3, status="killed", word="verify", decided=False, decision="b4a5", sf="RIGHT", answer="b4a5", target_sq=chess.B3, move=None),
            dict(id="x", set="s", run="r", order=2, fen=ESJQH, goal="win", stake=3, status="verified", word="chase", decided=True, decision="b4a5", sf="RIGHT", answer="b4a5", target_sq=chess.C7, move=None)]
    ex = GP._examples(rows)
    assert len(ex) == 2                                                                   # decision points k = 0, 1 before the decider at 2
    _b, _us, qs, open_idx, pos, _st = ex[1]
    assert pos == 2 and open_idx == [1, 2] and qs[0]["status"] == "killed" and qs[1]["status"] == "open" and qs[2]["status"] == "open"
    rows2 = [dict(r, sf="WRONG", decided=False, decision="h6h2") for r in rows]          # not RIGHT: the question about the answer's square is the target
    rows2[1]["target_sq"] = chess.A5
    ex2 = GP._examples(rows2)
    assert ex2 and ex2[0][4] == 1
