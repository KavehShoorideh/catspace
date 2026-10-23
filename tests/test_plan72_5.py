"""PLAN 72.5 THE SURVEY BEFORE THE QUESTIONS: for every unit the cheapest layer-2 reach to each job (regress.survey), its plain-words trace lines (once per root agenda), and its features for the
graph policy (version 1; a version-0 model keeps running on the old vectors). It orders, it never decides. Numbers are directional (OBS lines)."""
import os
import time

import chess
import pytest

from catspace.incubator.analyst.discovery import agenda as AGM
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.state import state_of

DISCOVERY = "r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1"
POS2 = "r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1"
KNIGHT = "4k3/5p2/8/8/8/8/8/4K1N1 w - - 0 1"
CAGED = "4k3/8/8/8/8/8/1P6/B3K3 w - - 0 1"
BISHOP_PATH = "4k3/8/8/8/2P5/4B3/8/4K3 w - - 0 1"


def _row(rows, sq, kind, job=None):
    return [r for r in rows if r.sq == sq and r.job_kind == kind and (job is None or r.job == job)]


def test_72_5_a_knight_that_joins_in_two_a_caged_bishop_and_a_held_path():
    rows = R.survey(chess.Board(KNIGHT), 3)
    (r,) = _row(rows, chess.G1, "attack", chess.F7)
    assert (r.material, r.tempi) == (0, 2) and r.via is not None                          # Nh3-g5 bears on f7
    assert _row(rows, chess.G1, "free")[0].count > 0
    caged = R.survey(chess.Board(CAGED), 3)
    assert _row(caged, chess.A1, "free")[0].count == 0 and not [r for r in caged if r.sq == chess.A1 and r.job_kind != "free"]      # the bishop behind its own pawn: no landing, no job
    held = R.survey(chess.Board(BISHOP_PATH), 3)
    assert any(r.tempi == 0 for r in _row(held, chess.E3, "path"))                        # Be3 bears on c5 now: the c-pawn's way
    assert not _row(held, chess.C4, "path")                                               # (a pawn does not hold its own path)


def test_72_5_the_printed_lines_are_plain_words_one_per_unit_at_most_the_fuse():
    held = R.survey_lines(chess.Board(BISHOP_PATH), True)
    assert "survey: our B on e3 holds our c-pawn's path" in held
    caged = R.survey_lines(chess.Board(CAGED), True)
    assert "survey: our B on a1 can join nothing within 3 and has no free landing" in caged
    knight = R.survey_lines(chess.Board("4k3/5r2/8/8/8/8/8/4K1N1 w - - 0 1"), True)
    assert any(l.startswith("survey: our N on g1 reaches ") and "bearing on their R on f7" in l for l in knight)
    for fen in (DISCOVERY, POS2):
        ls = R.survey_lines(chess.Board(fen), True)
        assert 0 < len(ls) <= R.SURVEY_LINES and all(l.startswith("survey: ") for l in ls)
        assert len({l.split(" reaches ")[0].split(" holds ")[0].split(" can ")[0].split(" comes ")[0] for l in ls}) == len(ls)       # one line per unit


def test_72_5_the_survey_is_cheap_obs():
    for name, fen in (("discovery", DISCOVERY), ("pos2", POS2)):
        b = chess.Board(fen)
        t = time.perf_counter()
        rows = R.survey(b)
        ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        R.survey_features(b, True, [chess.E8, None])
        ms2 = (time.perf_counter() - t) * 1000
        print(f"OBS survey {name}: {len(rows)} rows from {len(b.piece_map())} units in {ms:.0f} ms (rolls memoised after: features {ms2:.1f} ms)")
        assert len(rows) > len(b.piece_map())


def test_72_5_the_trace_prints_the_survey_once_per_root_and_not_in_inner_agendas():
    ag = AGM.Agenda(state_of(BISHOP_PATH, "w"))
    assert sum(1 for l in ag.lines if l.startswith("survey:")) == len(R.survey_lines(ag.board, ag.me)) > 0
    ag.ask()
    assert sum(1 for l in ag.trace().splitlines() if l.lstrip().startswith("survey:")) == len(R.survey_lines(ag.board, ag.me))     # the chase's inner agendas print none
    their = AGM.Agenda(state_of(BISHOP_PATH, "w"), side_depth=0)
    assert not any(l.startswith("survey:") for l in their.lines)
    AGM.SURVEY = False
    try:
        assert not any(l.startswith("survey:") for l in AGM.Agenda(state_of(BISHOP_PATH, "w")).lines)
    finally:
        AGM.SURVEY = True


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from catspace.incubator.analyst.discovery import graph_policy as GP


def test_72_5_feature_widths_version_0_unchanged_version_1_adds_four_and_one():
    b = chess.Board(DISCOVERY)
    qs = [dict(goal="win", stake=3, depth=0, target=chess.E7, status="open"), dict(goal="mate", stake=100, depth=0, target=None, status="open")]
    g0 = GP.build(b, True, qs, None)
    g1 = GP.build(b, True, qs, None, survey=GP.survey_for(b, True, qs))
    assert g1["piece"].x.shape[1] == g0["piece"].x.shape[1] + 4 and g1["question"].x.shape[1] == g0["question"].x.shape[1] + 1
    assert torch.equal(g1["piece"].x[:, : g0["piece"].x.shape[1]], g0["piece"].x) and torch.equal(g1["question"].x[:, : g0["question"].x.shape[1]], g0["question"].x)      # the old columns are untouched
    assert float(g1["question"].x[1, -1]) == 1.0                                          # no target: the none sentinel
    u, q = GP.survey_for(b, True, qs)
    assert all(0.0 <= x <= 1.0 for v in u.values() for x in v) and 0.0 <= q[0][0] <= 1.0


def _save(tmp_path, features):
    b = chess.Board()
    qs = [dict(goal="win", stake=1, depth=0, target=0, status="open")]
    rt = [dict(q=0, uci="e2e4", why="threat 1", cost=3, order=0)]
    g = GP.build(b, True, qs, None, routes=rt, survey=GP.survey_for(b, True, qs) if features else None)
    torch.manual_seed(0)
    m = GP.make_model((g.node_types, g.edge_types))
    with torch.no_grad():
        m(g)
        m(g, routes=True)
    blob = dict(state=m.state_dict(), meta=None, hidden=GP.HIDDEN)
    if features:
        blob["features"] = features
    p = tmp_path / f"m{features}.pt"
    torch.save(blob, p)
    return p


@pytest.fixture
def fresh_policy():
    saved = (GP._model[0], GP._loaded[0], GP._feat[0])
    GP._loaded[0] = False
    yield
    GP._model[0], GP._loaded[0], GP._feat[0] = saved


def test_72_5_an_old_model_still_loads_and_scores_on_the_old_vectors_and_a_new_one_on_the_new(tmp_path, fresh_policy):
    b = chess.Board(DISCOVERY)
    qs = [dict(goal="win", stake=3, depth=0, target=chess.E7, status="open"), dict(goal="mate", stake=100, depth=0, target=None, status="open")]
    rts = [dict(uci="e2e4", why="threat 1", cost=3, order=0)]
    GP.load(_save(tmp_path, 0))                                                           # a blob without the key = version 0
    assert GP._model[0] is not None and GP._feat[0] == 0
    s0 = GP.score_questions(b, True, qs, None)
    with torch.no_grad():
        direct = [float(x) for x in GP._model[0](GP.build(b, True, qs, None))]
    assert s0 == direct and len(s0) == 2                                                  # the survey changes no score of a version-0 model
    assert len(GP.score_routes(b, True, qs[0], rts, None)) == 1
    GP._loaded[0] = False
    GP.load(_save(tmp_path, 1))
    assert GP._feat[0] == 1 and len(GP.score_questions(b, True, qs, None)) == 2
    assert len(GP.score_routes(b, True, qs[0], rts, None)) == 1


def test_72_5_the_saved_model_in_data_derived_still_loads_and_scores(fresh_policy):
    from catspace.io import paths
    if not os.path.exists(paths.derived(GP.MODEL)):
        pytest.skip("no saved graph policy in this checkout")
    assert GP.load() is not None
    s = GP.score_questions(chess.Board(DISCOVERY), True, [dict(goal="win", stake=3, depth=0, target=chess.E7, status="open")], None)
    assert len(s) == 1
