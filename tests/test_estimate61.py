"""PLAN 61 stage B: the proved / estimated label (B1), the quiet-status estimate of a cut route (B2) and the calibration pass (B3).

An estimate never becomes proved; a claim with an estimated leaf is proved=False; the decision reads proved claims first and consults an estimated one only when no proved claim keeps the
status; the confidence is the PRODUCT of the estimated leaves' calibrated confidences and comes from the calibration file (0.5 only when the file or class is absent)."""
import json

import chess
import chess.engine
import pytest

import catspace.incubator.analyst  # noqa: F401
from catspace.incubator.analyst.discovery import agenda as AG
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import question_prior as QP
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.state import Claim, best_route_label, combine, state_of

FREE_PAWN = "7k/8/7p/8/8/8/8/2B3K1 w - - 0 1"                    # Bxh6 wins a pawn; both replies of the bare king leave a QUIET leaf (no check, no threat, no pass threat)
CHECKABLE = "6k1/8/8/8/8/8/8/R5K1 w - - 0 1"                      # Ra8+ is forcing: not quiet
NOT_QUIET_AFTER = "4k3/pp6/7p/8/8/8/5PPP/2B3K1 w - - 0 1"         # after Bxh6 a reply leaves a forcing move of ours: the gap stays a gap


@pytest.fixture(autouse=True)
def _setup(monkeypatch, tmp_path):
    BV.install_terms()
    monkeypatch.setattr(AG, "ESTIMATE", True)
    monkeypatch.setattr(QP.paths, "derived", lambda *a: str(tmp_path / a[-1]))
    monkeypatch.setattr(QP.paths, "experiment", lambda *a: str(tmp_path / a[-1]))
    monkeypatch.setattr(QP, "_cal", [None])


def _agenda(fen, us="w"):
    ag = AG.Agenda(state_of(fen, us), seconds=5)
    ag._chase_q = R.new_question("win", 1, dict(target=chess.H6), text="test")
    return ag


def _gap_estimate(fen):
    ag = _agenda(fen)
    res = dict(value=0, killed=None, gap="out of time: Bxh6 not started", parts=[], lemmas=[])
    ag._estimate_route(chess.Move.from_uci("c1h6"), res)
    return ag, res


def _claim(proved=True, conf=1.0, value=1, move="c1h6"):
    return Claim("win", "Bxh6", value, "l", "line", move=move, proved=proved, confidence=conf, estimator="" if proved else "quiet-status")


# ---- B1 the label ----------------------------------------------------------------------------------------------------------------------------------

def test_b1_a_claim_is_proved_by_default_and_the_estimate_is_labelled():
    c = Claim("win", "x", 1, "l", "line")
    assert (c.proved, c.confidence, c.estimator) == (True, 1.0, "")
    e = _claim(False, 0.4)
    assert (e.proved, e.confidence, e.estimator) == (False, 0.4, "quiet-status")


def test_b1_a_route_is_proved_iff_every_reply_is_and_its_confidence_is_the_product():
    assert combine([(True, 1.0), (True, 1.0)]) == (True, 1.0)
    proved, conf = combine([(True, 1.0), (False, 0.5), (False, 0.4)])
    assert proved is False and conf == pytest.approx(0.2)
    assert combine([(False, 0.9)])[0] is False                       # one estimated reply makes the route estimated; it never becomes proved
    assert combine([]) == (True, 1.0)


def test_b1_a_question_takes_its_best_route_and_a_proved_route_wins_a_tie():
    assert best_route_label([(3, False, 0.9), (2, True, 1.0)]) == (3, False, 0.9)
    assert best_route_label([(2, False, 0.9), (2, True, 1.0)]) == (2, True, 1.0)
    assert best_route_label([]) is None


def _decide(proved_best, est_best, fen=FREE_PAWN):
    ag = _agenda(fen)
    ag._pass_done, ag._pass_best = True, None                        # (their pass threat 0: the status is read without running their agenda)
    ag.best = None if proved_best is None else (proved_best.value, 0, proved_best)
    ag.best_est = None if est_best is None else (est_best.value, 0, est_best)
    return ag, ag.decision()


def test_b1_the_decision_prefers_a_proved_claim_that_keeps_the_status_and_never_prints_estimated():
    # status: B vs p -> balance +2, line 'stay ahead' (1): a proved +1 keeps it
    ag, d = _decide(_claim(True, value=1), _claim(False, 0.9, value=5, move="c1d2"))
    assert d.proved and d.value == 1
    assert not any("ESTIMATED" in l for l in ag.lines) and ag.decision_estimated() is False


def test_b1_an_estimated_claim_decides_only_when_no_proved_claim_keeps_the_status():
    # a proved claim of -9 does not keep it (2 - 9 < 1); the estimated +1 does (2 + 1 >= 1)
    ag, d = _decide(_claim(True, value=-9), _claim(False, 0.4, value=1))
    assert not d.proved and d.estimator == "quiet-status" and ag.decision_estimated()
    line = next(l for l in ag.lines if l.startswith("DECISION ESTIMATED"))
    assert "Bxh6" in line and "confidence 0.40" in line and "by quiet-status" in line and "no proved claim keeps the status" in line
    # no proved claim at all
    ag, d = _decide(None, _claim(False, 0.4, value=1))
    assert not d.proved
    # an estimated claim that does not keep the status does not decide either
    ag, d = _decide(None, _claim(False, 0.4, value=-9))
    assert ag.decision_estimated() is False and d is None


def test_b1_the_flag_off_ignores_an_estimated_claim(monkeypatch):
    monkeypatch.setattr(AG, "ESTIMATE", False)
    ag, d = _decide(None, _claim(False, 0.9, value=5))
    assert d is None


# ---- B2 the quiet-status estimate ------------------------------------------------------------------------------------------------------------------

def test_b2_a_quiet_gap_closes_as_an_estimated_claim_with_the_product_of_the_class_confidence(tmp_path):
    (tmp_path / QP.CALIBRATION).write_text(json.dumps({"quiet:us": {"n": 10, "held": 8, "confidence": 0.8}}))
    ag, res = _gap_estimate(FREE_PAWN)
    est = res["est"]
    assert res["gap"] is not None and res["killed"] is None             # still a GAP for the flow
    assert est["proved"] is False and est["estimator"] == "quiet-status" and est["value"] == 1
    assert len(est["leaves"]) == 2 and est["confidence"] == pytest.approx(0.64)     # two quiet leaves: 0.8 x 0.8, from the calibration file
    assert all(lf["cls"] == "quiet:us" and lf["status"] == 3 for lf in est["leaves"])


def test_b2_the_default_confidence_is_half_only_when_the_file_is_absent():
    ag, res = _gap_estimate(FREE_PAWN)
    assert res["est"]["confidence"] == pytest.approx(0.25)              # 0.5 x 0.5


def test_b2_the_estimated_claim_is_filed_apart_from_the_proved_ones_and_the_leaves_are_logged(tmp_path):
    ag, res = _gap_estimate(FREE_PAWN)
    routes = [R.Route(chess.Move.from_uci("c1h6"), "Bxh6", "capture 1")]
    ag._est_claims(ag._chase_q, routes, {0: res}, [])
    assert ag.state.claims == [] and ag.best is None                    # the proved side is untouched
    (c,) = ag.state.estimated
    assert c.proved is False and c.value == 1 and c.confidence == pytest.approx(0.25)
    assert ag.best_est[2] is c
    assert any("CLAIM ESTIMATED" in l for l in ag.lines)
    ag._pass_done, ag._pass_best = True, None
    d = ag.decision()
    assert d is c and ag.decision_estimated()
    rows = [json.loads(l) for l in (tmp_path / QP.EST_LOG).read_text().splitlines()]
    assert len(rows) == 2 and all(r["estimator"] == "quiet-status" and r["leaned"] is True and r["route"] == "c1h6" and r["cls"] == "quiet:us" for r in rows)
    assert {"fen", "value", "confidence", "estimator", "cls", "route", "set", "id", "leaned", "status", "us"} <= set(rows[0])


def test_b2_a_proved_route_at_least_as_good_needs_no_estimate():
    ag, res = _gap_estimate(FREE_PAWN)
    routes = [R.Route(chess.Move.from_uci("c1h6"), "Bxh6", "capture 1")]
    ag._est_claims(ag._chase_q, routes, {0: res}, [(1, routes[0], dict(parts=[]))])
    assert ag.state.estimated == [] and ag.best_est is None


def test_b2_a_non_quiet_gap_stays_a_gap():
    assert not _agenda(CHECKABLE)._quiet_leaf("6k1/8/8/8/8/8/8/R5K1 w - - 0 1")      # Ra8+ is a forcing move of ours
    ag, res = _gap_estimate(NOT_QUIET_AFTER)
    assert "est" not in res and res["gap"] is not None and ag.ctx.est_rows == []


def test_b2_an_estimate_below_the_stake_claims_nothing():
    ag = _agenda("7k/8/7p/8/8/8/8/2B3K1 w - - 0 1")
    res = dict(value=0, killed=None, gap="out of time: Kf1 not started", parts=[], lemmas=[])
    ag._estimate_route(chess.Move.from_uci("g1f1"), res)               # a king move: the material balance does not change (value 0 < 1)
    assert "est" not in res


def test_b2_the_flag_defaults_off(monkeypatch):
    monkeypatch.undo()
    assert AG.ESTIMATE is False


def test_b2_the_flag_on_changes_no_decision_on_a_plain_puzzle():
    fen = "1r2k3/1bp2p2/p3p2Q/q5B1/3b4/P7/5PPP/2RR2K1 b - - 4 29"
    out = []
    for on in (False, True):
        AG.ESTIMATE = on
        ag = AG.Agenda(state_of(fen, "b"))
        ag.ask()
        d = ag.decision()
        out.append((d.move, d.value, d.proved) if d else None)
    assert out[0] == out[1]


# ---- B3 the calibration pass -----------------------------------------------------------------------------------------------------------------------

def test_b3_held_when_the_sign_against_level_agrees_within_the_tolerance():
    assert QP.held(1, 120) is True                                      # we claimed level-or-better, Stockfish agrees
    assert QP.held(0, -30) is True                                      # level, and within 50 cp of it
    assert QP.held(0, -80) is False                                     # claimed level or better, Stockfish says clearly worse
    assert QP.held(-2, -300) is True                                    # claimed worse, Stockfish agrees
    assert QP.held(-2, 40) is False                                     # claimed worse, Stockfish says level or better
    assert QP.held(3, None) is None                                     # ungraded


class _Fake:
    def __init__(self, cps):
        self.cps, self.calls = cps, 0

    def analyse(self, board, limit):
        self.calls += 1
        return {"score": chess.engine.PovScore(chess.engine.Cp(self.cps[board.fen()]), board.turn)}


def test_b3_calibrate_scores_ungraded_leaves_and_writes_the_held_fraction_per_class(tmp_path, monkeypatch):
    from catspace.incubator.analyst.discovery import verdict as VD
    f1, f2, f3 = "6k1/8/7B/8/8/8/8/6K1 w - - 1 2", "8/7k/7B/8/8/8/8/6K1 w - - 1 2", "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    rows = [dict(fen=f1, status=3, cls="quiet:us", us="w"), dict(fen=f2, status=3, cls="quiet:us", us="w"), dict(fen=f3, status=1, cls="quiet:us", us="w"),
            dict(fen=f3, status=-2, cls="quiet:them", us="w"), dict(fen=f1, status=3, cls="quiet:us", us="w", sf_cp=500, held=True)]       # (the last one is already graded)
    log = tmp_path / "est.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fake = _Fake({f1: 400, f2: -300, f3: 0})
    monkeypatch.setattr(VD, "_get", lambda: fake)
    table = QP.calibrate(str(log), out=str(tmp_path / "cal.json"))
    assert fake.calls == 3                                              # one call per FEN not yet graded
    assert table["quiet:us"] == dict(n=4, held=3, confidence=0.75)       # f1 yes, f2 no (claimed +3, SF -300), f3 yes (1 vs 0), the graded one yes
    assert table["quiet:them"]["held"] == 0 and table["quiet:them"]["confidence"] == 0.0    # claimed -2, SF level -> not held
    again = [json.loads(l) for l in log.read_text().splitlines()]
    assert all(r["sf_cp"] is not None and r["held"] is not None for r in again)
    assert json.loads((tmp_path / "cal.json").read_text()) == table


def test_b3_the_confidence_is_the_calibration_file_and_nothing_else(tmp_path):
    assert QP.estimate_confidence("quiet:us") == QP.DEFAULT_CONFIDENCE == 0.5
    (tmp_path / QP.CALIBRATION).write_text(json.dumps({"quiet:us": {"n": 4, "held": 3, "confidence": 0.75}}))
    QP._cal[0] = None
    assert QP.estimate_confidence("quiet:us") == 0.75 and QP.estimate_confidence("quiet:them") == 0.5
