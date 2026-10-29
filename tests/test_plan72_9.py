"""PLAN 72.9 (docs/ANALYST_PLAN.md: loop target 1 made concrete -- the judge's rate, the share, the judge's order; W0VFO). The three rules are machinery, not budgets:
(a) a parry is priced at the rate of what it is (check, capture, quiet), learned in the run, the run's rate until a kind has its own;
(b) the judge's first question, 'can one more question here change the decision, cheaply': a route that DECIDES (its question can beat the best claim, the policy ranks it
    above chance, its replies are forced) takes the root's whole share and is read whatever the rate says; every other route is priced by the rate against its fair share;
(c) the judge reads a root question's routes in the policy's order. Numbers are directional (OBS lines)."""
import time

import chess

from catspace.incubator.analyst.discovery import agenda as AGM
from catspace.incubator.analyst.discovery import graph_policy as GP
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import State, state_of

W0VFO = "2r2rk1/R5pp/5p2/2q1p3/3pQ3/8/2P2PPP/R5K1 w - - 4 30"     # rand8_10: Qb7 Rc7 Qxc7 Qxc7 Rxc7
BARE_KING = "7k/8/8/8/8/8/8/R5K1 w - - 0 1"                          # Ra8+ : two parries (Kg7, Kh7), both quiet
LOSING = dict(balance=-1, threat=0, s0=-1, word="losing", line=0, deep=None)


def _rated(fen: str, spn: float = 0.01, seconds: float = 100) -> Agenda:
    ag = Agenda(state_of(fen, "w" if chess.Board(fen).turn else "b"), seconds=seconds)
    ag.ctx.judge_t, ag.ctx.judge_n = 100 * spn, 100                    # the run's rate: spn seconds per cost unit
    return ag


def test_72_9_a_a_parry_is_priced_by_what_it_is():
    # plan 76.2 (b): what a parry IS comes with its price, read by the Game surface (`Game.reply_kind`), not off its SAN
    _cost, kids = R.route_cost(chess.Board(BARE_KING), chess.Move.from_uci("a1a8"), 1, "w")
    assert [k for _s, _c, k in kids] == ["quiet", "quiet"]
    ag = _rated(BARE_KING)
    kids = [("Kxg7", 1, "capture"), ("Kh8", 2, "quiet")]
    assert abs(ag._priced(4, kids, 0.01)[0] - 4 * 0.01) < 1e-12          # no kind timed yet: cost x the run's rate (plan 61 C2 as it was)
    assert ag._priced(4, kids, 0.01)[1].startswith("at the run's rate 0.010 s/unit")
    ag.ctx.kind_rate = {"capture": (8.0, 8), "quiet": (0.5, 5)}                                # captures timed on 8 units (1 s each); quiet replies on 5 only
    need, words = ag._priced(4, kids, 0.01)
    print("OBS 72.9 (a) priced:", round(need, 3), "|", words)
    assert abs(need - (1 * 1.0 + 2 * 0.01 + 1 * 0.01)) < 1e-9                                  # the capture at its own rate, the quiet parry and the route's own node at the run's
    assert "1 capture at its own 1.000 s/unit (8 units timed)" in words and "2 quiet at the run's 0.010 s/unit (only 5 of its units timed)" in words and "the route's own node" in words
    assert ag._low_rate(0.01) == 0.01 and ag._low_rate(2.0) == 1.0                             # the cheapest rate a unit can be priced at bounds where pricing may stop
    ag._kind_tally("quiet", 3, 0.25)
    assert ag.ctx.kind_rate["quiet"] == (0.75, 8)                                               # a parry read to its end adds its seconds and its child's cost to its kind


def test_72_9_a_the_node_judge_prices_by_the_reply_that_made_the_node(monkeypatch):
    monkeypatch.setattr(R, "node_cost", lambda *a, **k: (10, "Ra7+"))
    ag = _rated(BARE_KING)
    ag.ctx.kind_rate = {"capture": (8.0, 8)}
    ag.ctx.deadline = time.monotonic() + 5.0
    ag._route_move = chess.Move.from_uci("a1a8")
    after = chess.Board("7k/6N1/8/8/8/8/8/R5K1 w - - 0 1")                                     # (the route Ra8+ with a knight on g7: the reply Kxg7 takes it)
    after.push(chess.Move.from_uci("a1a8"))
    ag._reply_san, ag._reply_at = "Kh7", (after, "h8h7")                                         # quiet, not timed: 10 x 0.01 s fits in 5 s
    assert ag._judge_node(BARE_KING) is False
    ag._reply_san, ag._reply_at = "Kxg7", (after, "h8g7")                                        # a capture, timed at 1 s a unit: 10 s does not fit
    assert ag._judge_node(BARE_KING) is True
    print("OBS 72.9 (a) node judge:", ag.ctx.c_lines[-1].strip())
    assert "10.0 s needed (capture at its own 1.000 s/unit (8 units timed))" in ag.ctx.c_lines[-1] and "-> GAP" in ag.ctx.c_lines[-1]


def test_72_9_a_the_route_judge_says_which_rate_it_used():
    ag = _rated(BARE_KING, spn=0.01)
    ag._chase_q = None
    ag.ctx.root_end, ag.ctx.deadline = time.monotonic() + 10.0, time.monotonic() + 1.0          # 10 s left of the root's share, a 1 s slice
    ag.ctx.kind_rate = {"quiet": (80.0, 8)}                                                     # quiet replies timed at 10 s a unit
    ag.ctx.route_costs[(AGM._pos_key(BARE_KING), "a1a8", 1)] = (5, [("Kg7", 2, "quiet"), ("Kh7", 2, "quiet")], False)      # the price of Ra8+: two quiet parries with children of cost 2
    res0, tok0 = ag._judge_begin(chess.Move.from_uci("a1a8"), False, False, None)       # 2026-10-29 (52.18) THE MEASUREMENT READ: the first route a kind's own rate would cut is read
    line0 = next(x for x in ag.lines if "judge: Ra8+" in x)                             # whatever its price, once per kind per pass, so the rate is re-measured (one slow early parry had fixed it)
    assert res0 is None and tok0 is not None and "read on to MEASURE the quiet rate" in line0 and "quiet at its own 10.000 s/unit (8 units timed)" in line0, line0
    assert ag._measured_kinds == {"quiet"} and ag.ctx.measuring == {"quiet"}
    ag.ctx.measuring = set()                                                                   # (the route's end clears it: _judge_end)
    ag.lines.clear()
    res, _tok = ag._judge_begin(chess.Move.from_uci("a1a8"), False, False, None)       # the second time the kind is measured for this pass: the price cuts, with the words as before
    line = next(x for x in ag.lines if "judge: Ra8+" in x)
    print("OBS 72.9 (a) route judge:", line.strip())
    assert res is not None and res["gap"].startswith("out of time")
    assert "40.0 s needed priced by what each parry is: 4 quiet at its own 10.000 s/unit (8 units timed), 1 at the run's 0.010 s/unit (the route's own node)" in line and "-> GAP" in line


def _routes(ag: Agenda, spec: list) -> list:
    return [R.Route(ag.board.parse_san(s), s, why) for s, why in spec]


SPEC = [("Qb7", "threat of mate"), ("Qf5", "delayed threat: after every reply a new threat of ours"), ("Qd3", "threat of mate")]


def _record_share(monkeypatch, ag: Agenda, q, probs: "dict | None") -> list:
    seen = []

    def fake(m, promote=False, cheap=False, override=None):
        if cheap:
            return dict(pending=True)
        seen.append((ag.board.san(m), ag.ctx.claim_all, ag.ctx.why_not, ag.ctx.deadline - time.monotonic(), ag.ctx.root_end - time.monotonic()))
        return dict(value=0, killed="test", gap=None, parts=[], lemmas=[])
    monkeypatch.setattr(ag, "_chase_route", fake)
    ag._chase_q = q
    routes = _routes(ag, SPEC)
    ag._policy_p = None if probs is None else ({r.move.uci(): probs[r.san] for r in routes}, len(routes))
    ag._chase_routes(q, routes, False, [], [], {}, ag.ctx.left)
    return seen


def _pool(word: str = "losing") -> Agenda:
    ag = Agenda(State(fen=W0VFO, us="w", pieces=[]), seconds=60)
    ag.ctx.pool, ag.ctx.one_route, ag.status_info = True, word == "losing", dict(LOSING, word=word, s0=-1 if word == "losing" else 0)
    return ag


def test_72_9_b_a_route_that_decides_takes_the_whole_share_the_others_their_fair_slice(monkeypatch):
    ag = _pool()
    q = max(ag.all, key=lambda x: x.stake)                                                      # a mate question: it can beat the best claim (none yet)
    seen = _record_share(monkeypatch, ag, q, {"Qb7": 0.5, "Qf5": 0.4, "Qd3": 0.1})
    print("OBS 72.9 (b) shares:", [(s, bool(c), w, round(d, 1)) for s, c, w, d, _e in seen])
    assert [s for s, *_ in seen] == ["Qb7", "Qf5", "Qd3"]
    assert seen[0][1].startswith("threat of mate (its replies are forced), the policy ranks it above chance (p 0.50 of 3 routes)") and seen[0][3] > 50      # forced, above chance: the whole share
    assert seen[1][1] == "" and seen[1][2] == "its replies are free (delayed threat)" and seen[1][3] < 0.6 * seen[1][4]                   # quiet: its fair slice (two routes open)
    assert seen[2][1] == "" and seen[2][2].startswith("the policy ranks it below chance (p 0.10 of 3 routes)")                              # forced, but below chance (1/3)
    # level: the root question has half the run (52.23 (11)); the route that decides takes all of that half
    ag2 = _pool("level")
    seen2 = _record_share(monkeypatch, ag2, max(ag2.all, key=lambda x: x.stake), {"Qb7": 0.5, "Qf5": 0.4, "Qd3": 0.1})
    assert seen2[0][1] and 25 < seen2[0][4] < 35 and abs(seen2[0][3] - seen2[0][4]) < 1.0
    # no policy ranked the routes (the tiers): nothing decides, the rate prices every route
    ag3 = _pool()
    assert all(c == "" and w == "" for _s, c, w, _d, _e in _record_share(monkeypatch, ag3, max(ag3.all, key=lambda x: x.stake), None))
    # the root question's share: the whole run while the status is lost and it can beat the best claim, else the half
    ag4 = _pool()
    assert ag4._whole_share(max(ag4.all, key=lambda x: x.stake)) != "" and _pool("level")._whole_share(max(ag4.all, key=lambda x: x.stake)) == ""


def test_72_9_b_the_judge_reads_a_route_that_decides_whatever_the_rate_says():
    cost, _kids = R.route_cost(chess.Board(BARE_KING), chess.Move.from_uci("a1a8"), 1, "w")
    for claim, read in (("check (its replies are forced), the policy ranks it above chance (test)", True), ("", False)):
        ag = _rated(BARE_KING, spn=20.0 / cost)                                                 # the rate says the route needs 20 s: more than the whole share
        ag._chase_q = None
        ag.ctx.pool, ag.ctx.one_route = True, True
        now = time.monotonic()
        ag.ctx.root_end, ag.ctx.deadline = now + 10.0, now + (10.0 if claim else 1.0)          # 10 s left of the root's share; the decider's slice is all of it, another route's 1 s
        ag.ctx.claim_all, ag.ctx.why_not = claim, "" if claim else "its replies are free (test)"
        res, _tok = ag._judge_begin(chess.Move.from_uci("a1a8"), False, False, None)
        line = next(x for x in ag.lines if "judge: Ra8+" in x)
        print("OBS 72.9 (b) judge:", line.strip())
        assert (res is None) == read
        if read:
            assert "decides, forced: read -- check (its replies are forced)" in line and "does not apply" in line and "the root's whole share" in line
        else:
            assert "judge: Ra8+ cut (cost" in line and "does not decide: its replies are free (test)" in line and "it may claim half of it" in line and "-> GAP" in line


def test_72_9_c_the_judge_reads_the_routes_in_the_policys_order(monkeypatch):
    captured, scored = [], []

    def score(board, us, qd, rd, *a, **k):                                                      # the policy ranks the LAST generated route first (score = its index)
        scored.append([r["uci"] for r in rd])
        return [float(i) for i in range(len(rd))]

    def run(pool: bool) -> list:
        ag = Agenda(State(fen=W0VFO, us="w", pieces=[]), seconds=60)
        ag.ctx.pool = pool
        q = next(x for x in ag.all if x.goal == "mate")
        monkeypatch.setattr(ag, "_chase_routes", lambda q_, routes, *a: captured.append([r.move.uci() for r in routes]) or False)
        ag._chase(q)
        return captured[-1]
    monkeypatch.setattr(GP, "load", lambda *a, **k: object())
    monkeypatch.setattr(GP, "score_routes", score)
    pooled = run(True)
    policy = list(reversed(scored[-1]))                                                         # the policy's order
    plain = run(False)                                                                          # no policy ordered (the pool off, no PRIOR): the most-forcing sort
    b = chess.Board(W0VFO)
    print("OBS 72.9 (c) policy order:", [b.san(chess.Move.from_uci(u)) for u in pooled][:6], "| forcing sort:", [b.san(chess.Move.from_uci(u)) for u in plain][:6])
    assert len(pooled) >= 3 and pooled == policy                                                # the judge reads exactly the policy's order ...
    assert sorted(plain) == sorted(pooled) and plain != pooled and b.gives_check(chess.Move.from_uci(plain[0]))      # ... where the forcing sort would put a check first


def test_72_9_w0vfo_at_the_count_budget_still_proves_qb7():
    ag = Agenda(State(fen=W0VFO, us="w", pieces=[]))
    ag.ask()
    d = ag.decision()
    print("OBS 72.9 W0VFO count budget:", d.what if d else None, d.value if d else None, d.proved if d else None)
    assert d is not None and d.what == "Qb7" and d.value == 3 and d.proved
    assert ag.ctx.kind_rate == {} and ag.ctx.claim_all == ""                                     # no clock: the judge's rate and share never apply
