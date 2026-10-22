"""PLAN 66 (docs/ANALYST_PLAN.md, 'proofs carry across tiers'). The Lean statements are mirrored and tested against small finite games; the agenda tests are one position each (directional)."""
import itertools
import random
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import agenda as AGM
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import Claim, state_of

LEAN = Path(__file__).resolve().parents[1] / "lean/ideas/Ideas/Forcing.lean"
J3YBL = "5rk1/5p1p/p1p5/1p3b1q/P2P1bp1/2P4P/1P1N1P2/R1BBrQK1 w - - 0 23"      # rand9_10 j3ybl: Qxe1 +2 is a schema claim of Q1 (no chase); the status is 'losing' (-2, their threat 4)


# ---------------------------------------------------------------------------------------------------------------- the Lean statements, mirrored

def reach(ours, theirs, goal, target, n: int, p) -> bool:
    """Python mirror of Lean `Ideas.Forced.Reach` over a finite game: within n of our moves, against every reply, the goal (on our move) or the target."""
    if target(p):
        return True
    if n == 0:
        return False
    for q in ours(p):
        if goal(q):
            return True
        rs = theirs(q)
        if rs and all(reach(ours, theirs, goal, target, n - 1, r) for r in rs):
            return True
    return False


def _game(rng: random.Random, size: int = 5):
    nodes = list(range(size))
    ours = {p: frozenset(q for q in nodes if rng.random() < 0.35) for p in nodes}
    theirs = {q: frozenset(r for r in nodes if rng.random() < 0.35) for q in nodes}
    goal = frozenset(q for q in nodes if rng.random() < 0.2)
    return nodes, ours, theirs, goal


def test_66_1_claim_mono_and_refutation_not_mono_are_the_lean_statements():
    """Lean `Ideas.Forcing.claim_mono` (a claim forced with route set R is forced with any R' >= R) and `refutation_not_mono` (a kill under R is not a kill under R') are in the file with no sorry, and
    hold on 300 random finite games, checked by the Python mirror of `Reach`: every proof under R survives every superset R'; some position refuted under R is proved under a superset."""
    src = LEAN.read_text()
    for name in ("theorem claim_mono", "theorem refutation_not_mono"):
        assert name in src
    assert "sorry" not in src
    rng = random.Random(66)
    widened = 0
    for _ in range(300):
        nodes, ours, theirs, goal = _game(rng)
        wider = {p: ours[p] | frozenset(q for q in nodes if rng.random() < 0.3) for p in nodes}
        for p, n in itertools.product(nodes, range(3)):
            narrow = reach(lambda x: ours[x], lambda x: theirs[x], lambda x: x in goal, lambda x: False, n, p)
            wide = reach(lambda x: wider[x], lambda x: theirs[x], lambda x: x in goal, lambda x: False, n, p)
            assert not narrow or wide                                   # claim_mono
            widened += (not narrow) and wide
    assert widened > 0                                                  # refutation_not_mono: refuted under R, proved under R'
    print(f"OBS 66.1 mirror: claim_mono held on 300 random games x 5 positions x 3 depths; {widened} refutations under R became proofs under a wider R'")


# ---------------------------------------------------------------------------------------------------------------- 66.1 the agenda

def _tier1(fen: str = J3YBL) -> Agenda:
    return Agenda(state_of(fen, "w" if chess.Board(fen).turn else "b"), seconds=15, tier=1)


def _q1(ag: Agenda):
    return next(q for q in ag.all if q.text.startswith("save our Q on f1"))


def _carry(ag: Agenda, q, net: int, dep: bool, balance: int, line: int) -> None:
    cl = Claim("save", "Qxe1", net, "delayed_balance_alive_d1", "Qxe1 (every reply: 38 replies)", move="f1e1")
    ag.ctx.carry_q = {ag._carry_key(q): dict(net=net, second=0, claim=cl, tier=0, dep=dep)}
    ag.ctx.carry_status = dict(balance=balance, threat=4, s0=balance - 4, word="losing", line=line)


def test_66_1_a_carried_claim_no_chase_made_is_served_and_not_asked_again():
    """A root question answered without the chase (a schema binding reads no vocabulary of a tier) is served its carried claim and not asked again; the best holds it; the trace says so."""
    ag = _tier1()
    q = _q1(ag)
    _carry(ag, q, 2, dep=False, balance=-2, line=1)
    assert ag._carry_question(q) is True
    assert ag.best is not None and ag.best[0] == 2 and ag.best[2].what == "Qxe1" and ag.best[2].proved
    line = ag.lines[-1]
    print("OBS 66.1 served:", line)
    assert "carried from tier 0: Qxe1 +2 (proved" in line and "not asked again" in line and q.status == "verified"
    assert ag.ctx.carried_q == 1 and ag.ctx.carried_q_kept == 1 and ag.ctx.carried_q_reasked == 0


def test_66_1_a_chased_claim_that_does_not_keep_the_status_is_re_asked_above_its_floor():
    """A carried claim that a chase made (the tier's vocabulary) and that does not keep the status is served AND re-asked: the question carries the floor; one that keeps the status is not re-asked."""
    ag = _tier1()
    q = _q1(ag)
    _carry(ag, q, 2, dep=True, balance=-2, line=1)                         # -2 + 2 = 0 < the raised line 1: not kept
    assert ag._carry_question(q) is False
    assert getattr(q, "_floor", None) == 2 and ag.best[0] == 2 and "re-asked for a better claim above the floor +2" in ag.lines[-1]
    ag2 = _tier1()
    q2 = _q1(ag2)
    _carry(ag2, q2, 2, dep=True, balance=-2, line=0)                       # -2 + 2 = 0 >= the line 0: kept
    assert ag2._carry_question(q2) is True and "it keeps the status: not asked again" in ag2.lines[-1]


def test_66_1_the_tier_hands_on_each_questions_own_best_proved_claim():
    """`_carried_out`: each verified question's own best PROVED claim (the one `_better` was offered for it), with whether a chase made it; an unproved claim never carries."""
    ag = _tier1()
    q = _q1(ag)
    cl = Claim("save", "Qxe1", 2, "delayed_balance_alive_d1", "line", move="f1e1", qid=q.id)
    ag._better(2, 0, cl)
    q.status = "verified"
    out = ag._carried_out()
    assert out[ag._carry_key(q)]["net"] == 2 and out[ag._carry_key(q)]["dep"] is False
    q._tier_dep = True
    assert ag._carried_out()[ag._carry_key(q)]["dep"] is True
    ag2 = _tier1()
    q2 = _q1(ag2)
    ag2._better(0, 0, Claim("win", "Qxe1", 0, "unfinished line", "x", move="f1e1", qid=q2.id, proved=False, word="unfinished line"))
    q2.status = "verified"
    assert ag2._carried_out() == {}
    assert AGM.CARRY is True                                                # ON by default (`analyse --no-carry` is the comparison condition)



# ---------------------------------------------------------------------------------------------------------------- 66.2 closed replies carry inside a route

LRTF = "2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 0 1"                                  # dev 0LRTF: a chase claim read in a fraction of a second under the count budget


def _route_low(gs, top=100):
    out = top
    for g in reversed(gs):
        out = min(g, out)
    return out


def test_66_2_route_low_mono_is_the_lean_statement():
    """Lean `Ideas.Forcing.routeLow_mono`: carried gains each at most the full read's give a route value at most the full read's (a carried proof is a floor). Mirrored on random lists."""
    src = LEAN.read_text()
    assert "def routeLow" in src and "theorem routeLow_mono" in src and "sorry" not in src
    rng = random.Random(662)
    for _ in range(500):
        ts = [rng.randint(-9, 100) for _ in range(rng.randint(0, 6))]
        gs = [t - rng.randint(0, 5) for t in ts]
        assert _route_low(gs) <= _route_low(ts)
        assert _route_low(ts) == (min(ts + [100]))


def _carry_ag(fen: str = LRTF) -> Agenda:
    return Agenda(state_of(fen, "w" if chess.Board(fen).turn else "b"), tier=1)


def test_66_2_a_carried_reply_is_served_only_when_it_meets_what_this_read_needs():
    """`_carried_reply`: served when the proof meets the agenda's target; read again below it, below a re-asked root question's floor + 1, or (a close by the exchange) below the goal's stake. `_proved_reply`
    keeps the larger proof of a reply; the refuter fix's replies are never filed."""
    ag = _carry_ag()
    key = (ag.pk, "h7h8")
    ag.ctx.carry_r = {(ag.pk, "h7h8", "c1h1"): dict(g=2, part="after Rh1: x -> 2", lemma=None, closed=True, tier=0)}
    assert ag._carried_reply(key, "c1h1") is not None and ag.ctx.carry_served == 1
    ag._floor = 2                                                           # a re-asked root question with the carried floor 2: the reply must beat it
    assert ag._carried_reply(key, "c1h1") is None and ag.ctx.carry_reread == 1
    ag._floor = None
    assert ag._carried_reply(key, "c1h1", gstake=5) is None                 # a close below the goal's stake would not have been made
    ag._proved_reply(key, "c1c2", 1, "p1")
    ag._proved_reply(key, "c1c2", 3, "p3")
    ag._proved_reply(key, "c1c2", 2, "p2")
    assert ag.ctx.proofs[(ag.pk, "h7h8", "c1c2")]["g"] == 3
    ag._proved_reply((ag.pk, "h7h8:refuter"), "c1c2", 9, "x")
    assert (ag.pk, "h7h8:refuter", "c1c2") not in ag.ctx.proofs


def test_66_2_a_route_read_again_with_its_proofs_carried_decides_the_same_and_opens_fewer_agendas():
    """0LRTF read once (tier 0, the count budget), then again on a fresh context carrying the replies tier 0 proved: the same decision and value, its replies served ('carried from tier 0'), fewer inner
    agendas opened. The chase claim that stands on carried replies is filed for this position only (their facts are the earlier tier's)."""
    ag0 = Agenda(state_of(LRTF, "w"))
    ag0.ask()
    d0 = ag0.decision()
    assert d0 is not None and ag0.ctx.proofs
    ag1 = _carry_ag()
    ag1.ctx.carry_r = dict(ag0.ctx.proofs)
    ag1.ask()
    d1 = ag1.decision()
    print(f"OBS 66.2 0LRTF: decision {d0.what} {d0.value} / carried {d1.what} {d1.value}; inner agendas {ag0.ctx.agendas} -> {ag1.ctx.agendas}; replies served {ag1.ctx.carry_served}, read again {ag1.ctx.carry_reread}")
    assert (d1.move, d1.value) == (d0.move, d0.value)
    assert ag1.ctx.carry_served > 0 and ag1.ctx.agendas < ag0.ctx.agendas
    assert any("[carried from tier 0]" in ln for ln in ag1.lines) or any("[carried from tier 0]" in c.line for c in ag1.state.claims)


# ---------------------------------------------------------------------------------------------------------------- 66.3 the stop of 65.2 with nothing re-read

def test_66_3_a_cut_routes_measured_seconds_is_no_bound_once_its_proofs_carry():
    """A route cut after t needed more than t only while a re-read started from nothing; with its proved replies carried the resumed read does not redo them, so the next tier prices it as a route not
    yet read (first-level cost x the rate). A finished read keeps its measured seconds."""
    secs = {("p", "a", 1): (2.0, False), ("p", "b", 1): (0.3, True)}
    out = Agenda._carried_secs(secs)
    assert out == {("p", "b", 1): (0.3, True)}
    assert Agenda._route_need(10, out.get(("p", "a", 1)), 0.01) == 10 * 0.01      # priced as never read, not max(cost x rate, 2.0)
    assert Agenda._route_need(10, secs[("p", "a", 1)], 0.01) == 2.0                # (without carry: more than t)


def test_66_3_the_open_line_says_what_was_left_when_the_widening_reached_its_end():
    ag = _tier1()
    assert ag._open_line(ag, 1.0).startswith("TIERS: ended at 1.0 s with no root route open")
    m = ag.board.parse_san("Qxe1")
    ag.ctx.root_gaps[(ag.pk, m.uci(), 5)] = (40, m.uci())
    ag.ctx.judge_t, ag.ctx.judge_n = 1.0, 100
    line = ag._open_line(ag, 2.0)
    assert "1 root route(s) still open" in line and "the cheapest, Qxe1, priced 40 x 0.0100 = 0.40 s" in line


# ---------------------------------------------------------------------------------------------------------------- 66.4 resume a chain-stopped line while it is still forcing

MH1Q8 = "3rrknQ/1pp2p2/p2b1q1p/3pNPp1/1P1P1BP1/P6P/8/R3R1K1 w - g6 0 29"      # rand6_10 mh1q8: Nd7+ checks and attacks the Q; Rxd7 is the only reply


def test_66_4_still_forcing_is_the_lean_definition():
    """Lean `Ideas.Forcing.stillForcing` (every step threatens and its reply parries), `stillForcing_append`, `quiet_breaks`, `stillForcing_parryList`: in the file, no sorry; the Python `_still_forcing` is the
    conjunction of `_step_forcing` over the steps (mirrored with stubbed steps), and a quiet step anywhere breaks it."""
    src = LEAN.read_text()
    for name in ("def stillForcing", "theorem stillForcing_append", "theorem quiet_breaks", "theorem stillForcing_parryList", "structure Step"):
        assert name in src
    assert "sorry" not in src
    ag = _carry_ag(MH1Q8)
    rng = random.Random(664)
    for _ in range(200):
        line = [("f", f"m{i}", f"r{i}", 1) for i in range(rng.randint(0, 5))]
        verdict = {st: (rng.random() < 0.8, "q") for st in line}
        ag.ctx.forcing_steps = dict(verdict)
        ok, _why = ag._still_forcing(line)
        assert ok == all(v[0] for v in verdict.values())               # stillForcing = List.all
        if line:
            ag.ctx.forcing_steps[line[0]] = (False, "quiet")
            assert ag._still_forcing(line)[0] is False                  # quiet_breaks


def test_66_4_a_check_is_parried_by_every_reply_and_a_quiet_move_names_no_threat():
    ag = _carry_ag(MH1Q8)
    b = ag.board
    nd7, rxd7 = b.parse_san("Nd7+"), "d8d7"
    assert ag._step_forcing((b.fen(), nd7.uci(), rxd7, 9)) == (True, "")
    a3 = b.parse_san("Kg2")
    c = b.copy()
    c.push(a3)
    r = next(iter(c.legal_moves)).uci()
    ok, why = ag._step_forcing((b.fen(), a3.uci(), r, 9))
    assert not ok and why == "Kg2 names no threat"


def test_66_4_the_extension_applies_only_while_the_line_is_still_forcing():
    ag = _carry_ag(MH1Q8)
    b = ag.board
    forcing = [(b.fen(), b.parse_san("Nd7+").uci(), "d8d7", 9)]
    ag.ctx.extend = 1
    ag.ctx.line = forcing
    assert ag._extension() == 1 and ag.ctx.ext_used == 1
    ag.ctx.line = forcing + [(b.fen(), b.parse_san("Kg2").uci(), "d8d7", 9)]
    assert ag._extension() == 0 and ag.ctx.ext_quiet == 1
    ag.ctx.extend = 0
    ag.ctx.line = forcing
    assert ag._extension() == 0


def test_66_4_a_chain_stop_memo_serves_only_its_own_chain_state():
    """mh1q8: the mate goal's chain stop of Nd7+ (allowance 1 under the 100-stake goal) was served to the queen goal ('seen before with no more fuse: not re-run'). A chain-stop GAP keeps the chain state it
    was cut under; under another one the route is read again."""
    ag = _carry_ag(MH1Q8)
    m = ag.board.parse_san("Nd7+")
    ag.ctx.mate_levels.append((1, 100))
    st100 = ag._chain_state()
    ag.ctx.mate_levels[-1] = (1, 9)
    assert ag._chain_state() != st100
    key = (ag.pk, m.uci())
    ag.ctx.routes[key] = dict(value=0, killed=None, gap="chain stop: Nd7+ not followed past Rxd7 (tempo arithmetic)", parts=[], lemmas=[], cut_left=10 ** 9, cut_slice=10 ** 9, chain_state=st100)
    ag.ctx.left = 0                                                     # (no inner agenda: the read stops at its first open reply, as a fuse GAP)
    ag._route_move = m
    res = ag._chase_route_body(m)
    assert not res.get("reused")                                        # read again (not 'seen before')
    ag.ctx.routes[key] = dict(ag.ctx.routes.get(key) or {}, gap="chain stop: x", chain_state=ag._chain_state(), cut_left=10 ** 9, cut_slice=10 ** 9, killed=None)
    assert ag._chase_route_body(m).get("reused")                        # the same chain state: the GAP stands


def test_66_4_the_next_tier_resumes_a_forcing_chain_stop_and_says_why_a_quiet_one_is_not():
    ag = _tier1(MH1Q8)
    b = ag.board
    q = next(q for q in ag.all if q.goal == "win")
    nd7 = b.parse_san("Nd7+")
    from catspace.incubator.analyst.discovery import regress as R
    forcing = [(b.fen(), nd7.uci(), "d8d7", q.stake)]
    quiet = [(b.fen(), b.parse_san("Kg2").uci(), "d8d7", q.stake)]
    ag.ctx.chain_gaps = [dict(key=ag._carry_key(q), route=R.Route(nd7, "Nd7+", "check"), uci=nd7.uci(), stake=q.stake, line=forcing, gap="chain stop: a", mkey=None),
                         dict(key=ag._carry_key(q), route=R.Route(b.parse_san("Kg2"), "Kg2", "x"), uci="g1g2", stake=q.stake, line=quiet, gap="chain stop: b", mkey=None)]
    ag.status_info = dict(balance=-1, threat=7, s0=-8, word="losing", line=0)
    nxt = ag._next_agenda(ag, 2, 5.0, ag.status_info, 1)
    assert nxt.ctx.resume_ext == {(ag._carry_key(q), nd7.uci()): dict(ext=1, moves=1)}
    assert any(ln.startswith("RESUME: not resumed Kg2: the line went quiet (move 1: Kg2 names no threat") for ln in ag.lines)
    assert AGM.RESUME is True


# ---------------------------------------------------------------------------------------------------------------- 66.5 the irreversible-move clock

ROR8T = "1R6/8/8/5p2/5rk1/1P5p/P7/6K1 b - - 1 50"                                # rand23_5 RoR8t: Kg3 Rg8+ Rg4 Rxg4+ (a CAPTURING check) Kxg4


def test_66_5_a_capture_by_either_side_not_bad_for_us_or_a_pawn_move_of_ours_resets_the_clock():
    ag = _carry_ag(ROR8T)
    b = ag.board
    ag.since_capture = 3
    ag._route_move = b.parse_san("Kg3")
    c = b.copy()
    c.push(ag._route_move)
    rg8 = c.parse_san("Rg8+").uci()
    a3 = c.parse_san("a3").uci()
    assert ag._since_after(rg8, 0) == 4 and not ag._resets(rg8)          # a quiet check: one more move of ours
    assert ag._since_after(a3, 0) == 4 and not ag._resets(a3)            # their pawn move is a reply like any other (the plan: a capture, or a pawn move OF OURS)
    ag._route_move = b.parse_san("h2+")                                     # OUR pawn move
    c2 = b.copy()
    c2.push(ag._route_move)
    kxh2 = c2.parse_san("Kxh2").uci()
    assert ag._since_after(kxh2, 0) == 0 and ag._resets(kxh2)
    ag._route_move = b.parse_san("Rf1+")                                    # our check, THEIR capture
    c3 = b.copy()
    c3.push(ag._route_move)
    kxf1 = c3.parse_san("Kxf1").uci()
    assert ag._since_after(kxf1, 0) == 0 and ag._resets(kxf1)              # a capture of theirs that is not bad for us (dbal >= 0): the clock starts again
    assert ag._since_after(kxf1, -5) == 4 and ag._resets(kxf1)             # one that wins material is their progress, not ours; the checks in a row still start again (no perpetual captures)
    ag._route_move = b.parse_san("Kg3")
    try:
        AGM.CLOCK_IRREVERSIBLE = False
        ag._route_move = b.parse_san("Rf1+")
        assert ag._since_after(kxf1, -5) == 4 and not ag._resets(kxf1)   # (the comparison condition: only our capture not bad for us / our pawn move)
    finally:
        AGM.CLOCK_IRREVERSIBLE = True
    assert AGM.CLOCK_IRREVERSIBLE is True and AGM.CHASE_CLOCK == 4      # the length stays a module parameter


def _stop_lines(fen: str) -> list:
    stops = []
    orig = AGM.Agenda._chain_gap

    def spy(self, q, why):
        if "second check" in why:
            steps = []
            for f, m, r, _s in self.ctx.line:
                bb = chess.Board(f)
                mv = chess.Move.from_uci(m)
                sm = bb.san(mv)
                bb.push(mv)
                steps.append((sm, bb.san(chess.Move.from_uci(r)), bb.is_capture(chess.Move.from_uci(r))))
            stops.append(steps)
        return orig(self, q, why)
    AGM.Agenda._chain_gap = spy
    try:
        b = chess.Board(fen)
        Agenda(state_of(fen, "w" if b.turn else "b")).ask()
    finally:
        AGM.Agenda._chain_gap = orig
    return stops


def test_66_5_a_capturing_check_is_not_a_second_check_in_a_row():
    """RoR8t after Kg3 Rg8+: the interposition Rg4 is answered by Rxg4+, a CAPTURING check. The stop for a second check of theirs exists for perpetual-check loops, which capture nothing: with the
    irreversible clock no stop is made right after Rxg4+ (Kxg4 is read); in the comparison condition it is."""
    b = chess.Board(ROR8T)
    for s in ("Kg3", "Rg8+"):
        b.push_san(s)
    on = _stop_lines(b.fen())
    assert on and not any(st[-1][1] == "Rxg4+" for st in on)
    assert all(not st[-1][2] for st in on)                              # every second-check stop follows a non-capturing check
    try:
        AGM.CLOCK_IRREVERSIBLE = False
        off = _stop_lines(b.fen())
    finally:
        AGM.CLOCK_IRREVERSIBLE = True
    assert any(st[-1][1] == "Rxg4+" for st in off)
    print(f"OBS 66.5 RoR8t after Kg3 Rg8+: second-check stops {len(on)} (none after a capturing check) against {len(off)} in the comparison condition")
