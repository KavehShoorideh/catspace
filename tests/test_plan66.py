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
