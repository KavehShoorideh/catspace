"""PLAN 61 stage A: the forced chain's words -- the threat as an object (A1), parries (A2), the delayed threat (A3), no safe move (A4). Each Python word is mirrored against the
Lean definition's cases (`Ideas.Forcing`: `closed_binds`, `parry_or_closed`, `delayed_forcing`, `noSafeMove_loses`, `free_move_breaks`), here as brute-force restatements over the
legal moves; the flags default OFF and ON leaves the decisions of the named puzzles alone."""
import chess
import pytest

import catspace.incubator.analyst  # noqa: F401
from catspace.incubator.analyst.discovery import agenda as AG
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.state import state_of

DOFB = "1r2k3/1bp2p2/p3p2Q/q5B1/3b4/P7/5PPP/2RR2K1 b - - 4 29"             # Qd5 (a5d5) is a threat of mate
MATE2 = "4R3/8/7Q/8/6K1/k7/8/8 w - - 0 1"                                   # Rh8 (e8h8): a threat of mate in two
DISCOVERY = "r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1"   # Nf6: threat 5 survives Bxf6
ZUG = "2K5/6N1/1k3n1Q/8/8/2p5/8/8 w - - 0 1"                                # Ne8 (g7e8): the pinned Nf6 cannot move, every black move leaves Qxf6 / Nxf6
DELAY = "3KR3/1k6/1N6/8/8/8/7n/8 w - - 0 1"                                 # Rg8 (e8g8): after every reply a new threat (Rh8 hits the knight on h2)


@pytest.fixture(autouse=True)
def _terms():
    BV.install_terms()


def _threat_after(fen, uci, stake, us, **kw):
    b = chess.Board(fen)
    m = chess.Move.from_uci(uci)
    why = R.forcing_why(b, m, stake, us, mate_threats=True, **kw)
    return b, m, why, R.threat_of(b, m, why, us)


# ---- A1 --------------------------------------------------------------------------------------------------------------------------------------------

def test_a1_the_threat_object_names_the_target_the_reason_names():
    b, m, why, th = _threat_after(DOFB, "a5d5", 100, "b")
    assert why == "threat of mate" and (th.kind, th.target, th.value) == ("mate", b.king(True), 100)
    b, m, why, th = _threat_after(MATE2, "e8h8", 100, "w", mate2=True)
    assert why == "threat of mate in two" and th.kind == "mate2" and th.target == chess.A3 and th.word == "mate_in_two"
    b, m, why, th = _threat_after(DISCOVERY, "e4f6", 5, "w")
    assert why.startswith("threat 5 survives") and th.kind == "threat" and th.value == 5
    assert G.see(G.with_turn(_after(DISCOVERY, "e4f6"), True), th.target) >= 5                    # the target is a piece of theirs the exchange wins that much on, after the move
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "0Nv29"][0]
    b, m, why, th = _threat_after(p["fen"], "b5b6", 8, "w")
    assert th.kind == "pass-claim" and chess.square_name(th.target) == "b8" and th.word == "unstoppablePasser2_sound"


def _after(fen, uci):
    b = chess.Board(fen)
    b.push(chess.Move.from_uci(uci))
    return b


def test_a1_every_forcing_reason_gives_an_object_with_a_target():
    seen = set()
    for name in ("dev_20", "fresh_60_threats_event"):
        for p in PA.load_set(name)[:25]:
            b = chess.Board(p["fen"])
            us = b.turn
            for stake in (1, 3, 5, 9, 100):
                for m in list(b.legal_moves)[:40]:
                    why = R.forcing_why(b, m, stake, us, mate_threats=True, mate2=True)
                    if why is None:
                        continue
                    th = R.threat_of(b, m, why, us)
                    assert th is not None and th.target is not None, why
                    seen.add(th.kind)
                    c = b.copy()
                    c.push(m)
                    if th.kind in ("check", "mate", "mate2"):
                        assert th.target == c.king(not us)
                    elif th.kind == "capture":
                        assert th.target == m.to_square
    assert {"check", "mate", "capture", "threat"} <= seen


# ---- A2: Lean `Ideas.Forcing` restated over the legal moves ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("fen,uci,stake,us,kw", [(DOFB, "a5d5", 100, "b", {}), (MATE2, "e8h8", 100, "w", {"mate2": True}), (DISCOVERY, "e4f6", 5, "w", {}),
                                                  (DISCOVERY, "e4d6", 5, "w", {})])
def test_a2_parries_and_closed_split_the_legal_replies_and_a_closed_reply_leaves_the_threat_bound(fen, uci, stake, us, kw):
    b, m, why, th = _threat_after(fen, uci, stake, us, **kw)
    after = b.copy()
    after.push(m)
    par, closed = R.parries(after, th, stake, us)
    legal = list(after.legal_moves)
    assert sorted([x.uci() for x in par] + [r.uci() for r, _ in closed]) == sorted(x.uci() for x in legal)      # parry_or_closed: a split of the legal replies
    me = G.colour_of(us)
    for r, v in closed:                                                          # closed_binds: the threat binds on the reply's own board, and the reply is no counter-threat
        d = after.copy()
        d.push(r)
        assert not d.is_check() and v >= (1 if th.kind in ("mate", "mate2") else stake) and R._binds(d, th, stake, me) == v
        assert not R._counter(after, r, stake, after.turn)
    for r in par:                                                                # the converse: a parry either ends the threat or is itself forcing against us
        d = after.copy()
        d.push(r)
        assert d.is_check() or R._binds(d, th, stake, me) == 0 or R._counter(after, r, stake, after.turn)


def test_a2_a_threat_of_mate_is_parried_by_the_replies_that_stop_the_mate_and_closed_for_the_rest():
    b, m, why, th = _threat_after(DOFB, "a5d5", 100, "b")
    after = b.copy()
    after.push(m)
    par, closed = R.parries(after, th, 100, "b")
    assert par and closed                                                        # some replies stop it (the Lean `parryList`), some do not (`closedList`)
    for r, v in closed:
        d = after.copy()
        d.push(r)
        mates = []
        for x in list(d.legal_moves):
            d.push(x)
            mates.append(d.is_checkmate())
            d.pop()
        assert v == 100 and any(mates)                                           # the mate in one stands on the reply's own board


def test_a2_the_flag_only_relabels_closed_replies_and_defaults_off():
    assert AG.PARRIES is False and AG.DELAYED_THREAT is True and AG.NO_SAFE_MOVE is False         # (63.3: the delayed threat is ON)
    from catspace.incubator.analyst.discovery import analyse as AN
    assert {"PARRIES", "DELAYED_THREAT", "NO_SAFE_MOVE"} <= set(AN._FLAGS)


# ---- A3 --------------------------------------------------------------------------------------------------------------------------------------------

def test_a3_a_delayed_threat_has_a_new_forcing_move_after_every_reply():
    b = chess.Board(DELAY)
    m = chess.Move.from_uci("e8g8")
    assert R.forcing_why(b, m, 3, "w", mate_threats=True) is None                 # quiet today
    why = R.forcing_why(b, m, 3, "w", mate_threats=True, delayed=True)
    assert why.startswith("delayed threat: after every reply a new threat of ours (tightest, after ")
    c = b.copy()
    c.push(m)
    for r in c.legal_moves:                                                      # Lean `delayed_forcing`: for every legal reply there is a forcing move of ours
        d = c.copy()
        d.push(r)
        assert any(R.forcing_why(d, x, 3, "w", mate_threats=False) for x in d.legal_moves)
    assert R.threat_of(b, m, why, "w").kind == "delayed"


def test_a3_a_move_with_one_reply_that_leaves_nothing_is_not_delayed_and_the_flag_is_off_by_default():
    b = chess.Board(DELAY)
    quiet = [m for m in b.legal_moves if R.forcing_why(b, m, 3, "w", mate_threats=True) is None]
    assert any(R.delayed_threat(b, m, 3, "w") is None for m in quiet)
    for m in quiet:                                                              # the plain call (the chase's default) never answers with a delayed reason
        assert R.forcing_why(b, m, 3, "w", mate_threats=True) is None


# ---- A4 --------------------------------------------------------------------------------------------------------------------------------------------

def test_a4_no_safe_move_every_reply_loses_the_stake():
    b = chess.Board(ZUG)
    m = chess.Move.from_uci("g7e8")
    assert R.forcing_why(b, m, 3, "w", mate_threats=True) is None                 # quiet by today's words (the knight hangs, the threat does not survive)
    why = R.forcing_why(b, m, 3, "w", mate_threats=True, nosafe=True)
    assert why.startswith("no safe move: every reply loses >= 3 (Ne8: cheapest ") and "wins 3" in why
    c = b.copy()
    c.push(m)
    assert list(c.legal_moves)                                                    # Lean `noSafeMove_loses`: there is a reply, and each one loses
    for r in c.legal_moves:
        d = c.copy()
        d.push(r)
        assert not d.is_check() and R._threat_square(G.with_turn(d, True), True)[0] >= 3
    assert R.threat_of(b, m, why, "w").kind == "zugzwang"


def test_a4_a_free_tempo_reply_makes_the_move_not_forcing_and_a_bigger_stake_is_not_met():
    b = chess.Board(ZUG)
    m = chess.Move.from_uci("g7e8")
    assert R.no_safe_move(b, m, 5, "w") is None                                   # Lean `free_move_breaks`: the replies lose 3, not 5
    free = chess.Board("2K5/6N1/1k3n1Q/8/8/8/8/7p w - - 0 1")                     # a pawn of theirs that can move for free: (h1 promotes -- a reply that is not a loss)
    free.turn = True
    assert R.no_safe_move(free, chess.Move.from_uci("g7e8"), 3, "w") is None
    assert R.no_safe_move(b, m, R.STAKE_MATE, "w") is None                       # a mate stake is read by checks and mate threats only


# ---- the chase with the flags ON leaves a plain puzzle's decision alone ---------------------------------------------------------------------------

def test_the_flags_on_change_no_decision_on_dofb_under_the_count_budget():
    saved = (AG.PARRIES, AG.DELAYED_THREAT, AG.NO_SAFE_MOVE)
    try:
        out = []
        for on in (False, True):
            AG.PARRIES = AG.DELAYED_THREAT = AG.NO_SAFE_MOVE = on
            ag = AG.Agenda(state_of(DOFB, "b"))
            ag.ask()
            d = ag.decision()
            out.append((d.move, d.value) if d else None)
        assert out[0] == out[1]
    finally:
        AG.PARRIES, AG.DELAYED_THREAT, AG.NO_SAFE_MOVE = saved
