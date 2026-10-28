"""PLAN 75.2 THE MOVE (docs/ANALYST_PLAN.md 75, Kaveh 2026-10-27: "the goal is actually very generic: first of all, don't lose, and then win ... some chess vocab has to be
provided by us ... everything else should be derived from the basic rules"). (a) THE GOAL LADDER AND KINDS FROM THE STORE: the GOAL entries (mate, promotion, win, save) and
the material VALUATION load; the stake of mate is read from the store at run time (a patched store moves the agenda's stake); material has one definition (Game.measure,
BaseWords.worth, selfmine's table and the Rust kernel's balance agree with the entry); the outcome ladder (don't lose, then win) is the worklist's generic arithmetic and the
agenda's status reads it. Count budgets only."""
import copy
import json

import chess
import pytest

import analyst_core as AC
from catspace.incubator.analyst.discovery import game as GM
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery import binder as BD
from catspace.incubator.analyst.discovery import selfmine as SM
from catspace.incubator.analyst.discovery import words as WD
from catspace.incubator.analyst.discovery import worklist as WL
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import State

RA8 = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"                 # Ra8# : a mate root on a8
PROMO = "8/4P3/8/8/8/8/k7/6K1 w - - 0 1"                  # e8=Q : a promote root one push away
POSITIONS = (RA8, PROMO, "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "8/8/8/K2pP2r/8/8/8/7k w - d6 0 2",
             "2r2rk1/R5pp/5p2/2q1p3/3pQ3/8/2P2PPP/R5K1 w - - 4 30")


def _entry(name: str) -> dict:
    return json.loads((LS.DIR / (LS.slug(name) + ".json")).read_text())


def _agenda(fen: str) -> Agenda:
    return Agenda(State(fen=fen, us="w" if chess.Board(fen).turn else "b", pieces=[]))


# ---------------------------------------------------------------------------------------------------------------- (a) the goal kinds and the valuation

def test_75a_goal_entries_load_with_their_roles_and_the_schema():
    store = LS.load(force=True)
    gk = LS.goal_kinds()
    print("OBS 75a goal kinds: " + ", ".join(f"{g.name} ({g.question}: {g.basis}, stake {g.stake})" for g in gk.by_question.values()) + f"; valuation {gk.valuation.name} {gk.valuation.worth}, unit {gk.valuation.unit}")
    assert {n for n, e in store.items() if e["kind"] == "GOAL"} >= {"mate", "promotion", "win", "save"}
    assert store["material"]["kind"] == "VALUATION"
    assert (gk.terminal.name, gk.promotion.name, gk.gain.name, gk.keep.name) == ("mate", "promotion", "win", "save")
    assert gk.terminal.basis == gk.promotion.basis == "RULE" and gk.gain.basis == gk.keep.basis == "VALUATION"
    assert gk.terminal.rule == "Game.terminal" and gk.promotion.rule == "Game.promotes"
    assert gk.terminal.ends_game and not gk.terminal.keep and gk.keep.keep and not gk.gain.keep
    assert gk.terminal.stake == _entry("mate")["params"]["stake"]                       # the stake is the entry's, read
    v = _entry("material")["params"]["worth"]
    assert gk.promotion.stake == v["q"] - v["p"] and gk.promotion.places == "q"          # the promotion's stake: the valuation's queen less its pawn
    assert gk.of("fork") is LS.NO_GOAL and not LS.NO_GOAL.ends_game and not LS.NO_GOAL.keep
    for n in ("mate", "promotion", "win", "save", "material"):
        assert LS.check_entry(store[n]) == [], n
    bad = copy.deepcopy(store["mate"])
    del bad["params"]["stake"]
    assert LS.check_entry(bad)                                                          # a GOAL without its stake breaks the schema
    bad = copy.deepcopy(store["material"])
    bad["params"]["royal"] = "x"
    assert LS.check_entry(bad)
    two = dict(store)
    two["mate twice"] = dict(copy.deepcopy(store["mate"]), name="mate twice")
    two["mate twice"]["params"]["question"] = "mate again"
    with pytest.raises(LS.LemmaStoreError):
        LS.GoalKinds(two)                                                               # exactly one goal ends the game


def test_75a_the_stake_of_mate_is_read_from_the_store(monkeypatch):
    """The agenda reads the stake at run time: a patched store (mate's stake 123, the queen worth 7) moves the agenda's terminal stake, its mate roots and its promotion stake."""
    ag = _agenda(RA8)
    mates = [q for q in ag.all if q.goal == "mate"]
    assert mates and all(q.stake == LS.goal_kinds().terminal.stake for q in mates) and ag.end_stake == R.STAKE_MATE == BD.MATE_VALUE == GM.MATE_VALUE
    patched = copy.deepcopy(LS.load())
    patched["mate"]["params"]["stake"] = 123
    patched["material"]["params"]["worth"]["q"] = 7
    monkeypatch.setattr(LS, "_STORE", [patched])                                       # (the loaded store: every read goes through it)
    ag2 = _agenda(RA8)
    assert ag2.end_stake == 123 and all(q.stake == 123 for q in ag2.all if q.goal == "mate")
    ag3 = _agenda(PROMO)
    assert ag3.goals.promotion.stake == 6 and [q.stake for q in ag3.all if q.goal == "promote"] == [6]
    print(f"OBS 75a patched store: the terminal stake {ag2.end_stake}, the promotion stake {ag3.goals.promotion.stake} (queen 7 less pawn 1)")


def test_75a_material_has_one_definition():
    v = LS.goal_kinds().valuation
    assert SM.PIECE_WORTH == v.worth == G.WORTH
    base = WD.CHESS_WORDS.base
    for sym in "pnbrqk":
        for colour in (chess.WHITE, chess.BLACK):
            pc = chess.Piece.from_symbol(sym if colour == chess.BLACK else sym.upper())
            assert base.worth(pc) == v.worth_of(pc) == G.worth(pc) == _entry("material")["params"]["worth"][sym]
    for fen in POSITIONS:
        b = chess.Board(fen)
        for side in (chess.WHITE, chess.BLACK):
            own = sum(v.worth_of(p) * (1 if p.color == side else -1) for p in b.piece_map().values())
            assert GM.CHESS.measure(b, side) == own
            assert AC.Pos(b.fen()).balance(side) - 64 == own                                 # the Rust kernel's copy of the worths agrees (a FINDING: rules.rs holds its own table)


def test_75a_the_outcome_ladder_is_the_worklists_and_the_status_reads_it():
    assert WL.ladder(0, 0, 1) == dict(balance=0, threat=0, s0=0, word=WL.LEVEL, line=0)
    assert WL.ladder(9, 0, 1)["word"] == WL.WINNING and WL.ladder(9, 0, 1)["line"] == 1
    assert WL.ladder(-1, 9, 1)["word"] == WL.LOSING and WL.ladder(-1, 9, 1)["s0"] == -10
    st = WL.ladder(-9, 0, 1)
    assert not WL.kept(st, None) and not WL.kept(st, 8) and WL.kept(st, 9) and WL.Worklist.need(st) == 9
    assert WL.raises(st, 9, 1) and not WL.raises(WL.ladder(9, 0, 1), -8, 1)              # level reached, not yet ahead: the line may rise
    up = WL.raised(WL.ladder(0, 0, 1), 1)
    assert up["line"] == 1 and up["raised"] and WL.line_name(up) == "stay ahead" and WL.line_name(WL.ladder(0, 0, 1)) == "reach level"
    assert WL.rung(-1, 1) == WL.LOSING and WL.rung(0, 1) == WL.LEVEL and WL.rung(1, 1) == WL.WINNING
    ag = _agenda("4k3/8/8/8/8/8/8/3qK3 w - - 0 1")
    st = ag.status()
    assert {k: st[k] for k in ("balance", "threat", "s0", "word", "line")} == WL.ladder(st["balance"], st["threat"], ag.unit)


# ---------------------------------------------------------------------------------------------------------------- (b) the stops

STOPS = {"root": ("budget",), "chase": ("chase fuse", "the chase clock", "a second check of theirs in a row", "tempo arithmetic", "the walk of checks", "the chase's follow list"),
         "judge": ("cut by the judge",)}


def _patched(monkeypatch, name: str, **kw) -> None:
    store = copy.deepcopy(LS.load())
    store[name]["params"].update(kw)
    monkeypatch.setattr(LS, "_STORE", [store])


def test_75b_the_stops_are_store_entries_with_their_shapes():
    store = LS.load(force=True)
    for ctx, names in STOPS.items():
        got = LS.stops_for(ctx)
        for n in names:
            assert store[n]["kind"] == "STOP" and store[n]["status"] == "HEURISTIC" and n in got, n
    print("OBS 75b stops: " + "; ".join(f"{n} {dict((k, v) for k, v in store[n]['params'].items() if k != 'context')}" for names in STOPS.values() for n in names))
    bad = copy.deepcopy(store["budget"])
    bad["params"]["context"] = "nowhere"
    assert LS.check_entry(bad)                                                          # a stop says where it applies


def test_75b_a_stops_parameter_changed_moves_the_stop(monkeypatch):
    """The agenda reads the stops' shapes at run time: the question budget per unit, the chase clock's cap from its tier, the follow list's power below the root."""
    ag = _agenda(RA8)
    units, caps = 2, sum(1 for m in ag.board.legal_moves if ag.board.is_capture(m))
    assert ag.budget == units * 8 + caps
    deep = Agenda(State(fen=RA8, us="w", pieces=[]), tier=3)
    assert deep._clock() == min(4, 4)
    win = next(q for q in ag.all if q.goal == "win")
    ag.chase_depth = 2
    assert not ag._can_chase(win)                                                       # a win question's power stops below the first level
    _patched(monkeypatch, "budget", per_unit=3)
    assert _agenda(RA8).budget == units * 3 + caps
    _patched(monkeypatch, "the chase clock", at_most=2)
    assert Agenda(State(fen=RA8, us="w", pieces=[]), tier=3)._clock() == 2
    _patched(monkeypatch, "the chase's follow list", power_depth={"promote": 1})
    ag2 = _agenda(RA8)
    ag2.chase_depth = 2
    assert ag2._can_chase(next(q for q in ag2.all if q.goal == "win"))
    print(f"OBS 75b patched stops: budget {ag.budget} -> {units * 3 + caps} (per unit 8 -> 3), the tier-3 clock 4 -> 2, a win's power at depth 2 off -> on")
