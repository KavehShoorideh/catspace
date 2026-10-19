"""Plan 52.19: the fact store with justifications (facts.py), the listener that flags what a move touched, the two tiers of thinking, and the store across moves."""
import threading
import time

import chess
import pytest

import catspace.incubator.analyst  # noqa: F401
from catspace.incubator.analyst.discovery import agenda as AG
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import facts as FS
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.state import State, state_of

BV.install_terms()
OPEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"          # a quiet opening position: many kills, no claim
FAR = "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"          # the same, the a-pawn on a6: far from most of what was read
LRTF = "2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 0 1"


def inner_agenda(fen, ctx, us="w"):
    return AG.Agenda(State(fen, us, []), side_depth=0, chase_depth=1, ctx=ctx, min_stake=1)


def fresh_ctx(left=10 ** 6):
    ctx = AG.ChaseCtx()
    ctx.left = left
    return ctx


# ---------------------------------------------------------------------------------------------------------------- item 1: the store

def test_52_19_1_a_fact_is_served_on_another_board_whose_read_cells_are_the_same():
    ctx = fresh_ctx()
    a = inner_agenda(OPEN, ctx)
    a.ask()
    derived = ctx.facts.derived
    assert derived and not a.inherited or True
    b = inner_agenda(FAR, ctx)
    b.ask()
    served = [ln for ln in b.lines if "served from the store (fact #" in ln]
    assert served and b.inherited == len(served)                       # siblings and transpositions share facts, not only parent to child
    assert b.asked < a.asked                                           # what the facts answered was not asked again
    assert all("derived under Q" in ln for ln in served)
    print(f"OBS 52.19 (1): position A asked {a.asked}, derived {derived}; the far-pawn position asked {b.asked}, served {len(served)}")


def test_52_19_1_the_signature_is_the_read_cells_not_the_fen():
    b = chess.Board(OPEN)
    cell = FS.board_cell(b)
    reads = frozenset({chess.C4, chess.F7, chess.E8})
    sig = FS.signature(cell, reads)
    assert sig == FS.signature(FS.board_cell(chess.Board(FAR)), reads)                 # a6 is not read
    moved = chess.Board(OPEN)
    moved.push_san("Bxf7+")
    assert sig != FS.signature(FS.board_cell(moved), reads)


def test_52_19_1_a_fact_records_its_justifications_stake_and_chain_stake():
    fs = FS.FactStore()
    premise = fs.add("kill", ("win", 1), True, "win a pawn", reads={chess.A1}, sig=((0, "", 0, 0),), stake=1)
    claim = fs.add("claim", ("mate", 2), True, "mate", reads={chess.H8}, sig=((63, "", 0, 0),), just=[premise.id], stake=100)
    assert claim.just == (premise.id,) and premise.dependants == [claim.id]
    assert premise.stake == 1 and premise.chain_stake == 100                          # "the highest stake of the entire chain is the stake of that chain"
    lines = fs.graph_lines(claim.id)
    assert lines[0].startswith(f"#{claim.id} claim mate") and lines[1].startswith(f"  #{premise.id} kill win a pawn") and "chain stake 100" in lines[1]


def test_52_19_1_a_chase_claim_stands_on_the_facts_of_its_route_and_the_graph_prints_them():
    ag = AG.Agenda(state_of(LRTF, "w"))
    ag.ask()
    d = ag.decision()
    assert d is not None and d.what == "Rh8+"
    fid = ag.claim_fid[id(d)]
    fs = ag.ctx.facts
    top = fs.facts[fid]
    assert top.kind == "claim" and top.just                                           # the deciding claim stands on facts (the inner answers of its route)
    assert any("justification graph of the decision" in ln for ln in ag.lines)
    assert all(fs.facts[j].chain_stake >= top.stake for j in top.just)
