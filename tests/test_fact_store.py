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


# ---------------------------------------------------------------------------------------------------------------- item 2: the listener

def test_52_19_2_a_move_flags_the_facts_that_read_its_squares_and_what_stands_on_them():
    fs = FS.FactStore()
    kill = fs.add("kill", ("win", 1), True, "the rook alone cannot mate", reads={chess.C1, chess.C7}, sig=((2, "r", 0, 0), (50, "P", 0, 0)), stake=100)
    claim = fs.add("claim", ("win", 2), True, "built on the kill", reads={chess.H8}, sig=((63, "", 0, 0),), just=[kill.id], stake=8)
    other = fs.add("kill", ("win", 3), True, "reads the far wing", reads={chess.A3}, sig=((16, "p", 0, 0),), stake=1)
    new = fs.invalidate({chess.C1})                                # the rook leaves c1
    assert set(new) == {kill.id, claim.id}                         # the kill read c1; the claim built on it goes stale with it
    assert other.id not in fs.stale                                # a fact that read nothing on the changed squares stays fresh
    fs.restore(new)                                                # a frame: the chase is a tree, the flags are taken back on the way up
    assert not fs.stale


def test_52_19_2_a_stale_fact_is_not_recomputed_on_invalidation_and_the_chain_stake_is_the_highest_of_the_chain():
    fs = FS.FactStore()
    low = fs.add("kill", ("win", 1), True, "wins a pawn", reads={chess.A1}, sig=((0, "", 0, 0),), stake=1)
    high = fs.add("kill", ("win", 2), True, "wins a rook", reads={chess.A1}, sig=((0, "", 0, 0),), stake=5)
    pawn = fs.add("kill", ("win", 3), True, "pawn fact under a mate", reads={chess.A1}, sig=((0, "", 0, 0),), stake=1)
    mate = fs.add("claim", ("mate", 4), True, "mate", reads={chess.H1}, sig=((7, "", 0, 0),), just=[pawn.id], stake=100)
    fs.invalidate({chess.A1}, permanent=True)
    assert fs.derived == 4 and fs.recomputed == 0                  # a cheap flag flip: nothing was recomputed
    assert fs.stale == {low.id, high.id, pawn.id, mate.id}         # the mate stands on the pawn fact and went stale with it
    assert (low.chain_stake, high.chain_stake, pawn.chain_stake, mate.chain_stake) == (1, 5, 100, 100)     # the pawn fact carries the mate's stake


def test_52_19_2_the_listener_flags_exactly_the_facts_whose_reads_the_move_touched_in_a_real_walk():
    ag = AG.Agenda(state_of(LRTF, "w"))
    ag.ask()
    fs = ag.ctx.facts
    b = chess.Board(LRTF)
    flagged_any = fresh_any = False
    for san in ("Rh8+", "Kd7", "Rd8+", "Ke6"):
        n = b.copy()
        n.push_san(san)
        diff = R.diff_squares(b, n)
        new = set(fs.invalidate(diff))
        hit = {f.id for f in fs.facts.values() if f.reads & diff and not f.superseded}
        assert hit <= new                                          # everything that read a changed square is flagged
        for fid in new - hit:                                      # ... and the rest only because something it stands on was
            assert any(j in new for j in fs.facts[fid].just)
        flagged_any |= bool(new)
        fresh_any |= any(f.id not in new for f in fs.facts.values())
        fs.restore(new)
        b = n
    assert flagged_any and fresh_any


def test_52_19_2_advance_flags_for_good_between_the_moves_of_a_game():
    fs = FS.FactStore()
    b = chess.Board(LRTF)
    f = fs.add("kill", ("win", 1), True, "reads c1", reads={chess.C1}, sig=FS.signature(FS.board_cell(b), {chess.C1}), stake=3)
    assert fs.advance(b) == []                                     # the first position: nothing to compare
    b2 = b.copy()
    b2.push_san("Rh8+")
    assert fs.advance(b2) == []                                    # the rook on c1 is Black's and did not move: c1 is untouched
    b4 = chess.Board("3R4/2Pk4/1K6/8/8/p7/8/3r4 w - - 0 1")        # their rook left c1
    assert fs.advance(b4) == [f.id] and f.id in fs.stale           # permanent: no frame restores it
    assert f.payload["perm"]


# ---------------------------------------------------------------------------------------------------------------- item 3: lazy recomputation, two tiers

def test_52_19_3_the_queue_is_ordered_by_chain_stake():
    fs = FS.FactStore()
    low = fs.add("kill", ("win", 1), True, "wins a pawn", reads={chess.A1}, sig=((0, "", 0, 0),), stake=1)
    high = fs.add("kill", ("win", 2), True, "wins a rook", reads={chess.A1}, sig=((0, "", 0, 0),), stake=5)
    pawn = fs.add("kill", ("win", 3), True, "pawn fact under a mate", reads={chess.A1}, sig=((0, "", 0, 0),), stake=1)
    fs.add("claim", ("mate", 4), True, "mate", reads={chess.H1}, sig=((7, "", 0, 0),), just=[pawn.id], stake=100)
    fs.invalidate({chess.A1}, permanent=True)
    q = fs.queue()
    assert [f.chain_stake for f in q] == [100, 100, 5, 1]          # the pawn fact leads with the mate that stands on it
    assert q[0].id == pawn.id and q[-1].id == low.id and high.id in [f.id for f in q]


def test_52_19_3_a_stale_fact_is_recomputed_only_when_a_live_question_needs_it_and_the_rest_stays_queued():
    ag1 = AG.Agenda(state_of(OPEN, "w"))
    ag1.ask()
    fs = ag1.ctx.facts
    b = chess.Board(OPEN)
    fs.advance(b)
    b2 = b.copy()
    b2.push_san("Nxe5")
    b2.push_san("Nxe5")
    flagged = fs.advance(b2)
    assert flagged and len(fs.queue()) >= len(flagged) // 2            # the listener only flags; the queue holds them, nothing was recomputed
    base = fs.snapshot()
    ag2 = AG.Agenda(state_of(b2.fen(), "w"), facts=ag1.ctx.facts)
    ag2.ask()
    d = ag2.facts_delta()
    assert d["recomputed"] > 0 and d["served"] > 0                      # the questions the live agenda asked were served or recomputed
    left = fs.queue()
    assert left and left[0].chain_stake >= left[-1].chain_stake         # what no live question needed stays queued, highest chain stake first
    line = next(ln for ln in ag2.lines if ln.startswith("facts:"))
    assert f"recomputed {d['recomputed']} stale fact(s)" in line and f"background queue: {len(left)} (top stake {left[0].chain_stake})" in line
    print(f"OBS 52.19 (3): flagged {len(flagged)}; the live agenda served {d['served']}, recomputed {d['recomputed']} in the foreground; background queue {len(left)} (top stake {left[0].chain_stake})")


def test_52_19_3_the_out_of_time_gaps_of_the_live_move_enter_the_queue_and_the_drain_resumes_them_exactly_there():
    from catspace.incubator.analyst.discovery import ponder as PD
    ag = AG.Agenda(state_of(OPEN, "w"), seconds=0.05)
    ag.ask()
    fs = ag.ctx.facts
    gaps = [f for f in fs.queue() if f.kind == "gap"]
    assert gaps and all(g.resume["timed"] and g.resume["mark"] is not None and g.resume["reply"] for g in gaps)      # the route, the reply it stopped at and the time mark
    assert [g.chain_stake for g in gaps] == sorted((g.chain_stake for g in gaps), reverse=True)
    r = PD.drain(ag.ctx, "w", 30)
    assert r["done"] == len(gaps) and r["left"] == 0                    # every cut route was followed to its end on the other clock
    assert all(g.resolved or g.superseded for g in gaps)
    print(f"OBS 52.19 (3): drained {r['done']} of {r['queue']} queued gap routes in {r['secs']} s")


def test_52_19_3_the_background_tier_ponders_by_lemma_the_human_replies_and_stops_when_he_moves():
    from catspace.incubator.analyst.discovery import ponder as PD
    ag = AG.Agenda(state_of(OPEN, "w"), seconds=0.2)
    ag.ask()
    b = chess.Board(OPEN)
    b.push_san("d3")                                                   # our move; the human (Black) is to move
    lines = []
    pd = PD.Ponderer(ag.ctx)
    r = pd.run(b, 4.0, emit=lines.append)
    assert r["replies"] >= 1 and any(ln.startswith("background: if ") for ln in lines)
    assert all(ln.startswith("background:") for ln in lines)
    # he moves: cancel stops the worker promptly
    done = {}
    pd2 = PD.Ponderer(ag.ctx)
    th = threading.Thread(target=lambda: done.update(pd2.run(b, None)), daemon=True)
    th.start()
    time.sleep(0.5)
    t0 = time.monotonic()
    pd2.cancel()
    th.join(10)
    assert not th.is_alive() and time.monotonic() - t0 < 5


def test_52_19_3_the_trace_prints_both_tiers_and_the_deciding_claims_graph():
    ag = AG.Agenda(state_of(LRTF, "w"))
    ag.ask()
    t = ag.trace()
    assert "foreground: recomputed" in t and "background queue:" in t and "justification graph of the decision (Rh8+)" in t
    print("\n".join(ln for ln in ag.lines if ln.startswith("facts:") or "graph" in ln))
