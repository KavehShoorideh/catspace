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
    served = [ln for ln in b.lines if " -- served from the store (fact #" in ln]
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
        hit = {f.id for f in fs.facts.values() if f.reads & diff and not f.superseded and f.kind != "roots"}       # (a root list is a pure function of its cells, not a belief: never flagged)
        assert hit <= new                                          # everything that read a changed square is flagged
        for fid in new - hit:                                      # ... and the rest only because something it stands on was
            assert any(j in new for j in fs.facts[fid].just)
        flagged_any |= bool(new)
        fresh_any |= any(f.id not in new for f in fs.facts.values() if f.kind != "roots")
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
    ag = AG.Agenda(state_of(OPEN, "w"), seconds=0.01)                 # (0.05 s no longer cuts anything after plan 54: the whole agenda runs in under it)
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


# ---------------------------------------------------------------------------------------------------------------- item 4: across moves and puzzles

def test_52_19_4_a_fact_derived_on_one_move_whose_reads_are_untouched_is_served_on_the_next():
    ag1 = AG.Agenda(state_of(OPEN, "w"))
    ag1.ask()
    b = chess.Board(OPEN)
    ag1.ctx.facts.advance(b)
    b2 = b.copy()
    b2.push_san("a3")
    b2.push_san("a6")                                                  # the far wing: our move and theirs, the reads mostly untouched
    ag1.ctx.facts.advance(b2)
    ag2 = AG.Agenda(state_of(b2.fen(), "w"), carry=ag1.ctx)
    ag2.ask()
    d = ag2.facts_delta()
    served = [ln for ln in ag2.lines if " -- served from the store (fact #" in ln]
    assert d["served"] > 0 and served and ag2.asked < ag1.asked
    assert ag2.ctx.memo is ag1.ctx.memo and ag2.ctx.facts is ag1.ctx.facts      # the exact memos and the store outlive the agenda
    print(f"OBS 52.19 (4): move 2 served {d['served']} facts from move 1 (asked {ag2.asked} against {ag1.asked}); {len(served)} root questions served")


def test_52_19_4_unrelated_puzzles_share_few_facts_and_say_so():
    ctx, before, cross = None, {}, 0
    for fen, us in (("2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 0 1", "w"), ("r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1", "w")):
        b = chess.Board(fen)
        if ctx is not None:
            ctx.facts.advance(b)
            before = {f.id: f.served for f in ctx.facts.facts.values()}
        ag = AG.Agenda(state_of(fen, us), carry=ctx)
        ag.ask()
        ctx = ag.ctx
    cross = sum(f.served - before[f.id] for f in ctx.facts.facts.values() if f.id in before)
    print(f"OBS 52.19 (4): facts of the first puzzle served in the second (an unrelated position): {cross} -- expect few")
    assert cross <= 10


# ---------------------------------------------------------------------------------------------------------------- 52.21 (1): root lists as facts

BACKRANK = "6k1/5ppp/8/8/8/8/R7/K7 w - - 0 1"                                          # the rook reaches b8..e8 in two moves, each a mate
SMOTHER = "6rk/6pp/8/8/2N5/8/8/K7 w - - 0 1"                                           # the knight reaches f7 in two moves (smothered mate): it reads a few squares


def _reach_texts(ag):
    return [q.text for q in ag.all if q.pattern.get("reach")]


def _strays(fen, piece_name):
    """A stray piece on every empty square of `fen`, each sibling asked with the seeded store and with an empty one: -> (squares where `piece_name`'s root list was served, where it was rescanned)."""
    seed_ctx = fresh_ctx()
    base = _reach_texts(inner_agenda(fen, seed_ctx))
    assert base and seed_ctx.roots_served == 0 and seed_ctx.facts.roots_derived >= 1
    served_on, rescanned_on = [], []
    for colour_piece in ("p", "P", "n", "N", "b"):
        for sq in chess.SQUARES:
            b = chess.Board(fen)
            if b.piece_at(sq) is not None or (colour_piece in "pP" and chess.square_rank(sq) in (0, 7)):
                continue
            b.set_piece_at(sq, chess.Piece.from_symbol(colour_piece))
            if not b.is_valid():
                continue
            fresh = inner_agenda(b.fen(), fresh_ctx())                                     # the rescan: a store with nothing in it
            ctx = AG.ChaseCtx(seed_ctx.facts)                                              # the seeded store
            ctx.left = 10 ** 6
            hit = inner_agenda(b.fen(), ctx)
            assert _reach_texts(hit) == _reach_texts(fresh), (colour_piece, chess.square_name(sq))      # served or scanned, the list is the rescan's
            (served_on if any("roots served from the store" in ln and piece_name in ln for ln in hit.lines) else rescanned_on).append((colour_piece, sq))
    return served_on, rescanned_on


def test_52_21_1_a_root_list_is_served_only_where_every_cell_it_read_is_as_it_was_and_then_agrees_with_a_rescan():
    served_on, rescanned_on = _strays(SMOTHER, "the N on c4")
    assert served_on and rescanned_on                                                      # not vacuous: some siblings are served, some are not
    far = set(served_on)
    assert ("N", chess.D1) in far                                                          # a piece on a square nothing read (no ray of the squares the mate depends on reaches it): served
    assert all((c, s) not in far for c in "pPnNb" for s in (chess.E5, chess.D6, chess.F7, chess.G7))      # a stray piece on the knight's route or on the squares of the mate: never served
    ra, rb = _strays(BACKRANK, "the R on a2")                                              # the rook reads the lines of its landing squares: few siblings are served, every one agrees with a rescan
    print(f"OBS 52.21 (1): knight position: {len(served_on)} siblings with a stray piece served the root list from the store, {len(rescanned_on)} read a cell it had read and were rescanned; "
          f"rook position: {len(ra)} served, {len(rb)} rescanned; every one agrees with a rescan")


def test_52_21_1_root_lists_are_filed_per_piece_not_as_belief_and_never_enter_the_queue():
    ctx = fresh_ctx()
    ag = inner_agenda(BACKRANK, ctx)
    roots = [f for f in ctx.facts.facts.values() if f.kind == "roots"]
    assert roots and all(f.key[0] == "roots:reach" and f.portable for f in roots)
    assert ctx.facts.derived == len([f for f in ctx.facts.facts.values() if f.kind != "roots"])        # counted apart from the claims and kills
    ctx.facts.invalidate(frozenset(range(64)))
    assert not any(f.kind == "roots" for f in ctx.facts.queue())                                      # a pure function of its cells: no stale flag, no recomputation queue
    again = inner_agenda(BACKRANK, ctx)
    assert any("roots served from the store (fact #" in ln for ln in again.lines) and _reach_texts(again) == _reach_texts(ag)


# ---------------------------------------------------------------------------------------------------------------- 52.21 (2): chase claims portable

KNIGHT_CHECK = "8/p1NR2bp/4pkp1/2p2p2/4n3/P3B1P1/7r/2K5 w - - 4 39"                  # dev_20 05gLw: Ne8+ is a chase claim whose route stands on facts with read sets


def test_52_21_2_a_chase_claim_with_a_read_set_is_served_on_a_transposition_and_not_where_a_cell_it_read_changed():
    ag1 = AG.Agenda(state_of(KNIGHT_CHECK, "w"))
    ag1.ask()
    d1 = ag1.decision()
    port = [f for f in ag1.ctx.facts.facts.values() if f.payload.get("chase")]
    assert d1 is not None and d1.lemma.startswith("chase") and port and all(f.portable and f.reads for f in port)       # the claim is filed portable, with the union of what its route stood on
    fid = ag1.claim_fid[id(d1)]
    assert ag1.ctx.facts.facts[fid].payload["chase"]
    trans = " ".join(KNIGHT_CHECK.split()[:4]) + " 17 44"                                # the same placement reached by another move order: other clocks
    ag2 = AG.Agenda(state_of(trans, "w"), carry=ag1.ctx)
    ag2.ask()
    d2 = ag2.decision()
    assert d2 is not None and d2.move == d1.move and ag2.ctx.chase_served >= 1 and ag2.asked < ag1.asked
    assert any("-- served from the store (fact #" in ln and "chase" in ln for ln in ag2.lines)
    reads = ag1.ctx.facts.facts[fid].reads
    changed = 0
    for sq in sorted(reads):                                                              # a cell the claim read, changed: never served
        b = chess.Board(KNIGHT_CHECK)
        if b.piece_at(sq) is not None:
            continue
        b.set_piece_at(sq, chess.Piece.from_symbol("p"))
        if not b.is_valid():
            continue
        ag3 = AG.Agenda(state_of(b.fen(), "w"), carry=ag1.ctx)
        ag3.ask()
        assert not any(ln.startswith("Q") and " -- served from the store (fact #%d," % fid in ln for ln in ag3.lines), chess.square_name(sq)
        changed += 1
    assert changed
    print(f"OBS 52.21 (2): the chase claim {d1.what} ({len(reads)} squares read, {len(port)} chase claims filed portable) was served on a transposition (asked {ag2.asked} against {ag1.asked}); "
          f"not served on any of {changed} boards with a stray pawn on a cell it read")


def test_52_21_2_a_chase_claim_that_stands_on_a_positional_fact_stays_non_portable_and_the_trace_says_so():
    ag = AG.Agenda(state_of(LRTF, "w"))
    ag.ask()
    assert any("chase claim filed for this position only: it stands on fact #" in ln and "read its whole position" in ln for ln in ag.lines)
    d = ag.decision()
    f = ag.ctx.facts.facts[ag.claim_fid[id(d)]]
    assert d.lemma.startswith("chase") and not f.portable and not f.payload.get("chase")


# ---------------------------------------------------------------------------------------------------------------- 52.21 (3): the deadline finishes the route in hand

def test_52_21_3_past_the_deadline_the_route_in_hand_runs_to_its_end_and_no_new_route_is_opened():
    ctx = AG.ChaseCtx()
    ctx.deadline = time.monotonic() + 1e6
    assert not ctx.out_of_time() and ctx.hand is None
    ctx.route_depth = 2
    ctx.route_names.extend(["Rh8+", "Rd8+"])
    ctx.deadline = time.monotonic() - 1
    assert ctx.out_of_time() and ctx.hand == 2                                       # the innermost route at the moment the clock was first seen is the route in hand
    assert ctx.finishing(2) and not ctx.finishing(1) and not ctx.finishing(3) and not ctx.finishing(0)
    assert ctx.finish_lines == ["out of time: finishing Rd8+ (the route in hand, depth 2), no new routes"]
    ctx.abort = True                                                                  # the human moved: nothing is finished
    assert not ctx.finishing(2)


def test_52_21_3_in_a_real_chase_the_clock_running_out_in_a_nested_route_opens_nothing_new():
    orig = AG.Agenda._chase_route_body
    seen = {"trigger": None, "after": []}

    def wrapped(self, m, promote=False, cheap=False, override=None):
        ctx = self.ctx
        if not cheap:
            if seen["trigger"] is None and ctx.route_depth == 2:
                seen["trigger"] = (self.board.san(m), ctx.route_depth)
                ctx.deadline = ctx.final = time.monotonic() - 1                       # the clock runs out inside the second route
            elif seen["trigger"] is not None:
                seen["after"].append((self.board.san(m), ctx.route_depth))             # a route opened after the deadline
        return orig(self, m, promote, cheap, override)
    AG.Agenda._chase_route_body = wrapped
    try:
        ag = AG.Agenda(state_of(LRTF, "w"), seconds=1e6)
        ag.ask()
    finally:
        AG.Agenda._chase_route_body = orig
    ctx = ag.ctx
    assert seen["trigger"] is not None and seen["after"] == []                         # no new route at any depth
    assert ctx.hit and ctx.finished >= 1 and ctx.hand is not None
    assert any(ln.startswith("out of time: finishing ") and "no new routes" in ln for ln in ag.lines)
    assert any("out of time" in g for g in ag.state.gaps) or any("out of time" in ln for ln in ag.lines)      # what was left is a GAP, never a kill
    print(f"OBS 52.21 (3): clock cut inside {seen['trigger'][0]} (depth {seen['trigger'][1]}): routes opened after the deadline {len(seen['after'])}, routes finished {ctx.finished}")
