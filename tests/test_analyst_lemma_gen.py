"""Invariants for the lemma machinery: ids are stable and content-derived; guards round-trip through json; the box
cover is exact; subsumption inherits every embedding; the applier reproduces the generator's own labels on the
frame's rows; prove at the generation cap reproduces the stored coverage."""
import json

import chess
import pytest

from catspace.incubator.analyst import lemma_gen as lg
from catspace.incubator.analyst.chess_domain import domain
from catspace.incubator.analyst.lemma_apply import Applier, Compiled


def test_lemma_id_is_content_derived():
    a = lg.lemma_id(("wQ", "wR"), (("attacks", (0, 1), True),), 1, "acting keeps", [(("lands_attacked", (), True),)])
    b = lg.lemma_id(("wQ", "wR"), (("attacks", (0, 1), True),), 1, "acting keeps", [(("lands_attacked", (), True),)])
    c = lg.lemma_id(("wQ", "wR"), (("attacks", (0, 1), True),), 0, "acting keeps", [(("lands_attacked", (), True),)])
    assert a == b and a != c and a.startswith("L-") and len(a) == 10


def test_guard_json_round_trip():
    g = [(("attacked", (1,), True), ("stays_aligned", (1,), False)), (("lands_attacked", (), True),)]
    js = [[[r, list(sl), pol] for r, sl, pol in t] for t in g]
    assert lg.guard_from_json(json.loads(json.dumps(js))) == g
    assert lg.guard_from_json([["lands_attacked", [], True]]) == [(("lands_attacked", (), True),)]    # the old flat format


def test_box_cover_is_exact_and_covers():
    held = {("wK", "wB"), ("wK", "wN"), ("bK", "bB"), ("bK", "bN"), ("wK", "bB")}
    boxes = lg.box_cover(held, 2)
    import itertools
    covered = set()
    for box in boxes:
        cells = set(itertools.product(*box))
        assert cells <= held, f"box {box} claims a cell that did not hold"
        covered |= cells
    assert covered == held


@pytest.fixture(scope="module")
def gen():
    return lg.Generator(domain(), cap=200, log=lambda *a, **k: None)


def test_subsumption_inherits_every_embedding(gen):
    from collections import defaultdict
    by = defaultdict(list)
    for t in ("wK", "wR"):
        by[t].append(gen.evaluate(lg.Frame((t,), ()), 0, []))
    fr2 = lg.Frame(("wK", "wR"), (("same_side", (0, 1), True),))
    l2 = gen.evaluate(fr2, 0, gen.applicable(fr2, 0, by)); by["wK"].append(l2)
    fr3 = lg.Frame(("wK", "wR", "wR"), (("same_side", (0, 1), True), ("same_side", (0, 2), True)))
    inh = gen.applicable(fr3, 0, by)
    texts = [t for _, _, t, _ in inh]
    assert any("as (0, 1)" in t for t in texts) and any("as (0, 2)" in t for t in texts), texts
    l3 = gen.evaluate(fr3, 0, inh)
    assert l3.broke == 0 or l3.broke < l3.support - l3.holds, "the sub-frames' guards must explain part of the exceptions"


def test_applier_reproduces_the_generator_labels(gen):
    """Every (placement, action) row the generator labelled must get the same verdict from the applier on that board."""
    fr = lg.Frame(("wQ", "wR", "bB"), (("attacks", (0, 1), True), ("attacks", (2, 1), True)))
    lem = gen.evaluate(fr, 1, [])
    assert lem is not None
    comp = Compiled(0, fr, 1, lem.claim, lem.full_guard(), lem.text())
    app = Applier(gen.dom, [comp])
    rows, _ = gen.rows_of(fr, 1)
    checked = 0; mismatches = 0
    for state, ids, a, d in rows[:300]:
        b = state.board.copy(stack=False); b.turn = b.piece_at(ids[1]).color
        res = app.apply(b, bodies=set(ids), actor=ids[1], only_action=a[1])
        vs = [v for (e, k), vl in res["verdicts"].items() for v in vl]
        if not vs:
            continue
        pred = vs[0][1]
        # the guard's own answer on this row: exception when the guard fires, else the claim
        fired = any(all(gen._lit(state, ids, r, sl, pol, action=a) for r, sl, pol in term) for term in lem.full_guard())
        expect = ("loses" if lem.claim == "acting keeps" else "keeps") if fired else ("keeps" if lem.claim == "acting keeps" else "loses")
        checked += 1; mismatches += pred != expect
    assert checked > 50
    assert mismatches == 0, f"{mismatches} of {checked} rows: the applier disagrees with the generator's own guard evaluation"


def test_prove_at_the_generation_cap_reproduces_coverage(gen, tmp_path):
    fr = lg.Frame(("wQ",), ())
    lem = gen.evaluate(fr, 0, [])
    lib = [lg.lemma_to_json(lem)]
    p = tmp_path / "lib.json"; json.dump(lib, open(p, "w"))
    out = lg.prove(gen.dom, p, cap=200, log=lambda *a, **k: None)
    assert out[0]["support"] == lem.support
    assert out[0]["exceptions"] == lem.support - lem.holds
    assert out[0]["coverage"] == pytest.approx((lem.covered) / max(1, lem.covered + lem.residual), abs=1e-9)


def _trade_rows(n_pos: int = 14):
    """Hand-made rows of a [wQ bQ bP] world where the white queen attacks a black queen defended by a pawn: the trade
    keeps material, a queen landing on any other attacked square loses it. Outcome by hand: captured value minus nine
    when black can take back, else the captured value."""
    import random
    from catspace.incubator.analyst.chess_domain import State, VAL
    rng = random.Random(3); rows = []
    while len(rows) < n_pos * 15:
        b = chess.Board(None)
        sq = rng.sample(range(64), 5)
        b.set_piece_at(sq[0], chess.Piece(chess.QUEEN, chess.WHITE)); b.set_piece_at(sq[1], chess.Piece(chess.QUEEN, chess.BLACK))
        b.set_piece_at(sq[2], chess.Piece(chess.PAWN, chess.BLACK)); b.set_piece_at(sq[3], chess.Piece(chess.KING, chess.WHITE)); b.set_piece_at(sq[4], chess.Piece(chess.KING, chess.BLACK))
        b.turn = chess.WHITE
        if not b.is_valid() or b.is_check() or not b.is_attacked_by(chess.WHITE, sq[1]) or sq[1] not in b.attacks(sq[2]) or chess.square_rank(sq[2]) in (0, 7):
            continue
        st = State(b, [sq[0], sq[1], sq[2]])
        for m in b.legal_moves:
            if m.from_square != sq[0]:
                continue
            cap = VAL[b.piece_at(m.to_square).piece_type] if b.is_capture(m) else 0
            nb = b.copy(stack=False); nb.push(m)
            d = cap - (9 if nb.attackers(chess.BLACK, m.to_square) else 0)
            rows.append((st, [sq[0], sq[1], sq[2]], (b, m), float(d)))
    return rows


def test_refinement_adds_the_except_when_literal(gen):
    """The trade: 'lands attacked' fires on Qxq, where the claim (keeps) still holds. The refinement adds the literal
    that separates the false positives ('except when it captures a piece worth at least the queen'), keeping the true
    exceptions (2026-09-24, from game 669)."""
    rows = _trade_rows(); n = len(rows); all_mask = (1 << n) - 1; cache = {}
    exc = sum(1 << i for i, r in enumerate(rows) if r[3] < 0)
    term = (("lands_attacked", (), True),)
    m = gen.term_mask(rows, term, cache, all_mask)
    fp = lg._popcount(m & ~exc)
    assert fp >= 10, f"the hand-made world must hold at least ten trades that land attacked, got {fp}"
    term2, m2 = gen.refine_term(rows, term, m, exc, cache, all_mask, 3)
    assert len(term2) == 2 and term2[0] == term[0]
    r, sl, pol = term2[1]
    assert pol is False and r in ("captures_worth_ge", "captures"), term2
    assert lg._popcount(m2 & ~exc) < fp and lg._popcount(m2 & exc) >= 0.9 * lg._popcount(m & exc)
    # the inherited path: a parent's 'lands attacked' arrives in this frame, is refined here, and the frame counts as new
    fr = lg.Frame(("wQ", "bQ", "bP"), (("attacks", (0, 1), True), ("defends", (2, 1), True)))
    saved = gen.rows_of
    gen.rows_of = lambda frame, slot: (rows, False)
    try:
        lem = gen.evaluate(fr, 0, [("acting keeps", [term], "[wQ] slot 0 as (0,)", ())])
    finally:
        gen.rows_of = saved
    assert lem.claim == "acting keeps" and lem.refined == fp - lg._popcount(m2 & ~exc) and lem.refined > 0
    assert lem.inherited == [term2] and lem.parents and lem.reversed is False
    js = lg.lemma_to_json(lem); assert js["refined"] == lem.refined and lg.guard_from_json(js["inherited"]) == [term2]


def test_guard_terms_are_not_nested_duplicates(gen):
    """After a term is refined, the greedy loop must not pick the same term, or one inside it, again for the few
    exceptions the refinement gave up (the smoke run of 2026-09-24 printed a term three times)."""
    rows = _trade_rows(); n = len(rows); cache = {}
    exc = sum(1 << i for i, r in enumerate(rows) if r[3] < 0)
    fr = lg.Frame(("wQ", "bQ", "bP"), (("attacks", (0, 1), True), ("defends", (2, 1), True)))
    guard, cov, pur, matched = gen.find_guard(rows, exc, fr, cache)
    sets = [frozenset(t) for t in guard]
    assert len(sets) == len(set(sets)), guard
    for a in sets:
        for b in sets:
            assert a is b or not (a < b), f"nested terms {a} < {b}"
