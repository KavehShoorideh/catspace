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
    return lg.Generator(domain(), cap=200, log=lambda *a, **k: None, sequential=False)   # the full cap: these tests compare against the re-test at the same cap


def test_sequential_sampling_decides_clear_cases_on_the_quarter():
    """A one-piece frame is decided on a quarter of the cap (flagged); the re-test at the full cap agrees."""
    g = lg.Generator(domain(), cap=400, log=lambda *a, **k: None, sequential=True)
    l = g.evaluate(lg.Frame(("wK",), ()), 0, [])
    assert l is not None and l.sampled_small and l.support < 400 * 10
    o = lg._check(domain(), lg.Frame(("wK",), ()), 0, l.claim, l.full_guard(), cap=400, min_cov=0.95, min_pur=0.8)
    assert o["status"] in ("PROVED", "THEOREM")


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


def test_exhaustive_check_yields_a_theorem():
    """Exhaustion (Kaveh 2026-09-24: "actual proofs, at least in the pure frame sense"): every placement of the frame's
    closed world is enumerated; a guard that covers every exception makes the lemma a THEOREM of that world."""
    from catspace.incubator.analyst import chess_domain as cd
    dom = domain()
    raw = cd.placement_count(("wK",))
    assert raw == 10 * 63
    o = lg._check(dom, lg.Frame(("wK",), ()), 0, "acting keeps", [], cap=raw, min_cov=0.95, min_pur=0.8)
    assert o["complete"] and o["status"] == "THEOREM" and o["exceptions"] == 0 and o["support"] > 100   # shapes: one placement per configuration
    o = lg._check(dom, lg.Frame(("wK",), ()), 0, "acting keeps", [], cap=100, min_cov=0.95, min_pur=0.8)
    assert not o["complete"] and o["status"] == "PROVED"                     # a sample never makes a theorem


def test_joins_build_a_frame_from_two_known_frames(gen):
    """Kaveh 2026-09-24: extend by a known frame. The join carries both parts' literals, remapped, no literal across,
    and is flagged; both parts embed so subsumption inherits both guards."""
    from collections import defaultdict
    gen.max_joins = 2
    a = lg.Frame(("wK", "wR"), (("same_side", (0, 1), True),))
    b = lg.Frame(("bK", "bB"), (("same_side", (0, 1), True),))
    la = gen.evaluate(a, 1, []); lb = gen.evaluate(b, 1, [])
    gen.library = [la, lb]
    js = gen.joins(a, 4)
    gen.max_joins = 0; gen.library = []
    assert js and all(j.joined and j.size == 4 for j in js)
    j = next(x for x in js if x.types == ("wK", "wR", "bK", "bB"))
    assert j.literals == (("same_side", (0, 1), True), ("same_side", (2, 3), True))
    assert "(join)" in j.text()
    by = defaultdict(list); by["wR"].append(la); by["bB"].append(lb)
    inh = gen.applicable(j, 1, by)
    assert any("as (0, 1)" in t for _, _, t, _ in inh)


def test_quantifier_lift_names_the_escape_square():
    """some_not_lands_attacked(king) = the king has a square to go to that is not attacked (Kaveh 2026-09-24: the
    escape square as a composition, not a primitive). A cornered king has none; a free king has one."""
    from catspace.incubator.analyst import chess_domain as cd
    dom = domain()
    rels = {r.name: r for r in lg.lift_some(dom)}
    assert "some_not_lands_attacked" in rels and rels["some_not_lands_attacked"].arity == 1
    b = chess.Board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")                       # a8 king, queen c7, king b6: every king square attacked, not yet mate? (b8, a7 covered)
    st = cd.State(b, [chess.A8])
    assert rels["some_not_lands_attacked"].fn(st, chess.A8) is False
    b2 = chess.Board("k7/8/1K6/8/8/8/8/8 b - - 0 1")                        # nothing but the kings: a8 king may go to b8? b8 is next to... b6 king covers a7, b7, c7; b8 is free
    st2 = cd.State(b2, [chess.A8])
    assert rels["some_not_lands_attacked"].fn(st2, chess.A8) is True
    assert rels["some_captures"].arity == 1 and rels["some_captures_worth_ge"].arity == 2


def test_threat_lift_rolls_the_threat_forward_and_the_quantifier_finds_the_escape():
    """Kaveh 2026-09-24: 'roll the game state forward, place the king in check, and the escape squares are the legal
    moves that remove the threat'. threat_some_not_lands_attacked(king) = once the opponent's best reply has landed,
    some king move lands unattacked."""
    from catspace.incubator.analyst import chess_domain as cd
    dom = lg.lifted(cd.domain(oracle="frame"), ("some", "threat"))
    rels = {r.name: r for r in dom.relations}
    assert "threat_some_not_lands_attacked" in rels and rels["threat_some_not_lands_attacked"].arity == 2
    # the threat must be their best reply, so it captures a frame piece: king a1, rook d2, black queen h2 -> ...Qxd2,
    # after which b1 is still free for the king
    b = chess.Board("6k1/8/8/8/8/8/3R3q/K7 w - - 0 1")
    st = cd.State(b, [chess.A1, chess.D2, chess.H2])
    assert rels["threat_some_not_lands_attacked"].fn(st, chess.A1, chess.A1) is True
    b2 = chess.Board("1r4k1/8/8/8/8/8/1R5q/K7 w - - 0 1")                  # rook b2: ...Qxb2 is check, covers a2 and b1, and the rook on b8 defends the queen
    st2 = cd.State(b2, [chess.A1, chess.B2, chess.H2])
    assert rels["threat_some_not_lands_attacked"].fn(st2, chess.A1, chess.A1) is False
    assert rels["threat_attacked"].fn(st2, chess.A1, chess.A1) is True      # the king is in check once the threat lands


def test_streamed_check_matches_the_mask_path(gen):
    """The re-test streams rows and counts; it must agree exactly with the bitset path the generator uses."""
    fr = lg.Frame(("wQ",), ()); guard = [(("lands_attacked", (), True), ("attacked", (0,), False))]
    o = lg._check(gen.dom, fr, 0, "acting keeps", guard, cap=200, min_cov=0.95, min_pur=0.8)
    g = lg.Generator(gen.dom, cap=200, log=lambda *a, **k: None)
    rows, complete = g.rows_of(fr, 0); n = len(rows); all_mask = (1 << n) - 1
    exc = sum(1 << i for i, r in enumerate(rows) if r[3] < 0)
    m = g.term_mask(rows, guard[0], {}, all_mask)
    assert o["support"] == n and o["exceptions"] == lg._popcount(exc)
    assert abs(o["coverage"] - lg._popcount(m & exc) / max(1, lg._popcount(exc))) < 1e-9
    assert abs(o["purity"] - lg._popcount(m & exc) / max(1, lg._popcount(m))) < 1e-9


def test_canonical_frame_identifies_slot_orders_and_colours():
    """Kaveh 2026-09-24: 'many of them should be equivalent'. [bN wQ] attacks(1,0) and [wQ bN] attacks(0,1) are one
    frame; so is the colour flip [bQ wN] attacks(0,1)."""
    from catspace.incubator.analyst.chess_domain import COLOUR_FLIP
    a = lg.canonical_frame(("bN", "wQ"), (("attacks", (1, 0), True),), [COLOUR_FLIP])
    b = lg.canonical_frame(("wQ", "bN"), (("attacks", (0, 1), True),), [COLOUR_FLIP])
    c = lg.canonical_frame(("bQ", "wN"), (("attacks", (0, 1), True),), [COLOUR_FLIP])
    assert a[:2] == b[:2] == c[:2]
    d = lg.canonical_frame(("wQ", "bN"), (("attacks", (1, 0), True),), [COLOUR_FLIP])   # the knight attacks the queen: a different frame
    assert d[:2] != a[:2]


def test_applier_applies_a_lemma_through_its_colour_image():
    """A canonical lemma about a white piece must fire for the black piece in the mirrored situation."""
    from catspace.incubator.analyst.chess_domain import domain
    dom = domain()
    fr = lg.Frame(("wQ",), ())
    c = Compiled(0, fr, 0, "acting keeps", [(("lands_attacked", (), True),)], "L-x [wQ] (no relation) :: slot 0 acting keeps unless lands_attacked()", {"holds": 0.5, "exception": -9.0})
    app = Applier(dom, [c])
    assert ("bQ",) in app.by_types
    b = chess.Board("4k3/8/8/8/8/8/7q/3QK3 b - - 0 1")                       # black queen h2 to move
    res = app.apply(b, actor=chess.H2)
    assert res["verdicts"], "the black queen's moves get verdicts from the white-queen lemma"


def test_checkpoint_round_trip(tmp_path, monkeypatch):
    """A killed run resumes from its checkpoint with the same library and frontier."""
    from catspace.incubator.analyst.chess_domain import domain
    monkeypatch.setattr(lg, "OUT", tmp_path)
    g = lg.Generator(domain(), cap=60, log=lambda *a, **k: None)
    lib = g.run(max_size=2)
    assert (tmp_path / "checkpoint_chess.json").exists()
    g2 = lg.Generator(domain(), cap=60, log=lambda *a, **k: None)
    st = g2.load_checkpoint()
    assert st is not None and len(g2.library) == len(lib)
    assert [l.id for l in g2.library] == [l.id for l in lib]


def test_type_tree_splits_and_embeds_across_levels():
    """Top-down types (Kaveh 2026-09-24): frames start at the root and split a slot into its children; a lemma at an
    abstract type embeds in any frame whose types are its descendants."""
    from catspace.incubator.analyst.chess_domain import domain
    dom = domain()
    assert dom.root == "*" and dom.leaves("Q") == ["wQ", "bQ"] and len(dom.leaves("*")) == 12
    assert dom.is_a("wQ", "Q") and dom.is_a("wQ", "*") and not dom.is_a("Q", "wQ")
    g = lg.Generator(dom, cap=200, log=lambda *a, **k: None, sequential=False)
    root = lg.Frame(("*", "*"), (("attacks", (0, 1), True),))
    kids = g.splits(root)
    assert len(kids) == 12 and all(k.size == 2 for k in kids) and any(k.types == ("Q", "*") for k in kids)
    l1 = g.evaluate(lg.Frame(("*",), ()), 0, [])
    assert l1 is not None and l1.support >= 30
    from collections import defaultdict
    by = defaultdict(list); by["*"].append(l1)
    inh = g.applicable(lg.Frame(("wQ", "bN"), (("attacks", (0, 1), True),)), 0, by)
    assert inh, "the root lemma must embed in a concrete frame"
    st = next(iter(dom.placements(("Q", "*"), 5, 0)))[0]
    assert st.board.piece_at(st.ids[0]).piece_type == 5                                     # a queen of either colour
