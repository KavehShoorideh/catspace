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
