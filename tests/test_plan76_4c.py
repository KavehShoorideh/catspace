"""plan 76.4c MINING OBJECTS AND MATERIAL (docs/ANALYST_PLAN.md 76.4; 76's borrowings (4) and (5)): (a) one term language with typed combinators over the ontology
shelf's relations (terms.py) -- its typing, its pricing, its implication, the atoms re-expressed in it equal to the old readings, the language naming no game
object; (b) mining objects (object_miner.py) -- the derivations' readings and flips, the explanation tests and the pruning by implication, the three
rediscoveries (LINE, BLOCKED, REGION) on fixed positions, the mined kinds as shelf entries and node kinds of feature-set version 3; (c) material as a mined
valuation (valuation_miner.py) -- the count term, the material fit on a fixture, the mined VALUATION beside the hand one."""
from __future__ import annotations

import ast
from pathlib import Path

import chess
import pytest

from catspace.incubator.analyst.discovery import gdl_chess as GC
from catspace.incubator.analyst.discovery import names as NM
from catspace.incubator.analyst.discovery import ontology as ON
from catspace.incubator.analyst.discovery import technique_measures as TM
from catspace.incubator.analyst.discovery import terms as TL

ROOT = Path(__file__).resolve().parents[1]
GENERIC = ("terms.py", "object_miner.py", "valuation_miner.py")      # the modules that must name no game object

# fixed positions: a middlegame (pos2 of the guard), the discovery position, two K+R v K positions (one the refuted measure's play-out start), a pawn ending
FENS = ("r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1", "r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1",
        "8/8/4K3/6k1/3R4/8/8/8 w - - 0 1", "8/8/8/8/1K6/1R6/k7/8 w - - 0 1", "8/3Pk3/4P3/8/pp6/4KP2/P7/8 w - - 0 1",
        "8/8/8/8/8/2k3R1/8/K7 b - - 0 1")     # their king in check along a line: the square behind it is ours only with the king lifted


@pytest.fixture(scope="module")
def lang():
    return TM.term_language()


@pytest.fixture(scope="module")
def dom(lang):
    return lang[0]


# ---------------------------------------------------------------------------------------------------------------- (a) the language

def _code_strings_and_names(path: Path) -> set:
    """Every string constant (docstrings aside) and every identifier of a module's code."""
    tree = ast.parse(path.read_text())
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant):
            docs.add(id(node.body[0].value))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            out.add(node.value)
        elif isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
    return out


def test_a_the_language_names_no_game_object(dom):
    """THE GENERIC TEST: terms.py's code holds no relation of the spec, no constant of any of its sorts, no word of the names table, no shelf kind's name -- its
    sorts are found in the spec by role (the product object, the roles, the placement fluent), every relation comes from the shelf by name."""
    prog = GC.program()
    rels = {r.split(".", 1)[1] if "." in r else r for r in prog.nodes}
    consts = {m for ms in dom.members.values() for m in ms if len(m) > 1}
    words = (set(NM.WORDS) | {e["name"] for e in ON.load().values()}) - set(TL.SORTS)     # (the language's own sorts, the plan's: location, side, unit, ...)
    for mod in GENERIC:                                    # 76.4c (b), (c): the object miner and the valuation miner too
        seen = _code_strings_and_names(ROOT / "catspace" / "incubator" / "analyst" / "discovery" / mod)
        hits = sorted(seen & (rels | consts | words))
        assert hits == [], f"{mod} names game objects: {hits}"
    assert dom.location_kind in words and dom.comps and dom.placement.startswith(TL.STATE)


def test_a_the_sorts_found_by_role(dom):
    """The language's sorts read off the spec: the location (a product object of the shelf; its components each a chain of successors), the side (the roles),
    the placement fluent and its unit sort, whose turn, the opponents."""
    assert len(dom.locs) == len(dom.members[dom.comps[0]]) * len(dom.members[dom.comps[1]])
    assert dom.roles and set(dom.other) == set(dom.roles)
    assert dom.lang_sort(dom.unit_sort) == TL.UNIT and dom.lang_sort(dom.side_sort) == TL.SIDE
    assert TL.SORTS == ("location", "side", "unit", "locations", "units", "count", "bool")
    s = dom.slots("true.cell")
    assert [x[0] for x in s] == [TL.LOC, TL.SIDE, TL.UNIT]


def test_a_typing_and_pricing(dom):
    """A term is typed by its constructor (a sort mismatch is an error), prints as its s-expression and is read back from it, and is priced by size: a literal
    one, a constructor one, a side term or a constant in a literal's slot none, a defined name one."""
    king_step = TL.lit(dom, "king_step", None, None)
    att = TL.lit(dom, "attacked", TL.US, None)
    assert king_step.sort == (TL.LOC, TL.LOC) and att.sort == TL.LOCS and king_step.size == att.size == 1
    with pytest.raises(TL.TermError):
        TL.mk("and", king_step, att)
    with pytest.raises(TL.TermError):
        TL.mk("reach", att, att, None)
    with pytest.raises(TL.TermError):
        TL.lit(dom, "attacked", "nobody", None)
    defs = TL.Defs()
    free = defs.define("free", TL.mk("not", att))
    reg = TL.mk("reach", TL.mk("restrict", king_step, free), TL.lit(dom, "true.cell", None, TL.THEM, "king"), None)
    assert reg.sort == TL.LOCS and reg.size == 5
    assert reg.sexpr() == "(reach (restrict (king_step ? ?) free) (cell ? them king))"
    assert TL.parse(reg.sexpr(), dom, defs) == reg
    assert TL.mk("count", reg).sort == TL.COUNT and TL.mk("lex", TL.mk("count", reg), TL.mk("area", reg)).size == 13
    assert reg.rels(dom, defs) == frozenset({"king_step", "attacked", "true.cell", dom.other_rel})


def test_a_implication(dom):
    """THE IMPLICATION on the syntax (sound): (and A B), (minus A B), (restrict A S) imply A; A implies (or A B) and (closure A); a bounded reach implies a longer
    one; and every implied pair holds row by row on the fixed positions."""
    a = TL.lit(dom, "attacked", TL.US, None)
    b = TL.lit(dom, "true.cell", None, TL.THEM, "king")
    occ = TL.lit(dom, "occupied", None)
    step = TL.lit(dom, "step", None, None, None)
    pairs = [(TL.mk("and", a, occ), a), (TL.mk("minus", a, occ), a), (a, TL.mk("or", a, occ)), (step, TL.mk("closure", step)),
             (TL.mk("restrict", step, occ), step), (TL.mk("compose", TL.mk("restrict", step, occ), step), TL.mk("compose", step, step))]
    ks = TL.lit(dom, "king_step", None, None)
    pairs.append((TL.mk("reach", ks, b, 1), TL.mk("reach", ks, b, 2)))
    pairs.append((TL.mk("reach", ks, b, 2), TL.mk("reach", ks, b, None)))
    reader = GC.TermReader()
    for x, y in pairs:
        assert TL.implies(x, y), (x.sexpr(), y.sexpr())
        for f in FENS:
            ctx = TL.Ctx(dom, reader, chess.Board(f), "white")
            assert ctx.ev(x) <= ctx.ev(y), (x.sexpr(), y.sexpr(), f)
    assert not TL.implies(a, TL.mk("and", a, occ)) and not TL.implies(TL.mk("reach", ks, b, None), TL.mk("reach", ks, b, 1))


def test_a_the_atoms_as_terms_equal_the_old_readings(lang):
    """Every technique atom re-expressed in the term language (technique_measures.ATOM_TERMS) reads python-chess's value on the fixed positions, both sides read
    for -- but the vocabulary's P on a side whose only non-royal units are pawns (its convention: pawns the most advanced first, a per-kind order the language's
    `first` -- the unit sort's declared order, then the location's -- does not have): there the six P atoms are not compared (the FINDING of 76.4c (a))."""
    p_atoms = {k for k in TM.ATOM_TERMS if "P" in k.split("_")[0]}
    assert len(p_atoms) == 6
    for f in FENS:
        b = chess.Board(f)
        for side in (chess.WHITE, chess.BLACK):
            hand, got = TM.atom_values(b, side), TM.term_atom_values(b, side, lang)
            pawns_only = not (b.occupied_co[side] & ~b.kings & ~b.pawns) and b.pawns & b.occupied_co[side]
            for k in TM.ATOM_TERMS:
                if pawns_only and k in p_atoms:
                    continue
                assert hand[k] == got[k], (f, side, k, hand[k], got[k])


def test_a_the_region_lifts_their_king(lang):
    """The region's free squares are read with their king lifted (a line through it runs on): on a check along a rank the square behind their king is attacked;
    read without lifting it the region would hold the squares behind it -- the hand reading's value is the lifted one."""
    dom, reader, defs, atoms = lang
    b = chess.Board(FENS[-1])
    lifted = TM.term_atom_values(b, chess.WHITE, lang)["region"]
    d2 = TL.Defs()
    for name, body in defs.body.items():
        d2.body[name] = body
    d2.body["free"] = TL.parse("(not (attacked us ?))", dom, d2)
    unlifted = TL.Ctx(dom, reader, b, "white", d2).ev(atoms["region"])
    assert lifted == TM.atom_values(b, chess.WHITE)["region"] and unlifted == lifted + 2      # b3 and a3: the line through c3 runs on


def test_a_the_readers_agree(lang):
    """The chess reader (the native core's prover) and the GDL oracle reader (gdl.py's interpreter) give every atom the same value on the fixed positions."""
    dom, reader, defs, atoms = lang
    oracle = TL.GDLReader(GC.program(), GC.inputs_of())
    for f in FENS[:3]:
        b = chess.Board(f)
        c1, c2 = TL.Ctx(dom, reader, b, "white", defs), TL.Ctx(dom, oracle, GC.state_of(b), "white", defs)
        assert {k: c1.ev(t) for k, t in atoms.items()} == {k: c2.ev(t) for k, t in atoms.items()}


# ---------------------------------------------------------------------------------------------------------------- (b) mining objects

def _ref_line() -> frozenset:
    return frozenset((a, b, c) for a in chess.SQUARES for b in chess.SQUARES if a != b and chess.BB_RAYS[a][b] for c in chess.scan_forward(chess.between(a, b)))


def _ref_blocked(board: chess.Board) -> frozenset:
    return frozenset((a, b) for a in chess.SQUARES for b in chess.SQUARES if a != b and chess.BB_RAYS[a][b] and chess.between(a, b) & board.occupied)


INTERPOSE = "8/8/8/8/1n6/2k3R1/8/K7 b - - 0 1"         # their king in check along the third rank; their knight can step between (b4 -> d3)


def test_b_the_fuse_is_the_specs_largest_body():
    from catspace.incubator.analyst.discovery import object_miner as OM
    assert OM.body_fuse(GC.program()) == 8


@pytest.fixture(scope="module")
def harvested(dom):
    from catspace.incubator.analyst.discovery import lemma_store as LS
    from catspace.incubator.analyst.discovery import object_miner as OM
    from catspace.incubator.analyst.discovery import regression as RG
    prog = GC.program()
    B = GC.Binding()
    R = RG.PyRegressor(RG.Spec(prog, certified=GC.CERTIFIED), B)
    b = chess.Board(INTERPOSE)
    goals = [RG.goal_of("mate", LS.gdl_goals()["mate"], B, "black")]
    return OM.harvest(R, dom, [(b, "black", goals)], log=lambda *_a: None)


def test_b_the_derivations_read_the_squares_between_and_the_interposition_flips(harvested):
    """The mate goal's derivation reads the rook's check along the rank: a slide reading (g3, c3) whose leaves stand on d3, e3, f3 (the empty squares between);
    the knight's interposition (a witness of their legal move) flips one of those leaves: a flip of the slide reading by the `occupied` leaf."""
    sl = [r for r in harvested.true if r[1] == "slide" and (r[2], r[3]) == (chess.G3, chess.C3)]
    assert sl and sl[0][4] == frozenset({chess.D3, chess.E3, chess.F3})
    assert any(f[2] == "slide" and (f[3], f[4]) == (chess.G3, chess.C3) and f[5] == "occupied" for f in harvested.flips)


def test_b_line_explains_the_readings_an_impostor_does_not_and_a_failure_prunes_what_it_implies(dom, harvested):
    """LINE (the empty board's squares between) explains the slide readings; (via (slide ? ? ?) (slide ? ? ?)) reads slide -- never an explanation of slide's own
    readings; a candidate that misses a leaf (one step each way) fails 'small', and a candidate implying it is PRUNED (borrowing (4)), not evaluated."""
    from catspace.incubator.analyst.discovery import object_miner as OM
    reader = GC.TermReader()
    defs = TL.Defs()
    tests = {t.name: t for t in OM.tests_of(dom, harvested, {})}
    sc = OM.Scorer(dom, reader, harvested, defs)

    def cand(s):
        t = TL.parse(s, dom, defs)
        return OM.Cand(t, (), t.size, t.rels(dom, defs))
    assert sc.cites(cand("(via (closure (step ? ? ?)) (closure (step ? ? ?)))"), tests["readings slide"])
    assert not sc.cites(cand("(via (slide ? ? ?) (slide ? ? ?))"), tests["readings slide"])
    short = "(via (step ? ? ?) (step ? ? ?))"
    assert not sc.cites(cand(short), tests["readings slide"])
    before = sc.pruned
    assert not sc.cites(cand(f"(restrict {short} (occupied ?))"), tests["readings slide"]) and sc.pruned == before + 1


def _mined():
    store = ON.load(force=True)
    dom, defs, ts = ON.mined_terms(store, GC.program())
    return dom, defs, ts, {n: e for n, e in store.items() if e["origin"] == "MINED"}


def test_b_the_three_rediscoveries_on_fixed_positions():
    """THE THREE TEST CASES, by the shelf's mined entries: LINE equals python-chess's squares between two squares on one line; BLOCKED a line with an occupied
    square between; REGION the 73.5 atom's region -- on the fixed positions, each side read for (the region but where their king is in check along a line with
    a free square behind it: the mined region reads our attack with their king standing, the 73.5 atom with it lifted -- (a)'s fixture pins the two apart)."""
    dom, defs, ts, ents = _mined()
    assert {"line", "blocked", "region"} <= set(ts)
    reader = GC.TermReader()
    line = _ref_line()
    for f in FENS + (INTERPOSE,):
        b = chess.Board(f)
        for side in (chess.WHITE, chess.BLACK):
            ctx = TL.Ctx(dom, reader, b, GC.COLOUR[side], defs)
            assert frozenset((r[-3], r[-2], r[-1]) for r in ctx.ev(ts["line"])) == line
            assert frozenset((r[-2], r[-1]) for r in ctx.ev(ts["blocked"])) == _ref_blocked(b), f
            them_checked = b.turn != side and b.is_check()
            if b.king(not side) is not None and not them_checked:
                assert frozenset(r[0] for r in ctx.ev(ts["region"])) == frozenset(chess.scan_forward(TM.region(b, side))), (f, side)


def test_b_the_mined_kinds_are_shelf_entries_and_node_kinds_of_a_later_feature_set():
    """An accepted relation is an ontology entry (OBJECT, origin MINED, its term, its reads the term's, its evidence) and a NODE KIND at the feature-set version
    it enters at (3): the graph network's version-0 kinds are unchanged (no model sees a node it was not trained with; nothing retrained)."""
    from catspace.incubator.analyst.discovery import graph_policy as GP
    dom, defs, ts, ents = _mined()
    assert ents and ON.mined_problems(ON.load(), GC.program()) == []
    for n, e in ents.items():
        assert e["kind"] == "OBJECT" and e["node"]["since"] == 3 and e["evidence"]["uses"] and ts[n] is not None
        assert e["reads"] == TL.base_reads(ts[n], dom, ON.load(), defs)
    assert ON.node_kinds(0) == list(GP.NODE_KINDS) == ["square", "piece", "question", "root", "route"]
    assert set(ON.node_kinds(3)) - set(ON.node_kinds(0)) == {e["node"]["kind"] for e in ents.values()}


# ---------------------------------------------------------------------------------------------------------------- (c) material as a mined valuation

def test_c_the_material_term_is_the_hand_reading_with_the_hand_worths(dom):
    from catspace.incubator.analyst.discovery import lemma_store as LS
    from catspace.incubator.analyst.discovery import valuation_miner as VM
    hand = GC.valuation_weights(LS.goal_kinds().valuation)
    t = VM.material_term(dom, "material", hand)
    reader = GC.TermReader()
    for f in FENS:
        b = chess.Board(f)
        for side in (chess.WHITE, chess.BLACK):
            assert TL.evaluate(t, dom, reader, b, GC.COLOUR[side]) == TM.material(b, side) - TM.material(b, not side)


def test_c_the_priced_search_recovers_the_worths_outcomes_were_drawn_from():
    """A FIXTURE: outcomes drawn from the ordinal model with worths (2, 1) on two kinds (and a third kind constant, as the royal unit is): the continuous fit
    and the priced search (integers first, the paired one-standard-error rule) find (2, 1); the constant kind takes no weight."""
    import numpy as np
    from catspace.incubator.analyst.discovery import valuation_miner as VM
    rng = np.random.default_rng(764)
    rows = []
    for g in range(3000):
        for _k in range(6):
            dx, dy = int(rng.integers(-2, 3)), int(rng.integers(-3, 4))
            pl, pd, pw = VM._probs(np.array([2.0 * dx + dy]), 0.6, 0.4)
            u = rng.random()
            o = 0.0 if u < pl[0] else 0.5 if u < pl[0] + pd[0] else 1.0
            rows.append((g, (0, dx, dy), o))
    C = VM.Corpus(["k", "x", "y"], rows)
    assert C.constant == ["k"] and C.kinds == ["x", "y"]
    got = VM.search(C, log=lambda *_a: None)
    assert got["choice"]["worth"] == {"x": 2, "y": 1}
    assert abs(got["beta"]["x"] / got["beta"]["y"] - 2) < 0.2


def test_c_the_mined_valuation_stands_beside_the_hand_one():
    """The store holds the MINED valuation (its worths fit to outcomes, its term, the exchange law over it as a stated obligation) beside the HAND one; the GOAL
    entries count in the hand one -- the agenda reads it, no decision moved."""
    from catspace.incubator.analyst.discovery import lemma_store as LS
    e = LS.load(force=True)["material (mined)"]
    assert e["kind"] == "VALUATION" and e["origin"] == "MINED" and LS.check_entry(e) == []
    assert set(e["params"]) == {"worth", "unit", "royal", "term", "obligation"} and "OBLIGATION" in e["params"]["obligation"] and "NOT proved" in e["params"]["obligation"]
    assert LS.goal_kinds().valuation.name == "material" and LS.load()["material"]["origin"] == "HAND"
    dom = TL.Domain(GC.program(), ON.load())                 # its term reads, in the term language, its own worths' material difference
    t = TL.parse(e["params"]["term"], dom, valuations={LS.slug(e["name"]): GC.valuation_weights(LS.Valuation(e))})
    b = chess.Board(FENS[0])
    worth = lambda side: sum(e["params"]["worth"][x.symbol().lower()] for x in b.piece_map().values() if x.color == side)
    assert TL.evaluate(t, dom, GC.TermReader(), b, "white") == worth(chess.WHITE) - worth(chess.BLACK)
