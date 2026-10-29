"""plan 76.4c MINING OBJECTS AND MATERIAL (docs/ANALYST_PLAN.md 76.4; 76's borrowings (4) and (5)): (a) one term language with typed combinators over the ontology
shelf's relations (terms.py) -- its typing, its pricing, its implication, the atoms re-expressed in it equal to the old readings, the language naming no game
object."""
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
    seen = _code_strings_and_names(ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "terms.py")
    prog = GC.program()
    rels = {r.split(".", 1)[1] if "." in r else r for r in prog.nodes}
    consts = {m for ms in dom.members.values() for m in ms if len(m) > 1}
    words = (set(NM.WORDS) | {e["name"] for e in ON.load().values()}) - set(TL.SORTS)     # (the language's own sorts, the plan's: location, side, unit, ...)
    hits = sorted(seen & (rels | consts | words))
    assert hits == [], f"terms.py names game objects: {hits}"
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
