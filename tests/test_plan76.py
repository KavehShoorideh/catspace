"""plan 76.1 THE ONTOLOGY SHELF and the GDL EXTENSIONS: every object and relation kind declared once (catspace/incubator/analyst/discovery/lemmas/ontology/, ontology.py)
with its bindings -- the spec's relation (rules/chess.gdl), the Rust core's, the Lean definition, the kernel predicate -- and the bindings agree where they can be
evaluated; the names table read from the shelf."""
from __future__ import annotations

import json

import chess
import pytest

from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import gdl_chess as GC
from catspace.incubator.analyst.discovery import names as NM
from catspace.incubator.analyst.discovery import ontology as ON


# ---------------------------------------------------------------------------------------------------------------- (a) the shelf

def test_a_the_shelf_loads_and_validates():
    """Every entry keeps the schema; every relation of the spec is bound by an entry (none unnamed); every binding resolves; the stored reads are the spec's; the
    bindings agree on the sample (python-chess and the Rust core: the known Rust draw cases of rules/surprises/ aside)."""
    lines = []
    assert ON.validate(lean=False, log=lines.append, file=False) == 0, "\n".join(lines)
    store = ON.load()
    _covered, unnamed = ON.coverage(store, ON.program())
    assert unnamed == []
    assert {e["kind"] for e in store.values()} == {"OBJECT", "RELATION"}


def test_a_the_names_table_is_read_from_the_shelf():
    """names.py's table is the shelf's entries marked `names` (the same words, kernel predicates, Lean twins, sentences), in the table's order."""
    shelf = {e["name"]: e for e in ON.names_table()}
    assert list(NM.NAMES) == list(NM.WORDS) and set(shelf) == set(NM.WORDS)
    for w, n in NM.NAMES.items():
        assert (n.kernel, n.lean, n.means) == (shelf[w]["python"], shelf[w]["lean"], shelf[w]["means"])
    assert NM.problems() == []
    assert NM.EVENT_CAPTURE == "capture" and NM.EVENT_PAWN == "pawn" and NM.DRAW == "draw"


def test_a_each_entry_has_one_name_and_the_spec_s_relations_one_entry():
    """No name is invented: an entry is a word of the names table or a relation of the spec by its own name (or an object kind of the plan: unit, side, board,
    clock, the spec's sorts, the machinery's question / root / route, the Rust Datalog's own relations)."""
    prog = ON.program()
    store = ON.load()
    owners = {}
    for e in store.values():
        for r in ON.covers(e, prog.nodes):
            owners.setdefault(r, []).append(e["name"])
    assert all(len(v) == 1 for v in owners.values()), {r: v for r, v in owners.items() if len(v) > 1}
    for e in store.values():
        g = e["gdl"]
        if e["names"] or e["kind"] == "OBJECT" or g is None or "keyword" in g:
            continue
        assert e["name"] == (g.get("relation") or g.get("fluent")), e["name"]


def test_a_the_agreement_fires_on_a_wrong_spec():
    """The negative control of the agreement: a spec whose bishop also slides on the files and ranks disagrees with python-chess and with the Rust core on the
    attacks, the legal moves and the next states (the comparison is not vacuous)."""
    text = GC.SPEC.read_text()
    assert text.count("(<= (line_of bishop ?d) (diagonal ?d))") == 1
    wrong = GDL.Program(text.replace("(<= (line_of bishop ?d) (diagonal ?d))", "(<= (line_of bishop ?d) (diagonal ?d))\n(<= (line_of bishop ?d) (orthogonal ?d))"), "mutant")
    fen = "r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1"
    res = ON.agreement([fen], prog=wrong)
    got = {(k, r) for k, r, _s, _f, _d in res["disagree"]}
    assert {("python", "attack"), ("rust", "attack"), ("python", "legal"), ("rust", "legal")} <= got
    assert not ON.agreement([fen])["disagree"]


def test_a_the_shelf_rejects_a_bad_entry(tmp_path):
    """The schema: a null binding needs its reason; a node only on an object; an unknown kind."""
    e = json.loads((ON.DIR / "attack.json").read_text())
    assert ON.check_entry(e) == []
    bad = dict(e, rust=None)
    assert any("null without a reason" in x for x in ON.check_entry(bad))
    assert any("node" in x for x in ON.check_entry(dict(e, node={"kind": "x", "order": 9})))
    assert any("kind" in x for x in ON.check_entry(dict(e, kind="THING")))


@pytest.mark.slow
def test_a_three_bindings_agree_with_the_lean_kernel():
    """SLOW (one `lake env lean` of ~20 s): the chess-lean kernel's decidable definitions (`parseFen`, `pieceAttacksB`, `isAttackedB`, `inCheckB`, `legalMoves`,
    `applyMove` / `printFen`, `moveIsCapture`, `checkmateB`, `stalemateB`) read the sample as the spec does: no disagreement with the Lean kernel."""
    res = ON.agreement(lean=True)
    assert all(ks.get("lean") == len(ON.SAMPLE) for r, ks in res["compared"].items() if r in ON.LEAN_RELATIONS)
    assert not [d for d in res["disagree"] if d[0] == "lean"], res["disagree"][:3]
    known = ON.surprise_keys()
    assert all((k, r, s) in known for k, r, s, _f, _d in res["disagree"])
