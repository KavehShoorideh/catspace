"""plan 76.1 THE ONTOLOGY SHELF and the GDL EXTENSIONS: every object and relation kind declared once (catspace/incubator/analyst/discovery/lemmas/ontology/, ontology.py)
with its bindings -- the spec's relation (rules/chess.gdl), the Rust core's, the Lean definition, the kernel predicate -- and the bindings agree where they can be
evaluated; the names table read from the shelf."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import chess
import pytest

from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import gdl_chess as GC
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import names as NM
from catspace.incubator.analyst.discovery import ontology as ON
from catspace.incubator.analyst.discovery.game import CHESS, Record

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean" / "ideas"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"plan761_{name}", ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _lake() -> str:
    lake = shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")
    if not Path(lake).exists():
        pytest.skip("no lake")
    return lake


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


# ---------------------------------------------------------------------------------------------------------------- (b) the five named extensions

def test_b1_the_spec_is_sorted_and_type_checks():
    """Extension (1): every relation of the spec has a signature, every clause keeps it; the clock's numerals and the ranks' are two sorts."""
    prog = GC.program()
    assert prog.typecheck() == []
    assert set(prog.sorts) == {"file", "rank", "side", "kind", "direction", "wing", "parity", "event", "clock", "seen", "score", "move"}
    assert all(prog.signature(r) is not None for r in prog.nodes)
    assert "1" in prog.sorts["rank"] and "1" in prog.sorts["clock"] and "58" in prog.sorts["clock"] and "58" not in prog.sorts["rank"]
    assert ("move", "file", "rank", "file", "rank") in prog.sorts["move"] and "noop" in prog.sorts["move"]


def test_b1_the_type_check_fires_on_a_wrong_spec():
    """The negative control of the sorts: a clock numeral in a rank slot, a relation retyped against its clauses."""
    text = GC.SPEC.read_text()
    assert "(pawn_start white 2)" in text and "(signature check side)" in text
    bad = GDL.Program(text.replace("(pawn_start white 2)", "(pawn_start white 58)"), "mutant").typecheck()
    assert any("58 is no member of the sort rank" in x for x in bad)
    assert GDL.Program(text.replace("(signature check side)", "(signature check kind)"), "mutant").typecheck()


def test_b2_b3_the_translator_s_binding_table_is_the_shelf():
    """Extensions (2), (3), (5): the fluents, the inputs and the move terms are read through the shelf's `bind` entries."""
    GL = _script("gdl_to_lean")
    sorts, rels = GL.binding_table()
    assert set(rels) == {"true.cell", "true.control", "true.castling_right", "true.ep_square", "true.halfmove", "does", "record"}
    assert "Ideas.Spec.Binding.right" in rels["true.castling_right"][0] and "Ideas.Spec.Binding.clock" in rels["true.halfmove"][0]
    assert rels["does"][1] == ("p", "mv") and rels["record"][1] == ("p", "rec")
    assert sorts["move"]["type"] == "Ideas.Spec.Binding.MoveTerm" and set(sorts["move"]["terms"]) == {"move", "promote", "noop"}


def test_b4_the_generated_spec_is_current_and_every_relation_translates(tmp_path, capsys):
    """The committed Ideas/Spec/Chess.lean is what the translator writes today; the survey translates every relation of the spec (76.0: 48 of 90)."""
    out = tmp_path / "Chess.lean"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "gdl_to_lean.py"), "spec", "--out", str(out)], check=True, capture_output=True,
                   env=dict(os.environ, PYTHONPATH=str(ROOT)))
    assert out.read_text() == (LEAN / "Ideas" / "Spec" / "Chess.lean").read_text()
    GL = _script("gdl_to_lean")
    GL.survey()
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("OBS gdl_to_lean survey:"))
    assert "not translated 0" in line


def test_b4_the_spec_s_next_state_is_the_kernels_on_the_corners():
    """The next state the spec's `next` relations give (the interpreter; `applyMoveSpec` assembles the same relations in Lean) is python-chess's push and the Rust
    core's on every rule corner, for every legal move."""
    CP = _script("crossplay")
    n = 0
    for fen in CP.corner_fens():
        if not chess.Board(fen).is_valid():
            continue
        res = CP.compare(fen)
        assert not [d for d in res["disagree"] if d[1] in ("next", "legal")], (fen, res["disagree"][:2])
        n += res["n_next"]
    assert n > 200


def test_b5_threefold_is_the_game_record_s_input():
    """Extension (5): the spec's draw by repetition reads the input `record` -- with the position's identity twice in the game record before it the draw holds,
    as Game.terminal's 'repetition'; once, not; no record, not."""
    b = chess.Board("8/8/8/4k3/8/8/8/KR6 w - - 4 60")
    st = GC.state_of(b)
    for n, expect in ((2, True), (1, False), (0, False)):
        rec = Record()
        if n:
            rec.add(CHESS.position_key(b), n)
        r = GC.Reading(st, inputs=GC.inputs_of(b, rec))
        assert ("draw_by_repetition" in r.reasons) is expect
        assert (CHESS.terminal(b, rec) == "repetition") is expect
        assert r.reasons == GC.kernel_reasons(b, rec)
    assert "draw_by_repetition" not in GC.Reading(st).reasons


def test_b5_the_dead_position_is_a_proved_lemma_of_the_store():
    """Extension (5): the dead position (FIDE 5.2.2) is not a clause; its one-colour-bishops case is a PROVED store entry citing theorems of the index."""
    e = LS.get("dead position (one-colour bishops)")
    assert e is not None and e["status"] == "PROVED" and e["kind"] == "CLOSING"
    idx = LS.lean_index()
    assert "Ideas.DeadPosition.one_colour_dead" in e["lean"] and all(idx.get(t) == "theorem" for t in e["lean"])


@pytest.mark.slow
def test_b4_b5_the_lean_proofs_build_with_no_sorry():
    """SLOW (a Lean build, ~10 s with the library built): the generated spec, the binding, the next-state proofs and the dead position build; the main theorems
    rest on no sorry and no native_decide (their axioms: propext, Classical.choice, Quot.sound)."""
    lake = _lake()
    for f in ("Binding.lean", "NextEquiv.lean", "DeadPosition.lean"):
        text = (LEAN / "Ideas" / "Spec" / f).read_text()
        assert "native_decide" not in text.replace("no `native_decide`", "") and "sorry" not in text.replace("No `sorry`", "")
    r = subprocess.run([lake, "build", "Ideas.Spec.NextEquiv", "Ideas.Spec.DeadPosition"], cwd=LEAN, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    thms = ["Ideas.NextEquiv.applyMoveSpec_applyMove", "Ideas.NextEquiv.applyMoveSpec_unique", "Ideas.NextEquiv.nextCell_functional",
            "Ideas.NextEquiv.moveOf_termOf", "Ideas.NextEquiv.drawByRepetition_kernel", "Ideas.DeadPosition.one_colour_dead",
            "Ideas.DeadPosition.dead_of_spec_draw"]
    src = "import Ideas.Spec.NextEquiv\nimport Ideas.Spec.DeadPosition\n" + "".join(f"#print axioms {t}\n" for t in thms)
    path = LEAN / ".plan761_axioms.lean"
    try:
        path.write_text(src)
        r = subprocess.run([lake, "env", "lean", str(path)], cwd=LEAN, capture_output=True, text=True)
    finally:
        path.unlink(missing_ok=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    lines = [x for x in r.stdout.splitlines() if "depends on axioms" in x]
    assert len(lines) == len(thms) and not any("sorryAx" in x or "ofReduceBool" in x for x in lines), lines


@pytest.mark.slow
def test_b4_the_next_state_is_the_lean_kernels_on_the_corners():
    """SLOW (one `lake env lean`): on the rule corners the spec's next state after every legal move is the chess-lean kernel's `applyMove` (read by `#eval`),
    and its legal moves are the kernel's `legalMoves`."""
    CP = _script("crossplay")
    fens = [f for f in CP.corner_fens() if chess.Board(f).is_valid()]
    res = ON.agreement(fens=fens, lean=True)
    assert res["compared"]["next"].get("lean") == len(fens)
    assert not [d for d in res["disagree"] if d[0] == "lean"], res["disagree"][:3]


# ---------------------------------------------------------------------------------------------------------------- (c) the terms are the relations

def test_c_every_vocabulary_primitive_is_bound_to_the_shelf():
    """Every primitive of the base vocabulary (base_vocabulary.PRIMITIVES: the symbols of graph.py's words and of the Rust compile_term) that reads the position or the
    board's geometry is bound to shelf relations; every bound symbol is a primitive or a reading of the technique library."""
    from catspace.incubator.analyst.discovery import base_vocabulary as BV
    assert ON.vocabulary_problems() == []
    vm = ON.vocabulary_map()
    store = ON.load()
    assert all(r in store for rs in vm.values() for r in rs)
    for s, rels in {"attacked": ["attack"], "reach": ["king_step"], "W": ["cell"], "halfmoves": ["halfmove"], "seen": ["record"], "moves": ["legal"],
                    "see": ["capture", "material"], "knight1": ["knight_jump"]}.items():
        assert vm[s] == rels, s
    assert all(t in BV.PRIMITIVES or t in ON.READINGS for t in vm)


def test_c_every_atom_reads_shelf_relations():
    """technique_measures.py's atoms (and every candidate of its library) read named relations of the shelf; the nine atoms of plan 73.5 as written in its
    docstring: the region reads where their king stands, its walk and our attack; the material atoms the material; the tempo the clock."""
    from catspace.incubator.analyst.discovery import technique_measures as TM
    store = ON.load()
    for a in TM.ATOMS + TM.AXIS_ATOMS:
        assert a.rels and all(r in store for r in a.rels), a.key
    assert all(t.rels for t in TM.library())
    by = {a.key: a.rels for a in TM.ATOMS}
    assert by["region"] == ("attack", "cell", "king_step") and by["edge_dist"] == ("cell", "king_step") and by["kings_dist"] == ("cell", "king_step")
    assert by["material_us"] == ("material",) and by["tempo"] == ("halfmove",) and TM.ATOM_BY_KEY["turn"].rels == ("tempo",)
    with pytest.raises(ON.OntologyError):
        ON.term_relations("(count (flurb L))")


# ---------------------------------------------------------------------------------------------------------------- (d) the GNN reads the shelf; the Datalog's names

def test_d_the_graph_network_s_node_kinds_are_the_shelf_s():
    """graph_policy.py's node types are the shelf's OBJECT kinds that declare a node (the square, the unit as `piece`, the question, the root, the route), its board
    edges the shelf's relations that declare an edge; THE FEATURE-VERSION CHECK: every kind enters at version 0 and the version-0 graph keeps its widths."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from catspace.incubator.analyst.discovery import graph_policy as GP
    assert list(GP.NODE_KINDS) == ON.node_kinds() == ON.node_kinds(0) == ["square", "piece", "question", "root", "route"]
    assert (GP.SQUARE, GP.PIECE, GP.QUESTION, GP.ROOT, GP.ROUTE) == ("square", "piece", "question", "root", "route")
    assert (GP.ON_EDGE, GP.ATTACKS_EDGE, GP.ADJ_EDGE) == ("on", "attacks", "adj") == (ON.edge_of("cell"), ON.edge_of("attack"), ON.edge_of("king_step"))
    b = chess.Board("8/1Rb3kp/p2q2p1/2p1pp2/pQ6/1bP4P/1P3PPK/8 w - - 0 35")
    qs = [dict(goal="win", stake=3, depth=0, target=chess.B3, status="open")]
    g = GP.build(b, True, qs, None, routes=[dict(q=0, uci="b4a5", why="threat", cost=3, order=0)])
    assert set(g.node_types) == set(ON.node_kinds(0))
    assert {(a, r, c) for a, r, c in g.edge_types if not r.startswith("rev_")} == {
        ("piece", "on", "square"), ("piece", "attacks", "square"), ("square", "adj", "square"), ("question", "targets", "square"), ("question", "about", "piece"),
        ("root", "sees", "question"), ("route", "of", "question"), ("route", "from", "square"), ("route", "to", "square")}
    widths = {t: g[t].x.shape[1] for t in g.node_types}
    assert widths == {"square": 6, "piece": 11, "question": len(GP.GOALS) + len(GP.STATUSES) + len(GP.WORDS) + 4, "root": 6, "route": len(GP.RWORDS) + 8}
    assert GP._train_version() == 0


def test_d_the_rust_datalog_s_relations_are_shelf_entries():
    """rules.rs's Datalog (the threat layer's prefilter) names its relations by the shelf: `cell` and `attack` (renamed from piece / attacks), and its own derived
    `defended`, `gain`, `best` -- each a shelf entry whose Rust binding is that relation."""
    import re
    src = (ROOT / "rust" / "analyst_core" / "src" / "rules.rs").read_text()
    block = src[src.index("ascent! {"):src.index("}", src.index("ascent! {"))]
    rels = re.findall(r"relation (\w+)\(", block)
    assert rels == ["cell", "attack", "defended", "gain", "best"]
    store = ON.load()
    for r in rels:
        assert store[r]["rust"] == f"rules.rs:{r}", r


# ---------------------------------------------------------------------------------------------------------------- (e) the Rust fix the surprises pinned

def test_e_the_rust_insufficient_material_is_the_kernels_on_the_nine_surprises():
    """game.rs `insufficient` is the kernel's rule (shakmaty's is_insufficient_material = python-chess's = the spec's `draw`): the nine surprises of plan 76.0 (the
    one-colour bishops the Rust core missed) are agreements now, every reading of both kernels at each, and each case says so."""
    import analyst_core as AC
    CP = _script("crossplay")
    cases = [c for c in CP.load_surprises() if c["kernel"] == "rust" and c["class"] == "draw"]
    assert len(cases) == 9
    for c in cases:
        assert c["expect"] == "agree" and "PATCHED by plan 76.1 (e)" in c["change"], c["file"]
        assert CP.replay(c) == [], c["file"]
        b = chess.Board(c["fen"])
        reads = dict(AC.game_draw_check(c["fen"], []))
        for m in b.legal_moves:
            s = b.copy(stack=False)
            s.push(m)
            if s.is_insufficient_material() and s.halfmove_clock < 100:
                assert reads.get(m.uci()) == "insufficient material", (c["file"], m.uci())
    # the near misses stay sufficient: bishops on both colours, a bishop and a knight
    for fen in ("8/8/8/4k3/8/2b5/8/KB6 w - - 0 1", "8/8/3k4/8/8/8/1B6/KB6 w - - 0 1", "8/8/8/4k3/8/8/3n4/KB6 w - - 0 1"):
        assert not [w for _u, w in AC.game_draw_check(fen, []) if w == "insufficient material"], fen
