"""PLAN 73.6 THE PROVER (docs/ANALYST_PLAN.md 73.6): the engine states a mined technique's obligation in Lean and off-the-shelf tools search for its proof.
(1) THE ATOMS: the per-axis distances and whose move are vocabulary terms reading only base words (movement, tempo), the same readings for every unit and pair;
their values are their terms' (the king-step distance to the term's set) and the vocabulary's (the Rust evaluator over `bv`); the 73.6 library holds them and
leaves the class's constants out, 73.5b's library is the record; the miner runs with it; the entry records the re-mine without changing its status.
(2) THE STATEMENT: the emitted files are the emitter's output; hprog has the shape of forced_by_measure's premise with one sorry; the file BUILDS with exactly
one sorry (the shared .lake: a cached build, seconds); the measure's Lean atoms read the miner's numbers."""
import random
import re

import chess

from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import technique_corpus as TC
from catspace.incubator.analyst.discovery import technique_measures as TM
from catspace.incubator.analyst.discovery import technique_miner as MN
from catspace.incubator.analyst.discovery import technique_sample as TS


def test_the_axis_atoms_read_only_base_words():
    assert len(TM.AXIS_ATOMS) == 3 * 2 + 3 * 2 + 1                               # every unit's two edge distances, every pair's two, whose move
    for a in TM.AXIS_ATOMS:
        r = LS.term_reads(a.sexpr)
        assert r and r <= set(LS.BASE), a.sexpr
        assert r == ({"tempo"} if a.key == "turn" else {"movement"}), (a.sexpr, r)
        assert not re.search(r"(rook|queen|bishop|knight|pawn|KRvK|KQvK|opposition)", a.sexpr + a.key + a.gloss, re.I), a.sexpr
    units = {u for u, _g in TM.UNITS}
    shapes = {re.sub(r"\b[WLP]\b", "X", a.sexpr) for a in TM.AXIS_ATOMS}         # the same readings for every unit and every pair
    assert shapes == {"(dist X (or (rank a1) (rank a8)))", "(dist X (or (file a1) (file h1)))", "(dist X (rank X))", "(dist X (file X))", "turn"}
    assert units == {"W", "L", "P"}


def test_the_axis_atoms_read_as_their_terms_and_the_vocabulary():
    """Each per-axis value is its term's (technique_measures.dist_to over the set the term names) and, where the vocabulary says it (`bv`), the Rust
    evaluator's, for the winning side and on the colour-mirrored board for the other colour."""
    BV.install_terms()
    edge_r, edge_f = chess.BB_RANK_1 | chess.BB_RANK_8, chess.BB_FILE_A | chess.BB_FILE_H
    rng = random.Random(5)
    n = 0
    for sig in ("KRvK", "KQvK"):
        for _ in range(25):
            b = TC.random_start(rng, sig, 1)
            for side, bb in ((True, b), (False, b.mirror())):
                v = TM.atom_values(bb, side)
                sq = dict(W=bb.king(side), L=bb.king(not side), P=TM.other_unit(bb, side))
                for u in "WLP":
                    assert v[f"{u}_rank_edge"] == TM.dist_to(sq[u], edge_r) and v[f"{u}_file_edge"] == TM.dist_to(sq[u], edge_f)
                for u, w in (("W", "L"), ("W", "P"), ("L", "P")):
                    assert v[f"{u}{w}_rank"] == TM.dist_to(sq[u], chess.BB_RANKS[chess.square_rank(sq[w])])
                    assert v[f"{u}{w}_file"] == TM.dist_to(sq[u], chess.BB_FILES[chess.square_file(sq[w])])
                assert v["turn"] == (0 if bb.turn == side else 1)
                for a in TM.AXIS_ATOMS:
                    w = G.word(bb, BV.parse(a.bv), side) if a.bv else None
                    if w is not None:
                        assert w == v[a.key], (bb.fen(), side, a.key, v[a.key], w)
                        n += 1
            G.clear_caches()
    print(f"OBS 73.6 axis atoms against the vocabulary: {n} reads, every one equal")
    assert n > 500


def test_the_736_library_holds_the_axis_atoms_and_leaves_the_constants_out():
    assert len(MN.ATOMS) == 8 and MN.ATOMS_736[:8] == MN.ATOMS                   # 73.5b's library is the record, unchanged
    assert {len(MN.candidates(r)) for r in (2, 3)} == {8 * 7, 8 * 7 * 6}
    n = len(MN.ATOMS_736)
    assert n == 8 + len(TM.AXIS_ATOMS) and len(MN.candidates(3, MN.ATOMS_736)) == n * (n - 1) * (n - 2)
    assert "halfmoves" not in {a.sexpr for a in MN.ATOMS_736}                    # the clock stays out (73.5b's reason); whose move enters
    assert MN.measure_of("(lex (area (reach L free)) (dist L corner) (dist W (file P)))").parts == ("region_area", "corner_dist", "WP_file")
    assert MN.measure_of("(lex (area (reach L free)) (dist L corner) (dist W L))") is not None     # the entry's term resolves as before
    g = MN.Graph(chess.WHITE)
    rng = random.Random(9)
    for _ in range(40):
        g.add(TC.random_start(rng, "KRvK", 1))
    const = {a.key for a in MN.constant_atoms(g, MN.ATOMS_736)}
    assert {"material_us", "material_them", "turn"} <= const and not const & {"region", "WL_rank", "WL_file", "L_rank_edge"}


def _fixture() -> list:
    """Won K+Q v K positions with a mate in 2 to 4 (73.5b's fixture)."""
    rng = random.Random(21)
    rows = []
    while len(rows) < 16:
        b = TC.random_start(rng, "KQvK", 3)
        if TC.tablebase().probe_dtz(b) <= 7:
            f2 = TS.floor_word(b, 2)
            rows.append(dict(fen=b.fen(), source="fixture", floor2=f2, floor3=f2 or TS.floor_word(b, 3)))
    return rows


def test_the_miner_runs_the_736_library():
    lines = []
    got = MN.mine("KQvK", _fixture(), log=lines.append, library="73.6")
    assert got["library"] == "73.6" and {"material_us", "material_them", "turn"} <= set(got["dropped"])
    assert any("the constants of the class left out" in x for x in lines)
    c = got["choice"]
    assert c is not None and c["measure"].sexpr == c["zero"][0]
    assert any(x.startswith("OBS miner KQvK library 73.6 every zero-failure measure of the chosen level") for x in lines)
    for s in c["zero"]:
        assert MN.measure_of(s).reads <= set(LS.BASE)


def test_the_entry_records_the_remine_and_keeps_its_status():
    e = LS.get("mined technique KRvK")
    assert e["status"] == "CONJECTURE" and e["measure"]["quantity"] == "(lex (area (reach L free)) (dist L corner) (dist W L))"
    line = [x for x in e["evidence"]["confirmed"] if "--library 73.6" in x]
    assert line and "every zero-failure measure there" in line[0] and "the entry's measure is among them" in line[0]


# ---------------------------------------------------------------------------------------------------------------- (2) THE STATEMENT

from pathlib import Path  # noqa: E402

from catspace.incubator.analyst.discovery import technique_lean as TL  # noqa: E402

ENTRY = "mined technique KRvK"


def test_the_emitted_file_is_the_emitters_output_and_states_the_obligation():
    """The committed file is the emitter's output (generated, never hand-edited); `hprog` is the premise of forced_by_measure at k = max(d, n) with one
    `sorry`; the corollary applies the generic theorem; the class comes from the signature and the atoms from the term."""
    e = LS.get(ENTRY)
    text = TL.emit_text(e)
    assert TL.lean_path(e).read_text() == text
    k = max(e["measure"]["depth"], e["measure"]["n"])
    assert f"theorem hprog : ∀ p, inClass p → Ideas.Forced.Reach ours theirs mate (fun r => inClass r ∧ mu r < mu p) {k} p := by\n  sorry\n" in text
    assert len(re.findall(r"^\s*sorry\s*$", text, re.M)) == 1
    assert f"Ideas.Forced.forced_by_measure inClass mu {k} hprog" in text
    src = (TL.LEAN_DIR / "Ideas" / "ForcedByMeasure.lean").read_text()
    assert "(hprog : ∀ p, S p → Reach ours theirs goal (fun r => S r ∧ μ r < μ p) k p)" in src           # the generic premise, the same shape
    assert "p.board P = some ⟨p.turn, .rook⟩" in text and "p.board L = some ⟨Color.opposite p.turn, .king⟩" in text
    assert "def atom1 (p : Position) (c : Color) (W L P O : Square) : Nat := (areaOf (oreach (some L) (fun u => !(attackedBy p c L u))))" in text
    assert f"* {MN.RADIX} + atom2" in text and "def ours (p q : Position) : Prop := ∃ m, m ∈ legalMoves p ∧ q = applyMove p m ∧ ¬ stalemate q ∧ ¬ unitLost q" in text
    assert "WITHOUT the claimable draws" in text                                         # the narrowest faithful version, labelled
    kq = dict(e, name="test technique KQvK", measure=dict(e["measure"], **{"class": "KQvK"}))  # the same code states another class and another term
    t2 = TL.emit_text(kq, "(lex (count (and (reach L free) (not edge))) (dist L corner) (dist W L))")
    assert "p.board P = some ⟨p.turn, .queen⟩" in t2 and "(count (fun u => (oreach (some L)" in t2
    t3 = TL.emit_text(e, "(lex (area (reach L free)) (dist L corner) (dist W (file P)))", "_t")
    assert "(odistSet (some W) (oset fileOf (some P)))" in t3 and "namespace Ideas.Mined.mined_technique_krvk_t" in t3


def test_the_alternative_statements_are_the_emitters_output():
    """Every zero-failure measure of the re-mined level has its statement (`<module>_z<i>.lean`), each the emitter's output."""
    e = LS.get(ENTRY)
    zs = sorted(TL.MINED_DIR.glob(f"{TL.module_name(e)}_z*.lean"), key=lambda p: int(re.search(r"_z(\d+)", p.stem).group(1)))
    assert zs
    for p in zs:
        m = re.search(r"here the ALTERNATIVE measure (\(lex .*?\)) \(a zero-failure", p.read_text())
        assert m, p
        assert TL.emit_text(e, m.group(1), p.stem[len(TL.module_name(e)):]) == p.read_text(), p


def test_the_emitter_names_no_unit_kind():
    t = Path(TL.__file__).read_text()
    t = re.sub(r'"""(?:.|\n)*?"""', "", t)
    code = "\n".join(ln.split("  #")[0] for ln in t.splitlines())
    assert not re.search(r"\b(rook|queen|bishop|knight|ROOK|QUEEN)\b", code)
    assert not re.search(r"['\"](KRvK|KQvK)['\"]", code)


def test_the_statement_builds_with_exactly_one_sorry():
    """The Lean build of the emitted module (the shared .lake: a cached build takes seconds)."""
    b = TL.build(TL.module_name(LS.get(ENTRY)))
    print(f"OBS 73.6 build: ok {b['ok']}, sorry warnings {b['sorries']}, {b['secs']} s")
    assert b["ok"] and b["sorries"] == 1, b["errors"]


def test_the_statement_measure_reads_as_the_miner():
    """Lean's atoms (#eval on the kernel's FEN parse) equal the miner's on positions of the class; the `reach` atom is not evaluated, and says why."""
    e = LS.get(ENTRY)
    c = TL.check_measure(e, TL.class_fens("KRvK", 12, seed=3))
    print("OBS 73.6 measure check: " + "; ".join(f"{n} {s} {eq if why is None else why}" for n, (s, eq, why) in c["atoms"].items()))
    assert not c["diffs"], c["diffs"]
    assert [n for n, (s, eq, why) in c["atoms"].items() if why is None and eq == 12] == ["atom2", "atom3"]
    assert "exponential" in c["atoms"]["atom1"][2]
