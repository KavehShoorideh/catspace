"""PLAN 73.6 THE PROVER (docs/ANALYST_PLAN.md 73.6): the engine states a mined technique's obligation in Lean and off-the-shelf tools search for its proof.
(1) THE ATOMS: the per-axis distances and whose move are vocabulary terms reading only base words (movement, tempo), the same readings for every unit and pair;
their values are their terms' (the king-step distance to the term's set) and the vocabulary's (the Rust evaluator over `bv`); the 73.6 library holds them and
leaves the class's constants out, 73.5b's library is the record; the miner runs with it; the entry records the re-mine without changing its status.
(2) THE STATEMENT: the emitted files are the emitter's output; hprog has the shape of forced_by_measure's premise with one sorry; the file BUILDS with exactly
one sorry (the shared .lake: a cached build, seconds); the measure's Lean atoms read the miner's numbers.
(3) THE HARNESS: the acceptance check rejects native_decide, sorry and a new axiom and accepts a genuine toy proof; the scan; everything local (no
network service but the local model server); the goal, the splice and the sketch helpers; the hint verbatim; the decomposition path end to end with a fake model;
a round with no candidate ends the loop.
(4) THE STORE: a sub-lemma's kind by role and its reads; a Lean theorem implements an entry; the run's sub-goal file compiles with exactly its open
sorries and the index reads it right; an entry is PROVED only when accepted; a run with no result changes nothing.
THE DEMONSTRATION (Kaveh 2026-10-27): his protection lemma stated by code, building with its one sorry; the model shown the imported definitions; the
transcript shows every attempt."""
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
    assert f"theorem conjecture_hprog : ∀ p, inClass p → Ideas.Forced.Reach ours theirs mate (fun r => inClass r ∧ mu r < mu p) {k} p := by\n  sorry\n" in text
    assert len(re.findall(r"^\s*sorry\s*$", text, re.M)) == 1
    assert f"Ideas.Forced.forced_by_measure inClass mu {k} conjecture_hprog" in text and "theorem conjecture_forced_mate" in text    # (open: the repo's name)
    assert "theorem hprog :" in TL.emit_text(e, proof=dict(proof="exact h", tool="t", run="r"))                                      # (proved: the plain name)
    src = (TL.LEAN_DIR / "Ideas" / "Core" / "ForcedByMeasure.lean").read_text()
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


# ---------------------------------------------------------------------------------------------------------------- (3) THE HARNESS

from catspace.incubator.analyst.discovery import prover as PR  # noqa: E402

TOYS = {
    "native_decide": ("theorem toy : 2 + 2 = 4 := by\n  native_decide\n", "native_decide"),
    "sorry": ("theorem toy : 2 + 2 = 5 := by\n  sorry\n", "sorry"),
    "a new axiom": ("axiom toyAx : 2 + 2 = 5\ntheorem toy : 2 + 2 = 5 := by\n  exact toyAx\n", "exact toyAx"),
    "genuine": ("theorem toy (a b : Nat) : a + b = b + a := by\n  exact Nat.add_comm a b\n", "exact Nat.add_comm a b"),
}


def test_the_acceptance_check_rejects_native_decide_sorry_and_a_new_axiom_and_accepts_a_genuine_proof():
    got = {}
    for k, (text, proof) in TOYS.items():
        chk = PR.lean_check(text, ("toy",), "ideas", tag="test_toy_" + re.sub(r"\W", "_", k))
        got[k] = PR.accept(chk, ("toy",), (proof,))
        print(f"OBS 73.6 acceptance {k}: {got[k]}, axioms {chk.axioms}")
    assert got["genuine"] == (True, [])
    assert not got["native_decide"][0] and "banned: native_decide" in got["native_decide"][1]
    assert any("native_decide" in w for w in got["native_decide"][1] if "depends on" in w)          # #print axioms sees it too (an auxiliary axiom)
    assert not got["sorry"][0] and any("sorryAx" in w for w in got["sorry"][1])
    assert not got["a new axiom"][0] and "toy depends on toyAx" in got["a new axiom"][1]


def test_the_scan_bans_enumeration_and_reads_past_comments_and_lemma_names():
    assert PR.scan("simp [decide_eq_true_eq]\n-- native_decide in a comment\nexact Nat.le_refl _") == []
    for bad in ("native_decide", "decide", "simp (config := {decide := true})", "fin_cases s", "interval_cases x", "exact Finset.univ", "sorry", "admit"):
        assert PR.scan(bad), bad


def test_everything_is_local():
    """The owner's rule (2026-10-27): no network service is called for proving -- the one URL the harness opens is the local model server's."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", Path(PR.__file__).read_text())
    urls = set(re.findall(r"https?://[^\s\"']+", src))
    assert PR.OLLAMA.startswith("http://localhost:") and {u for u in urls if "github.com" not in u} == {PR.OLLAMA}     # (github: the tool pins, cloned by setup)
    assert "aristotle" not in src.lower() and src.count("urlopen(") == 1


def test_the_goal_the_splice_and_the_sketch_helpers():
    """The emitted file's one goal (its docstring with it, the corollary's axioms checked too); a model's answer contributes only its proof and its new
    lemmas -- our definitions and statement are kept; a sketch's sorries are filled in order; a sub-goal's binders are read."""
    path = PR.MINED_DIR / "mined_technique_krvk.lean"
    (g,) = PR.goals_of(path)
    assert g.name == "conjecture_hprog" and g.full == "Ideas.Mined.mined_technique_krvk.conjecture_hprog"
    assert g.extra_axioms == ("Ideas.Mined.mined_technique_krvk.conjecture_forced_mate",)
    assert g.stmt.lstrip().startswith("/-- THE OBLIGATION") and g.text("sorry") == path.read_text()
    answer = ("def inClass (p : Position) : Prop := True\n\nlemma helper (n : Nat) : n ≤ n := le_refl n\n\n"
              "theorem conjecture_hprog : ∀ p, inClass p → True := by\n  intro p hp\n  exact helper\n")
    proof, lemmas = PR.take_proof(answer, g, {"inClass", "ours", "theirs", "mu", "conjecture_hprog"})
    assert proof.strip() == "intro p hp\n  exact helper".strip() and "def inClass" not in lemmas and lemmas.startswith("theorem helper")
    assert PR.fill_sorries("have h : A := by sorry\nhave h2 : B := sorry\nsorry", lambda k, f: f"<{k}:{f}>") == "have h : A := <0:by>\nhave h2 : B := <1:term>\n<2:tactic>"
    assert PR.binder_names("theorem s1 (p : Position) (hp : inClass p) {x : Nat} [inst : Inhabited Nat] (h1 : ∃ q, ours p q) :\n  mu p < 1 := by") == ["p", "hp", "h1"]


def test_the_hint_is_the_journal_text_verbatim():
    h = PR.hint_text()
    j = " ".join((Path(PR.paths.REPO_ROOT) / "JOURNAL.md").read_text().split())
    assert h in j and h.startswith('"doing an exhaustive check on all king+rook positions is dumb. The proof is simple.')
    assert h.endswith('"distance from opposition on each rank".') and "i don't want you to write this" not in h


SKETCH = """Plan: split the conjunction.
```lean4
theorem toyg (a b : Nat) (h : a ≤ b) : a < b + 1 ∧ b ≤ b := by
  have h1 : a < b + 1 := by
    sorry
  have h2 : b ≤ b := by sorry
  exact ⟨h1, h2⟩
```"""
TOY_GOAL = "theorem toyg (a b : Nat) (h : a ≤ b) : a < b + 1 ∧ b ≤ b := by\n  sorry\n"


def _fake(answer, calls):
    def gen(prompt, seed, think=False):
        calls.append(prompt)
        return dict(response=answer, thinking="", tokens=0, secs=0.0, prompt_tokens=0, done_reason="stop")
    return gen


def test_the_decomposition_states_the_sketchs_goals_proves_them_and_assembles_the_parent(monkeypatch, tmp_path):
    """SUBGOAL DECOMPOSITION without the model (a fake answer): Lean's extract_goal states each `sorry` of the sketch with its context; the automation proves
    each; the parent is assembled from the proved sub-lemmas and accepted; every sub-goal is recorded with its parent."""
    calls = []
    monkeypatch.setattr(PR, "generate", _fake(SKETCH, calls))
    f = tmp_path / "toy.lean"
    f.write_text(TOY_GOAL)
    (g,) = PR.goals_of(f)
    P = PR.Prover(f, "blind", ("automation", "llm"), PR.Log(tmp_path / "log"), 900)
    r = P.decompose(g, f.read_text())
    assert r is not None and r["tool"].startswith("decomposition") and "exact blind_s1 a b h" in r["proof"]
    assert [n for n, _p, _s in P.proposed] == ["blind_s1", "blind_s2"] and set(P.proved) == {"blind_s1", "blind_s2"}
    assert P.proposed[1][2] == "theorem blind_s2 (a b : Nat) (h : a ≤ b) (h1 : a < b + 1) : b ≤ b := by" and len(calls) == 1


def test_a_round_with_no_candidate_ends_the_goals_loop(monkeypatch, tmp_path):
    """A model that never answers the goal (here: always another theorem) gives Lean nothing to check in a round: the loop ends, it does not spin to the fuse."""
    calls = []
    monkeypatch.setattr(PR, "generate", _fake("```lean4\ntheorem other : True := trivial\n```", calls))
    f = tmp_path / "toy.lean"
    f.write_text(TOY_GOAL)
    (g,) = PR.goals_of(f)
    P = PR.Prover(f, "blind", ("llm",), PR.Log(tmp_path / "log"), 900)
    assert P.llm(g) is None and len(calls) == 2 and "no candidate in a whole round" in P.results["toyg"]["last"]


RESTATED = """```lean4
theorem toyg (a b : Nat) (h : a ≤ b) : a < b + 1 ∧ b ≤ b := by
  have hall : a < b + 1 ∧ b ≤ b := by sorry
  exact hall
```"""


def test_a_sketch_that_restates_the_goal_is_refused_as_no_progress(monkeypatch, tmp_path):
    """A sketch whose `have` is the goal again (the demonstration's first sketch did this) yields a sub-goal from which the goal follows by `apply` and its
    own hypotheses -- Lean decides -- so it is no progress: refused, the model asked to revise; the same refusal twice ends the step."""
    calls = []
    monkeypatch.setattr(PR, "generate", _fake(RESTATED, calls))
    f = tmp_path / "toy.lean"
    f.write_text(TOY_GOAL)
    (g,) = PR.goals_of(f)
    P = PR.Prover(f, "blind", ("automation", "llm"), PR.Log(tmp_path / "log"), 900)
    assert P.decompose(g, f.read_text()) is None and len(calls) == 2 and not P.proposed
    assert "restates the goal" in P.results["toyg"]["last"]
    assert "llm sketch refused" in (tmp_path / "log" / "attempts.jsonl").read_text()


def test_the_no_progress_check_sees_the_blind_runs_restatement(tmp_path):
    """The blind run's first sketch (2026-10-27) wrote `hprog` again as a `have`; extract_goal stated it as below. The check must flag it in the emitted file's
    own context -- it did not in that run: the test theorem was compiled with the goal's suffix, whose corollary names the goal (fixed: only the namespace is
    closed)."""
    (g,) = PR.goals_of(PR.MINED_DIR / "mined_technique_krvk.lean")
    info = ("theorem Ideas.Mined.mined_technique_krvk.hprog.extracted_1 (p : Position) (hp : inClass p) :\n"
            "  Forced.Reach ours theirs mate (fun r => inClass r ∧ mu r < mu p) 3 p := sorry")
    P = PR.Prover(PR.MINED_DIR / "mined_technique_krvk.lean", "blind", ("llm",), PR.Log(tmp_path / "log"), 600)
    assert P.restating(g, "", [info]) == "#1"


# ---------------------------------------------------------------------------------------------------------------- (4) THE STORE

def test_a_sub_lemma_entry_by_role_and_reads():
    s_force = "theorem blind_s1 (p : Position) (hp : inClass p) : Ideas.Forced.Reach ours theirs mate (fun r => inClass r ∧ mu r < mu p) 1 p := by"
    s_keep = "theorem blind_s2 (p q : Position) (h : ours p q) : ¬ unitLost q := by"
    s_mate = "theorem blind_s3 (q : Position) (h : inClass q) : ¬ checkmate q := by"
    assert (PR.role(s_force), PR.role(s_keep), PR.role(s_mate)) == ("FORCING", "DEFECT_FIX", "CLOSING")
    assert PR.reads_of(s_force, ("attack", "movement")) == ["attack", "movement", "tempo"]
    assert PR.reads_of(s_keep, ()) == ["attack", "material", "movement", "tempo"]
    assert PR.conclusion(s_keep) == "¬ unitLost q"
    e = dict(name="mined lemma test", kind=PR.role(s_keep), reads=PR.reads_of(s_keep, ()), terms=[], status="CONJECTURE", lean=[], origin="MINED", plan=["73.6"],
             journal="2026-10-27", measure=None, evidence=dict(uses={}, confirmed=[], refuted=[]),
             implementation=["lean/ideas/Ideas/Mined/mined_technique_krvk.lean:conjecture_hprog"], note="a test entry")
    assert LS.check_entry(e) == [] and LS.implementation_problems({e["name"]: e}) == []          # a Lean theorem implements an entry


NS = "Ideas.Mined.mined_technique_krvk"
OPEN_STMT = "theorem blind_s8 (p : Position) (hp : inClass p) : ∃ q, ours p q := by"
PROVED = {"blind_s9": dict(statement="theorem blind_s9 (n : Nat) : n + 0 = n := by", proof="omega", tool="automation: omega", parent="hprog",
                           text="theorem blind_s9 (n : Nat) : n + 0 = n := by\n  omega\n")}           # a toy: the pipeline, not a chess fact
OPEN = {"blind_s8": dict(statement=OPEN_STMT, last="depends on sorryAx (a toy last state)", parent="hprog")}


def test_the_runs_sub_goal_file_compiles_and_the_index_reads_it():
    """The run's file (prover.sub_file_text): the open sub-goal with `sorry` and its last state, the proved one with its proof; it compiles against the emitted
    module with exactly the open sorry; the index reads the proved theorem as proved and the open one as OPEN (the open ones are written first)."""
    from catspace.incubator.analyst import lean_index as LI
    text = PR.sub_file_text("mined_technique_krvk", NS, "run0", "blind", PROVED, OPEN)
    chk = PR.lean_check(text, (f"{NS}.blind_s9",), "ideas", tag="test_subfile")
    assert chk.ok and chk.sorries == 1, chk.errors                                       # (the open sub-goal's sorry; record checks each proved one on the built module)
    assert set(chk.axioms[f"{NS}.blind_s9"]) <= set(PR.ALLOWED_AXIOMS) and PR.scan("omega") == []
    f = Path(PR.paths.experiment("prover", "scratch", "test_subfile_index.lean"))
    f.write_text(text)
    got = {d["name"]: d["open"] for d in LI._decls(f) if d["kind"] == "theorem"}
    assert got == {"conjecture_blind_s8": True, "blind_s9": False}                        # (an open one under the repo's name for it)


def test_a_sub_goals_entry_is_proved_only_when_accepted():
    p = PR.sub_entry("blind_s9", PROVED["blind_s9"], NS, "mined_technique_krvk_blind", "run0", "blind", ENTRY, ("attack", "movement"), "2026-10-27", accepted=True)
    r = PR.sub_entry("blind_s9", PROVED["blind_s9"], NS, "mined_technique_krvk_blind", "run0", "blind", ENTRY, ("attack", "movement"), "2026-10-27",
                     accepted=False, why=("banned: decide",))
    o = PR.sub_entry("blind_s8", OPEN["blind_s8"], NS, "mined_technique_krvk_blind", "run0", "blind", ENTRY, ("attack", "movement"), "2026-10-27", accepted=None)
    for e in (p, r, o):
        assert LS.check_entry(e) == [] and e["origin"] == "MINED" and e["implementation"][0].startswith("lean/ideas/Ideas/Mined/mined_technique_krvk_blind.lean:")
    assert (p["status"], p["lean"]) == ("PROVED", [f"{NS}.blind_s9"]) and (r["status"], r["lean"]) == ("CONJECTURE", [])
    assert "REJECTED it: banned: decide" in r["evidence"]["confirmed"][0]
    assert o["status"] == "CONJECTURE" and "the prover's last state: depends on sorryAx" in o["note"] and o["kind"] == "DEFECT_FIX"


def test_a_tautology_or_a_restatement_is_not_a_word():
    """The demonstration's six sub-goals were three tautologies and three restatements of the lemma: Lean decides which a sub-lemma is (`prover.no_word`), in
    the built module's context, on toys here."""
    taut = "theorem t1 (n : Nat) (h : n = 1) : n = 1 := by"
    rest = "theorem t2 (n m : Nat) (h : n ≤ m) : n ≤ m + 1 := by"
    word = "theorem t3 (n : Nat) : n ≤ n + 1 := by"
    top = "theorem top : ∀ n m : Nat, n ≤ m → n ≤ m + 1 := by"
    assert "tautology" in PR.no_word("mined_technique_krvk", NS, taut, [top])
    assert "restatement" in PR.no_word("mined_technique_krvk", NS, rest, [top])
    assert PR.no_word("mined_technique_krvk", NS, word, [top]) is None
    assert PR.closed_prop(rest) == "∀ (n m : Nat) (h : n ≤ m), n ≤ m + 1"
    proposed = [dict(name="a", parent="top", statement=rest), dict(name="b", parent="a", statement=word)]
    assert PR.ancestors_of("b", proposed, top) == [rest, top]


def test_a_run_with_no_result_changes_nothing(tmp_path):
    """Nothing is upgraded without the check: a run that closed nothing and proposed nothing writes no entry and no file."""
    before = {n: dict(e) for n, e in LS.load(force=True).items()}
    s = tmp_path / "summary.json"
    s.write_text(PR.json.dumps(dict(module="mined_technique_krvk", mode="blind", run="test", goals={"hprog": dict(closed_by=None, proof=None, proved={}, open={})})))
    out = PR.record(str(s), log=lambda *a: None)
    assert out == dict(proved_top=False, entries=[], file=None, not_words={}) and LS.load(force=True) == before


# ---------------------------------------------------------------------------------------------------------------- THE DEMONSTRATION (Kaveh 2026-10-27)

from catspace.incubator.analyst.discovery import prover_demo as DEMO  # noqa: E402


def test_the_demo_states_the_protection_lemma_and_it_builds_with_one_sorry():
    """The owner's protection lemma, stated by code over the entry's class (the file is prover_demo.lean_text's output unless an accepted proof replaced the
    sorry); its statement is the chosen narrowest form; it builds with exactly its own sorry (the imported statement's replayed warning not counted)."""
    e = LS.get(ENTRY)
    path = TL.MINED_DIR / f"{DEMO.module(e)}.lean"
    text = path.read_text()
    assert text == DEMO.lean_text(e) or "was FOUND by" in text
    assert DEMO.STATEMENT.format(name="conjecture_protection") in text and text.startswith(f"import Ideas.Mined.{TL.module_name(e)}\n")
    assert "isAttackedB p (Color.opposite p.turn) s = true →" in DEMO.STATEMENT and "¬ stalemate (applyMove p m)" in DEMO.STATEMENT
    b = TL.build(DEMO.module(e))
    assert b["ok"] and b["sorries"] == (0 if "was FOUND by" in text else 1), b["errors"]


def test_the_demo_shows_the_model_the_imported_definitions():
    e = LS.get(ENTRY)
    ctx, names = PR.context_of(TL.MINED_DIR / f"{DEMO.module(e)}.lean")
    assert {"inClass", "ours", "theirs", "mate", "mu", "unitLost"} <= names and "def inClass" in ctx and "theorem hprog" not in ctx


def test_the_transcript_shows_every_attempt(tmp_path):
    e = LS.get(ENTRY)
    T = DEMO.Transcript(e, dict(run="r0", started="now", tools=["automation", "llm"], think=False, fuse=60), tmp_path / "demo.md")
    T.row(dict(i=1, t="12:00:00", tool="automation", tactic="aesop", goal="protection", depth=0, proof="aesop", accepted=False, secs=2.5,
               errors=["x.lean:9:2: error: aesop: failed to prove the goal after exhaustive search."]))
    T.row(dict(i=2, t="12:10:00", tool="llm whole", goal="protection", depth=0, tokens=900, gen_secs=100.0, done_reason="stop", response="Plan: move away.\n```lean4\n```",
               proof="intro p hp\n  exact foo", lemmas="", accepted=False, why=["Lean errors: 1"], errors=["x.lean:12:4: error: unknown identifier 'foo'"]))
    md = (tmp_path / "demo.md").read_text()
    assert "## The statement" in md and "theorem conjecture_protection" in md and "## The attempts (2 so far)" in md
    assert "### 1. 12:00:00 -- automation `aesop` on `protection` -- REJECTED" in md and "aesop: failed to prove the goal" in md
    assert "The model wrote 900 tokens in 100.0 s" in md and "unknown identifier 'foo'" in md and "> Plan: move away." in md
