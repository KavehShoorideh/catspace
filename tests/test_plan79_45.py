"""PLAN 79.45 (SYSTEM_SPEC 11, "One lemma machinery, used twice" and "The crux of layer 2 is a translator ..."; piece: rules; ledger rows 49, 45): the gate of a lemma read as a rewrite, the pass gated, and the record of the lemma's certificate by translated code.
  (1) the translator takes a lemma as data (`translate(text, module, passes, lemmas)`), reads no folder and names no game, no relation, sort or constant of any description under rules/ nor any relation the lemma deltas define; the make script (which may name a game) gives it only the lemmas of
      form SAME whose computed standing is CERTIFIED or PROVED; the kept file of the first game with every pass and its lemma is the fresh translation, its head names the lemma and the key of its certificate; with no lemma the text and the document are as before;
  (2) the rewrite is what the module text says (a flag of no places, two clauses of the replacement, every reader of the left relation reading the replacement), and the translator refuses, naming the lemma and the reason, a form that is not SAME, a standing that is neither CERTIFIED nor PROVED, a relation
      that is not in the text or has other places, a left relation that a question reads or that is in a recursive set, a right relation or a condition that reads the left one, and a lemma about relations that read the turn;
  (3) the pass gated: what it gates and refuses on small descriptions written here, each refusal naming its reader; the same decisions from a description with every name replaced (the first game's composed text with its lemma included);
  (4) a small game that is not a chess, translated with its lemma and each set of passes and compiled with rustc here, answers the moves and every relation as the reader's evaluator does, the replacement has the rows of the left relation, the flag is the condition, the pieces gated away make no row where
      the flag shuts their readers; and a lemma that is false on it is caught by the same comparison (the control of the harness, on a game that is not a chess);
  (5) the record of the lemma's certificate by translated code lies beside the lemma, names the key of the lemma's certificate and the kept module it was made with, found no disagreement, and read more states than the certificate of 79.44;
  (6) nothing on the engine's path, no manifest, no description, no reader, no certificate is changed since the commit before the plan; the gate lists this file after the one of 79.44.
The comparisons against the plain module of the first game are the cargo tests of cs_game (translated_lemma.rs; with `--cfg singles` the two modules of the scratch folder), and the large certificate is `certificate_large` of that file."""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
GEN = ENGINE / "game" / "tests" / "generated"
TRANSLATOR = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "gdl_to_rust.py"
BEFORE = "93a2c09d"                      # the commit before this plan's code (main when the plan was built)
GAME, LEMMA = "chess", "exposed_by_cases"


def sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def translator():
    from catspace.incubator.analyst.discovery import gdl_to_rust as T
    return T


def t40():
    return sibling("test_plan79_40")


def make_script():
    spec = importlib.util.spec_from_file_location("gdl_to_rust_make", ROOT / "scripts" / "gdl_to_rust_make.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def RL():
    from catspace.incubator.analyst.discovery import rule_lemma
    return rule_lemma


def RC():
    from catspace.incubator.analyst.discovery import rule_lemma_certify
    return rule_lemma_certify


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def doc_of(lem, standing=None) -> dict:
    return {"name": lem.name, "form": lem.form, "left": lem.left, "right": lem.right, "condition": lem.condition, "standing": standing or {"is": "CERTIFIED", "key": lem.standing.get("key", "x")}}


# ------------------------------------------------------------------------------------------------------------------------------------------ (1)

def lemma_names() -> set:
    """Every relation, signature and constant name that a lemma document of any game defines or names (the delta's clauses, the two relations, the condition)."""
    G = t40().reader()
    names: set = set()
    for path in sorted((ROOT / "rules" / "lemmas").rglob("*.json")):
        if path.name.endswith(".certificate.json"):
            continue
        doc = json.loads(path.read_text())
        names |= {doc["left"]["relation"], doc["right"]["relation"]} | {c["relation"] for c in doc["condition"]}
        for c in doc["delta"]:
            for x in G.parse(c):
                if isinstance(x, list) and x and x[0] in G.DECLARATIONS:
                    names.add(x[1])
                elif isinstance(x, list) and x and x[0] == "<=":
                    names.add(G.to_atom(x[1])[0].split(".")[-1])
                    for lit in x[2:]:
                        if isinstance(lit, list) and lit and lit[0] not in ("not", "distinct"):
                            names.add(G.to_atom(lit)[0].split(".")[-1])
    return {n for n in names if re.fullmatch(r"[a-z_][a-z0-9_]*", n)} - sibling("test_plan79_37").FIXED_WORDS


def test_the_translator_names_no_relation_sort_or_constant_of_any_description_nor_any_name_a_lemma_defines():
    words = set(re.findall(r"[a-z_][a-z0-9_]*", TRANSLATOR.read_text().lower()))
    names = t40().all_names() | lemma_names()
    hit = sorted(words & names)
    print(f"OBS 79.45 F scan: {len(names)} names of the descriptions under rules/ and of the lemma documents, {len(words)} words in the translator's source, {len(hit)} in both")
    assert len(lemma_names()) >= 8 and {"walled", "on_wall", "row_pair", "exposed_by_cases"} <= lemma_names()
    assert not hit, hit


def test_the_scan_sees_a_planted_word_of_a_lemma_delta():
    planted = "def f():\n    return 'row_pair'\n"
    assert {"row_pair"} <= set(re.findall(r"[a-z_][a-z0-9_]*", planted.lower())) & lemma_names()


def test_the_translator_reads_no_folder_of_lemmas_and_imports_no_reader_of_them():
    import inspect
    T = translator()
    for fn in (T.rewrite_lemmas, T.lemma_problems, T.lemma_atom, T.swap_atom):
        body = inspect.getsource(fn)
        assert not any(w in body for w in ("read_text", "open(", "Path(", "glob", "import ")), fn.__name__
    assert not any(l.startswith(("import rule_lemma", "from catspace.incubator.analyst.discovery import rule_lemma")) for l in TRANSLATOR.read_text().splitlines())


def test_the_make_script_gives_the_translator_only_certified_or_proved_lemmas_of_form_same(capsys):
    text, docs = make_script().lemma_docs(GAME)
    out = capsys.readouterr().out
    assert [d["name"] for d in docs] == [LEMMA] and docs[0]["standing"]["is"] == "CERTIFIED" and docs[0]["form"] == "SAME"
    assert out.count("not given to the translator") == 2 and "candidate_is_mover_unit" in out and "lone_rook_attacks_its_file" in out
    print(f"OBS 79.45 F make script: given {[d['name'] for d in docs]}; refused two lemmas of form IMPLIES")


def test_the_kept_file_of_the_first_game_with_every_pass_and_its_lemma_is_the_fresh_translation_and_its_head_names_the_lemma():
    T = translator()
    text, docs = make_script().lemma_docs(GAME)
    path = GEN / f"{GAME}_lemma.rs"
    src = path.read_text()
    lines = src.splitlines()
    head = {l.split(":", 1)[0][3:]: l.split(":", 1)[1].strip() for l in lines[:5] if l.startswith("// ") and ":" in l}
    lem = RL().load(GAME, LEMMA)
    cert = RC().load(lem)
    import hashlib
    assert head["description-sha256"] == hashlib.sha256(text.encode()).hexdigest() and head["translator-sha256"] == T.translator_hash() and head["passes"] == ",".join(T.PASSES)
    assert f"// lemma: {LEMMA} SAME exposed -> lemma1_rep, flag lemma1_flag, CERTIFIED {cert['key']}" in lines[:6]
    assert T.translate(text, f"{GAME}_lemma", list(T.PASSES), docs) == src
    assert 'pub const GATED: &[&str] = &["after_occupied", "exposed", "exposed_by_cases", "king_after", "king_ray", "on_wall", "row_pair", "walled"];' in src
    tracked = git("ls-files", "rust/engine/game/tests/generated").stdout.split()
    assert f"rust/engine/game/tests/generated/{GAME}_lemma.rs" in tracked and not any("scratch" in t for t in tracked)
    print(f"OBS 79.45 F kept file {GAME}_lemma.rs: {len(src)} bytes, {len(lines)} lines, the lemma line names the certificate key {cert['key']}, a fresh translation is identical")


def test_with_no_lemma_the_document_is_what_it_was():
    T = translator()
    text = t40().description_of("chess")
    assert T.translate(text, "chess", [], ()) == T.translate(text, "chess", [])
    assert "LEMMAS" not in (GEN / "chess_fast.rs").read_text() and "// lemma:" not in (GEN / "chess_fast.rs").read_text()


# ------------------------------------------------------------------------------------------------------------------------------------------ (2)

def small():
    return sibling("test_plan79_41").WALK


def small_lemma(delta=None, left="see", right="see_alt", cond=None, standing=None, form="SAME"):
    """A lemma of the small game of 79.41: when a piece is on the board `see` has the rows of `free` (the first clause of `see` gives `free` where a piece stands anywhere; the second clause gives a part of it)."""
    delta = delta if delta is not None else ["(signature see_alt cell)", "(<= (see_alt ?n) (free ?n))"]
    text = small() + "\n" + "\n".join(delta) + "\n"
    doc = {"name": "small", "form": form, "left": {"relation": left, "places": 1}, "right": {"relation": right, "places": 1}, "condition": cond if cond is not None else [{"relation": "piece", "args": ["?a"], "neg": False}],
           "standing": standing or {"is": "CERTIFIED", "key": "k0"}}
    return text, doc


def test_the_rewrite_is_a_flag_two_clauses_of_the_replacement_and_every_reader_of_the_left_relation_reading_the_replacement():
    T = translator()
    text, doc = small_lemma()
    new, infos = T.rewrite_lemmas(text, [doc])
    assert infos[0]["flag"] == "lemma1_flag" and infos[0]["rep"] == "lemma1_rep" and infos[0]["evidence"] == "CERTIFIED k0"
    lines = new.splitlines()
    assert "(<= lemma1_flag (true (piece ?a)))" in lines
    assert "(<= (lemma1_rep ?x1) lemma1_flag (see_alt ?x1))" in lines and "(<= (lemma1_rep ?x1) (not lemma1_flag) (see ?x1))" in lines
    assert "(signature lemma1_rep cell)" in lines and "(signature lemma1_flag)" in lines
    legal = [l for l in lines if l.startswith("(<= (legal ?w (go ?b))")]
    assert len(legal) == 2 and "(lemma1_rep ?b)" in legal[0] and "(see ?b)" not in legal[0]
    assert [l for l in lines if l.startswith("(<= (see ")] == [l for l in small().splitlines() if l.startswith("(<= (see ")]       # the clauses of the left relation are as they were
    print(f"OBS 79.45 F rewrite: the flag, the two clauses of the replacement and {sum('lemma1_rep' in l for l in lines)} lines that name the replacement (2 of them its own clauses, the signature, and the reader of the left relation)")


def test_the_translator_refuses_what_the_argument_does_not_cover_naming_the_lemma_and_the_reason():
    T = translator()
    text, doc = small_lemma()
    cases = [
        (dict(form="IMPLIES"), "only SAME is a rewrite"),
        (dict(standing={"is": "CONJECTURE"}), "only CERTIFIED and PROVED"),
        (dict(standing={"is": "CERTIFIED"}), "names its evidence"),
        (dict(right="no_such_relation"), "is not in the text"),
        (dict(left="reach", delta=["(signature see_alt cell)", "(<= (see_alt ?n) (free ?n))", "(signature reach cell)", "(<= (reach ?n) (true (piece ?n)))", "(<= (reach ?m) (reach ?n) (up ?n ?m))"]), "recursive"),
        (dict(cond=[{"relation": "see", "args": ["?a"], "neg": False}]), "the flag reads see"),
    ]
    for kw, why in cases:
        t, d = small_lemma(**kw)
        with pytest.raises(T.Untranslatable, match=why):
            T.translate(t, "game", [], [d])
    t, d = small_lemma(delta=["(signature end_alt)", "(<= end_alt (true (piece n7)))"], left="terminal", right="end_alt")
    d["left"]["places"] = d["right"]["places"] = 0
    with pytest.raises(T.Untranslatable, match="read by a question"):
        T.translate(t, "game", [], [d])
    t, d = small_lemma(delta=["(signature see_alt cell)", "(<= (see_alt ?n) (see ?n))"])
    with pytest.raises(T.Untranslatable, match="the right relation reads see"):
        T.translate(t, "game", [], [d])
    t, d = small_lemma(delta=["(signature see_alt cell)", "(<= (see_alt ?n) (free ?n))", "(signature by_turn cell)", "(<= (by_turn ?n) (does ?w (go ?n)))"], left="by_turn", right="see_alt")
    with pytest.raises(T.Untranslatable, match="reads the turn"):
        T.translate(t, "game", [], [d])
    d3 = dict(doc, right={"relation": "see_alt", "places": 2})
    with pytest.raises(T.Untranslatable, match="places"):
        T.translate(text, "game", [], [d3])
    print("OBS 79.45 F refusals: a form that is not SAME, a standing that is neither CERTIFIED nor PROVED or has no evidence, a relation not in the text, a left relation a question reads or in a recursive set, a flag or a right relation that reads the left one, a lemma about the turn, other places: each refused naming the lemma and why")


def test_two_lemmas_on_one_left_relation_nest_and_read_the_same_rows():
    T = translator()
    text, doc = small_lemma()
    t2 = text + "(signature see_beta cell)\n(<= (see_beta ?n) (free ?n))\n"
    doc2 = dict(doc, name="small2", right={"relation": "see_beta", "places": 1}, condition=[{"relation": "control", "args": ["p"], "neg": False}])
    new, infos = T.rewrite_lemmas(t2, [doc, doc2])
    assert [i["rep"] for i in infos] == ["lemma1_rep", "lemma2_rep"] and "(<= (lemma1_rep ?x1) (not lemma1_flag) (lemma2_rep ?x1))" in new.splitlines()
    T.Description(new)


# ------------------------------------------------------------------------------------------------------------------------------------------ (3)

FLAGGED = small() + "(signature fl)\n(<= fl (true (piece n7)))\n"
FLAGGED = FLAGGED.replace("(<= (legal ?w (go ?b)) (true (control ?w)) (true (piece ?a)) (ray_up ?a ?b) (free ?b) (see ?b))", "(<= (legal ?w (go ?b)) (true (control ?w)) (true (piece ?a)) (ray_up ?a ?b) (free ?b) (not fl) (see ?b))")
assert "(not fl) (see ?b)" in FLAGGED


def plan(passes, text):
    T = translator()
    return T.Plan(T.Description(text), passes)


def names_of(p, ks) -> list:
    return sorted("+".join(sorted(p.d.strata[k][0])) for k in ks)


def test_the_pass_gated_gates_a_relation_whose_readers_are_all_shut_by_a_flag_literal_and_what_only_that_relation_reads():
    p = plan(["gated"], FLAGGED)
    assert names_of(p, p.gated) == ["ahead", "ray_up", "see"], names_of(p, p.gated)       # ray_up is read by that one clause alone, and its own recursion is no reader
    k = next(k for k in p.gated if p.d.strata[k][0] == {"see"})
    assert p.gated[k] == "(!d.r_fl.order.is_empty())" and p.gated_flags[k] == {"fl"}
    ahead = next(k for k in p.gated if p.d.strata[k][0] == {"ahead"})
    assert p.gated[ahead] == f"(off(d, {k}))"
    print(f"OBS 79.45 F gated: on the small game with one flag literal, {names_of(p, p.gated)} are gated; refused: {len(p.gated_refused)} strata, e.g. {sorted(p.gated_refused.values())[0][:100]}")


def test_the_pass_gated_refuses_a_relation_with_a_reader_no_flag_literal_shuts_naming_the_reader():
    p = plan(["gated"], small() + "(signature fl)\n(<= fl (true (piece n7)))\n")
    assert not p.gated
    why = " ".join(p.gated_refused.values())
    assert "(<= (legal ?w (go ?b)) (true (control ?w)) (true (piece ?a)) (ray_up ?a ?b) (free ?b) (see ?b))" in why and "no flag literal shuts" in why
    assert not plan(["gated"], small()).gated                      # no flag at all, nothing is gated


def test_the_pass_gated_does_not_gate_what_a_flag_is_made_of():
    text = small() + "(signature fl)\n(<= fl (true (piece ?a)) (see ?a))\n"
    text = text.replace("(<= (legal ?w (go ?b)) (true (control ?w)) (true (piece ?a)) (ray_up ?a ?b) (free ?b) (see ?b))", "(<= (legal ?w (go ?b)) (true (control ?w)) (true (piece ?a)) (ray_up ?a ?b) (free ?b) (not fl) (see ?b))")
    p = plan(["gated"], text)
    assert names_of(p, p.gated) == ["ray_up"], names_of(p, p.gated)             # see and ahead are what the flag is made of: they stay
    assert any("(<= fl (true (piece ?a)) (see ?a))" in v for v in p.gated_refused.values())


def test_the_piece_order_has_the_flag_before_the_pieces_it_gates():
    p = plan(["gated"], FLAGGED)
    ks = sorted(k for k, (comp, _r) in enumerate(p.d.strata) if not p.d.static[sorted(comp)[0]])
    order = p.gate_order(ks)
    pos = {"+".join(sorted(p.d.strata[k][0])): i for i, k in enumerate(order)}
    assert pos["fl"] < pos["see"] and pos["fl"] < pos["ahead"] and pos["ahead"] < pos["see"] and sorted(order) == ks


def test_the_gated_pass_and_the_lemma_give_the_same_decisions_on_a_description_with_every_name_replaced():
    T = translator()
    m = t40()
    lem = RL().load(GAME, LEMMA)
    text = RL().composed_text(lem)
    doc = doc_of(lem)
    d = T.Description(text)
    names = [doc["left"]["relation"], doc["right"]["relation"]] + [c["relation"] for c in doc["condition"]]
    probe = text + "\n;;;;\n" + " ".join(names) + "\n"
    renamed = m.rename(probe, d)
    text2, tail = renamed.split("\n;;;;\n")
    new = tail.split()
    doc2 = dict(doc, left={"relation": new[0], "places": 4}, right={"relation": new[1], "places": 4}, condition=[dict(c, relation=n) for c, n in zip(doc["condition"], new[2:])])
    a = T.translate(text, "a", list(T.PASSES), [doc])
    b = T.translate(text2, "b", list(T.PASSES), [doc2])

    def made(t, dc):
        new_text, infos = T.rewrite_lemmas(t, [dc])
        dd = T.Description(new_text)
        dd.flags = {i["flag"] for i in infos}
        return T.Plan(dd, T.PASSES)
    pa, pb = made(text, doc), made(text2, doc2)
    shape = lambda p: (len(p.gated), sorted(len(v) for v in p.gated_flags.values()), len(p.gated_refused), len(p.bits), len(p.demanded))
    assert shape(pa) == shape(pb) and len(pa.gated) == 8, (shape(pa), shape(pb))
    assert len(a.splitlines()) == len(b.splitlines())
    print(f"OBS 79.45 F renamed: the first game with its lemma gates {len(pa.gated)} pieces, refuses {len(pa.gated_refused)}; the same with every name replaced")


# ------------------------------------------------------------------------------------------------------------------------------------------ (4)

DRIVER = r"""
include!("game.rs");
use game::*;
fn main() {
    let mut g = Game::new();
    let input = std::io::read_to_string(std::io::stdin()).unwrap();
    for line in input.lines() {
        let facts: Vec<(String, Vec<String>)> = line.split(';').filter(|x| !x.trim().is_empty()).map(|f| { let mut it = f.split_whitespace(); let r = it.next().unwrap().to_string(); (r, it.map(|x| x.to_string()).collect()) }).collect();
        let s = g.state_of_facts(&facts).unwrap();
        g.profiling = true;
        g.reset_profile();
        let mut mv: Vec<String> = g.moves(&s).iter().map(|m| g.render(*m)).collect();
        mv.sort();
        println!("MOVES\t{}", mv.join(" "));
        for ((n, _, ), p) in PIECES.iter().zip(g.prof.iter()) { println!("PIECE\t{}\t{}", n, p.rows); }
        g.profiling = false;
        let rels = g.all_relations(&s, &[]);
        for (name, rows) in rels {
            println!("REL\t{}", name);
            let mut out: Vec<String> = rows.iter().map(|r| r.iter().map(|t| g.render(*t)).collect::<Vec<_>>().join(" ")).collect();
            out.sort(); out.dedup();
            for o in out { println!("ROW\t{}\t{}", name, o); }
        }
        println!("END");
    }
}
"""


def small_states(n: int) -> list:
    """States of the small game: a piece at a cell (not always: the condition of the lemma does not hold without one), blocks, the side to move."""
    import random
    rng = random.Random(45)
    cells = [f"n{i}" for i in range(8)]
    out = []
    for i in range(n):
        facts = [("control", (rng.choice("pq"),))] + [("block", (b,)) for b in rng.sample(cells, rng.randint(0, 5))]
        if i % 3 != 2:
            facts.append(("piece", (rng.choice(cells),)))
        out.append(facts)
    return out


def evaluator_rows(text: str, sts: list) -> list:
    G = t40().reader()
    prog = G.Program(text)
    d = translator().Description(text)
    dynamic = [r for r in d.nodes if not d.static[r]]
    out = []
    for facts in sts:
        db = prog.run([(f"true.{f}", a) for f, a in facts], does=())
        rel = {r: sorted(" ".join(G.term_text(t) if not isinstance(t, str) else t for t in row) for row in db.facts(r)) for r in dynamic}
        legal = [tuple(G.term_text(t) if not isinstance(t, str) else t for t in row) for row in db.facts("legal")]
        moves = []
        if not db.facts("terminal"):
            for role in ("p", "q"):
                mine = sorted(m for r, m in legal if r == role and m != "noop")
                if mine:
                    moves = mine
                    break
        out.append((rel, moves))
    return out


def run_small(text, doc, passes, tmp_path, sts) -> list:
    T = translator()
    (tmp_path / "game.rs").write_text(T.translate(text, "game", passes, [doc]))
    (tmp_path / "main.rs").write_text(DRIVER)
    exe = tmp_path / "walk"
    r = subprocess.run(["rustc", "--edition", "2021", "-O", "-A", "warnings", "-o", str(exe), str(tmp_path / "main.rs")], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    inp = "\n".join(";".join(f"{f} {' '.join(a)}" for f, a in facts) for facts in sts) + "\n"
    r = subprocess.run([str(exe)], input=inp, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    out = []
    for blk in r.stdout.split("END\n")[:-1]:
        rows, pieces, moves = {}, {}, None
        for line in blk.splitlines():
            parts = line.split("\t")
            if parts[0] == "MOVES":
                moves = sorted(re.findall(r"\([^()]*\)|\S+", parts[1])) if len(parts) > 1 and parts[1] else []
            elif parts[0] == "PIECE":
                pieces[parts[1]] = int(parts[2])
            elif parts[0] == "REL":
                rows.setdefault(parts[1], [])
            elif parts[0] == "ROW":
                rows.setdefault(parts[1], []).append(parts[2])
        out.append((rows, pieces, moves))
    return out


def compare_small(text, doc, passes, tmp_path, sts) -> dict:
    """The module of the small game with the lemma against the evaluator on the text with the lemma's delta (no rewrite): -> the counts, or the first disagreement under "bad"."""
    ref = evaluator_rows(text, sts)
    got = run_small(text, doc, passes, tmp_path, sts)
    assert len(got) == len(sts)
    c = {"states": 0, "relations": 0, "held": 0, "gated_rows_zero": 0, "bad": None}
    cond = doc["condition"][0]["relation"]
    for i, ((rows, pieces, moves), (want, want_moves)) in enumerate(zip(got, ref)):
        c["states"] += 1
        held = any(f == cond for f, _a in sts[i])
        if moves != sorted(want_moves):
            c["bad"] = c["bad"] or (i, "moves", moves, want_moves, sts[i])
        for name in rows:
            if name in want:
                c["relations"] += 1
                if sorted(set(rows[name])) != want[name]:
                    c["bad"] = c["bad"] or (i, name, sorted(rows[name]), want[name], sts[i])
        if "lemma1_rep" in rows and sorted(set(rows["lemma1_rep"])) != want[doc["left"]["relation"]]:
            c["bad"] = c["bad"] or (i, "the replacement", sorted(rows["lemma1_rep"]), want[doc["left"]["relation"]], sts[i])
        if bool(rows.get("lemma1_flag")) != held:
            c["bad"] = c["bad"] or (i, "the flag", bool(rows.get("lemma1_flag")), held, sts[i])
        if "gated" in passes or passes == list(translator().PASSES):          # (with the pass demand the relations of the lemma are answered for a key and are no piece: the loop skips what is not a piece)
            shut = {"see": held, "ahead": held, "see_alt": not held}
            for n, is_shut in shut.items():
                if n in pieces and is_shut:
                    assert pieces[n] == 0, (i, n, pieces[n], "a piece whose readers are all shut made rows")
                    c["gated_rows_zero"] += 1
        c["held"] += held
    return c


@pytest.mark.skipif(shutil.which("rustc") is None, reason="no rustc on this machine")
@pytest.mark.parametrize("passes", [[], ["bits"], ["gated"], ["bits", "gated"], list(translator().PASSES)], ids=["no pass", "bits", "gated", "bits and gated", "every pass"])
def test_a_small_game_that_is_not_chess_with_its_lemma_answers_the_moves_and_every_relation_as_the_evaluator_does(passes, tmp_path):
    text, doc = small_lemma()
    sts = small_states(90)
    c = compare_small(text, doc, passes, tmp_path, sts)
    assert c["bad"] is None, c["bad"]
    assert 20 < c["held"] < 90 and c["relations"] > 90 * 4
    if "gated" in passes:
        assert c["gated_rows_zero"] > 20, c
    print(f"OBS 79.45 F small game {passes}: {c['states']} states ({c['held']} where the condition holds), {c['relations']} relations and the moves compared with the evaluator's, the replacement has the rows of the left relation, the flag is the condition, {c['gated_rows_zero']} gated pieces made no row where shut, 0 disagreements")


@pytest.mark.skipif(shutil.which("rustc") is None, reason="no rustc on this machine")
def test_a_lemma_that_is_false_on_the_small_game_is_caught_by_the_same_comparison_the_control_of_the_harness(tmp_path):
    text, doc = small_lemma(delta=["(signature see_alt cell)", "(<= (see_alt ?n) (free ?n) (distinct ?n n3))"])
    c = compare_small(text, doc, ["bits"], tmp_path, small_states(90))
    assert c["bad"] is not None and c["bad"][1] in ("the replacement", "moves", "see_alt"), c
    print(f"OBS 79.45 F control: a lemma that is false on the small game (the right relation leaves one cell out) is caught at state {c['bad'][0]} in {c['bad'][1]}: {c['bad'][2:4]}")


# ------------------------------------------------------------------------------------------------------------------------------------------ (5)

def test_the_record_of_the_certificate_by_translated_code_lies_beside_the_lemma_and_is_fresh():
    import importlib
    spec = importlib.util.spec_from_file_location("rule_lemma_translated", ROOT / "scripts" / "rule_lemma_translated.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lem = RL().load(GAME, LEMMA)
    p = mod.record_path(lem)
    assert p.parent == RL().lemma_dir(GAME) and p.name == f"{LEMMA}.translated.certificate.json"
    assert p.name.endswith(".certificate.json") and sorted(l.name for l in RL().load_all(GAME)) == ["candidate_is_mover_unit", LEMMA, "lone_rook_attacks_its_file"]     # the record is not loaded as a lemma
    assert mod.status(lem) == "fresh", mod.status(lem)
    r = json.loads(p.read_text())
    c = RC().load(lem)
    assert r["certificate_key"] == c["key"] and r["evidence"] == f"CERTIFIED {c['key']}" and r["module"] == f"{GAME}_lemma"
    assert r["disagreeing_states"] == 0 and r["disagreeing_rows"] == 0 and r["flag_disagreements"] == 0 and r["states_held"] > 0 and r["rows_compared"] > 0
    assert r["states"] > 100 * c["positions"], (r["states"], c["positions"])
    assert r["certificate_counts"]["positions"] == c["positions"]
    assert RC().status(lem) == "fresh"
    print(f"OBS 79.45 F record: {r['games']} games, {r['states']} states ({r['states'] / c['positions']:.0f} times the {c['positions']} of the certificate of 79.44), {r['states_held']} where the condition held, {r['rows_compared']} rows compared, 0 disagreements; names the key {r['certificate_key']} and the module hash {r['module_sha256'][:12]}")


# ------------------------------------------------------------------------------------------------------------------------------------------ (6)

def test_nothing_on_the_engine_s_path_no_manifest_no_description_no_reader_no_certificate_differs_from_the_commit_before_the_plan():
    if git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode != 0:
        pytest.skip("the commit before the plan is not on this machine")
    changed = git("diff", "--name-only", BEFORE, "--", ".").stdout.split()
    bad = [c for c in changed if re.match(r"rust/engine/[^/]+/src/", c) or c.endswith(("Cargo.toml", "Cargo.lock")) or (c.startswith("rules/") and not c.startswith("rules/lemmas/"))
           or c in ("catspace/incubator/analyst/discovery/gdl.py", "catspace/incubator/analyst/discovery/certify.py", "catspace/incubator/analyst/discovery/rule_lemma.py", "catspace/incubator/analyst/discovery/rule_lemma_certify.py")]
    assert not bad, bad
    assert f"rules/lemmas/{GAME}/{LEMMA}.translated.certificate.json" in changed
    for f in (f"rules/lemmas/{GAME}/{LEMMA}.json", f"rules/lemmas/{GAME}/{LEMMA}.certificate.json"):
        assert git("diff", "--quiet", BEFORE, "--", f).returncode == 0, f
    print(f"OBS 79.45 F: {len(changed)} files changed since {BEFORE}; none under a crate's src, no manifest, no description, not the reader, not the certificates of the lemmas; one record added beside the lemma")


def test_the_gate_lists_this_test_after_the_one_of_79_44():
    text = (ROOT / "scripts" / "gate.py").read_text()
    assert "test_plan79_45" in text and text.index("test_plan79_44") < text.index("test_plan79_45")
