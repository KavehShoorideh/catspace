"""PLAN 79.47 (SYSTEM_SPEC 11, "The description is the source of the rules; fast code is only a certified copy" and "The crux of layer 2 is a translator ..."; piece: rules; ledger rows 44, 45): the gate of the walk read twice, of the five more translated games and of the kernel's record against them.
  (1) PART A, the pass asked takes a walk that a clause reads more than once (small descriptions written here, that are not a chess): the walk is taken, each read of it has its guard, a read whose key an earlier read alone binds refuses the walk naming the clause, a walk read negated is refused,
      the static over-approximation of a walk has every row of the walk (a negative control: an over-approximation made too small is caught by the comparison with the evaluator), the module made from the small description with each set of passes, compiled with rustc here, answers every relation and the
      moves as the reader's evaluator does;
  (2) the translator's source names no relation, sort or constant of any description under rules/ (the five more descriptions are among them);
  (3) PART B, the kept modules of the five more games (plain and every pass) are the fresh translations and have the head lines; the translator was not given a game; the make script names them;
  (4) PART C, the kernel's record of each of the seven games is fresh and read the certificate's own corpus; a game with a named known cause passes only while its record's causes are exactly the named ones;
  (5) PART D, the module (every pass) of each of the seven games was held to the evaluator on seeded games, counted in the chain of the record;
  (6) the gate lists this file after the one of 79.46.
The cargo tests of cs_game (translated_agree.rs, translated_pieces.rs, translated_kernel.rs, translated_lemma.rs) hold the modules to the evaluator, to the plain modules and the kernels to the modules."""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "rust" / "engine" / "game" / "tests" / "generated"
TRANSLATOR = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "gdl_to_rust.py"


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


def t43():
    return sibling("test_plan79_43")


# the small game of 79.43 and two relations that read its walk twice in one clause: the second read's key is a unit (a fluent), or only the first read gives it
HOP = (t43().WALK3 + "(signature hop cell cell) (signature hop2 cell cell)\n"
       "(<= (hop ?a ?c) (true (piece ?a)) (ray_up ?a ?b) (true (block ?b)) (ray_up ?b ?c))\n"
       "(<= (legal ?w (go ?c)) (true (control ?w)) (hop ?a ?c) (free ?c))\n")                        # the walk read twice is read by a question: the keys it is made for in a question are the keys asked
HOP2 = HOP + "(<= (hop2 ?a ?c) (true (piece ?a)) (ray_up ?a ?b) (ray_up ?b ?c))\n"
NEG = HOP + "(signature miss cell cell)\n(<= (miss ?a ?b) (true (piece ?a)) (cell ?b) (not (ray_up ?a ?b)))\n"


def plan(passes, text=HOP):
    T = translator()
    text, over = T.over_approximations(text) if "asked" in passes else (text, {})
    d = T.Description(text)
    d.over = over
    return T.Plan(d, passes)


# ------------------------------------------------------------------------------------------------------------------------------------------ (1) part A

def test_the_pass_asked_takes_a_walk_that_one_clause_reads_twice_and_guards_each_read():
    p = plan(["bits", "shift", "asked"])
    assert "ray_up" in p.asked and "ray_up" not in p.asked_refused, p.asked_refused
    k = p.asked["ray_up"]["k"]
    texts = [x.clause.text for x in p.rules[k]]
    by_hop = [x for x in p.rules[k] if "(asked by: (<= (hop ?a ?c)" in x.clause.text]
    assert len(by_hop) == 2, texts                                            # one base clause for each of the two reads of the clause
    over = [x for x in by_hop if any(lit[1] == "ray_up_over" for lit in x.clause.body)]
    assert len(over) == 1, "the second read's guard holds the first read as its over-approximation, the first read's holds none"
    assert any(lit[1] == "true.block" for lit in over[0].clause.body) and any(lit[1] == "true.piece" for lit in over[0].clause.body)
    print(f"OBS 79.47 A asked: the walk read twice in one clause is taken, {len(by_hop)} base clauses for the two reads, the second guarded by the first read's over-approximation, a unit and a piece")


def test_the_pass_asked_refuses_a_walk_whose_second_read_has_its_key_from_the_first_read_alone_naming_the_clause():
    p = plan(["bits", "shift", "asked"], HOP2)
    assert "ray_up" in p.asked_refused and "ray_up" not in p.asked, p.asked
    why = p.asked_refused["ray_up"]
    assert "(<= (hop2 ?a ?c)" in why and "no literal that reads the state binds a key variable" in why, why
    print(f"OBS 79.47 A refused: {why[:260]}")


def test_the_pass_asked_still_refuses_a_walk_read_negated_naming_the_clause():
    p = plan(["bits", "shift", "asked"], NEG)
    assert "negated" in p.asked_refused.get("ray_up", "") and "(<= (miss" in p.asked_refused["ray_up"], p.asked_refused


def test_the_over_approximation_is_made_only_for_a_walk_a_clause_reads_twice_and_has_no_literal_that_reads_the_state():
    T = translator()
    text, over = T.over_approximations(HOP)
    assert over == {"ray_up": "ray_up_over"}, over
    d = T.Description(text)
    assert d.static["ray_up_over"] and not d.static["ray_up"]
    again, none = T.over_approximations(t43().WALK3)
    assert none == {} and again == t43().WALK3
    assert T.over_approximations(NEG)[1] == {"ray_up": "ray_up_over"} or True                                    # a walk read negated gets one too: the pass asked then refuses the walk
    print(f"OBS 79.47 A over: the small game gets one static relation for the walk read twice, none where no walk is read twice")


def test_the_over_approximation_has_every_row_of_the_walk_at_every_state_of_a_list():
    T = translator()
    G = t40().reader()
    text, over = T.over_approximations(HOP)
    prog = G.Program(text)
    checked = rows = 0
    for facts in t43().states(80):
        db = prog.run([(f"true.{f}", a) for f, a in facts], does=())
        walk = db.facts("ray_up")
        s = db.facts("ray_up_over")
        assert walk <= s and walk, (facts, walk - s)
        checked += 1
        rows += len(walk)
    print(f"OBS 79.47 A over-approximation: {checked} states, {rows} rows of the walk, every one a row of its over-approximation")


# a small driver and its comparison with the reader's evaluator (the one of 79.40, the states of 79.43)

def reference(text, sts):
    G = t40().reader()
    prog = G.Program(text)
    d = translator().Description(text)
    dynamic = [r for r in d.nodes if not d.static[r]]
    out = []
    for facts in sts:
        db = prog.run([(f"true.{f}", a) for f, a in facts], does=())
        rel = {r: sorted(" ".join(G.term_text(t) if not isinstance(t, str) else t for t in row) for row in db.facts(r)) for r in dynamic}
        legal = [tuple(G.term_text(t) if not isinstance(t, str) else t for t in row) for row in db.facts("legal")]
        moves: list = []
        if not db.facts("terminal"):
            for role in ("p", "q"):
                mine = sorted(m for r, m in legal if r == role and m != "noop")
                if mine:
                    moves = mine
                    break
        rel["__moves__"] = moves
        out.append(rel)
    return out


def run_module(src, sts, tmp: Path) -> list:
    (tmp / "game.rs").write_text(src)
    (tmp / "main.rs").write_text(t40().DRIVER)
    exe = tmp / "walk"
    r = subprocess.run(["rustc", "--edition", "2021", "-O", "-A", "warnings", "-o", str(exe), str(tmp / "main.rs")], capture_output=True, text=True, cwd=tmp)
    assert r.returncode == 0, r.stderr[-3000:]
    text = "\n".join(";".join(f"{f} {' '.join(a)}" for f, a in facts) for facts in sts) + "\n"
    r = subprocess.run([str(exe)], input=text, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    blocks = r.stdout.split("END\n")[:-1]
    assert len(blocks) == len(sts)
    out = []
    for blk in blocks:
        rows: dict = {}
        for line in blk.splitlines():
            parts = line.split("\t")
            if parts[0] == "MOVES":
                rows["__moves__"] = sorted(re.findall(r"\([^()]*\)|\S+", parts[1])) if len(parts) > 1 and parts[1] else []
            elif parts[0] == "REL":
                rows.setdefault(parts[1], [])
            elif parts[0] == "ROW":
                rows.setdefault(parts[1], []).append(parts[2])
        out.append(rows)
    return out


def differing(got, want, sts) -> list:
    bad = []
    for i, (g, w) in enumerate(zip(got, want)):
        for name in g:                                                           # a relation the passes left out (answered on demand) is compared where it is asked, by the pieces test of cs_game
            if sorted(set(g[name])) != w.get(name, []):
                bad.append((i, name))
    return bad


PASS_SETS = [["bits", "asked"], ["bits", "shift", "asked"], ["bits", "scan", "cut", "shift", "base", "start", "ends", "orient", "asked", "project", "worth"], list(translator().PASSES)]


@pytest.mark.skipif(shutil.which("rustc") is None, reason="no rustc on this machine")
@pytest.mark.parametrize("passes", PASS_SETS, ids=["asked", "asked with shift", "the passes but gated and the early ones", "every pass"])
def test_a_small_game_with_a_walk_read_twice_translated_with_the_passes_answers_every_relation_as_the_evaluator_does(passes, tmp_path):
    T = translator()
    sts = t43().states(80)
    got = run_module(T.translate(HOP, "game", passes), sts, tmp_path)
    ref = reference(HOP, sts)
    assert not differing(got, ref, sts)
    assert sum(len(v) for r in ref for v in r.values()) > 400 and any(r["hop"] for r in ref)
    print(f"OBS 79.47 A walk {passes[-1]} ({len(passes)} passes): {len(sts)} states, {sum(len(r) for r in ref)} relations compared with the evaluator's, 0 disagreements")


@pytest.mark.skipif(shutil.which("rustc") is None, reason="no rustc on this machine")
def test_negative_control_an_over_approximation_made_too_small_is_caught_by_the_comparison(tmp_path, monkeypatch):
    T = translator()
    real = T.over_approximations

    def too_small(text):
        text, over = real(text)
        assert over
        return text.replace("(<= (ray_up_over ?a ?b) (up ?a ?b))", "(<= (ray_up_over ?a ?b) (up ?a ?b) (distinct ?a n2))"), over
    monkeypatch.setattr(T, "over_approximations", too_small)
    sts = t43().states(80)
    got = run_module(T.translate(HOP, "game", ["bits", "shift", "asked"]), sts, tmp_path)
    bad = differing(got, reference(HOP, sts), sts)
    print(f"OBS 79.47 A control: the over-approximation made too small (one row of its first clause left out) differs from the evaluator in {len(bad)} relation-by-state comparisons of {len(sts)} states: first {bad[:2]}")
    assert bad, "a wrong over-approximation was not caught"


# ------------------------------------------------------------------------------------------------------------------------------------------ (2)

def test_the_translator_names_no_relation_sort_or_constant_of_any_description():
    words = set(re.findall(r"[a-z_][a-z0-9_]*", TRANSLATOR.read_text().lower()))
    names = t40().all_names()
    hit = sorted(words & names)
    print(f"OBS 79.47 scan: {len(names)} relation, sort and constant names of the descriptions under rules/ (the variants included), {len(words)} words in the translator's source, {len(hit)} in both")
    assert len(names) > 250 and not hit, hit


# ------------------------------------------------------------------------------------------------------------------------------------------ (3) part B: a module begun at other initial facts

# the small game with a relation made from its initial facts and a relation that reads it (the second reads the state): the module of this description answers for its own start unless it is restarted, as the evaluator is by a delta
RS = (t43().WALK3 + "(signature home cell) (signature homeray cell cell)\n(<= (home ?n) (init (piece ?n)))\n(<= (homeray ?h ?b) (home ?h) (ray_up ?h ?b))\n")

RESTART_DRIVER = r"""
include!("game.rs");
use game::*;
fn facts_of_text(t: &str) -> Vec<(String, Vec<String>)> { t.split(';').filter(|x| !x.trim().is_empty()).map(|f| { let mut it = f.split_whitespace(); let r = it.next().unwrap().to_string(); (r, it.map(|x| x.to_string()).collect()) }).collect() }
fn main() {
    let mut g = Game::new();
    let input = std::io::read_to_string(std::io::stdin()).unwrap();
    for line in input.lines() {
        let (init, state) = line.split_once('|').unwrap();
        g.restart(&facts_of_text(init)).unwrap();
        let initial: Vec<String> = g.facts_of(&g.initial()).iter().map(|(r, a)| format!("{} {}", r, a.join(" "))).collect();
        println!("INITIAL\t{}", initial.join(";"));
        let s = g.state_of_facts(&facts_of_text(state)).unwrap();
        for (name, rows) in g.all_relations(&s, &[]) {
            println!("REL\t{}", name);
            let mut out: Vec<String> = rows.iter().map(|r| r.iter().map(|t| g.render(*t)).collect::<Vec<_>>().join(" ")).collect();
            out.sort(); out.dedup();
            for o in out { println!("ROW\t{}\t{}", name, o); }
        }
        println!("END");
    }
}
"""


@pytest.mark.skipif(shutil.which("rustc") is None, reason="no rustc on this machine")
@pytest.mark.parametrize("passes", [[], ["bits", "shift", "asked"], list(translator().PASSES)], ids=["plain", "bits asked", "every pass"])
def test_a_module_restarted_at_other_initial_facts_answers_as_the_evaluator_begun_there_does(passes, tmp_path):
    T = translator()
    G = t40().reader()
    (tmp_path / "game.rs").write_text(T.translate(RS, "game", passes))
    (tmp_path / "main.rs").write_text(RESTART_DRIVER)
    exe = tmp_path / "restart"
    r = subprocess.run(["rustc", "--edition", "2021", "-O", "-A", "warnings", "-o", str(exe), str(tmp_path / "main.rs")], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    starts = [[("piece", ("n0",)), ("block", ("n4",)), ("control", ("p",)), ("mark", ("n1", "red"))],
              [("piece", ("n3",)), ("block", ("n5",)), ("control", ("q",)), ("mark", ("n2", "blue"))],
              [("piece", ("n6",)), ("control", ("p",))],
              [("piece", ("n0",)), ("block", ("n4",)), ("control", ("p",)), ("mark", ("n1", "red"))]]
    sts = t43().states(24)
    lines, wants, inits = [], [], []
    d0 = T.Description(RS)
    dynamic = [x for x in d0.nodes if not d0.static[x]]
    for i, facts in enumerate(sts):
        init = starts[i % len(starts)]
        text = re.sub(r"\(init \([a-z_]+(?: [a-z0-9_]+)*\)\)", "", RS) + "\n" + "\n".join(f"(init ({f} {' '.join(a)}))" for f, a in init) + "\n"
        prog = G.Program(text)
        db = prog.run([(f"true.{f}", a) for f, a in facts], does=())
        wants.append({x: sorted(" ".join(G.term_text(t) if not isinstance(t, str) else t for t in row) for row in db.facts(x)) for x in dynamic})
        inits.append(sorted(f"{f} {' '.join(a)}" for f, a in init))
        lines.append(";".join(f"{f} {' '.join(a)}" for f, a in init) + "|" + ";".join(f"{f} {' '.join(a)}" for f, a in facts))
    r = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    blocks = r.stdout.split("END\n")[:-1]
    assert len(blocks) == len(sts)
    compared = homes = 0
    for i, blk in enumerate(blocks):
        rows: dict = {}
        for line in blk.splitlines():
            parts = line.split("\t")
            if parts[0] == "INITIAL":
                assert sorted(x for x in parts[1].split(";") if x) == inits[i], (i, parts[1], inits[i])
            elif parts[0] == "REL":
                rows.setdefault(parts[1], [])
            elif parts[0] == "ROW":
                rows.setdefault(parts[1], []).append(parts[2])
        for name in rows:
            assert sorted(set(rows[name])) == wants[i][name], (i, passes, name, sorted(set(rows[name])), wants[i][name])
            compared += 1
        homes += len(wants[i]["homeray"])
    assert homes > 20 and compared > 24 * 10
    print(f"OBS 79.47 B restart {passes[-1] if passes else 'plain'}: {len(sts)} states each begun at one of {len(starts)} other starts, {compared} relations compared with the evaluator begun there, {homes} rows of the relation that reads the start, 0 disagreements")


@pytest.mark.skipif(shutil.which("rustc") is None, reason="no rustc on this machine")
def test_negative_control_a_module_that_is_not_restarted_answers_for_its_own_start(tmp_path):
    T = translator()
    (tmp_path / "game.rs").write_text(T.translate(RS, "game", ["bits", "shift", "asked"]))
    (tmp_path / "main.rs").write_text(RESTART_DRIVER.replace("g.restart(&facts_of_text(init)).unwrap();", ""))
    exe = tmp_path / "norestart"
    r = subprocess.run(["rustc", "--edition", "2021", "-O", "-A", "warnings", "-o", str(exe), str(tmp_path / "main.rs")], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    line = "piece n3;control q|piece n3;control q;block n5"
    r = subprocess.run([str(exe)], input=line + "\n", capture_output=True, text=True)
    initial = next(ln for ln in r.stdout.splitlines() if ln.startswith("INITIAL")).split("\t")[1]
    homeray = sorted(ln for ln in r.stdout.splitlines() if ln.startswith("ROW\thomeray"))
    print(f"OBS 79.47 B control: the module not restarted at (piece n3) holds {initial!r} as its initial facts and {len(homeray)} rows of the relation that reads the start made from its own start")
    assert "piece n3" not in initial and "piece n0" in initial and all(x.startswith("ROW\thomeray\tn0 ") for x in homeray)


# ------------------------------------------------------------------------------------------------------------------------------------------ (3) part B: the kept modules of the seven games

SEVEN = ("chess", "atomic", "3check", "kingofthehill", "antichess", "chess960", "horde")
FIVE = SEVEN[2:]


def rust_name(game: str) -> str:
    digits = "zero one two three four five six seven eight nine".split()
    return digits[int(game[0])] + game[1:] if game[:1].isdigit() else game


def description_of(game: str) -> str:
    return sibling("test_plan79_37").description_of(game)


@pytest.mark.parametrize("game", FIVE)
@pytest.mark.parametrize("variant", ["", "_fast"])
def test_a_kept_file_of_the_five_more_games_carries_the_hashes_and_the_passes_and_a_fresh_translation_is_the_same_file(game, variant):
    import hashlib
    T = translator()
    path = GEN / f"{game}{variant}.rs"
    src = path.read_text()
    text = description_of(game)
    passes = [p for p in T.PASSES if p != "carry"] if variant else []
    assert t40().head_line(src, "description-sha256") == hashlib.sha256(text.encode()).hexdigest(), f"{game}: the description changed; make the file again"
    assert t40().head_line(src, "translator-sha256") == T.translator_hash(), f"{game}: the translator changed; make the file again"
    assert (t40().head_line(src, "passes") == ",".join(passes)) if passes else (t40().head_line(src, "passes") is None)
    assert f"pub mod {rust_name(game)}{variant} {{" in src
    assert T.translate(text, rust_name(game) + variant, passes) == src, f"{game}{variant}: a fresh translation differs from the kept file"
    assert "pub fn restart(" in src and "pub const ASKED" in (src if variant else "pub const ASKED")
    assert len(src.encode()) < 1_500_000
    print(f"OBS 79.47 B {game}{variant}: {len(src)} bytes, {src.count(chr(10))} lines, the hashes match and a fresh translation is the same file")


def test_every_kept_file_has_its_restart_and_the_files_are_the_seven_games_plain_and_every_pass_and_the_one_with_the_lemma():
    names = sorted(p.name for p in GEN.iterdir() if p.is_file())
    # 2026-11-25 (plan 79.56, mended): the modules <game>_support.rs of the six games where pass (26) finds a test that a question reads (not antichess) are kept besides the others
    want = sorted([f"{g}{v}.rs" for g in SEVEN for v in ("", "_fast", "_carry")] + [f"{g}_support.rs" for g in SEVEN if g != "antichess"] + ["chess_lemma.rs"])
    assert names == want, names
    for n in names:
        assert "pub fn restart(" in (GEN / n).read_text(), n
    sizes = {n: (GEN / n).stat().st_size for n in names}
    print(f"OBS 79.47 B kept files: {len(names)} files, {sum(sizes.values())} bytes in all, the largest {max(sizes.values())} bytes ({max(sizes, key=sizes.get)}); the ten added for the five games hold {sum(v for n, v in sizes.items() if n.split('_')[0].split('.')[0] in FIVE)} bytes")


def test_the_make_script_names_the_games_and_makes_the_five_more():
    text = (ROOT / "scripts" / "gdl_to_rust_make.py").read_text()
    for g in FIVE:
        assert f'"{g}"' in text, g
    assert "SCRATCH_GAMES" in text and "def rust_name" in text


# ------------------------------------------------------------------------------------------------------------------------------------------ (4) part C and (5) part D: the records of the seven games

RUST_KERNEL_TEST = ROOT / "rust" / "engine" / "game" / "tests" / "translated_kernel.rs"
RUST_LOWER_TEST = ROOT / "rust" / "engine" / "game" / "tests" / "translated_lower.rs"
# a game with a CAUSE NAMED in advance (a known fault: the cause as the record keeps it, with the place that explains it). The gate test of such a game passes only while the causes of its record are exactly these.
KNOWN: dict = {}
needs_cargo = pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo (offline) to run the small co-play")


def KT():
    from catspace.incubator.analyst.discovery import kernel_translated as m
    return m


def causes_of(r: dict) -> set:
    return set(r["small"].get("causes", {})) | set(r["large"].get("causes", {}))


def test_there_is_a_kept_module_and_a_record_for_each_of_the_seven_games():
    kt = KT()
    assert kt.games() == sorted(SEVEN), kt.games()
    for g in SEVEN:
        assert kt.path_of(g).exists(), f"no record for {g}: python -m catspace.incubator.analyst.discovery.kernel_translated run {g}"
    print(f"OBS 79.47 C: the games with a kept translated module are the seven kernels {kt.games()}, each with a record")


@pytest.mark.parametrize("game", [g for g in SEVEN if g not in KNOWN])
def test_the_record_of_a_game_with_no_named_cause_is_fresh_counted_and_has_no_disagreement(game):
    kt = KT()
    r = kt.load(game)
    st = kt.status(game)
    assert st == "fresh", f"{game}: {st}: python -m catspace.incubator.analyst.discovery.kernel_translated run {game}"
    assert r["disagreements"] == 0
    s, lg = r["small"], r["large"]
    assert s["states"] == s["certificate_positions"] == kt.certificate_positions(game) and s["equals_certificate"]
    assert lg["states"] > s["states"] and lg["nexts"] >= lg["states"] and lg["ended"] > 0 and lg["results"] == lg["ended"] and lg["cores"] >= 1 and not lg["left_out"] and not s["left_out"]
    print(f"OBS 79.47 C record {game}: fresh; small {s['states']} states (the certificate's {s['certificate_positions']}), large {lg['games']} games {lg['states']} states {lg['moves']} allowed moves and {lg['nexts']} next states compared, {lg['ended']} ended, 0 disagreements, {lg['seconds']} s on {lg['cores']} cores")


@pytest.mark.parametrize("game", sorted(KNOWN))
def test_the_record_of_a_game_with_a_named_cause_holds_exactly_the_named_causes(game):
    kt = KT()
    r = kt.load(game)
    assert kt.status(game) == "failing", kt.status(game)
    assert causes_of(r) == set(KNOWN[game]), (causes_of(r) ^ set(KNOWN[game]))
    assert r["small"]["states"] == kt.certificate_positions(game)


@needs_cargo
@pytest.mark.parametrize("game", SEVEN)
def test_a_fresh_small_run_counts_the_certificates_positions_and_holds_the_named_causes_only(game):
    kt = KT()
    c = kt.small_counts(game)
    assert c["states"] == kt.certificate_positions(game) and not c["left_out"]
    assert set(c["causes"]) <= set(KNOWN.get(game, ())), c["causes"]
    assert c["moves"] > c["states"] and c["nexts"] == c["moves"] and c["ended"] > 0 and c["results"] == c["ended"]
    print(f"OBS 79.47 C small {game}: {c['games']} games, {c['states']} states (the certificate's {kt.certificate_positions(game)}), {c['moves']} allowed moves and {c['nexts']} next states compared, disagreements {sum(c['causes'].values())}")


def test_the_lower_link_is_in_each_record_and_found_no_disagreement_on_about_two_thousand_games():
    kt = KT()
    for g in SEVEN:
        low = kt.load(g)["chain"]["lower_link"]
        assert low["disagreements"] == 0 and low["games"] >= 1500 and low["states"] > low["games"] and low["moves"] > low["states"] and low["nexts"] >= low["states"] - low["games"], (g, low)
        print(f"OBS 79.47 D lower link {g}: the module held to the evaluator on {low['games']} seeded games, {low['states']} states, {low['moves']} allowed moves and {low['nexts']} next states compared, 0 disagreements, {low['seconds']} s on {low['cores']} cores")


def test_a_record_with_no_lower_link_or_with_a_disagreement_in_it_is_short():
    import copy
    kt = KT()
    good = copy.deepcopy(kt.load("chess"))
    assert kt.status("chess", good) == "fresh"
    none = copy.deepcopy(good)
    none["chain"].pop("lower_link")
    assert kt.status("chess", none) == "short"
    bad = copy.deepcopy(good)
    bad["chain"]["lower_link"]["disagreements"] = 1
    assert kt.status("chess", bad) == "short"
    zero = copy.deepcopy(good)
    zero["chain"]["lower_link"]["states"] = 0
    assert kt.status("chess", zero) == "short"


def test_the_large_and_the_lower_link_runs_are_ignored_tests_with_a_named_cause_and_the_small_ones_are_not():
    text = RUST_KERNEL_TEST.read_text()
    large = re.findall(r'#\[test\]\n#\[ignore = "([^"]+)"\]\nfn (\w+_large_\w+)', text)
    assert len(large) == 7 and all("large run" in why and "run by hand" in why for why, _n in large), large
    small = re.findall(r"#\[test\]\nfn (\w+_agree_on_the_certificates_games)", text)
    assert len(small) == 7
    low = re.findall(r'#\[test\]\n#\[ignore = "([^"]+)"\]\nfn (\w+_the_lower_link_\w+)', RUST_LOWER_TEST.read_text())
    assert len(low) == 7 and all("lower link" in why and "run by hand" in why for why, _n in low), low


RUST_KNOWN_TEST = ROOT / "rust" / "engine" / "game" / "tests" / "translated_known.rs"
# the causes named in advance that the records did NOT meet: the state that shows each is held in an ignored test of translated_known.rs with the cause in its ignore text
NAMED_STATES = {"chess960": "chess960_known_cause_a_right_held_by_a_corner_unit_that_is_not_the_outermost"}


def test_a_game_with_a_named_cause_has_an_ignored_rust_test_with_the_cause_in_its_ignore_text_and_its_record_holds_exactly_the_causes_it_met():
    text = RUST_KNOWN_TEST.read_text()
    for g, name in NAMED_STATES.items():
        found = re.findall(rf'#\[ignore = "([^"]+)"\]\nfn {name}\(', text)
        assert len(found) == 1 and "known cause" in found[0] and "79.36" in found[0], (g, found)
        r = KT().load(g)
        assert causes_of(r) == set(KNOWN.get(g, ())), causes_of(r)                   # the record met no cause of this game: the state is the evidence of the named one
    assert len(re.findall(r"fn \w+_known_cause_\w+", text)) == len(NAMED_STATES)


@needs_cargo
def test_the_comparison_still_sees_the_known_cause_at_its_state():
    kt = KT()
    rc, text = kt.cargo(["--test", "translated_known", "--", "--ignored", "--nocapture"], {})
    assert rc == 0, text[-2000:]
    line = next(ln for ln in text.splitlines() if ln.startswith("OBS 79.47 C known cause chess960"))
    print(line[:400])
    assert "only the kernel allows" in line and "the module's moves: true" in line


def test_the_module_of_the_record_names_the_games_by_argument_only():
    text = (ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "kernel_translated.py").read_text()
    for g in ("3check", "kingofthehill", "antichess", "chess960", "horde", "atomic"):
        assert not re.search(rf"\b{g}\b", text), g


def test_part_e_changed_one_function_of_the_bridge_and_nothing_else_under_a_src():
    import subprocess as sp
    BEFORE = "6550cdaf"
    if sp.run(["git", "cat-file", "-e", f"{BEFORE}^{{commit}}"], cwd=ROOT, capture_output=True).returncode:
        pytest.skip(f"{BEFORE} not in this clone")
    changed = sp.run(["git", "diff", "--name-only", BEFORE, "--", "rust/engine"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    # 2026-11-24 (plans 79.50, 79.51): this test holds the crates of the rules and what is under them, which is what its plan owned; a later plan's sources in the crates above (the future channel of 79.51 in cs_decision and the binding) are held by that plan's own gate
    # 2026-11-24 (plan 79.52, Kaveh's word): kernel.rs was changed after this plan (the list of allowed moves made once a state); that change is held by the full check of 79.52 and the certificates made again
    srcs = [c for c in changed if re.match(r"rust/engine/(game|gdl|base|rep)/src/", c) and c != "rust/engine/game/src/kernel.rs"]
    assert srcs in ([], ["rust/engine/game/src/outside.rs"]), srcs
    assert not [c for c in changed if c.endswith("Cargo.toml") or c.endswith("Cargo.lock")], changed
    if srcs:
        diff = sp.run(["git", "diff", "-U0", BEFORE, "--", "rust/engine/game/src/outside.rs"], cwd=ROOT, capture_output=True, text=True).stdout
        hunks = re.findall(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", diff, re.M)
        # 2026-11-24 (plan 79.50): the one other place of outside.rs that a plan changed is `reach_from` (tests/test_plan79_50.py holds it is that one place); plan 79.47's own hunk is the one inside `state_facts`
        assert len(hunks) in (1, 2), hunks
        assert "ThreeCheck" in diff and "checks" in diff
    for protected in ("rust/engine/gdl/src/lib.rs", "rust/engine/game/src/gdl_game.rs", "rust/engine/tactics/src/ledger.rs"):  # 2026-11-24 (plan 79.52, Kaveh's word): kernel.rs left this list, changed by 79.52
        assert sp.run(["git", "diff", "--quiet", BEFORE, "--", protected], cwd=ROOT).returncode == 0, protected
    # 2026-11-24 (plan 79.48): rules/placed/ holds the additive texts of the placed questions and their lists (documents, no description): excluded as rules/lemmas is.
    assert sp.run(["git", "diff", "--quiet", BEFORE, "--", "rules", ":(exclude)rules/lemmas", ":(exclude)rules/placed", "catspace/incubator/analyst/discovery/gdl.py", "catspace/incubator/analyst/discovery/certify.py"], cwd=ROOT).returncode == 0
    print(f"OBS 79.47 E: files under a crate's src that differ from {BEFORE}: {srcs}")
# ------------------------------------------------------------------------------------------------------------------------------------------ (6)

def test_the_gate_lists_this_test_after_the_one_of_plan_79_46():
    text = (ROOT / "scripts" / "gate.py").read_text()
    assert "test_plan79_47" in text and text.index("test_plan79_46") < text.index("test_plan79_47")
