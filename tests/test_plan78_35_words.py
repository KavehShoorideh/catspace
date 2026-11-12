"""Plan 78.35: words built from words (word_terms.py), names (word_names.py, rules/concepts/names.txt), the registry of a store's words and the general / game labels (word_registry.py),
naming a long admitted body (induce.name_bodies), the use / generation split (usage_split.py), the flags. Everything in a tmp sandbox; the real store is never written; no tablebase and no
Stockfish. A term compiled to rules is evaluated by the Rust evaluator on chess positions and by the Python mirror of it (gdl.py) on a made-up game, each against a brute-force computation."""
import inspect
import json
import re
from pathlib import Path

import chess
import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import abstraction as AB
from catspace.incubator.analyst.discovery import apply_audit as AU
from catspace.incubator.analyst.discovery import board_structure as BS
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import induce as IND
from catspace.incubator.analyst.discovery import initialize as IN
from catspace.incubator.analyst.discovery import targeted as TG
from catspace.incubator.analyst.discovery import usage_split as US
from catspace.incubator.analyst.discovery import word_names as WN
from catspace.incubator.analyst.discovery import word_registry as WR
from catspace.incubator.analyst.discovery import word_terms as WT
from catspace.io import paths

ROOT = Path(__file__).resolve().parents[1]
RULES = paths.REPO_ROOT / "rules"
DISC = ROOT / "catspace/incubator/analyst/discovery"
KEYS = "keys"
START = "8/8/4k3/8/3K4/8/5P2/8 w - - 0 1"
QUIET = lambda *_a, **_k: None          # noqa: E731

# ------------------------------------------------------------------------------------------------------------------------------ a made-up game (no chess in it)

CELLS = [(x, y) for x in "123" for y in "123"]


def _grid_steps():
    out = []
    for (x, y) in CELLS:
        for (u, v) in CELLS:
            if abs(int(x) - int(u)) + abs(int(y) - int(v)) == 1:
                out.append((x, y, u, v))
    return out


STEPS = _grid_steps()
LINKS = [("red", "1", "2"), ("red", "2", "3"), ("blue", "1", "3"), ("blue", "3", "1")]
WALL = ("2", "2")
GAME = ("(sort cx 1 2 3) (sort cy 1 2 3) (sort side red blue) (sort distance d0 d1 d2)\n"
        "(signature role side) (signature step cx cy cx cy) (signature mark side cx cy) (signature wall cx cy) (signature link side cx cx)\n"
        "(role red) (role blue) (init (mark red 1 1)) (<= (next (mark ?s ?x ?y)) (true (mark ?s ?x ?y))) (wall 2 2)\n"
        + " ".join(f"(step {a} {b} {c} {d})" for a, b, c, d in STEPS) + "\n" + " ".join(f"(link {s} {a} {b})" for s, a, b in LINKS) + "\n")
DESC = WT.read_description(GAME)
STATE = {("true.mark", ("red", "1", "1")), ("true.mark", ("blue", "3", "3")), ("true.mark", ("red", "3", "1"))}


def run_game(term_text: str, state=STATE) -> set:
    t = WT.parse(term_text)
    text = WT.compile("zed", t, DESC.signatures, fluents=DESC.fluents, sorts=DESC.sorts)
    prog = GDL.Program(GAME + text)
    assert prog.typecheck() == [], prog.typecheck()
    return {tuple(f) for f in prog.run(state).facts("zed")}


def step_pairs():
    return {((a, b), (c, d)) for a, b, c, d in STEPS}


def closure_of(pairs):
    cur = set(pairs)
    while True:
        new = cur | {(x, y) for (x, z) in cur for (z2, y) in cur if z == z2}
        if new == cur:
            return cur
        cur = new


def flat(pairs):
    return {a + b for a, b in pairs}


# ------------------------------------------------------------------------------------------------------------------------------ the term language

def test_parse_and_show_round_trip_every_constructor():
    texts = ["(r ?x _ c)", "(r)", "(and (a ?x) (b ?x ?y) (not (c ?y)) (distinct ?x ?y))", "(or (a ?x) (b ?x))", "(exists (?y ?z) (and (a ?x ?y) (b ?y ?z)))",
             "(compose (a ?a ?b) (b ?b ?c))", "(compose (a ?a ?b ?c ?d) (b ?c ?d ?e ?f) 2)", "(via (a ?a ?b) (b ?b ?c))", "(reach (a ?a ?b) 3)", "(reach0 (a ?a ?b ?c ?d) 2 2)",
             "(closure (a ?a ?b))", "(closure (a ?a ?b ?c ?d) 2)"]
    for t in texts:
        term = WT.parse(t)
        assert WT.show(term) == t and WT.parse(WT.show(term)) == term
    assert WT.parse("(a)") == WT.parse("a")                        # a bare name is a literal of no slot
    assert WT.size(WT.parse("(and (A ?x) (not (B ?x)))")) == 4 and WT.size(WT.parse("(exists (?y) (A ?x ?y))")) == 2
    assert WT.free_vars(WT.parse("(exists (?y) (and (A ?x ?y) (B ?z ?x)))")) == ["?x", "?z"]
    assert WT.free_vars(WT.parse("(compose (A ?p ?a ?b) (B ?p ?b ?c))")) == ["?p", "?a", "?c"]
    assert WT.free_vars(WT.parse("(via (A ?a ?b) (B ?b ?c))")) == ["?a", "?c", "?b"]


def test_parse_errors():
    for bad in ["(and)", "(not)", "(not A B)", "(exists ?x (A ?x))", "(exists () (A ?x))", "(distinct ?x _)", "(reach (A ?a ?b) 0)", "(reach (A ?a ?b) 9)", "(reach (A ?a ?b) x)",
                "(closure)", "(A ?_1)", "(A ?)", "(A ?x) (B ?x)", "()", "(compose (A ?a ?b))"]:
        with pytest.raises(WT.TermError):
            WT.parse(bad)


def test_sort_checking_and_its_errors():
    sigs, sorts = DESC.signatures, DESC.sorts
    assert WT.sort_of(WT.parse("(and (mark ?s ?x ?y) (not (wall ?x ?y)))"), sigs, sorts) == ("side", "cx", "cy")
    assert WT.sort_of(WT.parse("(compose (step ?a ?b ?c ?d) (step ?c ?d ?e ?f) 2)"), sigs, sorts) == ("cx", "cy", "cx", "cy")
    assert WT.sort_of(WT.parse("(via (link ?s ?a ?b) (link ?s ?b ?c))"), sigs, sorts) == ("side", "cx", "cx", "cx")
    assert WT.sort_of(WT.parse("(exists (?s) (mark ?s ?x ?y))"), sigs, sorts) == ("cx", "cy")
    for bad, why in [("(nosuch ?x)", "no signature"), ("(wall ?x)", "slot"), ("(and (wall ?x ?y) (mark ?x ?x ?y))", "sort"), ("(wall red 1)", "no member"),
                     ("(not (wall ?x ?y))", "inside an `and`"), ("(and (wall ?x ?y) (not (mark ?s ?x ?y)))", "unsafe negation"), ("(and (wall ?x ?y) (distinct ?x ?z))", "unsafe"),
                     ("(or (wall ?x ?y) (mark ?s ?x ?y))", "different free variables"), ("(or (wall ?x ?y) (wall ?y ?x))", "sort"),
                     ("(compose (step ?a ?b ?c ?d) (step ?c ?d ?e ?f))", "locations"),             # width 1 over four slots: x,z of one slot each, the prefix is two columns, B's sorts differ
                     ("(compose (link ?s ?a ?b) (link ?t ?b ?c))", None), ("(closure (mark ?s ?x ?y))", "same sorts"), ("(exists (?q) (wall ?x ?y))", "does not occur"),
                     ("(compose (wall ?a ?b) (wall ?c ?d))", None)]:
        t = WT.parse(bad)
        if why is None:
            continue
        with pytest.raises(WT.TermError, match=why):
            WT.sort_of(t, sigs, sorts)
    with pytest.raises(WT.TermError):                                                 # the result's names must be distinct
        WT.free_vars(WT.parse("(compose (link ?s ?a ?b) (link ?s ?b ?a))"))


def test_compile_checks_the_name_and_the_arity():
    t = WT.parse("(wall ?x ?y)")
    for bad in ["Zed", "1zed", "step", "and", ""]:
        with pytest.raises(WT.TermError):
            WT.compile(bad, t, DESC.signatures)
    out = WT.compile("zed", t, DESC.signatures)
    assert out.splitlines()[0] == "(signature zed cx cy)" and "<=" in out
    assert WT.compile("zed", WT.parse("(wall ?x ?y)"), DESC.signatures, head_vars=["?y"]).splitlines()[0] == "(signature zed cy)"
    with pytest.raises(WT.TermError):
        WT.compile("zed", t, DESC.signatures, head_vars=["?q"])
    wide = WT.parse("(and (step ?a ?b ?c ?d) (step ?c ?d ?e ?f) (step ?e ?f ?g ?h))")                 # 8 variables: right at the evaluator's limit
    assert len(WT.free_vars(wide)) == 8 and WT.compile("zed", wide, DESC.signatures).startswith("(signature zed ")
    with pytest.raises(WT.TermError, match="widest"):
        WT.compile("zed", WT.parse("(and (step ?a ?b ?c ?d) (step ?c ?d ?e ?f) (step ?e ?f ?g ?h) (mark ?s ?a ?b))"), DESC.signatures)


# ------------------------------------------------------------------------------------------------------------------------------ compiled and evaluated: the made-up game (Python mirror)

def test_made_up_game_reach_two_steps_of_a_product_location():
    got = run_game("(reach (step ?a ?b ?c ?d) 2 2)")
    r1 = step_pairs()
    r2 = r1 | {(x, y) for (x, z) in r1 for (z2, y) in r1 if z == z2}
    assert got == flat(r2) and (("1", "1") + ("1", "1")) in got and (("1", "1") + ("3", "3")) not in got
    assert run_game("(reach (step ?a ?b ?c ?d) 1 2)") == flat(r1)
    r3 = r2 | {(x, y) for (x, z) in r2 for (z2, y) in r1 if z == z2}
    assert run_game("(reach (step ?a ?b ?c ?d) 3 2)") == flat(r3)


def test_made_up_game_reach_without_and_with_the_start_itself():
    r1 = step_pairs()
    strict = run_game("(reach (step ?a ?b ?c ?d) 1 2)")
    assert not any(row[:2] == row[2:] for row in strict)                             # x itself is not included: no cycle of one step
    with_zero = run_game("(reach0 (step ?a ?b ?c ?d) 1 2)")
    assert with_zero == flat(r1) | {c + c for c in CELLS}                             # every location the relation mentions, itself included
    link = "(link ?s ?a ?b)"
    assert run_game(f"(reach {link} 2)") == {("red", "1", "2"), ("red", "2", "3"), ("red", "1", "3"), ("blue", "1", "3"), ("blue", "3", "1"), ("blue", "1", "1"), ("blue", "3", "3")}
    assert ("red", "2", "2") in run_game(f"(reach0 {link} 2)") and ("red", "2", "2") not in run_game(f"(reach {link} 2)")


def test_made_up_game_via_keeps_the_middle_and_compose_drops_it():
    r1 = step_pairs()
    via = run_game("(via (step ?a ?b ?c ?d) (step ?c ?d ?e ?f) 2)")                    # columns: a b e f c d (x, y, then the middle)
    assert via == {x + y + z for (x, z) in r1 for (z2, y) in r1 if z == z2}
    comp = run_game("(compose (step ?a ?b ?c ?d) (step ?c ?d ?e ?f) 2)")
    assert comp == {x + y for (x, z) in r1 for (z2, y) in r1 if z == z2} and len(comp) < len(via)
    assert run_game("(compose (link ?s ?a ?b) (link ?s ?b ?c))") == {("red", "1", "3"), ("blue", "1", "1"), ("blue", "3", "3")}          # the prefix `side` is held
    assert run_game("(via (link ?s ?a ?b) (link ?s ?b ?c))") == {("red", "1", "3", "2"), ("blue", "1", "1", "3"), ("blue", "3", "3", "1")}


def test_made_up_game_closure_is_recursive_and_holds_the_prefix():
    r1 = step_pairs()
    assert run_game("(closure (step ?a ?b ?c ?d) 2)") == flat(closure_of(r1))
    assert len(run_game("(closure (step ?a ?b ?c ?d) 2)")) == 81
    text = WT.compile("zed", WT.parse("(closure (step ?a ?b ?c ?d) 2)"), DESC.signatures)
    assert "(zed ?x0 ?x1 ?w0 ?w1)" in text                                              # a recursive rule
    assert run_game("(closure (link ?s ?a ?b))") == {("red", "1", "2"), ("red", "2", "3"), ("red", "1", "3"), ("blue", "1", "3"), ("blue", "3", "1"), ("blue", "1", "1"), ("blue", "3", "3")}


def test_made_up_game_and_with_not_or_exists_distinct():
    r1 = step_pairs()
    assert run_game("(and (mark ?s ?x ?y) (not (wall ?x ?y)))") == {("red", "1", "1"), ("blue", "3", "3"), ("red", "3", "1")}
    assert run_game("(and (step ?a ?b ?c ?d) (not (wall ?c ?d)) (distinct ?a ?c))") == {a + b for (a, b) in r1 if b != WALL and a[0] != b[0]}
    marked = {(x, y) for (_s, (x, y)) in [(s, a) for _t, (s, *a) in STATE]}
    assert run_game("(and (step ?a ?b ?c ?d) (not (mark _ ?c ?d)))") == {a + b for (a, b) in r1 if b not in marked}          # a negated literal with `_`: a helper relation
    assert run_game("(or (mark red ?x ?y) (wall ?x ?y))") == {("1", "1"), ("3", "1"), ("2", "2")}
    assert run_game("(exists (?s) (mark ?s ?x ?y))") == {("1", "1"), ("3", "3"), ("3", "1")}
    assert run_game("(and (exists (?s) (mark ?s ?x ?y)) (not (or (wall ?x ?y) (mark blue ?x ?y))))") == {("1", "1"), ("3", "1")}          # a not over an `or` is a helper
    assert run_game("(exists (?c ?d) (step ?a ?b ?c ?d))") == {c for c in CELLS}
    assert run_game("(mark ?s ?x ?y)", state=set()) == set()


def test_the_compiled_text_is_additive_and_only_adds_relations():
    t = WT.compile("zed", WT.parse("(reach (step ?a ?b ?c ?d) 3 2)"), DESC.signatures)
    for line in t.splitlines():
        assert line.startswith("(signature zed") or line.startswith("(<= (zed")
    heads = {l.split()[1].lstrip("(") for l in t.splitlines() if l.startswith("(<=")}
    sigs = {l.split()[1] for l in t.splitlines() if l.startswith("(signature")}
    assert heads == sigs == {"zed", "zed__1", "zed__2"}


# ------------------------------------------------------------------------------------------------------------------------------ compiled and evaluated: chess (the Rust evaluator)

FENS = [START, chess.STARTING_FEN, "4k3/8/8/3p4/4P3/8/8/4K2R w K - 0 1", "r3k2r/pp3ppp/2n1b3/3q4/8/2N2N2/PPP2PPP/R2QK2R b KQkq - 3 9", "8/8/8/8/8/8/1k6/4KR2 b - - 21 11", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"]
SQ = [(f, r) for f in "abcdefgh" for r in "12345678"]


def _sq(sq):
    return chess.FILE_NAMES[chess.square_file(sq)], str(chess.square_rank(sq) + 1)


def _dist(a, b):
    return max(abs("abcdefgh".index(a[0]) - "abcdefgh".index(b[0])), abs(int(a[1]) - int(b[1])))


def chess_facts(terms: dict, fen: str, text: str = None) -> dict:
    """{word: set of rows} of the words compiled onto the keys description and read by the engine at `fen` (white / black, not us / them)."""
    spec = GS.spec_texts("chess", KEYS)["chess"]
    d = WT.read_description(spec)
    rules = "".join(WT.compile(n, WT.parse(t), d.signatures, fluents=d.fluents, sorts=d.sorts) for n, t in terms.items())
    full = CE.compose(spec, rules)
    voc = json.dumps({"derived": list(terms), "delta": ["cell"], "units": None})
    out = {n: set() for n in terms}
    for r in CE.facts("chess", full, voc, fen, [], False):
        if r[0] in out:
            out[r[0]].add(tuple(r[1:]))
    return out


def test_chess_description_fluents_and_signatures_are_read_from_the_text():
    d = WT.read_description(GS.spec_texts("chess", KEYS)["chess"])
    assert "cell" in d.fluents and "occupied" not in d.fluents and d.signatures["king_step"] == ("file", "rank", "file", "rank")


def test_chess_words_equal_a_brute_force_computation_on_positions():
    terms = {"w_reach": "(reach (king_step ?a ?b ?c ?d) 2 2)",
             "w_via": "(via (king_step ?a ?b ?c ?d) (king_step ?c ?d ?e ?f) 2)",
             "w_free": "(and (king_step ?a ?b ?c ?d) (not (occupied ?c ?d)))",
             "w_flood": "(closure (and (king_step ?a ?b ?c ?d) (not (occupied ?c ?d))) 2)",
             "w_or": "(or (cell ?f ?r white king) (cell ?f ?r black pawn))",
             "w_ex": "(exists (?k) (cell ?f ?r ?s ?k))",
             "w_comp": "(compose (king_step ?a ?b ?c ?d) (king_step ?c ?d ?e ?f) 2)"}
    seen_nonempty = set()
    for fen in FENS:
        got = chess_facts(terms, fen)
        b = chess.Board(fen)
        occ = {_sq(s) for s in b.piece_map()}
        step = {(x, y) for x in SQ for y in SQ if _dist(x, y) == 1}
        free = {(x, y) for (x, y) in step if y not in occ}
        want = {"w_reach": {x + y for x in SQ for y in SQ if 1 <= _dist(x, y) <= 2 or x == y},
                "w_via": {x + y + z for (x, z) in step for (z2, y) in step if z == z2},
                "w_free": {x + y for (x, y) in free},
                "w_flood": {x + y for (x, y) in closure_of(free)},
                "w_or": {_sq(s) for s, p in b.piece_map().items() if (p.piece_type == chess.KING and p.color == chess.WHITE) or (p.piece_type == chess.PAWN and p.color == chess.BLACK)},
                "w_ex": {(*_sq(s), "white" if p.color else "black") for s, p in b.piece_map().items()},
                "w_comp": {x + y for (x, z) in step for (z2, y) in step if z == z2}}
        for n in terms:
            assert got[n] == want[n], (fen, n, sorted(got[n] ^ want[n])[:6])
            if want[n]:
                seen_nonempty.add(n)
    assert seen_nonempty == set(terms)


def test_a_two_step_reach_has_the_geometry_of_two_king_steps():
    got = chess_facts({"w_reach": "(reach (king_step ?a ?b ?c ?d) 2 2)", "w_one": "(reach (king_step ?a ?b ?c ?d) 1 2)", "w_zero": "(reach0 (king_step ?a ?b ?c ?d) 1 2)"}, START)
    assert ("e", "4", "g", "6") in got["w_reach"] and ("e", "4", "h", "4") not in got["w_reach"] and ("e", "4", "e", "4") in got["w_reach"]
    assert ("e", "4", "e", "4") not in got["w_one"] and ("e", "4", "e", "4") in got["w_zero"] and len(got["w_zero"]) == len(got["w_one"]) + 64


# ------------------------------------------------------------------------------------------------------------------------------ expansion and canonical form

def test_expand_replaces_every_word_by_its_term_and_canonical_names_variables_apart():
    reg = {"alba": "(and (key_gap ?s ?f ?r ?d) (not (king_in_front ?s ?f ?r)))", "bruno": "(and (alba ?s ?f ?r ?d) (not (hanging ?s ?f ?r)))"}
    e = WT.expand(WT.parse("(exists (?d) (bruno ?a ?b ?c ?d))"), reg)
    assert WT.show(e) == "(exists (?d) (and (and (key_gap ?a ?b ?c ?d) (not (king_in_front ?a ?b ?c))) (not (hanging ?a ?b ?c))))"
    assert WT.show(WT.expand(WT.parse("(alba white ?f ?r ?d)"), reg)) == "(and (key_gap white ?f ?r ?d) (not (king_in_front white ?f ?r)))"
    with pytest.raises(WT.TermError, match="more than once"):
        WT.expand(WT.parse("(alba us ?f _ ?d)"), reg)                                    # a word that uses a variable twice cannot take `_` there
    assert WT.canonical(WT.parse("(and (A ?x ?y) (B ?y ?z))")) == WT.canonical(WT.parse("(and (A ?p ?q) (B ?q ?r))"))
    assert WT.canonical(WT.parse("(and (A ?x ?y) (B ?y ?z))")) != WT.canonical(WT.parse("(and (A ?x ?y) (B ?z ?y))"))
    with pytest.raises(WT.TermError):
        WT.expand(WT.parse("(a ?x)"), {"a": "(a ?x)"})                                 # a word defined by itself
    with pytest.raises(WT.TermError):
        WT.expand(WT.parse("(w ?x ?y)"), {"w": "(wall ?x)"})                            # wrong number of slots
    with pytest.raises(WT.TermError, match="distinct variables"):
        WT.expand(WT.parse("(w ?a ?a ?c ?d)"), {"w": "(closure (step ?x ?y ?z ?q) 2)"})
    # a bound variable of the word is renamed apart from the caller's variables
    e2 = WT.expand(WT.parse("(w ?z)"), {"w": "(and (exists (?z) (a ?z ?y)) (b ?y))"})
    assert WT.free_vars(e2) == ["?z"] and WT.show(e2) != "(and (exists (?z) (a ?z ?z)) (b ?z))" and "(exists (?w" in WT.show(e2)


# ------------------------------------------------------------------------------------------------------------------------------ names

def test_names_are_distinct_legal_pronounceable_and_collide_with_no_relation_or_symbol():
    ns = WN.names()
    assert 180 <= len(ns) <= 260 and len(set(ns)) == len(ns)
    assert all(WN.LEGAL.match(n) and 3 <= len(n) <= 9 for n in ns)
    assert ns[:3] == ["alba", "bruno", "celia"] and [n[0] for n in ns[:26]] == list("abcdefghijklmnopqrstuvwxyz")        # the alphabet cycles
    symbols = set()
    for vocab in (None, "relational", KEYS):
        for text in GS.spec_texts("chess", vocab).values():
            symbols |= set(re.findall(r"[^\s()]+", "\n".join(l.split(";", 1)[0] for l in text.splitlines())))
    for p in (RULES / "chess.gdl", *(RULES / "concepts").glob("*.gdl"), *(RULES / "variants").glob("*.gdl")):
        symbols |= set(re.findall(r"[^\s()]+", "\n".join(l.split(";", 1)[0] for l in p.read_text().splitlines())))
    assert not set(ns) & (symbols | set(WT.KEYWORDS) | {"true", "next", "init", "does", "legal", "goal", "terminal", "role", "base", "input"})


def test_next_name_is_the_first_unused_in_the_list_order_and_the_list_can_run_out():
    ns = WN.names()
    assert WN.next_name(set()) == ns[0] and WN.next_name({ns[0]}) == ns[1] and WN.next_name([ns[0], ns[2]]) == ns[1]
    assert WN.next_name({ns[0], ns[1], ns[2]}) == ns[3] == WN.next_name({ns[2], ns[1], ns[0]})              # deterministic
    with pytest.raises(WN.NamesExhausted):
        WN.next_name(set(ns))


# ------------------------------------------------------------------------------------------------------------------------------ the registry in a sandbox store

@pytest.fixture
def store(tmp_path):
    old = GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT
    d = tmp_path / "learned" / "chess"
    d.mkdir(parents=True)
    vf = json.loads((RULES / "learned/chess/vocabulary.keys.json").read_text())
    vf["words"] = True
    (d / "vocabulary.keys.json").write_text(json.dumps(vf))
    GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT = tmp_path / "learned", tmp_path / "snapshots"
    try:
        yield tmp_path
    finally:
        GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT = old


ALBA = "(and (key_gap ?s ?f ?r ?d) (not (king_in_front ?s ?f ?r)))"
BRUNO = "(and (alba ?s ?f ?r ?d) (not (hanging ?s ?f ?r)))"


def test_define_names_in_list_order_dedupes_and_keeps_the_definition_order(store):
    a = WR.define("chess", ALBA, origin="hand", vocabulary=KEYS, note="a key gap of a pawn that is not blocked")
    assert a["name"] == "alba" and a["origin"] == "hand" and a["size"] == 4 and a["signature"] == ["side", "file", "rank", "distance"] and a["expansion"] == WT.show(WT.parse(ALBA))
    b = WR.define("chess", BRUNO, origin="generated", vocabulary=KEYS)
    assert b["name"] == "bruno" and b["expansion"] == "(and (and (key_gap ?s ?f ?r ?d) (not (king_in_front ?s ?f ?r))) (not (hanging ?s ?f ?r)))" and b["size"] == 4
    again, created = WR.define_new("chess", "(and (key_gap ?a ?b ?c ?e) (not (king_in_front ?a ?b ?c)))", origin="mined", vocabulary=KEYS)       # the same, other variable names
    assert again["name"] == "alba" and created is False
    nested, created = WR.define_new("chess", "(and (and (key_gap ?s ?f ?r ?d) (not (king_in_front ?s ?f ?r))) (not (hanging ?s ?f ?r)))", origin="mined", vocabulary=KEYS)    # the expansion of bruno
    assert nested["name"] == "bruno" and created is False
    assert [e["name"] for e in WR.load("chess")["words"]] == ["alba", "bruno"]
    d = WR.delta("chess")
    assert d.index("(signature alba ") < d.index("(signature bruno ") and d.index("(<= (alba ") < d.index("(<= (bruno ")
    c = WR.define("chess", "(and (attack ?s ?a ?b ?c ?d) (not (defended ?s ?c ?d)))", origin="mined", vocabulary=KEYS)
    assert c["name"] == "celia" and c["layer"] == "general"
    with pytest.raises(ValueError, match="itself"):
        WR.define("chess", "(key_gap ?a ?b ?c ?d)", origin="hand", vocabulary=KEYS)
    with pytest.raises(ValueError, match="already"):
        WR.define("chess", "(and (hanging ?s ?f ?r) (not (defended ?s ?f ?r)))", origin="hand", name="attack", vocabulary=KEYS)
    with pytest.raises(ValueError):
        WR.define("chess", "(and (hanging ?s ?f ?r) (not (defended ?s ?f ?r)))", origin="made up", vocabulary=KEYS)
    assert (GS.learned_dir("chess") / "words.json").is_file() and not (RULES / "learned/chess/words.json").exists()


def test_a_name_is_never_a_relation_of_the_description(store):
    used = WR._relations_of(GS.spec_texts("chess", KEYS)["chess"])
    assert "attack" in used and "key_gap" in used and "edge_gap_file" in used
    ns = WN.names()
    d = WR.define("chess", ALBA, origin="hand", vocabulary=KEYS)
    assert d["name"] == ns[0]
    assert WR.define("chess", "(and (hanging ?s ?f ?r) (not (defended ?s ?f ?r)))", origin="hand", vocabulary=KEYS)["name"] == ns[1]


def test_spec_texts_and_vocabulary_are_unchanged_without_words(store):
    vf = json.loads((RULES / "learned/chess/vocabulary.keys.json").read_text())
    assert "words" not in vf
    base = (RULES / vf["spec"]).read_text()
    t = base
    for c in vf["concepts"]:
        t = CE.compose(t, (RULES / c).read_text())
    by_hand = CE.compose(t, BS.structure_delta(t, vf))
    real = GS.LEARNED_ROOT
    GS.LEARNED_ROOT = None                                                       # the real store, read only: no `words` key
    try:
        assert GS.spec_texts("chess", KEYS)["chess"] == by_hand
        v = GS.vocabulary("chess", KEYS)
        assert v == {"derived": vf["derived"] + BS.relation_names(vf), "delta": vf["delta"], "units": None}
        for name in (None, "relational"):
            vf2 = GS.vocabulary_file("chess", name)
            assert "words" not in vf2 and GS.vocabulary("chess", name) == {"derived": list(vf2["derived"]) + BS.relation_names(vf2), "delta": list(vf2["delta"]), "units": vf2["units"]}
    finally:
        GS.LEARNED_ROOT = real
    assert GS.spec_texts("chess", KEYS)["chess"] == by_hand                          # `words: true` with no word defined: the same bytes
    assert GS.vocabulary("chess", KEYS)["derived"] == vf["derived"] + BS.relation_names(vf)


def test_a_word_is_a_derived_relation_the_engine_reads_and_a_lemma_with_it_matches(store):
    WR.define("chess", ALBA, origin="hand", vocabulary=KEYS)
    WR.define("chess", BRUNO, origin="hand", vocabulary=KEYS)
    specs, vocab = GS.spec_texts("chess", KEYS), GS.vocabulary("chess", KEYS)
    assert "(signature alba side file rank distance)" in specs["chess"] and vocab["derived"][-2:] == ["alba", "bruno"] and "alba" not in vocab["delta"]
    rows = CE.facts("chess", specs["chess"], json.dumps(vocab), START, [])
    by = {}
    for r in rows:
        by.setdefault(r[0], set()).add(tuple(r[1:]))
    assert by["alba"] and by["alba"] == by["key_gap"] - {t for t in by["key_gap"] if t[:3] in {u[:3] for u in by.get("king_in_front", set())}}
    assert by["bruno"] == by["alba"] - {t for t in by["alba"] if t[:3] in {u[:3] for u in by.get("hanging", set())}}
    hand = json.loads((RULES / "concepts/chess_keys.proposed.json").read_text())["rules"][0]                    # key_gap(us ?h0 ?h1 ?k0) -> key_gap(us ?h0 ?h1 d0)
    lem = {"name": "alba-response", "body": [AB.lit("alba", ["us", "?h0", "?h1", "?k0"])], "response": hand["response"], "stats": {"status": "HAND"}, "meta": {"origin": "hand", "vocabulary": KEYS}}
    GS.check_lemma(lem)
    cs = CE.build_cards([("chess", START, [], "p", "p")], specs, json.dumps(vocab), 1)
    named = AU.proposals(cs, lem)
    assert named and named == AU.proposals(cs, hand)                                                         # the same moves as the lemma written without the word
    vocab2 = dict(vocab)
    # a word listed under `word_delta` joins the vocabulary's `delta`
    vf = json.loads((GS.learned_dir("chess") / "vocabulary.keys.json").read_text())
    vf["word_delta"] = ["alba"]
    (GS.learned_dir("chess") / "vocabulary.keys.json").write_text(json.dumps(vf))
    assert GS.vocabulary("chess", KEYS)["delta"][-1] == "alba" and vocab2["delta"] == GS.vocabulary("chess", KEYS)["delta"][:-1]


def test_the_cli_prints_the_labels_and_the_words(store, capsys):
    WR.define("chess", ALBA, origin="hand", vocabulary=KEYS)
    WR.define("chess", "(and (attack ?s ?a ?b ?c ?d) (not (defended ?s ?c ?d)))", origin="mined", vocabulary=KEYS)
    WR.main(["chess", "--vocabulary", KEYS, "--store-root", str(store)])
    out = capsys.readouterr().out
    assert re.search(r"^OBS words layer chess keys: general \[.*'attack'.*\] game \[.*'hanging'.*\]$", out, re.M)
    assert f"OBS word alba (hand, game, size 4): {ALBA} = {ALBA}" in out
    assert re.search(r"^OBS word bruno \(mined, general, size 4\): ", out, re.M)


# ------------------------------------------------------------------------------------------------------------------------------ general or game

BASE = ("(sort c 1 2 3) (sort distance d0 d1 d2) (sort side red blue) (signature role side) (signature succ c c) (signature at side c)\n"
        "(role red) (role blue) (succ 1 2) (succ 2 3) (init (at red 1)) (<= (next (at ?s ?c)) (true (at ?s ?c)))\n"
        "(signature near c c) (<= (near ?a ?b) (succ ?a ?b))\n")
DELTA = ("(signature gen c c) (<= (gen ?a ?b) (near ?b ?a) (not (succ ?a ?a)))\n"
         "(signature far c c distance) (<= (far ?a ?b d1) (succ ?a ?b))\n"
         "(signature gap c c distance) (gap 1 1 d0) (gap 1 2 d1)\n"
         "(signature reads_gap c c) (<= (reads_gap ?a ?b) (gap ?a ?b d1))\n"
         "(signature named c) (<= (named ?a) (true (at red ?a)))\n"
         "(signature edge c distance) (edge 1 d0) (edge 2 d1)\n"
         "(signature uses_edge c) (<= (uses_edge ?c) (edge ?c d0))\n"
         "(signature loop c c) (<= (loop ?a ?b) (loop ?b ?a)) (<= (loop ?a ?b) (succ ?a ?b)) (<= (loop ?a ?b) (named ?a))\n"
         "(signature lonely c) (<= (lonely ?a) (true (at ?s ?a)) (not (named ?a)))\n")


def test_layer_of_made_up_descriptions():
    text = BASE + DELTA
    kw = dict(distance_sort="distance", base_text=BASE, generated=("edge",))
    lab = {r: WR.layer_of(r, text, **kw) for r in ("near", "gen", "far", "gap", "reads_gap", "named", "edge", "uses_edge", "loop", "lonely", "role", "at", "nosuch")}
    assert lab == {"near": "general", "gen": "general", "far": "general", "gap": "game", "reads_gap": "game", "named": "game", "edge": "general", "uses_edge": "general", "loop": "game",
                   "lonely": "game", "role": "general", "at": "general", "nosuch": "general"}
    a = WR.analyse(text, **kw)
    assert a["gap"]["why"] == "constant 1" and a["reads_gap"]["why"] == "reads gap -> constant 1" and a["named"]["why"] == "side constant red" and a["loop"]["why"].startswith("reads ")
    assert WR.layer_of("edge", text, distance_sort="distance", base_text=BASE) == "game"                       # without `generated` its coordinate facts count
    assert WR.layer_of("near", text, distance_sort="distance") == "game"                                      # without the base text a rule naming a constant... is judged by its clauses
    assert WR.layer_of("far", text, distance_sort="other") == "game"                                          # d1 is a distance member only for the named distance sort
    assert WR.layer_of("named", text, role_sorts=("side",), base_text=BASE, roles_general=True) == "general"      # (red is the only constant)
    assert WR.layer_of("lonely", text, base_text=BASE, role_sorts=("side",)) == "game"                       # (it reads `named`)
    assert WR.role_sorts_of(BASE) == ("side",)


def test_a_relation_the_description_defines_is_general_to_read_even_if_its_rules_name_constants():
    base = "(sort c 1 2) (sort kind a b) (signature rule2 c kind) (signature t c) (<= (rule2 ?x a) (t ?x)) (<= (t ?x) (rule2 ?x b))"
    delta = "(signature uses c) (<= (uses ?x) (rule2 ?x ?k))"
    assert WR.layer_of("rule2", base + delta, base_text=base) == "general" and WR.layer_of("uses", base + delta, base_text=base) == "general"
    assert WR.layer_of("rule2", base + delta) == "game"                                                        # with no base text its own constant `a` decides, and `uses` reads it
    assert WR.layer_of("uses", base + delta) == "game"


KEYS_TABLE = {"attack": "general", "check": "general", "defended": "general", "closer": "general", "hanging": "game", "passed_pawn": "game", "apart": "game", "flight": "game",
              "opposition": "game", "diagonal_opposition": "game", "to_promote": "game", "king_ahead": "game", "in_square": "game", "key_gap": "game", "key_guard": "game",
              "king_in_front": "game", "edge_gap_file": "general", "edge_gap_rank": "general", "goal_gap_rank": "general"}


def test_the_keys_vocabulary_labels_are_what_the_rule_says():
    lay = WR.layers("chess", KEYS, why=True)
    assert {r: l for r, (l, _w) in lay.items()} == KEYS_TABLE
    assert lay["hanging"][1] == "constant king" and lay["passed_pawn"][1] == "constant pawn" and lay["apart"][1] == "constant king" and lay["flight"][1] == "constant king"
    assert lay["opposition"][1] == "side constant white" and lay["diagonal_opposition"][1] == "side constant white" and lay["to_promote"][1] == "constant pawn"
    assert lay["king_ahead"][1] == "constant pawn" and lay["in_square"][1] == "constant king" and lay["king_in_front"][1] == "constant pawn"
    assert lay["key_gap"][1].startswith("reads ") and lay["key_gap"][1].endswith("constant king") and lay["key_guard"][1].endswith("constant king")
    assert WR.layers("chess", KEYS) == {r: l for r, (l, _w) in lay.items()}
    base = (RULES / "chess.gdl").read_text()
    text = GS.spec_texts("chess", KEYS)["chess"]
    assert WR.layer_of("attack", text, base_text=None) == "game" and WR.layer_of("attack", text, base_text=base) == "general"      # the base description's own rules are general to read


def test_rule_layer_and_origin():
    lay = {"closer": "general", "hanging": "game", "defended": "general"}
    gen = {"name": "g", "body": [AB.lit("closer", ["?a", "?b"]), AB.lit("cell", ["?f", "?r", "us", "?k"])], "response": {"removed": [], "added": [AB.lit("defended", ["us", "?f", "?r"])]}, "meta": {"origin": "hand"}}
    gm = {"name": "m", "body": [AB.lit("closer", ["?a", "?b"])], "response": {"removed": [AB.lit("hanging", ["us", "?f", "?r"])], "added": []}, "meta": {"origin": "induced"}}
    assert WR.rule_layer(gen, lay) == "general" and WR.rule_layer(gm, lay) == "game"
    assert WR.rule_layer({**gen, "body": gen["body"] + [AB.lit("hanging", ["us", "?f", "?r"])]}, lay) == "game"
    assert WR.origin_of(gen) == "hand" and WR.origin_of(gm) == "machine" and WR.origin_of({"name": "x"}) == "machine"
    keys_rules = json.loads((RULES / "concepts/chess_keys.proposed.json").read_text())["rules"]
    real = WR.layers("chess", KEYS)
    assert all(WR.rule_layer(r, real) == "game" and WR.origin_of(r) == "hand" for r in keys_rules)


# ------------------------------------------------------------------------------------------------------------------------------ naming a long body

def test_body_term_turns_the_lemma_body_into_a_term_text():
    lem = {"name": "x", "body": [AB.lit("key_gap", ["us", "?h0", "?h1", "?k0"]), AB.lit("key_guard", ["them", "?h0", "?h1", "?k1"]), {**AB.lit("closer", ["?k0", "?k1"]), "neg": True}],
           "response": {"removed": [], "added": [AB.lit("key_gap", ["us", "?h0", "?h1", "d0"])]}}
    t = IND.body_term(lem)
    assert t == "(exists (?k0 ?them ?k1) (and (key_gap ?us ?h0 ?h1 ?k0) (key_guard ?them ?h0 ?h1 ?k1) (not (closer ?k0 ?k1)) (distinct ?us ?them)))"
    assert WT.free_vars(WT.parse(t)) == ["?us", "?h0", "?h1"]


def tiny_corpus_of_chess():
    from test_plan78_32_exam_drill_induce import Fake, hashed, tiny_corpus          # noqa: E402
    return tiny_corpus(80), Fake(hashed)


def test_name_bodies_is_off_by_default_and_names_the_body_of_an_admitted_rule_without_changing_it(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "tests"))
    assert IND.SETTINGS["name_bodies"] == 0
    c, fake = tiny_corpus_of_chess()
    root = IN.prepare_sandbox("chess", tmp_path / "store")
    vpath = root / "learned" / "chess" / "vocabulary.keys.json"
    vf = json.loads(vpath.read_text())
    vf["words"] = True
    vpath.write_text(json.dumps(vf))
    rules = json.loads((RULES / "concepts/chess_keys.proposed.json").read_text())["rules"]
    hand = max(rules, key=lambda r: len(r["body"]))
    assert len(hand["body"]) >= 2
    found = {"status": "found", "lemma": None, "head": None, "dev": {"binds": 1, "right": 1, "wrong": 0, "trivial": 0, "names_a_move": 1}, "held": {"binds": 1, "right": 1, "wrong": 0, "trivial": 0, "names_a_move": 1},
             "why": "", "stage": "held", "heads_tried": 1, "candidates": [], "tier": "relative", "tiers": {}, "head_tier": None}
    start = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    rows = [{"start": start, "moves": [], "first_seen": 1, "history": []}]
    monkeypatch.setattr(IND.PS, "check", lambda ps, **kw: [{"move": "x", "keeps": False, "before": None, "after": None, "proved": False, "steps": [], "error": None} for _ in ps])

    def run(settings):
        lem = json.loads(json.dumps(hand))
        lem["name"] = "mined-by-hand-" + str(len(settings))
        monkeypatch.setattr(IND, "induce", lambda problem, corpus, **kw: {**found, "lemma": lem})
        lines = []
        with TG.sandbox(root):
            IND.solve_problems("chess", rows, c, labeller=fake, vocabulary=KEYS, spec=GS.spec_texts("chess", KEYS), settings=settings, say=lines.append)
            stored = GS.read_lemmas("chess", GS.ANY)
        return lines, lem, stored, json.loads(json.dumps(hand))

    lines, lem, stored, orig = run({})                                                                     # off: nothing is named
    assert "word" not in lem["meta"] and not any("OBS word" in l for l in lines) and not (root / "learned/chess/words.json").exists()
    lines, lem, stored, orig = run({"name_bodies": len(hand["body"]) + 1})                                  # a longer body than this one: nothing
    assert "word" not in lem["meta"] and not (root / "learned/chess/words.json").exists()
    lines, lem, stored, orig = run({"name_bodies": len(hand["body"])})
    reg = json.loads((root / "learned/chess/words.json").read_text())["words"]
    assert len(reg) == 1 and reg[0]["origin"] == "mined" and reg[0]["name"] == "alba" and lem["meta"]["word"] == "alba" and lem["meta"]["word_expansion"] == reg[0]["expansion"]
    assert lem["body"] == orig["body"] and lem["response"] == orig["response"]                              # what the rule matches is not changed
    assert [l for l in stored if l["name"] == lem["name"]][0]["meta"]["word"] == "alba"
    line = [l for l in lines if l.startswith("OBS word named alba:")]
    assert line and f"(rule {lem['name']}, size {reg[0]['size']})" in line[0]
    with TG.sandbox(root):
        assert "alba" in GS.vocabulary("chess", KEYS)["derived"] and "(signature alba " in GS.spec_texts("chess", KEYS)["chess"]      # available to the later rules (words: true)
        lines2 = []
        lem2 = json.loads(json.dumps(hand))
        lem2["name"] = "again"
        assert IND.name_body("chess", lem2, KEYS, say=lines2.append)["name"] == "alba" and lines2[0].startswith("OBS word reused alba:")       # one definition, one name


# ------------------------------------------------------------------------------------------------------------------------------ the use / generation split

class Keep:
    """A labeller by a table {start: {move: keeps}}."""
    def __init__(self, table):
        self.table = table

    def moves_keep(self, variant, start, moves):
        return self.table.get(start)


def test_usage_split_hits_every_bucket(tmp_path, monkeypatch):
    lemmas = [{"name": "r1", "body": [AB.lit("closer", ["?a", "?b"])], "response": {"removed": [], "added": []}, "meta": {"origin": "induced"}},
              {"name": "r2", "body": [AB.lit("hanging", ["us", "?f", "?r"])], "response": {"removed": [], "added": []}, "meta": {"origin": "hand"}}]
    snap = tmp_path / "s.json"
    snap.write_text(json.dumps({"vocabulary": {"derived": [], "delta": [], "units": None}, "lemmas": lemmas}))
    table = {"p0": {"a": True, "b": False, "c": False}, "p1": {"a": True, "b": False, "c": False}, "p2": {"a": True, "b": False}, "p3": {"a": True, "b": False}, "p4": {"a": True, "b": False},
             "p5": {"a": True, "b": False}, "p6": {"a": True, "b": False}, "p7": {"a": True, "b": False}}
    named = {"p0": {"r1": ["a"]}, "p1": {"r1": ["b"]}, "p2": {}, "p3": {"r1": ["a"]}, "p4": {"r1": ["a"], "r2": ["b"]}, "p5": {"r2": ["b"]}, "p6": {}, "p7": {"r2": ["a"]}}
    played = {"p0": ("a", True), "p1": ("a", True), "p2": ("a", True), "p3": ("b", False), "p4": ("b", False), "p5": ("b", False), "p6": ("b", False), "p7": ("a", True)}
    exam = {"positions": [{"start": s, "moves": [], "group": "g" + s[1], "before": "win"} for s in table]}
    graded = {"decisions": [{"i": i, "move": played[s][0], "keeps": played[s][1], "before": "win"} for i, s in enumerate(table)]}
    monkeypatch.setattr(US.CE, "build_cards", lambda items, *a, **k: items[0][1])               # the "card set" is just the position's text
    monkeypatch.setattr(US.AU, "proposals", lambda cs, lem: named[cs].get(lem["name"], []))
    res = US.split(exam, graded, variant="v", snapshot_path=str(snap), specs={}, labeller=Keep(table))
    assert res["kept"] == {"named_played": 2, "named_other": 1, "no_rule": 1}
    assert res["failed"] == {"usage": 2, "usage_conflict": 1, "wrong_rule": 1, "no_rule": 1}
    assert [r["bucket"] for r in res["rows"]] == ["named_played", "named_other", "no_rule", "usage", "usage_conflict", "wrong_rule", "no_rule", "named_played"]
    assert res["by_group"] == {"g3": 1, "g4": 1, "g5": 1, "g6": 1}
    lines = []
    US.say_split("t", res, say=lines.append)
    assert lines == ["OBS usage split t: kept 4 (a rule named the move played 2, rules named other moves only 1, no rule named a move 1); failed 4 (a rule named a keeping move and another was played 2, "
                     "of them with a rule also naming a losing move 1; rules named only losing moves 1; no rule named a move 1)"]
    lay = {"closer": "general", "hanging": "game"}
    lines.clear()
    US.say_split("t", res, lay, say=lines.append)
    assert lines[1] == ("OBS usage split by rule kind t: a rule named the move played in 2 decisions: named by a general rule 1 (only by game rules 1), by a hand rule 1 (only by machine rules 1)")
    lines.clear()
    c = US.say_rules_by_kind(lemmas, lay, say=lines.append)
    assert lines == ["OBS rules by kind: general 1 (hand 0, machine 1), game 1 (hand 1, machine 0)"] and c["game"] == {"hand": 1, "machine": 0}


def test_usage_split_with_the_engines_own_matcher(store):
    """One real rule of the sandbox store on a real position: the matcher names the move, the decision is graded by hand."""
    rules = json.loads((RULES / "concepts/chess_keys.proposed.json").read_text())["rules"]
    GS.write_lemma("chess", rules[0])
    sid = GS.publish("chess", vocabulary=KEYS)
    legal = CE.state("chess", START, [])[2]
    keep = {m: m == "d4e4" for m in legal}
    exam = {"positions": [{"start": START, "moves": [], "group": "g", "before": "win"}] * 2}
    graded = {"decisions": [{"i": 0, "move": "d4e4", "keeps": True, "before": "win"}, {"i": 1, "move": "d4d5", "keeps": False, "before": "win"}]}
    res = US.split(exam, graded, variant="chess", snapshot_path=str(GS.path_of("chess", sid)), specs=GS.spec_texts("chess", KEYS), labeller=Keep({START: keep}))
    assert res["rows"][0]["named"] == {rules[0]["name"]: ["d4e4"]} and res["kept"]["named_played"] == 1 and res["failed"]["usage"] == 1 and res["failed"]["usage_conflict"] == 0


# ------------------------------------------------------------------------------------------------------------------------------ the audit and the drill (78.34 open point 5)

def test_a_position_that_is_lost_whatever_black_plays_is_trivial_to_the_drill_though_history_makes_a_move_a_draw():
    """The four 'refuted' firings of run M's audit are ONE position (a rook against a lone king, the lone side to move, lost), reached with the history of a repetition: the move
    Kb2-a2 would complete a threefold repetition, which only a teacher that is given the moves sees; the exact labeller (a position, no history) says every move keeps the loss, so the drill
    counts the position trivial. A difference of standard, not a bug."""
    start, fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1", "8/8/8/8/8/8/1k6/4KR2 b - - 21 11"
    moves = "h1h7 e8f8 h7h1 f8e7 h1f1 e7d6 f1g1 d6c5 g1f1 c5b4 f1g1 b4a3 g1f1 a3a2 f1g1 a2b2 g1f1 b2a2 f1g1 a2b2 g1f1".split()
    assert CE.state("chess", start, moves)[0] == fen
    b = chess.Board(start)
    for m in moves:
        b.push_uci(m)
    assert b.fen() == fen
    after = b.copy()
    after.push_uci("b2a2")
    assert after.can_claim_threefold_repetition()                                                # the history makes Kb2-a2 a draw
    c = b.copy()
    c.push_uci("b2c1")
    assert not c.can_claim_threefold_repetition()


# ------------------------------------------------------------------------------------------------------------------------------ the switches, the gate, the structure file

def test_the_switches_are_off_by_default_and_the_flags_parse():
    assert inspect.signature(IN.run_size).parameters["usage_split"].default is False and IND.SETTINGS["name_bodies"] == 0
    src = type("S", (), {"grade_mode": "score"})()
    with pytest.raises(ValueError, match="exam"):
        IN.run_size("chess", 3, source=src, usage_split=True, say=QUIET)
    text = (DISC / "initialize.py").read_text()
    assert '"--usage-split"' in text and "usage_split=a.usage_split" in text and "name_bodies=N" in text
    assert IN.settings_pairs is not None


def test_the_new_modules_name_no_game_object_and_the_wake_never_reads_them():
    forbid = re.compile(r"\b(king|queen|rook|bishop|knight|pawn|file|files|rank|ranks|square|squares|chess\w*|piece|pieces|castl\w*|promot\w*|fen)\b", re.I)
    for name in ("word_terms.py", "word_names.py", "word_registry.py", "usage_split.py"):
        text = (DISC / name).read_text()
        assert forbid.search(text) is None, (name, forbid.search(text).group(0))
    for name in ("wake.py", "wdl_table.py"):
        t = (DISC / name).read_text()
        assert not any(w in t for w in ("word_terms", "word_registry", "usage_split", "word_names"))
    for p in sorted((ROOT / "rust/engine").glob("*/src/**/*.rs")):
        t = p.read_text()
        assert "word_terms" not in t and "words.json" not in t
    names_text = (RULES / "concepts/names.txt").read_text()
    assert "chess" not in names_text
    assert "terms import" not in (DISC / "word_terms.py").read_text() and "terms as" not in (DISC / "word_terms.py").read_text()          # October's evaluator is not on the path


def test_the_new_files_are_in_the_gate_and_in_repo_structure():
    assert "test_plan78_35_words" in (ROOT / "scripts/gate.py").read_text()
    rs = (ROOT / "repo_structure.md").read_text()
    for f in ("word_terms.py", "word_names.py", "word_registry.py", "usage_split.py", "names.txt", "name_body", "layer_of", "test_plan78_35_words.py"):
        assert f in rs
