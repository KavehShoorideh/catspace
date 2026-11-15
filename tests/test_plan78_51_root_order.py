"""Plan 78.51: the fitted number as the order of the root moves (rust/engine/tactics/src/ledger.rs `set_root_order`; `cs_engine.solve_batch` / `solve_equals_batch` `root_orders`, `root_order_mode`;
catspace/incubator/analyst/discovery/subgoal_order_measure.py).

1. The default path is unchanged: a golden file recorded from the build BEFORE the order existed (40 positions, ledger depth fixed at 3, no clock cut) holds the move, the ledger's depth, value and nodes of
   `solve_batch` and the equal moves of `solve_equals_batch`; the new build without an order reproduces every field.
2. The statement of the pick: in "strict" mode the move kept is the first one IN THE GIVEN ORDER among the moves `solve_equals_batch` reads as equal to the best (positions read whole); the given order changes
   neither the value, the depth nor the set of equals.  "ties" mode keeps a move that holds the best value.
3. The plumbing, the source scan (the script and the added engine code name no object of any game) and a tiny end-to-end match."""
import json
import random
import re
from pathlib import Path

import pytest

import cs_engine as CE

from catspace.incubator.analyst.discovery import subgoal_order_measure as OM
from catspace.incubator.analyst.discovery import subgoal_play_measure as PM

ROOT = Path(__file__).resolve().parents[1]
GAME_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
GOLDEN = json.loads((ROOT / "tests" / "golden_78_51_default_path.json").read_text())
VARIANT = GOLDEN["variant"]
DEPTH = GOLDEN["ledger_depth"]
SECONDS = GOLDEN["seconds"]
POSITIONS = [p["moves"] for p in GOLDEN["positions"]]


def store_and_specs():
    from catspace.incubator.analyst.discovery import game_store as GS
    store = str(GS.path_of(VARIANT, GS.head(VARIANT))) if GS.head(VARIANT) else None
    return store, (GS.spec_texts(VARIANT) if store else {})


def ledger_rows(trace_path, n):
    rows = {}
    for line in open(trace_path):
        r = json.loads(line)
        if r.get("kind") == "position":
            rows[int(r["label"].split(":")[1])] = next((s for s in r["steps"] if s.get("operator") == "ledger"), {})
    return [rows.get(i, {}) for i in range(n)]


def solve(positions, tmp_path, depth=DEPTH, **extra):
    store, specs = store_and_specs()
    tp = str(tmp_path / "trace.jsonl")
    Path(tp).unlink(missing_ok=True)
    out = CE.solve_batch([(VARIANT, None, m) for m in positions], SECONDS, snapshot="s", trace_path=tp, threads=1, store=store, specs=specs,
                         labels=[f"s:{i}:0:0" for i in range(len(positions))], ledger_depth=depth, **extra)
    return out, ledger_rows(tp, len(positions))


def equals(positions, depth=DEPTH, **extra):
    store, specs = store_and_specs()
    return [PM.equal_set(r) for r in CE.solve_equals_batch([(VARIANT, None, m) for m in positions], SECONDS, store=store, specs=specs, threads=1, ledger_share=0.8, ledger_depth=depth, **extra)]


# ------------------------------------------------------------------------------------------------------------------------------ 1. the default path is unchanged

def test_without_an_order_the_search_is_what_it_was(tmp_path):
    """Moves, ledger depths, values and nodes, and the equal moves, are exactly those the build before the order recorded."""
    out, led = solve(POSITIONS, tmp_path)
    for g, r, s in zip(GOLDEN["positions"], out, led):
        assert (r[0], bool(r[1]), s.get("nodes"), s.get("depth"), s.get("value")) == (g["move"], g["proved"], g["ledger_nodes"], g["ledger_depth"], g["ledger_value"]), g["moves"]
    for g, d in zip(GOLDEN["equals"], equals(POSITIONS)):
        assert (d["move"], d["equal_indices"], d["depth"], d["value"], d["legal"], bool(d["cut"])) == (g["move"], g["equal_indices"], g["ledger_depth"], g["ledger_value"], g["legal"], g["cut"])


def test_an_absent_order_is_the_same_whichever_way_it_is_absent(tmp_path):
    a, la = solve(POSITIONS[:12], tmp_path)
    b, lb = solve(POSITIONS[:12], tmp_path, root_orders=None)
    c, lc = solve(POSITIONS[:12], tmp_path, root_orders=[None] * 12, root_order_mode="ties")
    key = lambda out, led: [(r[0], bool(r[1]), s.get("nodes"), s.get("depth"), s.get("value")) for r, s in zip(out, led)]
    assert key(a, la) == key(b, lb) == key(c, lc)


def test_a_wrong_order_is_told_not_repaired(tmp_path):
    store, specs = store_and_specs()
    n = len(CE.state(VARIANT, None, POSITIONS[0])[2])
    pos = [(VARIANT, None, POSITIONS[0])]
    for bad in (list(range(n - 1)), list(range(n)) + [0], [n] + list(range(n - 1)), [0] * n):
        row = CE.solve_batch(pos, 0.2, store=store, specs=specs, threads=1, root_orders=[bad])[0]
        assert row[0] is None and "permutation" in row[4]
        assert CE.solve_equals_batch(pos, 0.2, store=store, specs=specs, threads=1, root_orders=[bad])[0][4] is not None
    with pytest.raises(ValueError):
        CE.solve_batch(pos, 0.2, root_orders=[None, None])                                # one entry per position
    with pytest.raises(ValueError):
        CE.solve_batch(pos, 0.2, root_orders=[list(range(n))], root_order_mode="fastest")
    with pytest.raises(ValueError):
        CE.solve_batch(pos, 0.2, root_orders=[list(range(n))], ordering=False)            # the order is read by the ordered search only


# ------------------------------------------------------------------------------------------------------------------------------ 2. what the search picks

def shuffled(rng, positions):
    out = []
    for m in positions:
        o = list(range(len(CE.state(VARIANT, None, m)[2])))
        rng.shuffle(o)
        out.append(o)
    return out


def test_strict_order_pick_is_the_first_in_the_order_among_the_equals(tmp_path):
    """The statement: with root_order_mode 'strict' the move kept is the first move in the given order among the moves held equal to the best at the finished depth (positions the equals entry read whole)."""
    rng = random.Random(51)
    legal = [CE.state(VARIANT, None, m)[2] for m in POSITIONS]
    plain_out, plain_led = solve(POSITIONS, tmp_path)
    eq = equals(POSITIONS)
    whole = [i for i, d in enumerate(eq) if not d["cut"]]
    assert len(whole) == len(POSITIONS) and any(len(eq[i]["equal_indices"]) > 1 for i in whole)         # (the test has several-equal positions to say something about)
    for _trial in range(3):
        orders = shuffled(rng, POSITIONS)
        out, led = solve(POSITIONS, tmp_path, root_orders=orders, root_order_mode="strict")
        eq_o = equals(POSITIONS, root_orders=orders, root_order_mode="strict")
        for i in whole:
            pick = legal[i].index(out[i][0])
            first = min(eq[i]["equal_indices"], key=orders[i].index)
            assert pick == first, (POSITIONS[i], pick, first, eq[i]["equal_indices"])
            assert (led[i].get("depth"), led[i].get("value")) == (plain_led[i].get("depth"), plain_led[i].get("value"))      # an order, never a verdict
            assert eq_o[i]["equal_indices"] == eq[i]["equal_indices"] and eq_o[i]["move"] == out[i][0]
    # the order the engine would have read anyway (index order) keeps the search's own pick only where that pick is the lowest-index equal move: the statement is about the given order, not the root's own
    ident = [list(range(len(lg))) for lg in legal]
    out, _led = solve(POSITIONS, tmp_path, root_orders=ident, root_order_mode="strict")
    assert all(legal[i].index(out[i][0]) == eq[i]["equal_indices"][0] for i in whole)


def test_strict_order_pick_at_a_deeper_finished_depth(tmp_path):
    rng = random.Random(52)
    pos = POSITIONS[:14]
    legal = [CE.state(VARIANT, None, m)[2] for m in pos]
    orders = shuffled(rng, pos)
    out, led = solve(pos, tmp_path, depth=4, root_orders=orders, root_order_mode="strict")
    eq = equals(pos, depth=4)
    for i, d in enumerate(eq):
        if d["cut"] or led[i].get("depth") != 4:
            continue
        assert legal[i].index(out[i][0]) == min(d["equal_indices"], key=orders[i].index)


def test_ties_mode_keeps_a_move_that_holds_the_best_value(tmp_path):
    rng = random.Random(53)
    legal = [CE.state(VARIANT, None, m)[2] for m in POSITIONS]
    plain_out, plain_led = solve(POSITIONS, tmp_path)
    eq = equals(POSITIONS)
    orders = shuffled(rng, POSITIONS)
    out, led = solve(POSITIONS, tmp_path, root_orders=orders, root_order_mode="ties")
    for i, d in enumerate(eq):
        assert legal[i].index(out[i][0]) in d["equal_indices"]
        assert (led[i].get("depth"), led[i].get("value")) == (plain_led[i].get("depth"), plain_led[i].get("value"))


# ------------------------------------------------------------------------------------------------------------------------------ 3. the plumbing

def test_the_order_of_numbers():
    assert OM.order_of({3: 0.5, 1: 0.9, 2: 0.5}, 5) == [1, 2, 3, 0, 4]              # best first; a tie in index order; the unnumbered after
    assert OM.order_of({}, 3) == [0, 1, 2]
    o = OM.random_order(9, random.Random("x:1"))
    assert sorted(o) == list(range(9)) and o == OM.random_order(9, random.Random("x:1")) and o != list(range(9))


def test_the_matches_are_registered_with_new_seeds_and_the_weights():
    assert OM.FIRST_SEED == 9000 and set(OM.MATCHES) == {"ordered_v_control", "ordered_ties_v_control", "random_order_v_control", "random_order_ties_v_control"}
    old = set(PM.opening_seeds(PM.FIRST_SEED, 300)) | set(PM.opening_seeds(PM.NOISE_SEED, 300))
    assert old.isdisjoint(PM.opening_seeds(OM.FIRST_SEED, 300))
    specs = PM.specs_of("ordered_v_control", VARIANT, 2, 0.05, "w.json", "scratch")
    assert [(s["seed"], s["colour"]) for s in specs] == [(9000, 0), (9000, 1), (9001, 0), (9001, 1)] and all(s["weights"] == "w.json" and s["plugin"] == OM.PLUGIN for s in specs)
    assert PM.specs_of("candidate_v_control", VARIANT, 1, 0.05, "w.json", "scratch")[0]["plugin"] is None          # the 78.50 matches are as they were
    assert OM.DEFAULT_WEIGHTS.endswith("weights_horizon8.json")


def test_the_sample_of_positions_is_seeded_and_the_batch_variants_are_all_there():
    recs = [{"game": g, "moves": ["a"] * 7} for g in range(10)]
    s = OM.sample_positions(recs, 25, 5)
    assert s == OM.sample_positions(recs, 25, 5) and len(set(s)) == 25 and all(0 <= t < 7 for _g, t in s)
    legal = [list("abcd"), list("ab")]
    v = OM.b_variants([(0, 0), (0, 1)], None, {(0, 0): {0: 1.0, 3: 2.0}, (0, 1): {}}, legal, 1)
    assert set(v) == {"today", "today_again", "number_strict", "random_strict", "number_ties", "random_ties"}
    assert v["today"] == (None, None) and v["number_strict"][0][0] == [3, 0, 1, 2] and v["number_ties"][1] == "ties"


# ------------------------------------------------------------------------------------------------------------------------------ the source is game-free

def test_the_script_and_the_added_engine_code_name_no_game_object():
    assert not re.findall(GAME_WORDS, (ROOT / "catspace/incubator/analyst/discovery/subgoal_order_measure.py").read_text(), re.I)
    keys = ("78.51", "root_pos", "root_strict", "RootOrder", "set_root_order", "root_order", "root_place", "order_is_valid")
    for f in ("rust/engine/tactics/src/ledger.rs", "rust/engine/tactics/src/operators.rs", "rust/engine/decision/src/lib.rs", "rust/engine/py/src/lib.rs"):
        for n, line in enumerate((ROOT / f).read_text().splitlines(), 1):
            if any(k in line for k in keys):
                assert not re.findall(GAME_WORDS, line, re.I), (f, n, line)
    rust = (ROOT / "rust/engine/py/src/lib.rs").read_text()
    block = rust[rust.index("/// Plan 78.51: the optional root order"):rust.index("type BatchItem")]
    assert block and not re.findall(GAME_WORDS, block, re.I)


# ------------------------------------------------------------------------------------------------------------------------------ a tiny match

def test_a_tiny_match_of_two_pairs_runs_end_to_end(tmp_path):
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.incubator.analyst.discovery import sheet_rust as SR
    model = SR.model(QS.Env(VARIANT), QS.store_worth(VARIANT))
    names = [n for n, _t in model.subgoal_names()]
    wfile = tmp_path / "w.json"
    wfile.write_text(json.dumps({"weights": [1.0 if n == "material" else 0.0 for n in names], "names": names}))
    for name in ("ordered_v_control", "random_order_ties_v_control"):
        games = PM.run_match(name, VARIANT, 2, 1, 0.03, str(wfile), tmp_path / name, say=lambda *a, **k: None)
        assert len(games) == 4 and sorted((g["seed"], g["colour"]) for g in games) == [(9000, 0), (9000, 1), (9001, 0), (9001, 1)]
        assert all(g["outcome"] is not None or g["unfinished"] for g in games) and len(PM.pair_scores(games)) == 2
        assert all(len(g["moves"]) > g["opening_len"] for g in games)
        lines = PM.match_lines(name, *OM.MATCHES[name][:2], games, draws=20)
        assert "games 4, opening pairs 2" in lines[0]
        a_rows = [r for g in games for r in g["log_a"]]
        assert a_rows and all(r[2]["legal"] >= 1 for r in a_rows) and any(r[2]["ordered"] for r in a_rows)
