"""Plan 78.54: what becomes of each unit a concept binds (rust/engine/tactics/src/sheet/fates.rs, base_graph.rs `through`, patterns.rs; catspace/incubator/analyst/discovery/unit_fates*.py).

A. the line through one unit for both sides: present in the hand positions and absent in their twins, equal to a brute walk over the layer-1 graph, in agreement with the rules' own answer for the side to move.
B. the tracker: against hand sequences of positions (a move, a taking, a unit changing kind, a unit that returns) and against the readable mirror on random games; two units of one owner and kind that are gone and
   appeared together are paired as they fall (a Rust unit test, counted).
C. the fates of a pattern's variables against the readable matcher on random games.
D. the rules of selection and the arithmetic of a cell.
F. the check by eye: the hand positions against the concepts the 500-game run kept (printed; skipped when the run's file is not there).
No module names an object of any game (a source scan); the hand positions and the names of units appear here and nowhere in the modules."""
import json
import random
import re
from pathlib import Path

import cs_engine as CE
import numpy as np
import pytest

from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2 as G2
from catspace.incubator.analyst.discovery import graph_layer2_match as GM
from catspace.incubator.analyst.discovery import unit_fates as UF
from catspace.incubator.analyst.discovery import unit_fates_measure as UM

ROOT = Path(__file__).resolve().parents[1]
GOAL = "king"
SCAN_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|diagonals?|[a-h][1-8]|pins?|pinned|forks?|skewers?|batter\w+|x-?rays?|discovered)\b"
MODULES = ["rust/engine/tactics/src/sheet/fates.rs", "rust/engine/tactics/src/sheet/patterns.rs", "rust/engine/tactics/src/sheet/base_graph.rs", "catspace/incubator/analyst/discovery/unit_fates.py",
           "catspace/incubator/analyst/discovery/unit_fates_measure.py", "catspace/incubator/analyst/discovery/graph_layer2.py", "catspace/incubator/analyst/discovery/graph_layer2_match.py"]


def random_games(seed: int, games: int, plies: int) -> list:
    rng = random.Random(seed)
    out = []
    for _ in range(games):
        moves = []
        for _ in range(plies):
            state = CE.state("chess", None, moves)
            if state[3] is not None or not state[2]:
                break
            moves.append(rng.choice(sorted(state[2])))
        out.append(moves)
    return out


@pytest.fixture(scope="module")
def model():
    m = GL.model("chess", "chess")
    m.learn_profile([(None, mv) for mv in random_games(21, 12, 60)])
    return m


@pytest.fixture(scope="module")
def kinds(model):
    return [n[len("kind:"):] for t, n in model.static_part()["nodes"] if t == "kind"]


def through_ids(model, fen: str) -> list:
    return GL.graph(model, fen).of_type("through")


def has_through(model, fen: str, kind: str, place: str, to: str) -> bool:
    return any(f":{kind}:{place}," in n and n.endswith(f"place:{to})") for n in through_ids(model, fen))


# ------------------------------------------------------------------------------------------------------------------------------ A: the line through one unit

def test_the_profile_is_read_from_the_bear_nodes_and_the_fresh_model_has_no_through_nodes(model, kinds):
    prof = model.profile()
    assert prof and all(k in kinds for k, _d, _f in prof)
    assert {k for k, _d, _f in prof} < set(kinds)                      # some kinds were never seen bearing two steps away
    fresh = GL.model("chess", "chess")
    assert fresh.profile() == []
    assert through_ids(fresh, "4k3/8/8/4n3/8/8/8/4R1K1 w - - 0 1") == []


HAND = {
    "a line through the unit between, the side to move holds it": ("4k3/8/8/4n3/8/8/8/4R1K1 w - - 0 1", "3k4/8/8/4n3/8/8/8/4R1K1 w - - 0 1", ("rook", "e1", "e8")),
    "a line through an own unit": ("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1", "3k4/8/8/8/4N3/8/8/4R1K1 w - - 0 1", ("rook", "e1", "e8")),
    "a line through the goal unit of theirs, held by the side NOT to move": ("4q3/8/8/4k3/8/8/8/4R1K1 b - - 0 1", "3q4/8/8/4k3/8/8/8/4R1K1 b - - 0 1", ("rook", "e1", "e8")),
    "the first one again, the other side to move": ("4k3/8/8/4n3/8/8/8/4R1K1 b - - 0 1", "3k4/8/8/4n3/8/8/8/4R1K1 b - - 0 1", ("rook", "e1", "e8")),
}


def test_the_line_through_one_unit_is_present_where_it_should_and_absent_in_the_twin(model):
    for what, (fen, twin, (kind, place, to)) in HAND.items():
        assert has_through(model, fen, kind, place, to), what
        assert not has_through(model, twin, kind, place, to), what
        assert not through_ids(model, twin) or all(f":{kind}:{place}," not in n for n in through_ids(model, twin)), what


def test_two_units_of_theirs_borne_on_by_a_unit_that_leaps_have_no_line_through_it(model):
    assert through_ids(model, "6k1/2r1q3/8/3N4/8/8/8/6K1 w - - 0 1") == []


def test_the_through_nodes_equal_the_brute_walk_over_the_graph(model):
    static = model.static_part()
    profile = set(model.profile())
    seen = 0
    for moves in random_games(5, 3, 60):
        for t in range(6, len(moves), 5):
            g = GL.graph(model, CE.state("chess", None, moves[:t])[0], static)
            if g.mover is None:
                continue
            walk = UF.through_walk(g, profile)
            engine = set()
            for nid in g.of_type("through"):
                ls = [(lt, o) for lt, _v, _x, o, fwd in g.neighbours(nid) if fwd]
                engine.add(([o for lt, o in ls if lt == "by"][0], [o for lt, o in ls if lt == "at_place"][0]))
            assert walk == engine, (moves[:t], walk ^ engine)
            seen += len(engine)
    assert seen > 20


def test_agreement_with_the_rules_for_the_side_to_move(model):
    tot = [0] * 5
    n = 0
    for moves in random_games(8, 6, 70):
        for t in range(4, len(moves), 3):
            tot = [a + b for a, b in zip(tot, model.through_check(CE.state("chess", None, moves[:t])[0]))]
            n += 1
    print(f"\nOBS through vs the rules' shut lines, {n} random positions: through {tot[0]}, shut {tot[1]}, in both {tot[2]}, through only {tot[3]}, shut only {tot[4]}")
    assert tot[0] > 30 and tot[3] == 0 and tot[4] == 0


# ------------------------------------------------------------------------------------------------------------------------------ B: the tracker

SEQUENCES = {
    "a taking": ("r3k3/8/8/8/8/8/8/R3K3 w - - 0 1", ["a1a8", "e8d7", "a8a7"]),
    "a unit changing kind": ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", ["a7a8q", "e8d7"]),
    "a unit that goes and comes back": ("4k3/8/8/8/8/8/P7/N3K3 w - - 0 1", ["a1b3", "e8d8", "b3a1", "d8e8"]),
}


def fates_of(model, start, moves, horizons):
    kinds_, plies = model.unit_fates((start, moves), horizons)
    return kinds_, [(mover, [(o, k, p, list(codes)) for o, k, p, codes in units]) for mover, units in plies]


def mirror_plies(model, start, moves):
    static = model.static_part()
    out = []
    for t in range(len(moves) + 1):
        out.append(UF.units_of(GL.graph(model, CE.state("chess", start, moves[:t])[0], static)))
    return out


def test_the_tracker_on_hand_sequences(model, kinds):
    # a taking
    start, moves = SEQUENCES["a taking"]
    _k, plies = fates_of(model, start, moves, [1, 2])
    at = {(o, k, p): codes for o, k, p, codes in plies[0][1]}
    rook = kinds.index("rook")
    assert at[("white", "rook", "a1")] == [UF.MOVED, UF.MOVED]
    assert at[("black", "rook", "a8")] == [3 + rook, 3 + rook]                 # taken by a unit of the kind rook
    assert at[("white", "king", "e1")] == [UF.STAYED, UF.STAYED]
    assert at[("black", "king", "e8")] == [UF.STAYED, UF.MOVED]
    # a unit changing kind
    start, moves = SEQUENCES["a unit changing kind"]
    _k, plies = fates_of(model, start, moves, [1, 2])
    at = {(o, k, p): codes for o, k, p, codes in plies[0][1]}
    assert at[("white", "pawn", "a7")] == [UF.OTHER, UF.OTHER]                 # gone, a unit of another kind stands elsewhere: counted as other, never guessed
    # a unit that goes and comes back: stayed means standing where it started
    start, moves = SEQUENCES["a unit that goes and comes back"]
    _k, plies = fates_of(model, start, moves, [1, 2, 3, 4])
    at = {(o, k, p): codes for o, k, p, codes in plies[0][1]}
    assert at[("white", "knight", "a1")] == [UF.MOVED, UF.MOVED, UF.STAYED, UF.STAYED]
    # a horizon past the end of the game has no fate
    assert at[("white", "king", "e1")] == [UF.STAYED] * 4
    _k, plies = fates_of(model, start, moves, [5])
    assert {codes[0] for *_x, codes in plies[0][1]} == {UF.NO_FATE}


def test_the_rust_tracker_equals_its_readable_mirror(model, kinds):
    horizons = [1, 2, 4, 7]
    cases = [(s, m) for s, m in SEQUENCES.values()] + [(None, mv) for mv in random_games(31, 2, 36)]
    for start, moves in cases:
        _k, plies = fates_of(model, start, moves, horizons)
        mirror = mirror_plies(model, start, moves)
        assert len(plies) == len(mirror)
        for t in range(len(moves) + 1):
            assert [(o, k, p) for o, k, p, _c in plies[t][1]] == mirror[t]
            for i, (_o, _k2, _p, codes) in enumerate(plies[t][1]):
                assert codes == [UF.fate(mirror, kinds, t, i, h) for h in horizons], (start, t, i)


def test_the_tracker_counts_its_steps(model):
    games = [(None, mv) for mv in random_games(41, 4, 50)]
    plies, steps, alive, taken, other, ambiguous = model.track_counts(games, 2)
    assert steps == alive + taken + other and plies > steps // 40 and ambiguous == 0
    assert taken > 0


# ------------------------------------------------------------------------------------------------------------------------------ C: the fates of a pattern against the readable matcher

def test_the_fates_of_a_patterns_variables_equal_the_readable_matcher(model, kinds):
    games = random_games(51, 3, 34)
    wires = [(None, mv) for mv in games]
    rows = [[t for t in range(3, len(mv) - 8, 2)] for mv in games]
    horizons = [2, 4, 6]
    kept = {}
    allowed = None
    for size in (1, 2):
        _found, rec = model.pattern_mine(wires, rows, size, list(G2.LEVELS), allowed, GOAL, 2, 6)
        allowed = [r[0] for r in rec]
        kept[size] = rec
    rng = random.Random(7)
    chosen = rng.sample(kept[1], min(8, len(kept[1]))) + rng.sample(kept[2], min(10, len(kept[2])))
    keys = [r[0] for r in chosen]
    nk = len(kinds)
    ncls = 3 + nk
    zeros = [0.0] * (len(horizons) * nk * 2 * ncls)
    nc, flat = model.fate_cells(wires, rows, keys, zeros, horizons, 3, GOAL, 2)
    assert nc == ncls
    cells = {(int(r[0]), int(r[1]), int(r[2])): r for r in np.array(flat).reshape(-1, 7 + 2 * ncls)}
    static = model.static_part()
    mirror = [mirror_plies(model, None, mv) for mv in games]
    compared = skipped = 0
    for ci, rec in enumerate(chosen):
        pat = G2.decode(bytes(rec[4]))
        ref: dict = {}
        ambiguous = False
        for gi, mv in enumerate(games):
            for t in rows[gi]:
                g = GL.graph(model, CE.state("chess", None, mv[:t])[0], static)
                if g.mover is None:
                    continue
                items, units = GM.graph_items(g, GOAL)
                side0, side1 = GM.match_binds(items, units, pat, kinds)
                seen = set()
                for sets in side0 + side1:
                    if len(sets) > 1:
                        ambiguous = True
                    for tup in sets:
                        for v, uid in enumerate(tup):
                            seen.add((v, uid))
                for v, uid in seen:
                    owner, kind, place = uid.split(":", 3)[1:]
                    i = mirror[gi][t].index((owner, kind, place))
                    for hi, h in enumerate(horizons):
                        code = UF.fate(mirror[gi], kinds, t, i, h)
                        if code != UF.NO_FATE:
                            ref.setdefault((v, hi), np.zeros(1 + ncls))[0] += 1
                            ref[(v, hi)][1 + code] += 1
        if ambiguous:
            skipped += 1
            continue
        for (v, hi), r in ref.items():
            row = cells[(ci, v, hi)]
            assert row[3] == r[0] and list(row[7:7 + ncls]) == list(r[1:]), (G2.plain_words(pat, kinds), v, hi)
            compared += 1
        assert all((ci, v, hi) in cells for (v, hi) in ref)
    print(f"\nfates against the matcher: {compared} (variable, horizon) cells compared, {skipped} patterns with a symmetry skipped")
    assert compared >= 30


# ------------------------------------------------------------------------------------------------------------------------------ D: the rules and the arithmetic

def test_a_cell_departure_and_its_standard_error_are_from_the_games_residuals():
    nc = 4
    # one cell: 100 units in 10 games, 20 taken against 10 expected; per game residual sums +1.0 each -> r1 = 10, r2 = 10
    row = np.zeros(7 + 2 * nc)
    row[3:7] = [100, 10, 10.0, 10.0]
    row[7 + 2] = 0                                    # counts by class (unused here)
    t = UM.cell_table(row, nc)
    assert t["d"][0] == pytest.approx(0.1)
    assert t["se"][0] == pytest.approx(np.sqrt(10.0) / 100)
    assert t["z"][0] == pytest.approx(10.0 / np.sqrt(10.0))


def test_base_rates_are_shares_by_kind_and_state():
    nk, ncls = 2, 5
    counts = np.zeros(3 * len(UM.HORIZONS) * nk * 2 * (ncls + 1), dtype=np.int64)
    c = counts.reshape(3, len(UM.HORIZONS), nk, 2, ncls + 1)
    c[1, 0, 1, 1] = [6, 2, 0, 1, 1, 10]
    r = UM.rates_of(counts, 1, nk, ncls)
    assert list(r[0, 1, 1]) == pytest.approx([0.6, 0.2, 0.0, 0.1, 0.1])
    assert (r[0, 0] == 0).all() and (r[1] == 0).all()


def test_the_stated_rules_are_in_the_module_and_in_the_plan_file():
    plan = (ROOT / "docs/ANALYST_PLAN.md").read_text()
    assert "78.54 RULES" in plan
    for token in (f"MIN_SUPPORT = {UM.MIN_SUPPORT}", f"MIN_CELL_N = {UM.MIN_CELL_N}", f"MIN_B_N = {UM.MIN_B_N}"):
        assert token.split(" = ")[1] in plan.split("78.54 RULES")[1]
    assert UM.__doc__ and "BEFORE the 500-game run" in UM.__doc__


def test_the_modules_name_no_object_of_any_game():
    for rel in MODULES:
        text = (ROOT / rel).read_text().split("#[cfg(test)]")[0]            # the unit tests of a Rust file write small boards of their own
        bad = [m.group(0) for m in re.finditer(SCAN_WORDS, text)]
        assert not bad, (rel, bad[:5])


# ------------------------------------------------------------------------------------------------------------------------------ F: the check by eye

FIVE = {
    "a unit pinned to the goal unit": ("4k3/8/8/4n3/8/8/8/4R1K1 w - - 0 1", "3k4/8/8/4n3/8/8/8/4R1K1 w - - 0 1"),
    "a line opened by moving the unit between (the unit between is mine)": ("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1", "3k4/8/8/8/4N3/8/8/4R1K1 w - - 0 1"),
    "two units of theirs borne on by one of mine": ("6k1/2r1q3/8/3N4/8/8/8/6K1 w - - 0 1", "6k1/2r4q/8/3N4/8/8/8/6K1 w - - 0 1"),
    "a unit of theirs behind their goal unit on one line (the line is held by the side NOT to move)": ("4q3/8/8/4k3/8/8/8/4R1K1 b - - 0 1", "3q4/8/8/4k3/8/8/8/4R1K1 b - - 0 1"),
    "the pinned unit with the other side to move": ("4k3/8/8/4n3/8/8/8/4R1K1 b - - 0 1", "3k4/8/8/4n3/8/8/8/4R1K1 b - - 0 1"),
}


def test_the_five_hand_positions_against_the_concepts_of_the_run_when_they_are_there(kinds):
    from catspace.io import paths
    d = Path(paths.experiment("unit_fates_78_54", "games_s5016_n500"))
    book_f, tsv_f = d / "concept_keys.json", d / "concepts.tsv"
    games_f = Path(paths.experiment("opportunity_78_44", "games_s5016_n500", "games.jsonl"))
    if not (book_f.is_file() and tsv_f.is_file() and games_f.is_file()):
        pytest.skip("the concepts file of the run is not there")
    recs = sorted((json.loads(x) for x in games_f.read_text().splitlines() if x.strip()), key=lambda r: r["game"])
    m = GL.model("chess", "chess")
    m.learn_profile([(r.get("start"), list(r["moves"])) for r in recs[:UM.PROFILE_GAMES]])           # the profile the run learned
    book = json.loads(book_f.read_text())
    rows = {ln.split("\t")[0]: ln.split("\t") for ln in tsv_f.read_text().splitlines()[1:]}
    names = list(book)
    keys = [int(book[n]["key"]) for n in names]
    fens = [x for pair in FIVE.values() for x in pair]
    vals = G2.evaluate(m, fens, keys, GOAL, 3)
    out = []
    for i, what in enumerate(FIVE):
        a, b = vals[2 * i], vals[2 * i + 1]
        fires = np.flatnonzero((a != 0) & (b == 0))
        out.append(f"OBS hand position {what}: {int((a != b).sum())} kept concepts differ from the twin; hold here and not in the twin: {len(fires)}")
        for j in fires[:10]:
            r = rows[names[j]]
            out.append(f"OBS   {names[j]} (value {a[j]:+.0f}): {book[names[j]]['words']} | variable {r[6]} at h={r[7]}: on the odd games n {r[17]}, taken {r[18]}, base {r[19]}, z {r[20]}")
        # the line through one unit is a node of the graph here, whether or not a kept concept names it
        has_line = bool(through_ids(m, FIVE[what][0]))
        out.append(f"OBS   the line through one unit is a node here: {has_line}; in the twin: {bool(through_ids(m, FIVE[what][1]))}")
    print("\n" + "\n".join(out))
    (d / "hand_positions_obs.txt").write_text("\n".join(out) + "\n")
    assert out
