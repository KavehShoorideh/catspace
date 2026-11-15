"""Plan 78.53: the graph, layer 2 -- patterns over the layer-1 graph (rust/engine/tactics/src/sheet/patterns.rs, catspace/incubator/analyst/discovery/graph_layer2*.py).

The language and its canonical form; the engine's counts against a brute-force matcher that reads the layer-1 graph as the Python mirror gives it (sizes 1 to 3, every level); the same pattern met two ways is
one key; the reduction test; the concepts file and the mapping file; the modules name no object of any game; and the check by eye (stage E): hand-made positions that hold a pinned unit, a line opened by moving
the unit between, two units of theirs borne on by one of ours, a unit of theirs behind their goal unit on one line, each beside its twin with the feature removed -- the test only asserts that some concept of size 2
or 3 tells each position from its twin, and prints which dcN fire. No rule is written that makes a named tactic fire; the concepts are mined from random play that never saw these positions.
The names of units and of lines appear here, in the hand positions, and nowhere in the modules."""
import random
import re
from pathlib import Path

import cs_engine as CE
import numpy as np
import pytest

from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2 as G2
from catspace.incubator.analyst.discovery import graph_layer2_match as GM

ROOT = Path(__file__).resolve().parents[1]
GOAL = "king"
SCAN_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|diagonals?|[a-h][1-8]|pins?|pinned|forks?|skewers?|batter\w+|x-?rays?|discovered)\b"
MODULES = ["rust/engine/tactics/src/sheet/patterns.rs", "catspace/incubator/analyst/discovery/graph_layer2.py", "catspace/incubator/analyst/discovery/graph_layer2_match.py",
           "catspace/incubator/analyst/discovery/graph_layer2_measure.py"]


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
    return GL.model("chess", "chess")


@pytest.fixture(scope="module")
def kinds(model):
    return [n[len("kind:"):] for t, n in model.static_part()["nodes"] if t == "kind"]


@pytest.fixture(scope="module")
def mined(model):
    games = random_games(11, 10, 40)
    wires = [(None, m) for m in games]
    rows = [list(range(len(m))) for m in games]
    return G2.enumerate_patterns(model, wires, rows, GOAL, 4, fuse=40, log=lambda m: None), wires, rows


def pattern_of(rec: dict) -> G2.Pattern:
    return G2.decode(rec["bytes"])


# ------------------------------------------------------------------------------------------------------------------------------ the language and its form

def test_every_kept_pattern_decodes_to_one_definition_and_one_hash(mined, kinds):
    kept = mined[0]["kept"]
    assert set(kept) == {1, 2, 3}
    texts: dict = {}
    for size, recs in kept.items():
        assert recs
        for h, r in recs.items():
            p = pattern_of(r)
            assert p.size == size and p.level == r["level"]
            text = G2.canonical_text(p, kinds)
            assert texts.setdefault(text, h) == h                       # a definition met twice would be two keys: the canonical form has failed
            assert len(G2.hash_of(text)) == 64 and G2.plain_words(p, kinds)
            assert all(ty in G2.TYPES for ty, *_ in p.atoms)
    assert len({G2.hash_of(t) for t in texts}) == len(texts)


def test_the_levels_say_what_they_say(mined, kinds):
    for recs in mined[0]["kept"].values():
        for r in recs.values():
            p = pattern_of(r)
            assert len(p.sides) >= 1
            if p.level == 0:
                assert not p.goal and not p.edge and not p.kind
            if p.level == 3:
                assert len(p.kind) == p.n_units() and all(k < len(kinds) for k in p.kind)
            if p.level in (0, 1, 2):
                assert not p.kind                                        # kind-free
    levels = {r["level"] for recs in mined[0]["kept"].values() for r in recs.values()}
    assert levels == {0, 1, 2, 3}


def test_a_kind_bound_pattern_names_its_kind_free_parent(mined):
    kept = mined[0]["kept"]
    free = {h for size in kept for h, r in kept[size].items() if r["level"] == 1}
    some = [r for size in kept for r in kept[size].values() if r["level"] == 3]
    assert some and all(r["free_parent"] != 0 for r in some)
    assert any(r["free_parent"] in free for r in some)


def test_growth_by_size_each_pattern_has_a_kept_part(mined):
    kept = mined[0]["kept"]
    for size in (2, 3):
        below = set(kept[size - 1])
        for r in kept[size].values():
            assert any(p in below for p in r["parts"])


# ------------------------------------------------------------------------------------------------------------------------------ the engine against the readable matcher

def test_the_engines_counts_equal_the_brute_force_matcher(model, mined, kinds):
    kept = mined[0]["kept"]
    rng = random.Random(3)
    chosen = []
    for size in (1, 2, 3):
        for level in G2.LEVELS:
            recs = sorted((h, r) for h, r in kept[size].items() if r["level"] == level)
            chosen += [(h, r) for h, r in rng.sample(recs, min(4 if size < 3 else 2, len(recs)))]
    keys = [h for h, _ in chosen]
    positions = []
    for moves in random_games(5, 3, 60)[:3]:
        for t in (14, 30, 44):
            if t < len(moves):
                positions.append(CE.state("chess", None, moves[:t])[0])
    raw = model.pattern_count([(f, []) for f in positions], [[0]] * len(positions), keys, 3, list(G2.LEVELS), GOAL, 2)
    us, them = G2.counts_matrix(raw, len(positions), len(keys))
    us, them = us.toarray(), them.toarray()
    static = model.static_part()
    compared = 0
    for i, fen in enumerate(positions):
        items, units = GM.graph_items(GL.graph(model, fen, static), GOAL)
        for j, (h, r) in enumerate(chosen):
            if r["support"] == 0:
                continue
            want = GM.match_counts(items, units, pattern_of(r), kinds)
            assert (int(us[i, j]), int(them[i, j])) == want, (fen, G2.plain_words(pattern_of(r), kinds))
            compared += 1
    assert compared >= 100
    assert us.sum() + them.sum() > 0


def test_a_position_and_its_colour_mirror_have_equal_values(model, mined):
    """Sides are relative: the same board with the colours swapped and turned over, the other colour to move, has the same value for every concept (mine and theirs are read from the side to move)."""
    kept = mined[0]["kept"]
    keys = [h for size in (1, 2, 3) for h in kept[size]]
    f = "4k3/8/8/4n3/8/8/8/4R1K1 w - - 0 1"
    g = "4r1k1/8/8/8/4N3/8/8/4K3 b - - 0 1"
    a, b = G2.evaluate(model, [f, g], keys, GOAL, 3)
    assert (a != 0).any()
    assert (a == b).all()


# ------------------------------------------------------------------------------------------------------------------------------ the reduction test and the files

def test_the_reduction_test_is_stated_and_works():
    n = 400
    rng = np.random.default_rng(0)
    parts = rng.integers(0, 4, size=(n, 2)).astype(float)
    half = np.arange(n) % 2 == 0
    assert G2.determined(parts, parts[:, 0] * parts[:, 1], half)                   # a function of the parts' counts
    assert not G2.determined(parts, rng.integers(0, 5, size=n).astype(float), half)    # not a function of them
    assert not G2.determined(np.zeros((n, 0)), np.zeros(n), half)                  # no part to reduce to
    lone = np.arange(n, dtype=float).reshape(-1, 1)
    assert not G2.determined(lone, lone[:, 0], half)                               # a vector never seen on the first half proves nothing


def test_the_concepts_file_has_an_empty_name_column_and_the_mapping_is_only_read(tmp_path, mined, kinds):
    rows = []
    for n, (h, r) in enumerate(list(mined[0]["kept"][2].items())[:5], 1):
        p = pattern_of(r)
        rows.append((f"dc{n}", G2.hash_of(G2.canonical_text(p, kinds)), 2, r["level"], G2.plain_words(p, kinds)))
    path = tmp_path / "concepts.tsv"
    G2.write_concepts(str(path), rows)
    lines = path.read_text().splitlines()
    assert lines[0].split("\t") == list(G2.HEADER)
    assert all(len(line.split("\t")) == 6 and line.split("\t")[5] == "" for line in lines[1:])
    assert G2.read_mapping(str(tmp_path / "mapping.tsv")) == {}
    (tmp_path / "mapping.tsv").write_text("# ours\n" + rows[0][1] + "\tmy name\n")
    assert G2.read_mapping(str(tmp_path / "mapping.tsv")) == {rows[0][1]: "my name"}
    assert path.read_text() == "\n".join(lines) + "\n"                              # reading the mapping wrote nothing


def test_the_modules_name_no_object_of_any_game():
    for rel in MODULES:
        text = (ROOT / rel).read_text()
        bad = [m.group(0) for m in re.finditer(SCAN_WORDS, text)]
        assert not bad, (rel, bad[:5])


def test_the_side_to_move_only_shut_line_gap_is_measured(model):
    """The first-blocker form through `without` agrees with the rules' own answer for the side to move only where the game accepts the hypothetical board; the gap is what keeps the other side's shut lines out."""
    tot = [0, 0]
    for moves in random_games(5, 3, 60):
        for t in range(10, len(moves), 7):
            a, b, _same = model.first_blocker_check(CE.state("chess", None, moves[:t])[0])
            tot[0] += a
            tot[1] += b
    print(f"first-blocker check: rules' answer {tot[0]} shut relations, through the hypothetical board {tot[1]}")
    assert tot[0] >= tot[1]


def test_unit_tables_count_the_units_by_kind_at_the_row_and_later(model, kinds):
    moves = random_games(9, 1, 40)[0]
    names, raw = model.unit_tables([(None, moves)], [[0, 10, len(moves) - 2, len(moves) + 5]], 4, 1)
    assert names == kinds
    tab = np.frombuffer(raw, dtype=np.int16).reshape(4, 1 + 4 * len(names))
    assert tab[0, 0] == 1 and tab[1, 0] == 1 and tab[2, 0] == 0 and tab[3, 0] == 0            # the horizon runs past the game; a row past the end
    assert tab[0, 1:1 + len(names)].sum() == tab[0, 1 + len(names):1 + 2 * len(names)].sum() == 16


# ------------------------------------------------------------------------------------------------------------------------------ E: the check by eye

HAND = {
    "a unit pinned to the goal unit": ("4k3/8/8/4n3/8/8/8/4R1K1 w - - 0 1", "3k4/8/8/4n3/8/8/8/4R1K1 w - - 0 1"),
    "a line opened by moving the unit between (the unit between is mine, on the line)": ("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1", "3k4/8/8/8/4N3/8/8/4R1K1 w - - 0 1"),
    "two units of theirs borne on by one of mine": ("6k1/2r1q3/8/3N4/8/8/8/6K1 w - - 0 1", "6k1/2r4q/8/3N4/8/8/8/6K1 w - - 0 1"),
    "a unit of theirs behind their goal unit on one line": ("4q3/8/8/4k3/8/8/8/4R1K1 b - - 0 1", "3q4/8/8/4k3/8/8/8/4R1K1 b - - 0 1"),
}


def test_hand_positions_differ_from_their_twins_by_some_concept_of_size_2_or_3(model, mined, kinds):
    kept = mined[0]["kept"]
    chosen = [(h, r) for size in (2, 3) for h, r in kept[size].items()]
    keys = [h for h, _ in chosen]
    fens = [f for pair in HAND.values() for f in pair]
    vals = G2.evaluate(model, fens, keys, GOAL, 3)
    for i, (what, (_feature, _twin)) in enumerate(HAND.items()):
        a, b = vals[2 * i], vals[2 * i + 1]
        differ = np.flatnonzero(a != b)
        fires = np.flatnonzero((a != 0) & (b == 0))
        print(f"\nstage E: {what}: {len(differ)} concepts of size 2 or 3 differ from the twin, {len(fires)} hold here and not in the twin")
        for j in fires[:12]:
            h, r = chosen[j]
            print(f"   dc{j + 1} fires (size {pattern_of(r).size}, level {r['level']}, value {a[j]:+.0f}): {G2.plain_words(pattern_of(r), kinds)}")
        assert len(differ) > 0, what
