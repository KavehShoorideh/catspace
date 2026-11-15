"""Plan 78.52: the graph, layer 1 (rust/engine/tactics/src/sheet/base_graph.rs and base_statements.rs, read through catspace/incubator/analyst/discovery/graph_layer1.py).

Stage D: the counts of places, directions, lines and edge places against what the rules imply, on two variants; each place's edge distance per direction; every bearing the trait reports is
exactly one relation node and no other bearing node exists; the static part is the same bytes at every position of one game; the ids are the same across two builds.
Stage E: the 26 statements of 78.49 as counts over the graph equal subgoal.rs on random positions of both variants; the new statements.
The names of a line family or of a unit appear here, in the expected counts of one game, and nowhere in the module."""
import json
import random
import re
from itertools import product
from pathlib import Path

import cs_engine as CE
import pytest

from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import question_sheet as QS

ROOT = Path(__file__).resolve().parents[1]
WORTH = {"k": 0, "q": 14, "r": 8, "b": 5, "n": 4, "p": 2}
VARIANTS = ["chess", "racingkings"]
SCAN_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|ranks?|files?|diagonals?|[a-h][1-8])\b"


@pytest.fixture(scope="module")
def models():
    return {v: GL.model("chess", v) for v in VARIANTS}


@pytest.fixture(scope="module")
def statics(models):
    return {v: m.static_part() for v, m in models.items()}


def random_positions(variant: str, seed: int, games: int = 3, plies: int = 45) -> list:
    rng = random.Random(seed)
    out = []
    for _ in range(games):
        moves = []
        for _ in range(plies):
            state = CE.state(variant, None, moves)
            if state[3] is not None or not state[2]:
                break
            out.append(state[0])
            moves.append(rng.choice(sorted(state[2])))
    return out


def coords(place: str) -> tuple:
    return ord(place[0]) - 97, int(place[1]) - 1


# ------------------------------------------------------------------------------------------------------------------------------ D: the static part

@pytest.mark.parametrize("variant", VARIANTS)
def test_counts_of_places_directions_lines_and_edge_places(models, statics, variant):
    g = GL.Graph(statics[variant])
    nodes, links = g.counts()
    assert (nodes["place"], nodes["direction"], nodes["edge"]) == (64, 8, 8)
    assert nodes["line"] == 8 + 8 + 26                                        # ranks, files, and the diagonals of at least two places in both families
    assert links["reverse"] == 8                                              # every direction has its reverse
    assert links["on_edge"] == 4 * 8 + 4 * 15                                 # edge places: a side of the board (8) for four directions, two sides meeting (15) for the others
    assert links["next"] == 4 * 56 + 4 * 49                                   # steps: 8 x 7 along a side, 7 x 7 across
    lens = sorted(len(l) for l in statics[variant]["lines"])
    assert lens == sorted([8] * 16 + [n for n in range(2, 8) for _ in range(4)] + [8, 8])
    assert links["on_line"] == sum(lens) and links["line_along"] == nodes["line"] and links["edge_of"] == 8
    assert links["edge_distance"] == 64 * 8


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_places_edge_distance_per_direction(models, statics, variant):
    g = GL.Graph(statics[variant])
    expect = {"east": lambda f, r: 7 - f, "west": lambda f, r: f, "north": lambda f, r: 7 - r, "south": lambda f, r: r,
              "northeast": lambda f, r: min(7 - f, 7 - r), "northwest": lambda f, r: min(f, 7 - r), "southeast": lambda f, r: min(7 - f, r), "southwest": lambda f, r: min(f, r)}
    assert sorted(statics[variant]["directions"]) == sorted(expect)
    seen = 0
    for p in g.of_type("place"):
        f, r = coords(p.split(":")[1])
        by_dir = {}
        for t, via, value, other, fwd in g.neighbours(p):
            if t == "edge_distance" and fwd:
                by_dir[other.split(":")[1]] = value if value is not None else 0
            if t == "on_edge" and fwd:
                assert expect[other.split(":")[1]](f, r) == 0
        assert by_dir == {d: fn(f, r) for d, fn in expect.items()}, p
        seen += 1
    assert seen == 64


def test_the_second_variant_has_the_same_board_by_the_rules_and_the_same_static_bytes(statics):
    assert json.dumps(statics["chess"], sort_keys=True) == json.dumps(statics["racingkings"], sort_keys=True)


def test_ids_are_definitions_and_the_same_across_two_builds(models, statics):
    again = GL.model("chess", "chess").static_part()
    assert json.dumps(again) == json.dumps(statics["chess"])
    ids = [i for _t, i in statics["chess"]["nodes"]]
    assert len(set(ids)) == len(ids)
    assert {"place:e4", "direction:north", "edge:north", "kind:rook", "side:white", "line:north@place:a1"} <= set(ids) or {"place:e4", "direction:north", "kind:rook"} <= set(ids)
    fen = CE.state("chess", None, ["e2e4", "e7e5"])[0]
    one, two = models["chess"].position(fen), GL.model("chess", "chess").position(fen)
    assert json.dumps(one) == json.dumps(two)
    assert any(i.startswith("unit:white:pawn:e4") for _t, i in one["nodes"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_static_part_is_identical_across_the_positions_of_one_game(models, statics, variant):
    base = json.dumps(statics[variant], sort_keys=True)
    for fen in random_positions(variant, 5, games=1, plies=30):
        g = GL.graph(models[variant], fen, statics[variant])
        assert json.dumps({"nodes": g.nodes[:g.offset], "links": g.links[:len(statics[variant]["links"])], "directions": statics[variant]["directions"],
                           "lines": statics[variant]["lines"]}, sort_keys=True) == base
        assert g.offset == len(statics[variant]["nodes"])


# ------------------------------------------------------------------------------------------------------------------------------ D: the relation nodes

@pytest.mark.parametrize("variant", VARIANTS)
def test_every_bearing_the_trait_reports_is_exactly_one_relation_node_and_no_others(models, statics, variant):
    m = models[variant]
    checked = 0
    for fen in random_positions(variant, 11):
        raw = m.bearings(fen)
        expected = sorted(f"bear(unit:{o}:{k}:{at},place:{p})" for o, k, at, ps in raw for p in ps)
        g = GL.graph(m, fen, statics[variant])
        got = sorted(i for t, i in g.nodes if t == "bear")
        assert got == expected, fen
        assert len(set(got)) == len(got)
        units = {i for t, i in g.nodes if t == "unit"}
        for t, nid in g.nodes:
            if t != "bear":
                continue
            ls = g.neighbours(nid)
            by = [o for lt, _v, _x, o, fwd in ls if lt == "by" and fwd]
            at_place = [o for lt, _v, _x, o, fwd in ls if lt == "at_place" and fwd]
            assert len(by) == 1 and by[0] in units and len(at_place) == 1
            along = [o for lt, _v, _x, o, fwd in ls if lt == "along" and fwd]
            if ":knight:" in nid:
                assert along == [] and not [1 for lt, *_ in ls if lt == "between"]       # a leap follows no line
            else:
                assert len(along) == 1                                                  # a step or a slide follows one line
            assert not [1 for lt, *_ in ls if lt == "between" and False]
            checked += 1
    assert checked > 500


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_bearing_has_no_unit_between_and_a_shut_line_has_some(models, statics, variant):
    m = models[variant]
    shuts = 0
    for fen in random_positions(variant, 13):
        g = GL.graph(m, fen, statics[variant])
        mover = m.position(fen)["mover"]
        for t, nid in g.nodes:
            ls = g.neighbours(nid) if t in ("bear", "shut", "may_move") else []
            between = [o for lt, _v, _x, o, fwd in ls if lt == "between" and fwd]
            if t == "bear":
                assert between == []                                                    # the trait's bearing stops at the first unit
            if t == "shut":
                assert between and [1 for lt, *_ in ls if lt == "along"]
                unit = [o for lt, _v, _x, o, fwd in ls if lt == "by" and fwd][0]
                assert unit.split(":")[1] == mover                                      # asked of the side to move
                shuts += 1
    assert shuts > 0


def test_a_known_position_by_hand(models, statics):
    fen = "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"
    g = GL.graph(models["chess"], fen, statics["chess"])
    nodes, links = g.counts()
    assert nodes["unit"] == 4 and links["unit_on"] == links["unit_kind"] == links["unit_side"] == 4
    assert "bear(unit:white:rook:d2,place:d5)" in g.of_type("bear") or "bear(unit:white:rook:d2,place:d5)" in [i for _t, i in g.nodes]
    ls = g.neighbours("bear(unit:white:rook:d2,place:d5)")
    assert ("at_unit" in [x[0] for x in ls]) and ("unit:black:queen:d5" in [x[3] for x in ls])
    assert [x[3] for x in ls if x[0] == "along"] == ["line:north@place:d1"]
    assert "unit:" in g.text() and "bear(" in g.text()


# ------------------------------------------------------------------------------------------------------------------------------ E: the statements

def sheet_of(variant: str):
    env = QS.Env("chess")
    med = env.medium()
    hops = [(k, s, f, t, bool(take)) for (k, s, f), row in med.hop.items() for t, take in row.items()]
    overs = [(k, s, f, t, o) for (k, s, f, t), places in med.over_of.items() for o in places]
    wf = env.worth_fn(WORTH)
    return CE.SheetModel(variant, hops, overs, sorted(med.takes), {k: wf(k) for k in env.kind_keys_all()}, "king", len(env.relations["can_go"]), {k: int(v) for k, v in WORTH.items()}, dict(env.kind_keys_all())), \
        {k: wf(k) for k in env.kind_keys_all()}


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_26_statements_over_the_graph_equal_the_sheets_numbers(models, variant):
    sheet, worth = sheet_of(variant)
    m = models[variant]
    names, board_names = m.statement_names()
    old = sheet.subgoal_names()
    assert [n for n, _t in names[:26]] == [n for n, _t in old] and [t for _n, t in names[:26]] == [t for _n, t in old]
    n = len(old)
    compared = 0
    for fen in random_positions(variant, 17, games=4, plies=60):
        row = sheet.subgoal_games([(fen, [])], 1)[0][0]
        if row is None:
            continue
        us, them, _board = m.statements(fen, worth, "king")
        assert us[:n] == row[:n] and them[:n] == row[n:], fen
        compared += 1
    assert compared > 100


def test_every_statement_is_one_line_of_plain_text(models):
    names, board = models["chess"].statement_names()
    assert len(names) == len({n for n, _t in names}) and len(board) == 3
    for n, t in list(names) + list(board):
        assert n and t.strip() and "\n" not in t and not t.endswith(".")
    assert [n for n, _t in names[26:]][:2] == ["units_on_edge", "edge_distance_nearest"] and names[-1][0] == "open_pairs" and names[-2][0] == "goal_edge_nearest"


def line_members(statics) -> dict:
    return {i: ls for (t, i), ls in zip([n for n in statics["nodes"] if n[0] == "line"], statics["lines"])}


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_new_statements_match_a_count_made_by_hand_from_the_board(models, statics, variant):
    sheet, worth = sheet_of(variant)
    m = models[variant]
    names = [n for n, _t in m.statement_names()[0]]
    dirs = statics[variant]["directions"]
    dist = {"east": lambda f, r: 7 - f, "west": lambda f, r: f, "north": lambda f, r: 7 - r, "south": lambda f, r: r,
            "northeast": lambda f, r: min(7 - f, 7 - r), "northwest": lambda f, r: min(f, 7 - r), "southeast": lambda f, r: min(7 - f, r), "southwest": lambda f, r: min(f, r)}
    members = list(statics[variant]["lines"])
    for fen in random_positions(variant, 19, games=3, plies=50):
        if sheet.subgoal_games([(fen, [])], 1)[0][0] is None:
            continue
        pos = m.position(fen)
        mover = pos["mover"]
        units = [tuple(i.split(":")[1:]) for t, i in pos["nodes"] if t == "unit"]          # (side word, kind, place)
        us, them, board = m.statements(fen, worth, "king")
        for vals, side in ((us, mover), (them, [s for s in ("white", "black") if s != mover][0])):
            mine = [(k, coords(p)) for s, k, p in units if s == side]
            row = dict(zip(names, vals))
            assert row["units_on_edge"] == sum(1 for _k, (f, r) in mine if min(fn(f, r) for fn in dist.values()) == 0)
            assert row["edge_distance_nearest"] == sum(min(fn(f, r) for fn in dist.values()) for _k, (f, r) in mine)
            for d in dirs:
                assert row[f"edge_distance_{d}"] == sum(dist[d](f, r) for _k, (f, r) in mine)
            kings = [(f, r) for k, (f, r) in mine if k == "king"]
            assert row["goal_edge_nearest"] == (min(fn(*kings[0]) for fn in dist.values()) if len(kings) == 1 else 0)
        # pairs of own units bearing on each other along one line with nothing between, by hand from the trait's own bearings
        raw = {(o, k, at): set(ps) for o, k, at, ps in m.bearings(fen)}
        occupied = {at for (_o, _k, at) in raw}
        def aligned_open(a, b):
            (fa, ra), (fb, rb) = coords(a), coords(b)
            df, dr = fb - fa, rb - ra
            if not (df == 0 or dr == 0 or abs(df) == abs(dr)) or (df, dr) == (0, 0):
                return False
            step = (df > 0) - (df < 0), (dr > 0) - (dr < 0)
            x, y = fa + step[0], ra + step[1]
            while (x, y) != (fb, rb):
                if f"{chr(97 + x)}{y + 1}" in occupied:
                    return False
                x, y = x + step[0], y + step[1]
            return True
        for vals, side in ((us, mover), (them, [s for s in ("white", "black") if s != mover][0])):
            own = sorted(u for u in raw if u[0] == side)
            expect = sum(1 for i, a in enumerate(own) for b in own[i + 1:] if b[2] in raw[a] and a[2] in raw[b] and aligned_open(a[2], b[2]))
            assert dict(zip(names, vals))["open_pairs"] == expect, fen
        # the open length of a line: the longest run of places with no unit on them
        lengths = dict(m.open_lengths(fen))
        runs = []
        for places in members:
            best = run = 0
            for p in places:
                run = 0 if p in occupied else run + 1
                best = max(best, run)
            runs.append(best)
        assert sorted(lengths.values()) == sorted(runs)
        assert board == [float(sum(1 for places in members if not any(p in occupied for p in places))), float(max(runs)), float(sum(runs))]


# ------------------------------------------------------------------------------------------------------------------------------ the module names no game

def test_the_modules_name_no_object_of_any_game():
    paths = [ROOT / "rust/engine/tactics/src/sheet/base_graph.rs", ROOT / "rust/engine/tactics/src/sheet/base_statements.rs", ROOT / "catspace/incubator/analyst/discovery/graph_layer1.py",
             ROOT / "catspace/incubator/analyst/discovery/graph_layer1_measure.py"]
    for p in paths:
        text = p.read_text()
        if p.suffix == ".rs":                                                     # the unit tests of the Rust module use a board of their own, past the marker
            text = text.split("#[cfg(test)]")[0]
        text = re.sub(r'"""[\s\S]*?"""', '""', text)
        text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
        text = re.sub(r"'(?:\\.|[^'\\])*'", "''", text) if p.suffix == ".py" else text
        hit = re.search(SCAN_WORDS, text, flags=re.I)
        assert hit is None, (p.name, hit.group(0) if hit else None)
