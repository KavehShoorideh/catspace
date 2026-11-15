"""Plan 78.43 stages A, B0 and B: the question sheet's light reading and its levels 0 and 1 by geometry, then the obstacles, the frames and the verdict of a way (stage B), in the Rust engine (rust/engine/tactics/src/sheet, the game's part behind
`cs_game::placed::Placed` in kernel.rs, the data table of the medium of movement supplied from Python). The Python sheet (question_sheet.py) is the oracle: on a handful of fixed
positions the two forms must give the same units, level 0 verdicts, level 1 route sets and verdicts. The measure over whole games is sheet_rust_measure.py."""
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import question_sheet as Q
from catspace.incubator.analyst.discovery import sheet_rust as SR
from catspace.incubator.analyst.discovery import sheet_rust_deep_measure as D
from catspace.incubator.analyst.discovery import sheet_rust_measure as M

WORTH = {"k": 0, "q": 14, "r": 8, "b": 5, "n": 4, "p": 2}
HANGING = "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"
PAWNS = "4k3/p7/8/3q4/8/8/P2R4/4K3 w - - 0 1"
KING_ROUTE = "8/6Np/8/1P6/2p1P2K/8/8/4k3 w - - 0 1"
OWN_IN_WAY = "4k3/8/8/3b4/8/8/2P5/N3K3 w - - 0 1"
PAWN_TAKES = "7k/8/8/3b4/8/3p4/8/N3K3 w - - 0 1"
THEIR_IN_WAY = "7k/8/8/7n/1p6/p7/8/R3K3 w - - 0 1"
CONFINED_NEAR = "7k/6pp/8/8/8/8/8/R3K3 w - - 0 1"
IN_CHECK = "4k3/8/8/4q3/8/8/3R4/4K3 w - - 0 1"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MIDDLE = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
CASTLING = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"
PROMOTING = "4k3/P7/8/8/8/8/p7/4K3 w - - 0 1"
EN_PASSANT = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
FIXED = [START, MIDDLE, HANGING, PAWNS, KING_ROUTE, OWN_IN_WAY, PAWN_TAKES, THEIR_IN_WAY, CONFINED_NEAR, IN_CHECK, CASTLING, PROMOTING, EN_PASSANT]
ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "rust" / "engine" / "tactics" / "src" / "sheet"
NEW_SOURCES = [SHEET / n for n in ("mod.rs", "medium.rs", "reading.rs", "goal_report.rs", "levels.rs", "cf.rs", "frame.rs", "settle.rs", "walk.rs", "mv.rs")] + [ROOT / "rust" / "engine" / "game" / "src" / "placed.rs", ROOT / "rust" / "engine" / "py" / "src" / "sheet_py.rs"]
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"


@pytest.fixture(scope="module")
def env():
    return Q.Env("chess")


@pytest.fixture(scope="module")
def state(env):
    params = Q.Params(compare_old=False)
    return {"env": env, "worth": WORTH, "params": params, "model": SR.model(env, WORTH)}


def _both(state, fen):
    py = M.canon_python(M.python_side(fen, state))
    rs = M.canon_rust(SR.read(state["model"], state["env"], fen, state["params"]))
    return py, rs


# ------------------------------------------------------------------------------------------------------------------------------ the pieces

def test_the_units_are_read_with_their_worth_and_ordered_by_stake(state):
    _py, rs = _both(state, START)
    assert len(rs["units"]) == 32
    assert ("us", "queen", "d1", 14.0) in rs["units"] and ("them", "pawn", "a7", 2.0) in rs["units"] and ("us", "king", "e1", 0.0) in rs["units"]
    first = rs["order"][:2]
    assert [o[:3] for o in first] == [("us", "king", "e1"), ("them", "king", "e8")]                            # the goal unit of each side first, ours then theirs
    assert all(o[4] for o in first) and not any(o[4] for o in rs["order"][2:])
    assert [o[5] for o in rs["order"]] == list(range(1, 33))


def test_level0_reads_what_strikes_and_guards_a_target_and_its_open_exits(state):
    _py, rs = _both(state, HANGING)
    queen = rs["level0"][("them", "queen", "d5")]
    assert ("rook", "d2") in queen["strikers"] and queen["status"] == "live" and queen["guards"] == []          # struck by our rook, nothing guards it
    assert rs["level0"][("us", "rook", "d2")]["strikers"] == [("queen", "d5")]
    cornered = _both(state, CONFINED_NEAR)[1]["level0"][("them", "king", "h8")]
    assert cornered["struck"] is False and cornered["open_exits"] >= 0 and cornered["exits"] >= 1             # the goal unit is read by the report


def test_the_goal_unit_in_check_has_its_exits_from_the_zone_not_a_pass(state):
    py, rs = _both(state, IN_CHECK)
    assert py["level0"] == rs["level0"] and py["level1"] == rs["level1"]
    assert rs["level0"][("us", "king", "e1")]["struck"] is True


def test_level1_names_the_ways_of_the_asking_side_with_their_routes_and_obstacles(state):
    _py, rs = _both(state, OWN_IN_WAY)
    key = ("them", "bishop", "d5")
    assert rs["level1"][key]["ways"], "some unit of ours can come to strike the bishop"
    assert any(way[1] == "knight" for way, _routes in rs["routes"][key])
    routes = [r for _way, rr in rs["routes"][key] for r in rr]
    assert all(len(r) == 8 for r in routes) and any(r[6] for r in routes)                                      # obstacles are listed on the routes that have them


def test_a_target_nothing_can_reach_is_closed_with_its_failed_link(state):
    _py, rs = _both(state, THEIR_IN_WAY)
    closed = [k for k, v in rs["level1"].items() if v["status"] == "closed"]
    assert closed and all(rs["level1"][k]["failed_at"] in (1, 2, 3) for k in closed)


def test_a_made_up_worth_table_is_seen_as_a_difference(state):
    """The comparison can fire: the same position under a different worth table differs in the units' worth."""
    other = dict(state, model=SR.model(state["env"], {**WORTH, "q": 99}))
    py = M.canon_python(M.python_side(HANGING, state))
    rs = M.canon_rust(SR.read(other["model"], state["env"], HANGING, state["params"]))
    assert M.diff_position(py, rs), "a wrong worth must differ"


def test_a_position_that_is_over_is_refused(state):
    with pytest.raises(ValueError):
        SR.read(state["model"], state["env"], "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", state["params"])           # stalemate: no legal move


# ------------------------------------------------------------------------------------------------------------------------------ rust against python

@pytest.mark.parametrize("fen", FIXED)
def test_the_rust_form_and_the_python_form_agree(state, fen):
    py, rs = _both(state, fen)
    assert M.diff_position(py, rs) == []
    assert py["routes"] == rs["routes"]


# ------------------------------------------------------------------------------------------------------------------------------ the layer stays game-free

def test_the_new_layer_sources_name_no_game_object_in_code_or_comments():
    for p in NEW_SOURCES:
        text = re.sub(r'"(?:\\.|[^"\\])*"', '""', p.read_text())
        hit = re.search(CHESS_WORDS, text, flags=re.I)
        assert hit is None, (p.name, hit.group(0) if hit else None)


def test_python_wrapper_names_no_game_object():
    for name in ("sheet_rust.py",):
        text = (ROOT / "catspace" / "incubator" / "analyst" / "discovery" / name).read_text()
        assert not re.search(r"\b(king|queen|rook|bishop|knight|pawn)\b", text, flags=re.I)


# ------------------------------------------------------------------------------------------------------------------------------ stage B: the obstacles, the frames, the verdict of a way

TWO_ROUTES = "r3k3/8/8/3p4/8/2N5/8/R3K3 w - - 0 1"
JOINT = "4k3/8/7p/7P/7R/8/8/4K3 b - - 0 1"            # a rook whose line to the far end has three units in it
MIDDLE_2 = "r1bq1rk1/ppp2ppp/2n2n2/3pp3/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 7"
BARE = "8/8/8/8/8/3n4/6K1/3kb3 w - - 2 57"            # the board without the stopper is one the game calls over (a lone knight): the route is still followed, as the Python codec asks the board
DEEP = [BARE, START, MIDDLE, HANGING, PAWNS, KING_ROUTE, OWN_IN_WAY, PAWN_TAKES, THEIR_IN_WAY, CONFINED_NEAR, IN_CHECK, CASTLING, PROMOTING, EN_PASSANT, JOINT, MIDDLE_2, TWO_ROUTES]


@pytest.fixture(scope="module")
def deep_state(state):
    return state


@pytest.mark.parametrize("fen", DEEP)
def test_the_deeper_levels_agree_with_the_python_sheet_step_by_step(deep_state, fen):
    """The Python sheet reads the position (schedule, levels 0 and 1, the flagged units to level 2); its steps of level 2 and on are replayed in the Rust engine in the same order: every
    finding (the verdict kind of every way, the obstacles, the stops, the cornering, the search, the census) is the same."""
    try:
        out = D.compare_position(deep_state, fen)
    except ValueError:
        pytest.skip("the position is over")
    assert out["stage_a"] == [], out["stage_a"][:2]
    assert out["deep"] == [], out["deep"][:3]


def test_some_deeper_steps_are_really_compared(deep_state):
    """The comparison has something to compare: frames with settled ways, and the goal units' search and census."""
    modes, ways = set(), 0
    for fen in (START, MIDDLE, KING_ROUTE, OWN_IN_WAY, THEIR_IN_WAY, MIDDLE_2):
        out = D.compare_position(deep_state, fen)
        for q, f in out["steps"]:
            modes.add(f.detail.get("mode", "").split(",")[0])
            ways += len([x for x in f.detail.get("results", []) if x.get("way") is not None])
    assert {"frame", "search", "census"} <= modes and ways > 0


def test_a_finding_of_another_question_is_seen_as_a_difference(deep_state):
    """The deeper comparison can fire: the Rust finding of one question set against the Python finding of another differs."""
    out = D.compare_position(deep_state, MIDDLE)
    frames = [(q, f) for q, f in out["steps"] if f.detail.get("mode") == "frame" and f.detail["results"][1:]]
    assert len(frames) >= 2
    (q1, f1), (q2, f2) = frames[0], frames[-1]
    sheet_steps = [(q.target.side, q.target.kind, Q.place_text(q.target.at), f.level) for q, f in (frames[0], frames[-1])]
    rs = SR.settle(deep_state["model"], deep_state["env"], MIDDLE, sheet_steps, deep_state["params"])
    mix = D.diff_finding(D.canon_finding_python(q1, f1), D.canon_finding_rust(rs["steps"][1]))
    assert mix, "another question's finding must differ"


def test_the_game_names_the_counterfactual_board_not_the_layer():
    """The counterfactual boards (a unit taken off, a unit put elsewhere) are the game's to make: the trait names them, the chess binding implements them beside the kernel."""
    placed = (ROOT / "rust" / "engine" / "game" / "src" / "placed.rs").read_text()
    assert "fn without" in placed and "fn relocate" in placed and "fn find_step" in placed and "fn why_not" in placed
    outside = (ROOT / "rust" / "engine" / "game" / "src" / "outside.rs").read_text()
    assert "fn without" in outside and "fn relocate" in outside


# ------------------------------------------------------------------------------------------------------------------------------ stage C, step one: the question of the move and the loss of a unit

from catspace.incubator.analyst.discovery import sheet_rust_move_measure as MM

MOVES = [START, MIDDLE, HANGING, PAWNS, KING_ROUTE, OWN_IN_WAY, PAWN_TAKES, THEIR_IN_WAY, CONFINED_NEAR, IN_CHECK, CASTLING, PROMOTING, EN_PASSANT, JOINT, MIDDLE_2, TWO_ROUTES, BARE]


def _move_both(state, fen):
    env, params = state["env"], state["params"]
    r = Q.Reading(env, fen, WORTH, light=True)
    py = Q.move_sheet(env, r, params)
    rs = SR.move_sheet(state["model"], env, fen, params)
    return py, rs


@pytest.mark.parametrize("fen", MOVES)
def test_the_question_of_every_move_agrees_with_the_python_sheet(state, fen):
    """Every legal move of the position: the order they are asked in, the verdict kind, the three answers, the figures and the units named are the Python sheet's."""
    try:
        py, rs = _move_both(state, fen)
    except ValueError:
        pytest.skip("the position is over")
    assert list(py.order) == rs["order"]
    a = {x["move"]: MM.canon_python(x) for x in py.results}
    b = {x["move"]: MM.canon_rust(x) for x in rs["results"]}
    assert a.keys() == b.keys()
    for mv in a:
        assert MM.diff_moves(a[mv], b[mv]) == [], mv


def test_the_move_question_names_a_loss_a_taking_and_a_safe_move(state):
    """A hanging piece moved onto a struck place LOSES, a move that takes a guarded unit is read as a balance, a quiet move is SAFE."""
    _py, rs = _move_both(state, "4k3/8/8/3p4/4P3/8/8/3RK3 w - - 0 1")
    by = {x["move"]: x for x in rs["results"]}
    assert by["e4d5"]["took"] == 2.0 and by["e4d5"]["took_units"] == [("them", "pawn", "d5")]
    assert by["d1d5"]["took"] == 2.0 and by["d1d5"]["verdict"] in ("SAFE", "LOSES")
    assert {x["verdict"] for x in rs["results"]} >= {"SAFE"}
    losing = [x for x in rs["results"] if x["verdict"] == "LOSES"]
    for x in losing:
        assert x["named"] is not None and x["high"] >= x["low"] > 0 and x["because"]


def test_a_move_that_is_not_ours_is_refused(state):
    with pytest.raises(ValueError):
        state["model"].move_question(START, "e7e5", 2, 1)


def test_a_wrong_move_answer_is_seen_as_a_difference(state):
    """The comparison can fire: the answer of one move set against the Python answer of another differs."""
    py, rs = _move_both(state, MIDDLE)
    a = {x["move"]: MM.canon_python(x) for x in py.results}
    b = {x["move"]: MM.canon_rust(x) for x in rs["results"]}
    keys = sorted(a)
    assert any(MM.diff_moves(a[keys[0]], b[k]) for k in keys[1:])


def test_the_general_format_of_a_number_is_the_python_one():
    """The words carry the numbers with Python's `:g`: the measure compares the words, so the format is the same (tested in the layer too)."""
    r = Q.Reading(Q.Env("chess"), "4k3/8/8/3p4/4P3/8/8/3RK3 w - - 0 1", WORTH, light=True)
    assert f"{2.0:g}" == "2" and f"{1e9:g}" == "1e+09" and f"{2.0:+g}" == "+2" and r.legal


def test_the_loss_of_a_unit_agrees_with_the_python_form_on_a_game_with_exchanges(state):
    env, model = state["env"], state["model"]
    worth_of = env.worth_fn(WORTH)
    moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "g1f3", "c8g4", "h2h3", "g4f3", "d1f3"]
    events = env.codec.events(None, moves)
    wire = SR.event_wire(events)
    by_ply, _taken = env.codec.tracks(None, moves)
    found = lost = 0
    for ply in range(len(moves) + 1):
        for _place, (uid, _s, _k) in by_ply[ply].items():
            for h in (4, 8):
                py, rs = Q.unit_loss(events, uid, ply, h, worth_of), SR.unit_loss(model, wire, uid, ply, h)
                assert (py is None) == (rs is None)
                if py is None:
                    continue
                found += 1
                assert (py["status"], float(py["balance"]), py["place"], py["ply"], list(py["run"]), py["arrival"]) == (rs["status"], rs["balance"], rs["place"], rs["ply"], rs["run"], rs["arrival"])
                if py["status"] == Q.LOST:
                    lost += 1
                    assert Q.charged_ply(events, py) == rs["charged"]
                else:
                    assert rs["charged"] is None
    assert found > 0 and lost > 0
