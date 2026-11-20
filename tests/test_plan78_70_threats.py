"""Plan 78.70: the threat list of a position (catspace/incubator/analyst/discovery/threat_list.py, the checks in threat_measure.py, the print in threat_print.py). SYSTEM_SPEC 8 and 4.
Synthetic rows for the list's logic; one engine read for the Rust extension (`questions_threats`, `QReader::answers`); a scan of the files that must name no game object."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import threat_list as TL
from catspace.incubator.analyst.discovery import threat_measure as TM

ROOT = Path(__file__).resolve().parents[1]
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
ENGINE = ROOT / "rust" / "engine"


def _between(path: Path, start: str, end: "str | None" = None) -> str:
    t = path.read_text()
    a = t.index(start)
    return t[a:t.index(end, a)] if end else t[a:]


def test_the_list_and_the_measure_name_no_game_object_and_no_relative_data_path():
    for mod in (TL, TM):
        src = Path(mod.__file__).read_text()
        hit = re.search(CHESS_WORDS, src, flags=re.I)
        assert hit is None, (mod.__name__, hit.group(0))
        assert not re.search(r"[\"']data/", src)
        assert "sys.path" not in src
        assert "experiments." not in src


def test_the_rust_added_for_the_list_names_no_game_object():
    """The code and the comments added in 78.70 (marked by their own headers) in the sheet crate and its bindings."""
    parts = [(ENGINE / "tactics" / "src" / "sheet" / "questions.rs", "    /// PLAN 78.70 A. WHAT THE OWNER OF A MEMBER", "    pub fn root_state_after_pass"),
             (ENGINE / "tactics" / "src" / "sheet" / "questions.rs", "/// 78.70: one answer to a taking", "/// What one reading found."),
             (ENGINE / "py" / "src" / "questions_py.rs", "// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ plan 78.70 stage A", None),
             (ENGINE / "tactics" / "src" / "sheet" / "threat_frames.rs", "/// The class and basis of member `id`", None),   # (plan 79.08 moved it out of the binding into the sheet crate; the whole of that file from there on is scanned)
             (ENGINE / "py" / "src" / "base_graph_py.rs", "    /// questions_threats(", "    /// frames_measure(")]
    for p, a, b in parts:
        text = _between(p, a, b)
        hit = re.search(CHESS_WORDS, text, flags=re.I)
        assert hit is None, (p.name, hit.group(0))


def test_the_interface_is_written_before_the_body():
    for mod in (TL, TM):
        for word in ("INTERFACE", "takes", "returns", "Called by"):
            assert word in mod.__doc__, (mod.__name__, word)
    assert "THE STAKE" in TL.__doc__ and "THE WIDENING" in TL.__doc__ and "HOW A LINE ENDED" in TL.__doc__


NAMES = {"kinds": ["k0", "k1", "k2"], "roles": ["r0", "r1"], "places": [f"p{i}" for i in range(10)]}


def _frame(fid, members, places, claims):
    return {"id": fid, "members": members, "places": places, "unread": 0, "claims": claims}


def _claim(member, cls, answers=(), bearers=((1, 5),), basis=2, tried=3):
    return {"member": member, "class": cls, "basis": basis, "take_basis": 255, "bearers": list(bearers), "answers": list(answers), "tried": tried, "unread": 0}


def _raw(frames, mover=0):
    return {"game": 0, "g": 0, "ply": 20, "mover": mover, "board": [], "played": None, "questions": 0, "frames": frames}


def test_a_unit_lost_in_the_smallest_frame_and_answered_in_a_wider_one_is_one_row_with_a_response():
    small = _frame("f/small", [(1, 0, 2), (0, 1, 5)], [2, 5], [_claim(0, 0)])                              # lost, no answers asked for a lost class
    wide = _frame("f", [(1, 0, 2), (0, 1, 5), (1, 2, 7)], [2, 5, 7], [_claim(0, 1, answers=[(7, 5, 2, 0)])])   # safe in the wider frame, with an answer
    rows = TL.threat_rows(_raw([small, wide]), NAMES)
    assert len(rows["mover"]) == 1 and rows["other"] == []
    r = rows["mover"][0]
    assert r["class"] == "lost" and r["widened"] and r["response"] and r["response"][0][1] == TL.HOW[2]
    assert [f["n_members"] for f in r["frames"]] == [2, 3]                                                     # both frames named, the smallest first
    assert r["win_frame"]["n_members"] == 2


def test_a_lost_unit_with_no_wider_frame_has_no_response_and_lost_stays_apart_from_contested():
    lost = _frame("a", [(1, 0, 2), (0, 1, 5)], [2, 5], [_claim(0, 0)])
    contested = _frame("b", [(1, 1, 3), (0, 1, 6)], [3, 6], [_claim(0, 6, answers=[(3, 4, 1, 1)])])
    rows = TL.threat_rows(_raw([lost, contested]), NAMES)["mover"]
    by = {r["place"]: r for r in rows}
    assert by["p2"]["class"] == "lost" and by["p2"]["response"] == [] and not by["p2"]["widened"]
    assert by["p3"]["class"] == "contested" and by["p3"]["response"] and not by["p3"]["widened"]


def test_the_two_sides_are_listed_apart_by_the_owner_of_the_unit():
    f = _frame("a", [(1, 0, 2), (0, 1, 5)], [2, 5], [_claim(0, 4, answers=[(2, 3, 1, 1)]), _claim(1, 0, bearers=((0, 2),))])
    rows = TL.threat_rows(_raw([f], mover=1), NAMES)
    assert [r["place"] for r in rows["mover"]] == ["p2"] and [r["place"] for r in rows["other"]] == ["p5"]
    assert rows["mover"][0]["owner"] == 1 and rows["other"][0]["owner"] == 0                                   # the absolute side follows the mover


def test_the_stake_orders_the_rows_and_is_the_callers_not_a_table_here():
    f = _frame("a", [(1, 0, 2), (1, 1, 3), (0, 1, 5)], [2, 3, 5], [_claim(0, 0), _claim(1, 0)])
    rows = TL.threat_rows(_raw([f]), NAMES, stake_of=lambda kind, own: {"k0": 0.1, "k1": 0.4}[kind])["mover"]
    assert [r["kind"] for r in rows] == ["k1", "k0"] and rows[0]["stake"] == 0.4
    none = TL.threat_rows(_raw([f]), NAMES)["mover"]
    assert all(r["stake"] is None for r in none)


def test_the_four_bands_and_their_order_by_both_ways():
    hi_no = {"stake": 0.5, "response": []}
    hi_yes = {"stake": 0.5, "response": [1]}
    lo_no = {"stake": 0.01, "response": []}
    lo_yes = {"stake": 0.01, "response": [1]}
    assert [TL.band_of(r, 0.1, None, "spread") for r in (hi_no, hi_yes, lo_no, lo_yes)] == [1, 2, 3, 4]
    assert [TL.band_of(r, None, h, "rank") for r, h in ((hi_no, True), (hi_yes, True), (lo_no, False), (lo_yes, False))] == [1, 2, 3, 4]
    assert TL.band_of({"stake": None, "response": []}, 0.1, None, "spread") is None
    assert TL.band_of(hi_no, None, None, "spread") is None


def test_the_rank_within_the_position_takes_the_top_half_and_never_a_negative_stake():
    rows = [{"stake": s} for s in (0.4, 0.3, 0.2, 0.1, -0.05, None)]
    hi = TL.rank_high_of(rows)
    assert [hi.get(id(r)) for r in rows] == [True, True, True, False, False, None]                              # five rows have a stake: the top three, the middle one included


def test_how_a_line_ended_is_one_of_the_four_reasons():
    c = {"frame_unread": 0, "unread": 0, "basis": 3, "take_basis": 255}
    assert TL.ended_of(c) == "quiet"
    assert TL.ended_of(dict(c, basis=0)) == "concrete" and TL.ended_of(dict(c, basis=1)) == "concrete"
    assert TL.ended_of(dict(c, unread=2)) == "out of budget" and TL.ended_of(dict(c, frame_unread=1)) == "out of budget"
    assert set(TL.ENDED) == {"concrete", "quiet", "lost beyond doubt", "out of budget"}


def test_the_read_set_is_the_members_and_places_and_the_side_to_move():
    f = _frame("a", [(1, 0, 2), (0, 1, 5)], [5, 2], [])
    assert TL.read_set_of(f, 0) == (0, ((0, 0, 2), (1, 1, 5)), (2, 5))
    assert TL.read_set_of(f, 1) == (1, ((0, 1, 5), (1, 0, 2)), (2, 5))                                        # the sides are absolute: the same unit is the other side's when the mover differs


@pytest.fixture(scope="module")
def reader():
    return TL.ThreatReader(threads=2)


def test_the_engine_returns_the_read_set_the_answers_and_the_played_move(reader):
    moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "g1f3", "g8f6"]
    res = reader.read([(None, moves)], [[5, 6, 8]])
    assert [r["ply"] for r in res["rows"]] == [5, 6, 8]
    places = res["places"]
    seen_answer = seen_member_claim = False
    for r in res["rows"]:
        assert r["played"] is None or all(0 <= x < len(places) for x in r["played"]) or r["ply"] == len(moves)
        for f in r["frames"]:
            assert len(f["members"]) >= 1 and set(f["places"]) <= set(range(len(places)))
            for c in f["claims"]:
                seen_member_claim = True
                assert 0 <= c["member"] < len(f["members"]) and c["class"] in range(7)
                assert c["tried"] >= len(c["answers"]) and c["unread"] <= c["tried"]
                for (a, b, how, mv) in c["answers"]:
                    seen_answer = True
                    assert how in range(5) and mv in (0, 1) and 0 <= a < len(places) and 0 <= b < len(places)
    assert seen_member_claim and seen_answer
    assert res["rows"][0]["played"] is not None and res["rows"][2]["played"] is None                           # ply 5: the game's next move is known; at the end of the record (ply 8) it is not


def test_threat_list_on_one_position_reports_both_sides_with_the_bands(reader):
    moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3"]                                                           # the side to move has a unit attacked
    lst = TL.threat_list(moves, 5, reader, stake_of=lambda kind, own: 0.2, spread=0.1)
    assert set(lst) >= {"mover", "other", "raw"} and lst["raw"]["mover"] == 1
    for r in lst["mover"] + lst["other"]:
        assert r["band_spread"] in (1, 2, 3, 4) and r["band_rank"] in (1, 2, 3, 4)
        assert r["class"] in {"lost", "both", "threat", "conditional", "contested"}
        assert r["ended"] in TL.ENDED
        assert (r["band_spread"] in (2, 4)) == bool(r["response"])


def test_the_new_read_agrees_with_the_state_read_on_what_it_shares(reader):
    """`questions_threats` reads the frames and classes of `questions_state` (78.62): in the frames as built (not the smallest-frame variants), the units the state read holds in the classes lost, both, threat or
    conditional are the claims the new read lists in the same classes, by owner and kind. The refactor that shares the class code leaves the state read as it was."""
    import collections
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d4", "e5d4", "e1g1", "f6e4", "f1e1", "d7d5", "c4d5", "d8d5", "b1c3", "d5a5", "c3e4", "c8e6"]
    plies = [[p for p in range(6, len(moves) + 1, 2)]]
    new = reader.model.questions_threats([(None, moves)], plies, reader.keys, reader.goal, TL.TS.FUSE, TL.TS.ROW_FUSE, TL.OLD.MAX_SEED_SIZE, True, TL.TS.FUSE, 2)
    old = reader.model.questions_state([(None, moves)], plies, reader.keys, reader.goal, TL.TS.FUSE, TL.TS.ROW_FUSE, TL.OLD.MAX_SEED_SIZE, True, 0, TL.TS.HORIZON, 2)
    assert len(new["rows"]) == len(old["rows"]) == len(plies[0])
    for a, b in zip(new["rows"], old["rows"]):
        assert a["ply"] == b["ply"] and a["mover"] == b["mover"]
        got = collections.Counter((f["members"][c["member"]][0], f["members"][c["member"]][1], c["class"]) for f in a["frames"] if not f["id"].endswith("/small") for c in f["claims"] if c["class"] in (0, 2, 4, 5))
        want = collections.Counter((u[0], u[1], u[2]) for u in b["units"] if u[2] in (0, 2, 4, 5))
        assert got == want, (a["ply"], got, want)
