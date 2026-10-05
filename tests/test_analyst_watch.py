"""The surprise watcher (catspace/incubator/analyst/watch): what the game did is booked square by square and adds up
to the material change; the exchange evaluator; the rule language, the selection rule (more specific, then newer),
smoothing toward the parent; the forecast is a distribution; no forecast sees its own answer."""
import chess
import numpy as np
import pytest

from catspace.incubator.analyst.watch.frames import material, observe, see


@pytest.fixture()
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv("CATSPACE_WATCH_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("CATSPACE_WATCH_RECORD", str(tmp_path / "record"))
    return tmp_path


def _game(ucis):
    moves = [chess.Move.from_uci(u) for u in ucis.split()]
    boards = [chess.Board()]
    for m in moves:
        nb = boards[-1].copy(); nb.push(m); boards.append(nb)
    return boards, moves


def test_booking_adds_up_to_the_material_change():
    pytest.importorskip("pyarrow")
    from catspace.incubator.analyst.watch.watcher import games
    checked = 0
    for g in games(0, 25):
        boards, moves = _game(g["moves"])
        for t in range(1, len(moves)):
            o = observe(boards, moves, t)
            mover = not boards[t].turn
            assert o["delta"] == material(boards[o["end"]], mover) - material(boards[t], mover), (g["game_id"], t)
            assert t <= o["end"] <= t + 12
            checked += 1
    assert checked > 1000


def test_window_closes_after_two_quiet_moves():
    boards, moves = _game("e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 g1f3 g8f6")
    o = observe(boards, moves, 2)          # after 2...d5, White takes on d5, Black retakes, then two quiet moves
    assert o["by_square"] == {chess.D5: 0} and o["delta"] == 0 and o["end"] == 4


def test_see():
    b = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")                     # exd5: a free pawn
    assert see(b, chess.D5, chess.WHITE) == 1
    b = chess.Board("4k3/2p5/3p4/8/8/8/3Q4/4K3 w - - 0 1")                   # Qxd6 cxd6: the queen would be lost, so 0
    assert see(b, chess.D6, chess.WHITE) == 0
    b = chess.Board("4k3/8/3q4/8/8/8/8/3RK3 w - - 0 1")                      # Rxd6: a free queen
    assert see(b, chess.D6, chess.WHITE) == 9
    b = chess.Board("4k3/8/3r4/2P5/8/8/8/3RK3 w - - 0 1")                    # cxd6 wins the rook, Rd1 behind
    assert see(b, chess.D6, chess.WHITE) == 5


def test_language_selection_and_smoothing(roots):
    from catspace.incubator.analyst.watch import rules as R
    with pytest.raises(R.LanguageError):
        R.check({"cmp": "n_att", "op": "!", "value": 1})
    assert R.holds({"all": [{"flag": "hanging"}, {"cmp": "occ_value", "op": ">=", "value": 3}]}, dict(hanging=True, occ_value=3))
    r1 = R.add_rule("frame", "F0", {"flag": "hanging"}, "a hanging piece", ["c1"])
    r2 = R.add_rule("frame", r1.id, {"flag": "attacker_to_move"}, "a hanging piece whose attacker moves next", ["c2"])
    r3 = R.add_rule("frame", "F0", {"cmp": "occ_value", "op": ">=", "value": 3}, "a piece", ["c3"])
    rb = R.RuleBase.load()
    a = dict(hanging=True, attacker_to_move=True, occ_value=3)
    rid, matched = rb.select("frame", a)
    assert set(matched) == {"F0", r1.id, r2.id, r3.id}
    assert rid == r3.id                                                      # r2 and r3 are both most specific; r3 is newer
    assert rb.select("frame", dict(hanging=True, attacker_to_move=True, occ_value=1))[0] == r2.id
    assert rb.dist(r2.id) == pytest.approx(rb.dist("F0"))                    # no cases yet: the parent's forecast
    rb.update(matched, 3)
    d = rb.dist(r2.id)
    assert sum(d) == pytest.approx(1.0) and d[13] > rb.dist("F0")[13] - 1e-12


def test_forecast_is_a_distribution_and_sees_no_answer(roots):
    from catspace.incubator.analyst.watch import rules as R
    from catspace.incubator.analyst.watch.watcher import convolve, watch_game
    p, off = convolve([np.full(21, 1 / 21)] * 3)
    assert p.sum() == pytest.approx(1.0) and off == -30
    rb = R.RuleBase.load()
    g = dict(game_id=1, white_elo=1500, black_elo=1500, result=1, termination="Normal",
             moves="e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5c6 d7c6 e1g1 f7f6 d2d4 e5d4 f3d4 c6c5 d4b3 d8d1 f1d1")
    cases = watch_game(g, rb)
    assert all(c["p_result"] == pytest.approx(1 / 3) for c in cases)        # the result is revealed only at the end
    assert sum(rb.counts["R0"]) == pytest.approx(1.0)                        # and then counted ONCE: a game is one observation
    first = next(c for c in cases if c["frames"])                           # no frame window has closed before it:
    assert all(f["p"] == pytest.approx(1 / 21) for f in first["frames"])     # the frame forecast is still uniform
    later = [f["p"] for c in cases if c["ply"] > first["end"] for f in c["frames"]]
    assert later and any(abs(p - 1 / 21) > 1e-9 for p in later)              # after a reveal, it has learned


def test_a_reference_shifts_the_forecast(roots):
    """A rule with a reference counts offsets: 'the exchange goes as counted' learns once for every piece value."""
    from catspace.incubator.analyst.watch import rules as R
    r = R.add_rule("frame", "F0", {"flag": "attacker_to_move"}, "the exchange happens", ["c"], ref={"attr": "see_mover"})
    rb = R.RuleBase.load()
    for v in (-3, -5, -9, -1):                                               # each time, the mover lost exactly the counted value
        a = dict(attacker_to_move=True, see_mover=v)
        rb.update(rb.select("frame", a)[1], v, a)
    d = rb.dist(r.id, dict(attacker_to_move=True, see_mover=-5))
    assert sum(d) == pytest.approx(1.0) and max(range(21), key=lambda i: d[i]) - 10 == -5
    with pytest.raises(R.LanguageError):
        R.add_rule("frame", "F0", {"flag": "hanging"}, "bad", ["c"], ref={"attr": "x", "scale": 2})
