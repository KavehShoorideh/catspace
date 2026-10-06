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


def test_position_words():
    from catspace.incubator.analyst.watch.watcher import mate_in_one, position_attrs, promotes_next
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert mate_in_one(b)                                                    # Ra8#
    assert not mate_in_one(chess.Board("6k1/5pp1/7p/8/8/8/8/R5K1 w - - 0 1"))   # luft on h7
    p = chess.Board("8/4P3/8/8/8/8/k7/6K1 b - - 0 1")                         # Black to move; White's e7 pawn will promote
    assert promotes_next(p, chess.WHITE) and not promotes_next(p, chess.BLACK)
    a = position_attrs(chess.Board("8/4P3/8/8/8/8/k7/6K1 w - - 0 1"), chess.BLACK, [], (1500, 1500))
    assert a["promo_tomove"] and not a["mate1_tomove"]


def test_memory_carries_only_what_the_move_cannot_change():
    """Every carried frame's attributes equal a full recomputation, on real games; memory counts age and declines."""
    pytest.importorskip("pyarrow")
    from catspace.incubator.analyst.watch.frames import FrameMemory, attributes
    from catspace.incubator.analyst.watch.watcher import games
    checked = carried = 0
    for g in games(0, 15):
        mem = FrameMemory()
        b = chess.Board()
        for i, u in enumerate(g["moves"].split()):
            m = chess.Move.from_uci(u)
            before = b.copy()
            b.push(m)
            got = mem.step(before, m, b, i + 1)
            for sq, a in got.items():
                full = attributes(b, sq, before.turn, m)
                assert {k: a[k] for k in full} == full, (g["game_id"], i + 1, chess.square_name(sq))
                checked += 1
        carried += mem.carried
    assert checked > 2000 and carried > 0                                    # carried where the move cannot reach, all of it right
    # (OBS 2026-10-05: about 11% of frames are carried -- moves happen where the contested squares are)


def test_memory_counts_declines():
    from catspace.incubator.analyst.watch.frames import FrameMemory
    mem = FrameMemory()
    b = chess.Board("4k3/8/8/3n4/8/8/8/3QK3 w - - 0 1")                       # White to move; nothing attacked yet
    seq = ["d1d4", "e8e7", "e1e2", "e7e8"]                                    # Qd4 hits the knight; Black moves the king instead
    for i, u in enumerate(seq):
        m = chess.Move.from_uci(u); before = b.copy(); b.push(m)
        out = mem.step(before, m, b, i + 1)
    a = out[chess.D5]                                                         # Black's d5 knight: attacked since ply 1
    assert a["age"] == 3 and a["declined"] == 1                               # White could take at ply 3 (Ke2) and did not


def test_backfill_blend_and_per_game(roots):
    """A new rule starts from the stored past cases it covers (results once per game; cases missing a word skipped);
    a young rule with no cases forecasts what the rule it supersedes forecasts; results average per game."""
    import json
    from catspace.incubator.analyst.watch import rules as R
    from catspace.incubator.analyst.watch.watcher import backfill, data_root, per_game_mean
    d = data_root(); d.mkdir(parents=True, exist_ok=True)
    fr = lambda h, o: dict(attrs=dict(hanging=h, occ_value=3), outcome=o)
    cases = [dict(game=1, frames=[fr(True, -3), fr(False, 0)], elsewhere=dict(outcome=0), pattrs=dict(material=4), result="win"),
             dict(game=1, frames=[fr(True, -3)], elsewhere=dict(outcome=0), pattrs=dict(material=5), result="win"),
             dict(game=2, frames=[dict(attrs=dict(occ_value=3), outcome=0)], elsewhere=dict(outcome=0), pattrs=dict(material=-1), result="loss")]
    (d / "batch_000.json").write_text(json.dumps(dict(summary={}, cases=cases)))
    rf = R.add_rule("frame", "F0", {"flag": "hanging"}, "hanging", ["1:1:e4"])
    rr = R.add_rule("result", "R0", {"cmp": "material", "op": ">=", "value": 3}, "up", ["1:1"])
    rb = R.RuleBase.load()
    used = backfill(rb, {rf.id, rr.id}, 1)
    assert used[rf.id] == [2, 1]                                             # two hanging frames; one case predates the word
    assert rb.counts[rf.id][-3 + 10] == 2
    assert sum(rb.counts[rr.id]) == pytest.approx(1.0)                       # game 1's two positions share one observation
    young = R.add_rule("frame", "F0", {"cmp": "occ_value", "op": ">=", "value": 3}, "a piece", ["1:2:e4"])
    rb2 = R.RuleBase.load(counts={k: list(v) for k, v in rb.counts.items()})
    a = dict(hanging=True, occ_value=3)
    d_blend, lead, matched, how = rb2.forecast("frame", a)
    assert lead == young.id and how["superseded"] == rf.id and how["weight"] == 0.0
    assert d_blend == pytest.approx(rb2.dist(rf.id, a))                      # no cases yet: the superseded rule speaks
    assert per_game_mean([dict(game=1, b=2.0), dict(game=1, b=4.0), dict(game=2, b=0.0)], "b") == pytest.approx(1.5)


def test_cases_exchange_outcome_and_retrieval(roots):
    """The case database: a capture's exchange count, success = the material back within N plies (or mate), and
    retrieval of the same event's nearest cases from the index."""
    from catspace.incubator.analyst.watch import cases as C
    assert C.exchange_for(chess.Board("4k3/8/2p5/3b4/8/8/8/3QK3 w - - 0 1"), chess.Move.from_uci("d1d5")) == -6
    assert C.exchange_for(chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"), chess.Move.from_uci("e4d5")) == 1
    # Bxf7+ Kxf7 then Qh5+ forking the king and the e8... built so White wins the material back with a queen fork
    boards, moves = _game("e2e4 e7e5 f1c4 g8f6 c4f7 e8f7 d1h5 f6h5")
    out = C.outcome(boards, 4, chess.WHITE)                                   # Bxf7+ gave a bishop for a pawn; the queen was then lost too
    assert out["success"] == 0
    boards, moves = _game("e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 c4f7 e8f7 f3g5 f7g8 d1f3 d8e7 f3f6 e7f6")
    assert C.outcome(boards, 6, chess.WHITE)["success"] == 0
    # the material comes back and stands: Bxf7+ Kxf7, Ne5+ forks king and queen, Ke7, Nxd7 Kxd7
    boards = [chess.Board("4k3/3q1p2/8/8/2B5/5N2/8/6K1 w - - 0 1")]
    for u_ in "c4f7 e8f7 f3e5 f7e7 e5d7 e7d7".split():
        nb = boards[-1].copy(); nb.push_uci(u_); boards.append(nb)
    assert C.exchange_for(boards[0], chess.Move.from_uci("c4f7")) == -2      # a bishop for a pawn: a sacrifice
    assert C.outcome(boards, 0, chess.WHITE)["success"] == 1
    ev = C.scan_game(dict(game_id=1, white_elo=1600, black_elo=1600, result=1, termination="Normal",
                          moves="e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 c4f7 e8f7"))
    assert [(e["capturer"], e["victim"], e["investment"], e["recapturer"]) for e in ev] == [("B", "P", 2, "K")]
    import sqlite3
    C.db_path().parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(C.db_path())
    con.execute(f"CREATE TABLE events ({', '.join(C.COLS)}, PRIMARY KEY (game, ply))")
    base = dict(fen="x", move="x", kind="sacrifice", capturer="Q", victim="B", recapturer="P", investment=6, gives_check=0, king_dist=4,
                king_zone_attackers=2, king_shield=3, material=0, phase=40, elo=1800, elo_diff=0, result=0, term="Normal", back_in=None, mate=0)
    rows_ = [dict(base, game=g, ply=1, king_dist=d, success=s) for g, d, s in ((1, 1, 1), (2, 1, 1), (3, 7, 0), (4, 6, 0), (5, 4, 1))]
    con.executemany(f"INSERT INTO events VALUES ({', '.join('?' * len(C.COLS))})", [[r.get(c) for c in C.COLS] for r in rows_])
    con.commit(); con.close()
    near = C.similar(dict(base, game=99, ply=1, success=0), k=3)
    assert [r["game"] for r in near][0] == 5                                 # the same event, the nearest king distance first
