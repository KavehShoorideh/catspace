"""PLAN 64 (docs/ANALYST_PLAN.md, 'the diagnosis round: every sound switch ends ON'). One position each: the numbers are directional (OBS lines)."""
import chess

import analyst_core as AC
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.analyse import analyse
from catspace.incubator.analyst.discovery.state import state_of

POS2 = "r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1"


def _pos2_after(*sans: str) -> str:
    b = chess.Board(POS2)
    for s in sans:
        b.push_san(s)
    return b.fen()


def test_64_0_principal_line_ends_at_our_last_capture():
    """After Nxf4 Rxf4 the exchange on f4 is Bxf4 exf4 Qxf4: two captures of ours, Qxf4 the last."""
    ag = Agenda(state_of(POS2, "w"))
    ((sq, caps),) = ag._principal_captures(AC.Pos(_pos2_after("Nxf4", "Rxf4")))
    print("OBS 64.0 principal captures after Nxf4 Rxf4:", [(s, last) for s, _b, last in caps])
    assert chess.square_name(sq) == "f4"
    assert [(s, last) for s, _b, last in caps] == [("Bxf4", False), ("Qxf4", True)]


def test_64_0_their_free_move_after_the_last_capture_is_read():
    """pos2: 1.Nxf4 Rxf4 2.Bxf4 exf4 3.Qxf4 Bg5 pins the queen to Kc1: the close by 'exchange +3' is NOT proved, the reply stays asked."""
    ag = Agenda(state_of(POS2, "w"))
    P2 = AC.Pos(_pos2_after("Nxf4", "Rxf4"))
    t2, cl2 = ag._their_after_capture(P2)
    print(f"OBS 64.0 their claim after our capture(s) on f4: {t2} after {cl2[0] if cl2 else None}")
    assert t2 >= 1 and cl2[0] == "Qxf4"
    assert ag._closed_by_rule(P2, -2) is None


def test_64_0_the_probe_finds_bg5():
    """The probe (data/derived/ledger_puzzles_probe_pos2.json): after 3.Qxf4 our own engine finds Bg5."""
    (p,) = PA.load_set("probe_pos2")
    r = analyse(p, quiet=True, memory=False, seconds=10)
    print("OBS 64.0 probe decision:", r["decision"], r["move"])
    assert str(r["move"]).startswith("h4g5")


def test_64_0_pos2_no_longer_decides_nxf4():
    """pos2 (data/derived/ledger_puzzles_kaveh_pos2.json): the false proof of Nxf4 +1 (the leaf of the binder's delayed_root_max claim is read by their agenda too) is gone."""
    (p,) = PA.load_set("kaveh_pos2")
    r = analyse(p, quiet=True, memory=False, seconds=10)
    print("OBS 64.0 pos2 decision:", r["decision"], r["move"])
    assert r["move"] != "h3f4"


def _chain_on(monkeypatch):
    from catspace.incubator.analyst.discovery import agenda as AGM
    for k in ("NEED_AWARE_CLOSE", "EVASION_CHASE", "NEED_PROPAGATION", "GOAL_JUDGE", "DEEP_WIN_CHASE"):
        monkeypatch.setattr(AGM, k, True)
    monkeypatch.setattr(AGM, "CHASE_CLOCK", 4)


def test_64_1_zB_after_rf1_is_never_killed_by_the_chain_state(monkeypatch):
    """2zB89 (probe zB_after = after 1.Qxa5 Qxa1+): with the chain items ON, 2.Rf1 Qd4+ is not 'no claim of ours' (a second check of theirs is a chain-stop GAP): Rf1 is a GAP or a claim."""
    _chain_on(monkeypatch)
    p = next(x for x in PA.load_set("probe_2zB89") if x["id"] == "zB_after")
    lines = []
    analyse(p, quiet=False, log=lines.append, memory=False, seconds=10)
    text = "\n".join(lines)
    print("OBS 64.1 Rf1: GAP lines", text.count("Rf1: GAP"), "killed lines", text.count("Rf1: killed"))
    assert "Rf1: killed" not in text
    assert "Rf1: GAP" in text or "Rf1: " in text


def test_64_1_a_chain_stop_is_a_gap_not_a_kill():
    """The chase clock / the tempo allowance / a second check cut the route as a GAP: counted as a chain stop (not a budget cut), `_cut` names it, the question is not filed as chased to its end."""
    ag = Agenda(state_of(POS2, "w"))
    q = ag.heap[0][4] if ag.heap else None
    assert q is not None
    before = (ag.ctx.budget_hits, ag.ctx.chain_hits)
    ag._chain_gap(q, "the chase clock: 4 moves of ours without a capture")
    assert (ag.ctx.budget_hits, ag.ctx.chain_hits) == (before[0], before[1] + 1) and q._chased is False
    assert ag._cut("Bxf4", "exf4", True, dict(chain=True, cut=False)).startswith("chain stop: Bxf4 not followed past exf4")
    assert ag._cut("Bxf4", "exf4", True, dict(chain=True, cut=True)).startswith("chase fuse")      # a real budget cut wins the label
    assert not hasattr(q, "_chain_cut")


def test_64_2_the_chain_items_are_on_by_default():
    from catspace.incubator.analyst.discovery import agenda as AGM
    assert all(getattr(AGM, k) for k in ("NEED_AWARE_CLOSE", "EVASION_CHASE", "NEED_PROPAGATION", "GOAL_JUDGE", "DEEP_WIN_CHASE"))
    assert AGM.CHASE_CLOCK == 4


def test_64_2_a_kill_made_for_a_big_need_is_not_a_standing_kill_for_need_one():
    """W0VFO (rand8_10) with NEED_PROPAGATION: 'Ra7+ Qxa7' killed for a need of several; it was served as a standing kill to Qb7 Qxa7, which needs 1 (Qb7 decides, Rxa7 recaptures)."""
    (p,) = [x for x in PA.load_set("rand8_10") if x["id"] == "W0VFO"]
    r = analyse(p, quiet=True, memory=False, seconds=10)
    print("OBS 64.2 W0VFO decision:", r["decision"], r["move"])
    assert r["move"] == "e4b7"


def test_64_2_07z0p_decides_qxd5_under_the_time_budget():
    """07Z0P (dev_20): with the chain items ON the COUNT fuse does not reach Qxd5+ (the in-check chase of the mate hunts, ~2x the questions); the time regime with the tiers does (30 s with the cost judge ON;
    10 s with --no-cost-judge: the judge's pricing of the 50-parry routes costs 4.5 of 10 s, see the hand-back)."""
    (p,) = [x for x in PA.load_set("dev_20") if x["id"] == "07Z0P"]
    r = analyse(p, quiet=True, memory=False, seconds=30)
    print("OBS 64.2 07Z0P decision at 30 s:", r["decision"], r["move"])
    assert str(r["decision"]).startswith("Qxd5")


def test_64_0_equal_exchanges_are_a_choice_of_ours_fork2():
    """FORK2 (Ne3 forks Rf1 and Rg2): after Ne3 Rc2 the captures Nxf1 and Nxc2 both win a rook; Nxf1 allows Rxc4+, Nxc2 does not: the close is proved by the safe one."""
    fen = "8/8/8/8/K1R5/P6k/6r1/3N1r2 w - - 0 1"
    b = chess.Board(fen)
    b.push_san("Ne3")
    b.push_san("Rc2")
    ag = Agenda(state_of(fen, "w"))
    lines = ag._principal_captures(AC.Pos(b.fen()))
    assert len(lines) == 2
    assert ag._their_after_capture(AC.Pos(b.fen()))[0] == 0


def test_64_3_the_cost_judge_is_on_and_a_route_read_carries_its_measured_seconds():
    """4pC83 (rand7_10): under a time budget the judge prices a route read before by its MEASURED seconds (finished: t; cut after t: more than t), and decides Nf3+."""
    from catspace.incubator.analyst.discovery import agenda as AGM
    assert AGM.COST_JUDGE is True
    (p,) = [x for x in PA.load_set("rand7_10") if x["id"] == "4pC83"]
    lines = []
    r = analyse(p, quiet=False, log=lines.append, memory=False, seconds=10)
    text = "\n".join(lines)
    print("OBS 64.3 4pC83 decision:", r["decision"], r["move"], "judge lines", text.count("judge:"))
    assert r["move"] == "d4f3"
    assert "judge:" in text


def test_64_5_the_play_server_fills_the_game_history_for_the_cycle_terminal():
    from catspace.incubator.analyst.discovery.agenda import _pos_key
    from catspace.incubator.analyst.discovery.play_server import game_history
    b = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8"):          # the start position is repeated after 4 plies
        b.push_san(san)
    keys = game_history(b)
    print("OBS 64.5 game history keys:", len(keys))
    assert _pos_key(b.fen()) in keys                    # the root itself repeats an earlier position of the game (white to move)
    assert len(keys) == 2                               # (start position == root, and the position after 1.Nf3 Nf6)
    assert game_history(chess.Board(POS2)) == set()


def test_64_5_the_estimate_is_on_and_inert_where_a_proved_claim_keeps_the_status():
    from catspace.incubator.analyst.discovery import agenda as AGM
    assert AGM.ESTIMATE is True and AGM.PARRIES is True and AGM.NO_SAFE_MOVE is True and AGM.DELAYED_THREAT is True
    (p,) = [x for x in PA.load_set("rand3_10") if x["id"] == "qCXKp"]
    r = analyse(p, quiet=True, memory=False, seconds=10)
    print("OBS 64.5 qCXKp:", r["decision"], r["move"])
    assert r["move"] == "a6b5"
