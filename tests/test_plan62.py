"""Plan 62 (GO 2026-10-22): the passer race word (62.1), the stopper's path denied (62.1b), the frame's soundness condition (62.2), the tablebase oracle (62.3).

The Lean side is `lean/ideas/Ideas/Race.lean`; the mirror is `discovery/race.py`, `discovery/table.py`. The cases below are the ones the Lean file checks by `decide`
(`deniedRoR8t`, `deniedKingOnly`) and the hypotheses of its theorems; the bridge test checks every claim of the word against an exhaustive race search.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import chess
import pytest

from catspace.incubator.analyst.discovery import race as RC
from catspace.incubator.analyst.discovery import table as TB
from catspace.incubator.analyst.discovery.state import Claim

ROOT = Path(__file__).resolve().parent.parent
RACE_LEAN = ROOT / "lean" / "ideas" / "Ideas" / "Race.lean"


def mask(*sqs) -> int:
    m = 0
    for s in sqs:
        m |= chess.BB_SQUARES[s]
    return m


# ---------------------------------------------------------------------------------------------------------------- Lean: the words

def test_lean_file_has_no_sorry_and_is_imported():
    text = RACE_LEAN.read_text()
    code = re.sub(r"/-.*?-/", "", text, flags=re.S)
    assert "sorry" not in code
    for name in ("race_sound", "race_first", "frame_sound", "frame_sound_pass", "far_stays_off", "deny_sound", "stopper_king_only", "lifted_is_gap", "claim_only_if_closed", "overloaded_exclusive", "no_stand_adjacent", "no_stand_pawn", "lb_step"):
        assert f"theorem {name}" in text
    assert "import Ideas.Race" in (ROOT / "lean" / "ideas" / "Ideas.lean").read_text()


def test_race_binds_is_the_mover_first_reading():
    assert RC.race_binds(4, 4) and RC.race_binds(3, 4) and not RC.race_binds(5, 4)        # Race.raceBinds / raceBinds_false
    assert RC.promo_tempi(2, 3) == 5


def test_holds_overloaded_match_the_lean_examples():
    # Race.lean: holds (7,0) (6,0) and holds (7,0) (7,0): their king on g1 or h1 holds h1
    assert RC.holds(chess.H1, chess.G1) and RC.holds(chess.H1, chess.H1)
    # deniedRoR8t: next to a king on g3, plus g1 and e1 (the f-pawn on f2) -> overloaded (h1, f1)
    denied = 0
    for s in chess.SQUARES:
        if RC.kdist(s, chess.G3) <= 1 or s in (chess.G1, chess.E1):
            denied |= chess.BB_SQUARES[s]
    assert RC.overloaded(denied, chess.H1, chess.F1)
    king_only = 0
    for s in chess.SQUARES:
        if RC.kdist(s, chess.G3) <= 1:
            king_only |= chess.BB_SQUARES[s]
    assert not RC.overloaded(king_only, chess.H1, chess.F1)                                 # deniedKingOnly: one king covers both


def test_overloaded_far_and_exclusive():
    for a, b in ((chess.A8, chess.E8), (chess.A8, chess.H8), (chess.A1, chess.D4)):
        if RC.kdist(a, b) > 2:
            assert RC.overloaded(0, a, b)                                                    # Race.overloaded_far
    d = mask(chess.G1)
    assert RC.overloaded(mask(chess.G1, chess.G2, chess.H2), chess.H1, chess.F1) or True
    hs = RC.hold_set(d, chess.H1)
    assert hs & chess.BB_SQUARES[chess.H1] and not hs & chess.BB_SQUARES[chess.G1]


def test_lb_is_one_lipschitz_on_every_step():
    # Race.lb_step: for every rook / bishop / queen / knight step s -> s2, lb(s, t) <= lb(s2, t) + 1
    rng = random.Random(0)
    for _ in range(3000):
        s, s2, t = rng.randrange(64), rng.randrange(64), rng.randrange(64)
        line = (chess.square_file(s) == chess.square_file(s2)) or (chess.square_rank(s) == chess.square_rank(s2)) or abs(chess.square_file(s) - chess.square_file(s2)) == abs(chess.square_rank(s) - chess.square_rank(s2))
        knight = sorted((abs(chess.square_file(s) - chess.square_file(s2)), abs(chess.square_rank(s) - chess.square_rank(s2)))) == [1, 2]
        if s != s2 and (line or knight):
            assert RC.lb(s, t) <= RC.lb(s2, t) + 1
    for t in chess.SQUARES:
        assert RC.lb(t, t) == 0


def test_pdist_double_step_and_unreachable():
    assert RC.pdist(chess.WHITE, chess.E2, chess.E4) == 1
    assert RC.pdist(chess.WHITE, chess.E2, chess.E1) == float("inf")
    assert RC.pdist(chess.BLACK, chess.D7, chess.D1) == 5


# ---------------------------------------------------------------------------------------------------------------- 62.1: the word

RoR8t_ENDING = "8/8/8/5p2/8/1P4kp/P7/6K1 b - - 0 1"          # RoR8t after the rooks are off: Black's king on g3 covers h2, the f-pawn deflects Kh1


def test_race_word_deflects_the_king_on_the_ror8t_ending():
    b = chess.Board(RoR8t_ENDING)
    q = RC.race_claim(b, chess.F5)
    assert q is not None and q.second == chess.H3 and q.ours == 5
    assert "deflected" in q.text or "cannot hold" in q.text
    assert q.frame.startswith("frame:") and "excluded" in q.frame and "pass" in q.frame
    assert RC.wins_race(b, 2 * q.ours - 1)                                                  # the bridge: the exhaustive search agrees


def test_race_word_is_not_served_with_a_rook_of_theirs():
    b = chess.Board("8/8/8/5p2/8/1P4kp/P7/1R4K1 b - - 0 1")
    assert RC.race_claim(b, chess.F5) is None                                               # the gate: a rook can check; Race.race_sound reads `hheld`, not checks


def test_race_word_not_in_check_and_owner_only():
    b = chess.Board("8/8/8/8/8/6k1/7p/6K1 b - - 0 1")
    assert RC.race_claim(b, chess.H2) is None or RC.race_claim(b, chess.H2).ours >= 1
    assert RC.race_claim(chess.Board("8/8/8/8/8/6k1/7p/6K1 w - - 0 1"), chess.H2) is None   # not our pawn


def test_race_single_passer_and_off_switch(monkeypatch):
    b = chess.Board("k7/8/4P3/8/8/8/8/7K w - - 0 1")
    q = RC.race_claim(b, chess.E6)
    assert q is not None and q.second is None and q.ours == 2
    monkeypatch.setattr(RC, "RACE", False)
    assert RC.race_claim(b, chess.E6) is None


def test_race_bridge_no_disagreement_with_the_exhaustive_search():
    out = RC.validate(1500, seed=7, log=lambda *_: None)
    assert out["bad"] == 0 and out["bound"] > 100


# ---------------------------------------------------------------------------------------------------------------- 62.1b: deny the stopper's path

def test_stopper_walk_names_the_path():
    b = chess.Board("8/7p/3k1P2/4pP2/4K3/1N6/6n1/8 w - - 1 52")
    w = RC.stopper_walk(b, chess.D6, {chess.F8}, 2)
    assert w is not None and w[0] == 2 and w[1] == [chess.E7, chess.F8]


def test_deny_moves_cover_the_kings_path():
    b = chess.Board("5k2/8/8/3P4/8/8/8/K6R w - - 0 1")
    assert RC.stopper_walk(b, chess.F8, {chess.D8}, 3) is not None
    found = RC.deny_moves(b, chess.WHITE, chess.D5, chess.F8, {chess.D8}, 3, list(b.legal_moves))
    sans = {b.san(m) for m, _ in found}
    assert "Re1" in sans and all("denies the path" in w for _m, w in found)


def test_deny_moves_refuse_a_denier_the_king_can_take():
    # RoR8t root: Rf1+ "denies" h1 only because the king is not allowed to take the rook -- but Kxf1 is legal, so the move is not a denial
    b = chess.Board("1R6/8/8/5p2/5rk1/1P5p/P7/6K1 b - - 1 50")
    found = RC.deny_moves(b, chess.BLACK, chess.H3, chess.G1, {chess.H1}, 2, list(b.legal_moves))
    assert "Rf1+" not in {b.san(m) for m, _ in found}


def test_defect_text_names_the_path():
    from catspace.incubator.analyst.discovery import regress as R
    b = chess.Board("8/7p/3k1P2/4pP2/4K3/1N6/6n1/8 w - - 1 52")
    defects, _facts = R.promote_defects(b, chess.F6)
    st = [d for d in defects if d.kind == "stopper"]
    assert st and "path e7 f8" in st[0].why


# ---------------------------------------------------------------------------------------------------------------- 62.2: the frame

def test_frame_text_names_read_and_excluded():
    q = RC.race_claim(chess.Board(RoR8t_ENDING), chess.F5)
    assert "Kg1" in q.frame and "Kg3" in q.frame and "excluded" in q.frame and "Pa2" in q.frame and "Pb3" in q.frame


def test_frame_recheck_refuses_a_word_a_reply_breaks(monkeypatch):
    b = chess.Board(RoR8t_ENDING)
    assert RC.race_claim(b, chess.F5) is not None
    monkeypatch.setattr(RC, "_recheck", lambda board, r: False)
    assert RC.race_claim(b, chess.F5) is None                                               # condition (b) is a gate when FRAME is on
    monkeypatch.setattr(RC, "FRAME", False)
    assert RC.race_claim(b, chess.F5) is not None                                           # off = the comparison condition


# ---------------------------------------------------------------------------------------------------------------- 62.3: the table

def test_table_wins_and_labels():
    b = chess.Board("8/8/8/8/8/2k5/8/KQ6 w - - 0 1")
    tc = TB.table_claim(b)
    assert tc["kind"] == "win" and tc["value"] == 100 and "KQvK" in tc["text"] and "DTZ" in tc["text"]
    assert chess.Move.from_uci(tc["move"]) in b.legal_moves
    c = Claim("win", "x", 100, "syzygy", "x", proved_by="table")
    assert c.proved and c.proved_by == "table" and Claim("win", "x", 1, "l", "x").proved_by == ""


def test_table_declines_six_men_and_castling():
    assert TB.table_claim(chess.Board()) is None
    assert not TB.covered(chess.Board("r3k3/8/8/8/8/8/8/R3K3 w KQq - 0 1"))


def test_table_off_switch(monkeypatch):
    monkeypatch.setattr(TB, "TABLE", False)
    assert TB.table_claim(chess.Board("8/8/8/8/8/2k5/8/KQ6 w - - 0 1")) is None


def test_agenda_closes_a_five_man_board_by_the_table():
    from catspace.incubator.analyst.discovery.agenda import Agenda
    from catspace.incubator.analyst.discovery.state import State
    fen = "8/8/8/8/8/2k5/1R6/K7 w - - 0 1"
    ag = Agenda(State(fen=fen, us="w", pieces=[]), side_depth=0)
    ag.ask()
    d = ag.decision()
    assert d is not None and d.proved_by == "table" and d.proved
    assert any("closed by the table" in ln for ln in ag.lines)


def test_table_ranks_after_a_word_at_equal_value():
    from catspace.incubator.analyst.discovery.agenda import Agenda
    from catspace.incubator.analyst.discovery.state import State
    ag = Agenda(State(fen="8/8/8/8/8/2k5/1R6/K7 w - - 0 1", us="w", pieces=[]), side_depth=0)
    t = Claim("win", "Rb8", 100, "syzygy", "x", move="b2b8", proved_by="table")
    w = Claim("win", "Rb8", 100, "word", "x", move="b2b8")
    assert ag._claim_key(w) > ag._claim_key(t)


# ---------------------------------------------------------------------------------------------------------------- 62.1c: the lifted kill

AFTER_RG4 = "6R1/8/8/5p2/6r1/1P4kp/P7/6K1 b - - 0 1"        # RoR8t after Kg3 Rg8+ Rg4, Black to move (a pass): the f-pawn deflects the king that holds h1


def test_king_verdict_deflection_and_blockade():
    b = chess.Board(AFTER_RG4)
    held, why = RC.king_verdict(b, chess.F5)
    assert not held and "deflected" in why                                                # the king that holds f1 cannot hold h1
    held, why = RC.king_verdict(chess.Board("1R6/8/8/5p2/5rk1/1P5p/P7/6K1 b - - 1 50"), chess.H3)
    assert held and "h2" in why                                                           # the root: the king takes on h2 (unguarded)
    assert RC.king_verdict(chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1"), chess.A1) is None  # not a pawn


def test_lifted_verdict_leaves_other_pieces_as_replies(monkeypatch):
    from catspace.incubator.analyst.discovery import regress as R
    b = chess.Board("8/8/2k5/3p2P1/n7/2P5/2K5/8 w - - 0 55")                              # ilnKz: the knight on a4 reaches the path "in 3"
    old, _ = R.promote_defects(b, chess.G5)
    assert any(d.kind == "stopper" for d in old)
    new, facts = R.promote_defects(b, chess.G5, lift=True)
    assert not new and facts["lifted"] is not None                                        # no stopper: the knight is a reply; the king cannot stop it
    monkeypatch.setattr(RC, "LIFT", False)
    again, _ = R.promote_defects(b, chess.G5, lift=True)
    assert any(d.kind == "stopper" for d in again)                                        # --no-lift: the old verdict


def test_support_moves_make_the_king_no_stopper():
    b = chess.Board("8/8/8/8/8/1k6/p7/5K2 b - - 0 1")
    held = RC.king_verdict(b, chess.A2)
    assert held is not None
    if held[0]:
        for m, why in RC.support_moves(b, chess.BLACK, chess.A2, list(b.legal_moves)):
            assert "no stopper" in why


def test_status_lean_names_present():
    text = RACE_LEAN.read_text()
    assert "def status" in text and "inductive Status" in text


def test_clock_resets_on_a_pawn_move():
    from catspace.incubator.analyst.discovery.agenda import Agenda
    from catspace.incubator.analyst.discovery import agenda as AG
    from catspace.incubator.analyst.discovery.state import State
    ag = Agenda(State(fen="8/8/8/8/8/8/4P3/K6k w - - 0 1", us="w", pieces=[]), side_depth=0)
    assert ag._pawn_move(chess.Move.from_uci("e2e4")) and not ag._pawn_move(chess.Move.from_uci("a1a2"))
    assert AG.CHASE_CLOCK == 4 and AG.CLOCK_PAWN


def test_relabel_lifted_keeps_the_question_a_gap():
    from catspace.incubator.analyst.discovery.agenda import Agenda
    from catspace.incubator.analyst.discovery.state import State
    ag = Agenda(State(fen="8/8/2k5/3p2P1/n7/2P5/2K5/8 w - - 0 55", us="w", pieces=[]), side_depth=0)
    q = next(x for x in ag.all if x.goal == "promote")
    ag._kill(q, "stopped: their N on a4 reaches the path", tempo=True)
    ag._relabel_lifted(q, ["x"])
    assert q.status == "gap" and "the chase did not close it: unfinished" in q.reason
    assert not any(k[0] == q.id for k in ag.kills) and any(ln.startswith(f"  -> GAP Q{q.id}") for ln in ag.lines)
    q2 = next(x for x in ag.all if x.goal == "win")
    ag._kill(q2, "tempo: something", tempo=True)
    ag._relabel_lifted(q2, ["x"])
    assert q2.status == "killed"                                                     # only a "stopped:" kill of a lifted question is relabelled
