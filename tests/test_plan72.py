"""PLAN 72 THE PRICED REACH (Kaveh 2026-10-23): one word -- a unit reaches a job within k of its owner's moves at a price (material, tempi), the other side's quiet moves passed and its paying
captures assumed -- and its callers at the static leaf. Numbers are directional (OBS lines)."""
import chess

from catspace.incubator.analyst.discovery import agenda as AGM
from catspace.incubator.analyst.discovery import regress as R

I7K2H_AFTER_B3 = "8/4K3/pk6/8/P1p5/1p6/2B5/8 w - - 0 50"              # I7k2H: e7 Bc5+ Kd7 Bxe7 Kxe7 b3 -- the bishop is deflected, the c-pawn runs
GWIGP_AFTER_RXE6 = "8/5kp1/4r2p/R6K/P6P/8/8/8 w - - 0 47"              # GWigp: after Rxe6 (the miss): White's a-pawn is a passer four pushes out, our rook stands in front of it
HUBC_LEAF = "1N3k2/7p/p4r2/1p3p2/3P4/P1P5/bP3K2/6R1 b - - 0 40"        # 1hUBC: Nxc6 Bxa2 Nxb8 -- the knight on b8 has one safe square (d7): it is NOT stuck


def test_72_1_the_reach_is_priced_and_its_zero_price_is_attackable_in():
    k = chess.Board("4k3/5p2/8/8/8/8/8/4K1N1 w - - 0 1")
    r = R.reach_price(k, chess.G1, chess.F7, 3)
    assert r is not None and r["material"] == 0 and r["tempi"] == 2 and r["path"][0] == "g1h3"          # Nh3-g5 bears on f7, every landing safe (Lean `attackableIn`)
    assert R.reach_price(k, chess.G1, chess.F7, 1) is None                                               # not within one move
    # a landing that is taken: the price is what the exchange wins there
    b = chess.Board("4k3/3r4/8/8/8/8/8/R3K3 w - - 0 1")                                                 # our rook to bear on e8 in one: Ra8 is safe (nothing attacks a8); Re1?? no -- Ra8 bears on e8 along the rank
    r = R.reach_price(b, chess.A1, chess.E8, 2)
    assert r is not None and r["tempi"] == 1 and r["material"] == 0
    b2 = chess.Board("4k3/8/8/8/8/8/6b1/R3K3 w - - 0 1")                                                 # a8 is attacked by their bishop on g2: the one-move landing costs the rook (5); the two-move path (Rb1-b8) is free
    r2 = R.reach_price(b2, chess.A1, chess.E8, 2)
    assert r2 is not None and (r2["material"], r2["tempi"]) == (0, 2)
    r1 = R.reach_price(b2, chess.A1, chess.E8, 1)
    assert r1 is not None and r1["material"] == 5 and r1["tempi"] == 1                                   # within one move only the priced landing exists


def test_72_1_the_word_is_read_from_its_owner_and_a_unit_bearing_already_costs_nothing():
    b = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    r = R.reach_price(b, chess.A1, chess.A8, 1)
    assert r is not None and r["tempi"] == 0 and r["path"] == []
    b.turn = chess.BLACK                                                                                 # the side to move is irrelevant: the owner rolls
    assert R.reach_price(b, chess.A1, chess.A8, 1)["tempi"] == 0


def test_72_2_a_stopper_that_can_only_arrive_by_being_taken_is_lifted():
    b = chess.Board(I7K2H_AFTER_B3)
    b.push_uci("c2d1")
    b.turn = chess.BLACK
    defects, facts = R.promote_defects(b, chess.C4)
    assert not any(d.kind == "stopper" for d in defects) and any("no stopper" in s for s in facts.get("priced", []))      # the B on d1 bears on c2, where bxc2 answers Bxc2
    g = chess.Board(GWIGP_AFTER_RXE6)
    defects, facts = R.promote_defects(g, chess.A4)
    assert R.is_passer(g, chess.A4) and not any(d.kind == "stopper" for d in defects)                    # their rook on e6 reaches the a-file only where our rook takes it


def test_72_2_a_far_passer_is_a_promote_root_when_the_tempo_arithmetic_calls_it_unstoppable():
    assert AGM.PASSER_ROOTS
    from catspace.incubator.analyst.discovery.state import state_of
    ag = AGM.Agenda(state_of("8/8/8/8/P7/8/8/k6K w - - 0 1", "w"))
    assert any(q.goal == "promote" and q.pattern.get("pawn") == chess.A4 for q in ag.all)              # four pushes, beyond the old count: their king is 7 away, nothing holds it -- a root
    ag2 = AGM.Agenda(state_of(GWIGP_AFTER_RXE6, "w"))
    assert not any(q.goal == "promote" and q.pattern.get("pawn") == chess.A4 for q in ag2.all)         # four pushes with a defect (our own rook blocks it): a plan, not a question (the cost rule of 2026-10-23)


def test_72_2_the_unit_that_cannot_run_is_a_target_only_when_reached_below_its_worth_and_the_knight_on_b8_runs():
    b = chess.Board(HUBC_LEAF)
    b.turn = chess.WHITE
    assert R.can_run(b, chess.B8, "w")[0]                                                                # Nd7 is safe: the knight is not stuck (the earlier reading of 1hUBC was wrong here)
    assert not any(d["sq"] == chess.B8 for d in R.stuck_threats(b, "w"))
    # a knight in the corner with no safe square, their king three steps away, no guard of ours in time: a threat with no parry
    c = chess.Board("N7/1p6/1pk5/8/8/8/8/7K w - - 0 1")
    st = R.stuck_threats(c, "w")
    assert st and st[0]["sq"] == chess.A8 and st[0]["parry"] is None and st[0]["tempi"] <= 3


def test_72_2_the_static_leaf_reads_the_parry_once_more_with_the_pass_after_it():
    """I7k2H's leaf: at the position after e7 Bc5+ Kd7 Bxe7 Kxe7 b3 the bishop hangs; the one-ply keep family accepts Bd1 (c3 keeps the balance one ply), and after Bd1 their agenda with
    the move passed claims the c-pawn's promotion (the stopper lifted): the parry is not a parry YET. Read on the position itself: the save question's best route is not a keep."""
    assert AGM.PARRY_PASS and AGM.STUCK_LEAF
    from catspace.incubator.analyst.discovery.state import state_of
    ag = AGM.Agenda(state_of(I7K2H_AFTER_B3, "w"))
    ag.ask()
    d = ag.decision()
    assert d is None or d.value < 1                                                                      # nothing keeps the bishop: no proved claim of value 1 or more
    b = chess.Board(I7K2H_AFTER_B3)
    b.push_uci("c2d1")
    their = ag._their_agenda(b)
    assert any(q.goal == "promote" and q.status == "gap" and q.stake >= R.STAKE_PROMOTE for q in their.all), [(q.text, q.status) for q in their.all]      # the defects say unstoppable (the stopper lifted), no lemma fires yet: OPEN -- which makes our leaf a GAP (72.2 d)
    assert AGM.THEIR_GAPS
