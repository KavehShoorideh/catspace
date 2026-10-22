"""PLAN 67 (docs/ANALYST_PLAN.md: the plan's own weaknesses; Lean `Ideas.Plan`). The mirror is tested against the Lean statements (the bounds are
1-Lipschitz along the piece's real steps, exhaustively over the 64x64 squares -- a TEST may enumerate; the proof does not) and on the cases.
Numbers are directional (OBS lines)."""
import itertools

import chess

from catspace.incubator.analyst.discovery import plan67 as P
from catspace.incubator.analyst.discovery.race import kdist, lb


def _rook_step(s, t):
    return s != t and (chess.square_file(s) == chess.square_file(t) or chess.square_rank(s) == chess.square_rank(t))


def _bishop_step(s, t):
    return s != t and abs(chess.square_file(s) - chess.square_file(t)) == abs(chess.square_rank(s) - chess.square_rank(t))


def test_67_rdist_is_lipschitz_along_rook_steps_and_self_zero():
    """Lean `Plan.rdist_step`, `rdist_self`."""
    for s, s2, t in itertools.product(chess.SQUARES, chess.SQUARES, chess.SQUARES):
        if _rook_step(s, s2):
            assert P.rdist(s, t) <= P.rdist(s2, t) + 1
    assert all(P.rdist(t, t) == 0 for t in chess.SQUARES)


def test_67_bdist_is_lipschitz_along_bishop_steps_and_never_crosses_colour():
    """Lean `Plan.bdist_step`, `bishop_keeps_colour`."""
    for s, s2, t in itertools.product(chess.SQUARES, chess.SQUARES, chess.SQUARES):
        if _bishop_step(s, s2):
            assert P.bdist(s, t) <= P.bdist(s2, t) + 1
    assert P.bdist(chess.A1, chess.A2) == float("inf") and P.bdist(chess.A1, chess.H8) == 1 and P.bdist(chess.A1, chess.C1) == 2


def test_67_a_pure_pawn_ending_closes_by_arithmetic():
    """K+P v K with the king far: no weak moment reachable: the promotion is proved by the arithmetic alone (`plan_sound`)."""
    b = chess.Board("8/8/8/8/4P3/8/8/k3K3 w - - 0 1")           # their king on a1, our pawn e4: e5 e6 e7 e8=Q (4 moves); their king needs 4 to e5's square only after the pawn has left it
    pl = P.read(b, chess.E4)
    print("OBS 67 pure ending:", P.trace(b, pl))
    assert pl.n == 4 and pl.closed
    r = P.read(chess.Board("8/8/8/8/4P3/8/p7/k3K3 w - - 0 1"), chess.E4)        # their a-pawn promotes in 1: ours is not a claim by this plan (the race word compares)
    print("OBS 67 their race:", P.trace(chess.Board("8/8/8/8/4P3/8/p7/k3K3 w - - 0 1"), r))
    assert not r.closed and r.their_race == 1


def test_67_the_mate_ends_the_plan_and_a_stalemate_refuses_it():
    b = chess.Board("8/5pkp/4p3/4R3/p3P3/5P2/1r4PP/6K1 b - - 0 32")          # 3JaeM: a3 a2 a1=Q#
    pl = P.read(b, chess.A4)
    print("OBS 67 3JaeM:", P.trace(b, pl))
    assert pl.mate_at == 3 and pl.stalemate_at is None
    assert [r.weak.kind for r in pl.own_reach] == ["blockade"] and pl.own_reach[0].weak.sq == chess.A1       # the one weakness of the plan: Re5-a5-a1 blocks
    s = chess.Board("7k/8/6K1/8/8/8/5P2/8 w - - 0 1")                            # f3 f4 f5 f6 f7: their bare king on h8 has no move after f7? (f7 covers g8; Kg6 covers g7,h7) -> stalemate
    ps = P.read(s, chess.F2)
    print("OBS 67 stalemate:", P.trace(s, ps))
    assert ps.stalemate_at is not None and not ps.closed


def test_67_the_rook_in_front_steps_aside_and_the_weak_moment_is_read():
    """RoR8t's f-pawn (Kaveh's example): our rook on f4 is IN FRONT; the plan has it step aside and the pawn then stands unprotected."""
    b = chess.Board("1r6/8/8/5p2/5rk1/7p/8/6K1 b - - 0 1")          # RoR8t's shape (rand23_5: Kg4 Rf4 f5 h3 v Kg1 Rb8; the far pawns left out)
    pl = P.read(b, chess.F5)
    print("OBS 67 RoR8t f-pawn:", P.trace(b, pl))
    assert pl.moves and b.piece_type_at(pl.moves[0].from_square) == chess.ROOK          # the rook steps aside first
    assert any(w.kind == "capture" for w in pl.weak)                                   # ... and the pawn stands unprotected at some moment


def test_67_the_support_fix_derives_the_rook_behind_the_pawn():
    """4efZ2 (rand3_10): Stockfish's Ra5. The a-pawn's plan a2 a1=Q leaves the new queen unprotected (their Rf7 reaches a1 in 2); the support search finds
    the placements after which no weak moment of ours is reachable: Ra5 (the rook BEHIND the pawn) is one. Derived, not handwritten."""
    b = chess.Board("3k4/3P1R1p/8/2r1pP2/4P3/p3K1P1/7P/8 b - - 0 41")
    pl = P.read(b, chess.A3)
    print("OBS 67 4efZ2:", P.trace(b, pl))
    assert not pl.immune and any(r.weak.sq == chess.A1 and r.piece == "R" for r in pl.own_reach)
    fixes = P.support_fixes(b, chess.A3)
    sans = [san for _m, san, _p in fixes]
    print("OBS 67 4efZ2 support fixes:", sans)
    assert "Ra5" in sans and all(p2.immune for _m, _s, p2 in fixes)


def test_67_a_check_square_is_not_a_weakness_of_ours_but_is_counted():
    """3U6wp (rand5_10): the d-pawn's plan d7 d8=Q has no reachable capture or blockade (immune) while their rook reaches check squares: the chase reads those."""
    b = chess.Board("7R/8/1p1P4/p1p5/P1P5/1P4k1/1K5p/7r w - - 3 62")
    pl = P.read(b, chess.D6)
    print("OBS 67 3U6wp:", P.trace(b, pl))
    assert pl.immune and not pl.closed and any(r.weak.kind == "check" for r in pl.reach)
    assert P.support_fixes(b, chess.D6) == []                                           # already immune: no support needed


def test_67_the_agenda_claims_a_pure_ending_by_the_plan_arithmetic_and_offers_support_routes(monkeypatch):
    """Through the agenda (the table off, so the rule machinery is read): K+P v K far away is CLAIMED with lemma plan_sound (the trace prints the plan and
    its weak moments); 4efZ2's promote question carries the support routes Ra5/Rc1 (67.4) into its chase."""
    from catspace.incubator.analyst.discovery import race as RC
    from catspace.incubator.analyst.discovery import table as TB
    from catspace.incubator.analyst.discovery.agenda import Agenda
    from catspace.incubator.analyst.discovery.state import state_of
    monkeypatch.setattr(TB, "TABLE", False)
    monkeypatch.setattr(RC, "RACE", False)
    ag = Agenda(state_of("8/8/8/4P3/8/8/8/k3K3 w - - 0 1", "w"))          # (promote roots exist for pawns within 3 pushes)
    ag.ask()
    assert any(c.kind == "promote" and c.lemma in ("plan_sound", "unstoppablePasser2_sound") for c in ag.state.claims)      # the full-rules lemma claims first where it fires
    ag = Agenda(state_of("8/8/4PK2/8/8/8/1B6/7k w - - 0 1", "w"))        # our bishop: the full-rules lemma's bare-king form excludes it; the plan arithmetic claims
    ag.ask()
    text = ag.trace()
    print("OBS 67 agenda plan claim:", [l for l in text.splitlines() if "plan e7" in l or "CLAIM promote" in l][:2])
    assert any(c.kind == "promote" and c.lemma == "plan_sound" for c in ag.state.claims)
    ag2 = Agenda(state_of("3k4/3P1R1p/8/2r1pP2/4P3/p3K1P1/7P/8 b - - 0 41", "b"), seconds=5)
    ag2.ask()
    t2 = ag2.trace()
    assert "support (plan 67.4): Ra5" in t2 and "Ra5 (support:" in t2
