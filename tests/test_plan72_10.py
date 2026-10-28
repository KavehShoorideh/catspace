"""PLAN 72.10 THE LOCAL INNER AGENDA (docs/ANALYST_PLAN.md 72.10, Kaveh 2026-10-23): the chased goal and its frame are carried into the inner agenda of a reply; after their move it asks
Q1 the goal on the new board, Q2 their threats through the reply's diff, Q3 the price of their move (the delays), and the full agenda only where the goal is dead. The diff is
`regress.move_diff` (Lean `Ideas.Frame.chessDiff`; the neglect theorem `Ideas.Frame.chess_neglect`). Count budgets only; numbers are OBS lines (directional)."""
import random

import analyst_core as AC
import chess

from catspace.incubator.analyst.discovery import agenda as AGM
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import State

W0VFO = "2r2rk1/R5pp/5p2/2q1p3/3pQ3/8/2P2PPP/R5K1 w - - 4 30"     # rand8_10: Qb7 threatens mate on g7
DELAY = "5rk1/5ppp/8/8/r7/1N1B4/4QPP1/6K1 w - - 0 1"               # Qh5 threatens Qxh7#; their Ra1+ is a delay (it touches none of the frame), Nxa1 takes it back
ROUTES = "chase: make them move -- 6 forcing route(s): Rxg7+ (check), Qxh7+ (check), Qd5+ (check), Qb7 (threat of mate), Qg6 (threat of mate), Qg4 (threat of mate)"
SPECIAL = ["r3k2r/pppq1ppp/2n2n2/3pp3/1b1PP1b1/2NB1N2/PPPQ1PPP/R3K2R w KQkq - 0 1",       # both castlings, both sides
           "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",                  # en passant
           "1r2k3/P1P5/8/8/8/8/5ppp/4K3 w - - 0 1",                                          # promotions, a capturing one
           "4k3/8/8/4N3/8/8/1p6/R3K2R b KQ - 0 1",                                           # their promotion with a capture on a1
           "4k3/8/8/4N3/8/8/8/4RK2 w - - 0 1",                                               # a discovered check
           "3r2k1/5ppp/8/8/3B4/8/3R1PPP/3R2K1 w - - 0 1"]                                     # batteries: a front piece leaves, the one behind sees through


def _view(b: chess.Board, t: int) -> tuple:
    """What a word that reads the square `t` reads (Ideas.FrameLemma's two inputs): the piece on it and the attackers of each colour onto it, as sets."""
    return (b.piece_at(t), frozenset(b.attackers(chess.WHITE, t)), frozenset(b.attackers(chess.BLACK, t)))


def _positions() -> list:
    rng = random.Random(7210)
    out = list(SPECIAL)
    while len(out) < 50:
        b = chess.Board()
        for _ in range(rng.randrange(8, 70)):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
        if any(True for _ in b.legal_moves):
            out.append(b.fen())
    return out


def test_72_10_1_the_diff_holds_every_changed_square_on_50_random_positions():
    """The Lean condition, brute force: every square whose piece or whose attackers (either colour, as a set) a legal move changes lies in `move_diff`. `Pos.diff` (attacker counts) is
    not this: the OBS line counts the changed squares it misses."""
    moves = misses = count_misses = sizes = 0
    for fen in _positions():
        b = chess.Board(fen)
        for m in list(b.legal_moves):
            a = b.copy(stack=False)
            a.push(m)
            changed = {t for t in chess.SQUARES if _view(b, t) != _view(a, t)}
            d = R.move_diff(b, m)
            moves += 1
            sizes += len(d)
            missed = changed - d
            misses += len(missed)
            assert not missed, (fen, m.uci(), sorted(chess.square_name(s) for s in missed))
            count_misses += len(changed - set(AC.Pos(b.fen()).diff(AC.Pos(a.fen()))))
    print(f"OBS 72.10 (1) diff vs brute force: positions 50, legal moves {moves}, changed squares missed by move_diff {misses}, by the count-based Pos.diff {count_misses}, "
          f"mean diff size {sizes / moves:.1f} of 64")
    assert moves > 1000


def _w0vfo_goal(reply: str):
    """The root agenda on W0VFO, its first mate question chased by Qb7, and what the inner agenda after `reply` is handed."""
    outer = Agenda(State(W0VFO, "w", []))
    outer._chase_q = next(q for q in outer.all if q.goal == "mate")
    m = chess.Move.from_uci("e4b7")
    c = outer.board.copy(stack=False)
    c.push(m)
    loc = outer._local_goal(m, c, reply)
    d = c.copy(stack=False)
    d.push(chess.Move.from_uci(reply))
    return outer, loc, d


def test_72_10_2_after_a_parry_three_questions_are_asked_and_nothing_else_while_the_claim_is_alive():
    """Qb7 Rf7 (the rook interposes on the queen's line to g7): Q1 is the mate of the unit that threatened it (the queen on b7: Qxf7+, "a rook can, and is lost"), Q2 the queen
    the rook now attacks, Q3 the rook's own square; every other root of the board waits and is never asked."""
    outer, loc, d = _w0vfo_goal("f8f7")
    assert loc["kind"] == "mate" and loc["units"] == frozenset({chess.B7}) and not loc["delay"] and loc["frame"] & loc["diff"]
    assert {chess.F7, chess.F8, chess.G7} <= loc["diff"]                                   # the rook's two squares and the line it closed (b7-g7)
    ag = Agenda(State(d.fen(), "w", []), side_depth=0, chase_depth=1, ctx=outer.ctx, min_stake=1, frame=loc["frame"], goal=loc, diff=loc["diff"])
    ag.ask()
    L = ag.local
    print("OBS 72.10 (2) Qb7 Rf7:", ag.lines[0][:400])
    queued = [i.payload for i in ag.wl.deferred(ag._wf)]                                    # (plan 74 (c): the roots that wait are QUEUED in the worklist, no longer a list on `local`)
    assert L is not None and not L["full"] and queued
    assert all(L["cls"].get(id(q)) in ("Q1", "Q2", "Q3") for q in ag.all)                 # nothing else was asked
    assert not any(q in ag.all for q in queued)
    assert {L["cls"][id(q)] for q in ag.all} >= {"Q1", "Q2"}
    assert ag.best is not None and ag.best[2].what == "Qxf7+" and ag.best[0] >= 100
    assert ag.lines[0].startswith("local after Rf7 (") and "Q1 re-asked" in ag.lines[0] and "Q2 threats: save our Q on b7" in ag.lines[0]


def test_72_10_3_the_full_agenda_is_asked_only_when_the_chase_is_dead_there():
    """Qg6 hxg6 under the goal 'win their Q on c5': the queen is gone and the need is 10, so the goal has no question on this board -- the chased claim is dead there, and the full
    agenda (the mate roots, the only ones that can still bring the need) is asked. A check of theirs is a delay: its roots are the answers to it, read as today."""
    outer = Agenda(State(W0VFO, "w", []))
    outer._chase_q = next(q for q in outer.all if q.goal == "win" and q.pattern.get("target") == chess.C5)
    m = chess.Move.from_uci("e4g6")
    c = outer.board.copy(stack=False)
    c.push(m)
    loc = outer._local_goal(m, c, "h7g6")
    d = c.copy(stack=False)
    d.push(chess.Move.from_uci("h7g6"))
    ag = Agenda(State(d.fen(), "w", []), side_depth=0, chase_depth=1, ctx=outer.ctx, min_stake=10, frame=loc["frame"], goal=loc, diff=loc["diff"])
    ag.ask()
    print("OBS 72.10 (3) Qg6 hxg6:", ag.lines[0][:300])
    assert ag.local["full"] and "full agenda: the goal is dead here" in ag.lines[0]
    assert any(q.goal == "mate" for q in ag.all) and not ag.local["live"]
    outer2 = Agenda(State(DELAY, "w", []))                                                  # Qh5 threatens Qxh7# (the B on d3 guards h7); Ra1+ is a check touching none of the frame: a DELAY,
    outer2._chase_q = next(q for q in outer2.all if q.goal == "mate")                      # priced by what the exchange takes back (Nxa1: the rook)
    m = chess.Move.from_uci("e2h5")
    c2 = outer2.board.copy(stack=False)
    c2.push(m)
    loc2 = outer2._local_goal(m, c2, "a4a1")
    assert loc2["delay"] and loc2["check"] and loc2["price"] == 5 and not loc2["frame"] & loc2["diff"]
    c2.push(chess.Move.from_uci("a4a1"))
    ag2 = Agenda(State(c2.fen(), "w", []), side_depth=0, chase_depth=1, ctx=outer2.ctx, min_stake=1, frame=loc2["frame"], goal=loc2, diff=loc2["diff"])
    ag2.ask()
    print("OBS 72.10 (3) Qh5 Ra1+:", ag2.lines[0][:300])
    assert not ag2.local["full"] and "Q3 delays: Ra1+ (costs 5)" in ag2.lines[0] and "in check: their check is a delay" in ag2.lines[0]
    assert ag2.best is not None and ag2.best[2].what == "Nxa1"


def _run(local: bool, monkeypatch) -> tuple:
    monkeypatch.setattr(AGM, "LOCAL_INNER", local)
    ag = Agenda(State(W0VFO, "w", []))
    ag.ask()
    return ag, ag.decision()


def test_72_10_4_W0VFO_at_the_count_budget_proves_Qb7_plus_3_with_the_same_route_list_and_fewer_questions_per_inner_agenda(monkeypatch):
    """The acceptance at the count budget: Qb7 +3 by proof, every root question's route list as before; the OBS line: questions per inner agenda before (LOCAL_INNER off, the trace
    guard's before only) and after."""
    off, d_off = _run(False, monkeypatch)
    on, d_on = _run(True, monkeypatch)
    lists_on = [l.strip() for l in on.trace().splitlines() if "chase: make them move" in l]
    lists_off = [l.strip() for l in off.trace().splitlines() if "chase: make them move" in l]
    per = lambda a: a.ctx.asked / max(a.ctx.agendas, 1)
    print(f"OBS 72.10 (4) W0VFO count budget: before inner agendas {off.ctx.agendas}, inner questions {off.ctx.asked} ({per(off):.1f} per inner agenda); "
          f"after inner agendas {on.ctx.agendas}, inner questions {on.ctx.asked} ({per(on):.1f} per inner agenda), local {on.ctx.local_agendas}, full {on.ctx.local_full}, "
          f"closed by the frame {on.ctx.local_closed}; decision before {d_off.what} {d_off.value} proved {d_off.proved}, after {d_on.what} {d_on.value} proved {d_on.proved}")
    assert d_on is not None and d_on.what == "Qb7" and d_on.value == 3 and d_on.proved
    assert lists_on == lists_off and lists_on and all(l == ROUTES.strip() for l in lists_on)
    assert per(on) < per(off) and on.ctx.asked < off.ctx.asked
    assert on.ctx.local_agendas and on.ctx.local_closed and "local inner agendas one level below the root" in on.trace()
