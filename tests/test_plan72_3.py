"""PLAN 72.3 THE JOIN QUESTION and THE STAKE RULE (Kaveh 2026-10-23): a defect that needs a piece of ours at L gets "bring U to bear on L", priced by the reach word (72.1); it is not
pushed when the price plus what THEIR counter-plan buys with the free moves exceeds what the plan wins. Constructed positions, count budgets only."""
import chess

from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.graph import clear_caches


def _join(fen, stake, kind, sq, pattern=None, k=None):
    clear_caches()
    b = chess.Board(fen)
    q = R.new_question("mate", stake, dict(pattern or {}))
    return b, R.join_fixes(b, "w", R.Defect(kind, sq), q, k=k)


def test_72_3_a_knight_joining_is_a_route_with_price_zero_and_its_text():
    b, f = _join("4k3/8/8/8/8/8/8/4K1N1 w - - 0 1", 100, "flight", chess.F7)
    assert f is not None
    r = next(x for x in f.routes if x.move.from_square == chess.G1)
    assert r.move == chess.Move.from_uci("g1h3") and r.price == (0, 2)
    assert r.why == "bring the N from g1 to bear on f7 in 2 (via h3), price 0"
    assert r.san == "Nh3"


def test_72_3_an_unsafe_landing_is_priced():
    b, f = _join("4k3/8/8/8/8/8/6b1/R3K3 w - - 0 1", 100, "flight", chess.E8, k=1)       # Ra8 bears on e8 but the bishop on g2 takes on a8: the rook is the price
    r = next(x for x in f.routes if x.move.from_square == chess.A1)
    assert r.price == (5, 1) and r.why.endswith("in 1, price 5")


def test_72_3_the_stake_rule_refuses_a_join_their_counter_plan_beats():
    fen = "4k2r/5p2/8/8/3N4/2K5/8/B7 w - - 0 1"                    # Nd4-f5 bears on... d6/h6 in two; meanwhile Rh8-h1 hits the loose bishop on a1 along the first rank
    b, f = _join(fen, 3, "flight", chess.F7)
    assert f is not None and not [x for x in f.routes if x.move.from_square == chess.D4]       # the fix is NOT PUSHED
    assert any("the N cannot join in time: their R on h8 reaches our B on a1 in 1" in w for w in f.why)
    b, f = _join(fen, 100, "flight", chess.F7)                       # the same race against a mate stake: the join stands, with its price in the route
    assert any(x.move.from_square == chess.D4 and x.price is not None for x in f.routes)


def test_72_3_joins_are_quiet_plans_kept_out_of_the_fixes_and_route_cost_pays_their_tempi():
    clear_caches()
    b = chess.Board("4k3/5p2/8/8/8/8/8/4K1N1 w - - 0 1")
    q = R.new_question("win", 3, {"target": chess.F7})
    q.defects = [R.Defect("not attacked", chess.F7)]
    R.fixes_of(q.defects[0], q, b, "w")
    assert not any(f.text.startswith("bring a unit") for f in q.fixes)                 # a quiet plan is no forcing fix: tempo_ok / all_routes / the chase never see it
    assert not R.tempo_ok(q)
    j = R.joins_of(q, b, "w")
    assert j and any(r.why.startswith("bring the N from g1") for r in j[0].routes) and q.joins == j
    m = chess.Move.from_uci("g1h3")
    base, _ = R._route_cost(b, m, 3, "w")
    assert R.route_cost(b, m, 3, "w")[0] == base + 1               # two tempi: one extra free move for them
    q2 = R.new_question("win", 3, {"target": chess.F7})
    q2.defects = q.defects
    R.JOIN_FIXES = False
    try:
        assert R.joins_of(q2, b, "w") == []
    finally:
        R.JOIN_FIXES = True


# ---- the cost, by scoping (2026-10-23 evening): the same answers, read where they can matter -------------------------------------------------------------

_SCOPE_FENS = ["4k3/8/8/8/8/8/8/4K1N1 w - - 0 1", "4k3/8/8/8/8/8/6b1/R3K3 w - - 0 1", "r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1",
               "r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1"]


def test_72_cost_a_reach_read_only_where_it_does_the_job_is_the_same_reach():
    """`reach_price` asks `reach_map` for the landings that do the job only (the last step's other landings are not priced); the answer is the one the whole map gives."""
    for fen in _SCOPE_FENS:
        b = chess.Board(fen)
        for sq in chess.SQUARES:
            pc = b.piece_at(sq)
            if pc is None or pc.color != chess.WHITE:
                continue
            for target in (t for t in (chess.F7, chess.E8, chess.A8, chess.H2) if t != sq):
                clear_caches()
                got = R.reach_price(b, sq, target, 3)
                clear_caches()
                full = [e for at, e in R.reach_map(b, sq, 3).items() if target in e["board"].attacks(at)]
                want = min(((e["material"], e["tempi"]) for e in full), default=None)
                if target in R.G.with_turn(b, pc.color).attacks(sq):
                    want = (0, 0)
                assert (None if got is None else (got["material"], got["tempi"])) == want


def test_72_cost_the_geometry_check_never_rejects_a_unit_that_can_bear():
    for fen in _SCOPE_FENS:
        b = chess.Board(fen)
        for sq in chess.SQUARES:
            pc = b.piece_at(sq)
            if pc is None or pc.color != chess.WHITE:
                continue
            for L in (x for x in chess.SQUARES if x != sq):
                clear_caches()
                r = R.reach_price(b, sq, L, 3)
                if r is not None:
                    assert R._can_bear_in_principle(pc, sq, L, 3), (fen, chess.square_name(sq), chess.square_name(L))


def test_72_cost_the_join_is_asked_at_the_root_not_in_the_inner_agendas_of_a_chase():
    from catspace.incubator.analyst.discovery.agenda import Agenda
    from catspace.incubator.analyst.discovery.state import State
    ag = Agenda(State(fen=_SCOPE_FENS[2], us="w", pieces=[]))
    q = R.new_question("mate", 100, {})
    assert ag._joins(q, chess.Board(_SCOPE_FENS[2]), "w") == [] or ag.chase_depth == 0
    inner = Agenda(State(fen=_SCOPE_FENS[2], us="w", pieces=[]), chase_depth=1, ctx=ag.ctx)
    assert inner._joins(q, chess.Board(_SCOPE_FENS[2]), "w") == []
