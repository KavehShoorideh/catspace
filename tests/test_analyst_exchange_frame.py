"""The exchange as a search inside its frame (exchange_frame.py): values against hand counts, the unpruned search and
the kernel's capture game; the recursive x-ray; Kaveh's examples."""
import chess

from catspace.incubator.analyst.exchange_frame import Search, brute, frame_of, orders, root_values


def _root(fen, sq):
    b = chess.Board(fen); s = chess.parse_square(sq)
    on_s, mine, theirs = frame_of(b, s)
    r = root_values(Search(True), on_s, mine, theirs)
    assert r == root_values(Search(False), on_s, mine, theirs)        # the lemmas never change a value
    return r, (on_s, mine, theirs)


def test_queen_takes_a_defended_pawn_is_down_eight():
    """Kaveh: 'you take a pawn with a queen ... you end up down a queen, up a pawn'; they stop after BxQ."""
    r, _ = _root("4k3/8/2b5/3p4/8/8/8/3QK3 w - - 0 1", "d5")
    assert r == {chess.D1: 1 - 9}


def test_pawn_takes_queen_when_they_must_recapture_with_the_queen():
    r, _ = _root("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", "d5")
    assert r[chess.E4] == 9


def test_the_gun_recursively():
    """Rooks d4, d3 and the queen d2 behind them against a bishop on e6: the x-ray chain joins one by one."""
    r, (on_s, mine, theirs) = _root("4k3/8/4b3/3n4/3R4/3R4/3Q4/1K6 w - - 0 1", "d5")
    assert [x.square for x in mine] == [chess.D4] and [y[1] for y in mine[0].behind] == [chess.D3, chess.D2]
    assert r[chess.D4] == 1                                   # Rxd5 Bxd5 Rxd5: +3 -5 +3
    assert brute(on_s, mine, theirs) == max(0, r[chess.D4])


def test_pawn_or_knight_recapture_both_tie_and_both_orders_are_returned():
    """Kaveh's case: two ways to win the knight on d5 with equal material; the search returns both for the plan
    layer (the pawn keeps or breaks the chain -- the subgoal's choice)."""
    r, (on_s, mine, theirs) = _root("4k3/8/8/3n4/4P3/2N5/8/4K3 w - - 0 1", "d5")
    assert r[chess.E4] == 3 and r[chess.C3] == 3
    lines = orders(on_s, mine, theirs)
    assert ("e4",) in [l[:1] for l in lines] and ("c3",) in [l[:1] for l in lines]


def test_alpha_beta_equals_the_unpruned_search_on_a_long_frame():
    b = chess.Board("r2qk2r/pQpn1ppp/3bp3/1N6/3P2b1/8/PPP3PP/R1B1K1NR w KQkq - 2 10")
    for s in (chess.C7, chess.D6):
        on_s, mine, theirs = frame_of(b, s)
        if not mine:
            continue
        for lem in (True, False):
            assert max(0, max(root_values(Search(lem), on_s, mine, theirs).values())) == brute(on_s, mine, theirs)


def test_a_prior_that_stops_is_counted_as_an_estimate():
    """The plain count as a prior: it answers every state below the root without search; the result is flagged as
    estimated, and on an honest frame it equals the exact value."""
    from catspace.incubator.analyst.exchange_frame import ExchangeFrame, exchange_lemmas
    from catspace.incubator.analyst.frame_search import FrameSearch, Prior

    def plain(st):                                             # the classical cheapest-first count from this state
        on_s, mine, theirs = st
        if not mine:
            return 0
        from catspace.incubator.analyst.exchange_frame import _play
        m2, t2 = _play(mine[0], mine, theirs)
        return max(0, on_s - plain((mine[0].value, t2, m2)))

    b = chess.Board("4k3/8/2b5/3p4/8/8/8/3QK3 w - - 0 1"); st = frame_of(b, chess.D5)
    fs = FrameSearch(ExchangeFrame(), exchange_lemmas(), Prior("plain count", estimate=plain))
    assert fs.root(st)[st[1][0]] == -8 and fs.estimated >= 1
    exact = FrameSearch(ExchangeFrame(), exchange_lemmas())
    assert exact.root(st)[st[1][0]] == -8 and exact.estimated == 0


def test_retract_reinstate_and_resume():
    """Kaveh: keep what the search did; retract what rested on a lemma, reinstate it without redoing, resume a stopped
    search where it left off."""
    b = chess.Board("r2qk2r/pQpn1ppp/3bp3/1N6/3P2b1/8/PPP3PP/R1B1K1NR w KQkq - 2 10")
    st = frame_of(b, chess.C7)
    exact = {r.square: v for r, v in Search(False).root(st).items()}
    fs = Search(True)
    assert root_values(fs, *st) == exact
    used = fs.decisions("both sides can exit")
    affected = fs.retract("both sides can exit")
    assert affected == used
    assert root_values(fs, *st) == exact                       # still exact without the lemma
    fs.reinstate("both sides can exit"); n = fs.nodes
    assert root_values(fs, *st) == exact and fs.nodes == n     # reinstated: nothing re-derived
    bud = Search(True); bud.budget = 1
    root_values(bud, *st)
    assert not bud.finished()
    bud.resume(None)
    assert root_values(bud, *st) == exact and bud.finished()
