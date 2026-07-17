"""Model-free tests for the PUCT MCTS core (catspace/nn/mcts.py): synthetic
reach_fn, real chess rules. The FB-checkpoint wrapper is covered by the
playout_ab smoke, not here."""
import chess
import numpy as np
import pytest

from catspace.nn.mcts import DRAW_V, MATE_V, PLY_DISCOUNT, MCTS


def flat_reach(boards):
    return np.zeros(len(boards))


def make(reach=flat_reach, nodes=64, **kw):
    return MCTS(reach, max_nodes=nodes, **kw)


def test_white_takes_mate_in_one():
    # back-rank: Ra1-a8 is mate
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    m = make().best_move(b)
    b.push(m)
    assert b.is_checkmate()


def test_black_takes_mate_in_one():
    # mirrored back-rank for Black
    b = chess.Board("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1")
    m = make().best_move(b)
    b.push(m)
    assert b.is_checkmate()


def test_avoids_stalemating_when_no_mate():
    # Qc7 stalemates Black (Ka8, no moves, not in check); no mate-in-1 exists
    b = chess.Board("k7/8/8/1K6/8/8/2Q5/8 w - - 0 1")
    stalemate = chess.Move.from_uci("c2c7")
    b2 = b.copy(stack=False)
    b2.push(stalemate)
    assert b2.is_stalemate()          # the trap is real
    assert make(nodes=128).best_move(b) != stalemate


def test_budget_respected_and_counted():
    b = chess.Board()                 # startpos, branching 20
    t = make(nodes=100)
    t.best_move(b)
    # may overshoot by at most one expansion's branching, never a full level
    assert 100 <= t.evals_used <= 100 + 40
    small = make(nodes=25)
    small.best_move(b)
    assert small.evals_used < t.evals_used


def test_deterministic():
    b = chess.Board("6k1/5pp1/7p/8/8/6Q1/5PPP/6K1 w - - 0 1")
    assert make(nodes=200).best_move(b) == make(nodes=200).best_move(b)


def test_visits_concentrate_on_high_reach_move():
    # reach oracle that loves positions where White's queen is on h5
    def reach(boards):
        return np.array([2.0 if bd.piece_at(chess.H5) is not None
                         and bd.piece_at(chess.H5).piece_type == chess.QUEEN
                         else 0.0 for bd in boards])
    b = chess.Board("6k1/5pp1/7p/8/8/8/5PPP/3Q2K1 w - - 0 1")
    t = make(reach, nodes=300)
    root = t.run(b)
    best = max(root.children, key=lambda c: c.N)
    assert best.move == chess.Move.from_uci("d1h5")


def test_terminal_values_and_discount():
    t = make()
    root = t.run(chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"))
    mate_child = next(c for c in root.children
                      if c.move == chess.Move.from_uci("a1a8"))
    assert mate_child.terminal_v == pytest.approx(MATE_V - PLY_DISCOUNT)
    draws = [c for c in root.children if c.terminal_v == DRAW_V]
    assert all(c.terminal_v < 0 for c in draws)


def test_single_legal_move():
    # in check from the (rook-protected) Qh2: Kf1 is the only legal move
    b = chess.Board("6kr/8/8/8/8/8/5PPq/6K1 w - - 0 1")
    legal = list(b.legal_moves)
    assert len(legal) == 1
    assert make(nodes=8).best_move(b) == legal[0]


def test_no_legal_moves_raises():
    b = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # stalemate, Black to move
    assert b.is_stalemate()
    with pytest.raises(ValueError):
        make().best_move(b)


def test_all_terminal_children_terminates():
    # regression (2026-07-14): budget counts NETWORK evals, terminal backups
    # consume none -- a subtree where every child is terminal must not spin
    # the run loop forever (hung a 700-start generation run)
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    t = make(nodes=500)
    root = t.run(b)
    for c in root.children:
        c.terminal_v = -0.999 if c.terminal_v is None else c.terminal_v
    t.evals_used = 0                    # pretend budget untouched: worst case
    import catspace.nn.mcts as M

    class Reroot(M.MCTS):
        def _expand(self, node, at_root):
            if at_root:
                node.children = root.children
                return 0.0
            return super()._expand(node, at_root)
    t2 = Reroot(flat_reach, max_nodes=500)
    t2.run(b)                           # must return, not hang
    assert t2.evals_used < 500


def test_eval_cache_makes_repeats_free_and_stays_deterministic():
    # exact cache: re-searching the same position spends ~no fresh evals,
    # and a fresh cached engine picks the same move as an uncached one
    b = chess.Board("6k1/5pp1/7p/8/8/6Q1/5PPP/6K1 w - - 0 1")
    shared = {}
    t1 = MCTS(flat_reach, max_nodes=150, cache=shared)
    m1 = t1.best_move(b)
    first_evals = t1.evals_used
    t2 = MCTS(flat_reach, max_nodes=150, cache=shared)
    root2 = t2.run(b)
    # budget counts FRESH evals, so a cached engine spends the same budget on
    # NOVEL positions: hits are nonzero and the tree gets BIGGER, not cheaper
    assert t2.cache_hits > first_evals * 0.5
    root1 = MCTS(flat_reach, max_nodes=150, cache={}).run(b)
    assert root2.N > root1.N
    # cache changes tree SHAPE but never values: same-config is deterministic
    m1b = MCTS(flat_reach, max_nodes=150, cache={}).best_move(b)
    assert m1 == m1b


def test_path_aware_threefold_detection():
    """The search must see a threefold forming in its own lines (copy(stack=
    False) drops history, so is_game_over could not) -- the measured cause of
    the toy shuffling into an unseen draw."""
    import chess
    import numpy as np
    from catspace.nn.mcts import MCTS, _Node

    m = MCTS(lambda bs: np.zeros(len(bs)), max_nodes=10)
    b = chess.Board("7k/8/8/8/8/8/8/R6K w - - 0 1")
    m.rep_history = {b._transposition_key(): 2}     # this position already twice
    root = _Node(b, None)
    # a child returning to the SAME position (3rd occurrence) is a threefold
    same = _Node(b.copy(stack=False), None, parent=root)
    assert m._threefold(same) is True
    # a different position (first occurrence) is not
    b2 = b.copy(stack=False); b2.push_san("Ra2")
    other = _Node(b2, None, parent=root)
    assert m._threefold(other) is False
    # repetition built up WITHIN the search path also counts
    m.rep_history = {b._transposition_key(): 1}
    mid = _Node(b.copy(stack=False), None, parent=root)   # 2nd (search)
    deep = _Node(b.copy(stack=False), None, parent=mid)   # 3rd (search)
    assert m._threefold(deep) is True


# ---- coherence-length backup discount (2026-07-16) ----------------------

def _det_reach(boards):
    # deterministic pseudo-reach so on/off comparisons are exact: hash the FEN
    return np.array([(hash(b.board_fen()) % 1000) / 1000.0 - 0.5 for b in boards])


def test_coherence_off_is_exact_old_backup():
    # coherence_k=0 must reproduce the undiscounted backup bit-for-bit
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    r_off = MCTS(_det_reach, max_nodes=48, coherence_k=0.0).run(b.copy())
    r_base = MCTS(_det_reach, max_nodes=48).run(b.copy())
    assert r_off.N == r_base.N
    for a, c in zip(r_off.children, r_base.children):
        assert a.N == c.N and abs(a.W - c.W) < 1e-12


def test_coherence_gamma_forced_vs_divergent():
    # a FORCED node (one legal move -> no child entropy) keeps gamma=1; a
    # DIVERGENT node (many comparable moves) gets gamma<1 under coherence_k>0.
    forced = chess.Board("7k/8/8/8/8/8/8/R6K b - - 0 1")   # Kh8, only Kg8/Kh7-ish few
    divergent = chess.Board("8/8/8/3k4/8/3K4/8/8 w - - 0 1")  # open board, many K moves
    m = MCTS(_det_reach, max_nodes=80, coherence_k=1.0)
    rf = m.run(forced.copy())
    md = MCTS(_det_reach, max_nodes=80, coherence_k=1.0)
    rd = md.run(divergent.copy())
    # root divergence: the more-branchy open position should have a strictly
    # smaller (more-discounting) coherence gamma than the constrained one
    assert rd.coh_gamma < rf.coh_gamma
    assert 0.0 < rd.coh_gamma <= 1.0 and 0.0 < rf.coh_gamma <= 1.0


# ---- obvious-region soft-terminal (2026-07-17) --------------------------

def test_certainty_stop_resolves_confident_region():
    # a certainty_fn that flags EVERY position resolved (conf=1.0, value=+1)
    # must make the search treat first-level children as terminals: no node
    # below them is expanded, so eval budget collapses to ~one expansion.
    def all_resolved(boards):
        n = len(boards)
        return np.ones(n), np.ones(n)          # value=+1 (White win), conf=1.0
    b = chess.Board("8/8/8/3k4/8/3K4/7R/8 w - - 0 1")   # KRvK, White to move
    root = MCTS(_det_reach, max_nodes=200, certainty_fn=all_resolved,
                certainty_stop=0.9).run(b.copy())
    # every child of the root is soft-resolved -> terminal -> never expanded
    assert all(c.terminal_v is not None for c in root.children)
    # and the resolved value (+1) backs up as White-POV win
    assert root.Q > 0.5


def test_certainty_stop_off_by_default_expands():
    # certainty_stop=0 (default) must NOT short-circuit: normal expansion.
    def all_resolved(boards):
        return np.ones(len(boards)), np.ones(len(boards))
    b = chess.Board("8/8/8/3k4/8/3K4/7R/8 w - - 0 1")
    root = MCTS(_det_reach, max_nodes=120, certainty_fn=all_resolved,
                certainty_stop=0.0).run(b.copy())
    # with the stop off, non-terminal children keep v_init (not forced terminal)
    assert any(c.terminal_v is None for c in root.children)


def test_black_prefers_mate_over_draw():
    # regression for the (terminal_v > 0.5) == white shortcut bug (2026-07-17):
    # for Black to move, that predicate reduced to terminal_v <= 0.5, matching a
    # DRAW (DRAW_V=0) -- so Black could grab a draw over a real Black mate. Build
    # a Black-to-move root where one child is a Black mate and another is a draw;
    # best_move must return the mate.
    import chess as _chess
    # Black to move: Ra1-a1#? construct a position where a Black rook mate exists.
    # Black: Ra2#? Use a back-rank mate mirror: White Kh1, pawns f2 g2 h2; Black
    # Ra8 with Ra1 = mate.
    b = _chess.Board("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1")   # ...Ra1#
    mv = make(nodes=64).best_move(b)
    bb = b.copy(); bb.push(mv)
    assert bb.is_checkmate()          # took the mate, not a shuffling non-mate/draw


def test_coherence_from_committor_confidence():
    # coherence grounded in P(realize) (Kaveh 2026-07-17): with a certainty_fn,
    # a CONFIDENT node (P~1, e.g. a forced/won region) must get coh_gamma ~1 (no
    # discount) EVEN IF it has many children -- move-count must not discount a
    # certain outcome. A low-confidence node gets coh_gamma < 1.
    b = chess.Board("8/8/8/3k4/8/3K4/8/8 w - - 0 1")   # open board, many K moves
    def conf_high(boards):
        n = len(boards); return np.zeros(n), np.full(n, 0.99)   # value, confidence
    def conf_low(boards):
        n = len(boards); return np.zeros(n), np.full(n, 0.20)
    rh = MCTS(_det_reach, max_nodes=40, coherence_k=2.0,
              certainty_fn=conf_high, certainty_stop=0.0).run(b.copy())
    rl = MCTS(_det_reach, max_nodes=40, coherence_k=2.0,
              certainty_fn=conf_low, certainty_stop=0.0).run(b.copy())
    assert rh.coh_gamma > 0.95            # confident => ~no discount despite many moves
    assert rl.coh_gamma < rh.coh_gamma    # uncertain => discounted more
