"""The chess domain's relations and placements: each relation on squares whose truth is known by the rules."""
import chess
import pytest

from catspace.incubator.analyst import chess_domain as cd


def st(fen, ids):
    return cd.State(chess.Board(fen), list(ids))


def test_relations_on_known_squares():
    b = "4k3/7K/8/8/7b/8/8/2r4R w - - 0 1"
    s = st(b, [chess.C1, chess.H1, chess.H4])
    assert cd.attacks(s, chess.C1, chess.H1) and cd.attacks(s, chess.H1, chess.C1)
    assert cd.attacks(s, chess.H4, chess.H1) is False               # a bishop does not attack along the file
    assert cd.aligned(s, chess.C1, chess.H1) and not cd.aligned(s, chess.H4, chess.C1)
    assert cd.attacked(s, chess.H1) and cd.attacked(s, chess.C1) and not cd.defended(s, chess.H1)
    assert cd.same_side(s, chess.C1, chess.H4) and not cd.same_side(s, chess.C1, chess.H1)
    assert cd.worth_ge(s, chess.H1, chess.H4) and not cd.worth_ge(s, chess.H4, chess.H1)
    s2 = st("4k3/8/8/8/8/8/8/Q2R2rK w - - 0 1", [chess.A1, chess.D1, chess.G1])
    assert cd.between(s2, chess.A1, chess.D1, chess.G1) and not cd.between(s2, chess.D1, chess.A1, chess.G1)


def test_action_relations():
    b = chess.Board("4k3/7K/8/8/7b/8/8/2r4R w - - 0 1")
    s = cd.State(b, [chess.C1, chess.H1, chess.H4])
    acts = {a[1].uci(): a for a in cd.actions(s, chess.H1)}
    assert cd.lands_attacked(s, acts["h1g1"]) and not cd.lands_attacked(s, acts["h1h2"])
    assert cd.stays_aligned(s, acts["h1g1"], chess.C1) and not cd.stays_aligned(s, acts["h1h2"], chess.C1)
    assert not cd.stays_aligned(s, acts["h1c1"], chess.C1)          # landing on the piece's square is not staying aligned with it
    assert cd.captures(s, acts["h1c1"]) and cd.captures_worth_ge(s, acts["h1c1"], chess.H4) and not cd.captures(s, acts["h1h2"])
    assert not cd.gives_check(s, acts["h1h2"])


def test_reply_scalar_and_null_and_resolved():
    b = chess.Board("4k3/7K/8/8/7b/8/8/2r4R w - - 0 1")
    s = cd.State(b, [chess.C1, chess.H1, chess.H4])
    acts = {a[1].uci(): a for a in cd.actions(s, chess.H1)}
    before = cd.scalar(s, chess.H1)
    assert cd.reply_scalar(cd.transition(s, acts["h1g1"]), chess.G1) - before == -5.0    # the rook is taken
    assert cd.reply_scalar(cd.transition(s, acts["h1c1"]), chess.C1) - before == 5.0     # the rook takes the rook
    assert cd.null_reply_scalar(s, chess.H1) - before == -5.0                            # passing loses the rook too
    dom = cd.domain(oracle="resolved")
    b2 = chess.Board("4k3/pppp4/5n2/8/8/1BN5/PPPP4/4K3 w - - 0 1")
    s2 = cd.State(b2, [chess.C3]); a = (b2, chess.Move.from_uci("c3d5"))
    assert dom.reply_scalar(dom.transition(s2, a), chess.D5) - dom.scalar(s2, chess.C3) == 0.0   # a trade, resolved


def test_placements_are_legal_and_typed():
    n = 0
    for state, exhaustive in cd.placements(("wQ", "wR", "bB"), cap=50, seed=0):
        b = state.board
        assert b.is_valid() and not b.is_check()
        assert b.piece_at(state.ids[0]) == chess.Piece(chess.QUEEN, True) and b.piece_at(state.ids[2]) == chess.Piece(chess.BISHOP, False)
        assert len(b.piece_map()) == 5
        n += 1
    assert n == 50
    assert list(cd.placements(("wK", "wK"), cap=10, seed=0)) == []          # two kings a side: no placements


def test_game_placements_keep_the_board():
    fens = ["r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"]
    pl = cd.game_placements(fens)
    out = list(pl(("wN", "bN"), cap=10, seed=0))
    assert len(out) == 4                                                  # two white knights x two black knights (b1, f3) x (b8, c6)
    assert all(len(s.board.piece_map()) == 32 for s, _ in out)


def test_entities_and_actor_ok():
    dom = cd.domain()
    b = chess.Board("4k3/7K/8/8/7b/8/8/2r4R w - - 0 1")
    ents = dict(dom.entities(b))
    assert ents[chess.H1] == "wR" and ents[chess.H4] == "bB" and ents[chess.E8] == "bK"
    assert dom.actor_ok(b, chess.H1) and not dom.actor_ok(b, chess.C1)


def test_frame_local_outcome_counts_only_the_frame():
    """The frame-local oracle: only the frame's pieces count; a capture elsewhere on the board is invisible; a capture
    inside the frame is counted once (no double count when two ids land on one square)."""
    dom = cd.domain(oracle="frame")
    b = chess.Board("r1bq1rk1/ppp2ppp/2n2n2/3pp3/1b1P4/2NBPN2/PPP2PPP/R2QK2R w KQ - 0 8")
    s = cd.State(b, [chess.C3, chess.B4])                                   # frame: the white knight c3, the black bishop b4 that attacks it
    base = dom.scalar(s, chess.C3)
    assert base == 0.0                                                      # knight 3 minus bishop 3
    null = dom.null_reply_scalar(s, chess.C3) - base
    assert null == -3.0                                                     # passing: Bxc3 takes the knight; the bishop is counted once
    a = (b, chess.Move.from_uci("c3e2")); after = dom.transition(s, a)
    assert dom.reply_scalar(after, after.ids[0]) - base == 0.0              # the knight steps away: the frame keeps its material
    a = (b, chess.Move.from_uci("c3d5")); after = dom.transition(s, a)
    assert dom.reply_scalar(after, after.ids[0]) - base == -3.0             # Nd5 exd5: the knight is lost; the pawn that took it is not in the frame and not counted
