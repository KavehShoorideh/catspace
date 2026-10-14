"""The surprise explorer's exact labels (goals.rs forced_moves) and its use of the mirror lemma (explorer.py)."""
import random

import chess
import chess.syzygy
import pytest

import catspace.incubator.analyst  # noqa: F401
import analyst_core as A
from catspace.io import paths
from catspace.incubator.analyst.discovery import explorer as E


def test_only_the_king_step_forces_the_promotion():
    lab = {u: (st, n) for u, st, n, _ in A.forced_moves("5k2/8/3KP3/8/8/8/8/8 w - - 2 2", 1, 12, 2_000_000)}
    assert lab["d6d7"] == ("forced", 3)                                # Kd7, e7, e8=Q
    assert all(st == "stopped" for u, (st, _) in lab.items() if u != "d6d7")


def _kpk(rng):
    while True:
        b = chess.Board(None)
        s = rng.sample(range(64), 3)
        if chess.square_rank(s[2]) in (0, 7):
            continue
        b.set_piece_at(s[0], chess.Piece(chess.KING, True)); b.set_piece_at(s[1], chess.Piece(chess.KING, False))
        b.set_piece_at(s[2], chess.Piece(chess.PAWN, True))
        if b.is_valid() and not b.is_game_over():
            return b


@pytest.mark.skipif(not paths.syzygy_dir().exists(), reason="no tablebases")
def test_forced_is_sound_against_the_tablebase():
    rng = random.Random(0)
    with chess.syzygy.open_tablebase(str(paths.syzygy_dir())) as tb:
        for _ in range(60):
            b = _kpk(rng)
            for u, st, n, _ in A.forced_moves(b.fen(), 1, 10, 2_000_000):
                x = b.copy(); x.push(chess.Move.from_uci(u))
                if st == "forced":
                    assert tb.probe_wdl(x) == -2, (b.fen(), u)       # a forced gain is a won position for us


def test_the_mirror_lemma_transfers_the_labels():
    rng = random.Random(1)
    for _ in range(20):
        b = _kpk(rng)
        lab = {u: (st, n) for u, st, n, _ in A.forced_moves(b.fen(), 1, 10, 2_000_000)}
        mir = {u: (st, n) for u, st, n, _ in A.forced_moves(E.mirror_fen(b.fen()), 1, 10, 2_000_000)}
        assert {u: x[:2] for u, x in E.mirror_labels({u: (st, n, []) for u, (st, n) in lab.items()}).items()} == mir


def test_the_defence_labels_name_the_moves_that_stop_their_pawn():
    # their a-pawn runs; our king must step into its square
    lab = {u: st for u, st, n, _ in A.defence_moves("6k1/6P1/7P/p7/4K3/8/8/8 w - - 4 14", 1, 6, 2_000_000)}
    assert lab["e4f4"] == "forced" and lab["e4d4"] == "stopped" and lab["e4d3"] == "stopped"


def test_the_tactic_profile_reads_both_sides():
    ours, theirs = A.tactic_profile("5k2/8/3KP3/8/8/8/8/8 w - - 2 2", 3, 200_000)
    assert ours[0] == (1, 3, 1, 0)                                     # a pawn's worth or more: in 3, by one first move (Kd7)
    assert all(c == 0 for _, _, c, _ in theirs)                        # their lone king forces nothing
