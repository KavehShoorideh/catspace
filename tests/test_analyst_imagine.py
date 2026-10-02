"""Imagined frames as goals (catspace/incubator/analyst/imagine.py)."""
import chess

from catspace.incubator.analyst.fates import clear_caches
from catspace.incubator.analyst.imagine import goals


def test_the_fork_is_the_first_imagined_goal():
    clear_caches()
    b = chess.Board("4k3/8/r3q3/8/8/3N4/8/7K w - - 0 1")             # Nc5 attacks the rook on a6 and the queen on e6 at once
    ms, squares = goals(b, 3)
    assert ms[0] == "d3c5"                                            # a double attack ranks first
    assert {chess.A6, chess.E6} <= squares                           # its targets become the search's goal squares
