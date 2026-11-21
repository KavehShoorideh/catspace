"""PLAN 79.22 THE WAKE READS LAYER 1'S TAGS AND SHARES ONE ROOT GRAPH (SYSTEM_SPEC 11 the layers, one graph, the history; piece: representation and its use by the wake).
Rust holds the tests of the kept position (cargo test --release --offline -p cs_decision --test keep_layer2: the root graph and the tags equal a fresh build and the rules' facts at every decision of whole games of both fixture games) and of layer 1's
keys met (rust/engine/rep/tests/history.rs). THIS file holds what belongs to Python: the number of times the root graph is built in one decision, through the binding (the guard of the decisions themselves is tests/test_plan79_19.py).
"""
from __future__ import annotations

import pytest

from catspace.incubator.analyst.discovery import whole_79_19_print as P
from catspace.incubator.analyst.discovery import whole_79_22 as W22


@pytest.fixture(scope="module")
def world():
    from catspace.incubator.analyst.discovery import whole_79_14 as W14
    try:
        return W14.world()
    except FileNotFoundError as e:                                                # the untracked inputs of the harness (a worktree without them)
        pytest.skip(str(e))


def _roots(lines: list) -> list:
    """the root graphs built, from the OBS lines of one position or decision"""
    return [int(l.split("the root's graph built ")[1].split(" times")[0]) for l in lines if " position " in l or " ply " in l]


def test_the_root_graph_is_built_once_a_decision_and_not_at_all_in_a_game_at_hand_after_the_first(world):
    no_goal = W22.builds(world, P.POSITIONS, "no_goal")
    assert _roots(no_goal) == [1] * len(P.POSITIONS), no_goal
    every = W22.builds(world, P.POSITIONS, "all_inputs")
    assert _roots(every) == [1] * 6, every
    played = W22.builds(world, P.POSITIONS, "played")
    roots = _roots(played)
    assert roots[0] == 1 and roots[1:] == [0] * (len(roots) - 1) and len(roots) >= 5, played      # the first decision asks the rules for the graph; the game's memory then keeps it and the moves played change it
    for l in no_goal + every + played:
        print(l)
