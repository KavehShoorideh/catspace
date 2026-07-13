"""planner/decompose.py on synthetic arc geometry.

Embeddings are unit vectors on a circle, so reach(a, b) = cos(angle between
them): monotonically decreasing in arc distance. A far pair with waypoints
strung along the arc MUST split near the middle (that's the geodesic-midpoint
property), and each give-up rule fires under the geometry built to demand it.
"""
import numpy as np
import pytest

from catspace.planner.decompose import (Decomposition, WaypointPool, decompose,
                                        hop_reach, waypoint_scores)


def unit(theta):
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)


def arc_pool(thetas):
    v = unit(np.asarray(thetas))
    return WaypointPool(F=v, B=v, labels=[f"m{i}" for i in range(len(v))])


S, G = 0.0, 2.0                      # arc endpoints (radians)


def test_waypoint_scores_peak_at_arc_middle():
    thetas = np.linspace(0.1, 1.9, 19)
    pool = arc_pool(thetas)
    scores = waypoint_scores(unit(S), unit(G), pool)
    assert thetas[np.argmax(scores)] == pytest.approx(1.0, abs=0.11)


def test_direct_hop_executable_no_split():
    dec = decompose(unit(0.0), unit(0.3), arc_pool([0.15]),
                    tau_exec=0.9, tau_floor=0.0)
    assert dec.executable and dec.waypoints == []
    assert dec.root.status == "executable"
    assert dec.root.reach == pytest.approx(np.cos(0.3))


def test_far_hop_decomposes_and_bottleneck_improves():
    pool = arc_pool(np.linspace(0.1, 1.9, 19))
    direct = hop_reach(unit(S), unit(G))
    dec = decompose(unit(S), unit(G), pool, tau_exec=np.cos(0.35), tau_floor=0.0)
    assert dec.executable
    assert dec.plan_bottleneck > direct
    assert dec.plan_bottleneck >= np.cos(0.35)          # every hop executable
    # waypoints come back in play order: monotone along the arc
    idx = dec.waypoints
    assert idx == sorted(idx)
    assert len(dec.subgoal_labels()) == len(idx)


def test_no_midpoint_when_pool_cannot_help():
    # every candidate sits BEHIND s: routing through any of them can't beat
    # the direct hop -- the hop is hard, not long
    pool = arc_pool([-0.5, -1.0, -1.5])
    dec = decompose(unit(S), unit(G), pool, tau_exec=0.9, tau_floor=-1.0)
    assert not dec.executable
    assert dec.block_rule == "no_midpoint"


def test_unlikely_territory_floor():
    # a single mediocre midpoint improves the bottleneck but stays below the floor
    pool = arc_pool([1.0])
    dec = decompose(unit(S), unit(G), pool, tau_exec=0.99, tau_floor=0.9)
    assert not dec.executable
    assert dec.block_rule == "unlikely_territory"


def test_budget_cap():
    pool = arc_pool(np.linspace(0.1, 1.9, 19))
    dec = decompose(unit(S), unit(G), pool, tau_exec=0.9999, tau_floor=0.0,
                    max_depth=1)
    assert not dec.executable
    assert dec.block_rule == "budget"
    assert all(leaf.depth <= 1 for leaf in dec.root.leaves())


def test_dry_out_on_vanishing_gains():
    # two waypoints barely off the endpoints: each split improves the
    # bottleneck by nearly nothing, twice in a row -> dry_out
    pool = arc_pool([0.02, 1.98])
    dec = decompose(unit(S), unit(G), pool, tau_exec=0.9999, tau_floor=-1.0,
                    min_gain=0.0, dry_gain=0.05, max_depth=6)
    assert not dec.executable
    assert dec.block_rule in ("dry_out", "no_midpoint")


def test_waypoints_not_reused():
    pool = arc_pool(np.linspace(0.1, 1.9, 19))
    dec = decompose(unit(S), unit(G), pool, tau_exec=np.cos(0.2), tau_floor=0.0,
                    max_depth=4)
    assert len(dec.waypoints) == len(set(dec.waypoints))


def test_anytime_tree_is_reported_on_block():
    pool = arc_pool(np.linspace(0.1, 1.9, 19))
    dec = decompose(unit(S), unit(G), pool, tau_exec=1.1, tau_floor=0.0,
                    max_depth=2)                        # nothing can be executable
    assert isinstance(dec, Decomposition)
    assert not dec.executable
    assert dec.plan_bottleneck > hop_reach(unit(S), unit(G))   # still improved


def test_custom_score_pairs_is_used_throughout():
    """A shifted scorer (dot - 0.5) must shift every reported reach/bottleneck
    -- proving hop_reach, waypoint_scores, and decompose all route through
    score_pairs rather than falling back to raw dots anywhere (the
    2026-07-12 quasimetric-calibration fix)."""
    def shifted(F, B):
        return F @ B.T - 0.5

    pool = arc_pool(np.linspace(0.1, 1.9, 19))
    assert hop_reach(unit(S), unit(G), shifted) == pytest.approx(np.cos(2.0) - 0.5)
    np.testing.assert_allclose(waypoint_scores(unit(S), unit(G), pool, shifted),
                               waypoint_scores(unit(S), unit(G), pool) - 0.5, atol=1e-6)

    # identical geometry, shifted thresholds -> identical tree shape
    base = decompose(unit(S), unit(G), pool, tau_exec=np.cos(0.35), tau_floor=0.0)
    shift = decompose(unit(S), unit(G), pool, tau_exec=np.cos(0.35) - 0.5,
                      tau_floor=-0.5, score_pairs=shifted)
    assert shift.executable == base.executable
    assert shift.waypoints == base.waypoints
    assert shift.plan_bottleneck == pytest.approx(base.plan_bottleneck - 0.5, abs=1e-6)
