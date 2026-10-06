"""Rule 1, the double threat (tactic_rules): the knight fork wins the rook, a fork the opponent can capture claims
nothing, the claim is a sound lower bound on the exact 3-ply value wherever it fires (the bridge), and the hook's
'at least' mode lifts the move to the claim."""
import chess
import pytest

from catspace.incubator.analyst import frame_player as FP
from catspace.incubator.analyst.tactic_rules import bridge, double_threats, exact_3ply, fork_claims, local_claim


def test_knight_fork_wins_the_rook():
    b = chess.Board("4k3/8/4r3/5N2/8/8/8/6K1 w - - 0 1")                        # Ng7+ forks Ke8 and Re6; Kd7/Ke7/Kf7 then recapture
    c = fork_claims(b)
    assert "f5g7" in c and c["f5g7"][0] == pytest.approx(2.0)                   # Nxe6 Kxe6: the knight for the rook, +2 at worst
    assert exact_3ply(b, chess.Move.from_uci("f5g7")) >= 2.0
    assert [m.uci() for m, _, _ in double_threats(b) if m.uci() == "f5g7"]


def test_a_fork_the_opponent_can_take_claims_nothing():
    b = chess.Board("4k2b/8/4r3/5N2/8/8/8/6K1 w - - 0 1")                       # the same, but Bh8 takes the knight on g7
    assert "f5g7" not in fork_claims(b)


def test_claims_are_sound_lower_bounds_on_dev_positions():
    pytest.importorskip("pyarrow")
    import json
    from catspace.io import paths
    from pathlib import Path
    f = Path(paths.derived("frame_puzzles_lichess_one_ply_off5_poc1.json"))
    if not f.exists():
        pytest.skip("no development dump")
    fens = [s["fen"] for r in json.load(open(f)) for s in r["steps"]][:200]
    r = bridge(fens)
    assert r["positions"] == len(fens) and r["wrong"] == 0, r["examples"]
    assert r["claims"] > 20


def test_at_least_mode_lifts_to_the_claim():
    old = list(FP.RULES)
    try:
        FP.RULES[:] = [FP.Rule("lift", "proved", lambda b: {"e2e4": (3.0, "a bounded outcome"), "d2d4": (-1.0, "below the ledger")}, "at_least")]
        from catspace.incubator.analyst.fates import clear_caches
        clear_caches()
        c = FP.choose(chess.Board())
        assert c["values"]["e2e4"] == 3.0 and c["best"] == ["e2e4"] and c["values"]["d2d4"] == 0.0   # never lowered
        assert "d2d4" not in c["rules"]
    finally:
        FP.RULES[:] = old
