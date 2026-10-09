import chess


def test_npe0_the_queen_is_chased_into_the_corner():
    """Lichess NPpe0: the king is skipped (not enough pieces there), the queen on g8 is the target, Bb7 clears the rank
    for the rook and is certified; Ba6 and Bd7 fail to Qxa8 (the rook hangs)."""
    from catspace.incubator.analyst.target_reasoner import reason
    b = chess.Board("r1b3Q1/p3kp1p/1p2p1p1/4q3/8/2n1B2P/P4PP1/R3KB1R w KQ - 0 20"); b.push_uci("a1c1")
    r = reason(b)
    assert r["move"] == b.parse_san("Bb7") and r["target"] == "queen g8" and r["gain"] >= 3
    assert any("why not Ba6: Qxa8" in t for t in r["trace"]) and any("skip" in t for t in r["trace"] if t.startswith("king"))
