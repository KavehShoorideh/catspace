"""The tactic census (catspace/incubator/analyst/tactic_census.py): tokens on positions whose properties are asserted
first, the null-move reading of the opponent's tactics, the Lean barrier mapping, structure features, and the rows of a
short game."""
from types import SimpleNamespace

import chess

from catspace.incubator.analyst import pin
from catspace.incubator.analyst import tactic_census as C


def _mates(b, uci):
    m = chess.Move.from_uci(uci)
    if m not in b.legal_moves:
        return False
    nb = b.copy(stack=False)
    nb.push(m)
    return nb.is_checkmate()


def test_a_back_rank_mate_is_on_for_us_and_a_threat_for_them():
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    assert _mates(b, "d1d8")                                        # the position's own property first
    assert ("back_rank", 0, "") in C.tokens(b)
    b2 = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1")     # the same, Black to move: White's threat
    nb = b2.copy(stack=False); nb.push(chess.Move.null())
    assert _mates(nb, "d1d8") and not b2.is_check()
    assert ("back_rank", 0, "") in C.opponent_tokens(b2)
    assert ("back_rank", 0, "") not in C.tokens(b2)                 # Black has no back-rank mate of its own


def test_a_pin_that_wins_is_on():
    b = chess.Board("r1b1k2r/pppp1ppp/6n1/6q1/3QP3/3B4/PPP2PPP/RN3RK1 b kq - 2 10")
    assert b.turn == chess.BLACK and pin.pins(b, chess.BLACK) and pin.strikes(b)
    assert ("pin", 0, "") in C.tokens(b)


def test_nothing_is_on_at_the_start():
    toks = C.tokens(chess.Board())
    assert not [t for t in toks if t[1] == 0]
    assert all(t[0] in C.FAMILIES for t in toks)


def test_lean_barrier_mapping():
    I = lambda status, failing=None, hyps=None: SimpleNamespace(status=status, failing=failing, hyps=hyps or {})
    assert C._lean_tokens([], "back_rank") == []
    assert C._lean_tokens([I("FAIL", hyps={"a": True, "check": False, "legal": False})], "back_rank") == [("back_rank", 2, "check+legal")]
    assert C._lean_tokens([I("FAIL", hyps={"check": False, "legal": False}), I("ONEAWAY", "noDefender")], "back_rank") == [("back_rank", 1, "noDefender")]
    assert C._lean_tokens([I("ONEAWAY", "x"), I("PROVED")], "passer") == [("passer", 0, "")]


def test_structure_features():
    b = chess.Board("8/8/8/8/8/8/PPP2PPP/4K3 w - - 0 1")
    s = C.structure(b, chess.WHITE)
    assert s == dict(open_files=2, half_open=0, passed=6, isolated=0, islands=2, shelter=1)
    b2 = chess.Board("4k3/3p4/8/8/8/8/P3P3/4K3 w - - 0 1")          # a2 isolated, e2 alone; d7 against e2
    s2 = C.structure(b2, chess.WHITE)
    assert s2["isolated"] == 2 and s2["islands"] == 2 and s2["passed"] == 1 and s2["half_open"] == 1


def test_rows_of_a_short_game():
    row = dict(game_id="7", white_elo="1600", black_elo="1650", result="1", moves="e2e4 e7e5 f1c4 b8c6 d1h5 g8f6 h5f7")
    rows = C._game_rows(row, engine=None)
    assert len(rows) == 7                                           # the final mate is game over: no row
    assert [r["side"] for r in rows] == ["w", "b"] * 3 + ["w"]
    assert rows[0]["result"] == 1 and rows[1]["result"] == -1
    assert rows[1]["elo"] == 1650 and rows[1]["elo_opp"] == 1600
    assert rows[6]["fen"] == "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    assert ("material_now", 0, "") in rows[6]["stm"] or ("double_threat", 0, "") in rows[6]["stm"]   # Qxf7 is on


def test_reconcile_keeps_a_rows_file_that_has_no_done_list(tmp_path):
    """Review 2026-10-07: a rows file from before checkpointing (no .done beside it) was reconciled against an empty list
    and every row dropped."""
    import json
    from catspace.incubator.analyst.tactic_census import reconcile
    out = tmp_path / "census_x.jsonl"
    out.write_text("".join(json.dumps(dict(game=g, ply=p)) + "\n" for g in (7, 8) for p in (1, 2)))
    done = reconcile(out, out.with_suffix(".done"))
    assert done == {"7", "8"} and sum(1 for _ in open(out)) == 4
    out.write_text(out.read_text() + json.dumps(dict(game=9, ply=1)) + "\n")   # a game cut off before its mark
    assert reconcile(out, out.with_suffix(".done")) == {"7", "8"} and sum(1 for _ in open(out)) == 4
