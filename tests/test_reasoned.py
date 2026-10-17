"""ANALYST_PLAN 51: the reasoned search (our moves from the reasoning, their replies all)."""
import chess

from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import reasoned as RS
from catspace.incubator.analyst.discovery import ruler as RU


def _puzzle(pid):
    return next(p for p in PA.load_set("dev_20") if p["id"] == pid)


def test_a_mate_in_one_is_the_only_recommendation():
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    moves, reasons = RS.recommend(b, "w", memory=False)
    assert [b.san(m) for m in moves] == ["Ra8#"] and reasons[-1].startswith("mate:")


def test_in_check_only_the_king_saving_moves_and_no_cut():
    b = chess.Board("7k/8/8/8/8/5n2/3n4/4K3 w - - 0 1")          # check from the knight on f3, guarded by the other knight: only king moves
    assert b.is_check()
    moves, reasons = RS.recommend(b, "w", memory=False)
    assert set(moves) == set(b.legal_moves) and all(b.piece_type_at(m.from_square) == chess.KING for m in moves)
    assert reasons[-1].startswith("in check:")


def test_a_hanging_queen_every_recommended_move_parries_checks_or_takes_nine():
    b = chess.Board("4k3/8/4p3/3Q4/8/8/8/4K3 w - - 0 1")           # the pawn on e6 attacks the unguarded queen
    moves, reasons = RS.recommend(b, "w", memory=False)
    assert reasons[-1].startswith("threat:") and moves
    for m in moves:                                                  # each: moves the queen (a parry), takes the attacker, gives check, or takes >= 9
        assert (m.from_square == chess.D5 or m.to_square == chess.E6 or b.gives_check(m)
                or (b.is_capture(m) and b.piece_type_at(m.to_square) == chess.QUEEN))
    assert chess.Move.from_uci("e1d1") not in moves and chess.Move.from_uci("e1f2") not in moves   # a king shuffle parries nothing
    assert chess.Move.from_uci("d5e6") in moves                      # taking the attacker


def test_a_quiet_position_only_captures_checks_promotions_or_threats():
    b = chess.Board("4k3/pppp4/8/8/8/8/PPPP4/4K2R w - - 0 1")
    moves, reasons = RS.recommend(b, "w", memory=False)
    assert moves and reasons[-1].startswith("observables:")
    assert chess.Move.from_uci("a2a3") not in moves and chess.Move.from_uci("h1h8") in moves
    assert all(b.is_capture(m) or b.gives_check(m) or m.promotion or RS._creates_hanging(b, m, "w") for m in moves)


def test_ours_restricts_the_nodes_and_never_lies():
    for pid in ("000Sa", "01IFU"):
        p = _puzzle(pid)
        full = RU.solve(p["fen"], int(p["value"]), 6, 0)
        red = RS.solve(p["fen"], int(p["value"]), 6, 0, memory=False)
        assert red["nodes"] <= full["nodes"]
        assert red["status"] == "forced" and red["line"][0] == full["line"][0]
        print(f"OBS reasoned {pid}: ruler nodes {full['nodes']}, reasoned nodes {red['nodes']}, cuts {red['by_reason']}")


def test_their_replies_are_all(monkeypatch):
    """The hook is never asked at a node where the other side moves."""
    seen = []
    p = _puzzle("01IFU")
    root = chess.Board(p["fen"]).turn

    def ours(b):
        seen.append(b.turn)
        return None
    RU.solve(p["fen"], int(p["value"]), 3, 0, ours=ours)
    assert seen and set(seen) == {root}
