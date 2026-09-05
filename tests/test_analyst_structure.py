"""L3.5 structure facts: pawn structure, files, outposts, bishop pair, king shelter, scope; diffed per move."""
import chess

from catspace.incubator.analyst.structure import structure, structure_diff, structure_changes, top_changes
from catspace.incubator.analyst.dynamics import move_effects


def facts(fen):
    return {(r.name, r.names(), r.owner, r.tags) for r in structure(chess.Board(fen))}


def test_pawn_structure_facts():
    # White: a2 isolated; c3+c5 doubled; d4 not passed (c7, e6 ahead on adjacent files). Black: a7 and c7 isolated.
    f = facts("1r1r2k1/p1p2ppp/b3pb2/2Pq4/2PP4/8/P2NNPPP/R2Q1RK1 b - - 0 16")
    assert ("isolated_pawn", ("a2",), chess.WHITE, ()) in f
    assert ("doubled_pawns", (), chess.WHITE, ("c",)) in f
    assert ("isolated_pawn", ("c7",), chess.BLACK, ()) in f
    assert not [x for x in f if x[0] == "passed_pawn"]
    # a passed pawn with two ranks to go, and a backward pawn: d6 cannot be supported (c- and e-pawns gone) and d5 is
    # controlled by the white pawn on e4
    f = facts("6k1/8/3p4/1P6/4P3/8/8/6K1 w - - 0 1")
    assert ("passed_pawn", ("b5",), chess.WHITE, ("3_to_go",)) in f
    assert ("backward_pawn", ("d6",), chess.BLACK, ()) in f


def test_files_pieces_king():
    # open b-file for the b8 rook, half-open d-file for d8; Black has the bishop pair (a6 light, f6 dark)
    f = facts("1r1r2k1/p1p2ppp/b3pb2/2Pq4/2PP4/8/P2NNPPP/R2Q1RK1 b - - 0 16")
    assert ("rook_on_open_file", ("b8",), chess.BLACK, ("b",)) in f
    assert ("rook_on_half_open_file", ("d8",), chess.BLACK, ("d",)) in f
    assert ("bishop_pair", (), chess.BLACK, ()) in f and not [x for x in f if x[0] == "bishop_pair" and x[2] == chess.WHITE]
    # outpost: a white knight on d5 held by e4, black has no c- or e-pawn that could ever attack d5
    f = facts("r4rk1/pp3ppp/3p4/3N4/4P3/8/PPP2PPP/R4RK1 w - - 0 1")
    assert ("outpost", ("d5", "e4"), chess.WHITE, ()) in f
    # not an outpost if a black e-pawn on e7 could come to e6
    f = facts("r4rk1/pp2pppp/3p4/3N4/4P3/8/PPP2PPP/R4RK1 w - - 0 1")
    assert not [x for x in f if x[0] == "outpost"]
    # king shelter only counts once castling rights are gone: castled white king with the g-file open
    f = facts("6k1/pppppppp/8/8/8/8/PPPPP2P/5RK1 w - - 0 1")
    assert ("king_open_file", (), chess.WHITE, ("g",)) in f and ("king_open_file", (), chess.WHITE, ("f",)) in f
    f = facts("4k3/pppppppp/8/8/8/8/PPPPP2P/R3K2R w KQ - 0 1")
    assert not [x for x in f if x[0] == "king_open_file" and x[2] == chess.WHITE]
    # the starting position has no bad bishops (nothing is blocked) and both sides have the pair
    f = facts(chess.STARTING_FEN)
    assert not [x for x in f if x[0] == "bad_bishop"] and len([x for x in f if x[0] == "bishop_pair"]) == 2


def test_structure_diff_phrases():
    # an even trade judged after the recapture: Bxf6 Bxf6 costs White the bishop pair (AUjG3sl4, 7.Bxf6)
    b = chess.Board("rnbqk2r/ppp1bppp/4pn2/3p2B1/3P4/3B1N2/PPP1PPPP/RN1QK2R w KQkq - 0 5")
    c, e = structure_diff(b, ["g5f6", "e7f6"])
    assert "White gives up the bishop pair" in structure_changes(c, e)
    # a quiet pawn move that opens a line for a piece that stayed put (12...e5 instead of Qd5)
    b = chess.Board("1rbq1rk1/p1p2ppp/2p1pb2/8/1P1P4/2P5/P3NPPP/RN1QK2R b KQ - 0 12")
    eff = move_effects(b, b.parse_san("e5"))
    assert top_changes(eff.structure_created, eff.structure_ended, k=1) == ["opens a line for the black bishop on c8 (+3 squares)"]
    # a passed pawn advancing is one push, not a loss and a gain
    b = chess.Board("6k1/8/8/1P6/8/8/8/6K1 w - - 0 1")
    eff = move_effects(b, b.parse_san("b6"))
    assert structure_changes(eff.structure_created, eff.structure_ended) == ["pushes the passed pawn to b6"]
    # the diff is deterministic across calls
    assert structure_changes(eff.structure_created, eff.structure_ended) == structure_changes(eff.structure_created, eff.structure_ended)
