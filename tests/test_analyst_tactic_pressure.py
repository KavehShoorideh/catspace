"""The pressure measurement's king-shelter features (catspace/incubator/analyst/tactic_pressure.py), on constructed
positions whose properties are asserted first."""
import importlib
import chess


def _king_feats():
    # the module runs its measurement at import; take the function from its source without running it
    import ast, types
    src = open(importlib.util.find_spec("catspace.incubator.analyst.tactic_pressure").origin).read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "king_feats")
    ns = {"chess": chess}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "king_feats", "exec"), ns)
    return ns["king_feats"]


def test_castled_king_with_f_pawn_home_and_moved():
    king_feats = _king_feats()
    b = chess.Board("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 6")
    assert b.king(chess.WHITE) == chess.G1 and b.piece_at(chess.F2) == chess.Piece(chess.PAWN, chess.WHITE)
    x = king_feats(b, chess.WHITE, "us")
    assert x["us_castled_k"] == 1 and x["us_fpawn_home"] == 1 and x["us_shelter_holes"] == 0 and x["us_shelter_moved"] == 0
    b2 = chess.Board("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1PP2/2N2N2/PPPP2PP/R1BQ1RK1 b - - 0 6")      # f2-f4 played
    assert b2.piece_at(chess.F2) is None and b2.piece_at(chess.F4) == chess.Piece(chess.PAWN, chess.WHITE)
    x2 = king_feats(b2, chess.WHITE, "us")
    assert x2["us_fpawn_home"] == 0 and x2["us_shelter_moved"] == 1 and x2["us_shelter_holes"] == 0
    b3 = chess.Board("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP2PP/R1BQ1RK1 w - - 0 6")      # no f-pawn at all
    x3 = king_feats(b3, chess.WHITE, "us")
    assert x3["us_fpawn_home"] == 0 and x3["us_shelter_holes"] == 1 and x3["us_king_files_open"] == 1


def test_black_mirror_and_exposure():
    king_feats = _king_feats()
    b = chess.Board("r1bq1rk1/pppp2pp/2n2n2/2b1pp2/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 7")        # Black played ...f5
    assert b.king(chess.BLACK) == chess.G8 and b.piece_at(chess.F7) is None
    x = king_feats(b, chess.BLACK, "them")
    assert x["them_castled_k"] == 1 and x["them_fpawn_home"] == 0 and x["them_shelter_moved"] == 1
    b2 = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    x2 = king_feats(b2, chess.BLACK, "them")
    assert x2["them_attackers_near"] == 0 and x2["them_defenders_near"] == 0 and x2["them_shelter_holes"] == 0
    b3 = chess.Board("6k1/5ppp/6N1/8/8/8/5PPP/3R2K1 w - - 0 1")                                   # a knight on g6
    assert chess.square_distance(chess.G6, chess.G8) == 2
    assert king_feats(b3, chess.BLACK, "them")["them_attackers_near"] == 1
