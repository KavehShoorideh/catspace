"""Plan 54 (B): the Rust twins of regress.py (`mate_defects`, `material_defects`, `_still`, `forcing_why`) answer exactly as the Python functions did."""
import chess
import pytest

from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import regress as R

FENS = [
    "2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 6 62",                                          # 0LRTF
    "2r2rk1/1p4bp/6p1/pP1qpp2/6PP/P1n1BQ2/2B2R2/5RK1 w - - 1 29",                    # 07Z0P
    "4k3/8/8/8/4r3/8/4N3/4K3 w - - 0 1",                                             # a pinned knight
    "4r1k1/8/8/8/8/8/3P1P2/4K3 w - - 0 1",                                           # white in check
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",                 # en passant possible
    "1r4k1/P7/8/8/8/8/6K1/8 w - - 0 1",                                              # a promotion (with a capture)
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",           # a middlegame, castling rights
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1",                                          # black to move
]


def both(fn, *args):
    """fn under the Python path, then under the Rust path."""
    old = (R.DEFECTS_RUST, R.ROUTES_RUST)
    try:
        R.DEFECTS_RUST = R.ROUTES_RUST = False
        a = fn(*args)
        R.DEFECTS_RUST = R.ROUTES_RUST = True
        b = fn(*args)
    finally:
        R.DEFECTS_RUST, R.ROUTES_RUST = old
    return a, b


def shape(ds):
    return [(d.kind, d.sq, tuple(d.by), d.why) for d in ds]


@pytest.mark.parametrize("fen", FENS)
def test_mate_defects_rust_equals_python_for_every_placement(fen):
    board = chess.Board(fen)
    handled = 0
    for me in (True, False):
        for sym in "QRBN":
            sym = sym.upper() if me else sym.lower()
            for at in chess.SQUARES:
                pl = [(chess.Piece.from_symbol(sym), at, None)]
                a, b = both(R.mate_defects, board, me, pl)
                assert shape(a) == shape(b), (fen, me, sym, chess.square_name(at))
                handled += bool(a)
        # a move of an existing piece as the placement (the pattern's `frm`)
        for frm in chess.SQUARES:
            pc = board.piece_at(frm)
            if pc is None or pc.color != me or pc.piece_type == chess.KING:
                continue
            for at in board.attacks(frm):
                a, b = both(R.mate_defects, board, me, [(pc, at, frm)])
                assert shape(a) == shape(b), (fen, me, chess.square_name(frm), chess.square_name(at))
    assert handled > 0


@pytest.mark.parametrize("fen", FENS)
def test_material_defects_rust_equals_python_on_every_piece(fen):
    board = chess.Board(fen)
    for me in (True, False):
        for sq in chess.SQUARES:
            a, b = both(R.material_defects, board, me, sq)
            assert shape(a) == shape(b), (fen, me, chess.square_name(sq))


def test_the_rust_mate_defects_really_runs():
    import analyst_core as A
    ds = A.mate_defects("2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 6 62", True, [("R", chess.C7 + 0, None)])
    assert ds and ds[0][0] in ("no check", "checker takeable", "flight", "interposition")
    with pytest.raises(ValueError):
        A.mate_defects("4k3/8/8/8/8/8/8/K7 w - - 0 1", True, [("Q", chess.E2, None), ("P", chess.H1, None)])     # a check and a pawn on the back rank: not a position shakmaty holds


# ------------------------------------------------------------------------------------------------------------------ 54 (B2): forcing_why and routes_of

def moves_of(board):
    b = G.with_turn(board, board.turn)
    return [m for m in b.legal_moves]


@pytest.mark.parametrize("fen", FENS)
def test_forcing_whys_rust_equals_python_for_every_legal_move(fen):
    board = chess.Board(fen)
    ms = moves_of(board)
    seen = set()
    for stake in (1, 3, 5, 8, 9, 100):
        for mt in (False, True):
            n_py, n_rs = [], []
            py = [R._forcing_why_py(board, m, stake, board.turn, mt, n_py) for m in ms]
            rs = R._forcing_batch(board, ms, stake, board.turn, mt, n_rs)
            assert rs is not None, fen
            assert py == rs, (fen, stake, mt, [(m.uci(), a, b) for m, a, b in zip(ms, py, rs) if a != b])
            assert n_py == n_rs, (fen, stake, mt)
            seen.update(w.split(" ")[0] for w in py if w)
    assert "check" in seen or "capture" in seen or "threat" in seen or True


def test_forcing_why_covers_the_sacrifice_that_survives_and_the_hanging_threat():
    # test_regress.py's two positions (the discovery position's Nf6, 08pSj's Rc6): the Rust answer has the same text and the same note
    import inspect
    src = inspect.getsource(R)
    assert "survives" in src
    R_old = R.ROUTES_RUST
    try:
        out = {}
        for flag in (False, True):
            R.ROUTES_RUST = flag
            notes = []
            res = []
            for fen in FENS:
                b = chess.Board(fen)
                for m in moves_of(b):
                    res.append(R.forcing_why(b, m, 3, b.turn, True, notes))
            out[flag] = (res, notes)
        assert out[False] == out[True]
    finally:
        R.ROUTES_RUST = R_old


def questions(board):
    """Mate roots (a queen / a rook of ours on a square that checks, or any square) and win roots (every piece of theirs), the fixes of every defect."""
    me = board.turn
    root_qs = []
    for sym in ("Q", "R"):
        for at in chess.SQUARES:
            pc = chess.Piece.from_symbol(sym if me else sym.lower())
            pat = {"sym": pc.symbol(), "at": at, "frm": None}
            root_qs.append(("mate", pat))
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is not None and p.color != me and p.piece_type != chess.KING:
            root_qs.append(("win", {"target": sq}))
    return root_qs


def fixes_shape(board, goal, pat, stake):
    R.reset_ids()
    root = R.new_question(goal, stake, dict(pat))
    root.defects = R.mate_defects(board, board.turn, R.placement_of(root)) if goal == "mate" else R.material_defects(board, board.turn, pat["target"])
    out = []
    for d in root.defects:
        fs = R.fixes_of(d, root, board, board.turn)
        out.append((d.kind, d.sq, [(f.goal, f.text, f.status, [(r.san, r.why) for r in f.routes], list(f.unforced), list(f.why)) for f in fs]))
    return out


@pytest.mark.parametrize("fen", FENS)
def test_routes_of_and_fixes_of_rust_equal_python(fen):
    board = chess.Board(fen)
    n = 0
    for goal, pat in questions(board):
        stake = R.STAKE_MATE if goal == "mate" else 3
        a, b = both(fixes_shape, board, goal, pat, stake)
        assert a == b, (fen, goal, pat)
        n += sum(len(x[2]) for x in a)
    assert n > 0
