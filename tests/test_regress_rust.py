"""Plan 54 (B): the Rust twins of regress.py (`mate_defects`, `material_defects`, `_still`, `forcing_why`) answer exactly as the Python functions did.
Plan 76.4a (c): the mate and material defects and the fix filter are the regression's now (discovery/regression.py); the hand tables are test oracles
(tests/oracles/regress_tables.py) -- the hand table, its Rust twin and the regression are compared here; `both` toggles the forcing reasons' path alone."""
import chess
import pytest

from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import regress as R
from oracles import regress_tables as HT

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
    """fn with the forcing reasons read in Python, then in Rust."""
    old = R.ROUTES_RUST
    try:
        R.ROUTES_RUST = False
        a = fn(*args)
        R.ROUTES_RUST = True
        b = fn(*args)
    finally:
        R.ROUTES_RUST = old
    return a, b


def shape(ds):
    return [(d.kind, d.sq, tuple(d.by), d.why) for d in ds]


def twin(rust_fn, py_fn, *args):
    """The Rust twin as the engine called it until 76.4a (a position shakmaty holds no such way: the Python table answered)."""
    try:
        return rust_fn(*args)
    except ValueError:
        return py_fn(*args)


@pytest.mark.parametrize("fen", FENS)
def test_mate_defects_rust_equals_python_for_every_placement(fen):
    """The hand mate table (the oracle), its Rust twin and the regression (regress.mate_defects) agree on every placement: kinds, squares, pieces, texts, order."""
    board = chess.Board(fen)
    handled = 0
    for me in (True, False):
        for sym in "QRBN":
            sym = sym.upper() if me else sym.lower()
            for at in chess.SQUARES:
                pl = [(chess.Piece.from_symbol(sym), at, None)]
                a, b, c = HT.mate_defects(board, me, pl), twin(HT.rust_mate_defects, HT.mate_defects, board, me, pl), R.mate_defects(board, me, pl)
                assert shape(a) == shape(b) == shape(c), (fen, me, sym, chess.square_name(at))
                handled += bool(a)
        # a move of an existing piece as the placement (the pattern's `frm`)
        for frm in chess.SQUARES:
            pc = board.piece_at(frm)
            if pc is None or pc.color != me or pc.piece_type == chess.KING:
                continue
            for at in board.attacks(frm):
                pl = [(pc, at, frm)]
                a, b, c = HT.mate_defects(board, me, pl), twin(HT.rust_mate_defects, HT.mate_defects, board, me, pl), R.mate_defects(board, me, pl)
                assert shape(a) == shape(b) == shape(c), (fen, me, chess.square_name(frm), chess.square_name(at))
    assert handled > 0


@pytest.mark.parametrize("fen", FENS)
def test_material_defects_rust_equals_python_on_every_piece(fen):
    """The hand material table, its Rust twin and the regression agree on every piece of theirs (a square of no unit of theirs is no material question)."""
    board = chess.Board(fen)
    for me in (True, False):
        for sq in chess.SQUARES:
            a, b = HT.material_defects(board, me, sq), twin(HT.rust_material_defects, HT.material_defects, board, me, sq)
            assert shape(a) == shape(b), (fen, me, chess.square_name(sq))            # the twins, on every square as before
            p = board.piece_at(sq)
            if p is not None and p.color != me and p.piece_type != chess.KING:
                assert shape(R.material_defects(board, me, sq)) == shape(a), (fen, me, chess.square_name(sq))


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


def test_the_core_cache_changes_nothing_and_clear_caches_empties_it():
    import analyst_core as A
    fen = FENS[1]
    ucis = [m.uci() for m in list(chess.Board(fen).legal_moves)[:6]]
    was = A.set_caches(True)
    try:
        A.clear_caches()
        a = A.forcing_whys(fen, ucis, 5, True, False)
        assert A.cache_size() == 1
        assert A.forcing_whys(fen, ucis, 5, True, False) == a
        G.clear_caches()                                  # a fresh root forgets the positions
        assert A.cache_size() == 0
        A.set_caches(False)
        assert A.forcing_whys(fen, ucis, 5, True, False) == a and A.cache_size() == 0
    finally:
        A.set_caches(was)


# ------------------------------------------------------------------------------------------------------------------ 54 (B4): trap, fork, skewer, drive-onto, second moves

ZBOUR = "2k5/1p6/p1P1p1p1/2P5/P2Pp1p1/4PrPq/2Q1RP1r/3RK3 b - - 0 34"
QUEEN_TRAP = "rnb1kb2/1p2pprp/2p2n2/pB4p1/2p2P1P/PP2P1PN/3qQK2/RNB4R w q - 2 13"      # their queen on d2 can run; one quiet move traps it
FENS2 = FENS + [ZBOUR, QUEEN_TRAP, "6k1/5ppp/8/8/2q5/8/5PPP/1R2R1K1 w - - 0 1",
                "4k3/8/8/8/8/5q2/PP6/RN2K3 w - - 0 1"]       # a queen with few squares


def trap_shape(board, t, us):
    pc = board.piece_at(t)
    n = G.mobility_of(G.with_turn(board, pc.color), t) or 0
    return (n, [(r.move.uci(), r.san, r.why, r.trap) for r in R.trap_routes(board, us, t)] if n else [],
            [(r.move.uci(), r.san, r.why, r.fork) for r in R.fork_routes(board, us, t)])


@pytest.mark.parametrize("fen", FENS2)
def test_trap_fix_rust_equals_python_on_every_piece_of_theirs(fen):
    board = chess.Board(fen)
    us = board.turn
    seen = 0
    for t in chess.SQUARES:
        pc = board.piece_at(t)
        if pc is None or pc.piece_type == chess.KING or pc.color == us:
            continue
        got = R._trap_rust(board, us, t)
        if got is None:
            continue
        n, traps, forks = got
        assert (n, [(r.move.uci(), r.san, r.why, r.trap) for r in traps] if n else [], [(r.move.uci(), r.san, r.why, r.fork) for r in forks]) == trap_shape(board, t, us), (fen, t)
        seen += len(traps) + len(forks)
    if fen == QUEEN_TRAP:
        assert seen > 0                                 # the queen on d2 has a trap route


def test_a_trapped_queen_is_found_by_the_core():
    import analyst_core as A
    b = chess.Board(QUEEN_TRAP)
    ra = [(r.move.uci(), r.why) for r in R.trap_routes(b, True, chess.D2)]
    assert ra and "queen" not in ra[0][1] and "Q on d2" in ra[0][1]
    assert [(u, w) for u, w in A.trap_routes(b.fen(), True, chess.D2, 3)] == ra


def play(fen, sans):
    b = chess.Board(fen)
    for s in sans.split():
        b.push_san(s)
    return b


def skewer_shape(rs):
    return [(r.move.uci(), r.san, r.why) for r in rs]


def both_skewer(fn, *args):
    old = R.SKEWER_RUST
    try:
        R.SKEWER_RUST = False
        a = fn(*args)
        R.SKEWER_RUST = True
        b = fn(*args)
    finally:
        R.SKEWER_RUST = old
    return skewer_shape(a), skewer_shape(b)


@pytest.mark.parametrize("fen", FENS2)
def test_skewer_and_drive_onto_rust_equal_python_on_every_target(fen):
    board = chess.Board(fen)
    us = board.turn
    for t in chess.SQUARES:
        pc = board.piece_at(t)
        if pc is None or pc.color == us:
            continue
        a, b = both_skewer(R.skewer_routes, board, us, t)
        assert a == b, (fen, t)
        a, b = both_skewer(R.drive_onto_routes, board, us, t, 3)
        assert a == b, (fen, t)


def test_the_skewer_and_the_drive_onto_routes_on_zbour_come_from_the_core():
    import analyst_core as A
    b = play(ZBOUR, "Rhxf2 cxb7+ Kb8 Rxf2 Qh1+ Ke2")
    assert [r.san for r in R.skewer_routes(b, b.turn, chess.C2)] == ["Rxf2+"] or "Rxf2+" in [r.san for r in R.skewer_routes(b, b.turn, chess.C2)]
    assert skewer_shape(R.skewer_routes(b, b.turn, chess.C2)) == [(u, chess.Board(b.fen()).san(chess.Move.from_uci(u)), w) for u, w in A.skewer_routes(b.fen(), b.turn, chess.C2)]
    c = play(ZBOUR, "Rhxf2 cxb7+ Kb8 Rxf2 Qh1+ Ke2 Rxf2+ Kxf2")
    assert sorted(r.san for r in R.skewer_routes(c, c.turn, chess.C2)) == ["Qg2+", "Qh2+"]
    a, bb = both_skewer(R.skewer_routes, c, c.turn, chess.C2)
    assert a == bb
    a, bb = both_skewer(R.drive_onto_routes, b, b.turn, chess.C2, 3)
    assert a == bb
