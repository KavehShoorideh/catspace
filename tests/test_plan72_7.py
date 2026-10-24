"""plan 72.7 THE RUST BOUNDARY (docs/RUST_BOUNDARY.md). Step 0, the sharper guard: the read set of every claim beside the trace (risk 2) and the cross-checks of
risks 4 and 5 -- python-chess's legal-move order and SAN against the Rust `Pos` over the guard's case set and the corners (the full sample is
`scripts/trace_identity.py crosscheck`). Step 1, the handle and the word calls on it: every new `Pos` method against python-chess on the case set's positions
(walked) and on random games pushed move by move, the compiled words against the FEN readers they replace, the word cache's replay of the read set, the
Pos-taking siblings against their FEN shims, and the illegal inputs (a ValueError, never a panic)."""
import importlib.util
import random
from pathlib import Path

import chess
import pytest

import analyst_core as AC
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import selfmine as SM

SPEC = importlib.util.spec_from_file_location("trace_identity", Path(__file__).resolve().parents[1] / "scripts" / "trace_identity.py")
TI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TI)


def test_72_7_0_the_corners_are_legal_positions():
    """A corner python-chess calls invalid would be dropped by the walk without a word: each must be a legal position."""
    assert all(chess.Board(f).is_valid() for f in TI.CORNERS)


def test_72_7_0_legal_order_and_san_match_python_chess_on_the_guard_case_set():
    """Risks 4 and 5 over the guard's ten positions and the corners, each walked WALK plies: `Pos.legal_ucis()` is python-chess's list, `Pos.san` its names."""
    r = TI.crosscheck(TI.crosscheck_positions(cases_only=True))
    print(f"OBS 72.7 step 0 crosscheck (case set and corners, walk {TI.WALK}): positions {r['positions']}, moves {r['moves']}, SAN mismatches {len(r['san'])}, "
          f"order mismatches {len(r['order'])}")
    assert r["positions"] >= len(TI.CORNERS) and not r["unparsable"]
    assert r["san"] == [] and r["order"] == []


def test_72_7_0_the_corners_name_what_they_are_for():
    """The corners exercise the cases python-chess and shakmaty could disagree on: castling (with check), promotions (under-promotions, with capture), en passant (and
    a pinned one, not legal), disambiguation by file, by rank and by both, mate, an evasion by en passant."""
    names = {f: [AC.Pos(f).san(u) for u in AC.Pos(f).legal_ucis()] for f in TI.CORNERS}
    every = {s for v in names.values() for s in v}
    assert {"O-O", "O-O-O", "O-O+", "axb8=Q+", "axb8=N", "a8=R", "b8=N", "exd6", "cxd6", "N4d3", "Nbd3", "Qa4d4", "Ra8#", "Qxf7#", "Kxe2"} <= every
    assert names["8/8/8/K2pP2r/8/8/8/7k w - d6 0 2"] == ["Kb6", "Ka6", "Kb5", "Kb4", "Ka4", "e6"]          # exd6 e.p. would expose the king on the rank
    assert names["7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"] == []
    assert "exd3" in names["8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1"]
    order = AC.Pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").legal_ucis()
    assert order[-2:] == ["e1g1", "e1c1"] and order.index("e1d1") < order.index("a1a8")      # castling after every piece move (the a1 rook's after the king's), the king side first
    assert order == [m.uci() for m in chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").legal_moves]


def test_72_7_0_illegal_input_raises_value_error():
    """Risk 7: no panic across the border -- an illegal FEN and an illegal move are a ValueError."""
    with pytest.raises(ValueError):
        AC.Pos("8/8/8/8/8/8/8/8 w - - 0 1")
    p = AC.Pos(chess.STARTING_FEN)
    for bad in ("e2e5", "e7e5", "zz", "", "e1g1"):
        with pytest.raises(ValueError):
            p.san(bad)
        with pytest.raises(ValueError):
            p.push(bad)


def test_72_7_0_the_read_set_dump_is_stable_and_names_every_claim():
    """Risk 2: the snapshot's read sets -- sorted square names, the root questions with their claims' reads, every fact of the store."""
    assert TI._sq(None) == "None" and TI._sq(frozenset()) == "{}"
    assert TI._sq(frozenset({chess.H8, chess.A1, chess.E4})) == "a1 e4 h8"
    trace, reads = TI._run("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "w")
    assert "DECISION Ra8# 100" in trace
    lines = reads.splitlines()
    assert lines[0] == "Q1 mate verified reads None | claimed a1 f7 g7 h7 a8 b8 c8 d8 e8 f8 g8 h8"
    assert any(l.startswith("#2 claim w ") and "reads a1 f7 g7 h7 a8 b8 c8 d8 e8 f8 g8 h8" in l for l in lines)
    assert TI._run("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "w")[1] == reads                                        # the same run, the same dump


# ---------------------------------------------------------------------------------------------------------------- step 1: the handle of a node and the words on it

def _sample() -> list:
    """The case set's positions and the corners, each walked TI.WALK plies (the cross-check's FENs)."""
    return TI.crosscheck_positions(cases_only=True)


def _parts(b: chess.Board):
    ep = b.ep_square if b.ep_square is not None and b.has_legal_en_passant() else None
    return AC.Pos.from_parts(b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings, b.occupied_co[True], b.occupied_co[False], b.turn,
                             b.clean_castling_rights(), ep, b.halfmove_clock, b.fullmove_number)


def _games(n: int = 12, plies: int = 40, seed: int = 7):
    """Random games from the start, the handle PUSHED beside the board (raw en passant squares, castling rights lost by moves, promotions): (board, handle) per ply."""
    rng = random.Random(seed)
    for _ in range(n):
        b, p = chess.Board(), AC.Pos(chess.STARTING_FEN)
        for _ in range(plies):
            ms = list(b.legal_moves)
            if not ms:
                break
            m = rng.choice(ms)
            b.push(m)
            p = p.push(m.uci())
            yield b.copy(stack=False), p


def _pairs():
    """(board, handle): the sample's positions read from their FEN, and the random games' positions pushed."""
    for f in _sample():
        b = chess.Board(f)
        yield b, AC.Pos(b.fen())
    yield from _games()


def test_72_7_1_fen_round_trip():
    """The handle prints the FEN python-chess prints -- read from the FEN, built from the board's parts (graph.pos_of), and pushed move by move."""
    n = 0
    for f in _sample():
        b = chess.Board(f)
        p = AC.Pos(b.fen())
        assert p.fen() == p.fen_exact() == b.fen() == _parts(b).fen() == G.pos_of(b).fen() == G.exact_fen(b)
        n += 1
    for b, p in _games():
        assert p.fen() == b.fen() and p.board_fen() == b.board_fen()
        n += 1
    print(f"OBS 72.7 step 1 fen round trip: {n} positions, every handle prints python-chess's FEN")


def test_72_7_1_key_is_the_transposition_key():
    """`Pos.key` is python-chess's `_transposition_key` as a hash: equal for transpositions (the clocks and the move order apart), different for another side to move,
    other castling rights, a legal en passant square."""
    a, s = chess.Board(), chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
        a.push_san(san)
    assert a.fen() != s.fen() and AC.Pos(a.fen()).key() == AC.Pos(s.fen()).key()
    x, y = chess.Board(), chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6"):
        x.push_san(san)
    for san in ("Nf3", "Nc6", "e4", "e5"):
        y.push_san(san)
    assert x._transposition_key() == y._transposition_key() and AC.Pos(x.fen()).key() == AC.Pos(y.fen()).key()
    ep, no_ep = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2"), chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 2")
    assert AC.Pos(ep.fen()).key() != AC.Pos(no_ep.fen()).key()
    boards = [chess.Board(f) for f in _sample()]
    keys = [AC.Pos(b.fen()).key() for b in boards]
    tks = [b._transposition_key() for b in boards]
    agree = sum((keys[i] == keys[j]) == (tks[i] == tks[j]) for i in range(len(boards)) for j in range(i + 1, len(boards)))
    pairs = len(boards) * (len(boards) - 1) // 2
    print(f"OBS 72.7 step 1 key: {agree} of {pairs} position pairs agree with python-chess's transposition key")
    assert agree == pairs


def test_72_7_1_with_turn_clears_en_passant_as_graph_with_turn():
    """`Pos.with_turn` is `graph.with_turn` (that side to move, no en passant square, castling rights and clocks kept); None exactly where python-chess keeps a board that
    is not a legal position (the side not to move in check). `pass_turn` is `game.pass_`."""
    n = none = 0
    for b, p in _pairs():
        for colour in (True, False):
            w = G.with_turn(b, colour)
            q = p.with_turn(colour)
            if q is None:
                assert not w.is_valid()
                none += 1
            else:
                assert q.fen() == w.fen() and q.ep_square() is None
            n += 1
        q, w = p.pass_turn(), G.with_turn(b, not b.turn)
        assert (q is None and not w.is_valid()) or q.fen() == w.fen()
    b = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
    assert AC.Pos(b.fen()).with_turn(True).fen() == "4k3/8/8/3pP3/8/8/8/4K3 w - - 0 2" == G.with_turn(b, True).fen()
    print(f"OBS 72.7 step 1 with_turn: {n} boards given to a side, {none} not a legal position (None)")


def test_72_7_1_legal_order_and_san_on_pushed_handles():
    """Risks 4 and 5 on handles PUSHED from the start (raw en passant squares and lost castling rights included), not only read from a FEN."""
    for b, p in _games():
        assert p.legal_ucis() == [m.uci() for m in b.legal_moves]
        assert [p.san(m.uci()) for m in b.legal_moves] == [b.san(m) for m in b.legal_moves]


def test_72_7_1_squares_masks_and_terminals_match_python_chess():
    """`piece_at`, `attackers`, `attacks` (u64 masks), `king`, `ep_square`, `board_fen`, `is_checkmate`, `is_stalemate`, `is_check` on every square of every position."""
    n = 0
    for b, p in _pairs():
        for sq in chess.SQUARES:
            pc = b.piece_at(sq)
            assert p.piece_at(sq) == ((pc.piece_type, pc.color) if pc is not None else None)
            assert p.attackers(sq, True) == int(b.attackers_mask(chess.WHITE, sq)) and p.attackers(sq, False) == int(b.attackers_mask(chess.BLACK, sq))
            assert p.attacks(sq) == int(b.attacks_mask(sq))
        assert p.king(True) == b.king(chess.WHITE) and p.king(False) == b.king(chess.BLACK) and p.ep_square() == b.ep_square
        assert p.board_fen() == b.board_fen() and p.is_checkmate() == b.is_checkmate() and p.is_stalemate() == b.is_stalemate() and p.is_check() == b.is_check()
        n += 1
    print(f"OBS 72.7 step 1 squares: {n} positions x 64 squares agree with python-chess")


def test_72_7_1_is_capture_and_gives_check_match_python_chess():
    n = 0
    for b, p in _pairs():
        for m in b.legal_moves:
            assert p.is_capture(m.uci()) == b.is_capture(m) and p.gives_check(m.uci()) == b.gives_check(m)
            n += 1
    print(f"OBS 72.7 step 1 moves: {n} legal moves, is_capture and gives_check agree with python-chess")


TERMS = [("see", "e4"), ("attackers", "d5"), ("defenders", "e5"), ("count", ("and", "ours", "attacked")), "balance", "material_us", ("count", ("mobility", "W")),
         ("pushes", "e2"), ("dist", "L", ("promo", "e2")), "ours", ("after", "e2", "e4", ("see", "d5")), "movecount", ("least", "balance"), ("count", ("moves", "W"))]


def test_72_7_1_the_compiled_words_read_what_the_fen_readers_read():
    """`Pos.read` is `vocab_eval` (val_at), `Pos.values` is `vocab_values` (evaluate: a set term reads 0), `graph.word` on a board and on its handle read what the FEN path
    read (the board given to the owner first, e.p. cleared), `graph.see` / `Pos.see` the word `see` -- the same evaluator, no FEN and no s-expression per read."""
    BV.install_terms()
    G.clear_caches()
    n = 0
    for f in _sample():
        b = chess.Board(f)
        p, mover = AC.Pos(b.fen()), "w" if b.turn else "b"
        ids = [SM.term_id(t, mover) for t in TERMS]
        sx = [BV.sexpr(SM.render(t, mover)) for t in TERMS]
        assert [p.read(i) for i in ids] == [AC.vocab_eval(b.fen(), s) for s in sx] == [SM.val_at(b.fen(), t, mover) for t in TERMS]
        assert p.values(ids) == [AC.vocab_values([b.fen()], s)[0] for s in sx]
        ev = SM.evaluate(TERMS, [dict(fen=b.fen(), mover=mover), dict(pos=p, mover=mover)])
        assert all(ev[s][0] == ev[s][1] == v for s, v in zip([BV.sexpr(t) for t in TERMS], p.values(ids)))
        for owner in (True, False):
            w = b if (b.turn == owner and b.ep_square is None) else G.with_turn(b, owner)
            for t in TERMS[:9]:
                exp = AC.vocab_eval(w.fen(), BV.sexpr(SM.render(t, "w" if owner else "b")))
                assert G.word(b, t, owner) == exp == G.word(p, t, owner)
        for sq in chess.SQUARES:
            assert G.see(b, sq) == (AC.vocab_eval(b.fen(), BV.sexpr(SM.render(("see", chess.square_name(sq)), mover))) or 0) == p.see(sq) == G.see(p, sq)
        n += 1
    print(f"OBS 72.7 step 1 words: {n} positions x {len(TERMS)} terms, the compiled words read what vocab_eval / vocab_values read")


def test_72_7_1_the_word_cache_replays_the_read_set_and_keeps_no_history_word():
    """Risk 2: a hit replays what the word added to the dependency trackers on its miss (the audit records the occupied squares through the cache's own path). A term that
    reads the clocks or the game's line is never cached; a pure one is cached once per position read (the clocks are not in the key)."""
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    miss, hit, same, served = AC.word_cache_replay_audit(fen)
    assert miss == int(chess.Board(fen).occupied) and miss == hit and same and served
    AC.dep_clear()
    assert AC.dep_peek() == (0, False, [0, 0, 0], 0)
    assert AC.compiled_pure(AC.compile_term("(see e4)")) and AC.compiled_pure(AC.compile_term("(count (and ours attacked))"))
    for t in ("halfmoves", "seen", "(ago 1 (count ours))", "(delta balance)", "(count (gained ours))", "(count (lost ours))", "forcing_us_3"):
        assert not AC.compiled_pure(AC.compile_term(t)), t
    a, s = chess.Board(), chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
        a.push_san(san)
    G.clear_caches()
    hm, pure = AC.compile_term("halfmoves"), AC.compile_term("(count (and ours attacked))")
    assert (AC.Pos(a.fen()).read(hm), AC.Pos(s.fen()).read(hm)) == (4, 0)                  # read afresh: no stale clock from the transposition
    n0 = AC.word_cache_size()
    v1, v2 = AC.Pos(a.fen()).read(pure), AC.Pos(s.fen()).read(pure)
    assert v1 == v2 and AC.word_cache_size() == n0 + 1                                       # one entry for the transposition
    G.clear_caches()
    assert AC.word_cache_size() == 0


def _outcome(f, *args):
    """A reader's answer, or its ValueError (a position the inner function refuses is refused by both the sibling and the shim)."""
    try:
        return ("ok", f(*args))
    except ValueError as e:
        return ("ValueError", str(e))


def test_72_7_1_the_pos_siblings_read_what_their_fen_shims_read():
    """The Pos-taking siblings of the FEN readers the agenda calls: the same inner functions, the same answers (or the same refusal)."""
    n = 0
    for f in _sample():
        b = chess.Board(f)
        p, me = AC.Pos(b.fen()), b.turn
        ucis = [m.uci() for m in b.legal_moves]
        assert _outcome(AC.forcing_whys_pos, p, ucis, 3, me, True) == _outcome(AC.forcing_whys, b.fen(), ucis, 3, me, True)
        assert _outcome(AC.see_after_pos, p, ucis, chess.E4) == _outcome(AC.see_after, b.fen(), ucis, chess.E4)
        for u in ucis[:6]:
            assert _outcome(AC.second_moves_pos, p, u) == _outcome(AC.second_moves, b.fen(), u)
        ksq = b.king(not me)
        for sq in chess.SQUARES:
            pc = b.piece_at(sq)
            if pc is None or pc.color == me:
                continue
            assert _outcome(AC.material_defects_pos, p, me, sq) == _outcome(AC.material_defects, b.fen(), me, sq)
            if pc.piece_type != chess.KING:
                assert _outcome(AC.skewer_routes_pos, p, me, sq) == _outcome(AC.skewer_routes, b.fen(), me, sq)
                assert _outcome(AC.drive_onto_routes_pos, p, me, sq, 3) == _outcome(AC.drive_onto_routes, b.fen(), me, sq, 3)
                assert _outcome(AC.trap_fix_routes_pos, p, me, sq) == _outcome(AC.trap_fix_routes, b.fen(), me, sq)
                spec = ("win", "win", ("", None), [], [], sq, None, None, sq)
                quiet = [u for u in ucis if not b.is_castling(chess.Move.from_uci(u))]
                assert _outcome(AC.still_routes_pos, p, me, quiet, spec) == _outcome(AC.still_routes, b.fen(), me, quiet, spec)
        if ksq is not None:
            for s in chess.SquareSet(chess.BB_KING_ATTACKS[ksq]):
                if b.piece_at(s) is None:
                    pl = [("Q" if me else "q", s, None)]
                    assert _outcome(AC.mate_defects_pos, p, me, pl) == _outcome(AC.mate_defects, b.fen(), me, pl)
        n += 1
    print(f"OBS 72.7 step 1 siblings: {n} positions, every Pos sibling reads what its FEN shim reads")


def test_72_7_1_the_bridge_node_pushes_both_and_pos_of_is_cached():
    G.clear_caches()
    root = G.Node(chess.Board())
    leaf = root.child("e2e4").child("c7c5").child("e4e5").child("d7d5")
    assert leaf.board.fen() == leaf.pos.fen() == "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    assert leaf.pos.ep_square() == leaf.board.ep_square == chess.D6 and root.board.fen() == chess.STARTING_FEN
    b = chess.Board()
    assert G.pos_of(b) is G.pos_of(b) and G.pos_for(b, True) is G.pos_of(b)
    assert G.pos_for(b, False).fen() == G.with_turn(b, False).fen()
    G.clear_caches()
    assert G._POS == {}


def test_72_7_1_illegal_input_raises_value_error():
    """Risk 7: an illegal FEN, an illegal move, a square off the board, an unknown term id, inconsistent bitboards, an unreadable term -- each a ValueError."""
    p = AC.Pos(chess.STARTING_FEN)
    for bad in ("e2e5", "e1g1", "zz"):
        for f in (p.push, p.san, p.is_capture, p.gives_check):
            with pytest.raises(ValueError):
                f(bad)
    for f in (p.piece_at, p.attacks, p.see):
        with pytest.raises(ValueError):
            f(64)
    with pytest.raises(ValueError):
        p.attackers(64, True)
    for f in (lambda: p.word(10 ** 6, True), lambda: p.read(10 ** 6), lambda: p.values([0, 10 ** 6]), lambda: p.words([10 ** 6], False), lambda: AC.compiled_pure(10 ** 6)):
        with pytest.raises(ValueError):
            f()
    with pytest.raises(ValueError):
        AC.Pos.from_parts(1, 1, 0, 0, 0, 0, 1, 0, True, 0, None, 0, 1)                       # a pawn and a knight on a1
    with pytest.raises(ValueError):
        AC.Pos.from_parts(0, 0, 0, 0, 0, 1, 1, 0, True, 0, None, 0, 1)                       # no black king: not a position
    with pytest.raises(ValueError):
        AC.compile_term("(see")
    with pytest.raises(ValueError):
        AC.forcing_whys_pos(p, ["e2e5"], 3, True, False)
    for fen in ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1", "8/8/8/8/8/8/8/P3K2k w - - 0 1", "not a fen"):
        with pytest.raises(ValueError):
            AC.Pos(fen)
    assert SM.val_at("not a fen", "balance", "w") is None and G.word(chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1"), "balance", True) == 64
