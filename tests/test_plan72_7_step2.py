"""plan 72.7 THE RUST BOUNDARY, step 2 (docs/RUST_BOUNDARY.md section 4 step 2): the chase's reply loop and the static leaf on the handle.
2a THE RUST BOARD (`rboard.RBoard`, RUST_BOUNDARY 3.3): every read of the subset against python-chess on the guard's positions walked and on seeded random games
(the legal moves in python-chess's ORDER, SAN, attackers / attacks, push and its FEN, check / capture / castling tests, `in` with python-chess's castling
aliases), its immutability, illegal input (a ValueError, never a panic: risk 7), the bridge's shadow (risk 9), and the loop's sites read on it against the
python-chess code they replace (the pass board, the principal captures)."""
import importlib.util
import random
from pathlib import Path

import chess
import pytest

import analyst_core as AC
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import rboard as RB
from catspace.incubator.analyst.discovery import words as WD
from catspace.incubator.analyst.discovery.agenda import Agenda, _pos_key
from catspace.incubator.analyst.discovery.state import State

SPEC = importlib.util.spec_from_file_location("trace_identity", Path(__file__).resolve().parents[1] / "scripts" / "trace_identity.py")
TI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TI)
W = WD.CHESS_WORDS
POS2 = TI.FENS["pos2"][0]
DISCOVERY = TI.FENS["discovery"][0]


def _games(n: int = 12, plies: int = 40, seed: int = 7272) -> list:
    """Seeded random games from the start and from the guard's positions, every position on them (python-chess boards with their move stacks)."""
    rng = random.Random(seed)
    roots = [chess.STARTING_FEN, POS2, DISCOVERY] + list(TI.CORNERS)
    out = []
    for k in range(n):
        b = chess.Board(roots[k % len(roots)])
        for _ in range(plies):
            out.append(b.copy())
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
    return out


def _same(rb: RB.RBoard, b: chess.Board) -> None:
    """`rb` built from `b`'s FEN or parts: every read equal. (FINDING of 2a: the e.p. square follows the handle -- a handle read from a FEN or from a board's parts holds
    only a LEGAL e.p. square, as the FEN prints it; a handle pushed holds python-chess's pushed square, legal or not: see the push test. The loop and the leaf read none.)"""
    assert rb.turn == b.turn and rb.fen() == b.fen() and rb.board_fen() == b.board_fen() and rb.ep_square == (b.ep_square if b.has_legal_en_passant() else None)
    assert [m.uci() for m in rb.legal_moves] == [m.uci() for m in b.legal_moves]              # python-chess's ORDER (risk 4)
    assert rb.legal_moves.count() == b.legal_moves.count() and bool(rb.legal_moves) == bool(b.legal_moves) and len(rb.legal_moves) == b.legal_moves.count()
    assert (rb.is_check(), rb.is_checkmate(), rb.is_stalemate()) == (b.is_check(), b.is_checkmate(), b.is_stalemate())
    assert [rb.king(c) for c in chess.COLORS] == [b.king(c) for c in chess.COLORS]
    for s in chess.SQUARES:
        assert rb.piece_at(s) == b.piece_at(s) and rb.attacks(s) == b.attacks(s) and rb.attacks_mask(s) == b.attacks_mask(s)
        for c in chess.COLORS:
            assert rb.attackers(c, s) == b.attackers(c, s) and rb.is_attacked_by(c, s) == b.is_attacked_by(c, s)
    for m in b.legal_moves:
        assert rb.san(m) == b.san(m) and rb.is_capture(m) == b.is_capture(m) and rb.gives_check(m) == b.gives_check(m) and rb.is_castling(m) == b.is_castling(m)
        assert m in rb.legal_moves


def test_72_7_2a_rboard_reads_what_python_chess_reads_on_random_games():
    boards = _games()
    for b in boards:
        _same(RB.RBoard(AC.Pos(b.fen())), b)
    print(f"OBS 72.7 2a RBoard against python-chess: positions {len(boards)}, every read of the subset equal")


def test_72_7_2a_push_is_a_new_board_and_its_fen_is_python_chess_s_push():
    """`push` returns a NEW RBoard (the handle is immutable, `copy` is the board itself); the FEN after it is python-chess's after `push`, move by move through a game."""
    rng = random.Random(27)
    b = chess.Board()
    rb = RB.RBoard(AC.Pos(b.fen()))
    assert rb.copy() is rb and rb.copy(stack=False) is rb
    for _ in range(60):
        ms = list(b.legal_moves)
        if not ms:
            break
        m = rng.choice(ms)
        before = rb.fen()
        nxt = rb.push(m)
        assert rb.fen() == before and nxt is not rb                                             # the board it was pushed from is unchanged
        b.push(m)
        assert nxt.fen() == b.fen() and nxt.ep_square == b.ep_square
        rb = nxt


def test_72_7_2a_in_reads_python_chess_is_legal_with_its_castling_aliases():
    """`move in board.legal_moves` is python-chess's `is_legal`: e1h1 / e1a1 are the castlings e1g1 / e1c1 (`_from_chess960`); a king stepping onto its own rook from
    another square, a promotion without its piece, a null move are not legal."""
    for fen in ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "r3k2r/8/8/8/8/8/8/R4RK1 w kq - 0 1", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"):
        b, rb = chess.Board(fen), RB.RBoard(AC.Pos(fen))
        for frm in chess.SQUARES:
            for to in chess.SQUARES:
                for promo in (None, chess.QUEEN, chess.KNIGHT):
                    m = chess.Move(frm, to, promo)
                    assert (m in rb.legal_moves) == (m in b.legal_moves), (fen, m.uci())
        assert (chess.Move.null() in rb.legal_moves) is False
    rb = RB.RBoard(AC.Pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"))
    assert chess.Move.from_uci("e1h1") in rb.legal_moves and chess.Move.from_uci("e1a1") in rb.legal_moves


def test_72_7_2a_illegal_input_is_a_value_error():
    """Risk 7: an illegal move pushed or named, an illegal FEN -- a ValueError, never a panic and never a corrupt board (`play_unchecked` only on legal moves)."""
    rb = RB.RBoard(AC.Pos(chess.STARTING_FEN))
    for bad in ("e2e5", "e7e5", "e1g1", "a1a1"):
        with pytest.raises(ValueError):
            rb.push(chess.Move.from_uci(bad))
        with pytest.raises(ValueError):
            rb.san(chess.Move.from_uci(bad))
        with pytest.raises(ValueError):
            rb.gives_check(chess.Move.from_uci(bad))
    with pytest.raises(ValueError):
        rb.piece_at(64)
    with pytest.raises(ValueError):
        RB.RBoard(AC.Pos("8/8/8/8/8/8/8/8 w - - 0 1"))
    assert rb.fen() == chess.STARTING_FEN


def test_72_7_2a_the_bridge_shadow_catches_a_split(monkeypatch):
    """Risk 9: with CATSPACE_BRIDGE_CHECK every RBoard read is compared with its python-chess shadow; a shadow that is another position fails the read."""
    monkeypatch.setattr(G, "BRIDGE_CHECK", True)
    b = chess.Board(POS2)
    rb = RB.RBoard.of(b)
    _same(rb, b)
    n0 = RB.BRIDGE_READS[0]
    rb.push(next(iter(b.legal_moves))).legal_moves.count()
    assert RB.BRIDGE_READS[0] > n0
    wrong = RB.RBoard(AC.Pos(POS2), chess.Board(POS2))
    wrong._shadow.push(next(iter(wrong._shadow.legal_moves)))                                   # the shadow moved, the handle did not
    with pytest.raises(AssertionError):
        wrong.legal_moves.count()


def test_72_7_2a_the_pass_board_on_the_handle_is_python_chess_s_pass():
    """`Agenda._pass_board` (their pass, read on the handle) prints the FEN, the turn and the legal moves of graph.with_turn(board, them) -- the board their agenda is built
    from -- and falls back to python-chess where the pass is not a legal position (we are in check)."""
    ag = Agenda(State(fen=POS2, us="w", pieces=[]))
    n = 0
    for b in _games(6, 30):
        if b.turn != ag.me:
            continue
        p = AC.Pos(b.fen())
        got, want = ag._pass_board(b.fen(), p), G.with_turn(chess.Board(b.fen()), not ag.me)
        assert got.fen() == want.fen() and got.turn == want.turn and [m.uci() for m in got.legal_moves] == [m.uci() for m in want.legal_moves]
        assert isinstance(got, RB.RBoard) == (not b.is_check())
        assert ag._pass_key(b.fen(), p) == _pos_key(want.fen())
        n += 1
    assert n > 20


def _principal_captures_python_chess(P2) -> list:
    """`Agenda._principal_captures` as it read python-chess boards before 72.7 step 2a (agenda.py at 015c1a33), the reference the RBoard version must equal."""
    b = chess.Board(P2.fen())
    me = b.turn
    best, sqs = 0, []
    for x in chess.SQUARES:
        p = b.piece_at(x)
        if p is None or p.color == me or p.piece_type == chess.KING or not b.is_attacked_by(me, x):
            continue
        v = W.base.see(P2, x)
        if v > best:
            best, sqs = v, [x]
        elif v == best and v > 0:
            sqs.append(x)
    lines = []
    for sq in sqs:
        c, gains, sans, boards = b.copy(stack=False), [], [], []
        while len(sans) < 12:
            caps = [m for m in c.legal_moves if m.to_square == sq]
            if not caps or c.piece_at(sq) is None:
                break
            m = min(caps, key=lambda m: (W.base.worth(c.piece_at(m.from_square)), m.uci()))
            gains.append(W.base.worth(c.piece_at(sq)))
            sans.append(c.san(m))
            c.push(m)
            boards.append(c.copy(stack=False))
        n = len(sans)
        f = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            f[i] = max(0, gains[i] - f[i + 1])
        taken = 0
        while taken < n and gains[taken] - f[taken + 1] > 0:
            taken += 1
        if taken:
            lines.append((sq, [(sans[i], boards[i], i == taken - 1, f[i + 1]) for i in range(taken) if i % 2 == 0]))
    return lines


def test_72_7_2a_the_principal_captures_on_the_handle_are_the_python_chess_lines():
    """64.0's principal lines read on RBoards: the same squares, SANs, ends, credits and boards (by FEN) as the python-chess code they replace, over random games."""
    ag = Agenda(State(fen=POS2, us="w", pieces=[]))
    n = 0
    for b in _games(8, 40):
        p = AC.Pos(b.fen())
        got, want = ag._principal_captures(p), _principal_captures_python_chess(p)
        assert [(sq, [(s, bd.fen(), last, cr) for s, bd, last, cr in caps]) for sq, caps in got] == \
               [(sq, [(s, bd.fen(), last, cr) for s, bd, last, cr in caps]) for sq, caps in want]
        n += bool(want)
    print(f"OBS 72.7 2a principal captures: {n} positions with a line, RBoard lines equal to python-chess's")
    assert n > 10
