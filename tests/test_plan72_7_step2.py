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


# ---------------------------------------------------------------------------------------------------------------- 2b: the static leaf's record from Rust

from catspace.incubator.analyst.discovery import base_vocabulary as BV          # noqa: E402
from catspace.incubator.analyst.discovery import puzzle_apply as PA             # noqa: E402
from catspace.incubator.analyst.discovery import selfmine as SM                 # noqa: E402


def _leaf_python_chess(board: chess.Board, uci: str, us: str, fam: str) -> tuple:
    """`Agenda._leaf_read`'s decomposition as it read before 72.7 step 2b (agenda.py at 015c1a33: python-chess boards pushed with their handles, the words by
    `SM.evaluate`): -> (the replies' ucis, `inner1` after each, {reply index: (top, [(uci, SAN, (after f t balance)) of the tied moves, natural first])})."""
    body = SM.tail_body(us, fam, "1000")
    inner1, inner2 = body[1], body[1][1]
    route = G.Node(board, G.handle(board)).child(uci)
    reps = list(route.board.legal_moves)
    kids = [route.child(rep.uci()) for rep in reps]
    v1 = SM.evaluate([inner1], [dict(pos=k.pos, mover=us) for k in kids])[BV.sexpr(inner1)] if kids else []
    per = {}
    for i, k in enumerate(kids):
        d = k.board
        leaves = [m for m in d.legal_moves if not (m.promotion or d.is_castling(m))]
        terms = [("after", chess.square_name(m.from_square), chess.square_name(m.to_square), inner2) for m in leaves]
        vals = SM.evaluate(terms, [dict(pos=k.pos, mover=us)]) if terms else {}
        v2 = [vals[BV.sexpr(t)][0] for t in terms]
        top = max(v2) if v2 else None
        tied = sorted((m for m, w in zip(leaves, v2) if w == top), key=lambda m: (not d.is_capture(m), not d.gives_check(m), m.uci()))
        bterms = [("after", chess.square_name(m.from_square), chess.square_name(m.to_square), "balance") for m in tied]
        bvals = SM.evaluate(bterms, [dict(pos=k.pos, mover=us)]) if bterms else {}
        per[i] = (top, [(m.uci(), d.san(m), bvals[BV.sexpr(t)][0]) for m, t in zip(tied, bterms)])
    return [r.uci() for r in reps], list(v1), per


def test_72_7_2b_the_leaf_record_is_the_python_leaf_decomposition():
    """`Pos.leaf_replies` / `Pos.leaf_tied` against the python-chess decomposition they replace, every family the leaf reads (puzzle_apply.ORDER), both colours,
    on the guard's positions and random games: the replies in python-chess's order, `inner1` after each, and after EVERY reply the top of `(after f t inner2)`, the
    tied moves in the natural order with their SAN, `(after f t balance)` and the board after them."""
    BV.install_terms()
    rng = random.Random(2722)
    n = tied_n = 0
    for b in _games(10, 24, seed=27222)[::3]:
        us = "w" if b.turn else "b"
        P0 = G.handle(b)
        for mv in rng.sample(list(b.legal_moves), min(2, b.legal_moves.count())):
            for fam in PA.ORDER:
                body = SM.tail_body(us, fam, "1000")
                i1, i2 = SM.term_id(body[1], us), SM.term_id(body[1][1], us)
                want_r, want_v1, want = _leaf_python_chess(b, mv.uci(), us, fam)
                rus, v1, kids = P0.leaf_replies(mv.uci(), i1)
                assert (rus, v1) == (want_r, want_v1)
                for i, k in enumerate(kids):
                    top, tied, reads = k.leaf_tied(i2, us == "w")
                    assert (top, [(u, s, bal) for u, s, bal, _p in tied]) == want[i]
                    assert reads == 0
                    for u, _s, _bal, P3 in tied:
                        c = chess.Board(k.fen())
                        c.push(chess.Move.from_uci(u))
                        assert P3.fen() == c.fen()
                    tied_n += len(tied)
                n += 1
    print(f"OBS 72.7 2b leaf record against python-chess: route reads {n}, tied leaves {tied_n}, all equal")
    assert n > 60 and tied_n > 200


def test_72_7_2b_the_leaf_record_refuses_illegal_input_and_records_no_read():
    """Risk 7: an illegal route move or an unknown term id is a ValueError. Risk 2: the leaf's reads add nothing to the dependency trackers (no vocabulary word records
    them) -- what was there stays, nothing is added."""
    BV.install_terms()
    P0 = AC.Pos(POS2)
    i1 = SM.term_id(SM.tail_body("w", "delayed_balance", "1000")[1], "w")
    with pytest.raises(ValueError):
        P0.leaf_replies("e2e5", i1)
    with pytest.raises(ValueError):
        P0.leaf_replies("h3g5", 10 ** 9)
    with pytest.raises(ValueError):
        AC.Pos(chess.STARTING_FEN).leaf_tied(10 ** 9, True)
    AC.dep_clear()
    before = AC.dep_peek()
    _rus, _v1, kids = P0.leaf_replies("h3g5", i1)
    i2 = SM.term_id(SM.tail_body("w", "delayed_balance", "1000")[1][1], "w")
    assert all(k.leaf_tied(i2, True)[2] == 0 for k in kids)
    assert AC.dep_peek() == before


def test_72_7_2b_the_guard_records_the_leaf_verdicts():
    """The guard's third file: every static leaf read of a run as it is computed (qCXKp reads four), the method restored after the run."""
    p = next(x for x in PA.load_set("rand3_10") if x["id"] == "qCXKp")
    us = "w" if chess.Board(p["fen"]).turn else "b"
    orig = Agenda._leaf_read
    leaf: list = []
    TI._run(p["fen"], us, leaf)
    assert Agenda._leaf_read is orig
    assert len(leaf) >= 1 and all(" -> " in x and "| obs [" in x for x in leaf)
