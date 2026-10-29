"""PLAN 76.4b THE REGRESSION IN RUST (docs/ANALYST_PLAN.md 76.4b): rules/chess.gdl loaded into the native core from its text (rust/analyst_core/src/gdl.rs: no
clause written in Rust), the binding's readers chosen by the ontology shelf's table, the regression (regression.rs) mirroring regression.py -- whose Python reading,
`PyRegressor`, is the test oracle -- and the python-chess board mirrored (pyboard.rs). Equal to the oracle on fixed positions (a negated atom read by its witnesses, a
recursive ray body read through its table), the cache replaying its read set, illegal input refused with a ValueError, the handle's `Pos.regress` / `Pos.fixes`."""
from __future__ import annotations

import random
import re
from pathlib import Path

import chess
import pytest

import analyst_core as A
from catspace.incubator.analyst.discovery import gdl_chess as GC
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery import regression as RG

B = GC.Binding()
SPEC = RG.Spec(B.prog, certified=GC.CERTIFIED)
NAMES = LS.defect_names()
GOALS = LS.gdl_goals()
UNIT = LS.goal_kinds().valuation.unit
ROOT = Path(__file__).resolve().parents[1]

DISCOVERY = "r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1"
POS2 = "r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1"
XRAY = "2r5/pp1k1p2/1b1Np1p1/8/4P1pq/8/PP2QP2/3R1RK1 w - - 1 21"
CORNERS = [DISCOVERY, POS2, XRAY, "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "8/8/8/K2pP2r/8/8/8/7k w - d6 0 2", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
           "4k3/8/8/2Pp4/8/8/8/4K3 w - d6 0 2", "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/3n4/R6K w - - 0 1", "4k3/8/8/8/8/8/3PPP2/R3K2R w KQ - 0 1"]


def _oracle() -> RG.PyRegressor:
    return RG.PyRegressor(SPEC, B)


def _native() -> RG.Regressor:
    return RG.Regressor(SPEC, B)


def _rset(r: RG.Regression) -> tuple:
    """Everything a regression answers, comparable: the head's truth, the derivation, the defects (witnesses, their actions and flips), the read set, the rows,
    the value, the leaves."""
    ds = [(d.kind, d.clause, d.lit, d.atom, d.deep, None if d.witness is None else (d.witness.atom, d.witness.body, d.witness.action, d.witness.flips), d.value)
          for d in r.defects]
    return (r.holds, r.clauses, ds, r.reads, r.rows, r.value, r.leaves)


def _questions(fen: str):
    """The mate counterfactuals of every non-king move of ours and the win goal on every unit of theirs (with the goals and their futures)."""
    b = chess.Board(fen)
    me = b.turn
    for m in b.legal_moves:
        pc = b.piece_at(m.from_square)
        if pc.piece_type == chess.KING:
            continue
        pl = [(pc, m.to_square, m.from_square)]
        yield b, RG.goal_of("mate", GOALS["mate"], B, B.role(me), extra=((pc.symbol(), m.to_square, m.from_square),)), R._future_spec(pl, not me)
    for t in chess.SquareSet(b.occupied_co[not me]):
        if b.piece_type_at(t) != chess.KING:
            yield b, RG.goal_of("win", GOALS["win"], B, B.role(me), target=t, unit=UNIT), None


def test_76_4b_a_the_spec_is_read_from_its_text_and_every_extension_is_the_interpreters():
    """(a) the native core's bottom-up evaluation (the description's own clauses, read at run time) gives every relation of the spec the extension gdl.py gives it
    -- the state alone and with a joint move (the next state's relations) -- on the corner positions."""
    GC.native()
    n = 0
    for fen in CORNERS:
        b = chess.Board(fen)
        state = GC.state_of(b) | frozenset(GC.inputs_of())
        db = B.prog.run(state)
        got = A.spec_run_bb(GC.parts(b), None)
        for rel in B.prog.nodes:
            assert set(db.facts(rel)) == set(got[rel]), (fen, rel)
            n += 1
        mover, other = ("white", "black") if b.turn else ("black", "white")
        for m in list(b.legal_moves)[:4]:
            dbn = B.prog.run(state, [(mover, GC.move_term(m.uci())), (other, "noop")])
            gn = A.spec_run_bb(GC.parts(b), m.uci())
            assert all(set(dbn.facts(rel)) == set(gn[rel]) for rel in B.prog.nodes), (fen, m.uci())
    assert n == len(CORNERS) * len(B.prog.nodes)


def test_76_4b_a_the_top_down_prover_is_the_interpreter():
    """(a) the regression's evaluator (top-down, tabled; the ray a table iterated to its fixpoint): every row of the relations the regression reads equals gdl.py's,
    with the certified relations read through the binding and from their clauses."""
    GC.native()
    rels = ("slide", "attack", "attacked", "check", "empty", "occupied", "move_by", "capture", "may_land", "has_move", "mate")
    for fen in CORNERS[:6]:
        b = chess.Board(fen)
        db = B.prog.run(GC.state_of(b) | frozenset(GC.inputs_of()))
        for certified in (True, False):
            for rel in rels + (("attack_by", "legal") if not certified else ()):
                arity = len(B.prog.signature(rel))
                assert set(A.spec_ans_bb(GC.parts(b), rel, [None] * arity, certified)) == db.facts(rel), (fen, rel, certified)


def test_76_4b_b_the_native_regression_is_the_oracles():
    """(b) on every question of the fixed positions the native regression equals the Python reading field by field (the derivation, the defects with their witnesses,
    actions and flipped leaves, the read set, the rows, the value, the leaves), and the names (kinds, squares, pieces, texts, order) -- ours and theirs."""
    po, na = _oracle(), _native()
    n = witnesses = 0
    for fen in CORNERS:
        for b, g, fut in _questions(fen):
            x = B.future(b, fut)
            ro, rn = po.regress(x, g), na.regress(x, g)
            assert _rset(ro) == _rset(rn), (fen, g.name, x.fen())
            for theirs in (False, True):
                no = [(m.kind, m.sq, m.by, m.why) for m in RG.name(ro, NAMES, B, x, theirs=theirs)]
                assert no == [(m.kind, m.sq, m.by, m.why) for m in na.names(b, g, NAMES, theirs=theirs, future=fut)], (fen, g.name, theirs)
            witnesses += sum(1 for d in rn.defects if d.kind == "witness")
            n += 1
    assert n > 150 and witnesses > 50


def test_76_4b_b_a_negated_atom_and_a_recursive_ray_body():
    """mate is `check(?c), not has_move`: the TRUE negated atom's witnesses (legal moves of theirs, actions) and the leaves each flips; the interposition's leaf is
    `not occupied` of `empty`, reached through slide's recursive clause -- the native reading equals the oracle's on both."""
    na, po = _native(), _oracle()
    g = RG.goal_of("mate", GOALS["mate"], B, "white")
    b = chess.Board("6k1/5pp1/8/8/8/8/8/R5K1 w - - 0 1")                    # Ra8+ Kh7: a witness flipping the king's cell
    cf = R.future_board(b, [(chess.Piece.from_symbol("R"), chess.A8, chess.A1)], chess.BLACK)
    r = na.regress(cf, g)
    ws = [d for d in r.defects if d.kind == "witness"]
    assert ws and all(d.atom == ("has_move", ()) and d.witness.action is not None for d in ws)
    assert any(lf.at == ("check", 0, 1) for lf in ws[0].witness.flips)
    assert _rset(r) == _rset(po.regress(cf, g))
    b2 = chess.Board("4k3/8/8/8/8/8/3n4/R6K w - - 0 1")                     # Re1+: the knight steps between on e4
    cf2 = R.future_board(b2, [(chess.Piece.from_symbol("R"), chess.E1, chess.A1)], chess.BLACK)
    r2 = na.regress(cf2, g)
    leaves = [lf for d in r2.defects if d.witness is not None for lf in d.witness.flips if lf.at == ("empty", 0, 1)]
    assert leaves and leaves[0].neg and leaves[0].atom[0] == "occupied"
    assert _rset(r2) == _rset(po.regress(cf2, g))
    assert [n.kind for n in na.names(cf2, g, NAMES)].count("interposition") == 1


def test_76_4b_b_the_cache_replays_the_read_set():
    """The native core's cache per (the board's key, the goal): a second reading is a hit (the stats move) and gives the read set and the names the fresh reading
    gave -- the same as the whole regression's read set read afresh."""
    g = RG.goal_of("mate", GOALS["mate"], B, "white")
    gid = B.native_goal(g)
    cf = R.future_board(chess.Board(XRAY), [(chess.Piece.from_symbol("N"), chess.E8, chess.D6)], chess.BLACK)
    A.spec_cache_clear()
    h0, r0 = A.reads_bb(GC.parts(cf), gid)
    s1 = A.spec_stats()
    h1, r1 = A.reads_bb(GC.parts(cf), gid)
    s2 = A.spec_stats()
    assert s2[1] == s1[1] + 1 and s2[0] == s1[0]                            # the second reading is served from the cache
    assert (h0, set(r0)) == (h1, set(r1)) and set(r1) == set(B.native_regress(cf, g).reads) and len(r1) > 10
    assert set(r1) == set(_oracle().regress(cf, g).reads)


def test_76_4b_b_the_fixes_are_the_oracles():
    """regression.fixes through the native core (every action in one crossing) equals the oracle's action by action: the XRAY position's cover of the flight e8 along
    the line through their king, every legal move of ours tested; the win root's gain goal on the discovery position."""
    b = chess.Board(XRAY)
    pl = [(chess.Piece.from_symbol("N"), chess.E8, chess.D6)]
    fut = R._future_spec(pl, chess.BLACK)
    g = RG.goal_of("mate", GOALS["mate"], B, "white")
    acts = RG.actions(_native(), b, "white")
    got = RG.fixes(_native(), b, g, ("flight", chess.E8), acts, fut, NAMES)
    assert got == RG.fixes(_oracle(), b, g, ("flight", chess.E8), acts, fut, NAMES)
    assert any(GC.uci_of(a[1]) == "e2b5" and mended and not new for a, mended, new in got)
    d = chess.Board(DISCOVERY)
    gw = RG.goal_of("win", GOALS["win"], B, "white", target=chess.A8, unit=UNIT)
    acts = RG.actions(_native(), d, "white")
    assert RG.fixes(_native(), d, gw, ("defended", chess.A8), acts, None, NAMES) == RG.fixes(_oracle(), d, gw, ("defended", chess.A8), acts, None, NAMES)


def test_76_4b_b_the_engine_reads_the_native_core():
    """The engine's defects (regress.mate_defects / material_defects / threat_defects) and its fix filter (routes_of's `_stills`) equal the oracle's reading."""
    po = _oracle()
    b = chess.Board(DISCOVERY)
    for m in list(b.legal_moves)[:25]:
        pc = b.piece_at(m.from_square)
        if pc.piece_type == chess.KING:
            continue
        pl = [(pc, m.to_square, m.from_square)]
        g = RG.goal_of("mate", GOALS["mate"], B, "white")
        assert [(x.kind, x.sq, x.by, x.why) for x in R.mate_defects(b, chess.WHITE, pl)] == \
               [(n.kind, n.sq, n.by, n.why) for n in po.names(b, g, NAMES, future=R._future_spec(pl, chess.BLACK))]
    for t in chess.SquareSet(b.occupied_co[chess.BLACK]):
        if b.piece_type_at(t) != chess.KING:
            g = RG.goal_of("win", GOALS["win"], B, "white", target=t, unit=UNIT)
            assert [(x.kind, x.sq, x.by, x.why) for x in R.material_defects(b, chess.WHITE, t)] == [(n.kind, n.sq, n.by, n.why) for n in po.names(b, g, NAMES)]
            gt = RG.goal_of("win", GOALS["win"], B, "white", target=t, unit=UNIT)          # (their gain goal on our unit, read for black, the keeper)
            assert [(x.kind, x.sq, x.by, x.why) for x in R.threat_defects(b, chess.BLACK, t)] == [(n.kind, n.sq, n.by, n.why) for n in po.names(b, gt, NAMES, theirs=True)]
    q = R.new_question("mate", 100, dict(sym="N", at=chess.E8, frm=chess.D6, move="d6e8"))
    f = R.new_question("cover", 100, dict(q.pattern, target=chess.E8), q, defect_key=("flight", chess.E8, (chess.D7,)))
    x = chess.Board(XRAY)
    moves = R._moves(x, chess.WHITE, chess.D6)
    stills = R._stills(f, x, chess.WHITE, moves, q)
    gm = RG.goal_of("mate", GOALS["mate"], B, "white")
    before = R._defects_before(f, x, chess.WHITE, q)
    exp = [not mended or bool(new) for _a, mended, new in po.fixes(x, gm, ("flight", chess.E8), [("white", GC.move_term(m.uci())) for m in moves],
                                                                    R._future_spec(R.placement_of(q), chess.BLACK), NAMES, before)]
    assert stills == exp and stills.count(False) >= 1


def test_76_4b_b_the_handle_reads_the_same():
    """`Pos.regress` / `Pos.fixes` on the Rust handle answer as the board-parts calls on the same position."""
    g = RG.goal_of("win", GOALS["win"], B, "white", target=chess.A8, unit=UNIT)
    gid = B.native_goal(g)
    b = chess.Board(DISCOVERY)
    p = A.Pos(DISCOVERY)
    assert p.regress(gid) == A.regress_bb(GC.parts(b), gid)
    moves = [(m.from_square, m.to_square, m.promotion or 0) for m in b.legal_moves]
    assert p.fixes(gid, ("defended", chess.A8), moves) == A.fixes_bb(GC.parts(b), gid, ("defended", chess.A8), moves, None, None)


def test_76_4b_b_illegal_input_is_refused():
    """RUST_BOUNDARY risk 7: every entry point answers a ValueError, never a panic -- inconsistent bitboards, a square out of range, a move with no unit on its
    square, an unknown goal or relation, a description that does not parse or is not stratified, a relation of the spec no reader binds."""
    g = B.native_goal(RG.goal_of("mate", GOALS["mate"], B, "white"))
    bb = GC.parts(chess.Board(DISCOVERY))
    with pytest.raises(ValueError):
        A.regress_bb(([1, 1, 0, 0, 0, 0, 1, 0], True, 0, None, (0, 1)), g)
    with pytest.raises(ValueError):
        A.regress_bb((bb[0], True, 0, 64, (0, 1)), g)
    with pytest.raises(ValueError):
        A.fixes_bb(bb, g, ("flight", None), [(chess.E5, chess.E6, 0)], None, None)        # e5 is empty
    with pytest.raises(ValueError):
        A.regress_bb(bb, 10 ** 6)
    with pytest.raises(ValueError):
        A.spec_goal("x", "no_such_relation", [], None, None)
    with pytest.raises(ValueError):
        A.spec_goal("x", "capture", [None] * 6, "white", ("no such law", [], 1, "attack", [None] * 5))
    with pytest.raises(ValueError):
        A.spec_load("(<= p (not q)) (<= q (not p))", [], [], ("file", "rank"), 100, [1, 3, 3, 5, 9, 0])      # not stratified
    with pytest.raises(ValueError):
        A.spec_load("(<= (p ?x) (q ?x)", [], [], ("file", "rank"), 100, [1, 3, 3, 5, 9, 0])                  # unbalanced
    with pytest.raises(ValueError):
        A.spec_load(GC.SPEC.read_text(), [x for x in GC.native_bindings() if x[0] != "true.cell"], list(GC.CERTIFIED), ("file", "rank"), 100, [1, 3, 3, 5, 9, 0])
    assert A.spec_loaded() and A.regress_bb(bb, g)[0] in (True, False)       # (the refused loads left the loaded spec in place)


def test_76_4b_b_the_binding_table_is_the_shelfs():
    """(a) every input relation of the spec and every certified one is bound by its ontology shelf entry's `rust` name, each a reader of the native binding."""
    got = dict(GC.native_bindings())
    want = {r for r in B.prog.nodes if r.startswith("true.") or r in ("does", "record")} | set(GC.CERTIFIED)
    assert set(got) == want
    assert got["true.cell"] == "rules.rs:cell" and got["legal"] == "pos.rs:legal_ucis" and got["attack_by"] == "pos.rs:attacks"


def test_76_4b_the_python_chess_board_is_mirrored():
    """pyboard.rs reads a board as python-chess holds it: the legal moves in python-chess's order and `push` field by field, on the corners (castling, en passant
    pinned and not, promotions, evasions), on boards shakmaty refuses (the side not to move in check, no king) and on random walks."""
    def parts_of(b):
        return GC.parts(b)
    boards = [chess.Board(f) for f in CORNERS]
    odd = chess.Board("4k3/8/8/8/8/8/4r3/4K2R b K - 0 1")
    odd.turn = chess.WHITE                                                   # white to move while black's rook stands on e2 -- fine; and the reverse:
    boards.append(odd)
    o2 = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
    o2.turn = chess.BLACK                                                    # black to move, white in check: shakmaty refuses it
    boards.append(o2)
    boards.append(chess.Board("8/8/8/3q4/8/8/8/R7 w - - 0 1"))              # no kings
    rng = random.Random(764)
    for p in PA.load_set("dev_20")[:10]:
        b = chess.Board(p["fen"])
        for _ in range(6):
            boards.append(b.copy(stack=False))
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
    n = 0
    for b in boards:
        assert A.py_legal_full_bb(parts_of(b)) == [m.uci() for m in b.legal_moves], b.fen()
        for m in b.legal_moves:
            c = b.copy(stack=False)
            c.push(m)
            assert A.py_push_bb(parts_of(b), m.uci()) == parts_of(c), (b.fen(), m.uci())
            n += 1
    assert n > 1000


CHESS_WORDS = r"\b(chess|python-chess|king|queen|rook|bishop|knight|pawn|piece|square|board|mate|check|capture|castl\w*|promot\w*|uci|san|fen|file|rank|" \
              r"KING|QUEEN|ROOK|BISHOP|KNIGHT|PAWN|[a-h][1-8])\b"          # (test_plan76_4a's: a generic module names no game object)


def test_76_4b_the_generic_core_names_no_game_object():
    """The native core's generic modules -- the description's evaluator (gdl.rs) and the regression (regression.rs) -- name no game object in their code (doc
    comments and comments aside); the chess binding is gdl_chess.rs and pyboard.rs."""
    for f in ("gdl.rs", "regression.rs"):
        t = (ROOT / "rust" / "analyst_core" / "src" / f).read_text()
        code = "\n".join(re.sub(r'"[^"\n]*"', '""', ln.split("//")[0]) for ln in t.splitlines())
        hits = [m.group(0) for m in re.finditer(CHESS_WORDS, code, flags=re.I)]
        assert hits == [], (f, hits)
