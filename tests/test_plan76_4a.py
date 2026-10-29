"""PLAN 76.4a DEFECTS AND FIXES FROM THE SPEC (docs/ANALYST_PLAN.md 76.4a; Kaveh 2026-10-28, "how are the defects found and raised?"): the regression of a goal
over the clause bodies of rules/chess.gdl (discovery/regression.py, generic) gives the hand tables' defects, reads negated atoms by their witnesses and recursive
ray bodies by tables, caches per (state, goal), names no game object, and its names are store entries bound to places of the spec."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import gdl_chess as GC
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery import regression as RG
from oracles import regress_tables as HT                            # plan 76.4a (c): the hand tables, test oracles

CHESS_WORDS = r"\b(chess|analyst_core|python-chess|king|queen|rook|bishop|knight|pawn|piece|square|board|mate|check|capture|castl\w*|promot\w*|uci|san|fen|file|rank|" \
              r"KING|QUEEN|ROOK|BISHOP|KNIGHT|PAWN|[a-h][1-8])\b"          # (tests/test_plan75.py's: a generic module names no game object)

B = GC.Binding()
SPEC = RG.Spec(B.prog, certified=GC.CERTIFIED)
NAMES = LS.defect_names()


def _rg() -> RG.Regressor:
    return RG.Regressor(SPEC, B)


def _mate(rg, cf, me: bool) -> list:
    g = RG.goal_of("mate", LS.gdl_goals()["mate"], B, B.role(me))
    return [(n.kind, n.sq, n.by, n.why) for n in RG.name(rg.regress(cf, g), NAMES, B, cf)]


def _win(rg, board, me: bool, t: int) -> list:
    g = RG.goal_of("win", LS.gdl_goals()["win"], B, B.role(me), target=t, unit=LS.goal_kinds().valuation.unit)
    return [(n.kind, n.sq, n.by, n.why) for n in RG.name(rg.regress(board, g), NAMES, B, board)]


def _hand_mate(cf, me) -> list:
    return [(d.kind, d.sq, d.by, d.why) for d in HT.mate_defects_on(cf, me)]


def _hand_win(board, me, t) -> list:
    return [(d.kind, d.sq, d.by, d.why) for d in HT.material_defects(board, me, t)]


DISCOVERY = "r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1"
POS2 = "r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1"
RA8 = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
XRAY = "2r5/pp1k1p2/1b1Np1p1/8/4P1pq/8/PP2QP2/3R1RK1 w - - 1 21"        # Nd6 moves: the rook on d1 checks; Qb5 covers e8 through the king on d7


def test_76_4a_the_regression_gives_the_hand_tables_defects():
    """Every counterfactual of every non-king move of ours in the fixed positions (the pattern placed, their move) regresses to the hand mate table's defects --
    kinds, squares, pieces involved, texts and order; every unit of theirs to the hand material table's."""
    rg = _rg()
    n = check = 0
    for fen in (DISCOVERY, POS2, RA8, XRAY, "8/8/8/8/8/2k5/8/K1Q5 w - - 0 1", "4k3/8/8/8/8/8/3PPP2/R3K2R w KQ - 0 1"):
        b = chess.Board(fen)
        me = b.turn
        for m in b.legal_moves:
            pc = b.piece_at(m.from_square)
            if pc.piece_type == chess.KING:
                continue
            cf = R.future_board(b, [(pc, m.to_square, m.from_square)], not me)
            assert _mate(rg, cf, me) == _hand_mate(cf, me), (fen, m.uci())
            n += 1
            check += cf.is_check()
        for t in chess.SquareSet(b.occupied_co[not me]):
            if b.piece_type_at(t) != chess.KING:
                assert _win(rg, b, me, t) == _hand_win(b, me, t), (fen, chess.square_name(t))
    assert n > 100 and check > 10


def test_76_4a_a_negated_atom_is_read_by_its_witnesses():
    """mate is `check(?c), not has_move`: in check with escapes the TRUE negated atom has_move is a defect per witness (each a legal move of theirs, an action),
    and each witness names the leaf of check's derivation it flips; mated, the head holds and nothing is a defect."""
    rg = _rg()
    b = chess.Board(RA8)
    pl = [(chess.Piece.from_symbol("R"), chess.A8, chess.A1)]
    mated = R.future_board(b, pl, chess.BLACK)
    g = RG.goal_of("mate", LS.gdl_goals()["mate"], B, "white")
    r = rg.regress(mated, g)
    assert r.holds and r.defects == []
    b2 = chess.Board("6k1/5pp1/8/8/8/8/8/R5K1 w - - 0 1")                    # h7 open: Ra8+ Kh7
    cf = R.future_board(b2, pl, chess.BLACK)
    r2 = rg.regress(cf, g)
    ws = [d for d in r2.defects if d.kind == "witness"]
    assert not r2.holds and ws and all(d.atom == ("has_move", ()) and d.witness.action is not None for d in ws)
    assert [(n.kind, n.sq) for n in RG.name(r2, NAMES, B, cf)] == [("flight", chess.H7)]
    assert any(lf.at == ("check", 0, 1) for lf in ws[0].witness.flips)       # the leaf it flips: the cell of the checked unit (check's literal 1)


def test_76_4a_a_recursive_ray_body_is_read_through_its_table():
    """The top-down tables equal the bottom-up interpreter (gdl.py) on every row of the derived relations the regression reads -- the ray (`slide`, recursive),
    the attacks, check, the empty squares, the moves and captures -- on positions with the certified relations read through the kernel and without; and the
    interposition's leaf is the `not occupied` literal of `empty` reached through the recursive clause of `slide`."""
    rels = ("slide", "attack", "attacked", "check", "empty", "occupied", "move_by", "capture", "may_land")
    fens = (DISCOVERY, POS2, XRAY, "8/8/8/K2pP2r/8/8/8/7k w - d6 0 2", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    spec0 = RG.Spec(B.prog, certified=())
    for fen in fens:
        b = chess.Board(fen)
        db = B.prog.run(GC.state_of(b) | frozenset(GC.inputs_of()))
        for spec in (spec0, SPEC):
            P = RG.Prover(spec, B.facts(b), B.certified(b))
            for rel in rels + (("attack_by",) if spec is spec0 else ()):
                arity = len(B.prog.signature(rel))
                assert set(P.ans(rel, (None,) * arity)) == db.facts(rel), (fen, rel)
    rg = _rg()
    b = chess.Board("4k3/8/8/8/8/8/3n4/R6K w - - 0 1")                      # Re1+: the knight on d2 steps between on e4
    cf = R.future_board(b, [(chess.Piece.from_symbol("R"), chess.E1, chess.A1)], chess.BLACK)
    g = RG.goal_of("mate", LS.gdl_goals()["mate"], B, "white")
    r = rg.regress(cf, g)
    inter = [n for n in RG.name(r, NAMES, B, cf) if n.kind == "interposition"]
    assert inter and _hand_mate(cf, chess.WHITE) == [(n.kind, n.sq, n.by, n.why) for n in RG.name(r, NAMES, B, cf)]
    leaf = [lf for d in inter[0].src for lf in d.witness.flips if lf.at == ("empty", 0, 1)]
    assert leaf and leaf[0].neg and leaf[0].atom[0] == "occupied"


def test_76_4a_the_regression_is_cached_per_state_and_goal():
    rg = _rg()
    cf = R.future_board(chess.Board(RA8), [(chess.Piece.from_symbol("R"), chess.A8, chess.A1)], chess.BLACK)
    g = RG.goal_of("mate", LS.gdl_goals()["mate"], B, "white")
    r1 = rg.regress(cf, g)
    r2 = rg.regress(cf.copy(), g)
    assert r1 is r2 and rg.stats["regressions"] == 1 and rg.stats["cached"] == 1
    g2 = RG.goal_of("win", LS.gdl_goals()["win"], B, "white", target=chess.F7, unit=1)
    assert rg.regress(cf, g2) is not r1 and rg.stats["regressions"] == 2


def test_76_4a_the_regression_is_generic():
    """regression.py names no game object (its code, docstrings and comments aside) and imports nothing of a game: the spec, the binding and the store say it all."""
    p = Path(RG.__file__)
    t = re.sub(r'"""(?:.|\n)*?"""', "", p.read_text())
    code = "\n".join(ln.split("#")[0] for ln in t.splitlines())
    hits = [m.group(0) for m in re.finditer(CHESS_WORDS, code, flags=re.I)]
    assert hits == [], hits
    imports = [ln.strip() for ln in code.splitlines() if ln.strip().startswith(("import ", "from "))]
    assert all(("discovery import gdl as GDL" in ln) or ln.split()[1] in ("__future__", "time", "dataclasses") for ln in imports), imports


def test_76_4a_every_name_is_bound_to_a_place_of_the_spec(tmp_path):
    """The DEFECT_FIX names of the mate and win goals are bound (`gdl`), every binding a literal of a clause of the spec; a binding to no literal is caught."""
    store = LS.load()
    assert LS.gdl_problems(store) == []
    names = {e["name"] for e in NAMES}
    assert {"no check", "checker takeable", "flight", "interposition", "not attacked", "defended"} <= names
    bad = json.loads(json.dumps(store))
    bad["flight"]["params"]["gdl"]["leaf"] = ["check", 0, 9]
    bad["mate"]["params"]["gdl"]["head"] = ["no_such_relation"]
    probs = LS.gdl_problems(bad)
    assert any("flight" in x for x in probs) and any("mate" in x for x in probs)
    rule = store["the rule: attack"]
    assert rule["kind"] == "RULE" and rule["status"] == "PROVED" and rule["params"]["gdl"] in GC.CERTIFIED


def test_76_4a_the_certified_attacks_are_the_specs():
    """The binding's `attack_by` rows (the kernel's, licensed by Ideas.LegalEquiv.attackBy_iff) equal the interpreter's on random positions, for every pattern the
    regression asks (a target bound, an origin bound, nothing bound)."""
    rng = random.Random(4)
    fens = [p["fen"] for p in PA.load_set("dev_20")[:6]]
    spec0 = RG.Spec(B.prog, certified=())
    for fen in fens:
        b = chess.Board(fen)
        for _ in range(4):
            P0 = RG.Prover(spec0, B.facts(b))
            cert = B.certified(b)
            for s in (chess.E4, chess.D5, chess.H8, chess.A1):
                f, r = GC.sq(s)
                for pat in ((None, "white", None, None, f, r), (None, None, f, r, None, None), (None, "black", None, None, f, r)):
                    assert sorted(cert("attack_by", pat)) == sorted(P0.ans("attack_by", pat)), (fen, pat)
            assert sorted(cert("attack_by", (None,) * 6)) == sorted(P0.ans("attack_by", (None,) * 6))
            b.push(rng.choice(list(b.legal_moves)))


def test_76_4a_c_the_engine_reads_its_defects_from_the_regression():
    """(c) THE SWITCH-OVER: regress.mate_defects / material_defects are the regression's (the hand tables are the oracles): equal to the oracles on the fixed
    positions, and a regression is read (the regressor's count moves); the plan 54 Rust twins are no longer read by the engine (no DEFECTS_RUST)."""
    import inspect
    assert not hasattr(R, "DEFECTS_RUST") and not hasattr(R, "_routes_rust") and not hasattr(R, "_goal_done")
    src = inspect.getsource(R)
    assert "mate_defects_pos" not in src and "material_defects_pos" not in src and "still_routes_pos" not in src
    n0 = R.regressor().stats["regressions"] + R.regressor().stats["cached"] + R.regressor().stats["validated"]
    b = chess.Board(DISCOVERY)
    for m in list(b.legal_moves)[:20]:
        pc = b.piece_at(m.from_square)
        if pc.piece_type == chess.KING:
            continue
        pl = [(pc, m.to_square, m.from_square)]
        assert [(d.kind, d.sq, d.by, d.why) for d in R.mate_defects(b, chess.WHITE, pl)] == [(d.kind, d.sq, d.by, d.why) for d in HT.mate_defects(b, chess.WHITE, pl)]
    for t in chess.SquareSet(b.occupied_co[chess.BLACK]):
        if b.piece_type_at(t) != chess.KING:
            assert [(d.kind, d.sq, d.by, d.why) for d in R.material_defects(b, chess.WHITE, t)] == _hand_win(b, chess.WHITE, t)
    assert R.regressor().stats["regressions"] + R.regressor().stats["cached"] + R.regressor().stats["validated"] > n0


def test_76_4a_c_a_keep_questions_defects_are_the_other_sides_goal_holding():
    """(c) the save goal is the other side's goal read for the keeper: in check, their terminal goal's positive literal `check` holds -- one 'in check' per
    attacker of our king; a unit that hangs, their gain goal holds on it -- 'hangs' with their capturers and the exchange's value for them."""
    assert R.threat_defects(chess.Board("4k3/8/8/8/1b6/8/3r4/4K3 w - - 0 1"), chess.WHITE) == []          # the rook on d2 blocks the bishop: no check
    one = R.threat_defects(chess.Board("4k3/8/8/8/8/8/3rr3/4K3 w - - 0 1"), chess.WHITE)
    assert [(x.kind, x.sq, x.by, x.why) for x in one] == [("in check", chess.E2, [], "R on e2 gives check")]
    two = R.threat_defects(chess.Board("4k3/8/8/8/1b6/8/8/r3K3 w - - 0 1"), chess.WHITE)                   # a double check: one defect per attacker
    assert [(x.kind, x.sq) for x in two] == [("in check", chess.A1), ("in check", chess.B4)]
    h = R.threat_defects(chess.Board("4k3/8/8/3n4/4P3/8/8/4K3 b - - 0 1"), chess.BLACK, chess.D5)          # the knight on d5, attacked by the e4 pawn, undefended
    assert [(x.kind, x.sq, x.by, x.why) for x in h] == [("hangs", chess.D5, [chess.E4], "they win 3 by the exchange on d5")]
    h2 = R.threat_defects(chess.Board("4k3/8/2p5/3n4/4P3/8/8/4K3 b - - 0 1"), chess.BLACK, chess.D5)       # defended by the c6 pawn: they still win 2 (N for P)
    assert [(x.kind, x.why) for x in h2] == [("hangs", "they win 2 by the exchange on d5")]
    assert R.threat_defects(chess.Board("4k3/8/2p5/3p4/4P3/8/8/4K3 b - - 0 1"), chess.BLACK, chess.D5) == []      # a pawn defended by a pawn: they win nothing


def test_76_4a_c_the_walls_flights_are_the_landings_of_the_names():
    """(c) the squares their king can step to are read off the names' bindings (a witness read at its landing), not the word 'flight'."""
    b = chess.Board("6k1/5pp1/8/8/8/8/8/R5K1 w - - 0 1")
    ds = R.mate_defects(b, chess.WHITE, [(chess.Piece.from_symbol("R"), chess.A8, chess.A1)])
    assert R.landings(ds) == [d.sq for d in ds if d.kind == "flight"] == [chess.H7]


def test_76_4a_d_a_goal_no_table_was_written_for_win_this_exchange():
    """(d) THE NEW GOAL, by the same code: 'win this exchange' (the store's GOAL `exchange`: the capture from a given unit, the exchange begun by it as the value)
    -- a capture that loses now names its defect (the value short, the attacks of theirs on the target read); a capture by a unit that does not bear names the
    false literal; regression.fixes finds the moves after which the same capture wins (nothing new); a winning capture has no defect."""
    rg = _rg()
    gx = LS.gdl_goals()["exchange"]
    b = chess.Board("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1")                    # Qxd5? the c6 pawn takes back: the exchange gives 1 - 9 = -8
    g = RG.goal_of("exchange", gx, B, "white", target=chess.D5, unit=1, frm=chess.D1)
    r = rg.regress(b, g)
    nm = RG.name(r, NAMES, B, b)
    assert not r.holds and [(n.kind, n.sq, n.by) for n in nm] == [("the exchange loses", chess.D1, [chess.C6])] and "-8" in nm[0].why
    g2 = RG.goal_of("exchange", gx, B, "white", target=chess.C6, unit=1, frm=chess.D1)      # the queen does not bear on c6
    assert [(n.kind, n.sq) for n in RG.name(rg.regress(b, g2), NAMES, B, b)] == [("the capturer does not bear", chess.D1)]
    b3 = chess.Board("4k3/8/2p5/3p4/8/8/3R4/3QK3 w - - 0 1")                 # the rook in front: Rxd5 cxd5 Qxd5 -- the exchange from the rook gives 1 - (5 - 1) = -3
    g3 = RG.goal_of("exchange", gx, B, "white", target=chess.D5, unit=1, frm=chess.D2)
    assert [(n.kind, "-3" in n.why) for n in RG.name(rg.regress(b3, g3), NAMES, B, b3)] == [("the exchange loses", True)]
    b4 = chess.Board("4k3/8/8/3p4/8/8/8/3QK3 w - - 0 1")                     # an undefended pawn: Qxd5 wins it
    assert rg.regress(b4, RG.goal_of("exchange", gx, B, "white", target=chess.D5, unit=1, frm=chess.D1)).holds
    e = chess.Board("8/1Rb3kp/p2q2p1/2p1pp2/pQ6/1bP4P/1P3PPK/8 w - - 0 35")  # eSJqH (the guard's ten): Rxc7+ Qxc7 loses 2 now
    ge = RG.goal_of("exchange", gx, B, "white", target=chess.C7, unit=1, frm=chess.B7)
    ne = RG.name(rg.regress(e, ge), NAMES, B, e)
    assert [(n.kind, n.by) for n in ne] == [("the exchange loses", [chess.D6])]
    acts = [a for a in RG.actions(rg, e, "white") if B.ends(a)[0] != chess.B7]
    fixes = [e.san(chess.Move(*B.ends(a))) for a, mended, new in RG.fixes(rg, e, ge, (ne[0].kind, ne[0].sq), acts, (lambda x: x), NAMES) if mended and not new]
    assert "Qb6" in fixes                                                    # after Qb6 the capture from b7 wins by the exchange, nothing new


def test_76_4a_d_the_kr_floor_mate_in_two_by_the_same_regression():
    """(d) the mined K+R technique's floor (its measure's floor: a mate within two moves): the mate goal regressed at the leaves of regression.within -- every
    reply read -- has Kb6 as a fix (Kb8 forced, Rh8#) and names the escapes of a first move that is no fix."""
    rg = _rg()
    b = chess.Board("k7/8/2K5/8/8/8/8/7R w - - 0 1")
    g = RG.goal_of("mate", LS.gdl_goals()["mate"], B, "white")
    res = RG.within(rg, b, g, "white", 2, NAMES)
    by = {GC.uci_of(w.action[1]): w for w in res}
    assert by["c6b6"].holds and not by["c6b6"].escapes
    assert not by["h1h2"].holds and by["h1h2"].escapes                        # a waiting rook move: their king walks out
    assert all(not w.holds for w in RG.within(rg, b, g, "white", 1, NAMES))     # no mate in one there


def test_76_4a_the_fix_test_reads_the_line_through_their_king():
    """A FIX is an action after which the defect is gone and nothing is new (regression.fixes). Nd6 discovers the rook's check on d7; Qb5 covers the flight e8 along
    the line THROUGH the king (the king on e8 would be attacked from b5): the derived fix test finds it -- the hand table's cover test (the flight square attacked
    with the king still standing on d7) does not: a table incompleteness, kept as a finding (plan 76.4a (b))."""
    rg = _rg()
    b = chess.Board(XRAY)
    pl = [(chess.Piece.from_symbol("N"), chess.E8, chess.D6)]
    fut = lambda x: R.future_board(x, pl, chess.BLACK)          # noqa: E731
    g = RG.goal_of("mate", LS.gdl_goals()["mate"], B, "white")
    keys = [(n.kind, n.sq) for n in RG.name(rg.regress(fut(b), g), NAMES, B, fut(b))]
    assert ("flight", chess.E8) in keys
    qb5 = ("white", GC.move_term("e2b5"))
    (_a, mended, new), = RG.fixes(rg, b, g, ("flight", chess.E8), [qb5], fut, NAMES)
    assert mended and new == []
    q = R.new_question("mate", 100, dict(sym="N", at=chess.E8, frm=chess.D6, move="d6e8"))
    f = R.new_question("cover", 100, dict(q.pattern, target=chess.E8), q, defect_key=("flight", chess.E8, (chess.D7,)))
    assert HT.still(f, b, chess.WHITE, chess.Move.from_uci("e2b5"), q)           # the hand table: still (not a fix)
    assert not R._still(f, b, chess.WHITE, chess.Move.from_uci("e2b5"), q)       # plan 76.4a (c): the engine's filter is the regression's -- a fix
