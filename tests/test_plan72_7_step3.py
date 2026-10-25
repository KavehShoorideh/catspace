"""plan 72.7 THE RUST BOUNDARY, step 3 (docs/RUST_BOUNDARY.md section 4 step 3): THE PYTHON WORDS IN RUST. Each ported word is compared with its Python body --
kept HERE, verbatim, as the TEST ORACLE (the engine's module now calls the Rust word) -- on seeded random games, on the guard's positions and on puzzle positions:
values, structures, orders, and what each wrote on the dependency trackers (RUST_BOUNDARY risk 2: `dep_peek` before and after); illegal input is a ValueError
(risk 7); a cache a word keeps is emptied by `graph.clear_caches` (the CLEAR_HOOKS).
3a THE MATE IN TWO: `selfmine.bind_mate2` (Lean `Ideas.Mined.Schemas.mate_in_two`) and `regress.mate2_threat` (plan 59 tier 2) -- words.rs `bind_mate2_pos`,
`mate2_threat_pos`."""
import importlib.util
import random
from pathlib import Path

import chess
import pytest

import analyst_core as AC
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery import selfmine as SM

SPEC = importlib.util.spec_from_file_location("trace_identity", Path(__file__).resolve().parents[1] / "scripts" / "trace_identity.py")
TI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TI)
POS2 = TI.FENS["pos2"][0]
DISCOVERY = TI.FENS["discovery"][0]


def _games(n: int = 12, plies: int = 40, seed: int = 7273) -> list:
    """Seeded random games from the start, the guard's two fixed positions and the cross-check's corners: every position on them."""
    rng = random.Random(seed)
    roots = [chess.STARTING_FEN, POS2, DISCOVERY] + list(TI.CORNERS)
    out = []
    for k in range(n):
        b = chess.Board(roots[k % len(roots)])
        for _ in range(plies):
            out.append(b.copy(stack=False))
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
    return out


def _puzzles(sets=("dev_20", "rand8_10", "rand14_10", "rand16_10", "rand21_10"), walk: int = 2, seed: int = 7274) -> list:
    """The guard's case set and puzzle positions (mates are common there), each walked `walk` plies (seeded)."""
    rng = random.Random(seed)
    roots = [fen for _pid, fen, _us in TI._cases("ten")]
    for s in sets:
        try:
            roots += [p["fen"] for p in PA.load_set(s)]
        except FileNotFoundError:
            continue
    out = []
    for f in roots:
        b = chess.Board(f)
        for _ in range(walk + 1):
            out.append(b.copy(stack=False))
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
    return out


def _deps():
    """What the dependency trackers hold now (nothing cleared)."""
    return AC.dep_peek()


# ---------------------------------------------------------------------------------------------------------------- 3a: the oracles (the Python bodies before the port, verbatim)

def bind_mate2_oracle(fen: str, mover: str, candidates: list | None = None) -> list:
    """BIND `mate_in_two` to a position: the candidates in order (every legal move but castling and promotions), each tail read in Rust (`premise_truth`);
    STOPS AT THE FIRST that reads (no mate in two beats another). -> [dict(depth=1, pairs, B=None, T=None, prem, atoms, family='mate2')] (zero or one)."""
    import chess
    if candidates is None:
        bd = chess.Board(fen)
        seen, candidates = set(), []
        for m in bd.legal_moves:
            if m.promotion or bd.is_castling(m):
                continue
            pr = (chess.square_name(m.from_square), chess.square_name(m.to_square))
            if pr not in seen:
                seen.add(pr)
                candidates.append([pr])
    for pairs in candidates:
        if len(pairs) != 1:
            continue
        (x, y), = pairs
        with SM.using_mover(mover):
            prem = [SM.atom_of(SM.mate2_tail(x, y), ">=", 1)]
        if SM.premise_truth(prem, [dict(fen=fen, mover=mover)])[0]:
            return [dict(depth=1, pairs=pairs, B=None, T=None, prem=prem, atoms=SM.with_legality(prem), family=SM.MATE2)]
    return []


def mate2_threat_oracle(board: chess.Board, m: chess.Move, us) -> bool:
    """plan 59 tier 2: after our (quiet) move m and a pass, we have a FORCED MATE IN TWO, read by the proved word `mate_in_two` (Lean `Ideas.Mined.Schemas.mate_in_two`: a move x -> y after which
    every reply is met by a mate; bound by `selfmine.bind_mate2`, the premise read in Rust) -- the same word the binder's mate_in_two schema uses, on the board after m with the move passed to them."""
    me = G.colour_of(us)
    c = board.copy(stack=False)
    c.push(m)
    if c.is_check() or not any(True for _ in G.with_turn(c, me).legal_moves):
        return False
    return bool(bind_mate2_oracle(G.with_turn(c, me).fen(), "w" if me else "b"))


def _sig(got: list) -> list:
    """A binding as comparable data: (pairs, the premise's atoms rendered, the atoms with their legs)."""
    return [(d["depth"], [tuple(p) for p in d["pairs"]], d["B"], d["T"], [(SM.BV.sexpr(a.term), a.op, a.v) for a in d["prem"]],
             [(SM.BV.sexpr(t), op, v) for t, op, v in d["atoms"]], d["family"]) for d in got]


# ---------------------------------------------------------------------------------------------------------------- 3a: the tests

def test_72_7_3a_bind_mate2_equals_its_oracle_on_puzzles_and_games():
    """bind_mate2 (Rust, one call) against the Python body on every position of the puzzles and the games, as given (the side to move the mover) and with a
    caller's candidate list: the same binding -- pairs, premise, atoms -- or none; the trackers untouched by either."""
    G.clear_caches()
    n = hits = 0
    for b in _puzzles() + _games():
        fen, mover = b.fen(), "w" if b.turn else "b"
        AC.dep_clear()
        want = bind_mate2_oracle(fen, mover)
        d_want = _deps()
        AC.dep_clear()
        got = SM.bind_mate2(fen, mover)
        assert _deps() == d_want, fen
        assert _sig(got) == _sig(want), fen
        assert _sig(SM.bind_mate2(None, mover, pos=G.pos_of(b))) == _sig(want), fen          # the handle, no FEN
        n += 1
        hits += bool(want)
    cands = [[("a1", "a2")], [("e2", "e4"), ("e7", "e5")], [("h1", "h8")]]
    for b in _puzzles()[:40]:
        fen, mover = b.fen(), "w" if b.turn else "b"
        mine = [[(chess.square_name(m.from_square), chess.square_name(m.to_square))] for m in b.legal_moves if not m.promotion][::-1]
        for cl in (cands, mine, cands + mine):
            assert _sig(SM.bind_mate2(fen, mover, [list(x) for x in cl])) == _sig(bind_mate2_oracle(fen, mover, [list(x) for x in cl])), fen
    assert hits > 0
    print(f"OBS 72.7 3a bind_mate2 against its oracle: positions {n}, bindings {hits}, every one equal")


def test_72_7_3a_mate2_threat_equals_its_oracle_on_every_move():
    """mate2_threat (Rust, on the handle) against the Python body for EVERY legal move of the puzzles' positions and a sample of the games', for the side to move and
    (the word asks either) for the other side as `us`."""
    G.clear_caches()
    n = hits = 0
    boards = _puzzles(walk=1) + _games(n=6, plies=30)
    for b in boards:
        for m in b.legal_moves:
            for us in (b.turn, not b.turn):
                want = mate2_threat_oracle(b, m, us)
                assert R.mate2_threat(b, m, us) == want, (b.fen(), m.uci(), us)
                n += 1
                hits += want
    assert hits > 0
    print(f"OBS 72.7 3a mate2_threat against its oracle: moves {n}, threats {hits}, every one equal")


def test_72_7_3a_illegal_input_is_a_value_error():
    p = AC.Pos(chess.STARTING_FEN)
    with pytest.raises(ValueError):
        AC.mate2_threat_pos(p, "e2e5", True)
    with pytest.raises(ValueError):
        AC.mate2_threat_pos(p, "zz", True)
    with pytest.raises(ValueError):
        AC.bind_mate2_pos(p, True, [("e9", "e4")])
    assert AC.bind_mate2_pos(p, True, []) is None
