"""plan 72.7 THE RUST BOUNDARY, step 3 (docs/RUST_BOUNDARY.md section 4 step 3): THE PYTHON WORDS IN RUST. Each ported word is compared with its Python body --
kept HERE, verbatim, as the TEST ORACLE (the engine's module now calls the Rust word) -- on seeded random games, on the guard's positions and on puzzle positions:
values, structures, orders, and what each wrote on the dependency trackers (RUST_BOUNDARY risk 2: `dep_peek` before and after); illegal input is a ValueError
(risk 7); a cache a word keeps is emptied by `graph.clear_caches` (the CLEAR_HOOKS).
3a THE MATE IN TWO: `selfmine.bind_mate2` (Lean `Ideas.Mined.Schemas.mate_in_two`) and `regress.mate2_threat` (plan 59 tier 2) -- words.rs `bind_mate2_pos`,
`mate2_threat_pos`.
3b THE ROLL AND THE STATIC WALKS: `regress.reach_map` / `reach_price` (plan 72.1), `race.stopper_walk` / `deny_moves` (62.1b), `graph.piece_moves_to` -- words.rs
`reach_map_bb`, `reach_price_bb`, `stopper_walk_bb`, `deny_moves_bb`, `piece_moves_to_bb`; boards whose side NOT to move is in check (python-chess holds them, the roll reaches them
after a quiet check) among the positions.
3c THE PASSER FAMILY: `regress.promote_defects` (52.6, 72.2's priced stopper), `stuck_threats` (72.2 (b)) with the passer duty it asks (`can_run` / `duty_after`
passers, 63.2), and python-chess's legal moves mirrored on a placement (`py_legal`) -- words.rs `promote_defects_bb`, `stuck_threats_pos`, `can_run_bb`, `py_legal_bb`.
3d THE KING VERDICT: `race.king_verdict` / `support_moves` (62.1c) -- words.rs `king_verdict_bb`, `support_moves_bb`."""
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


def _mate1_after_pass(b, m, us) -> bool:
    c = b.copy(stack=False)
    c.push(m)
    if c.is_check():
        return False
    P = G.pos_of(c)
    Q = P.with_turn(us) if P is not None else None
    return bool(Q is not None and Q.mates())


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
                got = R.mate2_threat(b, m, us)
                # plan 77.1 (a) (2026-11-04): the threat of mate in two is the composition (some move, then a mate in one after every reply), which counts a mate in ONE
                # after the pass too (forcing_why reads that first, as 'threat of mate'); otherwise equal to the mined word
                assert got == want or (got and _mate1_after_pass(b, m, us)), (b.fen(), m.uci(), us)
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


# ---------------------------------------------------------------------------------------------------------------- 3b: the oracles (the Python bodies before the port, verbatim)

def piece_moves_to_oracle(board: chess.Board, frm: int, targets: set, limit: int) -> int | None:
    """Fewest moves the non-pawn piece on `frm` needs to stand on one of `targets` (static board: the other pieces do not move, own-colour squares are not
    entered, a piece of the other colour may be taken; no pins, no checks), or None beyond `limit`. The `reach` of a slider / knight."""
    pc = board.piece_at(frm)
    seen, frontier = {frm}, [frm]
    for k in range(1, limit + 1):
        nxt = []
        for s in frontier:
            c = board.copy(stack=False)
            c.remove_piece_at(frm)
            c.set_piece_at(s, pc)
            for t in c.attacks(s):
                o = board.piece_at(t)
                if t in seen or (o is not None and o.color == pc.color and t != frm):
                    continue
                if t in targets:
                    return k
                seen.add(t)
                nxt.append(t)
        frontier = nxt
    return None


def stopper_walk_oracle(board: chess.Board, s: int, targets: set, limit: int, deny: int = 0) -> tuple | None:
    """The shortest walk of the piece on `s` to a square of `targets` in at most `limit` moves over the squares NOT in the `deny` mask (static board: the other pieces stand; a square holding a piece
    of its own colour is not entered; a king may take a piece of ours that is not denied). -> (moves, [the squares it lands on, in order]) or None. This is the PATH of the stopper
    (Kaveh 2026-10-22: my intent is to promote; their intent is to stop it; how -- which piece, along which path), and `deny` is the set of squares it cannot STAND and SURVIVE on
    (Race.deny_sound: a walk that needs a denied square is not a walk)."""
    pc = board.piece_at(s)
    if pc is None:
        return None
    seen, frontier = {s}, [(s, [])]
    for k in range(1, limit + 1):
        nxt = []
        for sq, path in frontier:
            if pc.piece_type == chess.KING:
                cand = chess.SquareSet(chess.BB_KING_ATTACKS[sq])
            else:
                c = board.copy(stack=False)
                c.remove_piece_at(s)
                c.set_piece_at(sq, pc)
                cand = chess.SquareSet(c.attacks_mask(sq))
            for t in cand:
                if t in seen:
                    continue
                o = board.piece_at(t)
                if o is not None and o.color == pc.color and t != s:
                    continue
                if deny & chess.BB_SQUARES[t]:
                    continue
                if t in targets:
                    return k, path + [t]
                seen.add(t)
                nxt.append((t, path + [t]))
        frontier = nxt
    return None


def denial_mask_oracle(after: chess.Board, me: bool, stopper: int, skip: int) -> int:
    """race.denial_mask (unchanged by the port; the oracle's own copy)."""
    pc = after.piece_at(stopper)
    c = after.copy(stack=False)
    mask = 0
    if pc is None:
        return 0
    if pc.piece_type == chess.KING:
        c.remove_piece_at(stopper)
        for t in chess.SquareSet(c.occupied_co[me]):
            if t != skip:
                mask |= c.attacks_mask(t)
        return mask
    if pc.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for t in chess.SquareSet(c.pawns & c.occupied_co[me]):
            if t != skip:
                mask |= c.attacks_mask(t)
    return mask


def deny_moves_oracle(board: chess.Board, me: bool, sq: int, stopper: int, targets: set, n: int, legal: list) -> list:
    """62.1b: the moves of ours (not the passer's own) after which the stopper's walk to `targets` no longer exists within n + 1 moves (they get a reply before our first push) over the
    squares it can stand on -- DENY THE STOPPER'S PATH. -> [(move, why)] with the path it had named."""
    before = stopper_walk_oracle(board, stopper, targets, n)
    if before is None:
        return []
    pc = board.piece_at(stopper)
    name = f"{pc.symbol().upper()}{chess.square_name(stopper)}"
    out = []
    for m in legal:
        if m.from_square == sq or m.to_square == stopper:
            continue
        after = board.copy(stack=False)
        after.push(m)
        if after.piece_at(stopper) is None:
            continue
        if after.attacks_mask(stopper) & chess.BB_SQUARES[m.to_square]:          # the stopper can take the piece that denies: only a guarded piece denies a KING (it cannot take into attack), none denies another piece
            if not (after.piece_at(stopper).piece_type == chess.KING and after.is_attacked_by(me, m.to_square)):
                continue
        deny = denial_mask_oracle(after, me, stopper, sq)
        if stopper_walk_oracle(after, stopper, targets, n + 1, deny) is None:
            out.append((m, f"denies the path of their {name} ({' '.join(chess.square_name(x) for x in before[1])}): it cannot stand on a square of it in {n + 1} moves"))
    return out


_REACH_O: dict = {}


def reach_map_oracle(board: chess.Board, sq: int, k: int = R.REACH_K, job: "tuple | None" = None) -> dict:
    """plan 72.1 THE ROLL FORWARD OF ONE UNIT (the Python body before plan 72.7 step 3b, verbatim but for its memo, `_REACH_O` here)."""
    pc = board.piece_at(sq)
    if pc is None:
        return {}
    key = ("rmap", board.board_fen(), sq, k, job)
    hit = _REACH_O.get(key)
    if hit is not None:
        return hit
    own = pc.color
    start = G.with_turn(board, own)
    out: dict = {}
    frontier = [(start, [], sq, 0)]                                           # (board, path, the unit's square, 0): the roll is quiet, so the board differs from the start only by the unit's square
    seen = {(sq, 0): 0}
    for depth in range(1, k + 1):
        nxt = []
        for b, path, at, taken in frontier:
            for m in b.generate_legal_moves(from_mask=chess.BB_SQUARES[at]):      # the ONE unit's moves (the roll reads nothing of the other pieces' moves: generating all of them and filtering was the word's whole cost)
                if b.is_castling(m) or b.is_capture(m) or (m.promotion is not None and m.promotion != chess.QUEEN):
                    continue                                                 # QUIET moves only: against a pass a capture is nonsense (the bishop "stopped" I7k2H's c-pawn by eating it); a capture is a one-move word or the chase's
                a = b.copy(stack=False)
                a.push(m)
                if job is not None and depth == k and not ((job[0] in a.attacks(m.to_square)) if job[1] else m.to_square == job[0]):
                    continue                                                 # (the last step, not the job: nobody reads it)
                # what they win by capturing on the landing (a king lands only where it is legal). A landing none of theirs attacks has no capture to exchange (the static read first: the priced exchange only where there is one)
                price = 0 if pc.piece_type == chess.KING or not a.is_attacked_by(not own, m.to_square) else G.see_for(a, m.to_square, not own)
                cand = (price, depth)
                old = out.get(m.to_square)
                if old is None or cand < (old["material"], old["tempi"]):
                    out[m.to_square] = dict(material=price, tempi=depth, path=path + [m.uci()], board=a)
                if price > 0 or depth == k:
                    continue
                taken2 = 0
                sk = (m.to_square, taken2)
                if sk in seen and seen[sk] <= depth:
                    continue
                seen[sk] = depth
                nxt.append((G.with_turn(a, own), path + [m.uci()], m.to_square, taken2))
        frontier = nxt
        if not frontier:
            break
    _REACH_O[key] = out
    return out


def reach_price_oracle(board: chess.Board, sq: int, target: int, k: int = R.REACH_K, bearing: bool = True) -> dict | None:
    """plan 72.1 THE PRICED REACH (the Python body before plan 72.7 step 3b, verbatim, reading `reach_map_oracle`)."""
    pc = board.piece_at(sq)
    if pc is None or target == sq:
        return None
    reads = {sq, target}
    if bearing and target in G.with_turn(board, pc.color).attacks(sq):
        return dict(material=0, tempi=0, first=None, path=[], reads=frozenset(reads))
    best = None
    for at, e in reach_map_oracle(board, sq, k, job=(target, bearing)).items():
        if (at == target) if not bearing else (target in e["board"].attacks(at)):
            if best is None or (e["material"], e["tempi"]) < (best["material"], best["tempi"]):
                best = e
    if best is None:
        return None
    for u in best["path"]:
        m = chess.Move.from_uci(u)
        reads.add(m.to_square)
        reads.update(chess.SquareSet(chess.between(m.from_square, m.to_square)))
    if bearing:
        reads.update(chess.SquareSet(chess.between(chess.Move.from_uci(best["path"][-1]).to_square, target)))
    return dict(material=best["material"], tempi=best["tempi"], first=best["path"][0], path=list(best["path"]), reads=frozenset(reads))


def _board_sig(b: chess.Board) -> tuple:
    """Every field of a python-chess board a reader can see: the placement, the side, the raw castling rights, the e.p. square, the clocks, the promoted mask, the stack."""
    return (G.bbs(b), b.turn, b.castling_rights, b.ep_square, b.halfmove_clock, b.fullmove_number, b.promoted, [m.uci() for m in b.move_stack])


def _flipped(boards: list) -> list:
    """Boards whose side NOT to move is in check (python-chess holds them; shakmaty does not): each board with the turn passed, where the side to move was in check."""
    out = []
    for b in boards:
        if b.is_check():
            c = b.copy(stack=False)
            c.turn = not c.turn
            c.ep_square = None
            out.append(c)
    return out


def _roll_boards() -> list:
    bs = _puzzles(walk=1) + _games(n=8, plies=40)
    return bs + _flipped(bs)


# ---------------------------------------------------------------------------------------------------------------- 3b: the tests

def test_72_7_3b_reach_map_equals_its_oracle():
    """reach_map (Rust) against the Python body for every unit of both sides, k = 1..3, with and without a job: the same landings IN THE SAME ORDER, the same
    (material, tempi, path), the unit's attacks from its landing, and the replayed board equal to the oracle's board field by field (castling, e.p., clocks, stack)."""
    n = landings = 0
    for i, b in enumerate(_roll_boards()):
        G.clear_caches()
        _REACH_O.clear()
        for sq in chess.SquareSet(b.occupied):
            for k in (1, 2, 3) if i % 3 == 0 else (R.REACH_K,):
                jobs = [None] + ([(t, x) for t in (0, 27, 36, 63) for x in (True, False)] if i % 5 == 0 else [])
                for job in jobs:
                    want = reach_map_oracle(b, sq, k, job)
                    got = R.reach_map(b, sq, k, job)
                    assert list(got) == list(want), (b.fen(), sq, k, job)
                    for at in want:
                        w, g = want[at], got[at]
                        assert (g["material"], g["tempi"], g["path"]) == (w["material"], w["tempi"], w["path"]), (b.fen(), sq, k, job, at)
                        assert g["board"].attacks(at) == w["board"].attacks(at), (b.fen(), sq, at)
                        if landings % 7 == 0:
                            assert _board_sig(g["board"]._real()) == _board_sig(w["board"]), (b.fen(), sq, at)
                            assert g["board"].fen() == w["board"].fen() and g["board"].attackers(chess.WHITE, at) == w["board"].attackers(chess.WHITE, at)
                        landings += 1
                    n += 1
    print(f"OBS 72.7 3b reach_map against its oracle: rolls {n}, landings {landings}, every one equal (order, price, path, attacks, the replayed board)")


def test_72_7_3b_reach_price_equals_its_oracle():
    """reach_price (Rust) against the Python body: every unit of every fourth board, six seeded targets each, bearing and landing."""
    rng = random.Random(7276)
    n = found = 0
    for i, b in enumerate(_roll_boards()):
        if i % 4:
            continue
        G.clear_caches()
        _REACH_O.clear()
        for sq in chess.SquareSet(b.occupied):
            for target in rng.sample(range(64), 6):
                for bearing in (True, False):
                    want = reach_price_oracle(b, sq, target, R.REACH_K, bearing)
                    assert R.reach_price(b, sq, target, R.REACH_K, bearing) == want, (b.fen(), sq, target, bearing)
                    n += 1
                    found += want is not None
    assert found > 0
    print(f"OBS 72.7 3b reach_price against its oracle: asks {n}, priced {found}, every one equal (price, path, first, read set)")


def test_72_7_3b_stopper_walk_and_piece_moves_to_equal_their_oracles():
    from catspace.incubator.analyst.discovery import race as RC
    rng = random.Random(7275)
    n = 0
    for b in _roll_boards():
        for s in chess.SquareSet(b.occupied):
            for _ in range(3):
                targets = set(rng.sample(range(64), rng.randint(1, 8)))
                limit = rng.randint(0, 6)
                deny = rng.getrandbits(64) & rng.getrandbits(64)
                assert RC.stopper_walk(b, s, targets, limit, deny) == stopper_walk_oracle(b, s, targets, limit, deny), (b.fen(), s)
                assert RC.stopper_walk(b, s, targets, limit) == stopper_walk_oracle(b, s, targets, limit), (b.fen(), s)
                assert G.piece_moves_to(b, s, targets, limit) == piece_moves_to_oracle(b, s, targets, limit), (b.fen(), s)
                skip = rng.randrange(64)
                for me in chess.COLORS:
                    assert RC.denial_mask(b, me, s, skip) == denial_mask_oracle(b, me, s, skip), (b.fen(), s, skip)
                n += 1
    print(f"OBS 72.7 3b stopper_walk / piece_moves_to / denial_mask against their oracles: asks {n}, every one equal")


def test_72_7_3b_deny_moves_equals_its_oracle():
    """deny_moves (Rust, one call) against the Python body: for every pawn of the side to move and every piece of the other side as the stopper, the passer's path
    (or its promotion square) as the targets -- the same moves, in the same order, with the same text (castling and en passant pushed as python-chess pushes them)."""
    from catspace.incubator.analyst.discovery import race as RC
    n = denied = 0
    for b in _puzzles(walk=1) + _games(n=10, plies=40) + [chess.Board(f) for f in TI.CORNERS]:
        me = b.turn
        legal = list(b.legal_moves)
        for sq in chess.SquareSet(b.pawns & b.occupied_co[me]):
            k = abs((7 if me else 0) - chess.square_rank(sq))
            if k < 1:
                continue
            path = R.pawn_path(b, sq, k)
            for stopper in chess.SquareSet(b.occupied_co[not me]):
                for tg in (set(path), {path[-1]}):
                    want = deny_moves_oracle(b, me, sq, stopper, tg, k, legal)
                    got = RC.deny_moves(b, me, sq, stopper, tg, k, legal)
                    assert [(m.uci(), w) for m, w in got] == [(m.uci(), w) for m, w in want], (b.fen(), sq, stopper)
                    n += 1
                    denied += len(want)
    assert denied > 0
    print(f"OBS 72.7 3b deny_moves against its oracle: asks {n}, denying moves {denied}, every one equal")


def test_72_7_3b_the_roll_cache_is_cleared_by_the_hook_and_input_is_checked():
    b = chess.Board(DISCOVERY)
    G.clear_caches()
    assert AC.reach_cache_size() == 0
    R.reach_map(b, chess.E4, 3)
    R.reach_price(b, chess.E4, chess.A8, 3)
    assert AC.reach_cache_size() == 2
    G.clear_caches()                                   # the CLEAR_HOOKS: regress's _REACH and the Rust memo together
    assert AC.reach_cache_size() == 0 and not R._REACH
    with pytest.raises(ValueError):
        AC.reach_map_bb((1, 1, 0, 0, 0, 0, 1, 0), 0, 3)          # a square in two piece boards
    with pytest.raises(ValueError):
        AC.reach_map_bb(G.bbs(b), 64, 3)
    with pytest.raises(ValueError):
        AC.stopper_walk_bb(G.bbs(b), 70, 1, 3)
    with pytest.raises(ValueError):
        AC.reach_price_bb(G.bbs(b), 0, 99, 3)
    AC.dep_clear()
    R.reach_map(b, chess.C3, 3)
    R.reach_price(b, chess.C3, chess.A8, 3)
    G.piece_moves_to(b, chess.C3, {chess.A8}, 3)
    assert AC.dep_peek() == (0, False, [0, 0, 0], 0)           # the roll writes nothing on the dependency trackers (the Python words wrote nothing there either)


# ---------------------------------------------------------------------------------------------------------------- 3c: the oracles (the Python bodies before the port, verbatim; each reads the other oracles)

def threat_value_oracle(board_after: chess.Board, us) -> int:
    """After our move (their turn) and a pass: the most we would win by the exchange on a piece of theirs (the Rust `see`, us on move)."""
    me = G.colour_of(us)
    c = G.with_turn(board_after, me)
    best = 0
    for s in chess.SQUARES:
        p = c.piece_at(s)
        if p and p.color != me and p.piece_type != chess.KING and c.is_attacked_by(me, s):
            best = max(best, G.see(c, s))
    return best


def is_passer_oracle(board: chess.Board, sq: int) -> bool:
    """plan 72.2 (a): a PASSED pawn -- no pawn of the other side ahead of it on its file or the adjacent files (the Rust region word `passer`, read directly)."""
    pc = board.piece_at(sq)
    if pc is None or pc.piece_type != chess.PAWN:
        return False
    f, r = chess.square_file(sq), chess.square_rank(sq)
    for s in chess.SquareSet(board.pawns & board.occupied_co[not pc.color]):
        if abs(chess.square_file(s) - f) <= 1 and ((chess.square_rank(s) > r) if pc.color else (chess.square_rank(s) < r)):
            return False
    return True


def promote_defects_oracle(board: chess.Board, sq: int, lift: bool = False, priced: "bool | None" = None) -> tuple:
    """What stands between the pawn on `sq` (any colour) and a queen (the Python body before plan 72.7 step 3c, verbatim, reading the 3b oracles)."""
    from catspace.incubator.analyst.discovery import race as RC
    pc = board.piece_at(sq)
    own = pc.color
    kv = None
    priced = R.PRICED_STOPPER if priced is None else priced
    if lift and RC.LIFT and board.turn == own and (board.occupied_co[not own] & (board.rooks | board.bishops | board.queens | board.knights)):
        kv = RC.king_verdict(board, sq)
    n = G.word(board, ("pushes", G.name(sq)), own)
    path = R.pawn_path(board, sq, n)
    promo = path[-1]
    facts = dict(pushes=n, path=path, promo=promo, king_dist=None, king_sq=board.king(not own))
    out = []
    lifted = []
    priced_notes = []
    ahead_ours = G.word(board, ("count", ("and", ("ahead", G.name(sq)), "ours")), own)
    ahead_theirs = G.word(board, ("count", ("and", ("ahead", G.name(sq)), "theirs")), own)
    if ahead_ours + ahead_theirs:
        for t in path:
            b_ = board.piece_at(t)
            if b_ is not None:
                out.append(R.Defect("blocked", t, [t], f"{'our' if b_.color == own else 'their'} {b_.symbol().upper()} stands on {G.name(t)}"))
    for t in path:
        a = G.word(board, ("defenders", G.name(t)), own)
        d = G.word(board, ("attackers", G.name(t)), own)
        if a > d:
            kind = "promo square attacked" if t == promo else "path attacked"
            by = [x for x in G.attackers_of(board, t, not own)]
            if kv is not None and not kv[0] and set(by) <= {board.king(not own)}:
                lifted.append(f"{G.name(t)} is attacked only by their king, which is no stopper ({kv[1]})")
                continue
            out.append(R.Defect(kind, t, by, f"{G.name(t)} is attacked by them {a}x and guarded by us {d}x"))
    ek = facts["king_sq"]
    if kv is not None:
        facts["lifted"] = lifted
        if kv[0]:
            out.append(R.Defect("stopper", promo, [ek], f"their K stops it: {kv[1]}"))
        else:
            lifted.append(f"their K is no stopper: {kv[1]}")
        facts["king_dist"] = G.word(board, ("dist", "L", ("promo", G.name(sq))), own)
        return out, facts
    if ek is not None:
        kd = G.word(board, ("dist", "L", ("promo", G.name(sq))), own)
        facts["king_dist"] = kd
        if kd <= n and ek not in path:
            w = stopper_walk_oracle(board, ek, {promo}, n)
            out.append(R.Defect("stopper", promo, [ek], f"their K on {G.name(ek)} reaches {G.name(promo)} in {kd}, the pawn needs {n}" + (f" (path {' '.join(G.name(x) for x in w[1])})" if w else "")))
    for s in chess.SQUARES:
        x = board.piece_at(s)
        if x is None or x.color == own or x.piece_type in (chess.KING, chess.PAWN):
            continue
        k = piece_moves_to_oracle(board, s, set(path), n)
        if k is not None:
            if priced:
                pr = []
                nb = G.with_turn(board, not own)
                for t in path:
                    if t in nb.attacks(s) and len(board.attackers(not own, t)) > len(board.attackers(own, t)):
                        pr.append(dict(material=0, tempi=0))
                if not pr:
                    mp = reach_map_oracle(board, s, min(n, R.REACH_K))
                    for at, e in mp.items():
                        if e["material"] != 0:
                            continue
                        cf = e["board"]
                        for t in path:
                            if t in cf.attacks(at) and len(cf.attackers(not own, t)) > len(cf.attackers(own, t)):
                                pr.append(dict(material=0, tempi=e["tempi"]))
                                break
                if not pr:
                    priced_notes.append(f"their {x.symbol().upper()} on {G.name(s)} reaches the path only at a price (taken where it lands): no stopper")
                    continue
                k = min(r["tempi"] for r in pr)
            w = stopper_walk_oracle(board, s, set(path), n)
            out.append(R.Defect("stopper", promo, [s], f"their {x.symbol().upper()} on {G.name(s)} reaches the path in {k} ({'free by the exchange' if priced else 'static board'}), the pawn needs {n}" + (f" (path {' '.join(G.name(x) for x in w[1])})" if w else "")))
    if priced_notes:
        facts["priced"] = priced_notes
    return out, facts


_PD_O: dict = {}


def promote_defects_cached_oracle(board: chess.Board, sq: int, priced: "bool | None" = None) -> list:
    key = (board.board_fen(), board.turn, sq, priced)
    hit = _PD_O.get(key)
    if hit is None:
        hit = _PD_O[key] = promote_defects_oracle(board, sq, priced=priced)[0]
    return hit


def iter_safe_moves_oracle(board: chess.Board, sq: int):
    pc = board.piece_at(sq)
    p = G.with_turn(board, pc.color)
    for m in p.legal_moves:
        if m.from_square != sq or p.is_castling(m):
            continue
        a = p.copy(stack=False)
        a.push(m)
        if G.see_for(a, m.to_square, not pc.color) == 0:
            yield m


def duty_after_oracle(board: chess.Board, r: chess.Move, us, ctx: int | None = None, need: int | None = None, passer: bool = False, mate: bool = True) -> tuple:
    """plan 63.2 duty_after (the Python body before plan 72.7 step 3c, verbatim; the mate branch is not ported and not compared: `mate` must be False)."""
    assert not mate
    me = G.colour_of(us)
    gain = 1 if board.is_en_passant(r) else (G.worth(board.piece_at(r.to_square)) if board.is_capture(r) else 0)
    a = board.copy(stack=False)
    a.push(r)
    if a.is_check():
        return 0, "check"
    now_ex = ctx if ctx is not None else threat_value_oracle(G.with_turn(board, me), me)
    v = max(0, threat_value_oracle(a, me) - gain - now_ex)
    if need is not None and v >= need:
        return v, f"the exchange wins {v}"
    if passer:
        held = board.attacks(r.from_square) | chess.SquareSet([r.from_square])
        for pw in chess.SquareSet(board.pawns & board.occupied_co[me]):
            if a.piece_at(pw) is None or not is_passer_oracle(board, pw):
                continue
            n_ = R._pushes_of(board, pw, me)
            if n_ < 1 or not (held & chess.SquareSet(R.pawn_path(board, pw, n_))):
                continue
            st = promote_defects_cached_oracle(a, pw, priced=False)
            ek = a.king(not me)
            if any(d.kind != "stopper" or d.by == [ek] for d in st):
                continue
            if st and promote_defects_cached_oracle(a, pw):
                continue
            return R.STAKE_PROMOTE, f"frees our passer on {G.name(pw)} (unstoppable by the tempo arithmetic after it)"
    return v, f"the exchange wins {v}" if v else "nothing"


_DUTY_O: dict = {}


def can_run_oracle(board: chess.Board, sq: int, us, passer: bool = False) -> tuple:
    pc = board.piece_at(sq)
    if pc.piece_type == chess.KING:
        return True, []
    key = ("run", board.board_fen(), board.castling_rights, sq, G.colour_of(us), passer)
    hit = _DUTY_O.get(key)
    if hit is not None:
        return hit
    me = G.colour_of(us)
    p = G.with_turn(board, pc.color)
    now = G.with_turn(board, me)
    ctx = threat_value_oracle(now, me)
    cut = []
    out = None
    for m in iter_safe_moves_oracle(board, sq):
        v, what = duty_after_oracle(p, m, us, ctx, need=G.worth(pc), passer=passer, mate=not passer)
        if v >= G.worth(pc):
            cut.append((p.san(m), v, what))
        else:
            out = (True, [])
            break
    if out is None:
        out = (False, cut)
    _DUTY_O[key] = out
    return out


def stuck_threats_oracle(board: chess.Board, us, k: int = R.REACH_K) -> list:
    """plan 72.2 (b) THE UNIT THAT CANNOT RUN (the Python body before plan 72.7 step 3c, verbatim, reading the oracles)."""
    me = G.colour_of(us)
    them = not me
    out = []
    ours = [s for s in chess.SQUARES if (pc := board.piece_at(s)) is not None and pc.color == me and pc.piece_type not in (chess.KING, chess.PAWN) and not board.is_attacked_by(them, s)]
    if not ours:
        return out
    theirs = G.with_turn(board, them)
    their_moves = None
    maps: dict | None = None
    for s in ours:
        pc = board.piece_at(s)
        worth = G.worth(pc)
        best = None
        if can_run_oracle(board, s, them, passer=True)[0]:
            if their_moves is None:
                their_moves = [m for m in theirs.legal_moves if not theirs.is_castling(m)]
            for m in their_moves:
                a = theirs.copy(stack=False)
                a.push(m)
                if a.is_check() or s not in a.attacks(m.to_square) or a.piece_at(s) is None:
                    continue
                price = 0 if theirs.piece_at(m.from_square).piece_type == chess.KING else G.see_for(a, m.to_square, me)
                if price >= worth:
                    continue
                after = G.with_turn(a, me)
                if can_run_oracle(after, s, them, passer=True)[0]:
                    continue
                e = dict(material=price, tempi=1, path=[m.uci()], board=a)
                if best is None or (price, 1) < (best[1]["material"], best[1]["tempi"]):
                    best = (m.from_square, e)
        else:
            if maps is None:
                maps = {a: reach_map_oracle(board, a, k) for a in chess.SQUARES if (ap := board.piece_at(a)) is not None and ap.color == them}
            for a, mp in maps.items():
                for at, e in mp.items():
                    if e["material"] >= worth or s not in e["board"].attacks(at):
                        continue
                    if best is None or (e["material"], e["tempi"]) < (best[1]["material"], best[1]["tempi"]):
                        best = (a, e)
        if best is None:
            continue
        a, e = best
        parry = None
        for g in chess.SQUARES:
            gp = board.piece_at(g)
            if gp is None or gp.color != me or g == s:
                continue
            if s in board.attacks(g):
                parry = f"our {gp.symbol().upper()} on {G.name(g)} guards it already"
                break
            if e["tempi"] > 1:
                pr = reach_price_oracle(board, g, s, e["tempi"] - 1)
                if pr is not None and pr["material"] == 0 and pr["tempi"] <= e["tempi"] - 1:
                    parry = f"our {gp.symbol().upper()} on {G.name(g)} guards it in {pr['tempi']} ({board.san(chess.Move.from_uci(pr['first']))} first)"
                    break
        if parry is None and e["tempi"] > 1:
            for m in board.legal_moves:
                if board.is_capture(m) or board.gives_check(m):
                    continue
                c = board.copy(stack=False)
                c.push(m)
                c = G.with_turn(c, me)
                if c.piece_at(s) is not None and can_run_oracle(c, s, them, passer=True)[0]:
                    parry = f"after {board.san(m)} it can run"
                    break
        reads = {chess.Move.from_uci(u).to_square for u in e["path"]} | {s, a}
        out.append(dict(sq=s, worth=worth, attacker=a, price=e["material"], tempi=e["tempi"], first=e["path"][0], path=list(e["path"]), parry=parry, reads=frozenset(reads)))
    out.sort(key=lambda d: -(d["worth"] - d["price"]))
    return out


def _clear_oracles():
    G.clear_caches()
    for c in (_REACH_O, _PD_O, _DUTY_O):
        c.clear()


def _defs(ds: list) -> list:
    return [(d.kind, d.sq, list(d.by), d.why) for d in ds]


# ---------------------------------------------------------------------------------------------------------------- 3c: the tests

def test_72_7_3c_py_legal_is_python_chess_s_legal_moves_with_their_order():
    """`py_legal` (words.rs) against python-chess's `legal_moves` (castling dropped; the e.p. square cleared, as `with_turn` clears it) on the games, the puzzles, the corners
    and the boards whose side NOT to move is in check (python-chess generates there; shakmaty does not hold them): the same moves in the same order."""
    n = flipped = 0
    for b in _roll_boards() + [chess.Board(f) for f in TI.CORNERS]:
        for side in (chess.WHITE, chess.BLACK):
            c = G.with_turn(b, side)
            want = [m.uci() for m in c.legal_moves if not c.is_castling(m)]
            assert AC.py_legal_bb(G.bbs(c), side) == want, (c.fen(), side)
            n += 1
            flipped += c.was_into_check()
    assert flipped > 0
    print(f"OBS 72.7 3c py_legal against python-chess: boards {n} (of them with the side not to move in check: {flipped}), every move list equal in order")


def test_72_7_3c_promote_defects_equals_its_oracle():
    """promote_defects (Rust) against the Python body for every pawn of both sides, priced and static, and with the lift (the king verdict read in Python): the same
    defects in the same order with the same text, the same facts (keys in order)."""
    n = stoppers = 0
    for i, b in enumerate(_roll_boards()):
        _clear_oracles()
        for sq in chess.SquareSet(b.pawns):
            own = b.piece_at(sq).color
            if not (0 < chess.square_rank(sq) < 7) or G.pos_for(b, own) is None:
                continue
            for priced in (True, False):
                for lift in ((False, True) if i % 2 == 0 else (False,)):
                    want = promote_defects_oracle(b, sq, lift=lift, priced=priced)
                    got = R.promote_defects(b, sq, lift=lift, priced=priced)
                    assert _defs(got[0]) == _defs(want[0]), (b.fen(), sq, priced, lift)
                    assert list(got[1].items()) == list(want[1].items()), (b.fen(), sq, priced, lift)
                    n += 1
                    stoppers += sum(1 for d in want[0] if d.kind == "stopper")
    assert stoppers > 0
    print(f"OBS 72.7 3c promote_defects against its oracle: asks {n}, stopper defects {stoppers}, every one equal (defects, order, text, facts)")


def test_72_7_3c_can_run_and_stuck_threats_equal_their_oracles():
    """can_run (the passer duty, as stuck_threats asks it) and stuck_threats (Rust) against the Python bodies on the puzzles' and the games' positions, us = the side to
    move: the same answers, the same rows in the same order (the parry's text, the read set)."""
    n = rows = runs = raised = 0
    for b in _puzzles(walk=1) + _games(n=8, plies=40):
        if b.is_check() and not any(True for _ in b.legal_moves):
            continue
        _clear_oracles()
        me = b.turn
        for s in chess.SquareSet(b.occupied_co[me] & ~b.kings & ~b.pawns):
            assert AC.can_run_bb(G.bbs(b), s, not me) == can_run_oracle(b, s, not me, passer=True)[0], (b.fen(), s)
            runs += 1
        try:                                          # FINDING (3c): with our king in check, python-chess lets them TAKE it on the board given to them; the duty then reads promote_defects
            want = stuck_threats_oracle(b, me)        # on a board with no king of ours, where graph.word reads None -- the Python word raises (TypeError); the port raises there too (ValueError)
        except TypeError:
            want = "raises"
        try:
            got = R.stuck_threats(b, me)
        except ValueError:
            got = "raises"
        assert got == want, b.fen()
        n += 1
        raised += want == "raises"
        rows += len(want) if want != "raises" else 0
    assert rows > 0
    print(f"OBS 72.7 3c stuck_threats against its oracle: positions {n}, rows {rows}, can_run asks {runs}, every one equal; both raise on {raised}")


def test_72_7_3c_the_passer_caches_are_cleared_by_the_hook():
    b = chess.Board(POS2)
    G.clear_caches()
    R.stuck_threats(b, b.turn)
    G.clear_caches()
    assert not R._PD and not R._DUTY
    AC.dep_clear()
    R.stuck_threats(b, b.turn)
    R.promote_defects(b, chess.G6)
    assert AC.dep_peek() == (0, False, [0, 0, 0], 0)
    with pytest.raises(ValueError):
        AC.promote_defects_bb(G.bbs(b), 64)
    with pytest.raises(ValueError):
        AC.can_run_bb(G.bbs(b), 64, True)


# ---------------------------------------------------------------------------------------------------------------- 3d: the oracles (the Python bodies before the port, verbatim; race.py's helpers they read stay in race.py for race_claim)

def _king_holds_oracle(board: chess.Board, me: bool, sq: int, with_second: bool) -> tuple | None:
    from catspace.incubator.analyst.discovery import race as RC
    ek, ok = board.king(not me), board.king(me)
    n = RC._pushes(me, sq)
    if n < 1 or n > 6:
        return None
    path = RC._path(me, sq, n)
    if any(board.piece_at(t) is not None for t in path):
        return None
    stable = RC._stable_attacks(board, me, ek, ok, n, {sq: path})
    why = RC._stage_held(me, ek, path, sq, stable, stable, n)
    if why is None:
        return False, "their king cannot stop it: no stage of the walk is reached in time (squares it cannot stand on denied)"
    if with_second:
        d = RC._deflect(board, me, sq, path, n, ek, ok, why, verdict_only=True)
        if d:
            return False, f"{d}; the king held it alone: {why}"
    return True, why


def king_verdict_oracle(board: chess.Board, sq: int) -> tuple | None:
    """62.1c: THEIR KING AS A STOPPER of our pawn on `sq` (the owner to move), judged by the proved analysis only (Race.deny_sound, race_sound restricted to the king): the stage test with the squares it cannot stand
    and survive on denied, then the deflection by a second passer (this pawn as the one that walks, or as the one that rests). -> (held, why): held=False = the king is no stopper (the verdict may not kill the
    question for it); None = not applicable (not our pawn, path not clear, in check). Their OTHER pieces are not read here at all: they are replies, and the chase reads every reply (Race.lifted_is_gap)."""
    me = board.turn
    pc = board.piece_at(sq)
    if pc is None or pc.piece_type != chess.PAWN or pc.color != me or board.is_check():
        return None
    if board.king(not me) is None or board.king(me) is None:
        return None
    first = _king_holds_oracle(board, me, sq, True)
    if first is None or not first[0]:
        return first
    for q in chess.SquareSet(board.pawns & board.occupied_co[me]):
        if q == sq:
            continue
        o = _king_holds_oracle(board, me, q, True)
        if o is not None and not o[0]:
            return False, f"the passer on {chess.square_name(q)} is not held by the king ({o[1]}); the king that holds {chess.square_name(sq)} cannot also stop it"
    return first


def support_moves_oracle(board: chess.Board, me: bool, sq: int, legal: list) -> list:
    """62.1c: our moves (not the pawn's own) after which their king is NO LONGER a stopper of the pawn on `sq` by `king_verdict` (we have the move again: the pass reading): a cover (Kg3 takes h2 and g2 from
    the king), a second passer's support. Each is a ROUTE of the promote question; the chase reads every reply to it. -> [(move, why)]."""
    before = king_verdict_oracle(board, sq)
    if before is None or not before[0]:
        return []
    out = []
    for m in legal:
        if m.from_square == sq:
            continue
        after = board.copy(stack=False)
        after.push(m)
        if after.is_check():
            continue
        after.turn = me
        after.ep_square = None
        v = king_verdict_oracle(after, sq)
        if v is not None and not v[0]:
            out.append((m, f"makes their king no stopper: {v[1]}"))
    return out


def _race_boards(n: int = 400, seed: int = 7277) -> list:
    """race.random_race_position's endings (our king and pawns against a king and pawns, sometimes a rook of ours), with the side to move either way."""
    from catspace.incubator.analyst.discovery import race as RC
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = RC.random_race_position(rng, rook=rng.random() < 0.3)
        if b is not None:
            out.append(b)
            c = b.copy(stack=False)
            c.turn = not c.turn
            if c.is_valid():
                out.append(c)
    return out


# ---------------------------------------------------------------------------------------------------------------- 3d: the tests

def test_72_7_3d_king_verdict_and_support_moves_equal_their_oracles():
    """race.king_verdict / support_moves (Rust) against the Python bodies for every pawn of the side to move on the race endings, the puzzles and the games: the same
    verdict and text, the same supporting moves in the same order."""
    from catspace.incubator.analyst.discovery import race as RC
    n = held = free = sup = 0
    for b in _race_boards() + _puzzles(walk=1) + _games(n=6, plies=40):
        legal = list(b.legal_moves)
        for sq in chess.SquareSet(b.pawns):
            want = king_verdict_oracle(b, sq)
            assert RC.king_verdict(b, sq) == want, (b.fen(), sq)
            n += 1
            held += bool(want and want[0])
            free += bool(want is not None and not want[0])
            if want is not None and want[0]:
                ws = support_moves_oracle(b, b.turn, sq, legal)
                gs = RC.support_moves(b, b.turn, sq, legal)
                assert [(m.uci(), w) for m, w in gs] == [(m.uci(), w) for m, w in ws], (b.fen(), sq)
                sup += len(ws)
    assert held and free and sup
    print(f"OBS 72.7 3d king_verdict / support_moves against their oracles: pawns {n}, held {held}, no stopper {free}, supporting moves {sup}, every one equal")
