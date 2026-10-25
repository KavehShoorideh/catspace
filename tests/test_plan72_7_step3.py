"""plan 72.7 THE RUST BOUNDARY, step 3 (docs/RUST_BOUNDARY.md section 4 step 3): THE PYTHON WORDS IN RUST. Each ported word is compared with its Python body --
kept HERE, verbatim, as the TEST ORACLE (the engine's module now calls the Rust word) -- on seeded random games, on the guard's positions and on puzzle positions:
values, structures, orders, and what each wrote on the dependency trackers (RUST_BOUNDARY risk 2: `dep_peek` before and after); illegal input is a ValueError
(risk 7); a cache a word keeps is emptied by `graph.clear_caches` (the CLEAR_HOOKS).
3a THE MATE IN TWO: `selfmine.bind_mate2` (Lean `Ideas.Mined.Schemas.mate_in_two`) and `regress.mate2_threat` (plan 59 tier 2) -- words.rs `bind_mate2_pos`,
`mate2_threat_pos`.
3b THE ROLL AND THE STATIC WALKS: `regress.reach_map` / `reach_price` (plan 72.1), `race.stopper_walk` / `deny_moves` (62.1b), `graph.piece_moves_to` -- words.rs
`reach_map_bb`, `reach_price_bb`, `stopper_walk_bb`, `deny_moves_bb`, `piece_moves_to_bb`; boards whose side NOT to move is in check (python-chess holds them, the roll reaches them
after a quiet check) among the positions."""
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
