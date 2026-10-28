"""PLAN 75.4 THE KERNEL'S RULE GAPS (Kaveh 2026-10-28: "item number two is a no-brainer"): the rules of the game read through the kernel, never re-written in the agenda.

(1) THE TERMINAL RULES IN THE KERNEL: `Game.terminal` reads the game record and the clock -- the third occurrence (counted by the kernel's identity: board, side to move,
    castling rights, a LEGAL en-passant square; never the raw FEN's fields) and the fifty-move clock at 100 with no mate on the move (Lean `Ideas.Game.Record.threefold`,
    `Ideas.Game.chess_threefold_iff`, `Ideas.Game.fifty`, `Ideas.Game.chess_fifty_iff`); the record keeps its count invariant along the game (`Ideas.Game.Record.step_sound`);
    the chase reads the rule at the positions of its line (the record before them: the game's, the root, the line's).
(2) THE CYCLE CLAIM RE-LABELLED AS WHAT IT IS: one repeat of a position of the line is a derived lemma ("repetition (a forced cycle: the threefold rule is reachable)"), a
    CONJECTURE in the store; its strict form is proved generically in Lean (`Ideas.Game.forced_cycle_threefold`), the chase's own reading is not.
(3) THE CHASE CLOCK READS THE KERNEL'S EVENTS: `Game.clock_events` (a capture, a pawn move: what resets the halfmove clock, Lean `Chess.applyMove_halfmove`) in place of the
    tests written in the agenda; the clock reads them exactly as before.
(4) THE POSITION KEY IS THE KERNEL'S: `agenda._pos_key` is the kernel's identity of the position a FEN names (a legal e.p. square only), not the FEN's first four fields.
(5) PROMOTION FROM THE KERNEL: the pushes the words build are the kernel's moves (`Game.moves_to`: every unit a promotion may place), the unit chosen by the store's
    valuation (`Game.placing_most`), not "always a queen"."""
import random

import chess
import pytest

import analyst_core as AC

import catspace.incubator.analyst  # noqa: F401
from catspace.incubator.analyst.discovery import agenda as AG
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import game as GM
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import names as NM
from catspace.incubator.analyst.discovery.play_server import game_history
from catspace.incubator.analyst.discovery.state import state_of

G = GM.CHESS
PERPETUAL = "1Q6/2R2pp1/5k1p/pB6/P1pb4/6PP/5r2/7K b - - 0 41"     # (test_cost61) lichess w91Hv: black a rook for a queen down, Rf1+ Kg2 Rf2+ Kh1 is a perpetual
PSEUDO_EP = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"      # after 1.e4 e5 as a FEN may write it: e6 named, no white pawn can take there
LEGAL_EP = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2"                     # exd6 e.p. is legal: the square is part of the identity


@pytest.fixture(autouse=True)
def _setup():
    BV.install_terms()


def _walk(seed: int, plies: int, fen: str = chess.STARTING_FEN) -> chess.Board:
    rng = random.Random(seed)
    b = chess.Board(fen)
    for _ in range(plies):
        ms = list(b.legal_moves)
        if not ms:
            break
        b.push(rng.choice(ms))
    return b


# ---------------------------------------------------------------------------------------------------------------- (1) the terminal rules in the kernel

def test_75_4_1_the_identity_is_the_kernels_a_pseudo_legal_en_passant_square_is_dropped_a_legal_one_kept():
    raw = chess.Board(PSEUDO_EP)
    assert raw.ep_square == chess.E6 and not raw.has_legal_en_passant()
    assert G.position_key(raw) == " ".join(PSEUDO_EP.split()[:3] + ["-"])                 # the identity drops e6 (Lean Chess.effectiveEnPassant)
    assert G.position_key(raw) == G.position_key(chess.Board(PSEUDO_EP.replace(" e6 ", " - ")))
    legal = chess.Board(LEGAL_EP)
    assert legal.has_legal_en_passant() and G.position_key(legal).endswith(" d6")
    assert G.position_key(legal) != G.position_key(chess.Board(LEGAL_EP.replace(" d6 ", " - ")))   # a legal capture is part of the position
    assert G.position_key(chess.Board(PERPETUAL)) == G.position_key(chess.Board(PERPETUAL.replace(" 0 41", " 7 60")))      # the clocks are not


def test_75_4_1_threefold_is_the_third_occurrence_counted_by_identity_not_by_the_raw_fen():
    s = chess.Board(PSEUDO_EP)
    same = chess.Board(PSEUDO_EP.replace(" e6 ", " - ").replace(" 0 2", " 4 4"))            # the same position written without the square, other clocks
    assert G.terminal(s) is None and G.terminal(s, GM.Record()) is None
    before = GM.Record().push(G, same)
    assert G.terminal(s, before) is None                                                     # the second occurrence: no rule yet (the cycle claim's case, a derived lemma)
    before.push(G, same)
    assert G.terminal(s, before) == "repetition"                                             # the third, though the raw FEN fields differ
    assert G.terminal(s, GM.Record({" ".join(PSEUDO_EP.split()[:4]): 2})) is None            # a record keyed by the raw fields counts another position: never read
    legal, gone = chess.Board(LEGAL_EP), chess.Board(LEGAL_EP.replace(" d6 ", " - "))
    assert G.terminal(gone, GM.Record().push(G, legal).push(G, legal)) is None                # the e.p. right differs: a different position


def test_75_4_1_the_record_built_by_the_game_is_python_chess_repetition_on_random_games():
    """The record (`game_history`: one push per move, Lean Record.step_sound) read by `Game.terminal` against python-chess's own `Board.is_repetition(3)` over its move stack
    (its transposition key is the kernel's identity), on knight-shuffle games and on random walks."""
    agree = third = 0
    games = []
    for sans in ("Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8 Nc3",                                        # the start position a third time after 8 plies
                 "e4 e5 Nf3 Nc6 Ng1 Nb8 Nf3 Nc6 Ng1 Nb8 Ke2",                                  # the position after 1.e4 e5 (its e.p. square e6 no pawn can use) a third time
                 "e4 Nf6 e5 d5 Nf3 Nc6 Ng1 Nb8 Nf3 Nc6 Ng1 Nb8 Nf3"):                          # after 3...d5 exd6 e.p. is LEGAL: the later positions without the right are others
        b = chess.Board()
        for x in sans.split():
            b.push_san(x)
        games.append(b)
    for seed in range(12):
        b = chess.Board()
        rng = random.Random(seed)
        for _ in range(16):                                                                   # shuffles: repetitions may come
            ms = [m for m in b.legal_moves if b.piece_type_at(m.from_square) in (chess.KNIGHT, chess.KING)] or list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
        games.append(b)
    games += [_walk(s, 40) for s in range(20)]
    for g in games:
        b = chess.Board()
        for mv in g.move_stack:
            b.push(mv)
            rec = game_history(b)
            assert rec.total() == len(b.move_stack)
            ours = G.terminal(b, rec) == "repetition"
            theirs = b.is_repetition(3) and not b.is_checkmate() and not b.is_stalemate() and not b.is_insufficient_material()
            assert ours == theirs, (b.fen(), [m.uci() for m in b.move_stack])
            agree += 1
            third += ours
        line = _line(g)[:-1]
        rec = game_history(g)
        assert all(rec.count(k) == sum(1 for x in line if G.position_key(x) == k) for k in rec)      # the count invariant (Lean Record.Sound)
    print(f"OBS 75.4 (1) record vs python-chess is_repetition(3): {agree} positions agree, {third} third occurrences")
    assert third >= 2


def _line(b: chess.Board) -> list:
    r = b.root()
    out = [r.copy()]
    for mv in b.move_stack:
        r.push(mv)
        out.append(r.copy(stack=False))
    return out


def test_75_4_1_fifty_moves_at_a_hundred_half_moves_and_never_on_a_mate():
    quiet = "8/8/4k3/8/8/3K4/8/R7 w - - {hm} 120"
    assert G.terminal(chess.Board(quiet.format(hm=99))) is None
    assert G.terminal(chess.Board(quiet.format(hm=100))) == "fifty-move"
    assert G.terminal(chess.Board(quiet.format(hm=150))) == "fifty-move"
    mated = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 99 80")
    mated.push_san("Ra8#")
    assert mated.halfmove_clock == 100 and G.terminal(mated) == "mate"                        # a mate on the move comes first (Lean Record.mated_not_draw)
    stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 100 90")
    assert G.terminal(stale) == "stalemate"
    for f in (quiet.format(hm=99), quiet.format(hm=100), PERPETUAL, chess.STARTING_FEN):
        assert G.key_clock(f) == G.clock(G.from_key(f))
    b = chess.Board(quiet.format(hm=98))
    b.push_san("Ra2")
    assert G.terminal(b) is None and b.halfmove_clock == 99
    b.push_san("Kd5")
    assert G.terminal(b) == "fifty-move"


def test_75_4_1_the_record_keeps_its_count_invariant_along_every_step():
    """Lean `Ideas.Game.Record.push_sound` / `step_sound` mirrored: pushing each position of a line gives every identity the count of its occurrences."""
    for seed in range(10):
        b = _walk(100 + seed, 60)
        line = _line(b)
        rec = GM.Record.of_line(G, line)
        for k in {G.position_key(x) for x in line}:
            assert rec.count(k) == sum(1 for x in line if G.position_key(x) == k)
        assert rec.total() == len(line)


def test_75_4_1_the_names_table_and_the_store_bind_the_two_draws_to_the_rule():
    idx = LS.lean_index()
    assert NM.problems(idx) == []
    for rule, word in NM.DRAW_BY.items():
        assert rule in GM.DRAWS_BY_RULE and word in NM.NAMES and NM.NAMES[word].lean in idx
        e = LS.load()[word]
        assert e["kind"] == "CLOSING" and e["origin"] == "BASE" and e["status"] == "PROVED" and all(n in idx for n in e["lean"])
        assert LS.says(word, "draw") and LS.says(word, "terminal")
    assert not LS.says("repetition", "terminal")                                             # the cycle claim is no rule


def test_75_4_1_the_chase_reads_the_rule_at_a_position_of_its_line():
    """PERPETUAL repeats the root after Rf1+ Kg2 Rf2+ Kh1 (test_cost61): with the game record holding the root once more, the repeat is the THIRD occurrence -- the kernel's
    draw (a terminal answer, the rule's word), not the cycle claim; without it, the cycle claim (the second occurrence) as before."""
    ag = AG.Agenda(state_of(PERPETUAL, "b"))
    ag._route_move, ag._reply_san = chess.Move.from_uci("f2f1"), "Kg2"
    assert ag._inner(PERPETUAL, 1)["claim"].lemma.startswith("repetition")                 # the second occurrence: the derived cycle claim
    ag = AG.Agenda(state_of(PERPETUAL, "b"))
    ag.ctx.history = GM.Record().push(G, chess.Board(PERPETUAL))                             # the game saw the root once before it
    ag._route_move, ag._reply_san = chess.Move.from_uci("f2f1"), "Kg2"
    out = ag._inner(PERPETUAL, 1)
    assert out["terminal"] == "repetition" and out["claim"].lemma == "draw by repetition" and out["claim"].kind == "draw" and out["value"] == 7
    assert ag.ctx.rule_draws == 1 and any(x.startswith("    closed: draw by repetition (the position after") for x in ag.ctx.c_lines)
    ahead = "1Q6/2R2pp1/5k1p/pB6/P1pb4/6PP/5r2/7K w - - 0 41"                                # white to move, a queen for a rook up: the draw takes the lead away
    ag = AG.Agenda(state_of(ahead, "w"))
    ag.ctx.history = GM.Record().push(G, chess.Board(ahead))
    out = ag._inner(ahead, 1)
    assert out["terminal"] == "repetition" and out["value"] == -7                            # (the cycle claim gives nothing ahead; the rule's draw costs the lead)


def test_75_4_1_the_root_and_an_open_position_of_the_line_make_a_repeat_the_third_occurrence():
    """test_regress's 52.16 cycle position: the root holds the identity and an open position of the line holds it again -- one more is the third occurrence (the move
    counters are no part of the identity), the rule's draw, read before the cycle."""
    a = "2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 10 64"
    ag = AG.Agenda(state_of(a, "w"))
    ag.ctx.left = 1000
    ag.ctx.active.add(AG._pos_key(a))
    r = ag._inner("2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 14 66", 1)
    assert r["terminal"] == "repetition" and r["claim"].lemma == "draw by repetition" and r["asked"] == 0


def test_75_4_1_a_route_into_a_third_occurrence_of_their_position_ends_there():
    """The position after our move (their move) occurs a third time: either side may claim, the route is worth the draw -- killed when ahead, a draw claim when behind."""
    b = chess.Board(PERPETUAL)
    m = b.parse_san("Rf1+")
    after = b.copy()
    after.push(m)
    ag = AG.Agenda(state_of(PERPETUAL, "b"))
    ag.ctx.history = GM.Record().push(G, after).push(G, after)
    k1, drawn = ag._route_rule(m)
    assert k1 == G.position_key(after) and drawn is not None and drawn["killed"] is None and drawn["value"] == 7 and drawn["lemmas"] == ["draw by repetition"]
    ag2 = AG.Agenda(state_of(PERPETUAL, "b"))
    assert ag2._route_rule(m)[1] is None                                                     # no record: no rule
    clocked = PERPETUAL.replace(" 0 41", " 99 41")
    ag3 = AG.Agenda(state_of(clocked, "b"))
    quiet = chess.Board(clocked).parse_san("Rf1+")
    k, d = ag3._route_rule(quiet)
    assert d is not None and d["lemmas"] == ["draw by the fifty-move rule"]                   # the clock reaches 100 with their move: no mate there


# ---------------------------------------------------------------------------------------------------------------- (2) the cycle claim re-labelled

def test_75_4_2_the_cycle_claim_is_the_derived_lemma_a_conjecture_not_the_rule():
    e = LS.load()["repetition"]
    assert e["kind"] == "CLOSING" and e["status"] == "CONJECTURE" and e["lean"] == [] and LS.says("repetition", "draw") and not LS.says("repetition", "terminal")
    assert AG.CYCLE_DETAIL == "a forced cycle: the threefold rule is reachable"
    word = f"repetition ({AG.CYCLE_DETAIL})"                                                    # how the claim reads in a route's parts and the trace
    assert LS.resolve(word) is not None and LS.says(word, "draw") and not LS.says(word, "terminal")       # the entry's (its detail in brackets): a draw, not the rule
    ag = AG.Agenda(state_of(PERPETUAL, "b"))
    ag._route_move, ag._reply_san = chess.Move.from_uci("f2f1"), "Kg2"
    out = ag._inner(PERPETUAL, 1)
    assert out["claim"].lemma == "repetition" and out["claim"].line.startswith(AG.CYCLE_DETAIL + "; ") and out["value"] == 7 and "terminal" not in out      # the decision logic is the cycle's, unchanged
    assert any(x.startswith("    closed: repetition (a forced cycle: the threefold rule is reachable; the position after") for x in ag.ctx.c_lines)


def test_75_4_2_the_strict_form_is_proved_in_lean_and_the_rule_is_not_the_cycle():
    idx = LS.lean_index()
    for n in ("Ideas.Game.forced_cycle_threefold", "Ideas.Game.forcedLine_replays", "Ideas.Game.Congruent", "Ideas.Game.ForcedLine"):
        assert n in idx, n
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "lean" / "ideas" / "Ideas" / "Draws.lean").read_text()
    assert "sorry" not in src and "theorem forced_cycle_threefold" in src


# ---------------------------------------------------------------------------------------------------------------- (3) the chase clock reads the kernel's events

CORNERS = ["r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "3k4/1P6/8/8/8/8/8/4K3 w - - 0 1", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
           "4k3/8/8/2Pp4/8/8/8/4K3 w - d6 0 2", "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1", "8/6P1/8/4R3/3p4/3k3P/4p1K1/5r2 b - - 0 54", chess.STARTING_FEN]


def _positions(n_walks: int = 40, plies: int = 60):
    for i, f in enumerate(CORNERS * (n_walks // len(CORNERS) + 1)):
        if i >= n_walks:
            break
        rng = random.Random(1000 + i)
        b = chess.Board(f)
        for _ in range(plies):
            yield b
            ms = list(b.legal_moves)
            if not ms:
                break
            b = b.copy(stack=False)
            b.push(rng.choice(ms))


def test_75_4_3_the_clock_events_are_the_kernels_what_resets_the_halfmove_clock():
    """`Game.clock_events` against python-chess's `Board.is_zeroing` and the clock after the move (Lean `Chess.applyMove_halfmove`, `Ideas.Game.chess_clock`): castling,
    en passant and promotions included."""
    moves = cap = pawn = 0
    for b in _positions():
        for m in b.legal_moves:
            ev = G.clock_events(b, m)
            after = G.apply(b, m)
            assert bool(ev) == b.is_zeroing(m) == (after.halfmove_clock == 0)
            assert (NM.EVENT_CAPTURE in ev) == b.is_capture(m) and (NM.EVENT_PAWN in ev) == (b.piece_type_at(m.from_square) == chess.PAWN)
            moves += 1
            cap += NM.EVENT_CAPTURE in ev
            pawn += NM.EVENT_PAWN in ev
    print(f"OBS 75.4 (3) clock events: {moves} moves, captures {cap}, pawn moves {pawn}; every one the kernel's is_zeroing and the clock after it")
    assert cap and pawn


def test_75_4_3_the_chase_clock_reads_them_exactly_as_before(monkeypatch):
    """The chase clock's reset (`_irreversible`, `_pawn_move`, `_since_after`, both CLOCK_IRREVERSIBLE settings) against the formulas it replaced (66.5 / 75.2 (b): our move
    `Board.is_zeroing`, their reply `Board.is_capture`, our pawn by the piece on the from-square) on every route and reply of a sample."""
    checked = 0
    for b in list(_positions(16, 12)):
        if b.is_game_over():
            continue
        ag = AG.Agenda(state_of(b.fen(), "w" if b.turn else "b"))
        for m in list(b.legal_moves)[:12]:
            ag._route_move = m
            after = b.copy(stack=False)
            after.push(m)
            for r in list(after.legal_moves)[:8]:
                old_irr = b.is_zeroing(m) or after.is_capture(r)
                assert ag._irreversible(r.uci()) == old_irr
                old_pawn = AG.CLOCK_PAWN and b.piece_type_at(m.from_square) == chess.PAWN
                assert ag._pawn_move(m) == old_pawn
                for dbal in (-1, 0, 1):
                    for irr in (True, False):
                        monkeypatch.setattr(AG, "CLOCK_IRREVERSIBLE", irr)
                        old = (0 if (old_irr and (dbal >= 0 or old_pawn)) else ag.since_capture + 1) if irr else \
                              (0 if (b.is_capture(m) and dbal >= 0) or old_pawn else ag.since_capture + 1)
                        assert ag._since_after(r.uci(), dbal) == old
                        checked += 1
    print(f"OBS 75.4 (3) the chase clock: {checked} (route, reply, balance, setting) readings equal to the formulas replaced")
    assert checked > 1000


# ---------------------------------------------------------------------------------------------------------------- (4) the position key is the kernel's

PERPETUAL_EP = "1Q6/2R2pp1/5k1p/pB6/P1pb4/6PP/5r2/7K b - a3 0 41"      # PERPETUAL as a FEN may write it after a2-a4: a3 named, no black pawn can take there


def test_75_4_4_the_key_is_the_kernels_identity_a_pseudo_legal_en_passant_square_dropped():
    assert AG._pos_key(PSEUDO_EP) == G.position_key(chess.Board(PSEUDO_EP)) != " ".join(PSEUDO_EP.split()[:4])
    assert AG._pos_key(PERPETUAL_EP) == AG._pos_key(PERPETUAL) and AG._pos_key(LEGAL_EP).endswith(" d6")
    n = 0
    for b in _positions(24, 30):                                                              # a FEN the kernel renders (python-chess, the Rust core): the old cut and the identity agree
        f = b.fen()
        assert AG._pos_key(f) == " ".join(f.split()[:4]) == G.position_key(b)
        r = AC.Pos(f).fen()
        assert AG._pos_key(r) == " ".join(r.split()[:4]) == G.position_key(b)
        n += 1
    print(f"OBS 75.4 (4) kernel-rendered FENs: {n} positions, the cut of python-chess's and of the Rust core's FEN equal to the identity on every one")


def test_75_4_4_a_root_written_with_a_pseudo_legal_square_is_the_position_the_chase_comes_back_to():
    """The root's key is the identity, so a line that returns to the root (Rf1+ Kg2 Rf2+ Kh1: the kernel renders the position without a3) is a repeat of it -- with the
    FEN's first four fields (a3 kept) the repeat of the root was missed and the position read again as a new one."""
    ag = AG.Agenda(state_of(PERPETUAL_EP, "b"))
    assert ag.pk == ag.ctx.root_pk == AG._pos_key(PERPETUAL) and ag.pk != " ".join(PERPETUAL_EP.split()[:4])
    ag._route_move, ag._reply_san = chess.Move.from_uci("f2f1"), "Kg2"
    out = ag._inner(PERPETUAL, 1)
    assert out["claim"] is not None and out["claim"].lemma == "repetition" and out["asked"] == 0 and any("repeats the root position" in x for x in ag.ctx.c_lines)


# ---------------------------------------------------------------------------------------------------------------- (5) promotion from the kernel

def test_75_4_5_every_promotion_piece_comes_from_the_kernel_and_the_valuation_chooses():
    b = chess.Board("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    assert sorted(chess.piece_symbol(m.promotion) for m in G.moves_to(b, chess.A7, chess.A8)) == ["b", "n", "q", "r"]       # every unit the promotion may place
    assert sorted(chess.piece_symbol(m.promotion) for m in G.moves_to(b, chess.A7, chess.B8)) == ["b", "n", "q", "r"]       # a capturing promotion too
    V = LS.goal_kinds().valuation
    push = G.push_to(b, chess.A7, chess.A8, V)
    assert push.promotion == chess.QUEEN and chess.piece_symbol(push.promotion) == LS.goal_kinds().promotion.places       # the valuation's choice is the promotion entry's unit

    class Patched:                                                                            # a valuation that prices the rook above the queen chooses the rook: the choice is read
        def of(self, sym):
            return 10 if sym == "r" else V.of(sym)
    assert G.push_to(b, chess.A7, chess.A8, Patched()).promotion == chess.ROOK
    assert len(G.moves_to(chess.Board(), chess.E2, chess.E4)) == 1 and G.push_to(chess.Board(), chess.E2, chess.E4, V).promotion is None
    pinned = chess.Board("8/r5PK/8/8/8/8/8/k7 w - - 0 1")                                    # the pawn on g7 is pinned along the rank: no legal push, though one is drawn on the board
    assert G.moves_to(pinned, chess.G7, chess.G8) == [] and G.push_to(pinned, chess.G7, chess.G8, V) is None
    assert len(G.moves_to(pinned, chess.G7, chess.G8, legal=False)) == 4                     # the static exchange's pseudo-legal reading still sees four


def test_75_4_5_the_kernels_push_is_the_old_one_wherever_that_was_legal():
    """On seeded walks: for every pawn and every square it may step to, the kernel's push equals the move the words built by hand (a queen on the last rank) whenever that
    was legal, and is None exactly when it was not."""
    V = LS.goal_kinds().valuation
    same = none = promos = 0
    for b in _positions(24, 40):
        for sq in chess.SquareSet(b.pawns & b.occupied_co[b.turn]):
            step = 8 if b.turn else -8
            for to in (sq + step, sq + 2 * step):
                if not 0 <= to < 64:
                    continue
                old = chess.Move(sq, to, promotion=chess.QUEEN if chess.square_rank(to) in (0, 7) else None)
                new = G.push_to(b, sq, to, V)
                if old in b.legal_moves:
                    assert new == old
                    same += 1
                    promos += old.promotion is not None
                else:
                    assert new is None
                    none += 1
    print(f"OBS 75.4 (5) pushes: {same} equal to the hand-built move ({promos} promotions), {none} with no legal push (the hand-built move illegal)")
    assert same and promos and none
