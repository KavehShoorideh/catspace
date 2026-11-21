"""PLAN 79.23 PART B: THE MAKER OF THE FIXTURES OF LAYER 2, THE RULES (SYSTEM_SPEC 11 "a layer's test uses only that layer and what is below it"; ledger row 36; piece: rules).
THIS IS NOT A TEST. It writes, once, the files rust/engine/game/tests/fixtures/<game>.fixture that the tests of cs_game read as opaque text. The reference is the python library `chess` (with chess.variant), which shares no code with the engine (the engine's
kernel is another library and its description is rules/*.gdl); the positions, the allowed moves, the ends, the facts, the position keys and the counts below are what that library says, never what the engine says. Run:
    .venv/bin/python -m catspace.incubator.analyst.discovery.layer2_fixtures_print [OUT_DIR]
The names of the game live here and in the files this writes, and nowhere in the layer's Rust (a scan, tests/test_plan79_23.py).

FILE FORMAT (tab separated, the tag first; every fact a token `relation,arg,arg`; a list of facts is joined by spaces):
  game   <name>  <declaration file under the rules root>                 the game the file is of
  ends   <labels>                                                         the ways this game ends (each is reached by some `pos` or `step`, the test checks)
  perft  <label> <text> <n1,n2,..>                                        the positions reached after exactly 1, 2, .. plies (an ended position has no moves)
  pos    <label> <text> <legal text> <end> <moves> <facts> <identity> <side to move>
                                                                          a position: its text form (the passed-over square always written), the text form that writes it only where a taking is legal, `-` or `<label>/<value of each side>`,
                                                                          the allowed moves (sorted), its facts (sorted), the hash of its identity (board, side, rights, a legal passing-square: what a repeat compares), the side to move
  step   <pos label> <move> <end> <added> <removed> <identity> <side to move>
                                                                          the position after one allowed move of a `pos`, as the facts added and removed (`-` for none)
  line   <label> <moves each `name|term`, joined by `;`>                  a game from the start by the moves' names in both readers' notation
  record <label> <moves> <repeats at each ply, comma separated> <limit>   a game from the start and, at each position, how many earlier positions have its identity; <limit> is the number of earlier occurrences that ends the game (the library's third occurrence)
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import chess
import chess.variant

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "rust" / "engine" / "game" / "tests" / "fixtures"
DECL = "learned/chess/graph_vocabulary.json"
KIND = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
SIDE = {True: "white", False: "black"}
CLOCK = 100
BOARDS = {"chess": chess.Board, "kingofthehill": chess.variant.KingOfTheHillBoard, "3check": chess.variant.ThreeCheckBoard}
ENDS = {"chess": ["beaten", "no_move_left", "dead_material", "counter"], "kingofthehill": ["beaten", "no_move_left", "counter", "variant_goal"],
        "3check": ["beaten", "no_move_left", "dead_material", "counter", "variant_goal"]}
START3 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3+3 0 1"

# ---------------------------------------------------------------------------------------------------------------------------- the reference, by the library


def new(game: str, text: str | None = None):
    b = BOARDS[game]()
    if text:
        b.set_fen(text)
    return b


def moves_of(b) -> list[str]:
    return sorted(m.uci() for m in b.legal_moves)


def given(b, colour: bool) -> int:
    """checks given so far, counted as the description counts them: one state late (a check stands at a state whose side to move is in check and is counted into the next state)"""
    return 3 - b.remaining_checks[colour] - (1 if b.is_check() and b.turn != colour else 0)


def end_of(game: str, b) -> str:
    """`-` while the game goes on; else `label/value of white,value of black`. The order of the reasons is the order of the description."""
    if game != "chess" and b.is_variant_end():
        if b.is_variant_win():
            w = b.turn
        elif b.is_variant_loss():
            w = not b.turn
        else:
            return "variant_goal/50,50"
        return f"variant_goal/{'100,0' if w else '0,100'}"
    if b.is_checkmate():
        return f"beaten/{'0,100' if b.turn else '100,0'}"
    if b.is_stalemate():
        return "no_move_left/50,50"
    if b.is_insufficient_material():
        return "dead_material/50,50"
    if b.halfmove_clock >= CLOCK:
        return "counter/50,50"
    return "-"


def allowed(game: str, b) -> list[str]:
    return [] if end_of(game, b) != "-" else moves_of(b)


def facts_of(game: str, b) -> list[str]:
    out = []
    for sq, p in b.piece_map().items():
        out.append(f"cell,{chess.FILE_NAMES[chess.square_file(sq)]},{chess.square_rank(sq) + 1},{SIDE[p.color]},{KIND[p.piece_type]}")
    out.append(f"control,{SIDE[b.turn]}")
    for c in (True, False):
        if b.has_kingside_castling_rights(c):
            out.append(f"castling_right,{SIDE[c]},kingside")
        if b.has_queenside_castling_rights(c):
            out.append(f"castling_right,{SIDE[c]},queenside")
    if b.ep_square is not None:
        out.append(f"ep_square,{chess.FILE_NAMES[chess.square_file(b.ep_square)]},{chess.square_rank(b.ep_square) + 1}")
    out.append(f"halfmove,{min(b.halfmove_clock, CLOCK)}")
    if game == "3check":
        for c in (True, False):
            out.append(f"checks,{SIDE[c]},{given(b, c)}")
    return sorted(out)


def identity_of(game: str, b) -> str:
    """the hash of what a repeat compares: the library's own transposition key (board, side, rights, a legal passing-square; for the checks game the checks left)"""
    return hashlib.md5(repr(b._transposition_key()).encode()).hexdigest()[:12]


def text_of(b) -> str:
    return b.fen(en_passant="fen")


def legal_text_of(b) -> str:
    return b.fen(en_passant="legal")


def spec_term(uci: str) -> str:
    """the move as the description's own term (the notation the second reader plays in)"""
    if len(uci) == 5:
        return f"(promote {uci[0]} {uci[1]} {uci[2]} {uci[3]} {dict(q='queen', r='rook', b='bishop', n='knight')[uci[4]]})"
    return f"(move {uci[0]} {uci[1]} {uci[2]} {uci[3]})"


# ---------------------------------------------------------------------------------------------------------------------------- what is named: the corners and the counts

PERFT_STANDARD = [
    ("start", None, 4),
    ("rich", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3),
    ("pinned_and_passing", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4),
    ("promotions", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3),
    ("promotions_mirrored", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 3),
    ("a_taking_in_passing", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", 3),
    ("promotion_among_pieces", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3),
    ("open_middle", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3),
]
CORNERS_STANDARD = {
    "passing_right_no_taker": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "passing_right_taker": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "passing_right_pinned": "8/8/8/KPp4r/8/8/8/7k w - c6 0 1",
    "passing_black_taker": "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
    "castle_both_ways": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "castle_through_attack": "4k3/8/8/8/8/8/5r2/R3K2R w KQ - 0 1",
    "castle_out_of_attack": "4k3/8/8/8/4r3/8/8/R3K2R w KQ - 0 1",
    "castle_with_rook_attacked": "1r2k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
    "castle_rights_partly_lost": "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1",
    "castle_no_rights": "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1",
    "castle_blocked": "r3k2r/8/8/8/8/8/8/RN2K1NR w KQkq - 0 1",
    "castle_black_to_move": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
    "promote_white": "8/P6k/8/8/8/8/7K/8 w - - 0 1",
    "promote_taking": "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "promote_black": "4k3/8/8/8/8/8/p7/1N2K3 b - - 0 1",
    "promote_and_give_attack": "k7/P7/8/8/8/8/8/K7 w - - 0 1",
    "mated": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "mated_at_the_counter": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 100 3",
    "no_move_left": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    "no_move_left_at_the_counter": "7k/5Q2/6K1/8/8/8/8/8 b - - 100 90",
    "bare_units": "8/8/8/4k3/8/8/8/4K3 w - - 0 1",
    "one_light_unit": "8/8/8/4k3/8/8/8/3BK3 w - - 0 1",
    "one_knight": "8/8/8/4k3/8/8/8/3NK3 w - - 0 1",
    "same_colour_bishops": "8/8/4k3/2b5/8/8/5B2/4K3 w - - 0 1",
    "opposite_colour_bishops": "8/8/4k3/2b5/8/8/4B3/4K3 w - - 0 1",
    "two_knights": "8/8/8/4k3/8/8/8/2NNK3 w - - 0 1",
    "one_rook": "8/8/8/4k3/8/8/8/R3K3 w - - 0 1",
    "counter_before": "8/8/8/4k3/8/8/8/R3K3 w - - 99 100",
    "counter_reached": "8/8/8/4k3/8/8/8/R3K3 w - - 100 120",
    "counter_reset_by_a_taking": "8/8/8/4k3/4p3/8/8/R3K3 b - - 99 100",
    "double_attack_on_the_ruler": "4k3/8/8/8/1b6/8/3n4/4K3 w - - 0 1",
    "single_unit_pinned": "4k3/4r3/8/8/8/8/4B3/4K3 w - - 0 1",
    "ruler_in_attack_only_ruler_moves": "4k3/8/8/8/8/2b5/1n6/R3K3 w - - 0 1",
}
CORNERS_KOTH = {
    "on_the_hill_white": "8/8/8/3K4/8/8/8/k7 b - - 0 1",
    "on_the_hill_black": "8/8/8/8/3k4/8/8/K7 w - - 0 1",
    "one_step_from_the_hill": "8/8/8/8/8/3K4/8/k7 w - - 0 1",
    "hill_step_into_attack": "8/8/8/8/3r4/3K4/8/k7 w - - 0 1",
    "bare_units_no_end": "8/8/8/8/8/8/8/k1K5 w - - 0 1",
    "one_light_unit": "8/8/8/4k3/8/8/8/3BK3 w - - 0 1",
    "mated": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "no_move_left": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    "counter_reached": "8/8/8/8/8/8/8/k1K1R3 w - - 100 120",
    "passing_right_taker": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "castle_both_ways": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "promote_white": "8/P6k/8/8/8/8/7K/8 w - - 0 1",
}
CORNERS_3CHECK = {
    "start": START3,
    "one_check_left_and_a_check_to_give": "4k3/8/8/8/8/8/R7/4K3 w - - 1+3 0 1",
    "one_check_left_black": "4k3/8/8/8/8/8/r7/4K3 b - - 3+1 0 1",
    "two_checks_given_white": "4k3/8/8/8/8/8/R7/4K3 w - - 1+2 0 1",
    "third_check_given": "R3k3/8/8/8/8/8/8/4K3 b - - 0+3 0 1",
    "checks_given_and_in_check": "4k3/8/8/8/8/8/4R3/4K3 b - - 2+3 0 1",
    "mated": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 3+2 1 3",
    "no_move_left": "7k/5Q2/6K1/8/8/8/8/8 b - - 3+3 0 1",
    "bare_units": "8/8/8/4k3/8/8/8/4K3 w - - 3+3 0 1",
    "one_light_unit": "8/8/8/4k3/8/8/8/3BK3 w - - 3+3 0 1",
    "counter_reached": "8/8/8/4k3/8/8/8/R3K3 w - - 3+3 100 120",
    "passing_right_taker": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 3+3 0 3",
    "castle_both_ways": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 3+3 0 1",
    "promote_taking": "1n2k3/P7/8/8/8/8/8/4K3 w - - 3+3 0 1",
}
PERFT_KOTH = [("start", None, 4), ("hill_run", "8/8/8/8/8/3K4/8/k7 w - - 0 1", 4), ("rich", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3),
              ("promotions", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3), ("a_taking_in_passing", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", 3)]
PERFT_3CHECK = [("start", START3, 4), ("checks_near_the_end", "4k3/8/8/8/8/8/R7/4K3 w - - 1+3 0 1", 4), ("rich", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 2+2 0 1", 3),
                ("promotions", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 3+3 0 1", 3), ("a_taking_in_passing", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 3+3 0 3", 3)]

LINES = {
    "chess": {
        "mate_in_four": "f2f3 e7e5 g2g4 d8h4",
        "no_move_left_in_nineteen": "e2e3 a7a5 d1h5 a8a6 h5a5 h7h5 h2h4 a6h6 a5c7 f7f6 c7d7 e8f7 d7b7 d8d3 b7b8 d3h7 b8c8 f7g6 c8e6",
        "a_taking_in_passing": "e2e4 a7a6 e4e5 d7d5 e5d6",
        "castle_short": "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1g1",
        "castle_long": "d2d4 d7d5 b1c3 b8c6 c1f4 c8f5 d1d3 d8d6 e1c1",
        "promote_taking": "a2a4 b7b5 a4b5 a7a6 b5a6 c8b7 a6b7 g8f6 b7a8q",
        "shuffle": " ".join(["g1f3", "g8f6", "f3g1", "f6g8"] * 25),
    },
    "kingofthehill": {
        "mate_in_four": "f2f3 e7e5 g2g4 d8h4",
        "the_hill_reached": "e2e3 e7e6 e1e2 e8e7 e2f3 e7f6 f3e4",
        "shuffle": " ".join(["g1f3", "g8f6", "f3g1", "f6g8"] * 25),
    },
    "3check": {
        "three_checks": None,
        "mate_in_four": "f2f3 e7e5 g2g4 d8h4",
        "castle_short": "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1g1",
        "shuffle": " ".join(["g1f3", "g8f6", "f3g1", "f6g8"] * 25),
    },
}
RECORDS = {"shuffle_to_three": "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8", "shuffle_and_leave": "g1f3 g8f6 f3g1 f6g8 b1c3 g8f6 c3b1 f6g8 g1f3 g8f6 f3g1 f6g8", "no_repeat": "e2e4 e7e5 g1f3 b8c6"}


def play(game: str, ucis: str, start: str | None = None):
    b = new(game, start)
    seq = [b.copy()]
    for u in ucis.split():
        m = chess.Move.from_uci(u)
        assert m in b.legal_moves, (game, u, b.fen())
        b.push(m)
        seq.append(b.copy())
    return seq


def find_three_checks(rng: random.Random) -> str:
    """a line from the start of the checks game in which a third check ends the game: random play that gives a check whenever it can (seeded)"""
    while True:
        b = new("3check")
        line = []
        for _ in range(80):
            ms = list(b.legal_moves)
            if not ms:
                break
            checking = []
            for m in ms:
                b.push(m)
                if b.is_check():
                    checking.append(m)
                b.pop()
            m = rng.choice(checking if checking and rng.random() < 0.9 else ms)
            b.push(m)
            line.append(m.uci())
            if b.is_variant_end():
                return " ".join(line)


# ---------------------------------------------------------------------------------------------------------------------------- walking


def perft(game: str, start: str | None, depth: int) -> list[int]:
    counts = [0] * (depth + 1)
    b = new(game, start)

    def walk(k: int):
        if end_of(game, b) != "-":
            return
        ms = list(b.legal_moves)
        counts[k + 1] += len(ms)
        if k + 1 < depth:
            for m in ms:
                b.push(m)
                walk(k + 1)
                b.pop()

    walk(0)
    return counts[1:]


def random_positions(game: str, rng: random.Random, games: int, plies: int, every: int):
    out = []
    for _ in range(games):
        b = new(game)
        for ply in range(plies):
            if end_of(game, b) != "-":
                out.append(b.copy())
                break
            if ply % every == 0:
                out.append(b.copy())
            b.push(rng.choice(list(b.legal_moves)))
    return out


def make(game: str) -> str:
    rng = random.Random(f"layer2-{game}-2026-11-20")
    rows = [f"game\t{game}\t{DECL}", "ends\t" + " ".join(ENDS[game])]
    perfts = {"chess": PERFT_STANDARD, "kingofthehill": PERFT_KOTH, "3check": PERFT_3CHECK}[game]
    for label, start, depth in perfts:
        s = start or ("" if game != "3check" else START3)
        b = new(game, s or None)
        rows.append(f"perft\t{label}\t{text_of(b)}\t{','.join(map(str, perft(game, s or None, depth)))}")
    corners = {"chess": CORNERS_STANDARD, "kingofthehill": CORNERS_KOTH, "3check": CORNERS_3CHECK}[game]
    positions: list[tuple[str, object, bool]] = [("start", new(game), True)]
    positions += [(f"corner:{k}", new(game, v), True) for k, v in corners.items()]
    lines = dict(LINES[game])
    if game == "3check":
        lines["three_checks"] = find_three_checks(rng)
    for name, ucis in lines.items():
        seq = play(game, ucis)
        for ply in sorted({0, 1, len(seq) - 3, len(seq) - 2, len(seq) - 1} | set(range(10, len(seq), 20))):
            if 0 <= ply < len(seq):
                positions.append((f"line:{name}:{ply}", seq[ply], False))
    for i, b in enumerate(random_positions(game, rng, 12 if game == "chess" else 9, 330, 20)):
        positions.append((f"random:{i}", b, False))
    seen_labels = set()
    reached_ends = set()
    for label, b, every_move in positions:
        text = text_of(b)
        end = end_of(game, b)
        if end != "-":
            reached_ends.add(end.split("/")[0])
        ms = allowed(game, b)
        rows.append("\t".join(["pos", label, text, legal_text_of(b), end, " ".join(ms) or "-", " ".join(facts_of(game, b)), identity_of(game, b), SIDE[b.turn]]))
        before = set(facts_of(game, b))
        picks = ms if every_move else rng.sample(ms, min(2, len(ms)))
        for u in picks:
            c = b.copy()
            c.push(chess.Move.from_uci(u))
            after = set(facts_of(game, c))
            e = end_of(game, c)
            if e != "-":
                reached_ends.add(e.split("/")[0])
            rows.append("\t".join(["step", label, u, e, " ".join(sorted(after - before)) or "-", " ".join(sorted(before - after)) or "-", identity_of(game, c), SIDE[c.turn]]))
    for name, ucis in lines.items():
        rows.append(f"line\t{name}\t" + ";".join(f"{u}|{spec_term(u)}" for u in ucis.split()))
    for name, ucis in RECORDS.items():
        seq = play(game, ucis)
        ids = [identity_of(game, x) for x in seq]
        reps = [ids[:i].count(ids[i]) for i in range(len(ids))]
        rows.append(f"record\t{name}\t{ucis}\t{','.join(map(str, reps))}\t2")
    missing = set(ENDS[game]) - reached_ends
    assert not missing, (game, "ends not reached", missing)
    return "\n".join(rows) + "\n"


def main(argv: list | None = None) -> int:
    out = Path((argv if argv is not None else sys.argv[1:])[0]) if (argv if argv is not None else sys.argv[1:]) else OUT
    out.mkdir(parents=True, exist_ok=True)
    for game in BOARDS:
        text = make(game)
        (out / f"{game}.fixture").write_text(text)
        print(f"{game}: {len(text.splitlines())} lines, {len(text)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
