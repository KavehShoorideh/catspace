"""Plan 78.21: the human concept vocabulary (rules/concepts/chess.gdl) -- every concept equals an independent python-chess oracle on seeded random positions."""
import json
import random

import chess
import pytest

import cs_engine as CE
from catspace.incubator.analyst import relations as REL
from catspace.incubator.analyst import structure as STR
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.io import paths

GV = GS.graph_vocabulary("chess")
TEXTS = GS.graph_spec_texts("chess")
TEXT = TEXTS["chess"]
VOC = json.dumps({"derived": GV["derived"], "delta": GV["delta"], "units": GV["units"]})
SIDE = {chess.WHITE: "white", chess.BLACK: "black"}
KIND = {chess.KING: "king", chess.QUEEN: "queen", chess.ROOK: "rook", chess.BISHOP: "bishop", chess.KNIGHT: "knight", chess.PAWN: "pawn"}
CONCEPTS = ["defended", "hanging", "pinned", "fork", "open_file", "half_open_file", "passed_pawn", "isolated_pawn", "doubled_pawn", "absent", "bishop_pair"]


def sqn(sq):  # (file, rank) as the facts print them
    return chess.FILE_NAMES[chess.square_file(sq)], str(chess.square_rank(sq) + 1)


def rows(fen, relation):
    return {tuple(r[1:]) for r in CE.facts("chess", TEXT, VOC, fen, [], False) if r[0] == relation}


def positions(n_games=30, seed=7):
    rng = random.Random(seed)
    out = []
    for _ in range(n_games):
        b = chess.Board()
        for ply in range(160):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
            if ply % 4 == 1:
                out.append(b.fen())
    return out


POS = positions()
HAND = {  # fens where the concept is present
    "pin": "4k3/4r3/8/8/8/8/4B3/4K3 w - - 0 1",
    "fork": "r3k3/8/8/8/3N4/8/8/4K3 w - - 0 1",        # a knight that attacks nobody
    "forked": "3k1r2/8/4N3/8/8/8/8/4K3 b - - 0 1",     # Ne6 attacks the king d8 and the hanging rook f8
    "passed": "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",
    "open": "4k3/pp3ppp/8/8/8/8/PP3PPP/4K3 w - - 0 1",
    "nobishops": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "pair": "4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1",
    "doubled": "4k3/8/8/8/P7/P7/8/4K3 w - - 0 1",
    "royalfork": "4k3/8/8/8/1q6/8/8/R3K3 w - - 0 1",
}


def board_of(fen):
    return chess.Board(fen)


def oracle(b):
    """{relation: set of rows} by python-chess, the definitions of rules/concepts/chess.gdl re-derived independently."""
    o = {k: set() for k in CONCEPTS}
    pm = b.piece_map()
    for sq, p in pm.items():
        c = p.color
        if b.attackers(c, sq):
            o["defended"].add((SIDE[c], *sqn(sq)))
        if p.piece_type != chess.KING and b.attackers(not c, sq) and not b.attackers(c, sq):
            o["hanging"].add((SIDE[c], *sqn(sq)))
        if p.piece_type != chess.KING and b.is_pinned(c, sq):
            o["pinned"].add((SIDE[c], *sqn(sq)))
    for sq, p in pm.items():
        c = p.color
        n = 0
        for t in b.attacks(sq):
            q = pm.get(t)
            if q is not None and q.color != c:
                # hanging(q.color, t): not king, attacked by c, not defended by q.color
                if q.piece_type == chess.KING or not b.attackers(q.color, t):
                    n += 1
        if n >= 2:
            o["fork"].add((SIDE[c], *sqn(sq)))
    pawns = {c: [s for s in pm if pm[s].piece_type == chess.PAWN and pm[s].color == c] for c in (chess.WHITE, chess.BLACK)}
    files = {c: {chess.square_file(s) for s in pawns[c]} for c in pawns}
    for f in range(8):
        name = chess.FILE_NAMES[f]
        if f not in files[chess.WHITE] and f not in files[chess.BLACK]:
            o["open_file"].add((name,))
        for c in (chess.WHITE, chess.BLACK):
            if f not in files[c] and f in files[not c]:
                o["half_open_file"].add((SIDE[c], name))
    for c in (chess.WHITE, chess.BLACK):
        for s in pawns[c]:
            f, r = chess.square_file(s), chess.square_rank(s)
            if not any(abs(chess.square_file(t) - f) <= 1 and ((chess.square_rank(t) > r) if c else (chess.square_rank(t) < r)) for t in pawns[not c]):
                o["passed_pawn"].add((SIDE[c], *sqn(s)))
            if not (files[c] & {f - 1, f + 1}):
                o["isolated_pawn"].add((SIDE[c], *sqn(s)))
            if sum(1 for t in pawns[c] if chess.square_file(t) == f) >= 2:
                o["doubled_pawn"].add((SIDE[c], *sqn(s)))
        for k, kn in KIND.items():
            if not b.pieces(k, c):
                o["absent"].add((SIDE[c], kn))
        cols = {(chess.square_file(s) + chess.square_rank(s)) % 2 for s in b.pieces(chess.BISHOP, c)}
        if len(cols) == 2:
            o["bishop_pair"].add((SIDE[c],))
    return o


def test_composed_description_loads_and_has_signatures():
    for variant, text in TEXTS.items():
        prog = GDL.Program(text)
        for rel in GV["derived"] + GV["static"] + GV["state"]:
            assert prog.signature(rel) is not None, (variant, rel)
        assert prog.typecheck() == [], (variant, prog.typecheck()[:5])


def test_square_object_matches_shelf():
    shelf = json.loads((paths.REPO_ROOT / "catspace/incubator/analyst/discovery/lemmas/ontology/square.json").read_text())
    assert GV["objects"]["square"] == shelf["sort"] == ["file", "rank"]


def test_diagonal_of_exact():
    got = {tuple(r[1:]) for r in CE.facts("chess", TEXT, json.dumps({"derived": ["diagonal_of"], "delta": ["cell"], "units": None}), None, [], False) if r[0] == "diagonal_of"}
    # direct computation: a diagonal is a maximal line of squares with f-r constant (rising) or f+r constant (falling), length >= 2
    want, names = set(), set()
    for rising in (True, False):
        for k in range(-7, 16):
            sq = [(f, (f - k) if rising else (k - f)) for f in range(8) if 0 <= ((f - k) if rising else (k - f)) < 8]
            if len(sq) < 2:
                continue
            (f1, r1), (f2, r2) = sq[0], sq[-1]
            name = f"{chess.FILE_NAMES[f1]}{r1 + 1}{chess.FILE_NAMES[f2]}{r2 + 1}"
            names.add(name)
            want |= {(chess.FILE_NAMES[f], str(r + 1), name) for f, r in sq}
    assert len(names) == 26
    assert got == want
    per_square = {}
    for f, r, d in got:
        per_square.setdefault((f, r), set()).add(d)
    assert len(per_square) == 64 and all(1 <= len(v) <= 2 for v in per_square.values())


def test_positions_count():
    assert len(POS) >= 300
    assert max(chess.Board(f).fullmove_number for f in POS) >= 40      # late-game positions, not only openings
    assert sum(1 for f in POS if len(chess.Board(f).piece_map()) <= 12) >= 30


@pytest.mark.parametrize("concept", CONCEPTS)
def test_concept_equals_oracle(concept):
    seen = 0
    for fen in POS + list(HAND.values()):
        got = rows(fen, concept)
        want = oracle(board_of(fen))[concept]
        assert got == want, (concept, fen, got ^ want)
        seen += bool(want)
    assert seen > 0, f"{concept} never present in the sample"


def test_hand_made_cases():
    assert ("white", "e", "2") in rows(HAND["pin"], "pinned")
    assert rows(HAND["fork"], "fork") == set()           # the knight on d4 attacks nobody
    assert ("white", "d", "5") in rows(HAND["passed"], "passed_pawn")
    assert ("a",) not in rows(HAND["open"], "open_file") and ("c",) in rows(HAND["open"], "open_file")
    assert rows(HAND["pair"], "bishop_pair") == {("white",)}
    assert rows(HAND["doubled"], "doubled_pawn") == {("white", "a", "3"), ("white", "a", "4")}
    assert rows(HAND["forked"], "fork") == {("white", "e", "6")}


def test_forks_present_and_subset_of_s0():
    # the sample has real forks (a knight on e6 against king d8 and rook f8, both only attacked by it)
    f = "3k1r2/8/4N3/8/8/8/8/4K3 b - - 0 1"
    assert ("white", "e", "6") in rows(f, "fork")
    # relations.py's forks add targets worth more than the forker, so ours are a subset of its forks
    for fen in POS[:120] + [f]:
        b = chess.Board(fen)
        s0 = {(SIDE[r.owner], *sqn(r.squares[0])) for r in REL.forks(b)}
        assert rows(fen, "fork") <= s0, fen


def test_s0_agreement():
    """Same definitions in structure.py / relations.py: passed, isolated, doubled (the file), bishop pair, hanging. (relations.py's `hanging` skips only a royal king: same.)"""
    for fen in POS[:150]:
        b = chess.Board(fen)
        s = STR.structure(b)
        assert {(SIDE[r.owner], *sqn(r.squares[0])) for r in s if r.name == "passed_pawn"} == rows(fen, "passed_pawn")
        assert {(SIDE[r.owner], *sqn(r.squares[0])) for r in s if r.name == "isolated_pawn"} == rows(fen, "isolated_pawn")
        assert {(SIDE[r.owner], r.tags[0]) for r in s if r.name == "doubled_pawns"} == {(c, f) for c, f, _ in rows(fen, "doubled_pawn")}
        assert {(SIDE[r.owner],) for r in s if r.name == "bishop_pair"} == rows(fen, "bishop_pair")
        h = {(SIDE[r.owner], *sqn(r.squares[0])) for r in REL.hanging_and_loose(b) if r.name == "hanging"}
        assert h == rows(fen, "hanging")


def test_absent():
    assert rows(HAND["nobishops"], "absent") >= {("white", "bishop"), ("black", "bishop")}
    assert rows(chess.STARTING_FEN, "absent") == set()


def test_demand_equals_full():
    a = CE.facts_audit("chess", TEXT, VOC, 10, 60, 1)
    print(a)
    assert a["unequal"] == 0
