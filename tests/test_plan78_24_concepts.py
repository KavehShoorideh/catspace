"""Plan 78.24 B: the relations between units (rules/concepts/chess_units.gdl) -- each equals an independent python-chess oracle on small (3-7 unit) and full-board positions."""
import json
import random
import time

import chess
import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.io import paths

RULES = paths.REPO_ROOT / "rules"
VJ = json.loads((RULES / "learned/chess/vocabulary.relational.json").read_text())
BASE = (RULES / VJ["spec"]).read_text()
TEXT = BASE
for _c in VJ["concepts"]:
    TEXT = CE.compose(TEXT, (RULES / _c).read_text())
NEW = ["apart", "closer", "flight", "opposition", "diagonal_opposition", "to_promote", "king_ahead", "in_square"]
VOC = json.dumps({"derived": NEW, "delta": ["cell"], "units": None})
SIDE = {chess.WHITE: "white", chess.BLACK: "black"}


def sqn(sq):
    return chess.FILE_NAMES[chess.square_file(sq)], str(chess.square_rank(sq) + 1)


def facts(fen):
    out = {k: set() for k in NEW}
    for r in CE.facts("chess", TEXT, VOC, fen, [], False):
        if r[0] in out:
            out[r[0]].add(tuple(r[1:]))
    return out


def small_positions(n=60, seed=24):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board(None)
        sqs = rng.sample(range(64), rng.randint(3, 7))
        b.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
        for s in sqs[2:]:
            kind = rng.choice([chess.PAWN, chess.PAWN, chess.ROOK, chess.QUEEN, chess.BISHOP, chess.KNIGHT])
            if kind == chess.PAWN and chess.square_rank(s) in (0, 7):
                continue
            b.set_piece_at(s, chess.Piece(kind, rng.choice([chess.WHITE, chess.BLACK])))
        b.turn = rng.choice([chess.WHITE, chess.BLACK])
        if b.is_valid() and len(b.piece_map()) >= 3:
            out.append(b.fen())
    return out


def full_positions(n=12, seed=5):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        b = chess.Board()
        for _ in range(rng.randint(20, 60)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        out.append(b.fen())
    return out


SMALL = small_positions()
FULL = full_positions()
HAND = ["4k3/8/4K3/8/8/8/8/8 w - - 0 1", "8/4P3/8/8/8/8/k7/4K3 w - - 0 1", "4k3/8/8/8/8/8/3pK3/8 b - - 0 1",
        "8/8/8/3k4/8/3K4/8/8 w - - 0 1", "8/8/5k2/8/3K4/8/8/8 b - - 0 1", "4k3/8/8/8/8/8/R7/4K3 b - - 0 1", "7k/8/8/8/8/8/p7/K7 w - - 0 1"]


def oracle(b):
    o = {k: set() for k in NEW}
    pm = b.piece_map()
    for i in range(8):
        for j in range(i + 1, 8):
            o["closer"].add((f"d{i}", f"d{j}"))
    kings = {c: b.king(c) for c in (chess.WHITE, chess.BLACK)}
    for s1, p1 in pm.items():
        for s2, p2 in pm.items():
            if s1 != s2 and (p1.piece_type == chess.KING or p2.piece_type == chess.KING):
                d = chess.square_distance(s1, s2)
                o["apart"].add((SIDE[p1.color], *sqn(s1), SIDE[p2.color], *sqn(s2), f"d{d}"))
    for c in (chess.WHITE, chess.BLACK):
        k = kings[c]
        for t in chess.SquareSet(chess.BB_KING_ATTACKS[k]):
            q = pm.get(t)
            if (q is None or q.color != c) and not b.attackers(not c, t):
                o["flight"].add((SIDE[c], *sqn(t)))
    wk, bk = kings[chess.WHITE], kings[chess.BLACK]
    df, dr = abs(chess.square_file(wk) - chess.square_file(bk)), abs(chess.square_rank(wk) - chess.square_rank(bk))
    if (df == 0 and dr == 2) or (dr == 0 and df == 2):
        o["opposition"].add(())
    if df == 2 and dr == 2:
        o["diagonal_opposition"].add(())
    for s, p in pm.items():
        if p.piece_type != chess.PAWN:
            continue
        c, f, r = p.color, chess.square_file(s), chess.square_rank(s)
        steps = 7 - r if c else r
        o["to_promote"].add((SIDE[c], *sqn(s), f"d{steps}"))
        kf, kr = chess.square_file(kings[c]), chess.square_rank(kings[c])
        if abs(kf - f) <= 1 and ((kr > r) if c else (kr < r)):
            o["king_ahead"].add((SIDE[c], *sqn(s)))
        ok = kings[not c]
        prom = chess.square(f, 7 if c else 0)
        if chess.square_distance(ok, prom) <= steps:
            o["in_square"].add((SIDE[c], *sqn(s)))
    return o


@pytest.mark.parametrize("rel", NEW)
def test_relation_equals_oracle(rel):
    seen = 0
    for fen in SMALL + FULL + HAND:
        got, want = facts(fen)[rel], oracle(chess.Board(fen))[rel]
        assert got == want, (rel, fen, got ^ want)
        seen += bool(want)
    assert seen > 0, rel


def test_sample_shape():
    assert len(SMALL) >= 30 and all(3 <= len(chess.Board(f).piece_map()) <= 7 for f in SMALL)
    assert len(FULL) >= 10
    assert {chess.Board(f).turn for f in SMALL} == {True, False}


def test_hand_cases():
    f = facts("4k3/8/8/8/8/8/R7/4K3 b - - 0 1")["flight"]   # black king e8 vs rook a2: the rook covers the 2nd rank only, so d8 d7(no) ...
    assert ("black", "d", "8") in f and ("black", "d", "7") in f
    assert facts("4k3/8/4K3/8/8/8/8/8 w - - 0 1")["opposition"] == {()}
    assert facts("4k3/8/8/8/8/8/3pK3/8 b - - 0 1")["flight"] >= {("black", "d", "7")}
    assert facts("8/4P3/8/8/8/8/k7/4K3 w - - 0 1")["to_promote"] == {("white", "e", "7", "d1")}
    assert facts("8/4P3/8/8/8/8/k7/4K3 w - - 0 1")["in_square"] == set()
    assert facts("5k2/4P3/8/8/8/8/8/4K3 b - - 0 1")["in_square"] == {("white", "e", "7")}


def test_flight_xray_caveat():
    # white Ke1 is checked by Re8 along the e-file; e1's flight squares include d1 f1 d2 f2 but ALSO ... the square behind the king on the file does not exist here, so use a rank check:
    fen = "4k3/8/8/8/8/8/8/r2K4 w - - 0 1"   # Ra1 checks Kd1 along the first rank; c1 lies between, e1 lies BEHIND the king
    f = facts(fen)["flight"]
    assert ("white", "e", "1") in f           # attack has no x-ray through the king: a player excludes e1
    assert ("white", "c", "1") not in f


def test_vocabulary_json_and_program():
    prog = GDL.Program(TEXT)
    assert prog.typecheck() == [], prog.typecheck()[:5]
    for rel in VJ["derived"] + VJ["delta"]:
        assert prog.signature(rel) is not None, rel
    assert set(VJ["delta"]) <= {"cell"} | set(VJ["derived"]) and "attack" not in VJ["delta"]
    assert VJ["concepts"] == ["concepts/chess.gdl", "concepts/chess_units.gdl"]
    base = json.loads((RULES / "learned/chess/vocabulary.json").read_text())
    assert set(base) <= set(VJ)


def test_distance_constants_disjoint_from_ranks():
    sorts = {}
    for line in TEXT.splitlines():
        if line.startswith("(sort "):
            parts = line.strip("()\n ").split()
            sorts[parts[1]] = parts[2:]
    assert sorts["distance"] == [f"d{i}" for i in range(8)]
    assert not set(sorts["distance"]) & (set(sorts["rank"]) | set(sorts["clock"] if False else []))
    assert not set(sorts["distance"]) & set(sum((v for k, v in sorts.items() if k != "distance"), []))


def test_obs_cost():
    def run(fens):
        n, t = [], time.perf_counter()
        for fen in fens:
            n.append(sum(len(v) for v in facts(fen).values()))
        return sum(n) / len(n), (time.perf_counter() - t) * 1000 / len(fens)
    sn, st = run(SMALL)
    fn, ft = run(FULL)
    print(f"OBS units vocabulary: {len(SMALL) + len(FULL)} positions, {sn:.0f} facts a position (small) / {fn:.0f} (full board), {st:.1f} ms a position (small) / {ft:.1f} (full board)")
