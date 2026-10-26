"""PLAN 73.5 THE GENERATOR (docs/ANALYST_PLAN.md 73.5; Kaveh 2026-10-25: the base concepts are movement and attack, tempo and material, everything else DERIVED):
the corpus is written by the table (the oracle only there); the measure library is built from the four base words only and never names a unit kind; its atoms read
the numbers the vocabulary's Rust evaluator reads; the miner chooses a measure with 0 violations when the corpus has one (short games written by the table) and
reports the finding when it has none (the table's play in K+R v K: no measure of the library is monotone on it -- no entry is written); the diagnostic's never-rules
refuse a stalemating move and a move that loses a unit by the exchange law."""
import random
import statistics

import chess

from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import technique_corpus as TC
from catspace.incubator.analyst.discovery import technique_measures as TM

UNIT_WORDS = {f"{s}{k}" for s in ("our", "their") for k in ("queens", "rooks", "bishops", "knights", "pawns")} | {"P", "O"} | {f"{x}{i}" for x in "PO" for i in range(2, 6)}


def _symbols(sexpr: str) -> set:
    return set(sexpr.replace("(", " ").replace(")", " ").split())


def test_the_signature_reads_ours_first():
    assert TC.parse_signature("KRvK") == (["K", "R"], ["K"])
    assert TC.parse_signature("KQvK") == (["K", "Q"], ["K"])
    for bad in ("KR", "RKvK", "KRvKK", "KRvQ", "KxvK"):
        try:
            TC.parse_signature(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    b = chess.Board("8/8/3k4/8/2R5/8/8/4K3 w - - 0 1")
    assert TC.signature(b, chess.WHITE) == "KRvK" and TC.signature(b, chess.BLACK) == "KvKR"
    assert TC.signature(b.mirror(), chess.BLACK) == "KRvK"


def test_the_library_is_built_from_the_base_only():
    lib = TM.library()
    print(f"OBS 73.5 library: {len(lib)} candidates ({sum(t.op == 'atom' for t in lib)} atoms, {sum(t.op == 'sum' for t in lib)} sums, {sum(t.op == 'product' for t in lib)} products)")
    assert len(lib) == 81 and sum(t.op == "atom" for t in lib) == 9
    want = {"region": {"movement", "attack"}, "region_off_edge": {"movement", "attack"}, "region_area": {"movement", "attack"}, "edge_dist": {"movement"},
            "corner_dist": {"movement"}, "kings_dist": {"movement"}, "material_us": {"material"}, "material_them": {"material"}, "tempo": {"tempo"}}
    for a in TM.ATOMS:
        assert LS.term_reads(a.sexpr) == want[a.key], a.key                     # each atom reads what the plan says, by the vocabulary's own table
    for t in lib:
        assert t.reads and t.reads <= set(LS.BASE), t.sexpr                      # a measure reads the base and nothing else
        assert t.reads == frozenset().union(*(want[k] for k in t.parts)), t.sexpr
        assert not (_symbols(t.sexpr) & UNIT_WORDS), t.sexpr                     # no unit kind is named: only our king W and their king L
        assert BV.parse(t.sexpr) and t.gloss
        assert TM.term(t.sexpr) == t
    assert TM.term("the rows beyond the rook's wall") is None                    # a hand-made entry's prose is no term of the library


def _small_corpus(tmp_path, sig: str, games: int, seed: int, floor: int):
    out = str(tmp_path / f"{sig}.jsonl")
    lines = []
    info = TC.write_corpus(sig, games=games, seed=seed, out=out, floor=floor, log=lines.append)
    return out, info, lines


def test_the_corpus_is_written_by_the_table(tmp_path):
    out, info, lines = _small_corpus(tmp_path, "KRvK", 3, 1, 20)
    print(lines[0])
    assert lines[0].startswith("OBS corpus KRvK: games 3, positions ") and ", mates 3;" in lines[0]
    games = TC.read_corpus(out)
    assert len(games) == 3
    assert games[0][0]["fen"] == "4K3/R7/8/8/2k5/8/8/8 w - - 0 1"               # the 2026-10-23 experiment's start (play_krk.py, seed 1, DTZ >= 20)
    for rows in games:
        assert rows[-1]["result"] == "mate" and rows[-1]["move"] is None
        assert rows[0]["dtz_before"] >= 20
        for r in rows[:-1]:
            assert chess.Move.from_uci(r["move"]) in chess.Board(r["fen"]).legal_moves
            if r["side"] == r["winner"]:
                assert abs(r["dtz_after"]) < r["dtz_before"]                     # the table's DTZ-best move shortens the distance
    floor, (lo, hi) = TC.dtz_floor("KRvK", sample=41, seed=3)
    rng = random.Random(3)
    sample = [TC.tablebase().probe_dtz(TC.random_start(rng, "KRvK", 1)) for _ in range(41)]
    assert floor == statistics.median_low(sample) and (lo, hi) == (min(sample), max(sample))


def test_the_atoms_read_as_the_vocabulary_reads_them(tmp_path):
    """Every atom the vocabulary also says (`bv`) reads the same number through the Rust evaluator (graph.word, the BaseWords' `word`), for the winning side and on the
    colour-mirrored board for the other colour; EDGE and CORNER, read from movement, are the vocabulary's `edge` and `corner`."""
    BV.install_terms()
    b0 = chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")
    assert G.word(b0, BV.parse("(count edge)"), True) == chess.popcount(TM.EDGE) == 28
    assert G.word(b0, BV.parse("(count (and edge (not corner)))"), True) == chess.popcount(TM.EDGE & ~TM.CORNER) == 24
    n = 0
    for sig in ("KRvK", "KQvK"):
        out, _info, _l = _small_corpus(tmp_path, sig, 2, 5, 9)
        for rows in TC.read_corpus(out):
            for r in rows:
                b = chess.Board(r["fen"])
                for side, bb in ((True, b), (False, b.mirror())):
                    v = TM.atom_values(bb, side)
                    for a in TM.ATOMS:
                        w = G.word(bb, BV.parse(a.bv), side) if a.bv else None
                        if w is not None:                                        # (None: the vocabulary does not hold the board read for that side)
                            assert w == v[a.key], (bb.fen(), side, a.key, v[a.key], w)
                            n += 1
                G.clear_caches()
    print(f"OBS 73.5 atoms against the vocabulary: {n} reads, every one equal")
    assert n > 500


def _short_games(tmp_path, sig: str, games: int, seed: int) -> str:
    """`games` games of the table from starts at DTZ 3 exactly (a mate in two: the shortest drive), written as a corpus file."""
    import json
    rng = random.Random(seed)
    rows, g = [], 0
    while g < games:
        b = TC.random_start(rng, sig, 3)
        if TC.tablebase().probe_dtz(b) != 3:
            continue
        rows += TC.play_game(b, g)[0]
        g += 1
    path = tmp_path / f"{sig}_short.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_the_miner_chooses_a_measure_with_no_violation(tmp_path):
    corpus = _short_games(tmp_path, "KQvK", 5, 2)
    lines = []
    got = TM.mine("KQvK", corpus, log=lines.append)
    print("\n".join(x for x in lines if "choice" in x or "library" in x))
    c = got["choice"]
    assert c is not None and got["candidates"] >= 1
    assert c["violations"] == 0 and c["never"] == 0 and c["found"] == c["floor_positions"] > 0       # no violation, a drop in every game, the mate words at the floor
    cands = [s for s in got["scores"] if s["candidate"]]
    assert c is min(cands, key=lambda s: (s["spread"], s["term"].simplicity()))      # the smallest floor spread, then the fewest base reads
    assert any(x.startswith("OBS miner KQvK choice: ") for x in lines)


def test_the_miner_reports_the_finding_on_the_tables_play(tmp_path):
    """K+R v K, the table's DTZ-best play from the experiment's starts: no measure of the library is monotone -- the miner relaxes nothing and chooses nothing."""
    out, _info, _l = _small_corpus(tmp_path, "KRvK", 5, 1, 20)
    lines = []
    got = TM.mine("KRvK", out, log=lines.append)
    print("\n".join(x for x in lines if "NO TECHNIQUE" in x or "best by" in x))
    assert got["choice"] is None and got["candidates"] == 0
    assert "no technique candidate" in got["finding"]
    for s in got["scores"]:
        if s["violations"] == 0:
            assert s["never"] == 5                                              # the zero-violation terms never drop: the class's constants
    assert any(x.startswith("OBS miner KRvK: NO TECHNIQUE CANDIDATE") for x in lines)
    assert any(x.startswith("OBS miner KRvK violation of ") for x in lines)     # a violation is shown, not only counted


def test_the_never_rules_refuse_stalemate_and_a_lost_unit():
    b = chess.Board("7k/8/6K1/8/8/8/8/5Q2 w - - 0 1")
    kids = {m.uci(): r for m, r in TM._children(b, chess.WHITE, lambda bb: 0)}
    assert "f1f7" not in kids                                                   # Qf7 stalemates
    assert kids["f1f8"] == "mate"                                               # Qf8 mates
    b = chess.Board("8/8/3k4/8/2R5/8/8/4K3 w - - 0 1")
    kids = {m.uci(): r for m, r in TM._children(b, chess.WHITE, lambda bb: 0)}
    assert "c4c5" not in kids and "c4c6" not in kids                            # the rook next to their king, unguarded: the exchange law loses it
    assert "c4c1" in kids and "c4c3" in kids
