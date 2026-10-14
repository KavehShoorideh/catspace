"""The base vocabulary's words for their side (Kaveh 2026-10-12: "base vocab should be complete with words for their
pieces"): O, hit, unhit, ours, theirs -- the Rust evaluator on a known board, the glosses, and the Lean renderings type-check
against Ideas.Base.Vocabulary."""
import subprocess
from pathlib import Path

import pytest

import catspace.incubator.analyst  # noqa: F401
import analyst_core as A
from catspace.incubator.analyst.discovery import base_vocabulary as V

# White (us, to move): K c3, R b4, R f4; Black: K e5, R g3
FEN = "8/8/8/4k3/1R3R2/2K3r1/8/8 w - - 0 1"


def val(term: str) -> int:
    return A.vocab_values([FEN], term)[0]


def test_their_piece_and_the_piece_sets():
    assert val("(count ours)") == 2
    assert val("(count theirs)") == 1
    assert val("(dist W O)") == 4                      # c3 to g3
    assert val("(dist L O)") == 2                      # e5 to g3
    assert val("(count (and theirs attacked))") == 0   # g3: neither rook nor the king reaches it
    assert val("(count (and ours hit))") == 1          # f4: the black king on e5 touches it


def test_hit_lifts_our_king():
    # the g3 rook's rank runs through our king on c3 to b3 and a3 once the king is lifted
    hit = A.vocab_values([FEN], "(count hit)")[0]
    beyond = A.vocab_values([FEN], "(count (and hit (rank W)))")[0]
    assert beyond == 7                                 # a3 b3 c3 d3 e3 f3 h3: the whole rank but g3
    assert hit + A.vocab_values([FEN], "(count unhit)")[0] == 64


def test_glosses_and_enumeration():
    assert V.gloss(V.parse("(count (and ours hit))")) == "how many of the squares of our pieces that are also the squares they attack"
    terms = {V.sexpr(t) for t in V.enumerate_numbers()}
    assert "(count (and hit ours))" in terms and "(dist W O)" in terms and "(count (reach W unhit))" in terms


@pytest.mark.skipif(not Path("lean/ideas/.lake").exists(), reason="the Lean project is not built")
def test_lean_renderings_typecheck(tmp_path):
    terms = ["(count (and ours hit))", "(count (and theirs (not unhit)))", "(dist W O)", "(count (reach W unhit))"]
    body = "\n".join(f"def t{i} (p : Position) (c : Color) (W L P O : Square) (hist : List Position) : Nat := {V.lean(V.parse(t))}" for i, t in enumerate(terms))
    src = tmp_path / "VocabCheck.lean"
    src.write_text(f"import Ideas.Base.Vocabulary\nset_option linter.unusedVariables false\nopen Chess Ideas.Vocabulary\n{body}\n")
    r = subprocess.run(["lake", "env", "lean", str(src)], cwd="lean/ideas", capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr


# THE SECOND BATCH (Kaveh 2026-10-12): geometry -- lines, diagonals, segment, knight hops, the centre, square classes under the
# board's symmetries (the mirror only with pawns) -- and per piece: what the piece on a square attacks

def test_geometry_and_pieces():
    assert val("(count (lines W))") == 14                     # c3: its rank and file
    assert val("(count (diagonals W))") == 11
    assert val("(count (segment W L))") == 3                  # c3, d4, e5
    assert val("(count (knight1 W))") == 8
    assert val("(count center)") == 4
    assert val("(count (attacks O))") == 12                   # the g3 rook: its rank to our king, its file
    assert val("(count (and (attacks O) (lines W)))") == 4    # d3 e3 f3 h3: its squares on our king's lines


def test_square_classes_and_symmetry():
    assert val("(count (cls a1))") == 4 and val("(count (cls b1))") == 8 and val("(count (cls d4))") == 4
    assert sum(A.vocab_values([FEN], f"(count (cls {x}))")[0] for x in V.SQUARE_CLASSES) == 64   # the classes cover the board
    assert A.vocab_values(["8/8/8/4k3/1R3R2/2K3r1/P7/8 w - - 0 1"], "(count (cls a1))")[0] == 2      # a pawn: the mirror only


@pytest.mark.skipif(not Path("lean/ideas/.lake").exists(), reason="the Lean project is not built")
def test_second_batch_lean_renderings_typecheck(tmp_path):
    terms = ["(count (and (attacks O) (lines W)))", "(count (diagonals P))", "(count (segment W L))", "(count (knight2 O))",
             "(count center)", "(count (and (cls a1) (not ours)))", "(count (cls d4))"]
    body = "\n".join(f"def t{i} (p : Position) (c : Color) (W L P O : Square) (hist : List Position) : Nat := {V.lean(V.parse(t))}" for i, t in enumerate(terms))
    src = tmp_path / "VocabCheck2.lean"
    src.write_text(f"import Ideas.Base.Vocabulary\nset_option linter.unusedVariables false\nopen Chess Ideas.Vocabulary\n{body}\n")
    r = subprocess.run(["lake", "env", "lean", str(src)], cwd="lean/ideas", capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_squares_colour_and_the_strongest_other_piece():
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    # K+B+N v K: the bishop on d2 (dark), the knight on a3 -- P is the bishop (the strongest), not the first by square
    fen = "8/8/8/3k4/8/N7/3B4/4K3 w - - 0 1"
    v = lambda t: A.vocab_values([fen], t)[0]
    assert v("(count light)") == v("(count dark)") == 32
    assert v("(count (colour P))") == 32 and v("(count (and (colour P) (diagonals P)))") == v("(count (diagonals P))")
    assert v("(count (and (cls a1) (colour P)))") == 2                    # the corners of the bishop's colour: a1, h8
    assert v("(count (and (cls a1) dark))") == 2 and v("(count (and (cls a1) light))") == 2


@pytest.mark.skipif(not Path("lean/ideas/.lake").exists(), reason="the Lean project is not built")
def test_colour_lean_renderings_typecheck(tmp_path):
    terms = ["(count light)", "(count (and dark (box L)))", "(count (and (cls a1) (colour P)))", "(count (and (colour W) (not (nearer W L))))"]
    body = "\n".join(f"def t{i} (p : Position) (c : Color) (W L P O : Square) (hist : List Position) : Nat := {V.lean(V.parse(t))}" for i, t in enumerate(terms))
    src = tmp_path / "VocabCheck3.lean"
    src.write_text(f"import Ideas.Base.Vocabulary\nset_option linter.unusedVariables false\nopen Chess Ideas.Vocabulary\n{body}\n")
    r = subprocess.run(["lake", "env", "lean", str(src)], cwd="lean/ideas", capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr


# EVERY SQUARE BY ITS NAME (Kaveh 2026-10-13): the 64 squares from our side, the square itself as a set; the class word unchanged

def test_named_squares_from_our_side():
    white = "8/8/8/4k3/8/2K5/P7/8 w - - 0 1"                  # our king c3, our pawn a2, their king e5
    black = "8/p7/2k5/8/4K3/8/8/8 b - - 0 1"                  # the same from Black's side: our king c6, pawn a7
    for fen in (white, black):
        v = lambda t: A.vocab_values([fen], t)[0]
        assert v("(dist W c3)") == 0 and v("(dist L e5)") == 0 and v("(dist P a2)") == 0
        assert v("(count (and (file P) (rank a8)))") == 1          # the pawn's promotion square
        assert v("(dist L a8)") == 4                               # their king to our promotion corner
        assert v("(count (and ours (at a2)))") == 1 and v("(count (and ours (at a3)))") == 0
        assert v("(count (cls a1))") == 2                          # a pawn: the class word keeps its mirror


def test_the_enumeration_names_squares_without_constants():
    terms = {V.sexpr(t) for t in V.enumerate_numbers()}
    assert "(dist L a8)" in terms and "(count (and ours (at e4)))" in terms and "(count (and (file P) (rank a8)))" in terms
    assert "(count (at e4))" not in terms and "(count (rank a8))" not in terms     # constants


@pytest.mark.skipif(not Path("lean/ideas/.lake").exists(), reason="the Lean project is not built")
def test_named_square_renderings_typecheck(tmp_path):
    terms = ["(dist L a8)", "(count (and ours (at e4)))", "(count (and (file P) (rank a8)))", "(count (cls d1))"]
    body = "\n".join(f"def t{i} (p : Position) (c : Color) (W L P O : Square) (hist : List Position) : Nat := {V.lean(V.parse(t))}" for i, t in enumerate(terms))
    src = tmp_path / "VocabCheck.lean"
    src.write_text(f"import Ideas.Base.Vocabulary\nset_option linter.unusedVariables false\nopen Chess Ideas.Vocabulary\n{body}\n")
    r = subprocess.run(["lake", "env", "lean", str(src)], cwd="lean/ideas", capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr


# RELATIVE GEOMETRY (Kaveh 2026-10-13: "the vocabulary needs to be complemented"): coordinates from our side, offsets
# between pieces (plus 7: level) -- "ahead of", "the opposition", an arrangement and its shift as compositions

def test_relative_geometry_from_both_sides():
    white = "8/8/8/4k3/8/2K5/P7/8 w - - 0 1"                  # our king c3, our pawn a2, their king e5
    black = "8/p7/2k5/8/4K3/8/8/8 b - - 0 1"                  # the same from Black's side
    for fen in (white, black):
        v = lambda t: A.vocab_values([fen], t)[0]
        assert v("(fileof W)") == 2 and v("(rankof W)") == 2 and v("(rankof P)") == 1
        assert v("(dfile L W)") == 7 + 2 and v("(drank L W)") == 7 + 2   # their king two files right, two ranks ahead
        assert v("(drank W P)") == 8 and v("(dfile W P)") == 9           # our king one rank ahead of the pawn, two files right


def test_the_opposition_is_a_composition():
    # kings face each other on a file, one square between: same file (7), two ranks apart (9)
    v = lambda fen, t: A.vocab_values([fen], t)[0]
    fen = "8/8/4k3/8/4K3/4P3/8/8 w - - 0 1"
    assert v(fen, "(dfile L W)") == 7 and v(fen, "(drank L W)") == 9
    assert v(fen, "(drank W P)") == 8                                     # our king ahead of the pawn


@pytest.mark.skipif(not Path("lean/ideas/.lake").exists(), reason="the Lean project is not built")
def test_relative_geometry_lean_renderings_typecheck(tmp_path):
    terms = ["(fileof W)", "(rankof P)", "(dfile L W)", "(drank W P)"]
    body = "\n".join(f"def t{i} (p : Position) (c : Color) (W L P O : Square) (hist : List Position) : Nat := {V.lean(V.parse(t))}" for i, t in enumerate(terms))
    src = tmp_path / "VocabCheck.lean"
    src.write_text(f"import Ideas.Base.Vocabulary\nset_option linter.unusedVariables false\nopen Chess Ideas.Vocabulary\n{body}\n")
    r = subprocess.run(["lake", "env", "lean", str(src)], cwd="lean/ideas", capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr


# THE VOCABULARY COMPLETED (Kaveh 2026-10-13: "add all of these ... and ... regular chess terms like pin ... in terms of
# simpler vocabulary"): every piece named, pawn direction, offsets, lines through pieces, control and exchanges, material and
# the game's context, time, combinators -- and the chess terms as compositions of them

F1 = "6k1/5p2/8/8/1r6/8/3P4/R3R1K1 w - - 0 1"     # ours: K g1, R a1, R e1, P d2; theirs: K g8, R b4, P f7


def test_every_piece_named_and_the_pawn_words():
    v = lambda t: A.vocab_values([F1], t)[0]
    assert (v("(fileof P)"), v("(fileof P2)"), v("(fileof P3)"), v("(fileof O)"), v("(fileof O2)")) == (0, 4, 3, 1, 5)
    assert v("(count ourrooks)") == 2 and v("(count theirpawns)") == 1
    assert v("(count (ahead P3))") == 6 and v("(rankof (front P3))") == 2 and v("(rankof (promo P3))") == 7
    assert v("(pushes P3)") == 6 and v("(pushes O2)") == 6 and v("(count (passer P3))") == 18
    assert (v("(fileof (offset W -1 2))"), v("(rankof (offset W -1 2))")) == (5, 2)


def test_lines_control_and_context():
    v = lambda t: A.vocab_values([F1], t)[0]
    assert v("(count (beyond P2 W))") == 1 and v("(defenders (front O))") == 1 and v("(see O)") == 0
    assert v("(count (moves L))") == 4 and v("material_us") == 11 and v("material_them") == 6
    assert v("(diff (pushes P3) (rankof W))") == 6 and v("(eq (fileof W) 6)") == 1
    pin = "4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1"
    assert A.vocab_values([pin], "(rankof (next W P))")[0] == 7 and A.vocab_values([pin], "(count (and (xray O) (at e1)))")[0] == 1
    A.set_vocab_line(["8/8/8/4k3/8/8/1R6/4K3 w - - 0 1", "8/8/8/4k3/8/8/1R6/4K3 b - - 1 1", "8/8/4k3/8/8/8/1R6/4K3 w - - 2 2"])
    assert A.vocab_values(["8/8/4k3/8/8/8/1R6/4K3 w - - 2 2"], "(ago 1 (rankof L))")[0] == 4
    A.set_vocab_line([])


def test_the_chess_terms_unfold_into_the_base():
    pin = "4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1"                 # our knight e2 pinned by the e8 rook
    assert A.vocab_values([pin], "(pinned_ours P)")[0] == 1
    assert A.vocab_values(["4r1k1/8/8/8/8/8/3N4/4K3 w - - 0 1"], "(pinned_ours P)")[0] == 0
    assert A.vocab_values(["8/8/4k3/8/4K3/4P3/8/8 w - - 0 1"], "opposition")[0] == 1
    assert A.vocab_values(["8/8/4k3/8/3K4/4P3/8/8 w - - 0 1"], "opposition")[0] == 0
    assert A.vocab_values(["7k/8/8/8/P7/8/8/K7 w - - 0 1"], "(outside_square P)")[0] == 1      # h8 king, a4 pawn: 4 to go, 7 away
    assert A.vocab_values(["8/8/8/2k5/P7/8/8/K7 w - - 0 1"], "(outside_square P)")[0] == 0
    assert V.gloss(V.parse("(pinned_ours P)")) == "whether our other piece is pinned to our king (1 or 0)"


@pytest.mark.skipif(not Path("lean/ideas/.lake").exists(), reason="the Lean project is not built")
def test_the_completed_vocabulary_renders_in_lean(tmp_path):
    terms = ["(fileof P2)", "(count ourrooks)", "(count (ahead P3))", "(rankof (front O2))", "(dist L (promo P3))", "(pushes O2)",
             "(count (passer P))", "(dist W (offset P -1 2))", "(count (beyond P2 W))", "(rankof (next W P))", "(count (xray O))",
             "(attackers P)", "(defenders (front P3))", "(see O)", "(count (moves L))", "material_us", "halfmoves", "seen",
             "(ago 2 (rankof L))", "(min (pushes P3) (pushes O2))", "(diff (fileof W) (fileof L))", "(eq (fileof W) 6)",
             "(count (or ours theirs))", "forcing_us_5", "forcing_them_mate"]
    terms += [V.sexpr(t) for t in V.layer1() if isinstance(t, tuple) and t[0] in V.CHESS_TERMS][:40] + ["opposition", "flights"]
    body = "\n".join(f"def t{i} (p : Position) (c : Color) (W L P O : Square) (hist : List Position) : Nat := {V.lean(V.parse(t))}" for i, t in enumerate(terms))
    src = tmp_path / "VocabCheck.lean"
    src.write_text(f"import Ideas.Base.Vocabulary\nset_option linter.unusedVariables false\nopen Chess Ideas.Vocabulary\n{body}\n")
    r = subprocess.run(["lake", "env", "lean", str(src)], cwd="lean/ideas", capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[:3000] + r.stderr[:2000]


# WITHOUT (Kaveh 2026-10-14, "what each piece's job is"): his position, Black to move; mirrored so the mover is White
KAVEH = "5k1r/1p1n1p2/p1qb1p1p/4p3/2rP2PP/2PQ1PB1/PP2NK2/1R5R b - - 0 1"


def test_without_and_job_on_kavehs_position():
    import chess
    import analyst_core as A
    V.install_terms()
    fen = chess.Board(KAVEH).mirror().fen()                        # the queen on c3, the rook on c5, the bishop on d3, the pawn on e4
    val = lambda t: A.vocab_values([fen], t)[0]
    job_qr, job_pb = val("(job (at c3) c5)"), val("(job (at e4) d3)")
    print(f"OBS without: see(c5) {val('(see c5)')}, see(c5) without the queen {val('(without (at c3) (see c5))')}, job(queen c3, rook c5) {job_qr}; "
          f"see(d3) {val('(see d3)')}, job(pawn e4, bishop d3) {job_pb}")
    assert job_qr > 0                                              # the queen is the rook's only guard against the queen
    assert job_pb == 0                                             # plain see lets the queen recapture on the bishop: the gap the effective exchange closes
    for t in ("(count ours)", "(see c5)", "(dist W P)", "(count (reach W free))"):
        assert val(f"(without (and ours theirs) {t})") == val(t)    # nothing in the set: T unchanged
    assert val("(without (at c3) (count ours))") == val("(count ours)") - 1
    # kings are never removed; a blocker taken off may leave a king attacked (built relaxed, read as it stands)
    assert val("(without (or ours theirs) (count (or (at W) (at L))))") == 2


@pytest.mark.skipif(not Path("lean/ideas/.lake").exists(), reason="the Lean project is not built")
def test_without_lean_rendering_typechecks(tmp_path):
    terms = ["(without (at P2) (see O))", "(job (at P) O)", "(without ours (dist W P))", "(without theirrooks (count (reach W unhit)))"]
    body = "\n".join(f"def t{i} (p : Position) (c : Color) (W L P O : Square) (hist : List Position) : Nat := {V.lean(V.parse(t))}" for i, t in enumerate(terms))
    src = tmp_path / "VocabCheckWithout.lean"
    src.write_text(f"import Ideas.Base.Vocabulary\nset_option linter.unusedVariables false\nopen Chess Ideas.Vocabulary\n{body}\n")
    r = subprocess.run(["lake", "env", "lean", str(src)], cwd="lean/ideas", capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
