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
    body = "\n".join(f"def t{i} (p : Position) (c : Color) (W L P O : Square) : Nat := {V.lean(V.parse(t))}" for i, t in enumerate(terms))
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
    body = "\n".join(f"def t{i} (p : Position) (c : Color) (W L P O : Square) : Nat := {V.lean(V.parse(t))}" for i, t in enumerate(terms))
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
    body = "\n".join(f"def t{i} (p : Position) (c : Color) (W L P O : Square) : Nat := {V.lean(V.parse(t))}" for i, t in enumerate(terms))
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
    body = "\n".join(f"def t{i} (p : Position) (c : Color) (W L P O : Square) : Nat := {V.lean(V.parse(t))}" for i, t in enumerate(terms))
    src = tmp_path / "VocabCheck.lean"
    src.write_text(f"import Ideas.Base.Vocabulary\nset_option linter.unusedVariables false\nopen Chess Ideas.Vocabulary\n{body}\n")
    r = subprocess.run(["lake", "env", "lean", str(src)], cwd="lean/ideas", capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
