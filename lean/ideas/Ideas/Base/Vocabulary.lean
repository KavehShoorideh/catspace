import Chess.Game
import Ideas.Mate
import Ideas.GeoLemmas
import Ideas.Passer

/-!
# The base vocabulary (Kaveh 2026-10-12)

"The important thing to give it is the vocabulary because that's what maps to human understanding in the end, enabling
interpretability. You can build arbitrary lemmas on top of it, but the base has to be deterministic and interpretable."

The discovery system (catspace/incubator/analyst/proposer.py, its evaluator rust/analyst_core/src/vocab.rs) builds every
concept as a term over these primitives and nothing else; each is a deterministic definition with a plain gloss, so a
discovered concept explains itself by unfolding into these words. Three layers:

* THE RULES -- squares, pieces, moves, attacks: the chess library's own definitions (`pieceAttacksB`, `isAttackedB`,
  `legalMoves`, `inCheckB`), used as they are.
* PERCEPTION -- what a player sees at a glance (below): distance in king steps, the king's box, the edge and the corners,
  a rank and a file, the line through two squares and the squares between them, the squares nearer one square than
  another, the squares a side attacks (a king lifted off the board), the squares of a side's pieces, the colour of a
  square. Both sides have their words: our king W, their king L, our other piece P, their other piece O.
* COMBINATORS -- count, the set where something holds, union / intersection / complement, sums, and REACH: the squares a
  king walks to by king steps over a set.

Chess concepts (the wall, the opposition, the box a queen cuts, the front zone, the side room) are deliberately NOT here:
they must be built from this base, or the rediscovery tests are void.
-/

namespace Ideas.Vocabulary

open Chess

/-- A set of squares, as the board's squares where a test holds. -/
abbrev SquareSet := Square → Bool

/-- **distance**: how many king steps from `s` to `t`. -/
def dist (s t : Square) : Nat := Passer.kdist s t

/-- **box**: the squares a king on `s` touches, and `s` itself (its 3×3 box, on the board). -/
def box (s : Square) : SquareSet := fun t => decide (dist s t ≤ 1)

/-- **edge**: the squares on the board's edge. -/
def edge : SquareSet := fun t => t.1.val = 0 || t.1.val = 7 || t.2.val = 0 || t.2.val = 7

/-- **corner**: the four corner squares. -/
def corner : SquareSet := fun t => (t.1.val = 0 || t.1.val = 7) && (t.2.val = 0 || t.2.val = 7)

/-- **rank of**: the squares on `s`'s rank. -/
def rankOf (s : Square) : SquareSet := fun t => decide (t.2 = s.2)

/-- **file of**: the squares on `s`'s file. -/
def fileOf (s : Square) : SquareSet := fun t => decide (t.1 = s.1)

/-- **line through**: the squares on the line (rank, file or diagonal) through `s` and `t`, when they share one; none
otherwise. -/
def lineThrough (s t : Square) : SquareSet := fun u =>
  queenGeometryB s t && (u == s || u == t || (queenGeometryB s u && queenGeometryB t u &&
    ((sameFileB s t && sameFileB s u) || (sameRankB s t && sameRankB s u) ||
     (sameDiagonalB s t && sameDiagonalB s u && sameDiagonalB t u))))

/-- **between**: the squares strictly between `s` and `t` on their line. -/
def between (s t : Square) : SquareSet := fun u => betweenB s u t

/-- **nearer**: the squares nearer to `s` than to `t` (in king steps) -- `s`'s side of the board against `t`. -/
def nearer (s t : Square) : SquareSet := fun u => decide (dist u s < dist u t)

/-- **attacked by**: the squares side `c` attacks with the king on `k` lifted off the board (it cannot hide behind
itself). -/
def attackedBy (p : Position) (c : Color) (k : Square) : SquareSet := fun u => isAttackedB (withEmpty p k) c u

/-- **pieces of**: the squares holding a piece of side `c` other than its king (Kaveh 2026-10-12: "base vocab should be
complete with words for their pieces" -- `ours` is `piecesOf p c`, `theirs` is `piecesOf p c.opposite`; the squares
they attack, `hit`, is `attackedBy p c.opposite W`, our king lifted). -/
def piecesOf (p : Position) (c : Color) : SquareSet := fun u =>
  match p.board u with
  | some pc => decide (pc.color = c) && decide (pc.kind ≠ PieceKind.king)
  | none => false

-- GEOMETRY AND PIECES, a second batch (Kaveh 2026-10-12: "we want to be able to specify sets of squares ... along a
-- diagonal or along a rank or a file from here to here ... reachable by a knight, one hop or two hops ... the four squares
-- on the center ... any particular square should be able to be named, but ... the symmetry ... except for the last rank
-- where there's promotion ... some per piece vocabulary"; a time vocabulary later)

/-- **lines**: the squares on `s`'s rank or file (a rook's lines through `s`, blockers ignored). -/
def lines (s : Square) : SquareSet := fun t => rookGeometryB s t

/-- **diagonals**: the squares on the diagonals through `s` (a bishop's lines, blockers ignored). -/
def diagonals (s : Square) : SquareSet := fun t => bishopGeometryB s t

/-- **segment**: the squares from `s` to `t` along the rank, file or diagonal they share, both ends included (none when they
share no line). -/
def segment (s t : Square) : SquareSet := fun u => queenGeometryB s t && (u == s || u == t || betweenB s u t)

/-- **one knight hop**: the squares a knight on `s` reaches. -/
def knight1 (s : Square) : SquareSet := fun t => knightGeometryB s t

/-- **two knight hops**: the squares a knight on `s` reaches in two hops. -/
def knight2 (s : Square) : SquareSet := fun t => allSquares.any (fun u => knightGeometryB s u && knightGeometryB u t)

/-- **the centre**: d4, e4, d5, e5. -/
def center : SquareSet := fun t => (t.1.val == 3 || t.1.val == 4) && (t.2.val == 3 || t.2.val == 4)

-- a square's colour (Kaveh 2026-10-13): a1 is dark -- file + rank even
/-- the light squares -/
def light : SquareSet := fun t => (t.1.val + t.2.val) % 2 == 1
/-- the dark squares -/
def dark : SquareSet := fun t => (t.1.val + t.2.val) % 2 == 0
/-- the squares of `s`'s colour -/
def colour (s : Square) : SquareSet := fun t => (t.1.val + t.2.val) % 2 == (s.1.val + s.2.val) % 2

/-- **attacks from**: the squares the piece standing on `s` attacks, for its own colour, with the board's blockers (the chess
library's `pieceAttacksB`); none when `s` is empty. -/
def attacksFrom (p : Position) (s : Square) : SquareSet := fun t =>
  match p.board s with
  | some q => pieceAttacksB p q.color s t
  | none => false

/-- The board's eight symmetries of a square (rotations and reflections). -/
def symmetries (s : Square) : List Square :=
  let (f, r) := s
  [(f, r), (Fin.rev f, r), (f, Fin.rev r), (Fin.rev f, Fin.rev r), (r, f), (Fin.rev r, f), (r, Fin.rev f), (Fin.rev r, Fin.rev f)]

/-- **the class of a square**: the squares like `s` under the board's symmetries -- all eight when no pawn is on the board,
only the left-right mirror when one is (the promotion direction breaks the rest); `s` is named from side `c`'s view (its
first rank, rank 1). -/
def cls (p : Position) (c : Color) (s : Square) : SquareSet := fun u =>
  let pawns := allSquares.any (fun v => match p.board v with | some q => q.kind == .pawn | none => false)
  let abs : Square := if c == Color.white then s else (s.1, Fin.rev s.2)
  if pawns then u == abs || u == (Fin.rev abs.1, abs.2) else (symmetries abs).contains u

/-- **a named square**: the square `s` named from side `c`'s view -- its first rank is rank 1, so `a8` is its promotion
corner on the a-file (Kaveh 2026-10-13: "the squares individually available"; the board's symmetries are a lemma of their
own, not part of the words). -/
def named (c : Color) (s : Square) : Square := if c == Color.white then s else (s.1, Fin.rev s.2)

/-- **the square itself**, as a set. -/
def atSq (s : Square) : SquareSet := fun u => u == s

/-- **colour of a square**: light or dark. -/
def squareColour (s : Square) : Nat := GeoLemmas.colour s

/-- **count**: how many squares a set holds. -/
def count (A : SquareSet) : Nat := (allSquares.filter A).length

/-- **reach**: the squares a king on `s` walks to by king steps over the squares of `A` (and `s`), at most 64 steps. -/
def reach (s : Square) (A : SquareSet) : SquareSet :=
  let step (R : SquareSet) : SquareSet := fun u => R u || (A u && allSquares.any (fun v => R v && dist v u == 1))
  Nat.repeat step 64 (fun u => u == s)

end Ideas.Vocabulary
