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
  another, the squares a side attacks (a king lifted off the board), the colour of a square.
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

/-- **colour of a square**: light or dark. -/
def squareColour (s : Square) : Nat := GeoLemmas.colour s

/-- **count**: how many squares a set holds. -/
def count (A : SquareSet) : Nat := (allSquares.filter A).length

/-- **reach**: the squares a king on `s` walks to by king steps over the squares of `A` (and `s`), at most 64 steps. -/
def reach (s : Square) (A : SquareSet) : SquareSet :=
  let step (R : SquareSet) : SquareSet := fun u => R u || (A u && allSquares.any (fun v => R v && dist v u == 1))
  Nat.repeat step 64 (fun u => u == s)

end Ideas.Vocabulary
