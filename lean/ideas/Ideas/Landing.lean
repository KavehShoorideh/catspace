import Ideas.Mate
import Ideas.GeoLemmas

/-!
# The landing: a piece placed on a square, the other side to move (ANALYST_PLAN 52.18 (1), Kaveh 2026-10-19)

"The target is a square where if my piece lands, I win -- mate, a place from which mate is inevitable, a double attack,
capturing a piece directly; ... if the piece lands here, I'm lost; if the piece lands here, I win."

A LANDING of side `c` is the counterfactual the roots already use (`placement_of` / `future_board`): the piece `pc` lifted
from `frm` and put on `to`, nothing else moved, the OTHER side to move (`land`). A landing is a TARGET of `c` when a claim
of `c` binds in it with no defect left (`Target`); a THREAT of the other side when the same holds from its side. The table
of landing answers is a memo keyed by the landing and the pieces on the squares the answer READ: `land_agree` is why that key
is enough -- two positions that agree on the read squares give landings that agree there, so an answer that reads only those
squares is the same answer.
-/

namespace Ideas.Landing

open Chess

/-- THE LANDING: `pc` lifted from `frm`, placed on `to`, the side other than `c` to move (no en-passant square, as a counterfactual
has no history). -/
def land (p : Position) (c : Color) (pc : Piece) (frm dst : Square) : Position :=
  { p with board := boardUpdate (boardUpdate p.board frm none) dst (some pc), turn := Color.opposite c, enPassant := none }

/-- The piece stands on its landing square. -/
theorem land_board_to (p : Position) (c : Color) (pc : Piece) (frm dst : Square) : (land p c pc frm dst).board dst = some pc := by
  simp [land, boardUpdate]

/-- THEM to move. -/
theorem land_turn (p : Position) (c : Color) (pc : Piece) (frm dst : Square) : (land p c pc frm dst).turn = Color.opposite c := rfl

/-- Nothing else moved: every square but the two keeps its piece. -/
theorem land_board_other (p : Position) (c : Color) (pc : Piece) (frm dst s : Square) (h1 : s ≠ frm) (h2 : s ≠ dst) :
    (land p c pc frm dst).board s = p.board s := by
  simp [land, boardUpdate, h1, h2]

/-- THE KEY OF THE TABLE: two positions that agree on a set of squares give landings that agree on it and on the two squares of
the landing -- the answer to a question that reads only those squares needs nothing else to be recomputed. -/
theorem land_agree (p p' : Position) (c : Color) (pc : Piece) (frm dst : Square)
    (S : Square → Prop) (h : ∀ s, S s → p.board s = p'.board s) :
    ∀ s, (s = frm ∨ s = dst ∨ S s) → (land p c pc frm dst).board s = (land p' c pc frm dst).board s := by
  intro s hs
  by_cases h1 : s = frm
  · subst h1
    by_cases h2 : s = dst <;> simp [land, boardUpdate, h2]
  · by_cases h2 : s = dst
    · subst h2; simp [land, boardUpdate]
    · rw [land_board_other p c pc frm dst s h1 h2, land_board_other p' c pc frm dst s h1 h2]
      rcases hs with hs | hs | hs
      · exact absurd hs h1
      · exact absurd hs h2
      · exact h s hs

/-- A TARGET of `c`: a claim binds in the landing (the claim is a statement about a position: a mate, an exchange won, a fork, a
promotion by the passer lemma). The same word read from the other side is a THREAT. -/
def Target (claim : Position → Prop) (p : Position) (c : Color) (pc : Piece) (frm dst : Square) : Prop :=
  claim (land p c pc frm dst)

/-- A landing that mates is a target. -/
theorem target_of_checkmate (p : Position) (c : Color) (pc : Piece) (frm dst : Square) (h : checkmate (land p c pc frm dst)) :
    Target (fun q => checkmate q) p c pc frm dst := h

end Ideas.Landing
