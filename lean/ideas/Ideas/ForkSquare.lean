import Chess

/-!
# The double threat of a landing: one move touches one of the two (ANALYST_PLAN 52.18 (4))

A fork is a landing that attacks two pieces of theirs, on `a` and on `b`. For the double threat to be one move too many, their
single move must not change both: `one_move_touches_one` -- a move other than a castle or en passant has a source and a
destination only, so it can touch both squares only as the move from one to the other (a piece of theirs taking the other, which
is the "move of one attacked piece that takes / guards the other" the fork root names as a defect). The other answers a fork has
-- a capture of the forking piece, a check -- are moves whose destination or effect is not on `a` or `b`; they are read by the
rules, root by root, not by this lemma.
-/

namespace Ideas.ForkSquare

open Chess

/-- A move touches a square: it leaves it or lands on it. -/
def touches (m : Move) (a : Square) : Prop := m.source = a ∨ m.destination = a

/-- A move that is not a castle and not en passant touches both of two different squares only as the move from one to the other. -/
theorem one_move_touches_one (m : Move) (hnc : m.isCastle = false) (hne : ∀ s d, m ≠ Move.enPassant s d) (a b : Square) (hab : a ≠ b)
    (ha : touches m a) (hb : touches m b) : (m.source = a ∧ m.destination = b) ∨ (m.source = b ∧ m.destination = a) := by
  unfold touches at ha hb
  rcases ha with ha | ha <;> rcases hb with hb | hb
  · exact absurd (ha.symm.trans hb) hab
  · exact Or.inl ⟨ha, hb⟩
  · exact Or.inr ⟨hb, ha⟩
  · exact absurd (ha.symm.trans hb) hab

end Ideas.ForkSquare
