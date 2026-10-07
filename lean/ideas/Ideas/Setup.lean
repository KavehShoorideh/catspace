import Ideas.Passer2

/-!
# Composition: a move that SETS UP a proved idea (Kaveh 2026-10-06: "compose certificates")

The deep families -- the passer: promotion whatever they do, proved by an invariant -- are stated at our turn. A
move `m` sets one up when, whatever they reply, the idea's hypotheses hold at our next turn. `setsUpPasser2_sound`
composes `unstoppablePasser2_sound` over that one enumerated ply: the promotion is then forced from the position
after `m`, one ply later than the idea's own bound. Found on development miss 00n9d (f6!): the pawn on f5 fails
`pawnsFar` only because an enemy pawn attacks f5 itself -- the square it leaves on the first push.
-/

namespace Ideas

open Chess

/-- After `m`, every legal reply leaves the second-form passer on `sq` with all its hypotheses holding. -/
def setsUpPasser2 (p : Position) (m : Move) (sq : Square) : Bool :=
  let q := applyMove p m
  let ms := legalMoves q
  !ms.isEmpty && ms.all (fun r => unstoppablePasser2.holds (applyMove q r) ⟨sq⟩)

/-- The pawn on `sq` promotes within `2 * pushes - 1 + 1` plies after `m`, whatever they do. -/
theorem setsUpPasser2_sound (p : Position) (m : Move) (sq : Square) (h : setsUpPasser2 p m sq = true) :
    forcedWithinB p.turn (Passer.promoted p.turn sq.1) (applyMove p m) (2 * Passer.pushes p.turn sq - 1 + 1) = true := by
  have hq : (applyMove p m).turn ≠ p.turn := by
    rw [applyMove_turn]; cases p.turn <;> simp [Color.opposite]
  have hr : ∀ r : Move, (applyMove (applyMove p m) r).turn = p.turn := by
    intro r
    rw [applyMove_turn, applyMove_turn, Color.opposite_opposite]
  rw [Passer.forcedWithinB_theirs hq]
  simp only [setsUpPasser2, Bool.and_eq_true, Bool.not_eq_true', List.all_eq_true] at h
  obtain ⟨hne, hall⟩ := h
  have hall' : (legalMoves (applyMove p m)).all (fun r =>
      forcedWithinB p.turn (Passer.promoted p.turn sq.1) (applyMove (applyMove p m) r)
        (2 * Passer.pushes p.turn sq - 1)) = true := by
    rw [List.all_eq_true]
    intro r hrm
    have hs := Passer2.unstoppablePasser2_sound (applyMove (applyMove p m) r) ⟨sq⟩ (hall r hrm)
    rw [hr r] at hs
    exact hs
  rw [hall']
  simp [hne]

end Ideas
