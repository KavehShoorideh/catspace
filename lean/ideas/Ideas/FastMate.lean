import Ideas.Fast
import Ideas.Mate

/-!
# The mate-side tests on the fast primitives

Each test of `Ideas/Mate.lean` again, with the fast attack and ray primitives of `Ideas/Fast.lean`, and a
theorem that it equals the original. Lemmas evaluate these; their soundness theorems rewrite to the
originals.
-/

namespace Ideas

open Chess

def flightOkB' (q : Position) (k d : Square) : Bool :=
  !kingGeometryB k d || !destinationFriendlyB q q.turn d ||
    isAttackedB' (withEmpty q k) (Color.opposite q.turn) d

theorem flightOkB'_eq (q : Position) (k d : Square) : flightOkB' q k d = flightOkB q k d := by
  simp only [flightOkB', flightOkB, isAttackedB'_eq]

def checksB' (q : Position) (k c : Square) : Bool :=
  pieceAttacksB' q (Color.opposite q.turn) c k

theorem checksB'_eq (q : Position) (k c : Square) : checksB' q k c = checksB q k c := by
  simp only [checksB', checksB, pieceAttacksB'_eq]

/-- Only the squares that can attack `c` at all are tried. -/
def noCaptureB' (q : Position) (k c : Square) : Bool :=
  (attackerTable[(sqIndex c).val]'(by rw [attackerTable_size]; exact (sqIndex c).isLt)).all
    (fun s => decide (s = k) || !pieceAttacksB' q q.turn s c)

theorem noCaptureB'_eq (q : Position) (k c : Square) : noCaptureB' q k c = noCaptureB q k c := by
  simp only [noCaptureB', noCaptureB, attackerTable_get, attackerSquares, pieceAttacksB'_eq]
  apply Bool.eq_iff_iff.mpr
  simp only [List.all_eq_true, List.mem_filter, Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true']
  constructor
  · intro h s hs
    by_cases hsk : s = k
    · exact Or.inl hsk
    · by_cases ha : pieceAttacksB q q.turn s c = true
      · exact h s ⟨hs, by simpa using pieceAttacksB_geometry ha⟩
      · exact Or.inr (by simpa using ha)
  · intro h s ⟨hs, _⟩
    exact h s hs

def couldReachEmptyB' (q : Position) (s u : Square) : Bool :=
  match q.board s with
  | some pc =>
      (pc.color == q.turn) && (pc.kind != .king) &&
        (if pc.kind = .pawn then
          pawnMoveGeometryB q.turn s u || pawnDoubleGeometryB q.turn s u
        else pieceAttacksB' q q.turn s u)
  | none => false

theorem couldReachEmptyB'_eq (q : Position) (s u : Square) :
    couldReachEmptyB' q s u = couldReachEmptyB q s u := by
  unfold couldReachEmptyB' couldReachEmptyB
  cases q.board s <;> simp [pieceAttacksB'_eq]

theorem couldReachEmptyB_geometry {q : Position} {s u : Square} (h : couldReachEmptyB q s u = true) :
    (queenGeometryB s u || knightGeometryB s u) = true := by
  unfold couldReachEmptyB at h
  split at h
  · rename_i pc hpc
    simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at h
    obtain ⟨⟨_, _⟩, h⟩ := h
    split at h
    · apply geometry_of_push s u
      cases hc : q.turn <;> rw [hc] at h <;> simp only [Bool.or_eq_true] at h ⊢ <;> tauto
    · exact pieceAttacksB_geometry h
  · simp at h

/-- Only the squares that can reach a between-square at all are tried. -/
def noBlockB' (q : Position) (k c : Square) : Bool :=
  (rayBetween c k).all fun u =>
    (attackerTable[(sqIndex u).val]'(by rw [attackerTable_size]; exact (sqIndex u).isLt)).all
      (fun s => !couldReachEmptyB' q s u)

theorem noBlockB'_eq (q : Position) (k c : Square) : noBlockB' q k c = noBlockB q k c := by
  simp only [noBlockB', noBlockB, rayBetween_eq, attackerTable_get, attackerSquares, couldReachEmptyB'_eq]
  apply Bool.eq_iff_iff.mpr
  simp only [List.all_eq_true, List.mem_filter, Bool.not_eq_true']
  constructor
  · intro h u hu s hs
    by_cases hr : couldReachEmptyB q s u = true
    · exact h u hu s ⟨hs, by simpa using couldReachEmptyB_geometry hr⟩
    · simpa using hr
  · intro h u hu s ⟨hs, _⟩
    exact h u hu s hs

end Ideas
