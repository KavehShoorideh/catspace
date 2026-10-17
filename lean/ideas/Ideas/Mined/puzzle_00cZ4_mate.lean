import Ideas.Base.Vocabulary
import Ideas.TreeStep
import Ideas.Root
import Ideas.Mined.Schemas

/-! MINED by catspace/incubator/analyst/discovery/selfmine.py (puzzle_00cZ4_mate): each theorem is a conjectured PREMISE over the base vocabulary with the OUTCOME
`forcingOf p c 100000 != 0` (Ideas.Vocabulary.forcingOf: side c, to move, forces a gain of 100000 or more in 1 to 3 moves against every defence).
Squares are named from c's side. A proof that failed stays as `sorry` with the error in the comment above it: the statement is the finding. -/

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 100000

namespace Ideas.Mined.puzzle_00cZ4_mate

open Chess hiding between
open Ideas.Vocabulary

/-- INSTANCE of mate_in_one at 8/2R5/6pk/p3p2p/4Nq2/5n1P/P4RP1/7K b - - 1 41 (b to move): f4->h2 leaves them no legal move, in check -/
theorem mate_in_one_instance (p : Position) (c : Color) (hc : p.turn = c) :
    ((count (fun u => (oset (movesOf p) (some (named c (((5 : Fin 8)), ((4 : Fin 8)))))) u && (oset atSq (some (named c (((7 : Fin 8)), ((6 : Fin 8)))))) u)) ≥ 1) ∧ ((count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some (named c (((5 : Fin 8)), ((4 : Fin 8)))))) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0) ∧ ((count (fun u => (piecesOf p c) u && (oset atSq (some (named c (((5 : Fin 8)), ((4 : Fin 8)))))) u)) ≥ 1) ∧ ((afterMove p (some (named c (((5 : Fin 8)), ((4 : Fin 8))))) (some (named c (((7 : Fin 8)), ((6 : Fin 8))))) (fun p => (moveCountOf p))) ≤ 0) ∧ ((afterMove p (some (named c (((5 : Fin 8)), ((4 : Fin 8))))) (some (named c (((7 : Fin 8)), ((6 : Fin 8))))) (fun p => (checkOf p))) ≥ 1) → forcingOf p c 100000 ≠ 0 := by
  intro hall
  obtain ⟨h0, h1, h2, h3, h4⟩ := hall
  exact Ideas.Mined.Schemas.mate_in_one p c hc (named c ((5 : Fin 8), (4 : Fin 8))) (named c ((7 : Fin 8), (6 : Fin 8))) 100000 h0 h1 h2 h3 h4

end Ideas.Mined.puzzle_00cZ4_mate
