import Ideas.Base.Vocabulary
import Ideas.TreeStep
import Ideas.Root
import Ideas.Mined.Schemas

/-! MINED by catspace/incubator/analyst/discovery/selfmine.py (puzzle_01IFU_mate2): each theorem is a conjectured PREMISE over the base vocabulary with the OUTCOME
`forcingOf p c 100000 != 0` (Ideas.Vocabulary.forcingOf: side c, to move, forces a gain of 100000 or more in 1 to 3 moves against every defence).
Squares are named from c's side. A proof that failed stays as `sorry` with the error in the comment above it: the statement is the finding. -/

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 100000

namespace Ideas.Mined.puzzle_01IFU_mate2

open Chess hiding between
open Ideas.Vocabulary

/-- INSTANCE of mate_in_two at r1b2rk1/pp3pp1/3b3p/3Bn3/P4q2/2P4P/1P2NPP1/RN1Q1RK1 b - - 2 17 (b to move): e5->f3, then over every reply a mate -/
theorem mate_in_two_instance (p : Position) (c : Color) (hc : p.turn = c) :
    ((count (fun u => (oset (movesOf p) (some (named c (((4 : Fin 8)), ((3 : Fin 8)))))) u && (oset atSq (some (named c (((5 : Fin 8)), ((5 : Fin 8)))))) u)) ≥ 1) ∧ ((count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some (named c (((4 : Fin 8)), ((3 : Fin 8)))))) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0) ∧ ((count (fun u => (piecesOf p c) u && (oset atSq (some (named c (((4 : Fin 8)), ((3 : Fin 8)))))) u)) ≥ 1) ∧ ((afterMove p (some (named c (((4 : Fin 8)), ((3 : Fin 8))))) (some (named c (((5 : Fin 8)), ((5 : Fin 8))))) (fun p => (leastOver p (fun p => (mostOver p (fun p => (min (checkOf p) (if (moveCountOf p) = (0) then 1 else 0)))))))) ≥ 1) → forcingOf p c 100000 ≠ 0 := by
  intro hall
  obtain ⟨h0, h1, h2, h3⟩ := hall
  exact Ideas.Mined.Schemas.mate_in_two p c hc (named c ((4 : Fin 8), (3 : Fin 8))) (named c ((5 : Fin 8), (5 : Fin 8))) 100000 h0 h1 h2 h3

end Ideas.Mined.puzzle_01IFU_mate2
