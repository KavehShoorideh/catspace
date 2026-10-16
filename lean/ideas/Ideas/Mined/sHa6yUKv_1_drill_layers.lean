import Ideas.Base.Vocabulary
import Ideas.TreeStep
import Ideas.Root
import Ideas.Mined.Schemas

/-! MINED by catspace/incubator/analyst/discovery/selfmine.py (sHa6yUKv_1_drill_layers): each theorem is a conjectured PREMISE over the base vocabulary with the OUTCOME
`forcingOf p c 6 != 0` (Ideas.Vocabulary.forcingOf: side c, to move, forces a gain of 6 or more in 1 to 3 moves against every defence).
Squares are named from c's side. A proof that failed stays as `sorry` with the error in the comment above it: the statement is the finding. -/

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 100000

namespace Ideas.Mined.sHa6yUKv_1_drill_layers

open Chess hiding between
open Ideas.Vocabulary

/-- INSTANCE of delayed_balance_d2 {'x1': 'f6', 'y1': 'c3', 'x2': 'c3', 'y2': 'a1'}, B=65, T=71, v=6 -/
theorem delayed_balance_d2_instance (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((5 : Fin 8)), ((2 : Fin 8))))) (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (fun p => (leastOver p (fun p => (count (fun u => (oset (movesOf p) (some (named c (((2 : Fin 8)), ((5 : Fin 8)))))) u && (oset atSq (some (named c (((0 : Fin 8)), ((7 : Fin 8)))))) u)))))) ≥ 1) ∧ ((afterMove p (some (named c (((5 : Fin 8)), ((2 : Fin 8))))) (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (fun p => (leastOver p (fun p => (count (fun u => (piecesOf p c) u && (oset atSq (some (named c (((2 : Fin 8)), ((5 : Fin 8)))))) u)))))) ≥ 1) ∧ ((afterMove p (some (named c (((5 : Fin 8)), ((2 : Fin 8))))) (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (fun p => (mostOver p (fun p => (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some (named c (((2 : Fin 8)), ((5 : Fin 8)))))) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)))))) ≤ 0) ∧ ((count (fun u => (oset (movesOf p) (some (named c (((5 : Fin 8)), ((2 : Fin 8)))))) u && (oset atSq (some (named c (((2 : Fin 8)), ((5 : Fin 8)))))) u)) ≥ 1) ∧ ((count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some (named c (((5 : Fin 8)), ((2 : Fin 8)))))) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0) ∧ ((count (fun u => (piecesOf p c) u && (oset atSq (some (named c (((5 : Fin 8)), ((2 : Fin 8)))))) u)) ≥ 1) ∧ ((balanceOf p c) ≤ 65) ∧ ((afterMove p (some (named c (((5 : Fin 8)), ((2 : Fin 8))))) (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (fun p => (leastOver p (fun p => (afterMove p (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (fun p => (leastOver p (fun p => (mostOver p (fun p => (leastOver p (fun p => (balanceOf p c))))))))))))) ≥ 71) → forcingOf p c 6 ≠ 0 := by
  intro hall
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7⟩ := hall
  exact Ideas.Mined.Schemas.delayed_balance_d2 p c  hc (named c ((5 : Fin 8), (2 : Fin 8))) (named c ((2 : Fin 8), (5 : Fin 8))) (named c ((2 : Fin 8), (5 : Fin 8))) (named c ((0 : Fin 8), (7 : Fin 8))) 65 71 6 (by omega) (by omega) h0 h1 h2 h3 h4 h5 h6 h7

end Ideas.Mined.sHa6yUKv_1_drill_layers
