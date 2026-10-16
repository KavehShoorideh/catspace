import Ideas.Base.Vocabulary
import Ideas.TreeStep
import Ideas.Root
import Ideas.Mined.Schemas

/-! MINED by catspace/incubator/analyst/discovery/selfmine.py (g_46LzXjoz_1_layers): each theorem is a conjectured PREMISE over the base vocabulary with the OUTCOME
`forcingOf p c 9 != 0` (Ideas.Vocabulary.forcingOf: side c, to move, forces a gain of 9 or more in 1 to 3 moves against every defence).
Squares are named from c's side. A proof that failed stays as `sorry` with the error in the comment above it: the statement is the finding. -/

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 100000

namespace Ideas.Mined.g_46LzXjoz_1_layers

open Chess hiding between
open Ideas.Vocabulary

/-- INSTANCE of delayed_balance_d1 {'x1': 'c4', 'y1': 'f1'}, B=67, T=76, v=9 -/
theorem delayed_balance_d1_instance (p : Position) (c : Color) (hc : p.turn = c) :
    ((count (fun u => (oset (movesOf p) (some (named c (((2 : Fin 8)), ((4 : Fin 8)))))) u && (oset atSq (some (named c (((5 : Fin 8)), ((7 : Fin 8)))))) u)) ≥ 1) ∧ ((count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some (named c (((2 : Fin 8)), ((4 : Fin 8)))))) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0) ∧ ((count (fun u => (piecesOf p c) u && (oset atSq (some (named c (((2 : Fin 8)), ((4 : Fin 8)))))) u)) ≥ 1) ∧ ((balanceOf p c) ≤ 67) ∧ ((afterMove p (some (named c (((2 : Fin 8)), ((4 : Fin 8))))) (some (named c (((5 : Fin 8)), ((7 : Fin 8))))) (fun p => (leastOver p (fun p => (mostOver p (fun p => (leastOver p (fun p => (balanceOf p c))))))))) ≥ 76) → forcingOf p c 9 ≠ 0 := by
  intro hall
  obtain ⟨h0, h1, h2, h3, h4⟩ := hall
  exact Ideas.Mined.Schemas.delayed_balance_d1 p c  hc (named c ((2 : Fin 8), (4 : Fin 8))) (named c ((5 : Fin 8), (7 : Fin 8))) 67 76 9 (by omega) (by omega) h0 h1 h2 h3 h4

end Ideas.Mined.g_46LzXjoz_1_layers
