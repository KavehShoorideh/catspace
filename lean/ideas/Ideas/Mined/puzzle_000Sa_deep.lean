import Ideas.Base.Vocabulary
import Ideas.TreeStep
import Ideas.Root
import Ideas.Mined.Schemas

/-! MINED by catspace/incubator/analyst/discovery/selfmine.py (puzzle_000Sa_deep): each theorem is a conjectured PREMISE over the base vocabulary with the OUTCOME
`forcingOf p c 2 != 0` (Ideas.Vocabulary.forcingOf: side c, to move, forces a gain of 2 or more in 1 to 3 moves against every defence).
Squares are named from c's side. A proof that failed stays as `sorry` with the error in the comment above it: the statement is the finding. -/

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 100000

namespace Ideas.Mined.puzzle_000Sa_deep

open Chess hiding between
open Ideas.Vocabulary

/-- INSTANCE of delayed_balance_deep_alive_d1 at 2Q2bk1/5p1p/p5p1/2p3P1/4B3/7P/qPr2P2/2K4R w - - 0 33 (w to move): e4->c2, the balance 63 now, the tail reads 65 -/
theorem delayed_balance_deep_alive_d1_instance (p : Position) (c : Color) (hc : p.turn = c) :
    ((count (fun u => (oset (movesOf p) (some (named c (((4 : Fin 8)), ((3 : Fin 8)))))) u && (oset atSq (some (named c (((2 : Fin 8)), ((1 : Fin 8)))))) u)) ≥ 1) ∧ ((count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some (named c (((4 : Fin 8)), ((3 : Fin 8)))))) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0) ∧ ((count (fun u => (piecesOf p c) u && (oset atSq (some (named c (((4 : Fin 8)), ((3 : Fin 8)))))) u)) ≥ 1) ∧ ((balanceOf p c) ≤ 63) ∧ ((afterMove p (some (named c (((4 : Fin 8)), ((3 : Fin 8))))) (some (named c (((2 : Fin 8)), ((1 : Fin 8))))) (fun p => (leastOver p (fun p => (mostOver p (fun p => (leastOver p (fun p => (mostOver p (fun p => (leastOver p (fun p => (min (balanceOf p c) ((materialOf p c) + (61))))))))))))))) ≥ 65) → forcingOf p c 2 ≠ 0 := by
  intro hall
  obtain ⟨h0, h1, h2, h3, h4⟩ := hall
  exact Ideas.Mined.Schemas.delayed_balance_deep_alive_d1 p c  hc (named c ((4 : Fin 8), (3 : Fin 8))) (named c ((2 : Fin 8), (1 : Fin 8))) 63 65 61 2 (by omega) (by omega) h0 h1 h2 h3 h4

end Ideas.Mined.puzzle_000Sa_deep
