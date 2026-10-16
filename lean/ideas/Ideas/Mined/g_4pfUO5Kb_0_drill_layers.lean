import Ideas.Base.Vocabulary
import Ideas.TreeStep
import Ideas.Root

/-! MINED by catspace/incubator/analyst/discovery/selfmine.py (g_4pfUO5Kb_0_drill_layers): each theorem is a conjectured PREMISE over the base vocabulary with the OUTCOME
`forcingOf p c 3 != 0` (Ideas.Vocabulary.forcingOf: side c, to move, forces a gain of 3 or more in 1 to 3 moves against every defence).
Squares are named from c's side. A proof that failed stays as `sorry` with the error in the comment above it: the statement is the finding. -/

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 100000

namespace Ideas.Mined.g_4pfUO5Kb_0_drill_layers

open Chess hiding between
open Ideas.Vocabulary

/-- THE ROOT on Qxb4: how many of the squares b5 can legally move to that are also the square b4 >= 1 AND how many of the squares of our pawns that are also the square b5 <= 0 AND how many of the squares of our pieces that are also the square b5 >= 1 AND how many of the squares of their pieces that are also the square b4 >= 1 AND the declared value of the piece on b4 (pawn 1, knight 3, bishop 3, rook 5, queen 9; a king or an empty square 0), whoever owns it >= 3 AND how many of the squares of our pieces that are also the squares they attack after the piece on b5 moves to b4 <= 0 AND how many of the squares of their pawns that are also the rank of a2 after the piece on b5 moves to b4 <= 0 AND our material >= 4 AND the least 1 after any legal move of the side to move after the piece on b5 moves to b4 >= 1 -/
theorem hanging_instance (p : Position) (c : Color) (W : Square) (hc : p.turn = c) (hW : kingSq p c = some W) :
    ((count (fun u => (oset (movesOf p) (some (named c (((1 : Fin 8)), ((4 : Fin 8)))))) u && (oset atSq (some (named c (((1 : Fin 8)), ((3 : Fin 8)))))) u)) ≥ 1) ∧ ((count (fun u => (kindSet p c .pawn) u && (oset atSq (some (named c (((1 : Fin 8)), ((4 : Fin 8)))))) u)) ≤ 0) ∧ ((count (fun u => (piecesOf p c) u && (oset atSq (some (named c (((1 : Fin 8)), ((4 : Fin 8)))))) u)) ≥ 1) ∧ ((count (fun u => (piecesOf p (Color.opposite c)) u && (oset atSq (some (named c (((1 : Fin 8)), ((3 : Fin 8)))))) u)) ≥ 1) ∧ ((onum (worthOf p) (some (named c (((1 : Fin 8)), ((3 : Fin 8)))))) ≥ 3) ∧ ((afterMove p (some (named c (((1 : Fin 8)), ((4 : Fin 8))))) (some (named c (((1 : Fin 8)), ((3 : Fin 8))))) (fun p => (count (fun u => (piecesOf p c) u && (attackedBy p (Color.opposite c) W) u)))) ≤ 0) ∧ ((afterMove p (some (named c (((1 : Fin 8)), ((4 : Fin 8))))) (some (named c (((1 : Fin 8)), ((3 : Fin 8))))) (fun p => (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((1 : Fin 8)))))) u)))) ≤ 0) ∧ ((materialOf p c) ≥ 4) ∧ ((afterMove p (some (named c (((1 : Fin 8)), ((4 : Fin 8))))) (some (named c (((1 : Fin 8)), ((3 : Fin 8))))) (fun p => (leastOver p (fun p => (1))))) ≥ 1) → forcingOf p c 3 ≠ 0 := by
  intro hall
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8⟩ := hall
  exact Ideas.Root.hanging p c _ W hc hW (named c ((1 : Fin 8), (4 : Fin 8))) (named c ((1 : Fin 8), (3 : Fin 8))) h2 h1 h0 h3 (by simp only [onum] at h4; omega) h5 h6 (by omega) h8

end Ideas.Mined.g_4pfUO5Kb_0_drill_layers
