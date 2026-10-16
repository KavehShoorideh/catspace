import Ideas.Base.Vocabulary
import Ideas.TreeStep
import Ideas.Root
import Ideas.Mined.Schemas

/-! MINED by catspace/incubator/analyst/discovery/selfmine.py (g_46LzXjoz_1_drill_layers): each theorem is a conjectured PREMISE over the base vocabulary with the OUTCOME
`forcingOf p c 9 != 0` (Ideas.Vocabulary.forcingOf: side c, to move, forces a gain of 9 or more in 1 to 3 moves against every defence).
Squares are named from c's side. A proof that failed stays as `sorry` with the error in the comment above it: the statement is the finding. -/

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 100000

namespace Ideas.Mined.g_46LzXjoz_1_drill_layers

open Chess hiding between
open Ideas.Vocabulary

-- PROOF FAILED: unsolved goals p : Position fst snd : Fin 8 hW : kingSq p p.turn = some (fst, snd) left : 1 ≤ count fun u => oset (movesOf p) (some (named p.turn (2, 4))) u && oset atSq (some (named p.turn (5, 7))) u left_1 : (count fun u => kindSet p p.turn PieceKind.pawn u && (oset atSq (some (named p.turn (2, 4)
/-- root + deviation: how many of the squares c5 can legally move to that are also the square f8 >= 1 AND how many of the squares of our pawns that are also the square c5 that are also the rank of a7 <= 0 AND how many of the squares of our pieces that are also the square c5 >= 1 AND how many of the squares of their pieces that are also the square f8 >= 1 AND the declared value of the piece on f8 (pawn 1, knight 3, bishop 3, rook 5, queen 9; a king or an empty square 0), whoever owns it >= 9 AND how many of the squares of our pieces that are also the squares they attack after the piece on c5 moves to f8 <= 1 AND the least the most the least our material minus theirs, plus 64 (64: level) after any legal move of the side to move after any legal move of the side to move after any legal move of the side to move after the piece on c5 moves to f8 >= 76 (covers plies k=[]) -/
theorem conjecture_01 (p : Position) (c : Color) (W : Square) (hc : p.turn = c) (hW : kingSq p c = some W) :
    ((count (fun u => (oset (movesOf p) (some (named c (((2 : Fin 8)), ((4 : Fin 8)))))) u && (oset atSq (some (named c (((5 : Fin 8)), ((7 : Fin 8)))))) u)) ≥ 1) ∧ ((count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some (named c (((2 : Fin 8)), ((4 : Fin 8)))))) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0) ∧ ((count (fun u => (piecesOf p c) u && (oset atSq (some (named c (((2 : Fin 8)), ((4 : Fin 8)))))) u)) ≥ 1) ∧ ((count (fun u => (piecesOf p (Color.opposite c)) u && (oset atSq (some (named c (((5 : Fin 8)), ((7 : Fin 8)))))) u)) ≥ 1) ∧ ((onum (worthOf p) (some (named c (((5 : Fin 8)), ((7 : Fin 8)))))) ≥ 9) ∧ ((afterMove p (some (named c (((2 : Fin 8)), ((4 : Fin 8))))) (some (named c (((5 : Fin 8)), ((7 : Fin 8))))) (fun p => (count (fun u => (piecesOf p c) u && (attackedBy p (Color.opposite c) (kingSqD p c)) u)))) ≤ 1) ∧ ((afterMove p (some (named c (((2 : Fin 8)), ((4 : Fin 8))))) (some (named c (((5 : Fin 8)), ((7 : Fin 8))))) (fun p => (leastOver p (fun p => (mostOver p (fun p => (leastOver p (fun p => (balanceOf p c))))))))) ≥ 76) → forcingOf p c 9 ≠ 0 := by
  sorry

end Ideas.Mined.g_46LzXjoz_1_drill_layers
