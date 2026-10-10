import Chess

/-!
# Stalemate lemmas (Kaveh 2026-10-10)

"If the opponent has a piece that can move, they cannot be stalemated -- and as long as that piece exists I don't have
to recheck"; "make every move in the endgame a check ... and I will never stalemate". ChessLean defines
`stalemateB p = !inCheckB p p.turn && (legalMoves p).isEmpty`, so both are consequences of the definition -- the kind of
rule the lemma miner (catspace/incubator/analyst/lemma_miner.py) is meant to find from vocabulary, and the kind a
definitional proof closes.
-/

namespace Ideas.StalemateLemmas

open Chess

/-- A side in check is never stalemated. -/
theorem check_not_stalemate (p : Position) (h : inCheckB p p.turn = true) : ¬ stalemate p := by
  simp [stalemate, stalemateB, h]

/-- A side with a legal move is never stalemated. -/
theorem legal_move_not_stalemate (p : Position) (m : Move) (h : m ∈ legalMoves p) : ¬ stalemate p := by
  intro hs
  have hs' : (!inCheckB p p.turn && (legalMoves p).isEmpty) = true := hs
  simp only [Bool.and_eq_true, List.isEmpty_iff] at hs'
  rw [hs'.2] at h
  simp at h

/-- THE POLICY FORM: a move after which the opponent is in check never stalemates the opponent ("give check with every
move and you never stalemate"). -/
theorem checking_move_never_stalemates (p : Position) (m : Move)
    (h : inCheckB (applyMove p m) (applyMove p m).turn = true) : ¬ stalemate (applyMove p m) :=
  check_not_stalemate (applyMove p m) h

/-- A side not in check is never checkmated. -/
theorem not_check_not_checkmate (p : Position) (h : inCheckB p p.turn = false) : ¬ checkmate p := by
  simp [checkmate, checkmateB, h]

/-- A side with a legal move is never checkmated. -/
theorem legal_move_not_checkmate (p : Position) (m : Move) (h : m ∈ legalMoves p) : ¬ checkmate p := by
  intro hc
  have hc' : (inCheckB p p.turn && (legalMoves p).isEmpty) = true := hc
  simp only [Bool.and_eq_true, List.isEmpty_iff] at hc'
  rw [hc'.2] at h
  simp at h

end Ideas.StalemateLemmas
