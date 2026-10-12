import Chess.Game
import Ideas.Mate

/-!
# The demand of a mate, from the rules (Kaveh 2026-10-12)

`Ideas.Mate` proves the SUFFICIENT direction -- a check that cannot be met is mate (`flightsCoveredB`, `noCaptureB`,
`noBlockB`). Here the NECESSARY direction, for any material: in a checkmate the defender's king stands attacked
(`checkmate_attacked`), and every square it could step to is no flight (`checkmate_flight`, in `Ideas.Mate`'s own
terms: `flightOkB` -- not a king step, or held by its own piece or the other king, or attacked with the king lifted off
the board). Why: the step to a square that is none of these is pseudo-legal, and after it the king is attacked exactly
when the square was attacked with the king lifted (the board differs only on the king's two squares, and no line to the
destination crosses it) -- so it would be a legal move, and a mated side has none.

This is THE DEMAND every mate lemma starts from: `Ideas.QueenMate` (a king and a queen mate only on the edge) and
`Ideas.InsufficientMaterial` (a king and one minor piece never mate).
-/

namespace Ideas

open Chess

/-- In a checkmate the defender's king (alone of its kind on `k`) is attacked. -/
theorem checkmate_attacked (q : Position) (k : Square) (hK : ∀ s, (q.board s = some ⟨q.turn, .king⟩ ↔ s = k))
    (hmate : checkmate q) : isAttackedB q (Color.opposite q.turn) k = true := by
  unfold checkmate checkmateB at hmate
  simp only [Bool.and_eq_true] at hmate
  have h := hmate.1
  unfold inCheckB at h
  rwa [kingSquare_of_only hK] at h

/-- THE FLIGHTS OF A MATED KING: no square next to it is a flight. -/
theorem checkmate_flight (q : Position) (k : Square) (hK : ∀ s, (q.board s = some ⟨q.turn, .king⟩ ↔ s = k))
    (hmate : checkmate q) (d : Square) : flightOkB q k d = true := by
  have hqk : q.board k = some ⟨q.turn, .king⟩ := (hK k).2 rfl
  cases hf : flightOkB q k d with
  | true => rfl
  | false =>
    exfalso
    simp only [flightOkB, Bool.or_eq_false_iff, Bool.not_eq_false'] at hf
    obtain ⟨⟨hgeo, hdest⟩, hatt⟩ := hf
    have hkd : k ≠ d := by rintro rfl; simp [kingGeometryB] at hgeo
    -- the step is pseudo-legal
    have hpl : pseudoLegal q (.normal k d) := by
      show normalPseudoLegalB q k d = true
      unfold normalPseudoLegalB
      rw [hqk]
      simp [normalPieceGeometryB, hgeo, hdest]
    -- after it the king stands on `d`, alone of its kind
    have hboard : (applyMove q (.normal k d)).board = boardMove q.board k d := rfl
    have hK' : ∀ x, ((applyMove q (.normal k d)).board x = some ⟨q.turn, .king⟩ ↔ x = d) := by
      intro x
      rw [hboard]
      simp only [boardMove, boardUpdate, Function.update_apply]
      by_cases hxd : x = d
      · subst hxd; simp [hqk]
      · by_cases hxk : x = k
        · subst hxk; simp [hxd]
        · simp [hxd, hxk, hK x]
    -- and is not attacked there: an attack after the step is an attack with the king lifted
    have hsafe : ¬ inCheck (applyMove q (.normal k d)) q.turn := by
      intro hc
      unfold inCheck inCheckB at hc
      rw [kingSquare_of_only hK'] at hc
      simp only [isAttackedB, List.any_eq_true] at hc
      obtain ⟨s, _, hs⟩ := hc
      have hsd : s ≠ d := pieceAttacksB_ne hs
      have hw : pieceAttacksB (withEmpty q k) (Color.opposite q.turn) s d = true := by
        apply pieceAttacksB_mono hs
        · rw [hboard]; simp [withEmpty, boardMove, boardUpdate, Function.update_apply, hsd]
        · intro u hu hn
          have hud : u ≠ d := between_not_destination hu
          rw [hboard] at hn
          simp only [withEmpty, boardMove, boardUpdate, Function.update_apply, hud, if_false] at hn ⊢
          exact hn
      have : isAttackedB (withEmpty q k) (Color.opposite q.turn) d = true := by
        simp only [isAttackedB, List.any_eq_true]
        exact ⟨s, mem_allSquares s, hw⟩
      rw [hatt] at this
      exact absurd this (by decide)
    -- so the step is legal: a mated side has none
    have hlegal : legalMove q (.normal k d) := ⟨hpl, hsafe, Or.inl (by simp [Move.isCastle])⟩
    unfold checkmate checkmateB at hmate
    simp only [Bool.and_eq_true, List.isEmpty_iff] at hmate
    have hmem : Move.normal k d ∈ legalMoves q := by
      unfold legalMoves
      exact List.mem_filter.2 ⟨mem_candidateMoves _, by simpa using hlegal⟩
    rw [hmate.2] at hmem
    simp at hmem

end Ideas
