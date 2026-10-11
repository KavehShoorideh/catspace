import Chess
import Ideas.PathLemma

/-!
# The frame-independence lemma (Kaveh 2026-10-11)

A FRAME is a set of squares the planner reasons about: a target, its escapes, the pieces bearing on them, the lines
between. A board-level proof about a frame reads two inputs and nothing else: the pieces ON the frame's squares, and
the attacks ONTO the frame's squares from anywhere (an attack on the king and an attack on the queen are the same kind
of input; they differ in value, not in kind). This file proves that both inputs are unchanged by any move that touches
no square that can BEAR on the frame -- a square of the frame, a square on a line through one (file, rank or diagonal,
any distance) or a knight's move from one:

* `attack_reach`: whoever attacks `t` stands on a square that bears on `t` (king and pawn attacks lie on the
  queen's lines; rook and bishop are the queen's; the knight is the knight).
* `between_reach`: every square strictly between an attacker and `t` lies on a line through `t`.
* `frame_unchanged`: so if no touched square bears on any frame square, the board on the frame and every attack onto
  the frame are the same before and after the move; `attacked_unchanged` is the per-square corollary.

What this licenses in the planner (rust/analyst_core): a proof whose zone no touched square bears on is REUSED without
search, provided the proof read only its zone's pieces and the attacks onto its zone (the obligation this theorem puts
on the code: whatever else a proof consults -- the opponent's other frames -- is part of its zone or is a separate
index entry whose change is tracked). `rules.rs exchange_affected` (queen rays on an empty board + knight + king + the
square) is this `reachB`; the earlier ply-level search used a frame-independence step without proof ("certificates
modulo frame independence", lib.rs) -- this is that proof, in the form that is actually true.
-/

namespace Ideas.FrameLemma

open Chess
open Ideas.PathLemma

/-- `x` can bear on `t`: it is `t`, lies on a line through `t`, or is a knight's move from `t`. -/
def reachB (t x : Square) : Bool := (t == x) || queenGeometryB x t || knightGeometryB x t

theorem absDiff_def (a b : Fin 8) : absDiff a b = if b.val ≤ a.val then a.val - b.val else b.val - a.val := rfl

/-- A king step lies on the queen's lines. -/
theorem king_queen (s t : Square) (h : kingGeometryB s t = true) : queenGeometryB s t = true := by
  simp only [kingGeometryB, queenGeometryB, rookGeometryB, bishopGeometryB, sameDiagonalB, sameFileB, sameRankB,
    Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, decide_eq_true_eq, Prod.ext_iff, Fin.ext_iff,
    absDiff_def] at *
  split_ifs at * <;> omega

/-- A pawn's attack lies on a diagonal. -/
theorem pawn_queen (c : Color) (s t : Square) (h : pawnAttackGeometryB c s t = true) : queenGeometryB s t = true := by
  by_cases hc : c = .white
  · simp only [pawnAttackGeometryB, hc, if_true, queenGeometryB, rookGeometryB, bishopGeometryB, sameDiagonalB,
      sameFileB, sameRankB, Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq,
      Prod.ext_iff, Fin.ext_iff, absDiff_def] at *
    split_ifs at * <;> omega
  · simp only [pawnAttackGeometryB, hc, if_false, queenGeometryB, rookGeometryB, bishopGeometryB, sameDiagonalB,
      sameFileB, sameRankB, Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq,
      Prod.ext_iff, Fin.ext_iff, absDiff_def] at *
    split_ifs at * <;> omega

/-- Whoever attacks `t` stands on a square that bears on `t`. -/
theorem attack_reach (p : Position) (c : Color) (s t : Square) (h : pieceAttacksB p c s t = true) :
    reachB t s = true := by
  cases hb : p.board s with
  | none => simp [pieceAttacksB, hb] at h
  | some q =>
      simp only [pieceAttacksB, hb, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, and_assoc, or_assoc] at h
      obtain ⟨hcol, h⟩ := h
      simp only [reachB, Bool.or_eq_true]
      rcases h with ⟨hk, hg, hs⟩ | ⟨hk, hg, hs⟩ | ⟨hk, hg, hs⟩ | ⟨hk, hg⟩ | ⟨hk, hg⟩ | ⟨hk, hg⟩
      · left; right; simp only [queenGeometryB, Bool.or_eq_true]; left; exact hg
      · left; right; simp only [queenGeometryB, Bool.or_eq_true]; right; exact hg
      · left; right; exact hg
      · right; exact hg
      · left; right; exact king_queen s t hg
      · left; right; exact pawn_queen c s t hg

/-- Every square strictly between `s` and `t` lies on a line through `t`. -/
theorem between_reach (s u t : Square) (h : between s u t) : reachB t u = true := by
  simp only [between, betweenB, reachB, queenGeometryB, rookGeometryB, bishopGeometryB, sameDiagonalB, sameFileB,
    sameRankB, betweenCoordB, ltB, Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq,
    decide_eq_true_eq, Prod.ext_iff, Fin.ext_iff, absDiff_def] at *
  split_ifs at * <;> omega

/-- THE FRAME-INDEPENDENCE LEMMA: if no touched square bears on any square of the frame `F`, the board on `F` and every
attack onto `F` are unchanged by the move. -/
theorem frame_unchanged (p : Position) (m : Move) (F : Square → Prop)
    (hF : ∀ t, F t → ∀ x, x ∈ touched p m → reachB t x = false) :
    (∀ t, F t → (applyMove p m).board t = p.board t) ∧
    (∀ t, F t → ∀ c s, pieceAttacksB (applyMove p m) c s t = pieceAttacksB p c s t) := by
  have off : ∀ t, F t → t ∉ touched p m := by
    intro t ht hin
    have := hF t ht t hin
    simp [reachB] at this
  refine ⟨?_, ?_⟩
  · intro t ht
    rw [applyMove_board]
    exact applyBoard_off p m t (off t ht)
  · intro t ht c s
    cases hr : reachB t s with
    | false =>
        -- `s` cannot bear on `t`: it attacks `t` on neither board
        have h1 : pieceAttacksB p c s t = false := by
          cases h : pieceAttacksB p c s t with
          | false => rfl
          | true => rw [attack_reach p c s t h] at hr; exact absurd hr (by decide)
        have h2 : pieceAttacksB (applyMove p m) c s t = false := by
          cases h : pieceAttacksB (applyMove p m) c s t with
          | false => rfl
          | true => rw [attack_reach (applyMove p m) c s t h] at hr; exact absurd hr (by decide)
        rw [h1, h2]
    | true =>
        -- `s` bears on `t`, so it is untouched; so is every square between them (each lies on a line through `t`)
        apply attack_unchanged p m c s t
        · intro hin; have := hF t ht s hin; rw [hr] at this; exact absurd this (by decide)
        · intro u hb hin
          have := hF t ht u hin
          rw [between_reach s u t hb] at this
          exact absurd this (by decide)

/-- The corollary the index uses: whether a frame square is attacked by either side is unchanged. -/
theorem attacked_unchanged (p : Position) (m : Move) (F : Square → Prop)
    (hF : ∀ t, F t → ∀ x, x ∈ touched p m → reachB t x = false) (t : Square) (ht : F t) (c : Color) :
    isAttackedB (applyMove p m) c t = isAttackedB p c t := by
  have h := (frame_unchanged p m F hF).2 t ht c
  simp only [isAttackedB]
  congr 1
  funext s
  exact h s

end Ideas.FrameLemma
