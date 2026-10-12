import Chess.Game
import Ideas.Mate
import Ideas.MateDemand
import Ideas.FrameLemma
import Ideas.QueenMate
import Ideas.Fork

/-!
# A king and a queen mate only on the edge -- on the board, from the rules (Kaveh 2026-10-12)

The demand of a mate (`Ideas.MateDemand`: the mated king's square is attacked, and every square next to it is attacked
with the king lifted, or holds its own piece or the other king) applied to a king and a queen: their attack is a king's
step or a queen's line (`attack_kq`); the lone king has no other piece, and our king is not next to it, so every square
of its box is covered in the sense of `Ideas.QueenMate.covered`. A mated lone king off the edge would contradict
`no_queen_mate_in_open`: THE LONE KING IS ON THE EDGE (`kq_mate_on_edge`).
-/

namespace Ideas.QueenMate

open Chess

/-- File and rank as integers. -/
def fx (s : Square) : Int := s.1.val
def fy (s : Square) : Int := s.2.val

theorem kingGeometry_hits (s t : Square) (h : kingGeometryB s t = true) : kingHits (fx s) (fy s) (fx t) (fy t) := by
  unfold kingHits fx fy
  simp only [kingGeometryB, Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq, Prod.ext_iff, Fin.ext_iff,
    Ideas.FrameLemma.absDiff_def] at h
  split_ifs at h <;> omega

theorem queenGeometry_line (s t : Square) (h : queenGeometryB s t = true) : queenLine (fx s) (fy s) (fx t) (fy t) := by
  unfold queenLine fx fy
  simp only [queenGeometryB, rookGeometryB, bishopGeometryB, sameDiagonalB, sameFileB, sameRankB, Bool.or_eq_true,
    Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, Prod.ext_iff, Fin.ext_iff, Ideas.FrameLemma.absDiff_def] at h
  split_ifs at h <;> omega

/-- AN ATTACK BY A KING AND A QUEEN: on a board where colour `c` has only a king on `w` and a queen on `q`, an attacked
square is a king's step from `w` or on a line of `q`. -/
theorem attack_kq (p : Position) (c : Color) (w q t : Square)
    (hc : ∀ s pc, p.board s = some pc → pc.color = c → (s = w ∧ pc.kind = .king) ∨ (s = q ∧ pc.kind = .queen))
    (h : isAttackedB p c t = true) : kingHits (fx w) (fy w) (fx t) (fy t) ∨ queenLine (fx q) (fy q) (fx t) (fy t) := by
  simp only [isAttackedB, List.any_eq_true] at h
  obtain ⟨s, _, hs⟩ := h
  unfold pieceAttacksB at hs
  cases hb : p.board s with
  | none => rw [hb] at hs; simp at hs
  | some pc =>
    rw [hb] at hs
    simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at hs
    obtain ⟨hcol, hk⟩ := hs
    rcases hc s pc hb hcol with ⟨rfl, hkind⟩ | ⟨rfl, hkind⟩
    · left
      rw [hkind] at hk
      simp at hk
      exact kingGeometry_hits _ _ hk
    · right
      rw [hkind] at hk
      simp at hk
      exact queenGeometry_line _ _ hk.1

/-- Exactly three pieces: the side to move has its king on `k` and nothing else; the other side a king on `w` and a queen
on `q`. -/
def kqk (p : Position) (k w q : Square) : Prop :=
  p.board k = some ⟨p.turn, .king⟩ ∧ p.board w = some ⟨Color.opposite p.turn, .king⟩ ∧
    p.board q = some ⟨Color.opposite p.turn, .queen⟩ ∧ ∀ s, s ≠ k → s ≠ w → s ≠ q → p.board s = none

/-- A KING AND A QUEEN MATE ONLY ON THE EDGE: in a checkmate with the lone king on `k` against a king on `w` (not next to
it -- kings never touch) and a queen on `q`, the lone king stands on the edge of the board. -/
theorem kq_mate_on_edge (p : Position) (k w q : Square) (h : kqk p k w q) (hfar : kingGeometryB k w = false)
    (hmate : checkmate p) : k.1.val = 0 ∨ k.1.val = 7 ∨ k.2.val = 0 ∨ k.2.val = 7 := by
  obtain ⟨hk, hw, hq, hnone⟩ := h
  have hop := opposite_ne p.turn
  have hkw : k ≠ w := by rintro rfl; rw [hk] at hw; simp only [Option.some.injEq, Piece.mk.injEq] at hw; exact hop hw.1.symm
  have hkq : k ≠ q := by rintro rfl; rw [hk] at hq; simp at hq
  have hwq : w ≠ q := by rintro rfl; rw [hw] at hq; simp at hq
  -- the lone king alone of its kind
  have hK : ∀ s, (p.board s = some ⟨p.turn, .king⟩ ↔ s = k) := by
    intro s; constructor
    · intro hs
      by_contra hsk
      by_cases hsw : s = w
      · subst hsw; rw [hw] at hs; simp only [Option.some.injEq, Piece.mk.injEq] at hs; exact hop hs.1
      · by_cases hsq : s = q
        · subst hsq; rw [hq] at hs; simp at hs
        · rw [hnone s hsk hsw hsq] at hs; simp at hs
    · rintro rfl; exact hk
  -- the attacker's pieces: the king on w and the queen on q (with the lone king lifted too)
  have hc : ∀ s pc, (withEmpty p k).board s = some pc → pc.color = Color.opposite p.turn →
      (s = w ∧ pc.kind = .king) ∨ (s = q ∧ pc.kind = .queen) := by
    intro s pc hs hcol
    simp only [withEmpty, boardUpdate, Function.update_apply] at hs
    by_cases hsk : s = k
    · simp [hsk] at hs
    · rw [if_neg hsk] at hs
      by_cases hsw : s = w
      · subst hsw; rw [hw] at hs; simp only [Option.some.injEq] at hs; subst hs; exact Or.inl ⟨rfl, rfl⟩
      · by_cases hsq : s = q
        · subst hsq; rw [hq] at hs; simp only [Option.some.injEq] at hs; subst hs; exact Or.inr ⟨rfl, rfl⟩
        · rw [hnone s hsk hsw hsq] at hs; simp at hs
  have hc0 : ∀ s pc, p.board s = some pc → pc.color = Color.opposite p.turn →
      (s = w ∧ pc.kind = .king) ∨ (s = q ∧ pc.kind = .queen) := by
    intro s pc hs hcol
    have hsk : s ≠ k := by rintro rfl; rw [hk] at hs; simp only [Option.some.injEq] at hs; subst hs; exact hop hcol.symm
    exact hc s pc (by simp only [withEmpty, boardUpdate, Function.update_apply, hsk, if_false]; exact hs) hcol
  -- off the edge?
  by_contra hin
  simp only [not_or] at hin
  obtain ⟨h1, h2, h3, h4⟩ := hin
  have hk1 := k.1.isLt
  have hk2 := k.2.isLt
  have hfar' : 2 ≤ fx w - fx k ∨ 2 ≤ fx k - fx w ∨ 2 ≤ fy w - fy k ∨ 2 ≤ fy k - fy w := by
    have hne : ¬ (k.1.val = w.1.val ∧ k.2.val = w.2.val) := fun ⟨a, b⟩ => hkw (Prod.ext (Fin.ext a) (Fin.ext b))
    simp only [kingGeometryB, Bool.and_eq_false_iff, bne_eq_false_iff_eq, decide_eq_false_iff_not,
      Prod.ext_iff, Fin.ext_iff, Ideas.FrameLemma.absDiff_def] at hfar
    unfold fx fy
    split_ifs at hfar <;> omega
  apply no_queen_mate_in_open (fx k) (fy k) (fx w) (fy w) (fx q) (fy q) hfar'
  intro i j hi1 hi2 hj1 hj2
  by_cases h0 : i = 0 ∧ j = 0
  · obtain ⟨rfl, rfl⟩ := h0
    simp only [add_zero]
    exact attack_kq p (Color.opposite p.turn) w q k hc0 (checkmate_attacked p k hK hmate)
  · -- the square (k + (i, j)) of the box, on the board since the king is off the edge
    have hx0 : 0 ≤ fx k + i := by unfold fx; omega
    have hx8 : (fx k + i).toNat < 8 := by unfold fx; omega
    have hy0 : 0 ≤ fy k + j := by unfold fy; omega
    have hy8 : (fy k + j).toNat < 8 := by unfold fy; omega
    let d : Square := (⟨(fx k + i).toNat, hx8⟩, ⟨(fy k + j).toNat, hy8⟩)
    have hdx : fx d = fx k + i := by show ((fx k + i).toNat : Int) = fx k + i; omega
    have hdy : fy d = fy k + j := by show ((fy k + j).toNat : Int) = fy k + j; omega
    have hgeo : kingGeometryB k d = true := by
      have e1 : d.1.val = (fx k + i).toNat := rfl
      have e2 : d.2.val = (fy k + j).toNat := rfl
      simp only [kingGeometryB, Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq, Prod.ext_iff, Fin.ext_iff,
        Ideas.FrameLemma.absDiff_def]
      simp only [fx, fy] at e1 e2
      split_ifs <;> omega
    have hdw : d ≠ w := by rintro hdw; rw [hdw] at hgeo; rw [hgeo] at hfar; exact absurd hfar (by decide)
    have hdk : d ≠ k := by rintro hdk; rw [hdk] at hgeo; simp [kingGeometryB] at hgeo
    have hdest : destinationFriendlyB p p.turn d = true := by
      unfold destinationFriendlyB
      by_cases hdq : d = q
      · rw [hdq, hq]; simp [hop]
      · rw [hnone d hdk hdw hdq]
    have hfl := checkmate_flight p k hK hmate d
    simp only [flightOkB, hgeo, hdest, Bool.not_true, Bool.false_or] at hfl
    have := attack_kq (withEmpty p k) (Color.opposite p.turn) w q d hc hfl
    rw [hdx, hdy] at this
    exact this

end Ideas.QueenMate
