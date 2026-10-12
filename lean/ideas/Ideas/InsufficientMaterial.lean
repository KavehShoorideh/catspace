import Chess.Game
import Ideas.Mate
import Ideas.MateDemand
import Ideas.GeoLemmas
import Ideas.FrameLemma
import Ideas.Fork

/-!
# A king and one minor piece never mate -- insufficient material, derived (Kaveh 2026-10-12)

"Derive an insufficient material rule from the laws already proven." The same process as `Ideas.QueenMate`:
1. THE DEMAND (`Ideas.MateDemand`): the mated king's square is attacked, and every square next to it is covered.
2. THE SUPPLY, by the colour laws of `Ideas.GeoLemmas`: a bishop attacks only its own colour (`bishop_keeps_colour`), a
   knight only the colour it does not stand on (`knight_flips_colour`). The minor piece gives the check (our king never
   touches theirs), so it attacks the mated king's colour -- and nothing else.
3. THE LEFTOVER: the orthogonal neighbours of the mated king are the other colour (`orthAdj_other_colour`), so they fall
   to our king alone; every square has two of them on the board, and a king guarding two orthogonal neighbours of a
   square stands next to it (`king_guards_two_orth`) -- the kings would touch.
So with a king and one knight or one bishop against a lone king there is no checkmate (`no_mate_king_minor`); the side
with the minor piece is never even in check (`never_in_check_by_lone_king`); and two kings alone never mate.
-/

namespace Ideas.InsufficientMaterial

open Chess
open Ideas.GeoLemmas

/-- A king's step, in file and rank numbers. -/
theorem kg_iff (s t : Square) : kingGeometryB s t = true ↔ (¬ (s.1.val = t.1.val ∧ s.2.val = t.2.val) ∧
    s.1.val ≤ t.1.val + 1 ∧ t.1.val ≤ s.1.val + 1 ∧ s.2.val ≤ t.2.val + 1 ∧ t.2.val ≤ s.2.val + 1) := by
  simp only [kingGeometryB, Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq, Prod.ext_iff, Fin.ext_iff,
    absDiff_def]
  split_ifs <;> omega

/-- Orthogonal neighbours, in file and rank numbers. -/
theorem orth_iff (s t : Square) : orthAdj s t ↔ ((s.1.val = t.1.val ∧ (s.2.val = t.2.val + 1 ∨ t.2.val = s.2.val + 1)) ∨
    (s.2.val = t.2.val ∧ (s.1.val = t.1.val + 1 ∨ t.1.val = s.1.val + 1))) := by
  simp only [orthAdj, absDiff_def]
  split_ifs <;> omega

/-- A king that guards two different orthogonal neighbours of `k` stands next to `k` (or on it). -/
theorem king_guards_two_orth (k w n1 n2 : Square) (h1 : orthAdj k n1) (h2 : orthAdj k n2) (hne : n1 ≠ n2)
    (g1 : kingGeometryB w n1 = true) (g2 : kingGeometryB w n2 = true) : w = k ∨ kingGeometryB k w = true := by
  have hne' : ¬ (n1.1.val = n2.1.val ∧ n1.2.val = n2.2.val) := fun ⟨a, b⟩ => hne (Prod.ext (Fin.ext a) (Fin.ext b))
  by_cases hwk : w = k
  · exact Or.inl hwk
  · right
    have hwk' : ¬ (k.1.val = w.1.val ∧ k.2.val = w.2.val) := fun ⟨a, b⟩ => hwk (Prod.ext (Fin.ext a.symm) (Fin.ext b.symm))
    rw [orth_iff] at h1 h2
    rw [kg_iff] at g1 g2 ⊢
    omega

/-- Every square has two different orthogonal neighbours on the board. -/
theorem two_orth (k : Square) : ∃ n1 n2 : Square, n1 ≠ n2 ∧ orthAdj k n1 ∧ orthAdj k n2 := by
  have hx := k.1.isLt
  have hy := k.2.isLt
  -- one along the file, one along the rank, each toward the board's inside
  obtain ⟨a, ha, ha'⟩ : ∃ a : Fin 8, (a.val = k.1.val + 1 ∨ a.val + 1 = k.1.val) ∧ True :=
    if h : k.1.val < 7 then ⟨⟨k.1.val + 1, by omega⟩, Or.inl rfl, trivial⟩ else ⟨⟨k.1.val - 1, by omega⟩, Or.inr (by simp; omega), trivial⟩
  obtain ⟨b, hb, hb'⟩ : ∃ b : Fin 8, (b.val = k.2.val + 1 ∨ b.val + 1 = k.2.val) ∧ True :=
    if h : k.2.val < 7 then ⟨⟨k.2.val + 1, by omega⟩, Or.inl rfl, trivial⟩ else ⟨⟨k.2.val - 1, by omega⟩, Or.inr (by simp; omega), trivial⟩
  refine ⟨(a, k.2), (k.1, b), ?_, ?_, ?_⟩
  · intro h; simp only [Prod.ext_iff, Fin.ext_iff] at h; omega
  · simp only [orthAdj, absDiff_def]; split_ifs <;> omega
  · simp only [orthAdj, absDiff_def]; split_ifs <;> omega

/-- THE MINOR PIECE'S COLOUR CLASS: a knight or a bishop on `m` that attacks `k` attacks only squares of `k`'s colour. -/
theorem minor_colour (r : PieceKind) (hr : r = .knight ∨ r = .bishop) (m k t : Square)
    (gk : (if r = .knight then knightGeometryB m k else bishopGeometryB m k) = true)
    (gt : (if r = .knight then knightGeometryB m t else bishopGeometryB m t) = true) : colour t = colour k := by
  rcases hr with rfl | rfl
  · simp only [if_true] at gk gt
    have h1 := knight_flips_colour m k gk
    have h2 := knight_flips_colour m t gt
    simp only [colour] at *
    omega
  · simp only [reduceCtorEq, if_false] at gk gt
    rw [← bishop_keeps_colour m t gt, bishop_keeps_colour m k gk]

/-- An attack by a minor piece of kind `r` standing on `m` is its geometry. -/
theorem minor_attack (p : Position) (c : Color) (r : PieceKind) (hr : r = .knight ∨ r = .bishop) (m t : Square)
    (hm : p.board m = some ⟨c, r⟩) (h : pieceAttacksB p c m t = true) :
    (if r = .knight then knightGeometryB m t else bishopGeometryB m t) = true := by
  unfold pieceAttacksB at h
  rw [hm] at h
  rcases hr with rfl | rfl <;> simp_all

/-- Exactly three pieces: the side to move has only its king, on `k`; the other side a king on `w` and one minor piece of
kind `r` (a knight or a bishop) on `m`. -/
def kmk (p : Position) (k w m : Square) (r : PieceKind) : Prop :=
  (r = .knight ∨ r = .bishop) ∧ p.board k = some ⟨p.turn, .king⟩ ∧ p.board w = some ⟨Color.opposite p.turn, .king⟩ ∧
    p.board m = some ⟨Color.opposite p.turn, r⟩ ∧ ∀ s, s ≠ k → s ≠ w → s ≠ m → p.board s = none

/-- A KING AND ONE MINOR PIECE NEVER MATE a lone king (the kings apart, as they always are). -/
theorem no_mate_king_minor (p : Position) (k w m : Square) (r : PieceKind) (h : kmk p k w m r)
    (hfar : kingGeometryB k w = false) : ¬ checkmate p := by
  intro hmate
  obtain ⟨hr, hk, hw, hm, hnone⟩ := h
  have hop := opposite_ne p.turn
  have hkw : k ≠ w := by rintro rfl; rw [hk] at hw; simp only [Option.some.injEq, Piece.mk.injEq] at hw; exact hop hw.1.symm
  have hkm : k ≠ m := by rintro rfl; rw [hk] at hm; simp only [Option.some.injEq, Piece.mk.injEq] at hm; exact hop hm.1.symm
  have hK : ∀ s, (p.board s = some ⟨p.turn, .king⟩ ↔ s = k) := by
    intro s; constructor
    · intro hs
      by_contra hsk
      by_cases hsw : s = w
      · subst hsw; rw [hw] at hs; simp only [Option.some.injEq, Piece.mk.injEq] at hs; exact hop hs.1
      · by_cases hsm : s = m
        · subst hsm; rw [hm] at hs; simp only [Option.some.injEq, Piece.mk.injEq] at hs; exact hop hs.1
        · rw [hnone s hsk hsw hsm] at hs; simp at hs
    · rintro rfl; exact hk
  -- an attacker's piece on a board equal to p off k: our king on w or the minor on m
  have hsrc : ∀ (b : Board), (∀ s, s ≠ k → b s = p.board s) → ∀ s pc, b s = some pc → pc.color = Color.opposite p.turn →
      s ≠ k → (s = w ∧ pc = ⟨Color.opposite p.turn, .king⟩) ∨ (s = m ∧ pc = ⟨Color.opposite p.turn, r⟩) := by
    intro b hb s pc hs _ hsk
    rw [hb s hsk] at hs
    by_cases hsw : s = w
    · subst hsw; rw [hw] at hs; exact Or.inl ⟨rfl, (Option.some.inj hs).symm⟩
    · by_cases hsm : s = m
      · subst hsm; rw [hm] at hs; exact Or.inr ⟨rfl, (Option.some.inj hs).symm⟩
      · rw [hnone s hsk hsw hsm] at hs; simp at hs
  -- 1 the check comes from the minor piece: our king is not next to theirs
  have hchk := checkmate_attacked p k hK hmate
  simp only [isAttackedB, List.any_eq_true] at hchk
  obtain ⟨s, _, hs⟩ := hchk
  obtain ⟨pc, hpc, hcol⟩ := pieceAttacksB_source hs
  have hsk : s ≠ k := pieceAttacksB_ne hs
  have gk : (if r = .knight then knightGeometryB m k else bishopGeometryB m k) = true := by
    rcases hsrc p.board (fun _ _ => rfl) s pc hpc hcol hsk with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · exfalso
      unfold pieceAttacksB at hs; rw [hw] at hs
      simp at hs
      rw [Passer.kingGeometryB_comm, hs] at hfar
      exact absurd hfar (by decide)
    · exact minor_attack p _ r hr s k hpc hs
  -- 2 two orthogonal neighbours of the mated king, both covered by our king alone
  obtain ⟨n1, n2, hne, ho1, ho2⟩ := two_orth k
  have guard : ∀ n, orthAdj k n → kingGeometryB w n = true := by
    intro n hn
    have hgeo : kingGeometryB k n = true := by
      rw [orth_iff] at hn
      rw [kg_iff]
      omega
    have hnw : n ≠ w := by rintro rfl; rw [hgeo] at hfar; exact absurd hfar (by decide)
    have hnk : n ≠ k := by rintro rfl; simp [kingGeometryB] at hgeo
    have hdest : destinationFriendlyB p p.turn n = true := by
      unfold destinationFriendlyB
      by_cases hnm : n = m
      · rw [hnm, hm]; rcases hr with rfl | rfl <;> simp [hop]
      · rw [hnone n hnk hnw hnm]
    have hfl := checkmate_flight p k hK hmate n
    simp only [flightOkB, hgeo, hdest, Bool.not_true, Bool.false_or, isAttackedB, List.any_eq_true] at hfl
    obtain ⟨s', _, hs'⟩ := hfl
    obtain ⟨pc', hpc', hcol'⟩ := pieceAttacksB_source hs'
    have hs'k : s' ≠ k := by
      rintro rfl; simp [withEmpty, boardUpdate] at hpc'
    have hb : ∀ x, x ≠ k → (withEmpty p k).board x = p.board x := by
      intro x hx; simp [withEmpty, boardUpdate, hx]
    rcases hsrc (withEmpty p k).board hb s' pc' hpc' hcol' hs'k with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · unfold pieceAttacksB at hs'; rw [hpc'] at hs'; simpa using hs'
    · -- the minor attacks only the mated king's colour; an orthogonal neighbour is the other colour
      exfalso
      have gt := minor_attack (withEmpty p k) _ r hr s' n hpc' hs'
      exact orthAdj_other_colour k n hn (minor_colour r hr s' k n gk gt).symm
  rcases king_guards_two_orth k w n1 n2 ho1 ho2 hne (guard n1 ho1) (guard n2 ho2) with h | h
  · exact hkw h.symm
  · rw [h] at hfar; exact absurd hfar (by decide)

/-- A LONE KING NEVER CHECKS: when the other side has nothing but a king on `w`, not next to ours on `k`, the side to move
is not mated. (Two kings alone; or a king and a minor piece with the minor piece's side to move.) -/
theorem no_mate_by_lone_king (p : Position) (k w : Square) (hK : ∀ s, (p.board s = some ⟨p.turn, .king⟩ ↔ s = k))
    (hopp : ∀ s pc, p.board s = some pc → pc.color = Color.opposite p.turn → s = w ∧ pc.kind = .king)
    (hfar : kingGeometryB k w = false) : ¬ checkmate p := by
  intro hmate
  have hchk := checkmate_attacked p k hK hmate
  simp only [isAttackedB, List.any_eq_true] at hchk
  obtain ⟨s, _, hs⟩ := hchk
  obtain ⟨pc, hpc, hcol⟩ := pieceAttacksB_source hs
  obtain ⟨rfl, hkind⟩ := hopp s pc hpc hcol
  unfold pieceAttacksB at hs
  rw [hpc] at hs
  simp [hkind] at hs
  rw [Passer.kingGeometryB_comm, hs.2] at hfar
  exact absurd hfar (by decide)

/-- INSUFFICIENT MATERIAL, the static rule: two kings apart and at most one more piece, a knight or a bishop of either
side -- no checkmate. -/
theorem insufficient_material (p : Position) (k w : Square) (hk : p.board k = some ⟨p.turn, .king⟩)
    (hw : p.board w = some ⟨Color.opposite p.turn, .king⟩) (hfar : kingGeometryB k w = false)
    (hrest : (∀ s, s ≠ k → s ≠ w → p.board s = none) ∨
      ∃ m c r, (r = .knight ∨ r = .bishop) ∧ p.board m = some ⟨c, r⟩ ∧ ∀ s, s ≠ k → s ≠ w → s ≠ m → p.board s = none) :
    ¬ checkmate p := by
  have hop := opposite_ne p.turn
  -- the side to move: its king on k, plus the minor piece if it is its own
  have hK : (∀ s, s ≠ k → s ≠ w → ∀ pc, p.board s = some pc → pc.kind ≠ .king) →
      ∀ s, (p.board s = some ⟨p.turn, .king⟩ ↔ s = k) := by
    intro hnk s; constructor
    · intro hs; by_contra hsk
      by_cases hsw : s = w
      · subst hsw; rw [hw] at hs; simp only [Option.some.injEq, Piece.mk.injEq] at hs; exact hop hs.1
      · exact hnk s hsk hsw _ hs rfl
    · rintro rfl; exact hk
  rcases hrest with hnone | ⟨m, c, r, hr, hm, hnone⟩
  · refine no_mate_by_lone_king p k w (hK fun s a b pc h _ => by rw [hnone s a b] at h; simp at h) ?_ hfar
    intro s pc hs hcol
    by_cases hsw : s = w
    · subst hsw; rw [hw] at hs; exact ⟨rfl, by rw [← Option.some.inj hs]⟩
    · by_cases hsk : s = k
      · subst hsk; rw [hk] at hs; rw [← Option.some.inj hs] at hcol; exact absurd hcol.symm hop
      · rw [hnone s hsk hsw] at hs; simp at hs
  · have hmk : m ≠ k := by rintro rfl; rw [hk] at hm; simp only [Option.some.injEq, Piece.mk.injEq] at hm; rcases hr with rfl | rfl <;> simp at hm
    have hmw : m ≠ w := by rintro rfl; rw [hw] at hm; simp only [Option.some.injEq, Piece.mk.injEq] at hm; rcases hr with rfl | rfl <;> simp at hm
    have hnk : ∀ s, s ≠ k → s ≠ w → ∀ pc, p.board s = some pc → pc.kind ≠ .king := by
      intro s a b pc h
      by_cases hsm : s = m
      · subst hsm; rw [hm] at h; rw [← Option.some.inj h]; rcases hr with rfl | rfl <;> simp
      · rw [hnone s a b hsm] at h; simp at h
    by_cases hc : c = p.turn
    · -- the minor piece is the mover's: the other side has a lone king
      refine no_mate_by_lone_king p k w (hK hnk) ?_ hfar
      intro s pc hs hcol
      by_cases hsw : s = w
      · subst hsw; rw [hw] at hs; exact ⟨rfl, by rw [← Option.some.inj hs]⟩
      · by_cases hsk : s = k
        · subst hsk; rw [hk] at hs; rw [← Option.some.inj hs] at hcol; exact absurd hcol.symm hop
        · by_cases hsm : s = m
          · subst hsm; rw [hm] at hs; rw [← Option.some.inj hs] at hcol; rw [hc] at hcol; exact absurd hcol.symm hop
          · rw [hnone s hsk hsw hsm] at hs; simp at hs
    · -- the minor piece is the other side's: a king and one minor piece never mate
      have hc' : c = Color.opposite p.turn := by cases c <;> cases h : p.turn <;> simp_all
      subst hc'
      exact no_mate_king_minor p k w m r ⟨hr, hk, hw, hm, hnone⟩ hfar

end Ideas.InsufficientMaterial
