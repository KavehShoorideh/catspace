import Chess.Legality
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Ring
import Ideas.Spec.Rook

/-!
# The generated rook is the kernel's rook (ANALYST_PLAN 76.0 (c), 2026-10-28)

`Ideas.Spec.Rook` is GENERATED from rules/chess.gdl by scripts/gdl_to_lean.py: the rook's attack (`attackByRook`), its slide (`slide`), the board's
geometry as the GDL states it (`step` from the successor tables `fileSucc` / `rankSucc`). This file proves, by reasoning, that the generated attack IS the
rule kernel's (chess-lean `Chess.pieceAttacks` of a rook: `rookGeometryB` and an empty segment `sliderAttackB` / `clearBetweenB`), on every position:
`attackByRook_iff`. And it re-proves one existing lemma, `Ideas.Region.rook_attacks_empty_file` ("a rook alone on its file attacks every other square of
the file"), from the GENERATED definition: `rook_attacks_empty_file_spec`.

How: the successor tables are the kernel's arithmetic (`fileSucc_iff`, `rankSucc_iff`; `file_all`, `rank_all`: each by cases on the table's rows -- the
GDL's successor IS an eight-row table, so the one place a case split on a coordinate's eight values is unavoidable); a step is a unit vector
(`step_iff`); a slide is a ray -- `n` steps along `d`, every square strictly before the last empty (`slide_iff`, by induction on the derivation one way and
on `n` the other); a rook's line on the kernel's side is the same ray (`kernel_rook_iff`, the kernel's `betweenB` read as arithmetic). No position and no
square is enumerated; no `decide`, no `native_decide`, no `sorry`.
-/

namespace Ideas.Spec.Rook

open Chess

/-! ## The tables: the GDL's coordinates are the kernel's -/

/-- Every file of the kernel is a GDL `file`. -/
theorem file_all (f : Fin 8) : file f := by
  obtain ⟨f, hf⟩ := f
  interval_cases f <;> constructor

/-- Every rank of the kernel is a GDL `rank`. -/
theorem rank_all (r : Fin 8) : rank r := by
  obtain ⟨r, hr⟩ := r
  interval_cases r <;> constructor

/-- The GDL's file successor is the kernel's `+ 1`. -/
theorem fileSucc_iff (a b : Fin 8) : fileSucc a b ↔ b.val = a.val + 1 := by
  constructor
  · intro h
    cases h <;> rfl
  · intro h
    obtain ⟨a, ha⟩ := a
    obtain ⟨b, hb⟩ := b
    simp only at h
    subst h
    interval_cases a <;> first | omega | constructor

/-- The GDL's rank successor is the kernel's `+ 1`. -/
theorem rankSucc_iff (a b : Fin 8) : rankSucc a b ↔ b.val = a.val + 1 := by
  constructor
  · intro h
    cases h <;> rfl
  · intro h
    obtain ⟨a, ha⟩ := a
    obtain ⟨b, hb⟩ := b
    simp only at h
    subst h
    interval_cases a <;> first | omega | constructor

/-! ## A step is a unit vector; a slide is a ray -/

/-- The file component of a direction. -/
def dx : Dir → Int
  | .east => 1
  | .west => -1
  | .north => 0
  | .south => 0
  | .northeast => 1
  | .northwest => -1
  | .southeast => 1
  | .southwest => -1

/-- The rank component of a direction. -/
def dy : Dir → Int
  | .east => 0
  | .west => 0
  | .north => 1
  | .south => -1
  | .northeast => 1
  | .northwest => 1
  | .southeast => -1
  | .southwest => -1

/-- One GDL step along `d` adds the direction's vector. -/
theorem step_iff (d : Dir) (f1 r1 f2 r2 : Fin 8) :
    step d f1 r1 f2 r2 ↔ ((f2.val : Int) = f1.val + dx d ∧ (r2.val : Int) = r1.val + dy d) := by
  constructor
  · intro h
    cases h <;> simp only [fileSucc_iff, rankSucc_iff, dx, dy] at * <;> constructor <;> omega
  · rintro ⟨hf, hr⟩
    cases d <;> simp only [dx, dy] at hf hr
    · have e : r2 = r1 := Fin.ext (by omega)
      subst e
      exact step.c3 (rank_all _) ((fileSucc_iff _ _).2 (by omega))
    · have e : f2 = f1 := Fin.ext (by omega)
      subst e
      exact step.c1 (file_all _) ((rankSucc_iff _ _).2 (by omega))
    · exact step.c5 ((fileSucc_iff _ _).2 (by omega)) ((rankSucc_iff _ _).2 (by omega))
    · exact step.c6 ((fileSucc_iff _ _).2 (by omega)) ((rankSucc_iff _ _).2 (by omega))
    · have e : f2 = f1 := Fin.ext (by omega)
      subst e
      exact step.c2 (file_all _) ((rankSucc_iff _ _).2 (by omega))
    · exact step.c7 ((fileSucc_iff _ _).2 (by omega)) ((rankSucc_iff _ _).2 (by omega))
    · exact step.c8 ((fileSucc_iff _ _).2 (by omega)) ((rankSucc_iff _ _).2 (by omega))
    · have e : r2 = r1 := Fin.ext (by omega)
      subst e
      exact step.c4 (rank_all _) ((fileSucc_iff _ _).2 (by omega))

/-- The square `k` steps from (f, r) along `d` is (f', r'). -/
def At (d : Dir) (f r : Fin 8) (k : Nat) (f' r' : Fin 8) : Prop :=
  (f'.val : Int) = f.val + k * dx d ∧ (r'.val : Int) = r.val + k * dy d

/-- THE RAY of length `n` from (f, r) along `d` is open: every square strictly before its last is empty. -/
def Ray (p : Position) (d : Dir) (f r : Fin 8) (n : Nat) : Prop :=
  ∀ k f' r', 1 ≤ k → k < n → At d f r k f' r' → p.board (f', r') = none

/-- `empty` is the kernel's empty square. -/
theorem empty_iff (p : Position) (f r : Fin 8) : empty p f r ↔ p.board (f, r) = none := by
  constructor
  · rintro ⟨_, hn⟩
    cases hb : p.board (f, r) with
    | none => rfl
    | some q => exact absurd (occupied.c1 (c := q.color) (k := q.kind) (by rw [hb])) hn
  · intro h
    refine empty.c1 (square.c1 (file_all f) (rank_all r)) ?_
    rintro ⟨hb⟩
    rw [h] at hb
    simp at hb

/-- Two squares the same number of steps along the same direction are one square. -/
theorem at_unique {d : Dir} {f r : Fin 8} {k : Nat} {f' r' f'' r'' : Fin 8} (h1 : At d f r k f' r') (h2 : At d f r k f'' r'') :
    f' = f'' ∧ r' = r'' := by
  obtain ⟨a1, b1⟩ := h1
  obtain ⟨a2, b2⟩ := h2
  exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩

/-- Between the start and a square `m + 1` steps away, the square `m` steps away is on the board. -/
theorem exists_at (d : Dir) (f1 r1 f2 r2 : Fin 8) (m : Nat) (h : At d f1 r1 (m + 1) f2 r2) : ∃ fm rm : Fin 8, At d f1 r1 m fm rm := by
  obtain ⟨h1, h2⟩ := h
  have hf1 := f1.isLt
  have hf2 := f2.isLt
  have hr1 := r1.isLt
  have hr2 := r2.isLt
  have hx : 0 ≤ (f1.val : Int) + m * dx d ∧ (f1.val : Int) + m * dx d < 8 := by
    cases d <;> simp only [dx] at h1 ⊢ <;> push_cast at h1 ⊢ <;> omega
  have hy : 0 ≤ (r1.val : Int) + m * dy d ∧ (r1.val : Int) + m * dy d < 8 := by
    cases d <;> simp only [dy] at h2 ⊢ <;> push_cast at h2 ⊢ <;> omega
  exact ⟨⟨((f1.val : Int) + m * dx d).toNat, by omega⟩, ⟨((r1.val : Int) + m * dy d).toNat, by omega⟩, by simp only; omega, by simp only; omega⟩

/-- THE SLIDE IS THE RAY: the generated `slide` reaches (f2, r2) from (f1, r1) along `d` exactly when (f2, r2) is `n ≥ 1` steps along `d` and the ray is open. -/
theorem slide_iff (p : Position) (d : Dir) (f1 r1 f2 r2 : Fin 8) :
    slide p d f1 r1 f2 r2 ↔ ∃ n, 1 ≤ n ∧ At d f1 r1 n f2 r2 ∧ Ray p d f1 r1 n := by
  constructor
  · intro h
    induction h with
    | c1 hs =>
      rw [step_iff] at hs
      refine ⟨1, le_refl _, ?_, ?_⟩
      · simpa [At] using hs
      · intro k _ _ h1 h2
        omega
    | @c2 f3 r3 f2 r2 _ he hs ih =>
      obtain ⟨n, hn, ⟨a1, a2⟩, hray⟩ := ih
      rw [step_iff] at hs
      refine ⟨n + 1, by omega, ⟨?_, ?_⟩, ?_⟩
      · rw [hs.1, a1]; push_cast; ring
      · rw [hs.2, a2]; push_cast; ring
      · intro k f r hk1 hk2 hk
        by_cases hkn : k < n
        · exact hray k f r hk1 hkn hk
        · have hk' : k = n := by omega
          subst hk'
          obtain ⟨e1, e2⟩ := at_unique hk ⟨a1, a2⟩
          subst e1
          subst e2
          exact (empty_iff p _ _).1 he
  · rintro ⟨n, hn, hat, hray⟩
    induction n generalizing f2 r2 with
    | zero => omega
    | succ m ih =>
      rcases Nat.eq_zero_or_pos m with hm | hm
      · subst hm
        apply slide.c1
        rw [step_iff]
        obtain ⟨a1, a2⟩ := hat
        simp only [Nat.zero_add] at a1 a2
        exact ⟨by simpa using a1, by simpa using a2⟩
      · obtain ⟨fm, rm, hatm⟩ := exists_at d f1 r1 f2 r2 m hat
        have h1 := ih fm rm hm hatm (fun k f r a b c => hray k f r a (by omega) c)
        refine slide.c2 h1 ((empty_iff p _ _).2 (hray m fm rm hm (by omega) hatm)) ?_
        rw [step_iff]
        obtain ⟨a1, a2⟩ := hat
        obtain ⟨b1, b2⟩ := hatm
        push_cast at a1 a2
        constructor
        · rw [a1, b1]; ring
        · rw [a2, b2]; ring

/-! ## The kernel's rook line is the same ray -/

/-- The kernel's `absDiff`, unfolded (as `Ideas.FrameLemma.absDiff_def`). -/
theorem absDiff_def (a b : Fin 8) : absDiff a b = if b.val ≤ a.val then a.val - b.val else b.val - a.val := rfl

/-- A square strictly between two squares of one file, on that file, is the kernel's `between`. -/
theorem between_file (s u t : Square) (h1 : s.1 = t.1) (h2 : u.1 = s.1)
    (h3 : (s.2.val < u.2.val ∧ u.2.val < t.2.val) ∨ (t.2.val < u.2.val ∧ u.2.val < s.2.val)) : between s u t := by
  simp only [between, betweenB, sameDiagonalB, sameFileB, sameRankB, betweenCoordB, ltB, Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq,
    beq_iff_eq, decide_eq_true_eq, Prod.ext_iff, Fin.ext_iff, absDiff_def] at *
  split_ifs at * <;> omega

/-- A square strictly between two squares of one rank, on that rank, is the kernel's `between`. -/
theorem between_rank (s u t : Square) (h1 : s.2 = t.2) (h2 : u.2 = s.2)
    (h3 : (s.1.val < u.1.val ∧ u.1.val < t.1.val) ∨ (t.1.val < u.1.val ∧ u.1.val < s.1.val)) : between s u t := by
  simp only [between, betweenB, sameDiagonalB, sameFileB, sameRankB, betweenCoordB, ltB, Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq,
    beq_iff_eq, decide_eq_true_eq, Prod.ext_iff, Fin.ext_iff, absDiff_def] at *
  split_ifs at * <;> omega

/-- On a rook's line the kernel's `between` is strictly between on that file or rank. -/
theorem between_rook (s u t : Square) (hg : rookGeometryB s t = true) (h : between s u t) :
    (s.1 = t.1 ∧ u.1 = s.1 ∧ ((s.2.val < u.2.val ∧ u.2.val < t.2.val) ∨ (t.2.val < u.2.val ∧ u.2.val < s.2.val))) ∨
    (s.2 = t.2 ∧ u.2 = s.2 ∧ ((s.1.val < u.1.val ∧ u.1.val < t.1.val) ∨ (t.1.val < u.1.val ∧ u.1.val < s.1.val))) := by
  simp only [rookGeometryB, between, betweenB, sameDiagonalB, sameFileB, sameRankB, betweenCoordB, ltB, Bool.or_eq_true, Bool.and_eq_true,
    bne_iff_ne, ne_eq, beq_iff_eq, decide_eq_true_eq, Prod.ext_iff, Fin.ext_iff, absDiff_def] at *
  split_ifs at * <;> omega

/-- THE KERNEL'S ROOK LINE IS THE RAY: `rookGeometryB s t && sliderAttackB p s t` (the kernel's rook attack, the rook's own square aside) holds exactly
when `t` is `n ≥ 1` steps from `s` along one of the four orthogonal directions with the ray open. -/
theorem kernel_rook_iff (p : Position) (s t : Square) :
    (rookGeometryB s t = true ∧ sliderAttackB p s t = true) ↔
      ∃ d, orthogonal d ∧ ∃ n, 1 ≤ n ∧ At d s.1 s.2 n t.1 t.2 ∧ Ray p d s.1 s.2 n := by
  have hclear : sliderAttackB p s t = true ↔ ∀ u, between s u t → p.board u = none := by
    have := clearBetween_iff p.board s t
    simp only [clearBetween, mem_allSquares, true_implies] at this
    exact this
  rw [hclear]
  constructor
  · rintro ⟨hg, hc⟩
    have hg' := hg
    simp only [rookGeometryB, sameFileB, sameRankB, Bool.and_eq_true, Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, Prod.ext_iff,
      Fin.ext_iff] at hg'
    obtain ⟨hne, hfr⟩ := hg'
    rcases hfr with hf | hr
    · rcases Nat.lt_or_gt_of_ne (fun e => hne ⟨hf, e⟩) with hlt | hgt
      · refine ⟨.north, .c1, t.2.val - s.2.val, by omega, ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩, ?_⟩
        rintro k f r hk1 hk2 ⟨a1, a2⟩
        simp only [dx, dy] at a1 a2
        exact hc (f, r) (between_file s (f, r) t (Fin.ext hf) (Fin.ext (by simp only; omega)) (Or.inl ⟨by simp only; omega, by simp only; omega⟩))
      · refine ⟨.south, .c2, s.2.val - t.2.val, by omega, ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩, ?_⟩
        rintro k f r hk1 hk2 ⟨a1, a2⟩
        simp only [dx, dy] at a1 a2
        exact hc (f, r) (between_file s (f, r) t (Fin.ext hf) (Fin.ext (by simp only; omega)) (Or.inr ⟨by simp only; omega, by simp only; omega⟩))
    · rcases Nat.lt_or_gt_of_ne (fun e => hne ⟨e, hr⟩) with hlt | hgt
      · refine ⟨.east, .c3, t.1.val - s.1.val, by omega, ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩, ?_⟩
        rintro k f r hk1 hk2 ⟨a1, a2⟩
        simp only [dx, dy] at a1 a2
        exact hc (f, r) (between_rank s (f, r) t (Fin.ext hr) (Fin.ext (by simp only; omega)) (Or.inl ⟨by simp only; omega, by simp only; omega⟩))
      · refine ⟨.west, .c4, s.1.val - t.1.val, by omega, ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩, ?_⟩
        rintro k f r hk1 hk2 ⟨a1, a2⟩
        simp only [dx, dy] at a1 a2
        exact hc (f, r) (between_rank s (f, r) t (Fin.ext hr) (Fin.ext (by simp only; omega)) (Or.inr ⟨by simp only; omega, by simp only; omega⟩))
  · rintro ⟨d, hd, n, hn, ⟨a1, a2⟩, hray⟩
    have hg : rookGeometryB s t = true := by
      simp only [rookGeometryB, sameFileB, sameRankB, Bool.and_eq_true, Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, Prod.ext_iff,
        Fin.ext_iff]
      cases hd <;> simp only [dx, dy] at a1 a2 <;> omega
    refine ⟨hg, ?_⟩
    intro u hu
    have hb := between_rook s u t hg hu
    obtain ⟨uf, ur⟩ := u
    simp only [Fin.ext_iff] at hb
    cases hd <;> simp only [dx, dy] at a1 a2
    · exact hray (ur.val - s.2.val) uf ur (by omega) (by omega) ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩
    · exact hray (s.2.val - ur.val) uf ur (by omega) (by omega) ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩
    · exact hray (uf.val - s.1.val) uf ur (by omega) (by omega) ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩
    · exact hray (s.1.val - uf.val) uf ur (by omega) (by omega) ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩

/-! ## THE EQUIVALENCE -/

/-- THE GENERATED ROOK ATTACK IS THE KERNEL'S, ON EVERY POSITION: the spec's `attack_by rook` holds from `s` to `t` exactly when a rook of colour `c`
stands on `s` and the kernel's `Chess.pieceAttacks` says it attacks `t`. -/
theorem attackByRook_iff (p : Position) (c : Color) (s t : Square) :
    attackByRook p c s.1 s.2 t.1 t.2 ↔ p.board s = some ⟨c, .rook⟩ ∧ pieceAttacks p c s t := by
  have hk : ∀ hb : p.board s = some ⟨c, .rook⟩, (pieceAttacks p c s t ↔ (rookGeometryB s t = true ∧ sliderAttackB p s t = true)) := by
    intro hb
    simp [pieceAttacks, pieceAttacksB, hb]
  constructor
  · rintro ⟨hb, ⟨hd⟩, hs⟩
    have hb' : p.board s = some ⟨c, .rook⟩ := hb
    refine ⟨hb', (hk hb').2 ((kernel_rook_iff p s t).2 ⟨_, hd, (slide_iff p _ _ _ _ _).1 hs⟩)⟩
  · rintro ⟨hb, ha⟩
    obtain ⟨d, hd, hray⟩ := (kernel_rook_iff p s t).1 ((hk hb).1 ha)
    exact attackByRook.c1 hb (lineOfRook.c1 hd) ((slide_iff p _ _ _ _ _).2 hray)

/-! ## The move and the capture -/

/-- `may_land` is the kernel's `destinationFriendlyB`: an empty square, or a unit of the other side that is not its king. -/
theorem mayLand_iff (p : Position) (c : Color) (f r : Fin 8) : mayLand p c f r ↔ destinationFriendlyB p c (f, r) = true := by
  constructor
  · rintro (⟨_, he⟩ | ⟨hb, ho, hj⟩)
    · simp [destinationFriendlyB, (empty_iff p f r).1 he]
    · cases ho <;> simp [destinationFriendlyB, hb, hj]
  · intro h
    cases hb : p.board (f, r) with
    | none =>
      exact mayLand.c1 (by cases c <;> constructor) ((empty_iff p f r).2 hb)
    | some q =>
      simp only [destinationFriendlyB, hb, Bool.and_eq_true, bne_iff_ne, ne_eq] at h
      obtain ⟨hc, hk⟩ := h
      obtain ⟨qc, qk⟩ := q
      refine mayLand.c2 hb ?_ hk
      cases c <;> cases qc <;> first | exact absurd rfl hc | constructor

/-- THE GENERATED ROOK MOVE IS THE KERNEL'S PSEUDO-LEGAL ROOK MOVE: the side to move is `c`, its rook stands on `s`, and the kernel's
`normalPseudoLegal` holds from `s` to `t` (its legality -- the king not exposed -- is the spec's `legal`, not this relation). -/
theorem moveByRook_iff (p : Position) (c : Color) (s t : Square) :
    moveByRook p c s.1 s.2 t.1 t.2 ↔ p.turn = c ∧ p.board s = some ⟨c, .rook⟩ ∧ normalPseudoLegal p s t := by
  constructor
  · rintro ⟨ht, ha, hl⟩
    obtain ⟨hb, hpa⟩ := (attackByRook_iff p c s t).1 ha
    have hk : rookGeometryB s t = true ∧ sliderAttackB p s t = true := by
      simpa [pieceAttacks, pieceAttacksB, hb] using hpa
    have hd := (mayLand_iff p c t.1 t.2).1 hl
    refine ⟨ht, hb, ?_⟩
    simp only [normalPseudoLegal, normalPseudoLegalB, hb, ht, normalPieceGeometryB]
    simp only [sliderAttackB] at hk
    simp [hd, hk.1, hk.2]
  · rintro ⟨ht, hb, hn⟩
    simp only [normalPseudoLegal, normalPseudoLegalB, hb, ht, normalPieceGeometryB, Bool.and_eq_true, beq_self_eq_true, true_and] at hn
    obtain ⟨⟨hd, _⟩, hg, hc⟩ := hn
    refine moveByRook.c1 ht ((attackByRook_iff p c s t).2 ⟨hb, ?_⟩) ((mayLand_iff p c t.1 t.2).2 hd)
    simp [pieceAttacks, pieceAttacksB, hb, hg, sliderAttackB, hc]

/-- THE GENERATED ROOK CAPTURE IS THE KERNEL'S: a rook move of the side to move onto a unit (the kernel's `moveIsCapture`). -/
theorem captureRook_iff (p : Position) (c : Color) (s t : Square) :
    captureRook p c s.1 s.2 t.1 t.2 ↔ moveByRook p c s.1 s.2 t.1 t.2 ∧ moveIsCapture p (.normal s t) = true := by
  constructor
  · rintro ⟨hm, hb, _⟩
    refine ⟨hm, ?_⟩
    have hb' : p.board t = some _ := hb
    simp [moveIsCapture, moveCaptureSquare, hb']
  · rintro ⟨hm, hcap⟩
    have hl : mayLand p c t.1 t.2 := by
      obtain ⟨_, _, hl⟩ := hm
      exact hl
    cases hq : p.board t with
    | none => simp [moveIsCapture, moveCaptureSquare, hq] at hcap
    | some q =>
      obtain ⟨qc, qk⟩ := q
      have hd := (mayLand_iff p c t.1 t.2).1 hl
      have hq' : p.board (t.1, t.2) = some ⟨qc, qk⟩ := hq
      simp only [destinationFriendlyB, hq', Bool.and_eq_true, bne_iff_ne, ne_eq] at hd
      refine captureRook.c1 hm hq' ?_
      cases c <;> cases qc <;> first | exact absurd rfl hd.1 | constructor

/-! ## An existing lemma, re-proved from the generated definition -/

/-- THE TWIN OF `Ideas.Region.rook_attacks_empty_file` ("a rook alone on its file (nothing else stands on it) attacks every other square of the file"),
over the GENERATED attack: a rook of colour `c` on (f, r) whose file holds nothing else attacks, by the spec's `attack_by rook`, every other square of the
file. Proved from the generated `slide` (the ray north or south of the rook, every square before it empty by the hypothesis), not from the kernel. -/
theorem rook_attacks_empty_file_spec (p : Position) (c : Color) (f r : Fin 8) (hrook : p.board (f, r) = some ⟨c, .rook⟩)
    (hempty : ∀ t : Fin 8, t ≠ r → p.board (f, t) = none) : ∀ y : Fin 8, y ≠ r → attackByRook p c f r f y := by
  intro y hy
  have hclear : ∀ (d : Dir) (n : Nat), orthogonal d → dx d = 0 → (∀ k, 1 ≤ k → k < n → (r.val : Int) + k * dy d ≠ r.val) → Ray p d f r n := by
    intro d n _ h0 hne k f' r' hk1 hk2 ⟨a1, a2⟩
    have ef : f' = f := Fin.ext (by rw [h0] at a1; omega)
    subst ef
    exact hempty r' (fun e => hne k hk1 hk2 (by rw [← a2, e]))
  rcases Nat.lt_or_gt_of_ne (fun e => hy (Fin.ext e)) with hlt | hgt
  · refine attackByRook.c1 hrook (lineOfRook.c1 orthogonal.c2) ((slide_iff p _ _ _ _ _).2 ⟨r.val - y.val, by omega, ⟨by simp only [dx]; omega,
      by simp only [dy]; omega⟩, hclear .south _ .c2 rfl (fun k h1 _ => by simp only [dy]; omega)⟩)
  · refine attackByRook.c1 hrook (lineOfRook.c1 orthogonal.c1) ((slide_iff p _ _ _ _ _).2 ⟨y.val - r.val, by omega, ⟨by simp only [dx]; omega,
      by simp only [dy]; omega⟩, hclear .north _ .c1 rfl (fun k h1 _ => by simp only [dy]; omega)⟩)

end Ideas.Spec.Rook
