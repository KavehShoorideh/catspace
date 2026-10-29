import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Ring
import Ideas.Spec.NextEquiv
import Ideas.Mate
import Ideas.Spec.DeadPosition

/-!
# The spec's `legal`, `terminal` and `goal` are the kernel's (ANALYST_PLAN 76.2 (a), 2026-10-28)

`Ideas.Spec.Chess` is GENERATED from rules/chess.gdl. `Ideas.Spec.NextEquiv` (76.1) proved its next state the kernel's `applyMove` on every pseudo-legal
move. This file proves, BY REASONING, the rest of the game layer:

* THE MOVES. `legal_iff`: on every position where the side to move has one king (`Chess.exactlyOneKing`) and the e.p. square stands where the rule puts it
  (`EpPlaced`: the sixth rank for White to move, the third for Black -- the spec's `ep_victim` reads no rank, the kernel's `epCapturedSquare` does), the
  spec's `legal p c mv` holds exactly when `c` is the side to move and `mv` is the term (`termOf`) of a move the kernel calls legal (`Chess.legalMove`), or
  `c` is the other side and `mv` is noop. `legal_iff_reachable`: on every reachable position, no hypothesis left (`Chess.reachable_positionInvariant`,
  `reachable_epPlaced`). With NextEquiv this makes the generated spec a certified model of the kernel for moves and next states. The hypotheses are needed:
  a side with no king is in check for the kernel (`Chess.inCheckB`), never exposed for the spec; a misplaced e.p. square lets the spec take en passant where
  the kernel has no captured square.
* HOW: every relation `legal` reads gets its closed form -- the steps and the slides are rays (`step_iff`, `slide_iff`), the kernel's lines the same rays
  (`rook_line_iff`, `bishop_line_iff`, `ray_clear`), so the spec's attack is the kernel's for every kind (`attackBy_iff`, `attacked_iff`, `check_iff`); its
  moves the kernel's pseudo-legal moves (`moveBy_iff`, `enPassant_of_kernel`, `kernel_of_enPassant`, `castle_iff`, `candidate_iff`); the counterfactual board
  of legality (`after`) the spec's own next board (`after_iff_nextCell`), hence the kernel's (`applyMoveSpec_applyMove`); and the spec's exposure the
  kernel's check after the move (`exposed_iff_inCheck_move`, `exposed_iff_inCheck_promotion`: a king attacked on a board read as a relation, `ExposedR`,
  is `Chess.isAttacked` there, `isAttacked_iff_exposedR`). The tables are read by their rows as in NextEquiv; no position and no square pair is enumerated;
  no `decide`, no `native_decide`, no `sorry`.
* THE END. `mate_iff` (= `Chess.checkmate`), `stalemate_iff` (= `Chess.stalemate`), `mate_iff_mated` / `stalemate_iff_stalemated` (= the machinery's
  `Ideas.Game.chess`), `drawByFiftyMoves_iff` (= `Chess.fiftyMoveClaimable` with a legal move), the repetition through the record (76.1's
  `drawByRepetition_kernel`); `goal_win_iff` / `goal_loss_iff` (a side scores 100 / 0 exactly when `Chess.gameStatus` says it won / lost).
  `terminal_sound`: a position the spec ends, `Chess.gameStatus` ends -- EXCEPT POSSIBLY the criterion's knight clause (K+N against kings and queens), whose
  dead position is proved only in the position itself: a CONJECTURE (the store's `dead position (the knight clause)`). `terminal_complete`: a position
  `Chess.gameStatus` ends, the spec ends, or it is a FIDE dead position beyond the spec's material criterion -- the spec is the engine's rule there, not
  FIDE 5.2.2 (a locked pawn chain is dead and not the spec's draw), so the two terminals are NOT equal and this is the exact obstacle.
-/

namespace Ideas.LegalEquiv

open Chess Ideas.Spec.Chess
open Ideas.NextEquiv (file_all rank_all square_all fileSucc_iff rankSucc_iff opponent_iff occupied_iff empty_iff homeRank_iff kingTo rookTo rookFrom
  castlingPath_iff pawnStart_iff forward_iff step_north_iff step_south_iff epVictim_iff home)
open Ideas.Spec.Binding (Wing right clock MoveTerm moveOf termOf promoKind castleMove)

/-! ## A step is a unit vector; a line is a ray -/

def dx : Dir → Int
  | .east => 1
  | .west => -1
  | .north => 0
  | .south => 0
  | .northeast => 1
  | .northwest => -1
  | .southeast => 1
  | .southwest => -1

def dy : Dir → Int
  | .east => 0
  | .west => 0
  | .north => 1
  | .south => -1
  | .northeast => 1
  | .northwest => 1
  | .southeast => -1
  | .southwest => -1

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

/-- Every direction is a line of the rook or of the bishop. -/
theorem orth_or_diag (d : Dir) : orthogonal d ∨ diagonal d := by
  cases d
  · exact Or.inl .c3
  · exact Or.inl .c1
  · exact Or.inr .c1
  · exact Or.inr .c2
  · exact Or.inl .c2
  · exact Or.inr .c3
  · exact Or.inr .c4
  · exact Or.inl .c4

theorem orthogonal_iff (d : Dir) : orthogonal d ↔ (dx d = 0 ∨ dy d = 0) := by
  constructor
  · intro h; cases h <;> simp [dx, dy]
  · intro h; cases d <;> simp [dx, dy] at h <;> constructor

theorem diagonal_iff (d : Dir) : diagonal d ↔ (dx d ≠ 0 ∧ dy d ≠ 0) := by
  constructor
  · intro h; cases h <;> simp [dx, dy]
  · intro h; cases d <;> simp [dx, dy] at h <;> constructor

/-- The square `k` steps from `s` along `d` is `x`. -/
def At (d : Dir) (s : Square) (k : Nat) (x : Square) : Prop :=
  (x.1.val : Int) = s.1.val + k * dx d ∧ (x.2.val : Int) = s.2.val + k * dy d

/-- THE OPEN RAY of length `n` from `s` along `d` over an occupancy `occ`: no square strictly before the last is occupied. -/
def Open (occ : Square → Prop) (d : Dir) (s : Square) (n : Nat) : Prop :=
  ∀ k x, 1 ≤ k → k < n → At d s k x → ¬ occ x

/-- THE RAY: `t` is `n ≥ 1` steps from `s` along `d`, the ray open. -/
def Ray (occ : Square → Prop) (d : Dir) (s t : Square) : Prop :=
  ∃ n, 1 ≤ n ∧ At d s n t ∧ Open occ d s n

/-- A board's occupancy. -/
def Occ (b : Board) (x : Square) : Prop := b x ≠ none

theorem at_unique {d : Dir} {s : Square} {k : Nat} {x y : Square} (h1 : At d s k x) (h2 : At d s k y) : x = y := by
  obtain ⟨a1, b1⟩ := h1
  obtain ⟨a2, b2⟩ := h2
  exact Prod.ext (Fin.ext (by omega)) (Fin.ext (by omega))

theorem exists_at (d : Dir) (s t : Square) (m : Nat) (h : At d s (m + 1) t) : ∃ x : Square, At d s m x := by
  obtain ⟨h1, h2⟩ := h
  have hf1 := s.1.isLt
  have hf2 := t.1.isLt
  have hr1 := s.2.isLt
  have hr2 := t.2.isLt
  have hx : 0 ≤ (s.1.val : Int) + m * dx d ∧ (s.1.val : Int) + m * dx d < 8 := by
    cases d <;> simp only [dx] at h1 ⊢ <;> push_cast at h1 ⊢ <;> omega
  have hy : 0 ≤ (s.2.val : Int) + m * dy d ∧ (s.2.val : Int) + m * dy d < 8 := by
    cases d <;> simp only [dy] at h2 ⊢ <;> push_cast at h2 ⊢ <;> omega
  exact ⟨(⟨((s.1.val : Int) + m * dx d).toNat, by omega⟩, ⟨((s.2.val : Int) + m * dy d).toNat, by omega⟩), by simp only; omega, by simp only; omega⟩

theorem at_one {d : Dir} {s x : Square} : At d s 1 x ↔ step d s.1 s.2 x.1 x.2 := by
  rw [step_iff]
  simp [At]

/-- One more step along the ray. -/
theorem at_succ {d : Dir} {s x y : Square} {n : Nat} (hx : At d s n x) (hs : step d x.1 x.2 y.1 y.2) : At d s (n + 1) y := by
  rw [step_iff] at hs
  obtain ⟨a1, a2⟩ := hx
  refine ⟨?_, ?_⟩
  · rw [hs.1, a1]; push_cast; ring
  · rw [hs.2, a2]; push_cast; ring

theorem at_pred {d : Dir} {s x y : Square} {n : Nat} (hx : At d s n x) (hy : At d s (n + 1) y) : step d x.1 x.2 y.1 y.2 := by
  rw [step_iff]
  obtain ⟨a1, a2⟩ := hx
  obtain ⟨b1, b2⟩ := hy
  push_cast at b1 b2
  constructor
  · rw [b1, a1]; ring
  · rw [b2, a2]; ring

/-- THE SLIDE IS THE RAY over the position's board. -/
theorem slide_iff' (p : Position) (d : Dir) (s : Square) (f2 r2 : Fin 8) : slide p d s.1 s.2 f2 r2 ↔ Ray (Occ p.board) d s (f2, r2) := by
  constructor
  · intro h
    induction h with
    | c1 hs =>
      refine ⟨1, le_refl _, at_one.2 hs, ?_⟩
      intro k x h1 h2
      omega
    | @c2 f3 r3 f2 r2 _ he hs ih =>
      obtain ⟨n, hn, hat, hopen⟩ := ih
      refine ⟨n + 1, by omega, at_succ (x := (f2, r2)) hat hs, ?_⟩
      intro k x hk1 hk2 hk
      by_cases hkn : k < n
      · exact hopen k x hk1 hkn hk
      · have hk' : k = n := by omega
        subst hk'
        rw [at_unique hk hat]
        exact fun ho => ho ((empty_iff p _ _).1 he)
  · rintro ⟨n, hn, hat, hopen⟩
    induction n generalizing f2 r2 with
    | zero => omega
    | succ m ih =>
      rcases Nat.eq_zero_or_pos m with hm | hm
      · subst hm
        exact slide.c1 (at_one.1 hat)
      · obtain ⟨x, hx⟩ := exists_at d s (f2, r2) m hat
        have h1 := ih x.1 x.2 hm hx (fun k y a b c => hopen k y a (by omega) c)
        have hxe : p.board x = none := by
          by_contra hc
          exact hopen m x hm (by omega) hx hc
        exact slide.c2 (f2 := x.1) (r2 := x.2) h1 ((empty_iff p _ _).2 hxe) (at_pred hx hat)

theorem slide_iff (p : Position) (d : Dir) (s t : Square) : slide p d s.1 s.2 t.1 t.2 ↔ Ray (Occ p.board) d s t := slide_iff' p d s t.1 t.2

/-! ## The kernel's lines are the rays -/

theorem absDiff_def (a b : Fin 8) : absDiff a b = if b.val ≤ a.val then a.val - b.val else b.val - a.val := rfl

theorem clear_iff (b : Board) (s t : Square) : clearBetweenB b s t = true ↔ ∀ u, between s u t → b u = none := by
  have := clearBetween_iff b s t
  simp only [clearBetween, mem_allSquares, true_implies] at this
  exact this

theorem rookGeometry_iff (s t : Square) : rookGeometryB s t = true ↔ s ≠ t ∧ (s.1 = t.1 ∨ s.2 = t.2) := by
  simp only [rookGeometryB, sameFileB, sameRankB, Bool.and_eq_true, Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq]

theorem bishopGeometry_iff (s t : Square) : bishopGeometryB s t = true ↔ s ≠ t ∧ absDiff s.1 t.1 = absDiff s.2 t.2 := by
  simp only [bishopGeometryB, sameDiagonalB, Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq]

/-- A ray's end is on its line's geometry (a rook's for an orthogonal direction, a bishop's for a diagonal one, never both). -/
theorem ray_geometry (occ : Square → Prop) (d : Dir) (s t : Square) (h : Ray occ d s t) :
    (orthogonal d → rookGeometryB s t = true ∧ bishopGeometryB s t = false) ∧ (diagonal d → bishopGeometryB s t = true ∧ rookGeometryB s t = false) := by
  obtain ⟨n, hn, ⟨a1, a2⟩, _⟩ := h
  obtain ⟨⟨sf, hsf⟩, ⟨sr, hsr⟩⟩ := s
  obtain ⟨⟨tf, htf⟩, ⟨tr, htr⟩⟩ := t
  simp only at a1 a2
  have hb : bishopGeometryB (⟨sf, hsf⟩, ⟨sr, hsr⟩) (⟨tf, htf⟩, ⟨tr, htr⟩) = true ↔ (sf ≠ tf ∨ sr ≠ tr) ∧
      (if tf ≤ sf then sf - tf else tf - sf) = (if tr ≤ sr then sr - tr else tr - sr) := by
    rw [bishopGeometry_iff, absDiff_def, absDiff_def]
    simp only [ne_eq, Prod.mk.injEq, Fin.mk.injEq, not_and_or]
  have hr : rookGeometryB (⟨sf, hsf⟩, ⟨sr, hsr⟩) (⟨tf, htf⟩, ⟨tr, htr⟩) = true ↔ (sf ≠ tf ∨ sr ≠ tr) ∧ (sf = tf ∨ sr = tr) := by
    rw [rookGeometry_iff]
    simp only [ne_eq, Prod.mk.injEq, Fin.mk.injEq, not_and_or]
  refine ⟨fun hd => ?_, fun hd => ?_⟩
  · cases hd <;> simp only [dx, dy] at a1 a2 <;> refine ⟨hr.2 ⟨by omega, by omega⟩, ?_⟩ <;>
      (cases hbb : bishopGeometryB _ _ <;> [rfl; (rw [hb] at hbb; split_ifs at hbb <;> omega)])
  · cases hd <;> simp only [dx, dy] at a1 a2 <;> refine ⟨hb.2 ⟨by omega, by split_ifs <;> omega⟩, ?_⟩ <;>
      (cases hrr : rookGeometryB _ _ <;> [rfl; (rw [hr] at hrr; omega)])

/-- A square the kernel's `between` puts strictly between a ray's ends is one of the ray's squares before its end. -/
theorem between_on_ray (d : Dir) (s t u : Square) (n : Nat) (hn : 1 ≤ n) (hat : At d s n t) (hu : between s u t) :
    ∃ k, 1 ≤ k ∧ k < n ∧ At d s k u := by
  obtain ⟨⟨sf, hsf⟩, ⟨sr, hsr⟩⟩ := s
  obtain ⟨⟨tf, htf⟩, ⟨tr, htr⟩⟩ := t
  obtain ⟨⟨uf, huf⟩, ⟨ur, hur⟩⟩ := u
  obtain ⟨a1, a2⟩ := hat
  simp only at a1 a2
  simp only [between, betweenB, sameDiagonalB, sameFileB, sameRankB, betweenCoordB, ltB, Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq,
    beq_iff_eq, decide_eq_true_eq, Prod.mk.injEq, Fin.mk.injEq, absDiff_def, true_and, and_true] at hu
  simp only [At]
  cases d <;> simp only [dx, dy] at a1 a2 ⊢
  · exact ⟨uf - sf, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega⟩
  · exact ⟨ur - sr, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega⟩
  · exact ⟨ur - sr, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega⟩
  · exact ⟨ur - sr, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega⟩
  · exact ⟨sr - ur, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega⟩
  · exact ⟨sr - ur, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega⟩
  · exact ⟨sr - ur, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega⟩
  · exact ⟨sf - uf, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega, by split_ifs at hu <;> omega⟩

/-- A ray's squares before its end are between its ends. -/
theorem on_ray_between (d : Dir) (s t u : Square) (n k : Nat) (hat : At d s n t) (hk1 : 1 ≤ k) (hk2 : k < n) (hu : At d s k u) : between s u t := by
  obtain ⟨⟨sf, hsf⟩, ⟨sr, hsr⟩⟩ := s
  obtain ⟨⟨tf, htf⟩, ⟨tr, htr⟩⟩ := t
  obtain ⟨⟨uf, huf⟩, ⟨ur, hur⟩⟩ := u
  obtain ⟨a1, a2⟩ := hat
  obtain ⟨b1, b2⟩ := hu
  simp only at a1 a2 b1 b2
  simp only [between, betweenB, sameDiagonalB, sameFileB, sameRankB, betweenCoordB, ltB, Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq,
    beq_iff_eq, decide_eq_true_eq, Prod.mk.injEq, Fin.mk.injEq, absDiff_def, true_and, and_true]
  cases d <;> simp only [dx, dy] at a1 a2 b1 b2 <;> split_ifs <;> omega

/-- THE RAY IS THE KERNEL'S LINE: an open ray's squares strictly between its ends are empty, and conversely. -/
theorem ray_clear (b : Board) (d : Dir) (s t : Square) (n : Nat) (hn : 1 ≤ n) (hat : At d s n t) :
    Open (Occ b) d s n ↔ clearBetweenB b s t = true := by
  rw [clear_iff]
  constructor
  · intro ho u hu
    obtain ⟨k, hk1, hk2, hk⟩ := between_on_ray d s t u n hn hat hu
    by_contra hc
    exact ho k u hk1 hk2 hk hc
  · intro hc k u hk1 hk2 hk ho
    exact ho (hc u (on_ray_between d s t u n k hat hk1 hk2 hk))

theorem rook_line_iff (b : Board) (s t : Square) :
    (rookGeometryB s t = true ∧ clearBetweenB b s t = true) ↔ ∃ d, orthogonal d ∧ Ray (Occ b) d s t := by
  constructor
  · rintro ⟨hg, hc⟩
    rw [rookGeometry_iff] at hg
    obtain ⟨⟨sf, hsf⟩, ⟨sr, hsr⟩⟩ := s
    obtain ⟨⟨tf, htf⟩, ⟨tr, htr⟩⟩ := t
    simp only [ne_eq, Prod.mk.injEq, Fin.mk.injEq, not_and_or] at hg
    obtain ⟨hne, hfr⟩ := hg
    have mk : ∀ d n, orthogonal d → 1 ≤ n → At d (⟨sf, hsf⟩, ⟨sr, hsr⟩) n (⟨tf, htf⟩, ⟨tr, htr⟩) →
        ∃ d, orthogonal d ∧ Ray (Occ b) d (⟨sf, hsf⟩, ⟨sr, hsr⟩) (⟨tf, htf⟩, ⟨tr, htr⟩) :=
      fun d n hd hn hat => ⟨d, hd, n, hn, hat, (ray_clear b d _ _ n hn hat).2 hc⟩
    rcases hfr with hf | hr
    · rcases Nat.lt_or_gt_of_ne (fun e => (hne.resolve_left (not_not.2 hf)) e) with hlt | hgt
      · exact mk .north (tr - sr) .c1 (by omega) ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩
      · exact mk .south (sr - tr) .c2 (by omega) ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩
    · rcases Nat.lt_or_gt_of_ne (fun e => (hne.resolve_right (not_not.2 hr)) e) with hlt | hgt
      · exact mk .east (tf - sf) .c3 (by omega) ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩
      · exact mk .west (sf - tf) .c4 (by omega) ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩
  · rintro ⟨d, hd, hray⟩
    refine ⟨((ray_geometry _ d s t hray).1 hd).1, ?_⟩
    obtain ⟨n, hn, hat, ho⟩ := hray
    exact (ray_clear b d s t n hn hat).1 ho

theorem bishop_line_iff (b : Board) (s t : Square) :
    (bishopGeometryB s t = true ∧ clearBetweenB b s t = true) ↔ ∃ d, diagonal d ∧ Ray (Occ b) d s t := by
  constructor
  · rintro ⟨hg, hc⟩
    rw [bishopGeometry_iff, absDiff_def, absDiff_def] at hg
    obtain ⟨⟨sf, hsf⟩, ⟨sr, hsr⟩⟩ := s
    obtain ⟨⟨tf, htf⟩, ⟨tr, htr⟩⟩ := t
    simp only [ne_eq, Prod.mk.injEq, Fin.mk.injEq, not_and_or] at hg
    obtain ⟨hne, he⟩ := hg
    have mk : ∀ d n, diagonal d → 1 ≤ n → At d (⟨sf, hsf⟩, ⟨sr, hsr⟩) n (⟨tf, htf⟩, ⟨tr, htr⟩) →
        ∃ d, diagonal d ∧ Ray (Occ b) d (⟨sf, hsf⟩, ⟨sr, hsr⟩) (⟨tf, htf⟩, ⟨tr, htr⟩) :=
      fun d n hd hn hat => ⟨d, hd, n, hn, hat, (ray_clear b d _ _ n hn hat).2 hc⟩

    split_ifs at he with h1 h2 h2
    · exact mk .southwest (sf - tf) .c4 (by omega) ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩
    · exact mk .northwest (sf - tf) .c2 (by omega) ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩
    · exact mk .southeast (tf - sf) .c3 (by omega) ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩
    · exact mk .northeast (tf - sf) .c1 (by omega) ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩
  · rintro ⟨d, hd, hray⟩
    refine ⟨((ray_geometry _ d s t hray).2 hd).1, ?_⟩
    obtain ⟨n, hn, hat, ho⟩ := hray
    exact (ray_clear b d s t n hn hat).1 ho

/-- The opposite direction. -/
def opp : Dir → Dir
  | .east => .west
  | .west => .east
  | .north => .south
  | .south => .north
  | .northeast => .southwest
  | .southwest => .northeast
  | .northwest => .southeast
  | .southeast => .northwest

theorem dx_opp (d : Dir) : dx (opp d) = - dx d := by cases d <;> rfl
theorem dy_opp (d : Dir) : dy (opp d) = - dy d := by cases d <;> rfl
theorem orthogonal_opp {d : Dir} (h : orthogonal d) : orthogonal (opp d) := by cases h <;> constructor
theorem diagonal_opp {d : Dir} (h : diagonal d) : diagonal (opp d) := by cases h <;> constructor

/-- A RAY READ BACKWARDS is a ray: the same squares from its end, the other way. -/
theorem ray_reverse (occ : Square → Prop) (d : Dir) (s t : Square) (h : Ray occ d s t) : Ray occ (opp d) t s := by
  obtain ⟨n, hn, ⟨a1, a2⟩, ho⟩ := h
  refine ⟨n, hn, ⟨?_, ?_⟩, ?_⟩
  · rw [dx_opp, a1]; ring
  · rw [dy_opp, a2]; ring
  · intro k x hk1 hk2 ⟨b1, b2⟩
    refine ho (n - k) x (by omega) (by omega) ⟨?_, ?_⟩
    · rw [dx_opp, a1] at b1
      rw [b1]
      push_cast [show k ≤ n by omega]
      ring
    · rw [dy_opp, a2] at b2
      rw [b2]
      push_cast [show k ≤ n by omega]
      ring

theorem opp_opp (d : Dir) : opp (opp d) = d := by cases d <;> rfl

theorem ray_reverse_iff (occ : Square → Prop) (s t : Square) (P : Dir → Prop) (hP : ∀ d, P d → P (opp d)) :
    (∃ d, P d ∧ Ray occ d s t) ↔ (∃ d, P d ∧ Ray occ d t s) := by
  constructor
  · rintro ⟨d, hd, hr⟩
    exact ⟨opp d, hP d hd, ray_reverse occ d s t hr⟩
  · rintro ⟨d, hd, hr⟩
    exact ⟨opp d, hP d hd, ray_reverse occ d t s hr⟩

/-! ## The jumps and the steps -/

theorem knightJump_iff (f1 r1 f2 r2 : Fin 8) : knightJump f1 r1 f2 r2 ↔ knightGeometryB (f1, r1) (f2, r2) = true := by
  simp only [knightGeometryB, absDiff_def, Bool.and_eq_true, Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, Prod.mk.injEq, Fin.ext_iff]
  constructor
  · intro h
    cases h <;> simp only [fileSucc_iff, rankSucc_iff] at * <;> split_ifs <;> omega
  · intro h
    have hf1 := f1.isLt
    have hf2 := f2.isLt
    have hr1 := r1.isLt
    have hr2 := r2.isLt
    split_ifs at h with e1 e2 e2
    · rcases h with ⟨_, ⟨h3, h4⟩ | ⟨h3, h4⟩⟩
      · exact knightJump.c4 (x := ⟨r2.val + 1, by omega⟩) ((fileSucc_iff _ _).2 (by omega)) ((rankSucc_iff _ _).2 rfl) ((rankSucc_iff _ _).2 (by simp only; omega))
      · exact knightJump.c8 (x := ⟨f2.val + 1, by omega⟩) ((fileSucc_iff _ _).2 rfl) ((fileSucc_iff _ _).2 (by simp only; omega)) ((rankSucc_iff _ _).2 (by omega))
    · rcases h with ⟨_, ⟨h3, h4⟩ | ⟨h3, h4⟩⟩
      · exact knightJump.c2 (x := ⟨r1.val + 1, by omega⟩) ((fileSucc_iff _ _).2 (by omega)) ((rankSucc_iff _ _).2 rfl) ((rankSucc_iff _ _).2 (by simp only; omega))
      · exact knightJump.c6 (x := ⟨f2.val + 1, by omega⟩) ((fileSucc_iff _ _).2 rfl) ((fileSucc_iff _ _).2 (by simp only; omega)) ((rankSucc_iff _ _).2 (by omega))
    · rcases h with ⟨_, ⟨h3, h4⟩ | ⟨h3, h4⟩⟩
      · exact knightJump.c3 (x := ⟨r2.val + 1, by omega⟩) ((fileSucc_iff _ _).2 (by omega)) ((rankSucc_iff _ _).2 rfl) ((rankSucc_iff _ _).2 (by simp only; omega))
      · exact knightJump.c7 (x := ⟨f1.val + 1, by omega⟩) ((fileSucc_iff _ _).2 rfl) ((fileSucc_iff _ _).2 (by simp only; omega)) ((rankSucc_iff _ _).2 (by omega))
    · rcases h with ⟨_, ⟨h3, h4⟩ | ⟨h3, h4⟩⟩
      · exact knightJump.c1 (x := ⟨r1.val + 1, by omega⟩) ((fileSucc_iff _ _).2 (by omega)) ((rankSucc_iff _ _).2 rfl) ((rankSucc_iff _ _).2 (by simp only; omega))
      · exact knightJump.c5 (x := ⟨f1.val + 1, by omega⟩) ((fileSucc_iff _ _).2 rfl) ((fileSucc_iff _ _).2 (by simp only; omega)) ((rankSucc_iff _ _).2 (by omega))

theorem knightGeometry_symm (s t : Square) : knightGeometryB s t = knightGeometryB t s := by
  apply Bool.eq_iff_iff.2
  simp only [knightGeometryB, absDiff_def, Bool.and_eq_true, Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, Prod.ext_iff, Fin.ext_iff]
  constructor <;> intro h <;> split_ifs at h ⊢ <;> omega

theorem kingStep_iff (f1 r1 f2 r2 : Fin 8) : kingStep f1 r1 f2 r2 ↔ kingGeometryB (f1, r1) (f2, r2) = true := by
  have key : kingStep f1 r1 f2 r2 ↔ ∃ d, step d f1 r1 f2 r2 := ⟨fun ⟨h⟩ => ⟨_, h⟩, fun ⟨_, h⟩ => kingStep.c1 h⟩
  rw [key]
  simp only [kingGeometryB, absDiff_def, Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq, Prod.mk.injEq, Fin.ext_iff, step_iff]
  constructor
  · rintro ⟨d, h1, h2⟩
    cases d <;> simp only [dx, dy] at h1 h2 <;> split_ifs <;> omega
  · rintro ⟨⟨hne, h1⟩, h2⟩
    split_ifs at h1 h2 with e1 e2 e2
    · rcases Nat.lt_or_ge f2.val f1.val with a | a <;> rcases Nat.lt_or_ge r2.val r1.val with b | b
      · exact ⟨.southwest, by simp only [dx]; omega, by simp only [dy]; omega⟩
      · exact ⟨.west, by simp only [dx]; omega, by simp only [dy]; omega⟩
      · exact ⟨.south, by simp only [dx]; omega, by simp only [dy]; omega⟩
      · exfalso; omega
    · rcases Nat.lt_or_ge f2.val f1.val with a | a
      · exact ⟨.northwest, by simp only [dx]; omega, by simp only [dy]; omega⟩
      · exact ⟨.north, by simp only [dx]; omega, by simp only [dy]; omega⟩
    · rcases Nat.lt_or_ge r2.val r1.val with b | b
      · exact ⟨.southeast, by simp only [dx]; omega, by simp only [dy]; omega⟩
      · exact ⟨.east, by simp only [dx]; omega, by simp only [dy]; omega⟩
    · exact ⟨.northeast, by simp only [dx]; omega, by simp only [dy]; omega⟩

theorem kingGeometry_symm (s t : Square) : kingGeometryB s t = kingGeometryB t s := by
  apply Bool.eq_iff_iff.2
  simp only [kingGeometryB, absDiff_def, Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq, Prod.ext_iff, Fin.ext_iff]
  constructor <;> intro h <;> split_ifs at h ⊢ <;> omega

theorem pawnStep_iff (c : Color) (f1 r1 f2 r2 : Fin 8) : (∃ d, pawnDiagonal c d ∧ step d f1 r1 f2 r2) ↔ pawnAttackGeometryB c (f1, r1) (f2, r2) = true := by
  simp only [pawnAttackGeometryB, absDiff_def, Bool.and_eq_true, beq_iff_eq, step_iff]
  constructor
  · rintro ⟨d, hd, h1, h2⟩
    cases hd <;> simp only [dx, dy] at h1 h2 <;> simp <;> split_ifs <;> omega
  · rintro ⟨h1, h2⟩
    cases c <;> simp at h2 <;> split_ifs at h1
    · exact ⟨.northwest, .c2, by simp only [dx]; omega, by simp only [dy]; omega⟩
    · exact ⟨.northeast, .c1, by simp only [dx]; omega, by simp only [dy]; omega⟩
    · exact ⟨.southwest, .c4, by simp only [dx]; omega, by simp only [dy]; omega⟩
    · exact ⟨.southeast, .c3, by simp only [dx]; omega, by simp only [dy]; omega⟩

theorem lineOf_iff (k : PieceKind) (d : Dir) : lineOf k d ↔
    (k = .rook ∧ orthogonal d) ∨ (k = .bishop ∧ diagonal d) ∨ (k = .queen ∧ (orthogonal d ∨ diagonal d)) := by
  constructor
  · intro h
    cases h with
    | c1 h => exact Or.inl ⟨rfl, h⟩
    | c2 h => exact Or.inr (Or.inl ⟨rfl, h⟩)
    | c3 h => exact Or.inr (Or.inr ⟨rfl, Or.inl h⟩)
    | c4 h => exact Or.inr (Or.inr ⟨rfl, Or.inr h⟩)
  · rintro (⟨rfl, h⟩ | ⟨rfl, h⟩ | ⟨rfl, h | h⟩)
    · exact .c1 h
    · exact .c2 h
    · exact .c3 h
    · exact .c4 h

/-! ## THE ATTACK IS THE KERNEL'S, for every kind, on every position -/

theorem ray_congr {o1 o2 : Square → Prop} (h : ∀ x, o1 x ↔ o2 x) (d : Dir) (s t : Square) : Ray o1 d s t ↔ Ray o2 d s t := by
  unfold Ray Open
  simp only [h]

/-- The kernel's piece attack once the unit is known. -/
theorem pieceAttacks_of (p : Position) (c : Color) (k : PieceKind) (s t : Square) (hb : p.board s = some ⟨c, k⟩) :
    pieceAttacks p c s t ↔ (k = .rook ∧ rookGeometryB s t = true ∧ clearBetweenB p.board s t = true) ∨
      (k = .bishop ∧ bishopGeometryB s t = true ∧ clearBetweenB p.board s t = true) ∨
      (k = .queen ∧ (rookGeometryB s t = true ∨ bishopGeometryB s t = true) ∧ clearBetweenB p.board s t = true) ∨
      (k = .knight ∧ knightGeometryB s t = true) ∨ (k = .king ∧ kingGeometryB s t = true) ∨ (k = .pawn ∧ pawnAttackGeometryB c s t = true) := by
  simp only [pieceAttacks, pieceAttacksB, hb, sliderAttackB, queenGeometryB, beq_self_eq_true, Bool.true_and, Bool.or_eq_true, Bool.and_eq_true,
    beq_iff_eq]
  cases k <;> simp

theorem slider_line_iff (b : Board) (k : PieceKind) (s t : Square) :
    (∃ d, lineOf k d ∧ Ray (Occ b) d s t) ↔ ((k = .rook ∧ rookGeometryB s t = true ∧ clearBetweenB b s t = true) ∨
      (k = .bishop ∧ bishopGeometryB s t = true ∧ clearBetweenB b s t = true) ∨
      (k = .queen ∧ (rookGeometryB s t = true ∨ bishopGeometryB s t = true) ∧ clearBetweenB b s t = true)) := by
  simp only [lineOf_iff]
  constructor
  · rintro ⟨d, ⟨rfl, hd⟩ | ⟨rfl, hd⟩ | ⟨rfl, hd | hd⟩, hr⟩
    · exact Or.inl ⟨rfl, (rook_line_iff b s t).2 ⟨d, hd, hr⟩⟩
    · exact Or.inr (Or.inl ⟨rfl, (bishop_line_iff b s t).2 ⟨d, hd, hr⟩⟩)
    · obtain ⟨h1, h2⟩ := (rook_line_iff b s t).2 ⟨d, hd, hr⟩
      exact Or.inr (Or.inr ⟨rfl, Or.inl h1, h2⟩)
    · obtain ⟨h1, h2⟩ := (bishop_line_iff b s t).2 ⟨d, hd, hr⟩
      exact Or.inr (Or.inr ⟨rfl, Or.inr h1, h2⟩)
  · rintro (⟨rfl, h⟩ | ⟨rfl, h⟩ | ⟨rfl, h1 | h1, h2⟩)
    · obtain ⟨d, hd, hr⟩ := (rook_line_iff b s t).1 h
      exact ⟨d, Or.inl ⟨rfl, hd⟩, hr⟩
    · obtain ⟨d, hd, hr⟩ := (bishop_line_iff b s t).1 h
      exact ⟨d, Or.inr (Or.inl ⟨rfl, hd⟩), hr⟩
    · obtain ⟨d, hd, hr⟩ := (rook_line_iff b s t).1 ⟨h1, h2⟩
      exact ⟨d, Or.inr (Or.inr ⟨rfl, Or.inl hd⟩), hr⟩
    · obtain ⟨d, hd, hr⟩ := (bishop_line_iff b s t).1 ⟨h1, h2⟩
      exact ⟨d, Or.inr (Or.inr ⟨rfl, Or.inr hd⟩), hr⟩

/-- THE GENERATED ATTACK IS THE KERNEL'S, FOR EVERY KIND, ON EVERY POSITION: the spec's `attack_by k` holds from `s` to `t` exactly when a unit of kind `k`
and colour `c` stands on `s` and the kernel's `Chess.pieceAttacks` says it attacks `t` (76.0 proved the rook's alone). -/
theorem attackBy_iff (p : Position) (k : PieceKind) (c : Color) (s t : Square) :
    attackBy p k c s.1 s.2 t.1 t.2 ↔ p.board s = some ⟨c, k⟩ ∧ pieceAttacks p c s t := by
  constructor
  · intro h
    cases h with
    | c1 hb hl hs =>
      refine ⟨hb, (pieceAttacks_of p c _ s t hb).2 ?_⟩
      have := (slider_line_iff p.board _ s t).1 ⟨_, hl, (slide_iff p _ s t).1 hs⟩
      rcases this with h | h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr (Or.inl h))
    | c2 hb hj => exact ⟨hb, (pieceAttacks_of p c _ s t hb).2 (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, (knightJump_iff _ _ _ _).1 hj⟩))))⟩
    | c3 hb hj => exact ⟨hb, (pieceAttacks_of p c _ s t hb).2 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, (kingStep_iff _ _ _ _).1 hj⟩)))))⟩
    | c4 hb hd hs => exact ⟨hb, (pieceAttacks_of p c _ s t hb).2 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨rfl, (pawnStep_iff c _ _ _ _).1 ⟨_, hd, hs⟩⟩)))))⟩
  · rintro ⟨hb, ha⟩
    rw [pieceAttacks_of p c k s t hb] at ha
    have hb' : p.board (s.1, s.2) = some ⟨c, k⟩ := hb
    rcases ha with h | h | h | ⟨rfl, h⟩ | ⟨rfl, h⟩ | ⟨rfl, h⟩
    · obtain ⟨d, hd, hr⟩ := (slider_line_iff p.board k s t).2 (Or.inl h)
      exact attackBy.c1 hb' hd ((slide_iff p d s t).2 hr)
    · obtain ⟨d, hd, hr⟩ := (slider_line_iff p.board k s t).2 (Or.inr (Or.inl h))
      exact attackBy.c1 hb' hd ((slide_iff p d s t).2 hr)
    · obtain ⟨d, hd, hr⟩ := (slider_line_iff p.board k s t).2 (Or.inr (Or.inr h))
      exact attackBy.c1 hb' hd ((slide_iff p d s t).2 hr)
    · exact attackBy.c2 hb' ((knightJump_iff _ _ _ _).2 h)
    · exact attackBy.c3 hb' ((kingStep_iff _ _ _ _).2 h)
    · obtain ⟨d, hd, hs⟩ := (pawnStep_iff c _ _ _ _).2 h
      exact attackBy.c4 hb' hd hs

theorem attack_iff (p : Position) (c : Color) (s t : Square) : attack p c s.1 s.2 t.1 t.2 ↔ pieceAttacks p c s t := by
  constructor
  · rintro ⟨h⟩
    exact ((attackBy_iff p _ c s t).1 h).2
  · intro h
    have h' := h
    simp only [pieceAttacks, pieceAttacksB] at h'
    cases hb : p.board s with
    | none => rw [hb] at h'; cases h'
    | some q =>
        rw [hb] at h'
        simp only [Bool.and_eq_true, beq_iff_eq] at h'
        obtain ⟨qc, qk⟩ := q
        simp only at h'
        obtain ⟨rfl, _⟩ := h'
        exact attack.c1 ((attackBy_iff p qk qc s t).2 ⟨hb, h⟩)

/-- THE GENERATED `attacked` IS THE KERNEL'S `Chess.isAttacked`, on every position. -/
theorem attacked_iff (p : Position) (c : Color) (t : Square) : attacked p c t.1 t.2 ↔ isAttacked p c t := by
  rw [isAttacked_iff]
  constructor
  · rintro ⟨h⟩
    exact ⟨_, (attack_iff p c (_, _) t).1 h⟩
  · rintro ⟨s, h⟩
    exact attacked.c1 (f1 := s.1) (r1 := s.2) ((attack_iff p c s t).2 h)

/-! ## The king -/

/-- One king of a side: its square. -/
theorem oneKing_sq {b : Board} {c : Color} (h : exactlyOneKing b c) : ∃ K, ∀ s, b s = some ⟨c, .king⟩ ↔ s = K := by
  obtain ⟨K, hK⟩ := Finset.card_eq_one.1 h
  refine ⟨K, fun s => ?_⟩
  have := Finset.ext_iff.1 hK s
  simp only [squaresOf, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton] at this
  exact this

theorem kingSquare_of_one {b : Board} {c : Color} {K : Square} (h : ∀ s, b s = some ⟨c, .king⟩ ↔ s = K) : kingSquare b c = some K :=
  Ideas.kingSquare_of_only h

/-- The kernel's check of a side with one king: its king's square is attacked by the other side. -/
theorem inCheck_one (q : Position) (c : Color) {K : Square} (h : ∀ s, q.board s = some ⟨c, .king⟩ ↔ s = K) :
    inCheck q c ↔ isAttacked q (Color.opposite c) K := by
  rw [inCheck_iff (kingSquare_of_one h)]

/-- THE GENERATED CHECK IS THE KERNEL'S: the side to move (one king) is in check exactly when the kernel says so. -/
theorem check_iff (p : Position) (hk : exactlyOneKing p.board p.turn) (c : Color) : check p c ↔ c = p.turn ∧ inCheck p p.turn := by
  obtain ⟨K, hK⟩ := oneKing_sq hk
  rw [inCheck_one p p.turn hK]
  constructor
  · rintro ⟨hc, hb, ho, ha⟩
    subst hc
    rw [opponent_iff] at ho
    subst ho
    have := (hK _).1 hb
    subst this
    exact ⟨rfl, (attacked_iff p _ _).1 ha⟩
  · rintro ⟨rfl, ha⟩
    exact check.c1 rfl ((hK K).2 rfl) ((opponent_iff _ _).2 rfl) ((attacked_iff p _ K).2 ha)

/-! ## THE MOVES ARE THE KERNEL'S PSEUDO-LEGAL MOVES -/

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

theorem attackByPawn_iff (p : Position) (c : Color) (f1 r1 f2 r2 : Fin 8) :
    attackByPawn p c f1 r1 f2 r2 ↔ p.board (f1, r1) = some ⟨c, .pawn⟩ ∧ pawnAttackGeometryB c (f1, r1) (f2, r2) = true := by
  constructor
  · intro h
    cases h with
    | c1 _ hl _ => cases hl
    | c2 hb hd hs => exact ⟨hb, (pawnStep_iff c _ _ _ _).1 ⟨_, hd, hs⟩⟩
  · rintro ⟨hb, hg⟩
    obtain ⟨d, hd, hs⟩ := (pawnStep_iff c _ _ _ _).2 hg
    exact attackByPawn.c2 hb hd hs

/-- THE KERNEL'S NORMAL MOVE, its promotion-rank condition aside: a unit of the side to move on `s`, a landing square for it, its geometry. -/
def NormalLike (p : Position) (s t : Square) : Prop :=
  ∃ q, p.board s = some q ∧ q.color = p.turn ∧ destinationFriendlyB p p.turn t = true ∧ normalPieceGeometryB p p.turn q s t = true

theorem normalPseudoLegal_iff (p : Position) (s t : Square) :
    normalPseudoLegal p s t ↔ NormalLike p s t ∧ ¬ (p.board s = some ⟨p.turn, .pawn⟩ ∧ t.2 = promotionRank p.turn) := by
  simp only [normalPseudoLegal, normalPseudoLegalB, NormalLike]
  cases hb : p.board s with
  | none => simp
  | some q =>
      obtain ⟨qc, qk⟩ := q
      simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq, Bool.or_eq_true, Option.some.injEq, Piece.mk.injEq, exists_eq_left']
      constructor
      · rintro ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩
        exact ⟨⟨h1, h2, h4⟩, fun ⟨⟨_, hk⟩, ht⟩ => by rcases h3 with h3 | h3 <;> simp_all⟩
      · rintro ⟨⟨h1, h2, h4⟩, h5⟩
        refine ⟨⟨⟨h1, h2⟩, ?_⟩, h4⟩
        by_cases hk : qk = .pawn
        · exact Or.inr (fun ht => h5 ⟨⟨h1, hk⟩, ht⟩)
        · exact Or.inl hk

/-- A pawn's push: one step forward along its file. -/
theorem push_iff (c : Color) (f r1 r2 : Fin 8) : (∃ d, forward c d ∧ step d f r1 f r2) ↔ pawnMoveGeometryB c (f, r1) (f, r2) = true := by
  simp only [pawnMoveGeometryB, Bool.and_eq_true, beq_iff_eq, true_and, forward_iff]
  constructor
  · rintro ⟨d, rfl, hs⟩
    cases c <;> simp only [step_iff, dx, dy] at hs <;> simp <;> omega
  · intro h
    cases c <;> simp at h
    · exact ⟨_, rfl, (step_iff _ _ _ _ _).2 ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩⟩
    · exact ⟨_, rfl, (step_iff _ _ _ _ _).2 ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩⟩

/-- The square between two squares two ranks apart on one file. -/
theorem clear_two (b : Board) (f : Fin 8) (r1 r3 : Fin 8) (h : r3.val = r1.val + 2 ∨ r1.val = r3.val + 2) :
    clearBetweenB b (f, r1) (f, r3) = true ↔ b (f, ⟨(r1.val + r3.val) / 2, by omega⟩) = none := by
  rw [clear_iff]
  obtain ⟨f, hf⟩ := f
  obtain ⟨r1, hr1⟩ := r1
  obtain ⟨r3, hr3⟩ := r3
  simp only at h
  constructor
  · intro hc
    apply hc
    simp only [between, betweenB, sameDiagonalB, sameFileB, sameRankB, betweenCoordB, ltB, Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq,
      beq_iff_eq, decide_eq_true_eq, Prod.mk.injEq, Fin.mk.injEq, absDiff_def, true_and, and_true]
    split_ifs <;> omega
  · intro hc u hu
    obtain ⟨⟨uf, huf⟩, ⟨ur, hur⟩⟩ := u
    simp only [between, betweenB, sameDiagonalB, sameFileB, sameRankB, betweenCoordB, ltB, Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq,
      beq_iff_eq, decide_eq_true_eq, Prod.mk.injEq, Fin.mk.injEq, absDiff_def, true_and, and_true] at hu
    have e1 : uf = f := by split_ifs at hu <;> omega
    have e2 : ur = (r1 + r3) / 2 := by split_ifs at hu <;> omega
    subst e1 e2
    exact hc

/-- A pawn's double push: two steps forward from its starting rank, the square passed over empty. -/
theorem double_iff (p : Position) (c : Color) (f r1 r3 : Fin 8) :
    (pawnStart c r1 ∧ ∃ d r2, forward c d ∧ step d f r1 f r2 ∧ empty p f r2 ∧ step d f r2 f r3) ↔
      (pawnDoubleGeometryB c (f, r1) (f, r3) = true ∧ clearBetweenB p.board (f, r1) (f, r3) = true) := by
  simp only [pawnDoubleGeometryB, Bool.and_eq_true, beq_iff_eq, true_and, forward_iff, pawnStart_iff, empty_iff]
  constructor
  · rintro ⟨hs, d, r2, rfl, h1, he, h2⟩
    cases c
    · simp only [step_iff, dx, dy] at h1 h2
      simp only at hs
      subst hs
      have e : r2 = ⟨(1 + r3.val) / 2, by omega⟩ := Fin.ext (by simp only [Fin.val_one] at h1 h2 ⊢; omega)
      rw [e] at he
      refine ⟨by simp; omega, ?_⟩
      rw [clear_two _ _ _ _ (by simp only [Fin.val_one] at h1 h2 ⊢; omega)]
      exact he
    · simp only [step_iff, dx, dy] at h1 h2
      simp only at hs
      subst hs
      have e : r2 = ⟨(6 + r3.val) / 2, by omega⟩ := Fin.ext (by simp at h1 h2 ⊢; omega)
      rw [e] at he
      refine ⟨by simp; omega, ?_⟩
      rw [clear_two _ _ _ _ (by simp at h1 h2 ⊢; omega)]
      exact he
  · rintro ⟨hg, hc⟩
    cases c <;> simp at hg
    · rw [clear_two _ _ _ _ (by omega)] at hc
      exact ⟨Fin.ext (by simp; omega), _, ⟨(r1.val + r3.val) / 2, by omega⟩, rfl, (step_iff _ _ _ _ _).2 ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩,
        hc, (step_iff _ _ _ _ _).2 ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩⟩
    · rw [clear_two _ _ _ _ (by omega)] at hc
      exact ⟨Fin.ext (by simp; omega), _, ⟨(r1.val + r3.val) / 2, by omega⟩, rfl, (step_iff _ _ _ _ _).2 ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩,
        hc, (step_iff _ _ _ _ _).2 ⟨by simp only [dx]; omega, by simp only [dy]; omega⟩⟩

/-- The kernel's geometry of a unit that is no pawn is its attack. -/
theorem normalGeometry_nonpawn (p : Position) (c : Color) (k : PieceKind) (hk : k ≠ .pawn) (s t : Square) (hb : p.board s = some ⟨c, k⟩) :
    normalPieceGeometryB p c ⟨c, k⟩ s t = true ↔ pieceAttacks p c s t := by
  rw [pieceAttacks_of p c k s t hb]
  cases k <;> simp_all [normalPieceGeometryB, queenGeometryB]

theorem destinationFriendly_empty (p : Position) (c : Color) (t : Square) (h : p.board t = none) : destinationFriendlyB p c t = true := by
  simp [destinationFriendlyB, h]

/-- THE GENERATED MOVE IS THE KERNEL'S PSEUDO-LEGAL MOVE, on every position: a unit of the side to move goes by the kernel's normal geometry onto a
square it may land on, or a pawn takes en passant. -/
theorem moveBy_iff' (p : Position) (k : PieceKind) (c : Color) (f1 r1 f2 r2 : Fin 8) :
    moveBy p k c f1 r1 f2 r2 ↔ (p.turn = c ∧ p.board (f1, r1) = some ⟨c, k⟩ ∧ destinationFriendlyB p c (f2, r2) = true ∧
      normalPieceGeometryB p c ⟨c, k⟩ (f1, r1) (f2, r2) = true) ∨ (k = .pawn ∧ enPassant p c f1 r1 f2 r2) := by
  constructor
  · intro h
    cases h with
    | c1 ht ha hk hl =>
        obtain ⟨hb, hpa⟩ := (attackBy_iff p _ c (_, _) (_, _)).1 ha
        exact Or.inl ⟨ht, hb, (mayLand_iff p c _ _).1 hl, (normalGeometry_nonpawn p c _ hk _ _ hb).2 hpa⟩
    | c2 ht hb hd hs he =>
        refine Or.inl ⟨ht, hb, destinationFriendly_empty p c _ ((empty_iff _ _ _).1 he), ?_⟩
        simp only [normalPieceGeometryB, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq]
        exact Or.inl (Or.inl ⟨(push_iff c _ _ _).1 ⟨_, hd, hs⟩, (empty_iff _ _ _).1 he⟩)
    | c3 ht hb hst hd hs1 he1 hs2 he2 =>
        refine Or.inl ⟨ht, hb, destinationFriendly_empty p c _ ((empty_iff _ _ _).1 he2), ?_⟩
        simp only [normalPieceGeometryB, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq]
        obtain ⟨h1, h2⟩ := (double_iff p c _ _ _).1 ⟨hst, _, _, hd, hs1, he1, hs2⟩
        exact Or.inr ⟨⟨h1, (empty_iff _ _ _).1 he2⟩, h2⟩
    | c4 ht ha hb ho hj =>
        obtain ⟨hs, hg⟩ := (attackByPawn_iff p c _ _ _ _).1 ha
        rw [opponent_iff] at ho
        subst ho
        refine Or.inl ⟨ht, hs, by cases c <;> simp [destinationFriendlyB, hb, hj], ?_⟩
        simp only [normalPieceGeometryB, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq]
        refine Or.inl (Or.inr ⟨hg, ?_⟩)
        simp [hb]
    | c5 he => exact Or.inr ⟨rfl, he⟩
  · rintro (⟨ht, hb, hd, hg⟩ | ⟨rfl, he⟩)
    · by_cases hk : k = .pawn
      · subst hk
        simp only [normalPieceGeometryB, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq] at hg
        rcases hg with (⟨hm, he⟩ | ⟨ha, hcap⟩) | ⟨⟨hdd, he⟩, hc2⟩
        · have hf : f1 = f2 := by simp only [pawnMoveGeometryB, Bool.and_eq_true, beq_iff_eq] at hm; exact hm.1
          subst hf
          obtain ⟨d, hd', hs⟩ := (push_iff c _ _ _).2 hm
          exact moveBy.c2 ht hb hd' hs ((empty_iff _ _ _).2 he)
        · cases hq : p.board (f2, r2) with
          | none => rw [hq] at hcap; cases hcap
          | some q =>
              rw [hq] at hcap
              simp only [destinationFriendlyB, hq] at hd
              obtain ⟨qc, qk⟩ := q
              simp only [beq_iff_eq, Bool.and_eq_true, bne_iff_ne, ne_eq] at hcap hd
              subst hcap
              exact moveBy.c4 ht ((attackByPawn_iff p c _ _ _ _).2 ⟨hb, ha⟩) hq ((opponent_iff _ _).2 rfl) hd.2
        · have hf : f1 = f2 := by simp only [pawnDoubleGeometryB, Bool.and_eq_true, beq_iff_eq] at hdd; exact hdd.1
          subst hf
          obtain ⟨hst, d, r', hd', hs1, he1, hs2⟩ := (double_iff p c _ _ _).2 ⟨hdd, hc2⟩
          exact moveBy.c3 ht hb hst hd' hs1 he1 hs2 ((empty_iff _ _ _).2 he)
      · exact moveBy.c1 ht ((attackBy_iff p k c (_, _) (_, _)).2 ⟨hb, (normalGeometry_nonpawn p c k hk _ _ hb).1 hg⟩) hk ((mayLand_iff p c _ _).2 hd)
    · exact moveBy.c5 he

theorem moveBy_iff (p : Position) (k : PieceKind) (c : Color) (s t : Square) :
    moveBy p k c s.1 s.2 t.1 t.2 ↔ (p.turn = c ∧ p.board s = some ⟨c, k⟩ ∧ destinationFriendlyB p c t = true ∧ normalPieceGeometryB p c ⟨c, k⟩ s t = true) ∨
      (k = .pawn ∧ enPassant p c s.1 s.2 t.1 t.2) := moveBy_iff' p k c s.1 s.2 t.1 t.2

/-! ## En passant -/

/-- THE E.P. SQUARE WHERE THE RULE PUTS IT: on the rank the side to move's pawn captures onto (sixth for White, third for Black). Every reachable position
has it (`reachable_epPlaced`); the kernel's capture square (`Chess.epCapturedSquare`) exists exactly there, the spec's `ep_victim` reads no rank. -/
def EpPlaced (p : Position) : Prop :=
  ∀ e : Square, p.enPassant = some e → e.2.val = (match p.turn with | .white => 5 | .black => 2)

theorem epPlaced_initial : EpPlaced initialPosition := by
  intro e h
  simp [initialPosition] at h

/-- A move leaves the e.p. square where the rule puts it (the kernel's `newEnPassant`: the square a double push passed over, for the other side). -/
theorem epPlaced_applyMove (p : Position) (m : Move) : EpPlaced (applyMove p m) := by
  intro e h
  simp only [applyMove, newEnPassant, pawnDoubleTarget] at h
  cases m with
  | normal s t =>
      simp only at h
      split_ifs at h with h1 h2
      · simp only [Option.some.injEq] at h
        subst h
        simp [h2]
      · simp only [Option.some.injEq] at h
        subst h
        cases hc : p.turn
        · exact absurd hc h2
        · simp [hc]
  | _ => cases h

theorem reachable_epPlaced (p : Position) (h : reachable p) : EpPlaced p := by
  unfold reachable at h
  induction h with
  | refl => exact epPlaced_initial
  | tail _ hstep _ =>
      obtain ⟨m, _, rfl⟩ := hstep
      exact epPlaced_applyMove _ m

/-- The kernel's en passant is the spec's (the spec's reading needs the e.p. square where the rule puts it). -/
theorem enPassant_of_kernel (p : Position) (s t : Square) (h : pseudoLegal p (.enPassant s t)) : enPassant p p.turn s.1 s.2 t.1 t.2 := by
  obtain ⟨he, hb, hg, ht, cs, hcs⟩ := Ideas.NextEquiv.ep_pl h
  have h' := h
  simp only [pseudoLegal, pseudoLegalB, pseudoLegalEnPassantB, he, hb, beq_self_eq_true, Bool.true_and, hg, ht, hcs] at h'
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h'
  have hcsb : p.board cs = some ⟨Color.opposite p.turn, .pawn⟩ := by simpa using h'
  exact enPassant.c1 (f3 := cs.1) (r3 := cs.2) rfl ((attackByPawn_iff p _ _ _ _ _).2 ⟨hb, hg⟩) he ((empty_iff _ _ _).2 ht)
    ((Ideas.NextEquiv.epVictim_cs hcs cs.1 cs.2).2 rfl) hcsb ((opponent_iff _ _).2 rfl)

theorem kernel_of_enPassant (p : Position) (hep : EpPlaced p) (c : Color) (f1 r1 f2 r2 : Fin 8) (h : enPassant p c f1 r1 f2 r2) :
    c = p.turn ∧ pseudoLegal p (.enPassant (f1, r1) (f2, r2)) := by
  cases h with | @c1 f3 r3 o hc ha he hempty hv hvb ho => ?_
  subst hc
  obtain ⟨hb, hg⟩ := (attackByPawn_iff p _ _ _ _ _).1 ha
  rw [opponent_iff] at ho
  subst ho
  have hr := hep _ he
  rw [epVictim_iff] at hv
  refine ⟨rfl, ?_⟩
  simp only [pseudoLegal, pseudoLegalB, pseudoLegalEnPassantB, he, hb, beq_self_eq_true, Bool.true_and, hg, (empty_iff _ _ _).1 hempty]
  obtain ⟨hf3, hv2⟩ := hv
  rw [hf3] at hvb
  cases hturn : p.turn <;> rw [hturn] at hr hv2 hvb <;> simp only at hr hv2
  · have e : epCapturedSquare Color.white (f2, r2) = some (f2, ⟨4, by omega⟩) := by simp [epCapturedSquare, hr]
    rw [e]
    have : (f2, (⟨4, by omega⟩ : Fin 8)) = (f2, r3) := by simp only [Prod.mk.injEq, true_and]; exact Fin.ext (by simp; omega)
    simp only [this]
    simpa using hvb
  · have e : epCapturedSquare Color.black (f2, r2) = some (f2, ⟨3, by omega⟩) := by simp [epCapturedSquare, hr]
    rw [e]
    have : (f2, (⟨3, by omega⟩ : Fin 8)) = (f2, r3) := by simp only [Prod.mk.injEq, true_and]; exact Fin.ext (by simp; omega)
    simp only [this]
    simpa using hvb

/-! ## Castling -/

theorem castlingEmpty_iff (w : Wing) (f : Fin 8) : castlingEmpty w f ↔ (w = .kingside ∧ (f = 5 ∨ f = 6)) ∨ (w = .queenside ∧ (f = 1 ∨ f = 2 ∨ f = 3)) := by
  constructor
  · intro h; cases h <;> simp
  · rintro (⟨rfl, rfl | rfl⟩ | ⟨rfl, rfl | rfl | rfl⟩) <;> constructor

theorem castlingSafe_iff (w : Wing) (f : Fin 8) : castlingSafe w f ↔ (w = .kingside ∧ (f = 4 ∨ f = 5 ∨ f = 6)) ∨ (w = .queenside ∧ (f = 4 ∨ f = 3 ∨ f = 2)) := by
  constructor
  · intro h; cases h <;> simp
  · rintro (⟨rfl, rfl | rfl | rfl⟩ | ⟨rfl, rfl | rfl | rfl⟩) <;> constructor

/-- What castling asks, in the kernel's words: the side to move, its right kept, its king and that rook home, the squares between empty, the king's three
squares not attacked. -/
def CastleOK (p : Position) (c : Color) (w : Wing) : Prop :=
  p.turn = c ∧ right c w ∈ p.castling ∧ p.board (4, home c) = some ⟨c, .king⟩ ∧ p.board (rookFrom w, home c) = some ⟨c, .rook⟩ ∧
    (∀ f, castlingEmpty w f → p.board (f, home c) = none) ∧ (∀ f, castlingSafe w f → ¬ isAttacked p (Color.opposite c) (f, home c))

theorem castle_spec_iff (p : Position) (c : Color) (w : Wing) : castle p c w ↔ CastleOK p c w := by
  constructor
  · rintro ⟨ht, hr, hh, hk, hp, hrook, hbl, hun⟩
    rw [homeRank_iff] at hh
    rw [castlingPath_iff] at hp
    obtain ⟨rfl, rfl, rfl⟩ := hp
    subst hh
    refine ⟨ht, hr, hk, hrook, fun f hf => ?_, fun f hf ha => ?_⟩
    · by_contra hn
      exact hbl (castleBlocked.c1 ((homeRank_iff _ _).2 rfl) hf ((occupied_iff _ _ _).2 hn))
    · exact hun (castleUnsafe.c1 ((homeRank_iff _ _).2 rfl) hf ((opponent_iff _ _).2 rfl) ((attacked_iff p _ _).2 ha))
  · rintro ⟨ht, hr, hk, hrook, he, hs⟩
    refine castle.c1 ht hr ((homeRank_iff _ _).2 rfl) hk ((castlingPath_iff w _ _ _).2 ⟨rfl, rfl, rfl⟩) hrook ?_ ?_
    · rintro ⟨hh, hf, ho⟩
      rw [homeRank_iff] at hh
      subst hh
      exact (occupied_iff _ _ _).1 ho (he _ hf)
    · rintro ⟨hh, hf, ho, ha⟩
      rw [homeRank_iff] at hh
      rw [opponent_iff] at ho
      subst hh ho
      exact hs _ hf ((attacked_iff p _ _).1 ha)

theorem kernel_castle_iff (p : Position) (c : Color) (w : Wing) :
    (pseudoLegal p (castleMove c w) ∧ castleSafeSquares p (castleMove c w)) ↔ CastleOK p c w := by
  unfold CastleOK
  simp only [castlingEmpty_iff, castlingSafe_iff]
  cases c <;> cases w <;>
    simp [castleMove, pseudoLegal, pseudoLegalB, pseudoLegalCastleB, castleColorOf, castleRightOf, castleRookSource, castleClearB,
      castleEmptySquares, castleSafeSquares, castleSafeSquaresB, castleTransitSquare, Move.source, Move.destination, home, rookFrom,
      Ideas.Spec.Binding.homeRank, isAttacked, or_imp, forall_and] <;> tauto

theorem castle_iff (p : Position) (c : Color) (w : Wing) : castle p c w ↔ pseudoLegal p (castleMove c w) ∧ castleSafeSquares p (castleMove c w) := by
  rw [castle_spec_iff, kernel_castle_iff]

theorem castlingMove_iff (p : Position) (c : Color) (w : Wing) (f1 r1 f2 r2 : Fin 8) :
    castlingMove p c w f1 r1 f2 r2 ↔ castle p c w ∧ f1 = 4 ∧ r1 = home c ∧ f2 = kingTo w ∧ r2 = home c := by
  constructor
  · rintro ⟨hc, hh, hp⟩
    rw [homeRank_iff] at hh
    rw [castlingPath_iff] at hp
    exact ⟨hc, rfl, hh, hp.1, hh⟩
  · rintro ⟨hc, rfl, rfl, rfl, rfl⟩
    exact castlingMove.c1 hc ((homeRank_iff _ _).2 rfl) ((castlingPath_iff w _ _ _).2 ⟨rfl, rfl, rfl⟩)

theorem castlingMoveE_iff (p : Position) (c : Color) (w : Wing) (r kt r' : Fin 8) :
    castlingMoveE p c w r kt r' ↔ castle p c w ∧ r = home c ∧ kt = kingTo w ∧ r' = home c := by
  constructor
  · rintro ⟨hc, hh, hp⟩
    rw [homeRank_iff] at hh
    rw [castlingPath_iff] at hp
    exact ⟨hc, hh, hp.1, hh⟩
  · rintro ⟨hc, rfl, rfl, rfl⟩
    exact castlingMoveE.c1 hc ((homeRank_iff _ _).2 rfl) ((castlingPath_iff w _ _ _).2 ⟨rfl, rfl, rfl⟩)

/-! ## The candidates -/

theorem enPassant_turn {p : Position} {c : Color} {f1 r1 f2 r2 : Fin 8} (h : enPassant p c f1 r1 f2 r2) : c = p.turn := by
  cases h with | c1 hc => exact hc.symm

theorem enPassant_pawn {p : Position} {c : Color} {f1 r1 f2 r2 : Fin 8} (h : enPassant p c f1 r1 f2 r2) : p.board (f1, r1) = some ⟨c, .pawn⟩ := by
  cases h with | c1 _ ha => exact ((attackByPawn_iff p _ _ _ _ _).1 ha).1

theorem castle_turn {p : Position} {c : Color} {w : Wing} (h : castle p c w) : c = p.turn := ((castle_spec_iff p c w).1 h).1.symm

/-- THE SPEC'S CANDIDATE MOVES, in the kernel's words: the side to move's normal move (its promotion-rank condition aside), its en passant, or its castling. -/
theorem candidate_iff (p : Position) (c : Color) (f1 r1 f2 r2 : Fin 8) :
    candidate p c f1 r1 f2 r2 ↔ c = p.turn ∧ (NormalLike p (f1, r1) (f2, r2) ∨ enPassant p c f1 r1 f2 r2 ∨
      ∃ w, castle p c w ∧ f1 = 4 ∧ r1 = home c ∧ f2 = kingTo w ∧ r2 = home c) := by
  constructor
  · rintro (⟨hm⟩ | ⟨hm⟩)
    · rcases (moveBy_iff' p _ c _ _ _ _).1 hm with ⟨ht, hb, hd, hg⟩ | ⟨_, he⟩
      · subst ht
        exact ⟨rfl, Or.inl ⟨_, hb, rfl, hd, hg⟩⟩
      · exact ⟨enPassant_turn he, Or.inr (Or.inl he)⟩
    · have := (castlingMove_iff p c _ _ _ _ _).1 hm
      exact ⟨castle_turn this.1, Or.inr (Or.inr ⟨_, this⟩)⟩
  · rintro ⟨rfl, ⟨⟨qc, qk⟩, hb, hqc, hd, hg⟩ | he | ⟨w, hw⟩⟩
    · simp only at hqc
      subst hqc
      exact candidate.c1 ((moveBy_iff' p qk _ _ _ _ _).2 (Or.inl ⟨rfl, hb, hd, hg⟩))
    · exact candidate.c1 (k := .pawn) ((moveBy_iff' p _ _ _ _ _ _).2 (Or.inr ⟨rfl, he⟩))
    · exact candidate.c2 ((castlingMove_iff p _ w _ _ _ _).2 hw)

/-- A candidate's unit belongs to the side to move. -/
theorem candidate_unit (p : Position) (c : Color) (f1 r1 f2 r2 : Fin 8) (h : candidate p c f1 r1 f2 r2) :
    c = p.turn ∧ ∃ k, p.board (f1, r1) = some ⟨c, k⟩ := by
  obtain ⟨rfl, hn | he | ⟨w, hc, rfl, rfl, _⟩⟩ := (candidate_iff p c _ _ _ _).1 h
  · obtain ⟨⟨qc, qk⟩, hb, hqc, _⟩ := hn
    simp only at hqc
    subst hqc
    exact ⟨rfl, qk, hb⟩
  · exact ⟨rfl, .pawn, enPassant_pawn he⟩
  · exact ⟨rfl, .king, ((castle_spec_iff p _ w).1 hc).2.2.1⟩

/-! ## THE BOARD AFTER A CANDIDATE is the spec's next board after its move term -/

theorem vacates_iff (p : Position) (f1 r1 f2 r2 fx rx : Fin 8) :
    vacates p f1 r1 f2 r2 fx rx ↔ ((∃ c, candidate p c f1 r1 f2 r2) ∧ fx = f1 ∧ rx = r1) ∨ (∃ c, enPassant p c f1 r1 f2 r2 ∧ epVictim c f2 r2 fx rx) ∨
      (∃ c w, castle p c w ∧ f1 = 4 ∧ r1 = home c ∧ f2 = kingTo w ∧ r2 = home c ∧ fx = rookFrom w ∧ rx = home c) := by
  constructor
  · intro h
    cases h with
    | c1 hc => exact Or.inl ⟨⟨_, hc⟩, rfl, rfl⟩
    | c2 he hv => exact Or.inr (Or.inl ⟨_, he, hv⟩)
    | c3 hm hp =>
        obtain ⟨hc, hr, hkt, _⟩ := (castlingMoveE_iff p _ _ _ _ _).1 hm
        rw [castlingPath_iff] at hp
        exact Or.inr (Or.inr ⟨_, _, hc, rfl, hr, hkt, hr, hp.2.2, hr⟩)
  · rintro (⟨⟨c, hc⟩, rfl, rfl⟩ | ⟨c, he, hv⟩ | ⟨c, w, hc, rfl, rfl, rfl, rfl, rfl, rfl⟩)
    · exact vacates.c1 hc
    · exact vacates.c2 he hv
    · exact vacates.c3 ((castlingMoveE_iff p c w _ _ _).2 ⟨hc, rfl, rfl, rfl⟩) ((castlingPath_iff w _ _ _).2 ⟨rfl, rfl, rfl⟩)

theorem fills_iff (p : Position) (f1 r1 f2 r2 fx rx : Fin 8) (c : Color) (k : PieceKind) :
    fills p f1 r1 f2 r2 fx rx c k ↔ (candidate p c f1 r1 f2 r2 ∧ p.board (f1, r1) = some ⟨c, k⟩ ∧ fx = f2 ∧ rx = r2) ∨
      (∃ w, castle p c w ∧ f1 = 4 ∧ r1 = home c ∧ f2 = kingTo w ∧ r2 = home c ∧ fx = rookTo w ∧ rx = home c ∧ k = .rook) := by
  constructor
  · intro h
    cases h with
    | c1 hc hb => exact Or.inl ⟨hc, hb, rfl, rfl⟩
    | c2 hm hp =>
        obtain ⟨hc, hr, hkt, _⟩ := (castlingMoveE_iff p _ _ _ _ _).1 hm
        rw [castlingPath_iff] at hp
        exact Or.inr ⟨_, hc, rfl, hr, hkt, hr, hp.2.1, hr, rfl⟩
  · rintro (⟨hc, hb, rfl, rfl⟩ | ⟨w, hc, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩)
    · exact fills.c1 hc hb
    · exact fills.c2 ((castlingMoveE_iff p c w _ _ _).2 ⟨hc, rfl, rfl, rfl⟩) ((castlingPath_iff w _ _ _).2 ⟨rfl, rfl, rfl⟩)

theorem filled_iff (p : Position) (f1 r1 f2 r2 fx rx : Fin 8) : filled p f1 r1 f2 r2 fx rx ↔ ∃ c k, fills p f1 r1 f2 r2 fx rx c k :=
  ⟨fun ⟨h⟩ => ⟨_, _, h⟩, fun ⟨_, _, h⟩ => filled.c1 h⟩

theorem stays_iff (p : Position) (f1 r1 f2 r2 fx rx : Fin 8) (c : Color) (k : PieceKind) :
    stays p f1 r1 f2 r2 fx rx c k ↔ (∃ m, candidate p m f1 r1 f2 r2) ∧ p.board (fx, rx) = some ⟨c, k⟩ ∧ ¬ vacates p f1 r1 f2 r2 fx rx ∧
      ¬ filled p f1 r1 f2 r2 fx rx :=
  ⟨fun ⟨h1, h2, h3, h4⟩ => ⟨⟨_, h1⟩, h2, h3, h4⟩, fun ⟨⟨_, h1⟩, h2, h3, h4⟩ => stays.c1 h1 h2 h3 h4⟩

theorem after_def (p : Position) (f1 r1 f2 r2 fx rx : Fin 8) (c : Color) (k : PieceKind) :
    after p f1 r1 f2 r2 fx rx c k ↔ fills p f1 r1 f2 r2 fx rx c k ∨ stays p f1 r1 f2 r2 fx rx c k := by
  constructor
  · intro h
    cases h with
    | c1 h => exact Or.inl h
    | c2 h => exact Or.inr h
  · rintro (h | h)
    · exact after.c1 h
    · exact after.c2 h

/-- (B1) A candidate onto the castling's squares with the king there IS the castling. -/
theorem castled_cand (p : Position) {c0 : Color} {f1 r1 f2 r2 : Fin 8} (hc : candidate p c0 f1 r1 f2 r2) (c : Color) (w : Wing) :
    castled p (.move (f1, r1) (f2, r2)) c w ↔ castle p c w ∧ f1 = 4 ∧ r1 = home c ∧ f2 = kingTo w ∧ r2 = home c := by
  rw [Ideas.NextEquiv.castled_iff]
  constructor
  · rintro ⟨hct, hm, hk⟩
    simp only [MoveTerm.move.injEq, Prod.mk.injEq] at hm
    obtain ⟨⟨rfl, rfl⟩, rfl, rfl⟩ := hm
    subst hct
    refine ⟨?_, rfl, rfl, rfl, rfl⟩
    obtain ⟨hc0, hn | he | ⟨w', hcw, -, -, hkt, -⟩⟩ := (candidate_iff p c0 _ _ _ _).1 hc
    · exfalso
      obtain ⟨q, hq, -, -, hg⟩ := hn
      rw [hk] at hq
      cases hq
      have := Ideas.NextEquiv.normal_king_step rfl hg
      rw [Ideas.NextEquiv.absDiff_four_kingTo] at this
      omega
    · exfalso
      have := enPassant_pawn he
      rw [hk] at this
      cases this
    · subst hc0
      have e : w = w' := Ideas.NextEquiv.kingTo_inj hkt
      subst e
      exact hcw
  · rintro ⟨hcw, rfl, rfl, rfl, rfl⟩
    have hok := (castle_spec_iff p c w).1 hcw
    exact ⟨hok.1.symm, rfl, hok.2.2.1⟩

/-- (B2) A pawn's sideways move onto the empty e.p. square, as a candidate, IS the en passant. -/
theorem passant_cand (p : Position) {c0 : Color} {f1 r1 f2 r2 : Fin 8} (hc : candidate p c0 f1 r1 f2 r2) (fx rx : Fin 8) :
    passantTaken p (.move (f1, r1) (f2, r2)) fx rx ↔ ∃ c, enPassant p c f1 r1 f2 r2 ∧ epVictim c f2 r2 fx rx := by
  rw [Ideas.NextEquiv.passantTaken_iff]
  constructor
  · rintro ⟨s, t, hm, hb, hep, ht, hne, hv⟩
    simp only [MoveTerm.move.injEq] at hm
    obtain ⟨rfl, rfl⟩ := hm
    obtain ⟨hc0, hn | he | ⟨w, hcw, rfl, rfl, _⟩⟩ := (candidate_iff p c0 _ _ _ _).1 hc
    · exfalso
      obtain ⟨q, hq, _, _, hg⟩ := hn
      rw [hb] at hq
      cases hq
      exact hne (Ideas.NextEquiv.normal_pawn_file rfl hg ht)
    · subst hc0
      exact ⟨_, he, hv⟩
    · exfalso
      have := ((castle_spec_iff p _ w).1 hcw).2.2.1
      rw [hb, ← hc0] at this
      cases this
  · rintro ⟨c, he, hv⟩
    have hct := enPassant_turn he
    subst hct
    cases he with
    | c1 _ ha hep hempty =>
        obtain ⟨hb, hg⟩ := (attackByPawn_iff p _ _ _ _ _).1 ha
        exact ⟨_, _, rfl, hb, hep, (empty_iff _ _ _).1 hempty, Ideas.NextEquiv.pawnAttack_file_ne hg, hv⟩

theorem not_promote_move {s t s' t' : Square} {k : PieceKind} : (MoveTerm.move s t) ≠ MoveTerm.promote s' t' k := by
  intro h
  cases h

theorem lifted_cand (p : Position) {c0 : Color} {f1 r1 f2 r2 : Fin 8} (hc : candidate p c0 f1 r1 f2 r2) (fx rx : Fin 8) :
    lifted p (.move (f1, r1) (f2, r2)) fx rx ↔ vacates p f1 r1 f2 r2 fx rx := by
  rw [Ideas.NextEquiv.lifted_iff, vacates_iff, passant_cand p hc]
  constructor
  · rintro (⟨t, hm⟩ | ⟨t, k, hm⟩ | ⟨c, w, hcw, hf, hr⟩ | hp)
    · simp only [MoveTerm.move.injEq, Prod.mk.injEq] at hm
      exact Or.inl ⟨⟨_, hc⟩, hm.1.1.symm, hm.1.2.symm⟩
    · exact absurd hm not_promote_move
    · obtain ⟨hcs, h1, h2, h3, h4⟩ := (castled_cand p hc c w).1 hcw
      exact Or.inr (Or.inr ⟨c, w, hcs, h1, h2, h3, h4, hf, hr⟩)
    · exact Or.inr (Or.inl hp)
  · rintro (⟨_, rfl, rfl⟩ | hp | ⟨c, w, hcs, h1, h2, h3, h4, hf, hr⟩)
    · exact Or.inl ⟨_, rfl⟩
    · exact Or.inr (Or.inr (Or.inr hp))
    · exact Or.inr (Or.inr (Or.inl ⟨c, w, (castled_cand p hc c w).2 ⟨hcs, h1, h2, h3, h4⟩, hf, hr⟩))

theorem landed_cand (p : Position) {c0 : Color} {f1 r1 f2 r2 : Fin 8} (hc : candidate p c0 f1 r1 f2 r2) (fx rx : Fin 8) :
    landed p (.move (f1, r1) (f2, r2)) fx rx ↔ filled p f1 r1 f2 r2 fx rx := by
  rw [Ideas.NextEquiv.landed_iff, filled_iff]
  obtain ⟨hc0, k0, hb0⟩ := candidate_unit p _ _ _ _ _ hc
  constructor
  · rintro (⟨s, hm⟩ | ⟨s, k, hm⟩ | ⟨c, w, hcw, hf, hr⟩)
    · simp only [MoveTerm.move.injEq, Prod.mk.injEq] at hm
      exact ⟨_, _, (fills_iff p _ _ _ _ _ _ _ _).2 (Or.inl ⟨hc, hb0, hm.2.1.symm, hm.2.2.symm⟩)⟩
    · exact absurd hm not_promote_move
    · obtain ⟨hcs, h1, h2, h3, h4⟩ := (castled_cand p hc c w).1 hcw
      exact ⟨c, .rook, (fills_iff p _ _ _ _ _ _ _ _).2 (Or.inr ⟨w, hcs, h1, h2, h3, h4, hf, hr, rfl⟩)⟩
  · rintro ⟨c, k, hfl⟩
    rcases (fills_iff p _ _ _ _ _ _ _ _).1 hfl with ⟨_, _, rfl, rfl⟩ | ⟨w, hcs, h1, h2, h3, h4, hf, hr, _⟩
    · exact Or.inl ⟨_, rfl⟩
    · exact Or.inr (Or.inr ⟨c, w, (castled_cand p hc c w).2 ⟨hcs, h1, h2, h3, h4⟩, hf, hr⟩)

/-- THE BOARD AFTER A CANDIDATE (legality's counterfactual board, the spec's `after`) IS THE SPEC'S NEXT BOARD after the candidate's move term. -/
theorem after_iff_nextCell (p : Position) {c0 : Color} {f1 r1 f2 r2 : Fin 8} (hc : candidate p c0 f1 r1 f2 r2) (fx rx : Fin 8) (c : Color) (k : PieceKind) :
    after p f1 r1 f2 r2 fx rx c k ↔ nextCell p (.move (f1, r1) (f2, r2)) fx rx c k := by
  rw [after_def, Ideas.NextEquiv.nextCell_iff, stays_iff, ← lifted_cand p hc, ← landed_cand p hc, fills_iff]
  obtain ⟨hc0, k0, hb0⟩ := candidate_unit p _ _ _ _ _ hc
  constructor
  · rintro ((⟨hc', hb, rfl, rfl⟩ | ⟨w, hcs, h1, h2, h3, h4, hf, hr, hk⟩) | ⟨_, hb, hl, hd⟩)
    · exact Or.inl ⟨_, rfl, (candidate_unit p _ _ _ _ _ hc').1, hb⟩
    · exact Or.inr (Or.inr (Or.inl ⟨w, (castled_cand p hc c w).2 ⟨hcs, h1, h2, h3, h4⟩, hf, hr, hk⟩))
    · exact Or.inr (Or.inr (Or.inr ⟨hb, hl, hd⟩))
  · rintro (⟨s, hm, hct, hb⟩ | ⟨s, hm, _⟩ | ⟨w, hcw, hf, hr, hk⟩ | ⟨hb, hl, hd⟩)
    · simp only [MoveTerm.move.injEq, Prod.mk.injEq] at hm
      obtain ⟨rfl, hx1, hx2⟩ := hm
      subst hx1 hx2
      subst hct
      have : c0 = p.turn := hc0
      subst this
      exact Or.inl (Or.inl ⟨hc, hb, rfl, rfl⟩)
    · exact absurd hm not_promote_move
    · obtain ⟨hcs, h1, h2, h3, h4⟩ := (castled_cand p hc c w).1 hcw
      exact Or.inl (Or.inr ⟨w, hcs, h1, h2, h3, h4, hf, hr, hk⟩)
    · exact Or.inr ⟨⟨_, hc⟩, hb, hl, hd⟩

/-! ## EXPOSURE IS THE KERNEL'S CHECK AFTER THE MOVE -/

theorem kingAfter_iff (p : Position) (c : Color) (f1 r1 f2 r2 kf kr : Fin 8) :
    kingAfter p c f1 r1 f2 r2 kf kr ↔ candidate p c f1 r1 f2 r2 ∧ after p f1 r1 f2 r2 kf kr c .king :=
  ⟨fun ⟨h1, h2⟩ => ⟨h1, h2⟩, fun ⟨h1, h2⟩ => kingAfter.c1 h1 h2⟩

theorem afterOccupied_iff (p : Position) (f1 r1 f2 r2 f r : Fin 8) : afterOccupied p f1 r1 f2 r2 f r ↔ ∃ c k, after p f1 r1 f2 r2 f r c k :=
  ⟨fun ⟨h⟩ => ⟨_, _, h⟩, fun ⟨_, _, h⟩ => afterOccupied.c1 h⟩

/-- THE KING'S RAY AFTER THE MOVE is a ray over the board after the move, from the mover's king. -/
theorem kingRay_iff (p : Position) (f1 r1 f2 r2 : Fin 8) (d : Dir) (f r : Fin 8) :
    kingRay p f1 r1 f2 r2 d f r ↔ ∃ c kf kr, kingAfter p c f1 r1 f2 r2 kf kr ∧ Ray (fun y => afterOccupied p f1 r1 f2 r2 y.1 y.2) d (kf, kr) (f, r) := by
  constructor
  · intro h
    induction h with
    | @c1 f r c kf kr hk hs =>
      refine ⟨c, kf, kr, hk, 1, le_refl _, at_one.2 hs, ?_⟩
      intro k x h1 h2
      omega
    | @c2 f3 r3 f r _ hno hs ih =>
      obtain ⟨c, kf, kr, hk, n, hn, hat, hopen⟩ := ih
      refine ⟨c, kf, kr, hk, n + 1, by omega, at_succ (x := (f, r)) hat hs, ?_⟩
      intro k x hk1 hk2 hkk
      by_cases hkn : k < n
      · exact hopen k x hk1 hkn hkk
      · have hk' : k = n := by omega
        subst hk'
        rw [at_unique hkk hat]
        exact hno
  · rintro ⟨c, kf, kr, hk, n, hn, hat, hopen⟩
    induction n generalizing f r with
    | zero => omega
    | succ m ih =>
      rcases Nat.eq_zero_or_pos m with hm | hm
      · subst hm
        exact kingRay.c1 hk (at_one.1 hat)
      · obtain ⟨x, hx⟩ := exists_at d (kf, kr) (f, r) m hat
        have h1 := ih x.1 x.2 hm hx (fun k y a b c => hopen k y a (by omega) c)
        exact kingRay.c2 (f := x.1) (r := x.2) h1 (hopen m x hm (by omega) hx) (at_pred hx hat)

/-- THE EXPOSURE OF A KING on a board given as a relation (a unit of colour and kind on a square): an enemy unit bears on the king's square -- a slider
along an open ray of its kind's lines, a knight's jump, a king's step, a pawn's diagonal step. -/
def ExposedR (A : Square → Color → PieceKind → Prop) (c : Color) (K : Square) : Prop :=
  (∃ d y k, lineOf k d ∧ Ray (fun z => ∃ c k, A z c k) d K y ∧ A y (Color.opposite c) k) ∨
    (∃ y, knightJump K.1 K.2 y.1 y.2 ∧ A y (Color.opposite c) .knight) ∨
    (∃ y, kingStep K.1 K.2 y.1 y.2 ∧ A y (Color.opposite c) .king) ∨
    (∃ d y, pawnDiagonal (Color.opposite c) d ∧ step d y.1 y.2 K.1 K.2 ∧ A y (Color.opposite c) .pawn)

theorem exposedR_congr {A B : Square → Color → PieceKind → Prop} {c : Color} {K : Square} (hocc : ∀ y, (∃ c k, A y c k) ↔ (∃ c k, B y c k))
    (hopp : ∀ y k, A y (Color.opposite c) k ↔ B y (Color.opposite c) k) : ExposedR A c K ↔ ExposedR B c K := by
  unfold ExposedR
  simp only [ray_congr hocc, hopp]

/-- The spec's exposure, read through a unique king after the move. -/
theorem exposed_iff (p : Position) (f1 r1 f2 r2 : Fin 8)
    (hU : ∀ (c : Color) (K : Square) (c' : Color) (K' : Square), kingAfter p c f1 r1 f2 r2 K.1 K.2 → kingAfter p c' f1 r1 f2 r2 K'.1 K'.2 → c = c' ∧ K = K') :
    exposed p f1 r1 f2 r2 ↔ ∃ c K, kingAfter p c f1 r1 f2 r2 K.1 K.2 ∧ ExposedR (fun y c k => after p f1 r1 f2 r2 y.1 y.2 c k) c K := by
  have hocc : ∀ y : Square, afterOccupied p f1 r1 f2 r2 y.1 y.2 ↔ ∃ c k, after p f1 r1 f2 r2 y.1 y.2 c k := fun y => afterOccupied_iff p _ _ _ _ _ _
  constructor
  · intro h
    cases h with
    | @c1 c kf kr d f r o k hk hr ha ho hl =>
        rw [opponent_iff] at ho
        subst ho
        obtain ⟨c', kf', kr', hk', hray⟩ := (kingRay_iff p _ _ _ _ _ _ _).1 hr
        obtain ⟨rfl, e⟩ := hU c (kf, kr) c' (kf', kr') hk hk'
        simp only [Prod.mk.injEq] at e
        obtain ⟨rfl, rfl⟩ := e
        exact ⟨c, (kf, kr), hk, Or.inl ⟨d, (f, r), k, hl, (ray_congr hocc d _ _).1 hray, ha⟩⟩
    | @c2 c kf kr f r o hk hj ha ho =>
        rw [opponent_iff] at ho
        subst ho
        exact ⟨c, (kf, kr), hk, Or.inr (Or.inl ⟨(f, r), hj, ha⟩)⟩
    | @c3 c kf kr f r o hk hj ha ho =>
        rw [opponent_iff] at ho
        subst ho
        exact ⟨c, (kf, kr), hk, Or.inr (Or.inr (Or.inl ⟨(f, r), hj, ha⟩))⟩
    | @c4 c kf kr o d f r hk ho hd hs ha =>
        rw [opponent_iff] at ho
        subst ho
        exact ⟨c, (kf, kr), hk, Or.inr (Or.inr (Or.inr ⟨d, (f, r), hd, hs, ha⟩))⟩
  · rintro ⟨c, K, hk, ⟨d, y, k, hl, hray, ha⟩ | ⟨y, hj, ha⟩ | ⟨y, hj, ha⟩ | ⟨d, y, hd, hs, ha⟩⟩
    · exact exposed.c1 hk ((kingRay_iff p _ _ _ _ _ _ _).2 ⟨c, K.1, K.2, hk, (ray_congr hocc d _ _).2 hray⟩) ha ((opponent_iff _ _).2 rfl) hl
    · exact exposed.c2 hk hj ha ((opponent_iff _ _).2 rfl)
    · exact exposed.c3 hk hj ha ((opponent_iff _ _).2 rfl)
    · exact exposed.c4 hk ((opponent_iff _ _).2 rfl) hd hs ha

theorem lineOf_opp {k : PieceKind} {d : Dir} (h : lineOf k d) : lineOf k (opp d) := by
  rw [lineOf_iff] at h ⊢
  rcases h with ⟨rfl, h⟩ | ⟨rfl, h⟩ | ⟨rfl, h | h⟩
  · exact Or.inl ⟨rfl, orthogonal_opp h⟩
  · exact Or.inr (Or.inl ⟨rfl, diagonal_opp h⟩)
  · exact Or.inr (Or.inr ⟨rfl, Or.inl (orthogonal_opp h)⟩)
  · exact Or.inr (Or.inr ⟨rfl, Or.inr (diagonal_opp h)⟩)

theorem occ_iff (b : Board) (y : Square) : Occ b y ↔ ∃ c k, b y = some ⟨c, k⟩ := by
  unfold Occ
  constructor
  · intro h
    cases hb : b y with
    | none => exact absurd hb h
    | some q => exact ⟨q.color, q.kind, rfl⟩
  · rintro ⟨c, k, h⟩ hn
    rw [hn] at h
    cases h

/-- THE KERNEL'S ATTACK ON A KING'S SQUARE is the exposure of a king there, read on the position's board. -/
theorem isAttacked_iff_exposedR (q : Position) (c : Color) (K : Square) :
    isAttacked q (Color.opposite c) K ↔ ExposedR (fun y c k => q.board y = some ⟨c, k⟩) c K := by
  rw [isAttacked_iff]
  unfold ExposedR
  have hR : ∀ d (y : Square), Ray (fun z => ∃ c k, q.board z = some ⟨c, k⟩) d K y ↔ Ray (Occ q.board) d K y :=
    fun d y => ray_congr (fun z => (occ_iff q.board z).symm) d K y
  simp only [hR]
  constructor
  · rintro ⟨y, h⟩
    have h' := h
    simp only [pieceAttacks, pieceAttacksB] at h'
    cases hb : q.board y with
    | none => rw [hb] at h'; cases h'
    | some pc =>
        rw [hb] at h'
        obtain ⟨pcc, k⟩ := pc
        simp only [Bool.and_eq_true, beq_iff_eq] at h'
        obtain ⟨rfl, -⟩ := h'
        rw [pieceAttacks_of q _ k y K hb] at h
        rcases h with h | h | h | ⟨rfl, h⟩ | ⟨rfl, h⟩ | ⟨rfl, h⟩
        · obtain ⟨d, hd, hr⟩ := (slider_line_iff q.board k y K).2 (Or.inl h)
          exact Or.inl ⟨opp d, y, k, lineOf_opp hd, ray_reverse _ d y K hr, hb⟩
        · obtain ⟨d, hd, hr⟩ := (slider_line_iff q.board k y K).2 (Or.inr (Or.inl h))
          exact Or.inl ⟨opp d, y, k, lineOf_opp hd, ray_reverse _ d y K hr, hb⟩
        · obtain ⟨d, hd, hr⟩ := (slider_line_iff q.board k y K).2 (Or.inr (Or.inr h))
          exact Or.inl ⟨opp d, y, k, lineOf_opp hd, ray_reverse _ d y K hr, hb⟩
        · rw [knightGeometry_symm] at h
          exact Or.inr (Or.inl ⟨y, (knightJump_iff _ _ _ _).2 h, hb⟩)
        · rw [kingGeometry_symm] at h
          exact Or.inr (Or.inr (Or.inl ⟨y, (kingStep_iff _ _ _ _).2 h, hb⟩))
        · obtain ⟨d, hd, hs⟩ := (pawnStep_iff _ _ _ _ _).2 h
          exact Or.inr (Or.inr (Or.inr ⟨d, y, hd, hs, hb⟩))
  · rintro (⟨d, y, k, hl, hray, hb⟩ | ⟨y, hj, hb⟩ | ⟨y, hj, hb⟩ | ⟨d, y, hd, hs, hb⟩)
    · refine ⟨y, (pieceAttacks_of q _ k y K hb).2 ?_⟩
      have := (slider_line_iff q.board k y K).1 ⟨opp d, lineOf_opp hl, ray_reverse _ d K y hray⟩
      rcases this with h | h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr (Or.inl h))
    · refine ⟨y, (pieceAttacks_of q _ _ y K hb).2 (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, ?_⟩))))⟩
      rw [knightGeometry_symm]
      exact (knightJump_iff _ _ _ _).1 hj
    · refine ⟨y, (pieceAttacks_of q _ _ y K hb).2 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, ?_⟩)))))⟩
      rw [kingGeometry_symm]
      exact (kingStep_iff _ _ _ _).1 hj
    · exact ⟨y, (pieceAttacks_of q _ _ y K hb).2 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨rfl, (pawnStep_iff _ _ _ _ _).1 ⟨d, hd, hs⟩⟩)))))⟩

/-- (E1) THE SPEC'S EXPOSURE IS THE KERNEL'S CHECK AFTER THE MOVE, given a board after the move that the spec's `after` reads as the kernel's board on
everything a check reads: the occupied squares, the other side's units, the mover's king. -/
theorem exposed_iff_inCheck_of (p : Position) (hk : exactlyOneKing p.board p.turn) (m : Move) (hm : pseudoLegal p m) (f1 r1 f2 r2 : Fin 8)
    (hc : candidate p p.turn f1 r1 f2 r2)
    (hocc : ∀ y : Square, (∃ c k, after p f1 r1 f2 r2 y.1 y.2 c k) ↔ (∃ c k, (applyMove p m).board y = some ⟨c, k⟩))
    (hopp : ∀ (y : Square) k, after p f1 r1 f2 r2 y.1 y.2 (Color.opposite p.turn) k ↔ (applyMove p m).board y = some ⟨Color.opposite p.turn, k⟩)
    (hking : ∀ y : Square, after p f1 r1 f2 r2 y.1 y.2 p.turn .king ↔ (applyMove p m).board y = some ⟨p.turn, .king⟩) :
    exposed p f1 r1 f2 r2 ↔ inCheck (applyMove p m) p.turn := by
  have hk' : exactlyOneKing (applyMove p m).board p.turn := by
    rw [applyMove_board]
    exact invariant_pseudo_preserves_king_count p m hm p.turn hk
  obtain ⟨K, hK⟩ := oneKing_sq hk'
  have hkA : ∀ c (y : Square), kingAfter p c f1 r1 f2 r2 y.1 y.2 ↔ c = p.turn ∧ y = K := by
    intro c y
    rw [kingAfter_iff]
    constructor
    · rintro ⟨hc', ha⟩
      have := (candidate_unit p _ _ _ _ _ hc').1
      subst this
      exact ⟨rfl, (hK y).1 ((hking y).1 ha)⟩
    · rintro ⟨rfl, hy⟩
      rw [hy]
      exact ⟨hc, (hking K).2 ((hK K).2 rfl)⟩
  rw [exposed_iff p _ _ _ _ (fun c K1 c' K2 h1 h2 => by
    obtain ⟨e1, e2⟩ := (hkA c K1).1 h1
    obtain ⟨e3, e4⟩ := (hkA c' K2).1 h2
    exact ⟨e1.trans e3.symm, e2.trans e4.symm⟩)]
  rw [inCheck_one _ _ hK, isAttacked_iff_exposedR]
  constructor
  · rintro ⟨c, y, hky, he⟩
    obtain ⟨rfl, hy⟩ := (hkA c y).1 hky
    rw [hy] at he
    exact (exposedR_congr hocc hopp).1 he
  · intro he
    exact ⟨p.turn, K, (hkA _ _).2 ⟨rfl, rfl⟩, (exposedR_congr hocc hopp).2 he⟩

/-- (E2) A kernel move that is no promotion: the board after its candidate IS the kernel's board after it. -/
theorem exposed_iff_inCheck_move (p : Position) (hk : exactlyOneKing p.board p.turn) (m : Move) (hm : pseudoLegal p m) (f1 r1 f2 r2 : Fin 8)
    (ht : termOf m = .move (f1, r1) (f2, r2)) (hc : candidate p p.turn f1 r1 f2 r2) :
    exposed p f1 r1 f2 r2 ↔ inCheck (applyMove p m) p.turn := by
  have hb : ∀ (y : Square) c k, after p f1 r1 f2 r2 y.1 y.2 c k ↔ (applyMove p m).board y = some ⟨c, k⟩ := by
    intro y c k
    rw [after_iff_nextCell p hc, ← ht]
    exact (Ideas.NextEquiv.applyMoveSpec_applyMove p m hm).1 y.1 y.2 c k
  exact exposed_iff_inCheck_of p hk m hm f1 r1 f2 r2 hc (fun y => by simp only [hb]) (fun y k => hb y _ k) (fun y => hb y _ _)

theorem promotion_parts {p : Position} {s t : Square} {pk : PromotionKind} (h : pseudoLegal p (.promotion s t pk)) :
    p.board s = some ⟨p.turn, .pawn⟩ ∧ t.2 = promotionRank p.turn ∧
      ((pawnMoveGeometryB p.turn s t = true ∧ p.board t = none) ∨
        (pawnAttackGeometryB p.turn s t = true ∧ ∃ k, p.board t = some ⟨Color.opposite p.turn, k⟩ ∧ k ≠ .king)) := by
  have hb := Ideas.NextEquiv.promotion_pl h
  simp only [pseudoLegal, pseudoLegalB, pseudoLegalPromotionB, hb, beq_self_eq_true, Bool.true_and, Bool.and_eq_true, beq_iff_eq,
    Bool.or_eq_true] at h
  refine ⟨hb, h.1, ?_⟩
  rcases h.2 with h2 | ⟨h2, h3⟩
  · exact Or.inl ⟨h2.1, by simpa using h2.2⟩
  · refine Or.inr ⟨h2, ?_⟩
    cases ht : p.board t with
    | none => rw [ht] at h3; cases h3
    | some q =>
        rw [ht] at h3
        obtain ⟨qc, qk⟩ := q
        simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at h3
        exact ⟨qk, by rw [h3.1], h3.2⟩

theorem pawnMove_ne {c : Color} {s t : Square} (h : pawnMoveGeometryB c s t = true) : s ≠ t := by
  intro e
  subst e
  cases c <;> simp [pawnMoveGeometryB] at h

theorem pawnAttack_ne {c : Color} {s t : Square} (h : pawnAttackGeometryB c s t = true) : s ≠ t := by
  intro e
  exact Ideas.NextEquiv.pawnAttack_file_ne h (congrArg Prod.fst e)

/-- A kernel promotion's squares are a candidate of the side to move (its push or its capture), never its en passant nor its castling. -/
theorem promotion_cand {p : Position} {s t : Square} {pk : PromotionKind} (h : pseudoLegal p (.promotion s t pk)) :
    candidate p p.turn s.1 s.2 t.1 t.2 ∧ (∀ c, ¬ enPassant p c s.1 s.2 t.1 t.2) ∧ (∀ c w, ¬ (castle p c w ∧ s.1 = 4 ∧ s.2 = home c)) ∧ s ≠ t := by
  obtain ⟨hb, _, hg⟩ := promotion_parts h
  refine ⟨?_, ?_, ?_, ?_⟩
  · refine (candidate_iff p _ _ _ _ _).2 ⟨rfl, Or.inl ⟨_, hb, rfl, ?_, ?_⟩⟩
    · rcases hg with ⟨_, ht⟩ | ⟨_, k, ht, hk⟩
      · exact destinationFriendly_empty p _ _ ht
      · cases ht' : p.turn <;> simp [destinationFriendlyB, ht, hk, ht']
    · simp only [normalPieceGeometryB, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq]
      rcases hg with ⟨hm, ht⟩ | ⟨ha, k, ht, _⟩
      · exact Or.inl (Or.inl ⟨hm, by simpa using ht⟩)
      · exact Or.inl (Or.inr ⟨ha, by simp [ht]⟩)
  · intro c he
    cases he with
    | c1 _ ha _ hempty =>
        rcases hg with ⟨hm, _⟩ | ⟨_, k, ht, _⟩
        · obtain ⟨_, hga⟩ := (attackByPawn_iff p _ _ _ _ _).1 ha
          simp only [pawnMoveGeometryB, Bool.and_eq_true, beq_iff_eq] at hm
          exact Ideas.NextEquiv.pawnAttack_file_ne hga hm.1
        · rw [(empty_iff _ _ _).1 hempty] at ht
          cases ht
  · rintro c w ⟨hcw, h1, h2⟩
    have hk := ((castle_spec_iff p c w).1 hcw).2.2.1
    have e : s = (4, home c) := Prod.ext h1 h2
    rw [← e, hb] at hk
    cases hk
  · rcases hg with ⟨hm, _⟩ | ⟨ha, _⟩
    · exact pawnMove_ne hm
    · exact pawnAttack_ne ha

/-- The board after a promotion's candidate: the pawn moved (the unit it places is the kernel's choice, not the board's: only the kind at the landing square
differs from the kernel's, and it is the mover's unit and no king). -/
theorem after_promotion {p : Position} {s t : Square} {pk : PromotionKind} (h : pseudoLegal p (.promotion s t pk)) (y : Square) (c : Color) (k : PieceKind) :
    after p s.1 s.2 t.1 t.2 y.1 y.2 c k ↔ (y = t ∧ p.board s = some ⟨c, k⟩) ∨ (y ≠ s ∧ y ≠ t ∧ p.board y = some ⟨c, k⟩) := by
  obtain ⟨hc, hne, hnc, hst⟩ := promotion_cand h
  have hv : ∀ x : Square, vacates p s.1 s.2 t.1 t.2 x.1 x.2 ↔ x = s := by
    intro x
    rw [vacates_iff]
    constructor
    · rintro (⟨_, h1, h2⟩ | ⟨c', he, _⟩ | ⟨c', w, hcw, h1, h2, _⟩)
      · exact Prod.ext h1 h2
      · exact absurd he (hne c')
      · exact absurd ⟨hcw, h1, h2⟩ (hnc c' w)
    · intro e
      subst e
      exact Or.inl ⟨⟨_, hc⟩, rfl, rfl⟩
  have hfi : ∀ (x : Square) c k, fills p s.1 s.2 t.1 t.2 x.1 x.2 c k ↔ x = t ∧ p.board s = some ⟨c, k⟩ := by
    intro x c k
    rw [fills_iff]
    constructor
    · rintro (⟨_, hb, h1, h2⟩ | ⟨w, hcw, h1, h2, _⟩)
      · exact ⟨Prod.ext h1 h2, hb⟩
      · exact absurd ⟨hcw, h1, h2⟩ (hnc c w)
    · rintro ⟨e, hb⟩
      subst e
      have hct := (candidate_unit p _ _ _ _ _ hc)
      obtain ⟨k0, hb0⟩ := hct.2
      rw [hb0] at hb
      simp only [Option.some.injEq, Piece.mk.injEq] at hb
      obtain ⟨rfl, rfl⟩ := hb
      exact Or.inl ⟨hc, hb0, rfl, rfl⟩
  have hfd : ∀ x : Square, filled p s.1 s.2 t.1 t.2 x.1 x.2 ↔ x = t := by
    intro x
    rw [filled_iff]
    simp only [hfi]
    constructor
    · rintro ⟨_, _, e, _⟩
      exact e
    · intro e
      obtain ⟨k0, hb0⟩ := (candidate_unit p _ _ _ _ _ hc).2
      exact ⟨_, _, e, hb0⟩
  rw [after_def, stays_iff, hfi, hv, hfd]
  constructor
  · rintro (h1 | ⟨_, hb, h2, h3⟩)
    · exact Or.inl h1
    · exact Or.inr ⟨h2, h3, hb⟩
  · rintro (h1 | ⟨h2, h3, hb⟩)
    · exact Or.inl h1
    · exact Or.inr ⟨⟨_, hc⟩, hb, h2, h3⟩

/-- (E3) A kernel promotion: the spec's exposure (read on the pawn moved) IS the kernel's check after the promotion. -/
theorem exposed_iff_inCheck_promotion (p : Position) (hk : exactlyOneKing p.board p.turn) (s t : Square) (pk : PromotionKind)
    (hm : pseudoLegal p (.promotion s t pk)) : exposed p s.1 s.2 t.1 t.2 ↔ inCheck (applyMove p (.promotion s t pk)) p.turn := by
  obtain ⟨hc, _, _, hst⟩ := promotion_cand hm
  have hb := Ideas.NextEquiv.promotion_pl hm
  have hkb : ∀ y : Square, (applyMove p (.promotion s t pk)).board y = if y = t then some ⟨p.turn, pk.toPieceKind⟩ else if y = s then none else p.board y := by
    intro y
    rw [applyMove_board]
    simp only [applyBoard, Ideas.NextEquiv.boardUpdate_apply]
  have hBt : (applyMove p (.promotion s t pk)).board t = some ⟨p.turn, pk.toPieceKind⟩ := by rw [hkb, if_pos rfl]
  have hBs : (applyMove p (.promotion s t pk)).board s = none := by rw [hkb, if_neg hst, if_pos rfl]
  have hBy : ∀ y : Square, y ≠ t → y ≠ s → (applyMove p (.promotion s t pk)).board y = p.board y := fun y h1 h2 => by rw [hkb, if_neg h1, if_neg h2]
  have hpk : pk.toPieceKind ≠ .king := by cases pk <;> simp [PromotionKind.toPieceKind]
  have hon := Ideas.DeadPosition.opp_ne p.turn
  refine exposed_iff_inCheck_of p hk _ hm _ _ _ _ hc ?_ ?_ ?_
  · intro y
    simp only [after_promotion hm]
    by_cases hyt : y = t
    · subst hyt
      exact ⟨fun _ => ⟨_, _, hBt⟩, fun _ => ⟨_, _, Or.inl ⟨rfl, hb⟩⟩⟩
    · by_cases hys : y = s
      · subst hys
        constructor
        · rintro ⟨c, k, h1 | ⟨h2, _⟩⟩
          · exact absurd h1.1 hyt
          · exact absurd rfl h2
        · rintro ⟨c, k, h⟩
          rw [hBs] at h
          cases h
      · constructor
        · rintro ⟨c, k, h1 | ⟨_, _, h⟩⟩
          · exact absurd h1.1 hyt
          · exact ⟨c, k, by rw [hBy y hyt hys]; exact h⟩
        · rintro ⟨c, k, h⟩
          rw [hBy y hyt hys] at h
          exact ⟨c, k, Or.inr ⟨hys, hyt, h⟩⟩
  · intro y k
    rw [after_promotion hm]
    by_cases hyt : y = t
    · subst hyt
      rw [hBt]
      constructor
      · rintro (⟨_, h⟩ | ⟨_, h2, _⟩)
        · rw [hb] at h
          simp only [Option.some.injEq, Piece.mk.injEq] at h
          exact absurd h.1 hon.symm
        · exact absurd rfl h2
      · intro h
        simp only [Option.some.injEq, Piece.mk.injEq] at h
        exact absurd h.1 hon.symm
    · by_cases hys : y = s
      · subst hys
        rw [hBs]
        constructor
        · rintro (⟨h1, _⟩ | ⟨h2, _⟩)
          · exact absurd h1 hyt
          · exact absurd rfl h2
        · intro h
          cases h
      · rw [hBy y hyt hys]
        constructor
        · rintro (⟨h1, _⟩ | ⟨_, _, h⟩)
          · exact absurd h1 hyt
          · exact h
        · intro h
          exact Or.inr ⟨hys, hyt, h⟩
  · intro y
    rw [after_promotion hm]
    by_cases hyt : y = t
    · subst hyt
      rw [hBt]
      constructor
      · rintro (⟨_, h⟩ | ⟨_, h2, _⟩)
        · rw [hb] at h
          simp only [Option.some.injEq, Piece.mk.injEq] at h
          cases h.2
        · exact absurd rfl h2
      · intro h
        simp only [Option.some.injEq, Piece.mk.injEq] at h
        exact absurd h.2 hpk
    · by_cases hys : y = s
      · subst hys
        rw [hBs]
        constructor
        · rintro (⟨h1, _⟩ | ⟨h2, _⟩)
          · exact absurd h1 hyt
          · exact absurd rfl h2
        · intro h
          cases h
      · rw [hBy y hyt hys]
        constructor
        · rintro (⟨h1, _⟩ | ⟨_, _, h⟩)
          · exact absurd h1 hyt
          · exact h
        · intro h
          exact Or.inr ⟨hys, hyt, h⟩

/-! ## THE EQUIVALENCE: the spec's `legal` IS the kernel's `legalMove` -/

theorem lastRank_iff (c : Color) (r : Fin 8) : lastRank c r ↔ r = promotionRank c := by
  constructor
  · intro h; cases h <;> rfl
  · rintro rfl; cases c <;> constructor

theorem promotes_iff (p : Position) (c : Color) (f1 r1 f2 r2 : Fin 8) :
    promotes p c f1 r1 f2 r2 ↔ candidate p c f1 r1 f2 r2 ∧ p.board (f1, r1) = some ⟨c, .pawn⟩ ∧ r2 = promotionRank c := by
  constructor
  · rintro ⟨h1, h2, h3⟩
    exact ⟨h1, h2, (lastRank_iff _ _).1 h3⟩
  · rintro ⟨h1, h2, h3⟩
    exact promotes.c1 h1 h2 ((lastRank_iff _ _).2 h3)

theorem promotable_iff (k : PieceKind) : promotable k ↔ k ≠ .king ∧ k ≠ .pawn := by
  constructor
  · intro h; cases h <;> simp
  · intro h; cases k <;> simp at h <;> constructor

theorem promotable_toPieceKind (pk : PromotionKind) : promotable pk.toPieceKind := by cases pk <;> constructor

theorem promoKind_promotable {k : PieceKind} (h : promotable k) : (promoKind k).toPieceKind = k := by cases h <;> rfl

/-- The last rank's index, by side. -/
def promoRankVal : Color → Nat
  | .white => 7
  | .black => 0

theorem rank_of_promotionRank {r : Fin 8} {c : Color} (h : r = promotionRank c) : r.val = promoRankVal c := by
  subst h
  cases c <;> rfl

/-- The side to move's normal move of a pawn onto its last rank is the kernel's promotion (to any unit). -/
theorem promotion_of_normalLike (p : Position) (s t : Square) (hn : NormalLike p s t) (hb : p.board s = some ⟨p.turn, .pawn⟩)
    (hr : t.2 = promotionRank p.turn) (k : PromotionKind) : pseudoLegal p (.promotion s t k) := by
  obtain ⟨q, hq, _, hd, hg⟩ := hn
  rw [hb] at hq
  simp only [Option.some.injEq] at hq
  subst hq
  simp only [normalPieceGeometryB, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq] at hg
  simp only [pseudoLegal, pseudoLegalB, pseudoLegalPromotionB, hb, beq_self_eq_true, Bool.true_and, Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, hr,
    true_and]
  rcases hg with (⟨hm, he⟩ | ⟨ha, hcap⟩) | ⟨⟨hdd, _⟩, _⟩
  · exact Or.inl ⟨hm, by simpa using he⟩
  · refine Or.inr ⟨ha, ?_⟩
    cases ht : p.board t with
    | none => rw [ht] at hcap; cases hcap
    | some q' =>
        rw [ht] at hcap
        simp only [destinationFriendlyB, ht, Bool.and_eq_true, bne_iff_ne, ne_eq] at hd
        simp only [beq_iff_eq] at hcap
        simp [hcap, hd.2]
  · exfalso
    have hr' := rank_of_promotionRank hr
    simp only [pawnDoubleGeometryB, Bool.and_eq_true, beq_iff_eq] at hdd
    cases hturn : p.turn <;> rw [hturn] at hdd hr' <;> simp [promoRankVal] at hdd hr' <;> omega

theorem castleMove_isCastle (c : Color) (w : Wing) : (castleMove c w).isCastle = true := by cases c <;> cases w <;> rfl

theorem castle_of_move (m : Move) (h : m.isCastle = true) : ∃ c w, m = castleMove c w := by
  cases m <;> simp [Move.isCastle] at h
  · exact ⟨.white, .kingside, rfl⟩
  · exact ⟨.white, .queenside, rfl⟩
  · exact ⟨.black, .kingside, rfl⟩
  · exact ⟨.black, .queenside, rfl⟩

theorem not_isCastle_normal (s t : Square) : (Move.normal s t).isCastle = false := rfl
theorem not_isCastle_promotion (s t : Square) (k : PromotionKind) : (Move.promotion s t k).isCastle = false := rfl
theorem not_isCastle_ep (s t : Square) : (Move.enPassant s t).isCastle = false := rfl

theorem legal_of_castle (p : Position) (hk : exactlyOneKing p.board p.turn) (c : Color) (w : Wing) (h : legalMove p (castleMove c w)) :
    legal p p.turn (termOf (castleMove c w)) := by
  obtain ⟨hpl, hnc, hcs⟩ := h
  have hsafe : castleSafeSquares p (castleMove c w) := hcs.resolve_left (by simp [castleMove_isCastle])
  have hcw := (castle_iff p c w).2 ⟨hpl, hsafe⟩
  have hct := castle_turn hcw
  subst hct
  rw [Ideas.NextEquiv.termOf_castle]
  have hcand : candidate p p.turn 4 (home p.turn) (kingTo w) (home p.turn) := candidate.c2 ((castlingMove_iff p _ w _ _ _ _).2 ⟨hcw, rfl, rfl, rfl, rfl⟩)
  refine legal.c1 hcand (fun hpro => ?_) (fun hx => hnc ((exposed_iff_inCheck_move p hk _ hpl _ _ _ _ (Ideas.NextEquiv.termOf_castle _ _) hcand).1 hx))
  obtain ⟨_, hb, _⟩ := (promotes_iff p _ _ _ _ _).1 hpro
  have := ((castle_spec_iff p _ w).1 hcw).2.2.1
  rw [hb] at this
  cases this

/-- THE SPEC'S `legal` IS THE KERNEL'S `legalMove`, on every position with one king of the side to move and the e.p. square where the rule puts it (every
reachable position: `legal_iff_reachable`): the side to move plays exactly the terms of the kernel's legal moves (`termOf`: castling the king's two-file
move, en passant the pawn's diagonal move, a promotion with the unit it places); the other side plays noop. -/
theorem legal_iff (p : Position) (hk : exactlyOneKing p.board p.turn) (hep : EpPlaced p) (c : Color) (mv : MoveTerm) :
    legal p c mv ↔ (c = p.turn ∧ ∃ m, legalMove p m ∧ termOf m = mv) ∨ (c = Color.opposite p.turn ∧ mv = .noop) := by
  constructor
  · intro h
    cases h with
    | @c1 f1 r1 f2 r2 hcand hnp hne =>
        obtain ⟨hct, hn | he | ⟨w, hcw, rfl, rfl, rfl, rfl⟩⟩ := (candidate_iff p _ _ _ _ _).1 hcand
        · subst hct
          have hpl : pseudoLegal p (.normal (f1, r1) (f2, r2)) :=
            (normalPseudoLegal_iff p _ _).2 ⟨hn, fun ⟨hb, hr⟩ => hnp ((promotes_iff p _ _ _ _ _).2 ⟨hcand, hb, hr⟩)⟩
          exact Or.inl ⟨rfl, .normal (f1, r1) (f2, r2), ⟨hpl, fun hchk => hne ((exposed_iff_inCheck_move p hk _ hpl _ _ _ _ rfl hcand).2 hchk),
            Or.inl (by simp [not_isCastle_normal])⟩, rfl⟩
        · obtain ⟨hct', hpl⟩ := kernel_of_enPassant p hep _ _ _ _ _ he
          subst hct
          exact Or.inl ⟨rfl, .enPassant (f1, r1) (f2, r2), ⟨hpl, fun hchk => hne ((exposed_iff_inCheck_move p hk _ hpl _ _ _ _ rfl hcand).2 hchk),
            Or.inl (by simp [not_isCastle_ep])⟩, rfl⟩
        · subst hct
          obtain ⟨hpl, hsafe⟩ := (castle_iff p _ w).1 hcw
          exact Or.inl ⟨rfl, castleMove p.turn w, ⟨hpl, fun hchk => hne ((exposed_iff_inCheck_move p hk _ hpl _ _ _ _
            (Ideas.NextEquiv.termOf_castle _ _) hcand).2 hchk), Or.inr hsafe⟩, Ideas.NextEquiv.termOf_castle _ _⟩
    | @c2 f1 r1 f2 r2 k hpro hne hpk =>
        obtain ⟨hcand, hb, hr⟩ := (promotes_iff p _ _ _ _ _).1 hpro
        obtain ⟨hct, hn | he | ⟨w, hcw, rfl, rfl, rfl, rfl⟩⟩ := (candidate_iff p _ _ _ _ _).1 hcand
        · subst hct
          have hpl := promotion_of_normalLike p (f1, r1) (f2, r2) hn hb hr (promoKind k)
          refine Or.inl ⟨rfl, .promotion (f1, r1) (f2, r2) (promoKind k), ⟨hpl, fun hchk => hne
            ((exposed_iff_inCheck_promotion p hk (f1, r1) (f2, r2) _ hpl).2 hchk), Or.inl (by simp [not_isCastle_promotion])⟩, ?_⟩
          show MoveTerm.promote (f1, r1) (f2, r2) (promoKind k).toPieceKind = _
          rw [promoKind_promotable hpk]
        · exfalso
          cases he with
          | c1 hct' _ hepq =>
              have := hep _ hepq
              subst hct
              have hr' := rank_of_promotionRank hr
              simp only at this
              cases hturn : p.turn <;> rw [hturn] at this hr' <;> simp [promoRankVal] at this hr' <;> omega
        · exfalso
          have := ((castle_spec_iff p _ w).1 hcw).2.2.1
          rw [hb] at this
          cases this
    | @c3 c' hc' ho =>
        rw [opponent_iff] at ho
        exact Or.inr ⟨by rw [ho, hc'], rfl⟩
  · rintro (⟨rfl, m, ⟨hpl, hnc, hcs⟩, rfl⟩ | ⟨rfl, rfl⟩)
    · cases m with
      | normal s t =>
          have hn := (normalPseudoLegal_iff p s t).1 hpl
          have hcand : candidate p p.turn s.1 s.2 t.1 t.2 := (candidate_iff p _ _ _ _ _).2 ⟨rfl, Or.inl hn.1⟩
          exact legal.c1 hcand (fun hpro => hn.2 (let ⟨_, h2, h3⟩ := (promotes_iff p _ _ _ _ _).1 hpro; ⟨h2, h3⟩))
            (fun hx => hnc ((exposed_iff_inCheck_move p hk _ hpl _ _ _ _ rfl hcand).1 hx))
      | promotion s t pk =>
          have hcand := (promotion_cand hpl).1
          obtain ⟨hb, hr, _⟩ := promotion_parts hpl
          exact legal.c2 ((promotes_iff p _ _ _ _ _).2 ⟨hcand, hb, hr⟩) (fun hx => hnc ((exposed_iff_inCheck_promotion p hk s t pk hpl).1 hx))
            (promotable_toPieceKind pk)
      | enPassant s t =>
          have he := enPassant_of_kernel p s t hpl
          have hcand : candidate p p.turn s.1 s.2 t.1 t.2 := (candidate_iff p _ _ _ _ _).2 ⟨rfl, Or.inr (Or.inl he)⟩
          refine legal.c1 hcand (fun hpro => ?_) (fun hx => hnc ((exposed_iff_inCheck_move p hk _ hpl _ _ _ _ rfl hcand).1 hx))
          obtain ⟨_, _, hr⟩ := (promotes_iff p _ _ _ _ _).1 hpro
          obtain ⟨_, _, _, _, cs, hcs'⟩ := Ideas.NextEquiv.ep_pl hpl
          have hr' := rank_of_promotionRank hr
          unfold epCapturedSquare at hcs'
          split_ifs at hcs' with h1 h2
          · rw [h1.1] at hr'
            simp [promoRankVal] at hr'
            omega
          · rw [h2.1] at hr'
            simp [promoRankVal] at hr'
            omega
      | whiteKingCastle => exact legal_of_castle p hk .white .kingside ⟨hpl, hnc, hcs⟩
      | whiteQueenCastle => exact legal_of_castle p hk .white .queenside ⟨hpl, hnc, hcs⟩
      | blackKingCastle => exact legal_of_castle p hk .black .kingside ⟨hpl, hnc, hcs⟩
      | blackQueenCastle => exact legal_of_castle p hk .black .queenside ⟨hpl, hnc, hcs⟩
    · exact legal.c3 rfl ((opponent_iff _ _).2 rfl)

/-- THE EQUIVALENCE ON EVERY REACHABLE POSITION (no hypothesis left: one king each side, `Chess.reachable_positionInvariant`; the e.p. square where the
rule puts it, `reachable_epPlaced`). -/
theorem legal_iff_reachable (p : Position) (hr : reachable p) (c : Color) (mv : MoveTerm) :
    legal p c mv ↔ (c = p.turn ∧ ∃ m, legalMove p m ∧ termOf m = mv) ∨ (c = Color.opposite p.turn ∧ mv = .noop) := by
  have hinv := reachable_positionInvariant p hr
  have hk : exactlyOneKing p.board p.turn := by
    cases ht : p.turn
    · exact hinv.1
    · exact hinv.2.1
  exact legal_iff p hk (reachable_epPlaced p hr) c mv

/-- The move binding reads every legal term back to the kernel's move it came from (`moveOf`). -/
theorem legal_moveOf (p : Position) (hk : exactlyOneKing p.board p.turn) (hep : EpPlaced p) (mv : MoveTerm) (h : legal p p.turn mv) (hn : mv ≠ .noop) :
    ∃ m, moveOf p mv = some m ∧ legalMove p m := by
  rcases (legal_iff p hk hep _ mv).1 h with ⟨_, m, hm, rfl⟩ | ⟨he, _⟩
  · exact ⟨m, Ideas.NextEquiv.moveOf_termOf p m hm.1, hm⟩
  · exact absurd he (Ideas.DeadPosition.opp_ne p.turn).symm

/-! ## THE END OF THE GAME: the spec's terminals and goals in the kernel's words -/

/-- The side to move HAS A MOVE in the spec exactly when the kernel lists a legal move. -/
theorem hasMove_iff (p : Position) (hk : exactlyOneKing p.board p.turn) (hep : EpPlaced p) : hasMove p ↔ legalMoves p ≠ [] := by
  constructor
  · rintro ⟨hc, hl⟩
    subst hc
    rcases (legal_iff p hk hep _ _).1 hl with ⟨_, m, hm, _⟩ | ⟨he, _⟩
    · intro h
      have := (legalMoves_correct p m).2 hm
      rw [h] at this
      cases this
    · exact absurd he (Ideas.DeadPosition.opp_ne p.turn).symm
  · intro h
    obtain ⟨m, hm⟩ := List.exists_mem_of_ne_nil _ h
    exact hasMove.c1 rfl ((legal_iff p hk hep _ _).2 (Or.inl ⟨rfl, m, (legalMoves_correct p m).1 hm, rfl⟩))

/-- THE SPEC'S MATE IS THE KERNEL'S CHECKMATE (`Chess.checkmate`: in check, no legal move). -/
theorem mate_iff (p : Position) (hk : exactlyOneKing p.board p.turn) (hep : EpPlaced p) : mate p ↔ checkmate p := by
  simp only [checkmate, checkmateB, Bool.and_eq_true, List.isEmpty_iff]
  constructor
  · rintro ⟨hc, hn⟩
    obtain ⟨_, hchk⟩ := (check_iff p hk _).1 hc
    refine ⟨hchk, ?_⟩
    by_contra h
    exact hn ((hasMove_iff p hk hep).2 h)
  · rintro ⟨hchk, he⟩
    exact mate.c1 ((check_iff p hk p.turn).2 ⟨rfl, hchk⟩) (fun h => (hasMove_iff p hk hep).1 h he)

/-- THE SPEC'S STALEMATE IS THE KERNEL'S (`Chess.stalemate`: not in check, no legal move). -/
theorem stalemate_iff (p : Position) (hk : exactlyOneKing p.board p.turn) (hep : EpPlaced p) : Ideas.Spec.Chess.stalemate p ↔ Chess.stalemate p := by
  simp only [Chess.stalemate, stalemateB, Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_iff]
  constructor
  · rintro ⟨hc, hnc, hn⟩
    subst hc
    refine ⟨?_, ?_⟩
    · cases h : inCheckB p p.turn
      · rfl
      · exact absurd ((check_iff p hk _).2 ⟨rfl, h⟩) hnc
    · by_contra h
      exact hn ((hasMove_iff p hk hep).2 h)
  · rintro ⟨hchk, he⟩
    refine stalemate.c1 rfl (fun h => ?_) (fun h => (hasMove_iff p hk hep).1 h he)
    have := ((check_iff p hk _).1 h).2
    have h' : inCheckB p p.turn = true := this
    rw [hchk] at h'
    cases h'

/-- The spec's mate and stalemate are the machinery's game's (`Ideas.Game.chess`: the kernel's legal moves and check). -/
theorem mate_iff_mated (p : Position) (hk : exactlyOneKing p.board p.turn) (hep : EpPlaced p) : mate p ↔ Ideas.Game.chess.mated p = true := by
  rw [mate_iff p hk hep]
  simp only [checkmate, checkmateB, Ideas.Game.Game.mated, Ideas.Game.Game.terminal, Ideas.Game.chess, Bool.and_eq_true]
  tauto

theorem stalemate_iff_stalemated (p : Position) (hk : exactlyOneKing p.board p.turn) (hep : EpPlaced p) :
    Ideas.Spec.Chess.stalemate p ↔ Ideas.Game.chess.stalemated p = true := by
  rw [stalemate_iff p hk hep]
  simp only [Chess.stalemate, stalemateB, Ideas.Game.Game.stalemated, Ideas.Game.Game.terminal, Ideas.Game.chess, Bool.and_eq_true]
  tauto

/-- THE SPEC'S FIFTY-MOVE DRAW IS THE KERNEL'S CLAIMABLE ONE (`Chess.fiftyMoveClaimable`: a hundred half-moves) with a legal move to make (a mate or a
stalemate on the move comes first, as in `Chess.gameStatus`). -/
theorem drawByFiftyMoves_iff (p : Position) (hk : exactlyOneKing p.board p.turn) (hep : EpPlaced p) :
    drawByFiftyMoves p ↔ fiftyMoveClaimable p ∧ legalMoves p ≠ [] := by
  constructor
  · rintro ⟨hl, hc, hm⟩
    rw [Ideas.NextEquiv.clockLimit_iff] at hl
    subst hl
    simp only [Ideas.Spec.Binding.clock] at hc
    exact ⟨by simp only [fiftyMoveClaimable]; omega, (hasMove_iff p hk hep).1 hm⟩
  · rintro ⟨hf, hm⟩
    simp only [fiftyMoveClaimable] at hf
    exact drawByFiftyMoves.c1 ((Ideas.NextEquiv.clockLimit_iff _).2 rfl) (by simp only [Ideas.Spec.Binding.clock]; omega) ((hasMove_iff p hk hep).2 hm)

/-- THE SPEC'S WIN AND LOSS ARE THE KERNEL'S (`Chess.gameStatus` of the game's history -- the record's positions, then this one -- reads a checkmate first):
a side scores 100 exactly when the kernel says it won, 0 exactly when it says the other side won. -/
theorem goal_win_iff (p : Position) (hk : exactlyOneKing p.board p.turn) (hep : EpPlaced p) (rec : Ideas.Game.Record Position RepetitionKey) (c : Color) :
    goal p rec c 100 ↔ gameStatus (rec.before ++ [p]) = (match c with | .white => GameStatus.whiteWin | .black => GameStatus.blackWin) := by
  have hst : gameStatus (rec.before ++ [p]) = if checkmateB p then (if p.turn = .white then .blackWin else .whiteWin) else
      (gameStatus (rec.before ++ [p])) := by
    split_ifs with h1 h2
    · simp only [gameStatus, List.getLast?_concat, h1, if_true, h2]
    · simp only [gameStatus, List.getLast?_concat, h1, if_true, h2, if_false]
    · rfl
  constructor
  · intro h
    cases h with
    | @c2 o hm ht ho =>
        rw [opponent_iff] at ho
        subst ho
        have hcm : checkmateB p = true := (mate_iff p hk hep).1 hm
        rw [hst, if_pos hcm]
        subst ht
        cases hturn : p.turn <;> simp [hturn]
  · intro h
    by_cases hcm : checkmateB p = true
    · rw [hst, if_pos hcm] at h
      have hm := (mate_iff p hk hep).2 hcm
      refine goal.c2 hm rfl ((opponent_iff _ _).2 ?_)
      cases hturn : p.turn <;> cases c <;> simp_all
    · exfalso
      simp only [gameStatus, List.getLast?_concat, hcm, Bool.false_eq_true, if_false] at h
      cases c <;> split_ifs at h <;> cases h

theorem goal_loss_iff (p : Position) (hk : exactlyOneKing p.board p.turn) (hep : EpPlaced p) (rec : Ideas.Game.Record Position RepetitionKey) (c : Color) :
    goal p rec c 0 ↔ gameStatus (rec.before ++ [p]) = (match c with | .white => GameStatus.blackWin | .black => GameStatus.whiteWin) := by
  constructor
  · intro h
    cases h with
    | c1 hm ht =>
        have hcm : checkmateB p = true := (mate_iff p hk hep).1 hm
        subst ht
        simp only [gameStatus, List.getLast?_concat, hcm, if_true]
        cases hturn : p.turn <;> simp [hturn]
  · intro h
    by_cases hcm : checkmateB p = true
    · have hm := (mate_iff p hk hep).2 hcm
      simp only [gameStatus, List.getLast?_concat, hcm, if_true] at h
      refine goal.c1 hm ?_
      cases hturn : p.turn <;> cases c <;> simp_all
    · exfalso
      simp only [gameStatus, List.getLast?_concat, hcm, Bool.false_eq_true, if_false] at h
      cases c <;> split_ifs at h <;> cases h

/-- `Chess.gameStatus` of a history ending at `p` is ongoing exactly when none of its terminal tests fires. -/
theorem gameStatus_ongoing_iff (hs : List Position) (p : Position) (h : hs.getLast? = some p) :
    gameStatus hs = .ongoing ↔ ¬ checkmateB p = true ∧ ¬ stalemateB p = true ∧ ¬ deadPosition p ∧ ¬ fivefoldRepetition hs ∧ ¬ 150 ≤ p.halfmove ∧
      ¬ threefoldClaimable hs ∧ ¬ 100 ≤ p.halfmove := by
  unfold gameStatus
  simp only [h]
  split_ifs <;> simp_all

/-- THE SPEC'S TERMINAL IS A KERNEL TERMINAL, with one clause left open: on a sound game record, with one king each side standing apart (every reachable
position), a position the spec ends is one `Chess.gameStatus` ends -- except possibly a `draw` of the criterion's KNIGHT clause (a king and one knight
against kings and queens), whose dead position is proved only in the position itself (`Ideas.InsufficientMaterial`), not for every continuation: the
store's CONJECTURE `the spec's knight draw is a dead position`. -/
theorem terminal_sound (p : Position) (rec : Ideas.Game.Record Position RepetitionKey) (hs : rec.Sound repetitionKey)
    (hK : ∀ c, ∃ k, ∀ s, p.board s = some ⟨c, .king⟩ ↔ s = k)
    (hfar : ∀ k w, p.board k = some ⟨p.turn, .king⟩ → p.board w = some ⟨Color.opposite p.turn, .king⟩ → kingGeometryB k w = false)
    (hep : EpPlaced p) (ht : terminal p rec) : (draw p ∧ anyKnight p) ∨ gameStatus (rec.before ++ [p]) ≠ .ongoing := by
  have hk : exactlyOneKing p.board p.turn := by
    obtain ⟨k, hk⟩ := hK p.turn
    unfold exactlyOneKing squaresOf
    rw [Finset.card_eq_one]
    exact ⟨k, Finset.ext fun s => by simp [hk]⟩
  by_cases hkn : draw p ∧ anyKnight p
  · exact Or.inl hkn
  right
  intro hon
  obtain ⟨h1, h2, h3, _, _, h6, h7⟩ := (gameStatus_ongoing_iff _ p List.getLast?_concat).1 hon
  cases ht with
  | c1 hm => exact h1 ((mate_iff p hk hep).1 hm)
  | c2 hsm => exact h2 ((stalemate_iff p hk hep).1 hsm)
  | c3 hd => exact h3 (Ideas.DeadPosition.dead_of_spec_draw p hK hfar hd (fun hn => hkn ⟨hd, hn⟩))
  | c4 hf => exact h7 ((drawByFiftyMoves_iff p hk hep).1 hf).1
  | c5 hr => exact h6 ((Ideas.NextEquiv.drawByRepetition_kernel p rec hs).1 hr)

/-- A KERNEL TERMINAL IS A SPEC TERMINAL OR A DEAD POSITION: on a sound record (one king of the side to move, the e.p. square placed), every position
`Chess.gameStatus` ends is one the spec ends, or a FIDE dead position (5.2.2) -- the kernel states the rule, the spec its sufficient material criterion, and
a dead position beyond the criterion (a locked pawn chain) is not the spec's terminal: there the spec is the engine's rule (python-chess's), not FIDE's. -/
theorem terminal_complete (p : Position) (rec : Ideas.Game.Record Position RepetitionKey) (hs : rec.Sound repetitionKey)
    (hk : exactlyOneKing p.board p.turn) (hep : EpPlaced p) (hg : gameStatus (rec.before ++ [p]) ≠ .ongoing) : terminal p rec ∨ deadPosition p := by
  have hlast : (rec.before ++ [p]).getLast? = some p := List.getLast?_concat
  have hrep : threefoldClaimable (rec.before ++ [p]) → terminal p rec := fun h => terminal.c5 ((Ideas.NextEquiv.drawByRepetition_kernel p rec hs).2 h)
  have hfifty : 100 ≤ p.halfmove → ¬ checkmateB p = true → ¬ stalemateB p = true → terminal p rec := by
    intro h100 hnm hns
    refine terminal.c4 ((drawByFiftyMoves_iff p hk hep).2 ⟨h100, fun he => ?_⟩)
    cases hc : inCheckB p p.turn
    · exact hns (by simp [stalemateB, hc, he])
    · exact hnm (by simp [checkmateB, hc, he])
  by_contra hn
  apply hg
  refine (gameStatus_ongoing_iff _ p hlast).2 ⟨fun h => hn (Or.inl (terminal.c1 ((mate_iff p hk hep).2 h))),
    fun h => hn (Or.inl (terminal.c2 ((stalemate_iff p hk hep).2 h))), fun h => hn (Or.inr h), fun h => ?_, fun h => ?_, fun h => hn (Or.inl (hrep h)),
    fun h => ?_⟩
  · apply hn
    refine Or.inl (hrep ?_)
    simp only [fivefoldRepetition, fivefoldRepetitionB, hlast, decide_eq_true_eq] at h
    simp only [threefoldClaimable, threefoldClaimableB, hlast, decide_eq_true_eq]
    omega
  · by_cases hm : checkmateB p = true
    · exact hn (Or.inl (terminal.c1 ((mate_iff p hk hep).2 hm)))
    · by_cases hsm : stalemateB p = true
      · exact hn (Or.inl (terminal.c2 ((stalemate_iff p hk hep).2 hsm)))
      · exact hn (Or.inl (hfifty (by omega) hm hsm))
  · by_cases hm : checkmateB p = true
    · exact hn (Or.inl (terminal.c1 ((mate_iff p hk hep).2 hm)))
    · by_cases hsm : stalemateB p = true
      · exact hn (Or.inl (terminal.c2 ((stalemate_iff p hk hep).2 hsm)))
      · exact hn (Or.inl (hfifty h hm hsm))

end Ideas.LegalEquiv
