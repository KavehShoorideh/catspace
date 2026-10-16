import Ideas.TreeStep
import Ideas.InsufficientMaterial

/-!
# THE ROOT AND ITS LAYERS (Kaveh 2026-10-15, ANALYST_PLAN 42)

The goal of the game is mate; mate needs pieces; so the simplest lemma is the ROOT: a piece that hangs is worth its value to the
side that can take it. Everything else is a layer on the root.

* 42.3 item 0, a lone king cannot give check, so cannot mate: ALREADY PROVED in `Ideas.InsufficientMaterial` --
  `no_mate_by_lone_king` (a lone king cannot mate), `no_mate_king_minor` (king and one knight or bishop cannot mate),
  `insufficient_material`. Not re-proved here.
* Item 1: material after a move (`material_boardMove`: a move of one piece changes a side's material by exactly what that side had on
  the destination), item 2: `capture_gains`.
* Items 3-4: the reply bound for w = 0 and THE ROOT `hanging`.
* 42.4 the first layer, a defended piece: no new theorem. It is the root with w = the capturer's worth (their recapture on y is a
  capture of a piece worth `worth x` on an attacked square, so the maximum over our attacked pieces counts it). The full exchange
  sequence is the exchange law (ExchangeLaw.lean, unproved) and is not in this build.
-/

namespace Ideas.Root

open Chess
open Ideas.Vocabulary
open Ideas.TreeStep

/-! ## a sum over a list with two entries changed -/

theorem sum_two_changed (f g : Square → Nat) (s t : Square) (hst : s ≠ t) (hoff : ∀ u, u ≠ s → u ≠ t → g u = f u) :
    ∀ l : List Square, l.Nodup →
      (l.map g).sum + (if s ∈ l then f s else 0) + (if t ∈ l then f t else 0) =
      (l.map f).sum + (if s ∈ l then g s else 0) + (if t ∈ l then g t else 0) := by
  intro l
  induction l with
  | nil => intro _; simp
  | cons v l ih =>
    intro hnd
    have hv : v ∉ l := (List.nodup_cons.mp hnd).1
    have ih' := ih (List.nodup_cons.mp hnd).2
    simp only [List.map_cons, List.sum_cons, List.mem_cons]
    by_cases h1 : v = s
    · subst h1
      have h2 : ¬ (t = v) := fun h => hst h.symm
      have hsl : v ∉ l := hv
      simp only [true_or, if_true, h2, false_or, hsl, if_false] at ih' ⊢
      split_ifs at ih' ⊢ <;> omega
    · by_cases h2 : v = t
      · subst h2
        have h1' : ¬ (s = v) := fun h => h1 h.symm
        have htl : v ∉ l := hv
        simp only [true_or, if_true, h1', false_or, htl, if_false] at ih' ⊢
        split_ifs at ih' ⊢ <;> omega
      · have := hoff v h1 h2
        have h1' : ¬ (s = v) := fun h => h1 h.symm
        have h2' : ¬ (t = v) := fun h => h2 h.symm
        simp only [h1', h2', false_or] at ih' ⊢
        split_ifs at ih' ⊢ <;> omega

/-- Two positions whose boards agree off `s`, `t`: their material differs by what the two squares hold. -/
theorem material_two_changed (p p' : Position) (c : Color) (s t : Square) (hst : s ≠ t)
    (hoff : ∀ u, u ≠ s → u ≠ t → p'.board u = p.board u) :
    materialOf p' c + contrib p c s + contrib p c t = materialOf p c + contrib p' c s + contrib p' c t := by
  rw [materialOf_eq_sum, materialOf_eq_sum]
  have h := sum_two_changed (contrib p c) (contrib p' c) s t hst
    (fun u h1 h2 => by unfold contrib; rw [hoff u h1 h2]) allSquares allSquares_nodup
  simp only [mem_allSquares, if_true] at h
  exact h

/-! ## item 1: material after a move -/

/-- ITEM 1 (a, b together): a piece moves from `s` to `t`; each side's material afterwards plus what that side had on `t` is what it was
before. The mover keeps all of its material (it had nothing on `t`, or a capture of its own piece is not a move); the other side loses
exactly the worth of what stood on `t`. -/
theorem material_boardMove (p p' : Position) (s t : Square) (hst : s ≠ t) (q : Piece) (hs : p.board s = some q)
    (hb : p'.board = boardMove p.board s t) (d : Color) :
    materialOf p' d + contrib p d t = materialOf p d := by
  have hoff : ∀ u, u ≠ s → u ≠ t → p'.board u = p.board u := by
    intro u h1 h2
    rw [hb]; simp [boardMove, boardUpdate, Function.update_apply, h1, h2]
  have h := material_two_changed p p' d s t hst hoff
  have h1 : contrib p' d s = 0 := by
    unfold contrib; rw [hb]; simp [boardMove, boardUpdate, Function.update_apply, hst]
  have h2 : contrib p' d t = contrib p d s := by
    have ht : p'.board t = some q := by rw [hb]; simp [boardMove, boardUpdate, hs]
    unfold contrib; rw [ht, hs]
  omega

theorem contrib_other (p : Position) (c : Color) (u : Square) (r : Piece) (h : p.board u = some r) (hc : r.color ≠ c) :
    contrib p c u = 0 := by
  unfold contrib; rw [h]; simp [beq_iff_eq, hc]

theorem contrib_own (p : Position) (c : Color) (u : Square) (r : Piece) (h : p.board u = some r) (hc : r.color = c) :
    contrib p c u = (pieceValue r.kind).toNat := by
  unfold contrib; rw [h]; simp [beq_iff_eq, hc]

theorem contrib_none (p : Position) (c : Color) (u : Square) (h : p.board u = none) : contrib p c u = 0 := by
  unfold contrib; rw [h]

theorem worthOf_eq (p : Position) (u : Square) (r : Piece) (h : p.board u = some r) : worthOf p u = (pieceValue r.kind).toNat := by
  unfold worthOf; rw [h]

/-! ## item 2: a capture gains the worth of the captured piece -/

/-- ITEM 2: a normal move of the side to move onto a piece of the other side: its material is unchanged, the other side's falls by
`worthOf p y` (stated as the two equations, so no truncated subtraction hides in them). -/
theorem capture_gains (p : Position) (x y : Square) (hxy : x ≠ y) (q r : Piece) (hx : p.board x = some q) (hy : p.board y = some r)
    (hr : r.color ≠ q.color) :
    materialOf (applyMove p (Move.normal x y)) q.color = materialOf p q.color ∧
    materialOf (applyMove p (Move.normal x y)) r.color + worthOf p y = materialOf p r.color := by
  have hb : (applyMove p (Move.normal x y)).board = boardMove p.board x y := rfl
  constructor
  · have := material_boardMove p _ x y hxy q hx hb q.color
    rw [contrib_other p q.color y r hy hr] at this
    omega
  · have := material_boardMove p _ x y hxy q hx hb r.color
    rw [contrib_own p r.color y r hy rfl] at this
    rw [worthOf_eq p y r hy]
    omega

/-- The same in the inequality form the root uses, in the balance: after the capture our balance is at least what it was plus the worth. -/
theorem capture_balance (p : Position) (x y : Square) (hxy : x ≠ y) (q r : Piece) (hx : p.board x = some q) (hy : p.board y = some r)
    (hr : r.color ≠ q.color) :
    (materialOf (applyMove p (Move.normal x y)) q.color : Int) - materialOf (applyMove p (Move.normal x y)) r.color =
      (materialOf p q.color : Int) - materialOf p r.color + worthOf p y := by
  obtain ⟨h1, h2⟩ := capture_gains p x y hxy q r hx hy hr
  omega

/-! ## the replies: nothing is gained by a move that takes nothing -/

/-- A piece moves onto an empty square: nobody's material changes. -/
theorem material_boardMove_empty (p p' : Position) (s t : Square) (hst : s ≠ t) (q : Piece) (hs : p.board s = some q)
    (ht : p.board t = none) (hb : p'.board = boardMove p.board s t) (d : Color) : materialOf p' d = materialOf p d := by
  have := material_boardMove p p' s t hst q hs hb d
  rw [contrib_none p d t ht] at this
  omega

/-- Castling moves a king and a rook onto empty squares: nobody's material changes. -/
theorem castle_material (p : Position) (m : Move) (hm : pseudoLegalCastleB p m = true) (d : Color) :
    materialOf (applyMove p m) d = materialOf p d := by
  have key : ∀ (ks kd rs rd : Square) (k r : Piece), p.board ks = some k → p.board rs = some r → p.board kd = none → p.board rd = none →
      ks ≠ kd → rs ≠ rd → rs ≠ ks → rs ≠ kd → rd ≠ ks → rd ≠ kd → 
      materialOf { p with board := boardMove (boardMove p.board ks kd) rs rd } d = materialOf p d := by
    intro ks kd rs rd k r hk hr hkd hrd h1 h2 h3 h4 h5 h6
    let p1 : Position := { p with board := boardMove p.board ks kd }
    have e1 : materialOf p1 d = materialOf p d := material_boardMove_empty p p1 ks kd h1 k hk hkd rfl d
    have hr1 : p1.board rs = some r := by
      show boardMove p.board ks kd rs = some r
      simp [boardMove, boardUpdate, Function.update_apply, h3, h4, hr]
    have hrd1 : p1.board rd = none := by
      show boardMove p.board ks kd rd = none
      simp [boardMove, boardUpdate, Function.update_apply, h5, h6, hrd]
    have e2 := material_boardMove_empty p1 { p with board := boardMove (boardMove p.board ks kd) rs rd } rs rd h2 r hr1 hrd1 rfl d
    rw [e2, e1]
  cases m with
  | whiteKingCastle =>
    simp only [pseudoLegalCastleB, castleColorOf, castleRightOf, castleRookSource, castleClearB, castleEmptySquares, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq,
      List.all_cons, List.all_nil, Bool.and_true, if_true, ite_true, ↓reduceIte] at hm
    obtain ⟨⟨⟨⟨_, _⟩, hk⟩, hr⟩, hf, hg⟩ := hm
    exact key _ _ _ _ _ _ hk hr hg hf (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  | whiteQueenCastle =>
    simp only [pseudoLegalCastleB, castleColorOf, castleRightOf, castleRookSource, castleClearB, castleEmptySquares, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq,
      List.all_cons, List.all_nil, Bool.and_true, if_true, ite_true, ↓reduceIte] at hm
    obtain ⟨⟨⟨⟨_, _⟩, hk⟩, hr⟩, hb1, hc1, hd1⟩ := hm
    exact key _ _ _ _ _ _ hk hr hc1 hd1 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  | blackKingCastle =>
    simp only [pseudoLegalCastleB, castleColorOf, castleRightOf, castleRookSource, castleClearB, castleEmptySquares, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq,
      List.all_cons, List.all_nil, Bool.and_true, if_true, ite_true, ↓reduceIte] at hm
    obtain ⟨⟨⟨⟨_, _⟩, hk⟩, hr⟩, hf, hg⟩ := hm
    exact key _ _ _ _ _ _ hk hr hg hf (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  | blackQueenCastle =>
    simp only [pseudoLegalCastleB, castleColorOf, castleRightOf, castleRookSource, castleClearB, castleEmptySquares, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq,
      List.all_cons, List.all_nil, Bool.and_true, if_true, ite_true, ↓reduceIte] at hm
    obtain ⟨⟨⟨⟨_, _⟩, hk⟩, hr⟩, hb1, hc1, hd1⟩ := hm
    exact key _ _ _ _ _ _ hk hr hc1 hd1 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  | normal s t => simp [pseudoLegalCastleB, castleColorOf] at hm
  | promotion s t k => simp [pseudoLegalCastleB, castleColorOf] at hm
  | enPassant s t => simp [pseudoLegalCastleB, castleColorOf] at hm

/-- What lands on a piece attacks it: a normal capture's geometry is the attack. -/
theorem normal_capture_attacked (q : Position) (s t : Square) (q0 r : Piece) (hs : q.board s = some q0) (ht : q.board t = some r)
    (hcol : q0.color = q.turn) (hg : normalPieceGeometryB q q.turn q0 s t = true) : isAttackedB q q.turn t = true := by
  unfold isAttackedB
  rw [List.any_eq_true]
  refine ⟨s, mem_allSquares s, ?_⟩
  unfold pieceAttacksB
  rw [hs]
  have hc : (q0.color == q.turn) = true := by rw [beq_iff_eq]; exact hcol
  unfold normalPieceGeometryB at hg
  show (q0.color == q.turn && _) = true
  rw [hc, Bool.true_and]
  cases hk : q0.kind <;> rw [hk] at hg <;> simp_all [sliderAttackB]

theorem legal_pseudo (p : Position) (m : Move) (hm : m ∈ legalMoves p) : pseudoLegalB p m = true := by
  have hl : legalMove p m := by
    unfold legalMoves at hm
    have h2 := (List.mem_filter.mp hm).2
    exact @of_decide_eq_true _ (legalMoveDecidable p m) h2
  exact hl.1

theorem color_of_ne_opposite (a c : Color) (h : a ≠ Color.opposite c) : a = c := by
  cases a <;> cases c <;> simp_all [Color.opposite]

/-- ITEM 3 for w = 0: a reply of theirs that is neither a promotion nor (with no en-passant square) an en passant, from a position where
nothing of ours (not the king) is attacked, loses us no material and gains them none. (The general bound with the largest attacked
worth `w` and a pawn about to promote is the same argument with the capture's worth in place of 0.) -/
theorem reply_no_loss (q : Position) (c : Color) (hq : q.turn = Color.opposite c) (hep : q.enPassant = none)
    (hsafe : ∀ t r, q.board t = some r → r.color = c → r.kind ≠ PieceKind.king → isAttackedB q (Color.opposite c) t = false)
    (m : Move) (hm : m ∈ legalMoves q) (hnp : ∀ s t k, m ≠ Move.promotion s t k) :
    materialOf q c ≤ materialOf (applyMove q m) c ∧ materialOf (applyMove q m) (Color.opposite c) ≤ materialOf q (Color.opposite c) := by
  have hps := legal_pseudo q m hm
  cases m with
  | promotion s t k => exact absurd rfl (hnp s t k)
  | enPassant s t => simp [pseudoLegalB, pseudoLegalEnPassantB, hep] at hps
  | whiteKingCastle | whiteQueenCastle | blackKingCastle | blackQueenCastle =>
    all_goals
      have h := castle_material q _ (by unfold pseudoLegalB at hps; exact hps)
      rw [h c, h (Color.opposite c)]
      exact ⟨le_refl _, le_refl _⟩
  | normal s t =>
    simp only [pseudoLegalB, normalPseudoLegalB] at hps
    cases hs : q.board s with
    | none => simp [hs] at hps
    | some q0 =>
      rw [hs] at hps
      simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at hps
      obtain ⟨⟨⟨hcol, hdest⟩, _⟩, hgeo⟩ := hps
      have hst : s ≠ t := by
        rintro rfl
        unfold destinationFriendlyB at hdest
        rw [hs] at hdest
        simp [hcol] at hdest
      have hb : (applyMove q (Move.normal s t)).board = boardMove q.board s t := rfl
      have e1 := material_boardMove q _ s t hst q0 hs hb c
      have e2 := material_boardMove q _ s t hst q0 hs hb (Color.opposite c)
      refine ⟨?_, by omega⟩
      have h0 : contrib q c t = 0 := by
        cases ht : q.board t with
        | none => exact contrib_none q c t ht
        | some r =>
          unfold destinationFriendlyB at hdest
          rw [ht] at hdest
          simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hdest
          have hr : r.color = c := color_of_ne_opposite _ _ (by rw [← hq]; exact hdest.1)
          have hatt := normal_capture_attacked q s t q0 r hs ht hcol hgeo
          rw [hq] at hatt
          have := hsafe t r ht hr hdest.2
          rw [this] at hatt
          exact absurd hatt (by simp)
      omega

end Ideas.Root
