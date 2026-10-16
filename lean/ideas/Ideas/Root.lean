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

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

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

/-! ## item 4: THE ROOT -/

/-- A legal move from a piece that is not a pawn onto a piece (so not onto an empty square) is a normal move. -/
theorem move_is_normal (p : Position) (m : Move) (hm : m ∈ legalMoves p) (q0 r : Piece) (hs : p.board m.source = some q0)
    (hq : q0.kind ≠ PieceKind.pawn) (ht : p.board m.destination = some r) : m = Move.normal m.source m.destination := by
  have hps := legal_pseudo p m hm
  cases m with
  | normal s t => rfl
  | promotion s t k =>
    exfalso
    simp only [Move.source] at hs
    simp only [pseudoLegalB, pseudoLegalPromotionB, hs, Bool.and_eq_true, beq_iff_eq] at hps
    obtain ⟨⟨⟨_, hpawn⟩, _⟩, _⟩ := hps
    exact hq hpawn
  | enPassant s t =>
    exfalso
    simp only [Move.source] at hs
    simp only [pseudoLegalB, pseudoLegalEnPassantB] at hps
    cases hep : p.enPassant with
    | none => simp [hep] at hps
    | some e =>
      simp only [hep, hs, Bool.and_eq_true, beq_iff_eq] at hps
      exact hq (by simp_all)
  | _ =>
    exfalso
    all_goals
      simp only [Move.destination] at ht
      simp only [pseudoLegalB, pseudoLegalCastleB, castleColorOf, castleRightOf, castleRookSource, castleClearB, castleEmptySquares, Bool.and_eq_true,
        beq_iff_eq, decide_eq_true_eq, List.all_cons, List.all_nil, Bool.and_true, if_true, ite_true, ↓reduceIte] at hps
      simp_all

theorem kingSq_board (p : Position) (c : Color) (W : Square) (h : kingSq p c = some W) : p.board W = some ⟨c, PieceKind.king⟩ := by
  unfold kingSq at h
  have hm := List.mem_of_mem_head? h
  have := (List.mem_filter.mp hm).2
  revert this
  cases hb : p.board W with
  | none => simp
  | some q =>
    intro h2
    simp only [Bool.and_eq_true, beq_iff_eq] at h2
    obtain ⟨h3, h4⟩ := h2
    cases q; simp_all

/-- Lifting our king off the board only adds to what they attack. -/
theorem attacked_withEmpty (q : Position) (d : Color) (W t : Square) (hW : ∀ pc, q.board W = some pc → pc.color ≠ d)
    (h : isAttackedB q d t = true) : isAttackedB (withEmpty q W) d t = true := by
  unfold isAttackedB at h ⊢
  rw [List.any_eq_true] at h ⊢
  obtain ⟨s, hs, hatt⟩ := h
  refine ⟨s, hs, ?_⟩
  unfold pieceAttacksB at hatt ⊢
  cases hb : q.board s with
  | none => rw [hb] at hatt; simp at hatt
  | some pc =>
    rw [hb] at hatt
    have hcol : pc.color = d := by simpa using (Bool.and_eq_true_iff.mp hatt).1
    have hsW : s ≠ W := by
      rintro rfl
      exact hW pc hb hcol
    have hb' : (withEmpty q W).board s = some pc := by
      simp [withEmpty, boardUpdate, hsW, hb]
    rw [hb']
    have hclear : ∀ u, q.board u = none → (withEmpty q W).board u = none := by
      intro u hu
      by_cases huW : u = W
      · simp [withEmpty, boardUpdate, huW]
      · simp [withEmpty, boardUpdate, huW, hu]
    have hcb : clearBetweenB q.board s t = true → clearBetweenB (withEmpty q W).board s t = true := by
      intro hc
      unfold clearBetweenB at hc ⊢
      rw [List.all_eq_true] at hc ⊢
      intro u hu
      have := hc u hu
      simp only [beq_iff_eq] at this ⊢
      exact hclear u this
    simp only [sliderAttackB] at hatt ⊢
    clear hb hb' hclear hW
    cases h1 : clearBetweenB q.board s t <;> cases h2 : clearBetweenB (withEmpty q W).board s t <;> simp_all
    all_goals tauto

/-- No pawn of theirs on the rank before promotion (our second rank, named from our side): they have no promotion. -/
theorem no_promotion (q : Position) (c : Color) (hq : q.turn = Color.opposite c)
    (h : count (fun u => kindSet q (Color.opposite c) PieceKind.pawn u && oset rankOf (some (named c ((0 : Fin 8), (1 : Fin 8)))) u) ≤ 0)
    (m : Move) (hm : m ∈ legalMoves q) : ∀ s t k, m ≠ Move.promotion s t k := by
  intro s t k hmk
  subst hmk
  have hps := legal_pseudo q _ hm
  have h0 := count_zero_not _ h s
  simp only [pseudoLegalB, pseudoLegalPromotionB] at hps
  cases hb : q.board s with
  | none => simp [hb] at hps
  | some q0 =>
    simp only [hb, Bool.and_eq_true, beq_iff_eq] at hps
    obtain ⟨⟨⟨hcol, hpawn⟩, htr⟩, hgeo⟩ := hps
    simp only [kindSet, hb, oset, rankOf, hcol, hpawn, hq, beq_self_eq_true, Bool.true_and, Bool.and_eq_false_iff, decide_eq_false_iff_not] at h0
    cases c <;> simp only [Color.opposite] at hq htr h0 hgeo <;> rw [hq] at htr hgeo <;>
      simp only [promotionRank, named, pawnMoveGeometryB, pawnAttackGeometryB, Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true] at htr hgeo h0 <;>
      (rcases hgeo with ⟨⟨_, h1⟩, _⟩ | ⟨⟨_, h1⟩, _⟩ <;> (simp only [Fin.ext_iff] at htr h0; (try simp (config := {decide := true}) at h1 h0); (try simp [Fin.rev] at h0); omega))

theorem newEnPassant_nonpawn (p : Position) (x y : Square) (q0 : Piece) (hx : p.board x = some q0) (hq : q0.kind ≠ PieceKind.pawn) :
    (applyMove p (Move.normal x y)).enPassant = none := by
  show newEnPassant p (Move.normal x y) = none
  unfold newEnPassant pawnDoubleTarget
  have : moveIsPawn p (Move.normal x y) = false := by
    unfold moveIsPawn movingPiece
    simp [Move.source, hx, hq]
  simp [this]

/-- THE ROOT (plan 42, w = 0). A piece of theirs stands on `y` and our non-pawn piece on `x` can legally take it; after the capture
nothing of ours (but the king) is attacked, they have no pawn on the rank before promotion, and whatever they reply the game is alive.
Then `forcingOf p c v` is nonzero for every `v` up to the worth of what was taken: the gain is forced in one move of ours and their reply.
Hypotheses are the words as `base_vocabulary.lean` renders them (`W` is our king, as for every mined statement that names it). -/
theorem hanging (p : Position) (c : Color) (v : Int) (W : Square) (hc : p.turn = c) (hW : kingSq p c = some W) (x y : Square)
    (hown : count (fun u => piecesOf p c u && oset atSq (some x) u) ≥ 1)
    (hnp : count (fun u => kindSet p c PieceKind.pawn u && oset atSq (some x) u) ≤ 0)
    (hmv : count (fun u => oset (movesOf p) (some x) u && oset atSq (some y) u) ≥ 1)
    (hth : count (fun u => piecesOf p (Color.opposite c) u && oset atSq (some y) u) ≥ 1)
    (hv : (worthOf p y : Int) ≥ v)
    (hhit : afterMove p (some x) (some y) (fun q => count (fun u => piecesOf q c u && attackedBy q (Color.opposite c) W u)) ≤ 0)
    (hpr : afterMove p (some x) (some y) (fun q =>
      count (fun u => kindSet q (Color.opposite c) PieceKind.pawn u && oset rankOf (some (named c ((0 : Fin 8), (1 : Fin 8)))) u)) ≤ 0)
    (halive : afterMove p (some x) (some y) (fun q => leastOver q (fun r => if deadDraw r then 0 else 1)) ≥ 1) :
    forcingOf p c v ≠ 0 := by
  obtain ⟨q0, hx, hqc, hqk⟩ := ours_at p c x hown
  have hqp := not_pawn_at p c x hnp q0 hx hqc
  obtain ⟨r, hy, hrc, -⟩ := ours_at p (Color.opposite c) y hth
  obtain ⟨m, hm, hms, hmd, hafter⟩ := after_of_moves p x y q0 hx (by rw [hqc, hc]) hqp (moves_of_count p x y hmv)
  have hmn : m = Move.normal x y := by
    have := move_is_normal p m hm q0 r (by rw [hms]; exact hx) hqp (by rw [hmd]; exact hy)
    rw [hms, hmd] at this; exact this
  subst hmn
  rw [hafter] at hhit hpr halive
  beta_reduce at hhit hpr halive
  have hxy : x ≠ y := by
    rintro rfl
    rw [hx] at hy
    have := Option.some.inj hy
    subst this
    exact absurd hrc (by rw [hqc]; cases c <;> simp [Color.opposite])
  have hrq : r.color ≠ q0.color := by rw [hrc, hqc]; cases c <;> simp [Color.opposite]
  obtain ⟨hm1, hm2⟩ := capture_gains p x y hxy q0 r hx hy hrq
  rw [hqc] at hm1
  rw [hrc] at hm2
  -- the position after our capture
  have hturn : (applyMove p (Move.normal x y)).turn = Color.opposite c := by rw [applyMove_turn, hc]
  have hep := newEnPassant_nonpawn p x y q0 hx hqp
  have hWb := kingSq_board p c W hW
  have hWx : W ≠ x := by
    intro h
    rw [h, hx] at hWb
    exact hqk (by rw [Option.some.inj hWb])
  have hWy : W ≠ y := by
    intro h
    rw [h, hy] at hWb
    have := congrArg Piece.color (Option.some.inj hWb)
    rw [hrc] at this
    revert this; cases c <;> simp [Color.opposite]
  obtain ⟨q, hqdef⟩ : ∃ q, applyMove p (Move.normal x y) = q := ⟨_, rfl⟩
  rw [hqdef] at hm1 hm2 hturn hep hhit halive hpr
  have hqb : q.board = boardMove p.board x y := by rw [← hqdef]; rfl
  have hqW : q.board W = some ⟨c, PieceKind.king⟩ := by
    rw [hqb]; simp [boardMove, boardUpdate, Function.update_apply, hWx, hWy, hWb]
  have hsafe : ∀ t r', q.board t = some r' → r'.color = c → r'.kind ≠ PieceKind.king → isAttackedB q (Color.opposite c) t = false := by
    intro t r' ht hrc' hk
    have h0 := count_zero_not _ hhit t
    have hp : piecesOf q c t = true := by
      unfold piecesOf; rw [ht]; simp [hrc', hk]
    rw [hp, Bool.true_and] at h0
    by_contra hcon
    simp only [Bool.not_eq_false] at hcon
    have := attacked_withEmpty q (Color.opposite c) W t (by
      intro pc hpc; rw [hqW] at hpc; have := Option.some.inj hpc; subst this; cases c <;> simp [Color.opposite]) hcon
    unfold attackedBy at h0
    rw [this] at h0
    exact absurd h0 (by simp)
  apply forcingOf_ne_zero p c v hc 1 (by simp)
  apply step_ours c _ p 1 hc _ hm
  rw [hqdef]
  have hturn' : q.turn ≠ c := by rw [hturn]; cases c <;> simp [Color.opposite]
  apply step_theirs c _ q 0 hturn' (least_pos_nonempty _ _ halive)
  intro rr hrr
  apply leaf
  have hnl := reply_no_loss q c hturn hep hsafe rr hrr (no_promotion q c hturn hpr rr hrr)
  apply gainGoal_of_material c _ v _ (by rw [← hqdef]; exact turn_after_two p c hc _ rr)
  · have hv' : (worthOf p y : Int) ≥ v := hv
    omega
  · have hd := least_floor _ _ 1 halive rr hrr
    beta_reduce at hd
    by_contra hcon
    simp only [Bool.not_eq_false] at hcon
    simp [hcon] at hd

end Ideas.Root
