import Chess.Game
import Ideas.InsufficientMaterial
import Ideas.Spec.NextEquiv

/-!
# THE DEAD POSITION OF ONE-COLOUR BISHOPS (ANALYST_PLAN 76.1 (b), extension (5))

FIDE 5.2.2: the game is drawn when no series of legal moves can end in a checkmate -- a DEAD POSITION (the kernel's `Chess.deadPosition`: no checkmate is
reachable). The rule is not state-local (it quantifies over every continuation), so the spec states it as a LEMMA: the `insufficient` clauses of
rules/chess.gdl (python-chess's criterion) are a SOUND test of it. `Ideas.InsufficientMaterial` proved that a king and ONE minor piece never mate in the position
itself; here the BISHOPS case of the criterion, for any number of bishops of either side on squares of one colour, and for EVERY CONTINUATION:

* `no_mate_one_colour`: kings and bishops on one colour only, one king each, the kings apart -- no checkmate. (The mated king would stand on the bishops'
  colour, which is the only colour they attack; its two orthogonal neighbours are of the other colour, so they fall to the other king alone, which then
  stands next to it: `Ideas.InsufficientMaterial`'s argument, every bishop of the board in the place of its one minor piece.)
* `one_colour_step`: a legal move keeps the class -- a bishop slides on its colour, a king steps, a capture removes, no pawn can promote, no castling has its
  rook; and after a legal move the mover's king is not attacked, so the kings stay apart.
* `one_colour_dead`: so the position is dead (`Chess.deadPosition`).

By reasoning on the colour classes (`Ideas.GeoLemmas.bishop_keeps_colour`, `orthAdj_other_colour`) and the kernel's definitions; no position is enumerated.
-/

namespace Ideas.DeadPosition

open Chess Ideas.GeoLemmas Ideas.InsufficientMaterial

/-- THE ONE-COLOUR-BISHOPS CLASS: every unit is a king, or a bishop on a square of colour `col`; each side has exactly one king; the kings stand apart. -/
def OneColour (p : Position) (col : Nat) : Prop :=
  (∀ s q, p.board s = some q → q.kind = .king ∨ (q.kind = .bishop ∧ colour s = col)) ∧
    (∀ c, ∃ k, ∀ s, p.board s = some ⟨c, .king⟩ ↔ s = k) ∧
    (∀ k w, p.board k = some ⟨p.turn, .king⟩ → p.board w = some ⟨Color.opposite p.turn, .king⟩ → kingGeometryB k w = false)

theorem opp_ne (c : Color) : Color.opposite c ≠ c := by cases c <;> simp [Color.opposite]

/-- An attacker on the board of the class: the other side's king, or a bishop of the colour class, attacking only its colour. -/
theorem attacker_class {p : Position} {col : Nat} (h : OneColour p col) {b : Board} (hb : ∀ s, b s = none ∨ b s = p.board s)
    {c : Color} {s t : Square} (q : Position) (hq : q.board = b) (ha : pieceAttacksB q c s t = true) :
    (b s = some ⟨c, .king⟩ ∧ kingGeometryB s t = true) ∨ colour t = col := by
  obtain ⟨pc, hpc, hcol⟩ := pieceAttacksB_source ha
  rw [hq] at hpc
  have hps : p.board s = some pc := by
    rcases hb s with h0 | h0
    · rw [h0] at hpc; cases hpc
    · rw [← h0]; exact hpc
  rcases h.1 s pc hps with hk | ⟨hk, hc⟩
  · left
    refine ⟨by rw [hpc, ← hcol, ← hk], ?_⟩
    unfold pieceAttacksB at ha
    rw [hq, hpc] at ha
    cases pc
    simp only at hk hcol
    subst hk hcol
    simpa using ha
  · right
    have hg := minor_attack q c .bishop (Or.inr rfl) s t (by rw [hq, hpc, ← hcol, ← hk]) ha
    simp only [reduceCtorEq, if_false] at hg
    rw [← bishop_keeps_colour s t hg, hc]

/-- KINGS AND ONE-COLOUR BISHOPS NEVER MATE. -/
theorem no_mate_one_colour (p : Position) (col : Nat) (h : OneColour p col) : ¬ checkmate p := by
  intro hmate
  obtain ⟨k, hK⟩ := h.2.1 p.turn
  obtain ⟨w, hW⟩ := h.2.1 (Color.opposite p.turn)
  have hk : p.board k = some ⟨p.turn, .king⟩ := (hK k).2 rfl
  have hw : p.board w = some ⟨Color.opposite p.turn, .king⟩ := (hW w).2 rfl
  have hfar : kingGeometryB k w = false := h.2.2 k w hk hw
  have hkw : k ≠ w := by
    rintro rfl
    rw [hk] at hw
    simp only [Option.some.injEq, Piece.mk.injEq] at hw
    exact opp_ne _ hw.1.symm
  -- 1 the check is a bishop's: the mated king stands on the bishops' colour
  have hchk := checkmate_attacked p k hK hmate
  simp only [isAttackedB, List.any_eq_true] at hchk
  obtain ⟨s, _, hs⟩ := hchk
  have hcolk : colour k = col := by
    rcases attacker_class h (b := p.board) (fun _ => Or.inr rfl) p rfl hs with ⟨hsk, hg⟩ | hc
    · have : s = w := (hW s).1 hsk
      subst this
      rw [Passer.kingGeometryB_comm, hfar] at hg
      cases hg
    · exact hc
  -- 2 two orthogonal neighbours, each guarded by the other king alone
  obtain ⟨n1, n2, hne, ho1, ho2⟩ := two_orth k
  have guard : ∀ n, orthAdj k n → kingGeometryB w n = true := by
    intro n hn
    have hgeo : kingGeometryB k n = true := by
      rw [orth_iff] at hn
      rw [kg_iff]
      omega
    have hcoln : colour n ≠ col := by rw [← hcolk]; exact (orthAdj_other_colour k n hn).symm
    have hnk : n ≠ k := by rintro rfl; simp [kingGeometryB] at hgeo
    have hnw : n ≠ w := by rintro rfl; rw [hgeo] at hfar; cases hfar
    have hempty : p.board n = none := by
      cases hb : p.board n with
      | none => rfl
      | some q =>
          rcases h.1 n q hb with hq | ⟨_, hc⟩
          · exfalso
            cases q with
            | mk qc qk =>
                simp only at hq
                subst hq
                cases qc <;> cases ht : p.turn <;> simp_all [Color.opposite]
          · exact absurd hc hcoln
    have hfl := checkmate_flight p k hK hmate n
    simp only [Ideas.flightOkB, hgeo, Bool.not_true, Bool.false_or, destinationFriendlyB, hempty, Bool.not_true, isAttackedB,
      List.any_eq_true] at hfl
    obtain ⟨s', _, hs'⟩ := hfl
    have hb' : ∀ x, (Ideas.withEmpty p k).board x = none ∨ (Ideas.withEmpty p k).board x = p.board x := by
      intro x
      by_cases hx : x = k
      · left; simp [Ideas.withEmpty, boardUpdate, hx]
      · right; simp [Ideas.withEmpty, boardUpdate, hx]
    rcases attacker_class h (b := (Ideas.withEmpty p k).board) hb' (Ideas.withEmpty p k) rfl hs' with ⟨hsk, hg⟩ | hc
    · have hs'k : s' ≠ k := by
        rintro rfl
        simp [Ideas.withEmpty, boardUpdate] at hsk
      have : s' = w := by
        apply (hW s').1
        simpa [Ideas.withEmpty, boardUpdate, hs'k] using hsk
      subst this
      exact hg
    · exact absurd hc hcoln
  rcases king_guards_two_orth k w n1 n2 ho1 ho2 hne (guard n1 ho1) (guard n2 ho2) with e | e
  · exact hkw e.symm
  · rw [e] at hfar
    cases hfar

/-! ## A legal move keeps the class -/

/-- A pseudo-legal normal move: a unit of the side to move on `s`, its geometry, and a landing square holding no unit of its own side and no king. -/
theorem normal_pl' {p : Position} {s t : Square} (h : pseudoLegal p (.normal s t)) :
    ∃ q, p.board s = some q ∧ q.color = p.turn ∧ normalPieceGeometryB p p.turn q s t = true ∧ destinationFriendlyB p p.turn t = true := by
  simp only [pseudoLegal, pseudoLegalB, normalPseudoLegalB] at h
  cases hb : p.board s with
  | none => rw [hb] at h; cases h
  | some q =>
      rw [hb] at h
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      exact ⟨q, rfl, h.1.1.1, h.2, h.1.1.2⟩

theorem not_king_of_friendly {p : Position} {c : Color} {t : Square} (h : destinationFriendlyB p c t = true) (c' : Color) :
    p.board t ≠ some ⟨c', .king⟩ := by
  intro e
  simp [destinationFriendlyB, e] at h

/-- After a unit's move from `s` to `t` (no king taken), each side still has exactly one king. -/
theorem kings_after_boardMove {b : Board} {s t : Square} {q : Piece} (hs : b s = some q) (ht : ∀ c', b t ≠ some ⟨c', .king⟩) (c : Color)
    (hk : ∃ k, ∀ x, b x = some ⟨c, .king⟩ ↔ x = k) : ∃ k', ∀ x, boardMove b s t x = some ⟨c, .king⟩ ↔ x = k' := by
  obtain ⟨k, hK⟩ := hk
  simp only [Ideas.NextEquiv.boardMove_apply]
  by_cases hq : q = ⟨c, .king⟩
  · subst hq
    have hsk : s = k := (hK s).1 hs
    refine ⟨t, fun x => ?_⟩
    by_cases hxt : x = t
    · simp [hxt, hs]
    · rw [if_neg hxt]
      by_cases hxs : x = s
      · rw [if_pos hxs]
        exact ⟨fun e => (by cases e), fun e => absurd e hxt⟩
      · rw [if_neg hxs, hK x]
        exact ⟨fun e => absurd (e.trans hsk.symm) hxs, fun e => absurd e hxt⟩
  · refine ⟨k, fun x => ?_⟩
    by_cases hxt : x = t
    · rw [if_pos hxt, hs]
      refine ⟨fun e => absurd (Option.some.inj e) hq, fun e => ?_⟩
      rw [hxt] at e
      exact absurd ((hK t).2 e) (ht c)
    · rw [if_neg hxt]
      by_cases hxs : x = s
      · rw [if_pos hxs]
        refine ⟨fun e => (by cases e), fun e => ?_⟩
        have := (hK k).2 rfl
        rw [← e, hxs, hs] at this
        exact absurd (Option.some.inj this) hq
      · rw [if_neg hxs]
        exact hK x

/-- After a legal move the kings stand apart: the mover's king is not attacked, and the other king attacks every square next to it. -/
theorem apart_after {p : Position} {m : Move} (hm : legalMove p m) (hK : ∃ k, ∀ x, (applyMove p m).board x = some ⟨p.turn, .king⟩ ↔ x = k) :
    ∀ k w, (applyMove p m).board k = some ⟨(applyMove p m).turn, .king⟩ → (applyMove p m).board w = some ⟨Color.opposite (applyMove p m).turn, .king⟩ →
      kingGeometryB k w = false := by
  intro k w hk hw
  obtain ⟨w0, hW⟩ := hK
  rw [applyMove_turn, Color.opposite_opposite] at hw
  rw [applyMove_turn] at hk
  have hww : w = w0 := (hW w).1 hw
  subst hww
  have hsafe := legalMove_kingSafe hm
  simp only [inCheck, inCheckB, kingSquare_of_only hW, isAttackedB, List.any_eq_true, not_exists, not_and] at hsafe
  have := hsafe k (mem_allSquares k)
  unfold pieceAttacksB at this
  rw [hk] at this
  simpa using this

/-- A LEGAL MOVE KEEPS THE CLASS. -/
theorem one_colour_step {p : Position} {col : Nat} (h : OneColour p col) {m : Move} (hm : legalMove p m) : OneColour (applyMove p m) col := by
  have hpl := legalMove_pseudoLegal hm
  cases m with
  | normal s t =>
      obtain ⟨q, hq, _, hg, hd⟩ := normal_pl' hpl
      have hkings : ∀ c, ∃ k', ∀ x, (applyMove p (.normal s t)).board x = some ⟨c, .king⟩ ↔ x = k' := fun c =>
        kings_after_boardMove hq (not_king_of_friendly hd) c (h.2.1 c)
      refine ⟨?_, hkings, apart_after hm (hkings p.turn)⟩
      intro x q' hx
      rw [applyMove_board] at hx
      simp only [applyBoard, Ideas.NextEquiv.boardMove_apply] at hx
      by_cases hxt : x = t
      · rw [if_pos hxt, hq] at hx
        cases hx
        rcases h.1 s q hq with hk | ⟨hk, hc⟩
        · exact Or.inl hk
        · refine Or.inr ⟨hk, ?_⟩
          simp only [normalPieceGeometryB, hk, Bool.and_eq_true] at hg
          rw [hxt, ← bishop_keeps_colour s t hg.1, hc]
      · rw [if_neg hxt] at hx
        by_cases hxs : x = s
        · rw [if_pos hxs] at hx
          cases hx
        · rw [if_neg hxs] at hx
          exact h.1 x q' hx
  | promotion s t k =>
      have := Ideas.NextEquiv.promotion_pl hpl
      rcases h.1 s _ this with hk | ⟨hk, _⟩ <;> cases hk
  | enPassant s t =>
      have := (Ideas.NextEquiv.ep_pl hpl).2.1
      rcases h.1 s _ this with hk | ⟨hk, _⟩ <;> cases hk
  | whiteKingCastle =>
      have := (Ideas.NextEquiv.castle_pl (c := .white) (w := .kingside) hpl).2.2.2.1
      rcases h.1 _ _ this with hk | ⟨hk, _⟩ <;> cases hk
  | whiteQueenCastle =>
      have := (Ideas.NextEquiv.castle_pl (c := .white) (w := .queenside) hpl).2.2.2.1
      rcases h.1 _ _ this with hk | ⟨hk, _⟩ <;> cases hk
  | blackKingCastle =>
      have := (Ideas.NextEquiv.castle_pl (c := .black) (w := .kingside) hpl).2.2.2.1
      rcases h.1 _ _ this with hk | ⟨hk, _⟩ <;> cases hk
  | blackQueenCastle =>
      have := (Ideas.NextEquiv.castle_pl (c := .black) (w := .queenside) hpl).2.2.2.1
      rcases h.1 _ _ this with hk | ⟨hk, _⟩ <;> cases hk

/-- Every position reachable from the class is in it. -/
theorem one_colour_reach {p q : Position} {col : Nat} (h : OneColour p col) (hr : reachableFrom p q) : OneColour q col := by
  induction hr with
  | refl => exact h
  | tail _ hst ih =>
      obtain ⟨m, hm, rfl⟩ := hst
      exact one_colour_step ih hm

/-- THE DEAD POSITION (FIDE 5.2.2), the one-colour-bishops case: kings and bishops on squares of one colour, one king each, the kings apart -- no series
of legal moves ends in a checkmate. -/
theorem one_colour_dead (p : Position) (col : Nat) (h : OneColour p col) : deadPosition p := by
  rintro ⟨q, hr, hm⟩
  exact no_mate_one_colour q col (one_colour_reach h hr) hm

/-! ## The spec's criterion: its draw, with no knight on the board, is a dead position -/

/-- The parity a number has. -/
def parityOf (n : Nat) : Ideas.Spec.Chess.Parity := if n % 2 = 0 then .even else .odd

theorem fileParity_iff (f : Fin 8) (q : Ideas.Spec.Chess.Parity) : Ideas.Spec.Chess.fileParity f q ↔ q = parityOf f.val := by
  constructor
  · intro h
    cases h <;> rfl
  · rintro rfl
    obtain ⟨f, hf⟩ := f
    interval_cases f <;> simp only [parityOf, Nat.reduceMod, ↓reduceIte, one_ne_zero] <;> constructor

theorem rankParity_iff (r : Fin 8) (q : Ideas.Spec.Chess.Parity) : Ideas.Spec.Chess.rankParity r q ↔ q = parityOf r.val := by
  constructor
  · intro h
    cases h <;> rfl
  · rintro rfl
    obtain ⟨r, hr⟩ := r
    interval_cases r <;> simp only [parityOf, Nat.reduceMod, ↓reduceIte, one_ne_zero] <;> constructor

theorem parityOf_eq (a b : Nat) : parityOf a = parityOf b ↔ a % 2 = b % 2 := by
  unfold parityOf
  split_ifs <;> simp <;> omega

/-- The spec's dark squares are the kernel's colour class 0 (a1 dark). -/
theorem dark_iff (f r : Fin 8) : Ideas.Spec.Chess.dark f r ↔ colour (f, r) = 0 := by
  constructor
  · rintro ⟨h1, h2⟩
    rw [fileParity_iff] at h1
    rw [rankParity_iff] at h2
    have := (parityOf_eq _ _).1 (h1.symm.trans h2)
    simp only [colour]
    omega
  · intro h
    simp only [colour] at h
    exact Ideas.Spec.Chess.dark.c1 ((fileParity_iff _ _).2 rfl) ((rankParity_iff _ _).2 ((parityOf_eq _ _).2 (by omega)).symm)

theorem light_iff (f r : Fin 8) : Ideas.Spec.Chess.light f r ↔ colour (f, r) = 1 := by
  constructor
  · rintro ⟨h1, h2, hne⟩
    rw [fileParity_iff] at h1
    rw [rankParity_iff] at h2
    subst h1 h2
    have : ¬ f.val % 2 = r.val % 2 := fun e => hne ((parityOf_eq _ _).2 e)
    simp only [colour]
    omega
  · intro h
    simp only [colour] at h
    refine Ideas.Spec.Chess.light.c1 ((fileParity_iff _ _).2 rfl) ((rankParity_iff _ _).2 rfl) ?_
    intro e
    have := (parityOf_eq _ _).1 e
    omega

/-- No mating material, no knight, no bishops of both colours: every unit is a king or a bishop of one colour class. -/
theorem units_of_criterion (p : Position) (hm : ∀ c, ¬ Ideas.Spec.Chess.matingMaterial p c) (hn : ¬ Ideas.Spec.Chess.anyKnight p)
    (hmix : ¬ Ideas.Spec.Chess.mixedBishops p) : ∃ col, ∀ s q, p.board s = some q → q.kind = .king ∨ (q.kind = .bishop ∧ colour s = col) := by
  have hlt : ∀ s : Square, colour s < 2 := fun s => Nat.mod_lt _ (by omega)
  have other : ∀ s q, p.board s = some q → q.kind = .king ∨ q.kind = .bishop := by
    rintro ⟨f, r⟩ ⟨c, k⟩ hb
    cases k
    · exact Or.inl rfl
    · exact absurd (Ideas.Spec.Chess.matingMaterial.c3 hb) (hm c)
    · exact absurd (Ideas.Spec.Chess.matingMaterial.c2 hb) (hm c)
    · exact Or.inr rfl
    · exact absurd (Ideas.Spec.Chess.anyKnight.c1 hb) hn
    · exact absurd (Ideas.Spec.Chess.matingMaterial.c1 hb) (hm c)
  by_cases hd : ∃ (s : Square) (c : Color), p.board s = some ⟨c, .bishop⟩ ∧ colour s = 0
  · obtain ⟨⟨f0, r0⟩, c0, hb0, hc0⟩ := hd
    refine ⟨0, fun s q hb => ?_⟩
    rcases other s q hb with hk | hk
    · exact Or.inl hk
    · refine Or.inr ⟨hk, ?_⟩
      by_contra hne
      have h1 : colour s = 1 := by have := hlt s; omega
      obtain ⟨f, r⟩ := s
      cases q
      simp only at hk
      subst hk
      exact hmix (Ideas.Spec.Chess.mixedBishops.c1 hb0 ((dark_iff _ _).2 hc0) hb ((light_iff _ _).2 h1))
  · refine ⟨1, fun s q hb => ?_⟩
    rcases other s q hb with hk | hk
    · exact Or.inl hk
    · refine Or.inr ⟨hk, ?_⟩
      by_contra hne
      have h0 : colour s = 0 := by have := hlt s; omega
      cases q
      simp only at hk
      subst hk
      exact hd ⟨s, _, hb, h0⟩

/-- THE SPEC'S DRAW WITH NO KNIGHT ON THE BOARD IS A DEAD POSITION: when the spec's insufficient-material `draw` holds and no knight stands on the board
(both sides insufficient by the bishops clause or the bare king's), in a position with one king each and the kings apart, no checkmate is reachable. The
criterion's knight clause (a king and a knight against kings and queens) is not covered here: `Ideas.InsufficientMaterial` proves it in the position itself. -/
theorem dead_of_spec_draw (p : Position) (hk : ∀ c, ∃ k, ∀ s, p.board s = some ⟨c, .king⟩ ↔ s = k)
    (hfar : ∀ k w, p.board k = some ⟨p.turn, .king⟩ → p.board w = some ⟨Color.opposite p.turn, .king⟩ → kingGeometryB k w = false)
    (hd : Ideas.Spec.Chess.draw p) (hn : ¬ Ideas.Spec.Chess.anyKnight p) : deadPosition p := by
  obtain ⟨hw, hb⟩ := hd
  have side : ∀ c, Ideas.Spec.Chess.insufficient p c → ¬ Ideas.Spec.Chess.matingMaterial p c ∧
      (Ideas.Spec.Chess.hasKind p c .bishop → ¬ Ideas.Spec.Chess.mixedBishops p) := by
    intro c h
    cases h with
    | c1 _ hm hknight _ _ _ =>
        obtain ⟨hb⟩ := hknight
        exact absurd (Ideas.Spec.Chess.anyKnight.c1 hb) hn
    | c2 _ hm _ _ hmix _ _ => exact ⟨hm, fun _ => hmix⟩
    | c3 _ hm _ hnb => exact ⟨hm, fun hb => absurd hb hnb⟩
  have hm : ∀ c, ¬ Ideas.Spec.Chess.matingMaterial p c := by
    intro c
    cases c
    · exact (side _ hw).1
    · exact (side _ hb).1
  have hmix : ¬ Ideas.Spec.Chess.mixedBishops p := by
    intro hx
    have hx' := hx
    rcases hx with @⟨f1, r1, c1, f2, r2, c2, h1, _, _, _⟩
    have hkb : Ideas.Spec.Chess.hasKind p c1 .bishop := Ideas.Spec.Chess.hasKind.c1 h1
    cases c1
    · exact (side _ hw).2 hkb hx'
    · exact (side _ hb).2 hkb hx'
  obtain ⟨col, hu⟩ := units_of_criterion p hm hn hmix
  exact one_colour_dead p col ⟨hu, hk, hfar⟩

end Ideas.DeadPosition
