import Chess.Game

/-!
# A check that cannot be met is mate

The reusable core under every mate lemma. In the position `q` the side to move (the defender)
is in check from the attacker's piece on `c`; its king stands on `k`. If

* every square next to the king is held by the defender's own piece or attacked by the
  attacker once the king is lifted off the board (`flightOk`),
* no defender's piece but the king attacks `c` (`noCaptureB`, pins ignored), and
* no defender's piece can step onto a square between `c` and `k` (`noBlockB`, pawn pushes
  counted without checking their path),

then the defender is checkmated. Every condition over-approximates what the defender can do,
so the hypotheses are stronger than needed, never weaker: a detector built on them can miss a
mate, it never claims a false one.
-/

namespace Ideas

open Chess

/-- `q` with square `s` emptied: the defender's king lifted off the board, so squares it could
flee to are judged with the attacker's lines running through its old square. -/
def withEmpty (q : Position) (s : Square) : Position :=
  { q with board := boardUpdate q.board s none }

/-- The colour-`c` king stands on `k` and nowhere else. -/
def kingOnlyAtB (b : Board) (c : Color) (k : Square) : Bool :=
  allSquares.all (fun s => decide (b s = some ⟨c, .king⟩) == decide (s = k))

/-- The square `d` is no flight for the king on `k`: not a king step, or held by the defender's
own piece, or attacked by the attacker with the king lifted off the board. -/
def flightOkB (q : Position) (k d : Square) : Bool :=
  !kingGeometryB k d || !destinationFriendlyB q q.turn d ||
    isAttackedB (withEmpty q k) (Color.opposite q.turn) d

/-- No square next to the king is a flight. -/
def flightsCoveredB (q : Position) (k : Square) : Bool :=
  allSquares.all (fun d => flightOkB q k d)

/-- The attacker's piece on `c` attacks the king on `k`. -/
def checksB (q : Position) (k c : Square) : Bool :=
  pieceAttacksB q (Color.opposite q.turn) c k

/-- No defender's piece but the king attacks `c` (pins ignored: an over-approximation). -/
def noCaptureB (q : Position) (k c : Square) : Bool :=
  allSquares.all (fun s => decide (s = k) || !pieceAttacksB q q.turn s c)

/-- The defender's piece on `s` could step onto the empty square `u`: a pawn by a single or
double push (path not checked), any other piece but the king along its attack lines. -/
def couldReachEmptyB (q : Position) (s u : Square) : Bool :=
  match q.board s with
  | some pc =>
      (pc.color == q.turn) && (pc.kind != .king) &&
        (if pc.kind = .pawn then
          pawnMoveGeometryB q.turn s u || pawnDoubleGeometryB q.turn s u
        else pieceAttacksB q q.turn s u)
  | none => false

/-- No defender's piece can step onto a square between `c` and `k`. -/
def noBlockB (q : Position) (k c : Square) : Bool :=
  (squaresBetween c k).all (fun u => allSquares.all (fun s => !couldReachEmptyB q s u))

/-! ## Board facts -/

theorem kingOnlyAtB_iff {b : Board} {c : Color} {k : Square} :
    kingOnlyAtB b c k = true ↔ ∀ s, (b s = some ⟨c, .king⟩ ↔ s = k) := by
  simp [kingOnlyAtB, List.all_eq_true]

theorem kingSquare_of_only {b : Board} {c : Color} {k : Square}
    (h : ∀ s, (b s = some ⟨c, .king⟩ ↔ s = k)) : kingSquare b c = some k := by
  unfold kingSquare
  cases hf : allSquares.find? (fun s => decide (b s = some ⟨c, .king⟩)) with
  | none =>
      rw [List.find?_eq_none] at hf
      exact absurd ((h k).2 rfl) (by simpa using hf k (mem_allSquares k))
  | some x =>
      have hx := List.find?_some hf
      simp only [decide_eq_true_eq] at hx
      rw [(h x).1 hx]

theorem pieceAttacksB_ne {p : Position} {c : Color} {s t : Square}
    (h : pieceAttacksB p c s t = true) : s ≠ t := by
  intro hst
  subst hst
  unfold pieceAttacksB at h
  split at h
  · simp [rookGeometryB, bishopGeometryB, sameDiagonalB, queenGeometryB, knightGeometryB,
      kingGeometryB, pawnAttackGeometryB, absDiff] at h
  · simp at h

theorem pieceAttacksB_source {p : Position} {c : Color} {s t : Square}
    (h : pieceAttacksB p c s t = true) : ∃ pc, p.board s = some pc ∧ pc.color = c := by
  unfold pieceAttacksB at h
  split at h
  · rename_i pc hpc
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    exact ⟨pc, hpc, h.1⟩
  · simp at h

theorem clearBetweenB_mono {b b' : Board} {s t : Square} (h : clearBetweenB b s t = true)
    (hb : ∀ u, betweenB s u t = true → b u = none → b' u = none) :
    clearBetweenB b' s t = true := by
  simp only [clearBetweenB, squaresBetween, List.all_eq_true, List.mem_filter] at h ⊢
  intro u hu
  have hbu := h u hu
  simp only [beq_iff_eq] at hbu ⊢
  exact hb u hu.2 hbu

/-- An attack survives any change of the board that keeps the attacker on its square and keeps
empty every square between that was empty. -/
theorem pieceAttacksB_mono {p p' : Position} {c : Color} {s t : Square}
    (h : pieceAttacksB p c s t = true) (hs : p'.board s = p.board s)
    (hb : ∀ u, betweenB s u t = true → p.board u = none → p'.board u = none) :
    pieceAttacksB p' c s t = true := by
  have hc : clearBetweenB p.board s t = true → clearBetweenB p'.board s t = true :=
    fun h' => clearBetweenB_mono h' hb
  unfold pieceAttacksB sliderAttackB at *
  rw [hs]
  cases hq : p.board s with
  | none => simp [hq] at h
  | some pc =>
      simp only [hq, Bool.and_eq_true, Bool.or_eq_true] at h ⊢
      obtain ⟨hcol, h⟩ := h
      refine ⟨hcol, ?_⟩
      rcases h with (((((⟨hk, hcl⟩ | ⟨hk, hcl⟩) | ⟨hk, hcl⟩) | h) | h) | h)
      · exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl ⟨hk, hc hcl⟩))))
      · exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inr ⟨hk, hc hcl⟩))))
      · exact Or.inl (Or.inl (Or.inl (Or.inr ⟨hk, hc hcl⟩)))
      · exact Or.inl (Or.inl (Or.inr h))
      · exact Or.inl (Or.inr h)
      · exact Or.inr h

/-! ## Moves of the defender -/

theorem capture_geom_attacks {p : Position} {s t : Square} {pc x : Piece}
    (hs : p.board s = some pc) (hcol : pc.color = p.turn) (hk : pc.kind ≠ .king)
    (ht : p.board t = some x) (hg : normalPieceGeometryB p p.turn pc s t = true) :
    pieceAttacksB p p.turn s t = true := by
  obtain ⟨col, kind⟩ := pc
  simp only at hcol hk
  subst hcol
  unfold pieceAttacksB sliderAttackB
  rw [hs]
  cases kind <;> simp_all [normalPieceGeometryB]

theorem empty_geom_reaches {p : Position} {s t : Square} {pc : Piece}
    (hs : p.board s = some pc) (hcol : pc.color = p.turn) (hk : pc.kind ≠ .king)
    (ht : p.board t = none) (hg : normalPieceGeometryB p p.turn pc s t = true) :
    couldReachEmptyB p s t = true := by
  obtain ⟨col, kind⟩ := pc
  simp only at hcol hk
  subst hcol
  unfold couldReachEmptyB pieceAttacksB sliderAttackB
  rw [hs]
  cases kind <;> simp_all [normalPieceGeometryB]
  tauto

theorem not_noBlock {q : Position} {k c s u : Square} (hBlock : noBlockB q k c = true)
    (hu : betweenB c u k = true) (hr : couldReachEmptyB q s u = true) : False := by
  have hmem : u ∈ squaresBetween c k := by simp [squaresBetween, hu]
  have h1 := (List.all_eq_true.1 hBlock) u hmem
  have h2 := (List.all_eq_true.1 h1) s (mem_allSquares s)
  simp [hr] at h2

theorem not_noCapture {q : Position} {k c s : Square} (hCapture : noCaptureB q k c = true)
    (hsk : s ≠ k) (ha : pieceAttacksB q q.turn s c = true) : False := by
  have h := (List.all_eq_true.1 hCapture) s (mem_allSquares s)
  simp [hsk, ha] at h

theorem attacked_of_checks {q : Position} {k c : Square} (hCheck : checksB q k c = true) :
    isAttackedB q (Color.opposite q.turn) k = true := by
  simp only [isAttackedB, List.any_eq_true]
  exact ⟨c, mem_allSquares c, hCheck⟩

/-- The checker's colour is not the defender's. -/
theorem checker_ne_defender {q : Position} {k c s : Square} {pc : Piece}
    (hCheck : checksB q k c = true) (hs : q.board s = some pc) (hcol : pc.color = q.turn) :
    c ≠ s := by
  rintro rfl
  obtain ⟨cpc, hcpc, hccol⟩ := pieceAttacksB_source hCheck
  rw [hs] at hcpc
  cases hcpc
  rw [hcol] at hccol
  cases h : q.turn <;> rw [h] at hccol <;> simp [Color.opposite] at hccol

/-- A king step leaves the king in check: every flight is held or covered. -/
theorem king_step_in_check {q : Position} {k : Square}
    (hK : ∀ s, (q.board s = some ⟨q.turn, .king⟩ ↔ s = k))
    (hF : ∀ d, flightOkB q k d = true) (t : Square)
    (hpl : normalPseudoLegalB q k t = true) :
    inCheckB (applyMove q (.normal k t)) q.turn = true := by
  have hqk : q.board k = some ⟨q.turn, .king⟩ := (hK k).2 rfl
  unfold normalPseudoLegalB at hpl
  rw [hqk] at hpl
  simp only [normalPieceGeometryB, Bool.and_eq_true, beq_self_eq_true, true_and] at hpl
  obtain ⟨⟨hfr, _⟩, hgeo⟩ := hpl
  have hf := hF t
  simp only [flightOkB, hgeo, hfr, Bool.not_true, Bool.false_or] at hf
  have hkt : k ≠ t := by
    rintro rfl
    simp [kingGeometryB] at hgeo
  have hboard : (applyMove q (.normal k t)).board = boardMove q.board k t := rfl
  have hK' : ∀ x, ((applyMove q (.normal k t)).board x = some ⟨q.turn, .king⟩ ↔ x = t) := by
    intro x
    rw [hboard]
    simp only [boardMove, boardUpdate, Function.update_apply]
    by_cases hxt : x = t
    · subst hxt
      simp [hqk]
    · by_cases hxk : x = k
      · subst hxk
        simp [hxt]
      · simp [hxt, hxk, hK x]
  unfold inCheckB
  rw [kingSquare_of_only hK']
  simp only [isAttackedB, List.any_eq_true] at hf ⊢
  obtain ⟨s', _, hs'⟩ := hf
  refine ⟨s', mem_allSquares s', ?_⟩
  have hs't : s' ≠ t := pieceAttacksB_ne hs'
  apply pieceAttacksB_mono hs'
  · rw [hboard]
    simp [withEmpty, boardMove, boardUpdate, Function.update_apply, hs't]
  · intro u hu hqu
    have hut : u ≠ t := between_not_destination hu
    rw [hboard]
    simp only [withEmpty, boardMove, boardUpdate, Function.update_apply] at hqu ⊢
    simp only [hut, if_false]
    exact hqu

/-- Any other ordinary move leaves the king in check: it cannot take the checker, cannot step
between, and does not move the king. -/
theorem other_move_in_check {q : Position} {k c : Square}
    (hK : ∀ s, (q.board s = some ⟨q.turn, .king⟩ ↔ s = k))
    (hCheck : checksB q k c = true) (hCapture : noCaptureB q k c = true)
    (hBlock : noBlockB q k c = true) (s t : Square) (hsk : s ≠ k)
    (hpl : normalPseudoLegalB q s t = true) :
    inCheckB (applyMove q (.normal s t)) q.turn = true := by
  unfold normalPseudoLegalB at hpl
  cases hqs : q.board s with
  | none => simp [hqs] at hpl
  | some pc =>
    rw [hqs] at hpl
    simp only [Bool.and_eq_true, beq_iff_eq] at hpl
    obtain ⟨⟨⟨hcol, hfr⟩, _⟩, hgeo⟩ := hpl
    have hpk : pc.kind ≠ .king := by
      intro hk
      apply hsk
      apply (hK s).1
      rw [hqs]
      obtain ⟨col, kind⟩ := pc
      simp only at hcol hk
      rw [hcol, hk]
    have htk : t ≠ k := by
      rintro rfl
      have hq := (hK t).2 rfl
      simp [destinationFriendlyB, hq] at hfr
    have hboard : (applyMove q (.normal s t)).board = boardMove q.board s t := rfl
    have hK' : ∀ x, ((applyMove q (.normal s t)).board x = some ⟨q.turn, .king⟩ ↔ x = k) := by
      intro x
      rw [hboard]
      simp only [boardMove, boardUpdate, Function.update_apply]
      by_cases hxt : x = t
      · subst hxt
        rw [if_pos rfl, hqs]
        constructor
        · intro h
          cases h
          exact absurd rfl hpk
        · intro h
          exact absurd h htk
      · by_cases hxs : x = s
        · subst hxs
          simp [hxt, hsk]
        · simp [hxt, hxs, hK x]
    have hcs : c ≠ s := checker_ne_defender hCheck hqs hcol
    have hct : c ≠ t := by
      rintro rfl
      obtain ⟨cpc, hcpc, _⟩ := pieceAttacksB_source hCheck
      exact not_noCapture hCapture hsk (capture_geom_attacks hqs hcol hpk hcpc hgeo)
    unfold inCheckB
    rw [kingSquare_of_only hK']
    simp only [isAttackedB, List.any_eq_true]
    refine ⟨c, mem_allSquares c, ?_⟩
    apply pieceAttacksB_mono hCheck
    · rw [hboard]
      simp [boardMove, boardUpdate, hcs, hct]
    · intro u hu hqu
      rw [hboard]
      simp only [boardMove, boardUpdate, Function.update_apply]
      by_cases hut : u = t
      · subst hut
        exact (not_noBlock hBlock hu (empty_geom_reaches hqs hcol hpk hqu hgeo)).elim
      · by_cases hus : u = s
        · subst hus
          simp [hut]
        · simp [hut, hus, hqu]

/-- A promotion leaves the king in check: the pawn cannot take the checker or step between. -/
theorem promotion_in_check {q : Position} {k c : Square}
    (hK : ∀ s, (q.board s = some ⟨q.turn, .king⟩ ↔ s = k))
    (hCheck : checksB q k c = true) (hCapture : noCaptureB q k c = true)
    (hBlock : noBlockB q k c = true) (s t : Square) (pk : PromotionKind)
    (hpl : pseudoLegalPromotionB q (.promotion s t pk) = true) :
    inCheckB (applyMove q (.promotion s t pk)) q.turn = true := by
  simp only [pseudoLegalPromotionB] at hpl
  cases hqs : q.board s with
  | none => simp [hqs] at hpl
  | some pc =>
    simp only [hqs, Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true] at hpl
    obtain ⟨⟨⟨hcol, hpawn⟩, _⟩, hgeo⟩ := hpl
    have hsk : s ≠ k := by
      rintro rfl
      have hq := (hK s).2 rfl
      rw [hqs] at hq
      cases hq
      simp at hpawn
    have htk : t ≠ k := by
      rintro rfl
      have hq := (hK t).2 rfl
      rw [hq] at hgeo
      simp at hgeo
    have hboard : (applyMove q (.promotion s t pk)).board =
        boardUpdate (boardUpdate q.board s none) t (some ⟨q.turn, pk.toPieceKind⟩) := rfl
    have hK' : ∀ x,
        ((applyMove q (.promotion s t pk)).board x = some ⟨q.turn, .king⟩ ↔ x = k) := by
      intro x
      rw [hboard]
      simp only [boardUpdate, Function.update_apply]
      by_cases hxt : x = t
      · subst hxt
        cases pk <;> simp [PromotionKind.toPieceKind, htk]
      · by_cases hxs : x = s
        · subst hxs
          simp [hxt, hsk]
        · simp [hxt, hxs, hK x]
    have hattack : pawnAttackGeometryB q.turn s c = true → pieceAttacksB q q.turn s c = true := by
      intro hg
      unfold pieceAttacksB
      rw [hqs]
      simp [hcol, hpawn, hg]
    have hcs : c ≠ s := checker_ne_defender hCheck hqs hcol
    have hct : c ≠ t := by
      rintro rfl
      obtain ⟨cpc, hcpc, _⟩ := pieceAttacksB_source hCheck
      rw [hcpc] at hgeo
      simp only [reduceCtorEq, and_false, false_or] at hgeo
      exact not_noCapture hCapture hsk (hattack hgeo.1)
    unfold inCheckB
    rw [kingSquare_of_only hK']
    simp only [isAttackedB, List.any_eq_true]
    refine ⟨c, mem_allSquares c, ?_⟩
    apply pieceAttacksB_mono hCheck
    · rw [hboard]
      simp [boardUpdate, hcs, hct]
    · intro u hu hqu
      rw [hboard]
      simp only [boardUpdate, Function.update_apply]
      by_cases hut : u = t
      · subst hut
        rw [hqu] at hgeo
        simp only [and_true, Bool.false_eq_true, and_false, or_false] at hgeo
        have hr : couldReachEmptyB q s u = true := by
          unfold couldReachEmptyB
          rw [hqs]
          simp [hcol, hpawn, hgeo]
        exact (not_noBlock hBlock hu hr).elim
      · by_cases hus : u = s
        · subst hus
          simp [hut]
        · simp [hut, hus, hqu]

/-- A castling move starts from its king square. -/
theorem castle_source_king {q : Position} {m : Move} (hm : m.isCastle = true)
    (hpl : pseudoLegalCastleB q m = true) :
    q.board m.source = some ⟨q.turn, .king⟩ := by
  cases m <;> simp only [Move.isCastle, reduceCtorEq] at hm <;>
    simp [pseudoLegalCastleB, castleColorOf, castleRightOf, castleRookSource,
      Move.source] at hpl ⊢ <;>
    · obtain ⟨⟨⟨⟨ht, _⟩, hk⟩, _⟩, _⟩ := hpl
      rw [ht]
      exact hk

theorem castle_turn {q : Position} {m : Move} (hm : m.isCastle = true)
    (hpl : pseudoLegalCastleB q m = true) : castleColorOf m = some q.turn := by
  cases m <;> simp only [Move.isCastle, reduceCtorEq] at hm <;>
    simp [pseudoLegalCastleB, castleColorOf, castleRightOf, castleRookSource] at hpl ⊢ <;>
    · obtain ⟨⟨⟨⟨ht, _⟩, _⟩, _⟩, _⟩ := hpl
      exact ht.symm

/-- Castling is never legal out of check. -/
theorem castle_illegal {q : Position} {k : Square}
    (hK : ∀ s, (q.board s = some ⟨q.turn, .king⟩ ↔ s = k))
    (hatt : isAttackedB q (Color.opposite q.turn) k = true) (m : Move)
    (hm : m.isCastle = true) : legalMoveB q m = false := by
  have hps : pseudoLegalB q m = pseudoLegalCastleB q m := by
    cases m <;> simp_all [Move.isCastle, pseudoLegalB]
  cases hp : pseudoLegalCastleB q m
  · simp [legalMoveB, hps, hp]
  · have hmk : m.source = k := (hK _).1 (castle_source_king hm hp)
    have hcol := castle_turn hm hp
    obtain ⟨u, hu⟩ : ∃ u, castleTransitSquare m = some u := by
      cases m <;> simp_all [castleTransitSquare, Move.isCastle]
    have hsafe : castleSafeSquaresB q m = false := by
      simp [castleSafeSquaresB, hcol, hu, hmk, hatt]
    simp [legalMoveB, hps, hp, hm, hsafe]

/-- **A check that cannot be met is mate.** If the defender's king stands alone on `k`, the
attacker's piece on `c` checks it, en passant is off, no square next to the king is a flight,
no defender's piece but the king attacks `c`, and none can step between `c` and `k`, then the
side to move is checkmated. -/
theorem checkmate_of_unmeetable_check (q : Position) (k c : Square)
    (hKing : kingOnlyAtB q.board q.turn k = true)
    (hCheck : checksB q k c = true)
    (hEp : q.enPassant = none)
    (hFlights : flightsCoveredB q k = true)
    (hCapture : noCaptureB q k c = true)
    (hBlock : noBlockB q k c = true) :
    checkmateB q = true := by
  have hK := kingOnlyAtB_iff.1 hKing
  have hF : ∀ d, flightOkB q k d = true :=
    fun d => (List.all_eq_true.1 hFlights) d (mem_allSquares d)
  have hatt := attacked_of_checks hCheck
  have hin : inCheckB q q.turn = true := by
    unfold inCheckB
    rw [kingSquare_of_only hK]
    exact hatt
  have hno : ∀ m, legalMoveB q m = false := by
    intro m
    cases m with
    | normal s t =>
        by_cases hsk : s = k
        · subst hsk
          cases hp : normalPseudoLegalB q s t
          · simp [legalMoveB, pseudoLegalB, hp]
          · simp [legalMoveB, pseudoLegalB, hp, king_step_in_check hK hF t hp]
        · cases hp : normalPseudoLegalB q s t
          · simp [legalMoveB, pseudoLegalB, hp]
          · simp [legalMoveB, pseudoLegalB, hp,
              other_move_in_check hK hCheck hCapture hBlock s t hsk hp]
    | promotion s t pk =>
        cases hp : pseudoLegalPromotionB q (.promotion s t pk)
        · simp [legalMoveB, pseudoLegalB, hp]
        · simp [legalMoveB, pseudoLegalB, hp,
            promotion_in_check hK hCheck hCapture hBlock s t pk hp]
    | enPassant s t =>
        simp [legalMoveB, pseudoLegalB, pseudoLegalEnPassantB, hEp]
    | whiteKingCastle => exact castle_illegal hK hatt _ rfl
    | whiteQueenCastle => exact castle_illegal hK hatt _ rfl
    | blackKingCastle => exact castle_illegal hK hatt _ rfl
    | blackQueenCastle => exact castle_illegal hK hatt _ rfl
  unfold checkmateB
  rw [hin, Bool.true_and]
  simp only [legalMoves, List.isEmpty_iff, List.filter_eq_nil_iff, decide_eq_true_eq]
  intro m _ hl
  have h := (legalMove_iff_legalMoveB q m).1 hl
  rw [hno m] at h
  exact Bool.false_ne_true h

end Ideas
