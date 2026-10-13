import Ideas.Passer
import Chess.Generator

/-!
# The mirror lemma: the left-right reflection is a symmetry of the rules

Kaveh (2026-10-13): "you want to have a mirroring lemma separately, and then you want to have the squares individually
available. So then basically later on you want to say, if I want to search this position, I don't need to because I can
just search that position and then apply the symmetry."

The reflection `σ` exchanges the a- and h-files, the b- and g-files, ... and keeps ranks and colours. It is a symmetry of
every rule but castling (the king starts on the e-file, so the reflected castle is no castle); pawns keep their direction,
so it holds WITH pawns -- the one symmetry that does. For a position without castling rights and with at most one king of
each colour (what every legal game keeps):

* `legalMove_mirror`      a move is legal exactly when its reflection is legal in the reflected position;
* `applyMove_mirror`      playing the reflected move in the reflected position gives the reflected result;
* `forcedWithin_mirror`   a side forces a reflection-invariant goal within `n` plies in the reflected position exactly when
                          it does in the position -- search one, apply the symmetry to the other.

Every step is reasoned: each geometry the rules read is invariant (`absDiff` of reflected files is the same distance,
"between" is reversed twice), the board read through `σ` is the reflected board, and the king lookup agrees when kings are
unique.
-/

namespace Ideas.Mirror

open Chess

/-! ## The reflection -/

/-- The left-right reflection of a square: a ↔ h, b ↔ g, c ↔ f, d ↔ e; the rank kept. -/
def σ (s : Square) : Square := (Fin.rev s.1, s.2)

@[simp] theorem σ_fst (s : Square) : (σ s).1 = Fin.rev s.1 := rfl
@[simp] theorem σ_snd (s : Square) : (σ s).2 = s.2 := rfl

@[simp] theorem σ_σ (s : Square) : σ (σ s) = s := by
  simp [σ]

theorem σ_injective : Function.Injective σ := by
  intro a b h
  have := congrArg σ h
  simpa using this

@[simp] theorem σ_inj {a b : Square} : σ a = σ b ↔ a = b := σ_injective.eq_iff

theorem σ_eq_iff {a b : Square} : σ a = b ↔ a = σ b := by
  constructor <;> intro h <;> subst h <;> simp

/-- The reflected board: what stands on `u` is what stood on its reflection. -/
def board (b : Board) : Board := fun u => b (σ u)

@[simp] theorem board_σ (b : Board) (s : Square) : board b (σ s) = b s := by
  simp [board]

/-- The reflected move (castles are left as they are: without castling rights neither side may castle). -/
def move : Move → Move
  | .normal s t => .normal (σ s) (σ t)
  | .promotion s t k => .promotion (σ s) (σ t) k
  | .enPassant s t => .enPassant (σ s) (σ t)
  | m => m

@[simp] theorem move_move (m : Move) : move (move m) = m := by
  cases m <;> simp [move]

/-- The reflected position. Castling rights are dropped: the theorems below are for positions without them. -/
def pos (p : Position) : Position :=
  { board := board p.board, turn := p.turn, castling := ∅, enPassant := p.enPassant.map σ,
    halfmove := p.halfmove, fullmove := p.fullmove }

@[simp] theorem pos_board (p : Position) : (pos p).board = board p.board := rfl
@[simp] theorem pos_turn (p : Position) : (pos p).turn = p.turn := rfl
@[simp] theorem pos_castling (p : Position) : (pos p).castling = ∅ := rfl
@[simp] theorem pos_enPassant (p : Position) : (pos p).enPassant = p.enPassant.map σ := rfl

/-! ## The geometry the rules read is invariant -/

theorem absDiff_rev (a b : Fin 8) : absDiff (Fin.rev a) (Fin.rev b) = absDiff a b := by
  unfold absDiff
  simp only [Fin.val_rev]
  split_ifs <;> omega

theorem ltB_rev (a b : Fin 8) : ltB (Fin.rev a) (Fin.rev b) = ltB b a := by
  simp only [ltB, Fin.val_rev, decide_eq_decide]
  omega

/-- "u lies between a and b" on a reflected line is the same statement read from the other end. -/
theorem betweenCoordB_rev (a u b : Fin 8) :
    betweenCoordB (Fin.rev a) (Fin.rev u) (Fin.rev b) = betweenCoordB a u b := by
  simp only [betweenCoordB, ltB_rev]
  cases ltB a u <;> cases ltB u b <;> cases ltB b u <;> cases ltB u a <;> rfl

@[simp] theorem rev_beq (a b : Fin 8) : (Fin.rev a == Fin.rev b) = (a == b) := by
  simp

@[simp] theorem σ_bne (a b : Square) : (σ a != σ b) = (a != b) := by
  simp [bne]

@[simp] theorem σ_beq (a b : Square) : (σ a == σ b) = (a == b) := by
  rw [Bool.eq_iff_iff]
  simp

theorem sameFileB_σ (s t : Square) : sameFileB (σ s) (σ t) = sameFileB s t := by
  simp [sameFileB]

theorem sameRankB_σ (s t : Square) : sameRankB (σ s) (σ t) = sameRankB s t := by
  simp [sameRankB]

theorem sameDiagonalB_σ (s t : Square) : sameDiagonalB (σ s) (σ t) = sameDiagonalB s t := by
  simp [sameDiagonalB, absDiff_rev]

theorem rookGeometryB_σ (s t : Square) : rookGeometryB (σ s) (σ t) = rookGeometryB s t := by
  simp [rookGeometryB, sameFileB_σ, sameRankB_σ]

theorem bishopGeometryB_σ (s t : Square) : bishopGeometryB (σ s) (σ t) = bishopGeometryB s t := by
  simp [bishopGeometryB, sameDiagonalB_σ]

theorem queenGeometryB_σ (s t : Square) : queenGeometryB (σ s) (σ t) = queenGeometryB s t := by
  simp [queenGeometryB, rookGeometryB_σ, bishopGeometryB_σ]

theorem knightGeometryB_σ (s t : Square) : knightGeometryB (σ s) (σ t) = knightGeometryB s t := by
  simp [knightGeometryB, absDiff_rev]

theorem kingGeometryB_σ (s t : Square) : kingGeometryB (σ s) (σ t) = kingGeometryB s t := by
  unfold kingGeometryB
  have h1 : absDiff (σ s).1 (σ t).1 = absDiff s.1 t.1 := absDiff_rev _ _
  rw [σ_bne, h1, σ_snd, σ_snd]

theorem pawnMoveGeometryB_σ (c : Color) (s t : Square) :
    pawnMoveGeometryB c (σ s) (σ t) = pawnMoveGeometryB c s t := by
  simp [pawnMoveGeometryB]

theorem pawnAttackGeometryB_σ (c : Color) (s t : Square) :
    pawnAttackGeometryB c (σ s) (σ t) = pawnAttackGeometryB c s t := by
  simp [pawnAttackGeometryB, absDiff_rev]

theorem pawnDoubleGeometryB_σ (c : Color) (s t : Square) :
    pawnDoubleGeometryB c (σ s) (σ t) = pawnDoubleGeometryB c s t := by
  simp [pawnDoubleGeometryB]

theorem betweenB_σ (s u t : Square) : betweenB (σ s) (σ u) (σ t) = betweenB s u t := by
  simp only [betweenB, σ_bne, sameFileB_σ, sameRankB_σ, sameDiagonalB_σ, σ_fst, σ_snd, rev_beq, absDiff_rev,
    betweenCoordB_rev]

/-! ## The board read through the reflection: lines, attacks, the king, check -/

theorem clearBetweenB_board (b : Board) (s t : Square) :
    clearBetweenB (board b) (σ s) (σ t) = clearBetweenB b s t := by
  rw [Bool.eq_iff_iff]
  change clearBetween (board b) (σ s) (σ t) ↔ clearBetween b s t
  rw [clearBetween_iff, clearBetween_iff]
  constructor
  · intro h v _ hv
    have hv' : between (σ s) (σ v) (σ t) := by unfold between; rw [betweenB_σ]; exact hv
    simpa using h (σ v) (mem_allSquares _) hv'
  · intro h u _ hu
    have hu' : between s (σ u) t := by unfold between; rw [← betweenB_σ, σ_σ]; exact hu
    exact h (σ u) (mem_allSquares _) hu'

theorem pieceAttacksB_board {p q : Position} (hq : q.board = board p.board) (c : Color) (s t : Square) :
    pieceAttacksB q c (σ s) (σ t) = pieceAttacksB p c s t := by
  unfold pieceAttacksB sliderAttackB
  rw [hq, board_σ, clearBetweenB_board]
  cases h : p.board s <;>
    simp [rookGeometryB_σ, bishopGeometryB_σ, queenGeometryB_σ, knightGeometryB_σ, kingGeometryB_σ,
      pawnAttackGeometryB_σ]

theorem isAttackedB_board {p q : Position} (hq : q.board = board p.board) (c : Color) (t : Square) :
    isAttackedB q c (σ t) = isAttackedB p c t := by
  rw [Bool.eq_iff_iff]
  simp only [isAttackedB, List.any_eq_true, mem_allSquares, true_and]
  constructor
  · rintro ⟨s, hs⟩
    exact ⟨σ s, by rw [← pieceAttacksB_board hq, σ_σ]; exact hs⟩
  · rintro ⟨s, hs⟩
    exact ⟨σ s, by rw [pieceAttacksB_board hq]; exact hs⟩

/-- At most one king of each colour -- what every legal game keeps (`uniqueKings_applyBoard`); the rules find "the" king
by scanning the squares in order, so with two kings of a colour the reflection could find the other one. -/
def UniqueKings (b : Board) : Prop :=
  ∀ c s t, b s = some ⟨c, .king⟩ → b t = some ⟨c, .king⟩ → s = t

theorem kingSquare_board {b : Board} (hb : UniqueKings b) (c : Color) :
    kingSquare (board b) c = (kingSquare b c).map σ := by
  cases h : kingSquare b c with
  | none =>
    rw [kingSquare_none_iff] at h
    simp only [Option.map_none]
    rw [kingSquare_none_iff]
    intro u hu
    exact h (σ u) hu
  | some s =>
    obtain ⟨_, hs⟩ := kingSquare_some h
    cases h' : kingSquare (board b) c with
    | none =>
      rw [kingSquare_none_iff] at h'
      exact absurd (by simpa using hs) (h' (σ s))
    | some u =>
      obtain ⟨_, hu⟩ := kingSquare_some h'
      have : σ u = s := hb c _ _ hu hs
      simp [← this]

theorem inCheckB_board {p q : Position} (hq : q.board = board p.board) (hb : UniqueKings p.board) (c : Color) :
    inCheckB q c = inCheckB p c := by
  unfold inCheckB
  rw [hq, kingSquare_board hb]
  cases kingSquare p.board c with
  | none => rfl
  | some s => exact isAttackedB_board hq _ _

/-! ## Pseudo-legal moves -/

theorem epCapturedSquare_σ (c : Color) (t : Square) :
    epCapturedSquare c (σ t) = (epCapturedSquare c t).map σ := by
  unfold epCapturedSquare
  simp only [σ_snd]
  split_ifs <;> rfl

theorem destinationFriendlyB_board {p q : Position} (hq : q.board = board p.board) (c : Color) (t : Square) :
    destinationFriendlyB q c (σ t) = destinationFriendlyB p c t := by
  unfold destinationFriendlyB
  rw [hq, board_σ]

theorem normalPieceGeometryB_board {p q : Position} (hq : q.board = board p.board) (c : Color) (x : Piece)
    (s t : Square) : normalPieceGeometryB q c x (σ s) (σ t) = normalPieceGeometryB p c x s t := by
  unfold normalPieceGeometryB
  rw [hq, board_σ, clearBetweenB_board]
  cases x.kind <;>
    simp [rookGeometryB_σ, bishopGeometryB_σ, queenGeometryB_σ, knightGeometryB_σ, kingGeometryB_σ,
      pawnMoveGeometryB_σ, pawnAttackGeometryB_σ, pawnDoubleGeometryB_σ]

theorem pseudoLegalB_pos {p : Position} (hc : p.castling = ∅) (m : Move) :
    pseudoLegalB (pos p) (move m) = pseudoLegalB p m := by
  cases m with
  | normal s t =>
    simp only [move, pseudoLegalB, normalPseudoLegalB, pos_board, board_σ, pos_turn]
    cases p.board s with
    | none => rfl
    | some x =>
      simp only [destinationFriendlyB_board (p := p) (q := pos p) rfl, σ_snd,
        normalPieceGeometryB_board (p := p) (q := pos p) rfl]
  | promotion s t k =>
    simp only [move, pseudoLegalB, pseudoLegalPromotionB, pos_board, board_σ, pos_turn, σ_snd,
      pawnMoveGeometryB_σ, pawnAttackGeometryB_σ]
  | enPassant s t =>
    simp only [move, pseudoLegalB, pseudoLegalEnPassantB, pos_board, board_σ, pos_turn, pos_enPassant,
      pawnAttackGeometryB_σ, epCapturedSquare_σ]
    cases p.enPassant with
    | none => rfl
    | some ep =>
      simp only [Option.map_some]
      cases epCapturedSquare p.turn t <;> simp
  | whiteKingCastle | whiteQueenCastle | blackKingCastle | blackQueenCastle =>
    simp [move, pseudoLegalB, pseudoLegalCastleB, castleColorOf, castleRightOf, castleRookSource, hc]

/-! ## Playing a move -/

theorem board_update (b : Board) (s : Square) (v : Option Piece) :
    board (boardUpdate b s v) = boardUpdate (board b) (σ s) v := by
  funext u
  simp only [board, boardUpdate, Function.update_apply, σ_eq_iff]

theorem board_move (b : Board) (s t : Square) :
    board (boardMove b s t) = boardMove (board b) (σ s) (σ t) := by
  unfold boardMove
  rw [board_update, board_update, board_σ]

theorem applyBoard_pos (p : Position) {m : Move} (hm : m.isCastle = false) :
    applyBoard (pos p) (move m) = board (applyBoard p m) := by
  cases m with
  | normal s t => simp only [move, applyBoard, pos_board, board_move]
  | promotion s t k => simp only [move, applyBoard, pos_board, pos_turn, board_update]
  | enPassant s t =>
    simp only [move, applyBoard, pos_board, pos_turn, epCapturedSquare_σ]
    cases epCapturedSquare p.turn t with
    | none => simp only [Option.map_none, board_move]
    | some cs => simp only [Option.map_some, board_update, board_move]
  | whiteKingCastle | whiteQueenCastle | blackKingCastle | blackQueenCastle => simp [Move.isCastle] at hm

theorem moveCaptureSquare_pos (p : Position) {m : Move} (hm : m.isCastle = false) :
    moveCaptureSquare (pos p) (move m) = (moveCaptureSquare p m).map σ := by
  cases m with
  | normal s t =>
    simp only [move, moveCaptureSquare, pos_board, board_σ]
    split_ifs <;> rfl
  | promotion s t k =>
    simp only [move, moveCaptureSquare, pos_board, board_σ]
    split_ifs <;> rfl
  | enPassant s t => simp only [move, moveCaptureSquare, pos_turn, epCapturedSquare_σ]
  | whiteKingCastle | whiteQueenCastle | blackKingCastle | blackQueenCastle => simp [Move.isCastle] at hm

theorem moveIsCapture_pos (p : Position) {m : Move} (hm : m.isCastle = false) :
    moveIsCapture (pos p) (move m) = moveIsCapture p m := by
  unfold moveIsCapture
  rw [moveCaptureSquare_pos p hm]
  cases moveCaptureSquare p m <;> simp

theorem moveIsPawn_pos (p : Position) {m : Move} (hm : m.isCastle = false) :
    moveIsPawn (pos p) (move m) = moveIsPawn p m := by
  cases m <;> simp_all [move, moveIsPawn, movingPiece, Move.source, Move.isCastle]

theorem pawnDoubleTarget_pos (p : Position) (m : Move) :
    pawnDoubleTarget (pos p) (move m) = (pawnDoubleTarget p m).map σ := by
  cases m with
  | normal s t =>
    have hp := moveIsPawn_pos p (m := .normal s t) rfl
    simp only [move] at hp
    simp only [move, pawnDoubleTarget, hp, σ_fst, σ_snd, Fin.rev_inj, pos_turn]
    split_ifs <;> rfl
  | _ => rfl

/-- A position equals another when every field does. -/
theorem position_ext {a b : Position} (h1 : a.board = b.board) (h2 : a.turn = b.turn) (h3 : a.castling = b.castling)
    (h4 : a.enPassant = b.enPassant) (h5 : a.halfmove = b.halfmove) (h6 : a.fullmove = b.fullmove) : a = b := by
  cases a; cases b; simp_all

/-- Playing the reflected move in the reflected position gives the reflected result (every move but a castle). -/
theorem applyMove_pos (p : Position) {m : Move} (hm : m.isCastle = false) :
    applyMove (pos p) (move m) = pos (applyMove p m) := by
  apply position_ext
  · rw [applyMove_board, applyBoard_pos p hm]; rfl
  · rfl
  · simp [applyMove, rightsAfterMove]
  · simp only [applyMove, newEnPassant, pawnDoubleTarget_pos, pos_enPassant]
  · show (applyMove (pos p) (move m)).halfmove = (applyMove p m).halfmove
    rw [applyMove_halfmove, applyMove_halfmove, moveIsPawn_pos p hm, moveIsCapture_pos p hm]; rfl
  · simp [applyMove, pos]

/-! ## One king of each colour is kept by every move -/

theorem uniqueKings_update {b : Board} (hb : UniqueKings b) (s : Square) {v : Option Piece}
    (hv : ∀ c, v ≠ some ⟨c, .king⟩) : UniqueKings (boardUpdate b s v) := by
  intro c u w hu hw
  simp only [boardUpdate, Function.update_apply] at hu hw
  split_ifs at hu hw with h1 h2 h2
  · exact h1.trans h2.symm
  · exact absurd hu (hv c)
  · exact absurd hw (hv c)
  · exact hb c u w hu hw

theorem uniqueKings_move {b : Board} (hb : UniqueKings b) (s t : Square) : UniqueKings (boardMove b s t) := by
  intro c u w hu hw
  simp only [boardMove, boardUpdate, Function.update_apply] at hu hw
  -- the king that moved stands on t alone (its old square is empty); every other king is where it was
  split_ifs at hu hw <;> first
    | (subst_vars; rfl)
    | simp at hu
    | simp at hw
    | (have := hb c _ _ hu hw; subst_vars; contradiction)
    | exact hb c u w hu hw

theorem promotion_not_king (c c' : Color) (k : PromotionKind) :
    (some ⟨c', k.toPieceKind⟩ : Option Piece) ≠ some ⟨c, .king⟩ := by
  cases k <;> simp [PromotionKind.toPieceKind]

theorem uniqueKings_applyBoard {p : Position} (hb : UniqueKings p.board) (m : Move) :
    UniqueKings (applyBoard p m) := by
  have hnone : ∀ c, (none : Option Piece) ≠ some ⟨c, .king⟩ := fun _ => by simp
  cases m with
  | normal s t => exact uniqueKings_move hb s t
  | promotion s t k =>
    exact uniqueKings_update (uniqueKings_update hb s hnone) t (fun c => promotion_not_king c p.turn k)
  | enPassant s t =>
    simp only [applyBoard]
    cases epCapturedSquare p.turn t with
    | none => exact uniqueKings_move hb s t
    | some cs => exact uniqueKings_update (uniqueKings_move hb s t) cs hnone
  | whiteKingCastle | whiteQueenCastle | blackKingCastle | blackQueenCastle =>
    simp only [applyBoard, castleBoard, castleRookSource, castleRookDestination]
    exact uniqueKings_move (uniqueKings_move hb _ _) _ _

theorem uniqueKings_applyMove {p : Position} (hb : UniqueKings p.board) (m : Move) :
    UniqueKings (applyMove p m).board := uniqueKings_applyBoard hb m

/-- Without castling rights no castle is pseudo-legal. -/
theorem not_castle_of_pseudoLegal {p : Position} (hc : p.castling = ∅) {m : Move} (h : pseudoLegalB p m = true) :
    m.isCastle = false := by
  cases m <;> simp_all [Move.isCastle, pseudoLegalB, pseudoLegalCastleB, castleColorOf, castleRightOf, castleRookSource]

theorem castling_applyMove {p : Position} (hc : p.castling = ∅) (m : Move) : (applyMove p m).castling = ∅ := by
  simp [applyMove, rightsAfterMove, hc]

/-! ## Legality -/

/-- **The mirror lemma, one move.** Without castling rights and with one king of each colour, a move is legal exactly when
its reflection is legal in the reflected position. -/
theorem legalMoveB_pos {p : Position} (hc : p.castling = ∅) (hb : UniqueKings p.board) (m : Move) :
    legalMoveB (pos p) (move m) = legalMoveB p m := by
  unfold legalMoveB
  rw [pseudoLegalB_pos hc]
  cases hps : pseudoLegalB p m with
  | false => rfl
  | true =>
    have hm := not_castle_of_pseudoLegal hc hps
    have hm' : (move m).isCastle = false := by cases m <;> simp_all [move, Move.isCastle]
    rw [applyMove_pos p hm, pos_turn, hm, hm',
      inCheckB_board (p := applyMove p m) (q := pos (applyMove p m)) rfl (uniqueKings_applyMove hb m)]
    simp

theorem legalMove_pos {p : Position} (hc : p.castling = ∅) (hb : UniqueKings p.board) (m : Move) :
    legalMove (pos p) (move m) ↔ legalMove p m := by
  rw [legalMove_iff_legalMoveB, legalMove_iff_legalMoveB, legalMoveB_pos hc hb]

theorem mem_legalMoves_pos {p : Position} (hc : p.castling = ∅) (hb : UniqueKings p.board) (m : Move) :
    m ∈ legalMoves (pos p) ↔ move m ∈ legalMoves p := by
  rw [legalMoves_correct, legalMoves_correct, ← legalMove_pos hc hb, move_move]

/-! ## Search one, know the other -/

/-- **The mirror lemma for search.** Without castling rights and with one king of each colour, side `c` forces a goal the
reflection does not change within `n` plies in the reflected position exactly when it does in the position: search one,
apply the symmetry to the other. -/
theorem forcedWithin_mirror (c : Color) (goal : Position → Bool)
    (hgoal : ∀ q : Position, q.castling = ∅ → UniqueKings q.board → goal (pos q) = goal q) :
    ∀ (n : Nat) (p : Position), p.castling = ∅ → UniqueKings p.board →
      forcedWithinB c goal (pos p) n = forcedWithinB c goal p n := by
  intro n
  induction n with
  | zero => intro p hc hb; simp only [forcedWithinB, hgoal p hc hb]
  | succ n ih =>
    intro p hc hb
    -- every legal move of the reflected position is the reflection of one of the position's, and leads to the reflection
    -- of where that one leads
    have step : ∀ m, m ∈ legalMoves p →
        forcedWithinB c goal (applyMove (pos p) (move m)) n = forcedWithinB c goal (applyMove p m) n := by
      intro m hm
      have hmc := not_castle_of_pseudoLegal hc (legalMoves_sound hm).1
      rw [applyMove_pos p hmc]
      exact ih _ (castling_applyMove hc m) (uniqueKings_applyMove hb m)
    have back : ∀ m, m ∈ legalMoves (pos p) →
        forcedWithinB c goal (applyMove (pos p) m) n = forcedWithinB c goal (applyMove p (move m)) n := by
      intro m hm
      have := step (move m) ((mem_legalMoves_pos hc hb m).1 hm)
      rwa [move_move] at this
    have hany : (legalMoves (pos p)).any (fun m => forcedWithinB c goal (applyMove (pos p) m) n) =
        (legalMoves p).any (fun m => forcedWithinB c goal (applyMove p m) n) := by
      rw [Bool.eq_iff_iff]
      simp only [List.any_eq_true]
      constructor
      · rintro ⟨m, hm, h⟩
        exact ⟨move m, (mem_legalMoves_pos hc hb m).1 hm, by rw [← back m hm]; exact h⟩
      · rintro ⟨m, hm, h⟩
        refine ⟨move m, (mem_legalMoves_pos hc hb (move m)).2 (by rwa [move_move]), ?_⟩
        rw [step m hm]; exact h
    have hall : (legalMoves (pos p)).all (fun m => forcedWithinB c goal (applyMove (pos p) m) n) =
        (legalMoves p).all (fun m => forcedWithinB c goal (applyMove p m) n) := by
      rw [Bool.eq_iff_iff]
      simp only [List.all_eq_true]
      constructor
      · intro h m hm
        have hm' : move m ∈ legalMoves (pos p) := (mem_legalMoves_pos hc hb (move m)).2 (by rwa [move_move])
        rw [← step m hm]; exact h _ hm'
      · intro h m hm
        rw [back m hm]; exact h _ ((mem_legalMoves_pos hc hb m).1 hm)
    have hempty : (legalMoves (pos p)).isEmpty = (legalMoves p).isEmpty := by
      rw [Bool.eq_iff_iff]
      simp only [List.isEmpty_iff, List.eq_nil_iff_forall_not_mem]
      constructor
      · intro h m hm
        exact h (move m) ((mem_legalMoves_pos hc hb (move m)).2 (by rwa [move_move]))
      · intro h m hm
        exact h _ ((mem_legalMoves_pos hc hb m).1 hm)
    simp only [forcedWithinB, hgoal p hc hb, pos_turn, hany, hall, hempty]

end Ideas.Mirror
