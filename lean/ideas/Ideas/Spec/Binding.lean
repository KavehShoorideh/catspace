import Chess.Legality
import Chess.History
import Ideas.Draws

/-!
# THE BINDING of the rule spec to the kernel (ANALYST_PLAN 76.1 (b), extensions (2), (3), (5))

Hand-written, rule-level: the definitions the generated spec (`Ideas.Spec.Chess`, from rules/chess.gdl) reads its fluents, its moves and its inputs through --
the ontology shelf's `bind` entries (catspace/incubator/analyst/discovery/lemmas/ontology/) name them. Nothing here is chess knowledge: each is a rule of the
game as the kernel (chess-lean) states it, read in the spec's vocabulary.

* (2) THE FLUENTS. `right`: the spec's castling right of a side on a wing is the kernel's `CastleRight` (the kernel's set `Position.castling` IS the spec's
  `castling_right`); `clock`: the spec's clock is the kernel's halfmove count saturated at the rule's limit, 100 (the rule reads only "100 or more").
* (3) THE MOVE TERM AND THE MOVE BINDING. `MoveTerm` is the spec's sort `move` (its text is the UCI notation); `Does` the spec's joint move (the side to move
  plays the term, the other side noop); `termOf` a kernel move's term; `moveOf` the kernel's move a term means AT A POSITION -- e1g1 is castling only when the
  side to move's king stands on e1, a pawn's sideways move onto the EMPTY e.p. square is en passant only then (`moveOf_termOf`: every pseudo-legal move is
  read back).
* (5) THE GAME RECORD AS AN INPUT. `seen`: how often the record holds the position's identity (the kernel's `Chess.repetitionKey`) before it, saturating at 2
  (the rule reads only "2 or more": `seen_two_iff_threefold`, the third occurrence of `Ideas.Game.Record.threefold`).
-/

namespace Ideas.Spec.Binding

open Chess

/-! ## (2) The fluents -/

/-- The wing a castling goes to (the spec's sort `wing`). -/
inductive Wing where
  | kingside
  | queenside
  deriving DecidableEq, Repr

/-- THE CASTLING RIGHT of a side on a wing: the kernel's `CastleRight` (the spec's `castling_right ?c ?w` is `right c w ∈ p.castling`). -/
def right : Color → Wing → CastleRight
  | .white, .kingside => .whiteKingSide
  | .white, .queenside => .whiteQueenSide
  | .black, .kingside => .blackKingSide
  | .black, .queenside => .blackQueenSide

/-- Every kernel right is one side's right on one wing. -/
theorem right_surj (r : CastleRight) : ∃ c w, right c w = r := by
  cases r
  · exact ⟨.white, .kingside, rfl⟩
  · exact ⟨.white, .queenside, rfl⟩
  · exact ⟨.black, .kingside, rfl⟩
  · exact ⟨.black, .queenside, rfl⟩

/-- A right names its side and its wing. -/
theorem right_inj {c c' : Color} {w w' : Wing} (h : right c w = right c' w') : c = c' ∧ w = w' := by
  cases c <;> cases w <;> cases c' <;> cases w' <;> simp_all [right]

/-- THE CLOCK THE SPEC READS: the kernel's halfmove count saturated at the rule's limit. -/
def clock (p : Position) : Nat := min p.halfmove 100

/-! ## (3) The move term and the move binding -/

/-- THE MOVE TERM (the spec's sort `move`): a unit's move between two squares, a promotion with the unit it places, or noop (the side not to move). -/
inductive MoveTerm where
  | move (s t : Square)
  | promote (s t : Square) (k : PieceKind)
  | noop
  deriving DecidableEq, Repr

/-- THE JOINT MOVE (the spec's `does`): the side to move plays `mv`, the other side plays noop. -/
def Does (p : Position) (mv : MoveTerm) (c : Color) (t : MoveTerm) : Prop :=
  (c = p.turn ∧ t = mv) ∨ (c = Color.opposite p.turn ∧ t = MoveTerm.noop)

/-- `does` with a unit's move: the side to move plays it. -/
theorem does_move {p : Position} {mv : MoveTerm} {c : Color} {s t : Square} : Does p mv c (.move s t) ↔ c = p.turn ∧ mv = .move s t := by
  constructor
  · rintro (⟨h1, h2⟩ | ⟨_, h2⟩)
    · exact ⟨h1, h2.symm⟩
    · cases h2
  · rintro ⟨h1, h2⟩
    exact Or.inl ⟨h1, h2.symm⟩

/-- `does` with a promotion: the side to move plays it. -/
theorem does_promote {p : Position} {mv : MoveTerm} {c : Color} {s t : Square} {k : PieceKind} :
    Does p mv c (.promote s t k) ↔ c = p.turn ∧ mv = .promote s t k := by
  constructor
  · rintro (⟨h1, h2⟩ | ⟨_, h2⟩)
    · exact ⟨h1, h2.symm⟩
    · cases h2
  · rintro ⟨h1, h2⟩
    exact Or.inl ⟨h1, h2.symm⟩

/-- The kernel's promotion kind of the unit a promotion term places (a king or a pawn, which no pseudo-legal promotion names, reads as a queen). -/
def promoKind : PieceKind → PromotionKind
  | .rook => .rook
  | .bishop => .bishop
  | .knight => .knight
  | _ => .queen

theorem promoKind_toPieceKind (k : PromotionKind) : promoKind k.toPieceKind = k := by
  cases k <;> rfl

/-- A side's home rank (its king's and rooks' rank). -/
def homeRank : Color → Fin 8
  | .white => 0
  | .black => 7

/-- The kernel's castling move of a side on a wing. -/
def castleMove : Color → Wing → Move
  | .white, .kingside => .whiteKingCastle
  | .white, .queenside => .whiteQueenCastle
  | .black, .kingside => .blackKingCastle
  | .black, .queenside => .blackQueenCastle

/-- THE MOVE BINDING: the kernel's move a term means at the position `p` (none: noop). -/
def moveOf (p : Position) : MoveTerm → Option Move
  | .noop => none
  | .promote s t k => some (.promotion s t (promoKind k))
  | .move s t =>
      if p.board s = some ⟨p.turn, .king⟩ ∧ s = (4, homeRank p.turn) ∧ t = (6, homeRank p.turn) then some (castleMove p.turn .kingside)
      else if p.board s = some ⟨p.turn, .king⟩ ∧ s = (4, homeRank p.turn) ∧ t = (2, homeRank p.turn) then some (castleMove p.turn .queenside)
      else if p.board s = some ⟨p.turn, .pawn⟩ ∧ p.enPassant = some t ∧ p.board t = none ∧ s.1 ≠ t.1 then some (.enPassant s t)
      else some (.normal s t)

/-- A kernel move's term (its UCI text: castling the king's two-file move, en passant the pawn's diagonal move). -/
def termOf : Move → MoveTerm
  | .normal s t => .move s t
  | .enPassant s t => .move s t
  | .promotion s t k => .promote s t k.toPieceKind
  | .whiteKingCastle => .move (4, 0) (6, 0)
  | .whiteQueenCastle => .move (4, 0) (2, 0)
  | .blackKingCastle => .move (4, 7) (6, 7)
  | .blackQueenCastle => .move (4, 7) (2, 7)

/-! ## (5) The game record as an input -/

/-- THE RECORD'S READING (the spec's input `record`): how often the game record holds the position's identity before it, saturating at 2. -/
def seen (r : Ideas.Game.Record Position RepetitionKey) (p : Position) : Nat :=
  min (r.count (repetitionKey p)) 2

/-- The record reads 2 exactly when the position occurs a third time (the rule's `Ideas.Game.Record.threefold`). -/
theorem seen_two_iff_threefold (r : Ideas.Game.Record Position RepetitionKey) (p : Position) :
    seen r p = 2 ↔ r.threefold repetitionKey p = true := by
  simp only [seen, Ideas.Game.Record.threefold, decide_eq_true_eq]
  omega

end Ideas.Spec.Binding
