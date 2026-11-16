import Ideas.Core.Game
import Ideas.Race
import Ideas.Base.Vocabulary

/-!
# The GAME structure: the axioms a domain supplies, as one Lean object (ANALYST_PLAN 71.4; Kaveh 2026-10-22 "machinery generic; chess-specific Lean axioms")

The machinery's theorems were already abstract in their games: `Ideas.Forced.Reach` and `forced_by_measure` quantify over relations `ours theirs : α → α → Prop`;
`Ideas.Race.race_sound` over the same with two tempo measures; `Ideas.Forcing.parries` over a claim `bind : Position → Bool`. What was missing is the ONE object that
names what a domain must supply so those relations exist: this file. A `Game State Move` is the mover's MOVES, APPLY, the PASS (the counterfactual every threat word is
defined through), the MEASURE for the mover (material in chess: the status and every claim's value are differences of it) and IN-CHECK (the forced-reply condition).
From it the machinery derives the step relation, the terminals (no move: mated when in check, stalemated when not), `Reach`, and the race -- each a restatement of the
abstract theorem with the game's own hypotheses, nothing reproved.

Chess is the instance `chess` (the kernel's `legalMoves` / `applyMove`). Two variants show the layering: Chess960 is the SAME instance (the laws are those of chess, only
the start differs: not a Lean object at all); king-of-the-hill is the same instance with ANOTHER GOAL (`kothGoal`: a predicate over states, which is what `Reach` takes).
A domain whose laws differ is a second instance.

The Python mirror is `discovery/game.py` (`Game` protocol, `ChessGame`): its methods are these fields plus the names and the geometry the policy's features need.
-/

namespace Ideas.Game

open Chess

/-! ## Chess, the instance -/

/-- Chess: the kernel's legal moves and `applyMove`; the pass flips the side to move (and drops the en-passant right, which the other side cannot use);
the measure is the mover's material less the other side's; in check by the kernel's `inCheckB`. Chess960 is THIS instance: its laws are chess's. -/
def chess : Game Position Chess.Move where
  moves := legalMoves
  apply := applyMove
  pass := fun p => { p with turn := Color.opposite p.turn, enPassant := none }
  measure := fun p => (Ideas.Vocabulary.materialOf p p.turn : Int) - (Ideas.Vocabulary.materialOf p (Color.opposite p.turn) : Int)
  inCheck := fun p => inCheckB p p.turn

theorem chess_pass_turn (p : Position) : (chess.pass p).turn = Color.opposite p.turn := rfl

/-- The pass leaves the board as it is: what every threat word reads. -/
theorem chess_pass_board (p : Position) : (chess.pass p).board = p.board := rfl

/-- The measure is antisymmetric across the pass: their balance is minus ours. -/
theorem chess_measure_pass (p : Position) : chess.measure (chess.pass p) = - chess.measure p := by
  simp only [chess, Color.opposite_opposite]
  show (↑(Ideas.Vocabulary.materialOf p (Color.opposite p.turn)) : Int) - ↑(Ideas.Vocabulary.materialOf p p.turn)
      = -(↑(Ideas.Vocabulary.materialOf p p.turn) - ↑(Ideas.Vocabulary.materialOf p (Color.opposite p.turn)))
  omega

/-! ## A variant by its goal: king of the hill (71.6)

The same game, another goal: the side `c` wins when its king stands on one of the four centre squares. `Reach chess (kothGoal c)` is the forced win of that
variant; mate (the kernel's `checkmateB`) remains a goal too, so the variant's goal is the disjunction. Nothing in the machinery changes: a goal is a predicate. -/

/-- The four centre squares (files d, e; ranks 4, 5) hold the mover's king of colour `c`. -/
def kothGoal (c : Color) (p : Position) : Prop :=
  ∃ s : Square, (s.1.val = 3 ∨ s.1.val = 4) ∧ (s.2.val = 3 ∨ s.2.val = 4) ∧ p.board s = some ⟨c, .king⟩

end Ideas.Game
