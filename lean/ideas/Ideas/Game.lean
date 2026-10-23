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

/-- The axioms a domain supplies. `State` and `Move` are the domain's own types; the machinery never looks inside them. -/
structure Game (State Move : Type) where
  moves : State → List Move                                     -- the mover's legal moves
  apply : State → Move → State
  pass : State → State                                           -- the same state with the other side to move
  measure : State → Int                                          -- the mover's measure (material balance in chess)
  inCheck : State → Bool                                         -- the mover must answer a threat to the royal unit

variable {State Move : Type} (g : Game State Move)

namespace Game

/-- ONE STEP of the game: some legal move leads from `p` to `q`. Ours and theirs are the same relation, read from the side to move. -/
def step (p q : State) : Prop := ∃ m, m ∈ g.moves p ∧ q = g.apply p m

/-- No legal move. -/
def terminal (p : State) : Bool := (g.moves p).isEmpty

/-- The mover is MATED: no move, and in check. -/
def mated (p : State) : Bool := g.terminal p && g.inCheck p

/-- The mover is STALEMATED: no move, not in check. -/
def stalemated (p : State) : Bool := g.terminal p && !g.inCheck p

theorem step_of_mem {p : State} {m : Move} (hm : m ∈ g.moves p) : g.step p (g.apply p m) := ⟨m, hm, rfl⟩

/-- A terminal state has no step out of it. -/
theorem terminal_no_step {p : State} (h : g.terminal p = true) : ¬ ∃ q, g.step p q := by
  rintro ⟨q, m, hm, _⟩
  simp only [terminal, List.isEmpty_iff] at h
  simp [h] at hm

/-- A non-terminal state has a step (the mover has a reply: `hlive` of the race). -/
theorem step_of_not_terminal {p : State} (h : g.terminal p = false) : ∃ q, g.step p q := by
  simp only [terminal, List.isEmpty_eq_false_iff_exists_mem] at h
  obtain ⟨m, hm⟩ := h
  exact ⟨g.apply p m, g.step_of_mem hm⟩

/-- Mated or stalemated is exactly terminal. -/
theorem terminal_iff {p : State} : g.terminal p = true ↔ (g.mated p = true ∨ g.stalemated p = true) := by
  simp only [mated, stalemated, Bool.and_eq_true, Bool.not_eq_true']
  constructor
  · intro h; cases hc : g.inCheck p <;> simp [h, hc]
  · rintro (⟨h, _⟩ | ⟨h, _⟩) <;> exact h

/-- REACH in a game: from `p` (us to move) we force the `goal` (read after our move) within `n` of our moves against every reply (`Ideas.Forced.Reach` with the game's
step relation on both sides; a stalemate -- no reply and no goal -- is no success, as there). -/
def Reach (goal : State → Prop) (n : Nat) (p : State) : Prop :=
  Ideas.Forced.Reach g.step g.step goal (fun _ => False) n p

theorem Reach.mono {goal : State → Prop} {n m : Nat} {p : State} (h : g.Reach goal n p) (hle : n ≤ m) : g.Reach goal m p :=
  Ideas.Forced.Reach.mono h hle

/-- THE RACE IN A GAME (`Ideas.Race.race_sound` with the game's hypotheses): `μ` our remaining tempi, `ν` theirs; our move lowers `μ` by one while there is one
(`hprog`); `μ = 0` is the goal (`hgoal`); no reply raises `μ` or hands us the goal (`hheld`); they are never stalemated while we have tempi (`hlive`); a reply lowers
`ν` by at most one and our move never lowers it (`hfirst`, `hours`). Then from `0 < μ p ≤ ν p` we force the goal within `μ p` of our moves. -/
theorem race_sound {goal : State → Prop} (μ ν : State → Nat)
    (hprog : ∀ p, 0 < μ p → ∃ m, m ∈ g.moves p ∧ μ (g.apply p m) + 1 = μ p)
    (hgoal : ∀ q, μ q = 0 → goal q)
    (hheld : ∀ q m, m ∈ g.moves q → 0 < μ q → 0 < μ (g.apply q m) ∧ μ (g.apply q m) ≤ μ q)
    (hlive : ∀ q, 0 < μ q → g.terminal q = false)
    (hfirst : ∀ q m, m ∈ g.moves q → ν q ≤ ν (g.apply q m) + 1)
    (hours : ∀ p m, m ∈ g.moves p → ν p ≤ ν (g.apply p m)) :
    ∀ p, 0 < μ p → μ p ≤ ν p → g.Reach goal (μ p) p := by
  intro p hpos hle
  refine Ideas.Race.race_sound (ours := g.step) (theirs := g.step) (goal := goal) μ ν ?_ hgoal ?_ ?_ ?_ ?_ p hpos hle
  · intro p hp
    obtain ⟨m, hm, h⟩ := hprog p hp
    exact ⟨g.apply p m, g.step_of_mem hm, h⟩
  · rintro q r ⟨m, hm, rfl⟩ hq
    exact hheld q m hm hq
  · intro q hq
    exact g.step_of_not_terminal (hlive q hq)
  · rintro q r ⟨m, hm, rfl⟩
    exact hfirst q m hm
  · rintro p q ⟨m, hm, rfl⟩
    exact hours p m hm

end Game

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
