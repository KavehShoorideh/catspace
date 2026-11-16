import Ideas.Core.Race

/-!
# The GAME structure, game-free (PLAN 78.56 stage B; moved out of `Ideas/Game.lean`, proofs unchanged)

The axioms a domain supplies, as one Lean object, and what the machinery derives from them: the step relation, the terminals, `Reach`, the race. This file imports no chess
file; the chess instance `Ideas.Game.chess` and the king-of-the-hill goal stay in `Ideas/Game.lean`. Names and namespace are as they were (`Ideas.Game.Game`,
`Ideas.Game.Game.race_sound`, ...). See `Ideas/Game.lean` for the story (ANALYST_PLAN 71.4).
-/

namespace Ideas.Game

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

end Ideas.Game
