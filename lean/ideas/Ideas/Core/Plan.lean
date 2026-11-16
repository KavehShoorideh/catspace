import Ideas.Core.Race

/-!
# The plan's weaknesses, over places (PLAN 78.56 stage D1; moved out of `Ideas/Plan.lean`)

A weak moment is a PLACE and a deadline; a mover is a unit with a lower bound `d` on the moves it needs to reach a place. `safe`, `safe_mono`, `safe_mono_movers` and
`plan_sound` were stated over the chess square type and read nothing of a square; they are now over an arbitrary type `Place` (the old names are these, with `Place := Square`).
The proof of `plan_sound` is unchanged: it rests on `Ideas.Race.far_not_bearing`. No chess file is imported. The story and the chess bounds are in `Ideas/Plan.lean` (ANALYST_PLAN 67).
-/

namespace Ideas.Plan

open Ideas.Race

variable {Place : Type}

/-- A weak moment: a square that must be free of their pieces when they reply to our move `deadline` (counted from 1), and before. -/
structure Weak (Place : Type) where
  sq : Place
  deadline : Nat

/-- A piece of theirs for the arithmetic: where it stands and the lower bound `d` on the moves it needs to any square. -/
structure Mover (Place : Type) where
  pos : Place
  d : Place → Place → Nat

/-- THE CONDITION: every mover needs more moves to every weak square than the deadline allows. -/
def safe (W : List (Weak Place)) (M : List (Mover Place)) : Prop :=
  ∀ w ∈ W, ∀ m ∈ M, w.deadline < m.d m.pos w.sq

/-- More weak moments only make the condition harder: the mirror may list every square that could matter. -/
theorem safe_mono {W W' : List (Weak Place)} {M : List (Mover Place)} (hsub : ∀ w ∈ W', w ∈ W) (h : safe W M) : safe W' M := by
  intro w hw m hm
  exact h w (hsub w hw) m hm

/-- More movers only make the condition harder: every piece of theirs is listed. -/
theorem safe_mono_movers {W : List (Weak Place)} {M M' : List (Mover Place)} (hsub : ∀ m ∈ M', m ∈ M) (h : safe W M) : safe W M' := by
  intro w hw m hm
  exact h w hw m (hsub m hm)

/-- THE SOUNDNESS: with `safe`, a mover whose real steps respect its bound (1-Lipschitz: each step lowers the bound by at most one) stands on no weak square
after `j ≤ deadline` of its moves, whatever those moves are. The opponent's whole reply sequence is quantified over: nothing of theirs can capture on, block,
or check from a weak square in time. -/
theorem plan_sound {W : List (Weak Place)} {M : List (Mover Place)} (h : safe W M)
    {step : Place → Place → Prop} {m : Mover Place} (hm : m ∈ M) (hself : ∀ t, m.d t t = 0)
    (hstep : ∀ s s' t, step s s' → m.d s t ≤ m.d s' t + 1)
    {w : Weak Place} (hw : w ∈ W) {j : Nat} {s' : Place} (hst : Steps step j m.pos s') (hj : j ≤ w.deadline) : s' ≠ w.sq := by
  have hfar : ∀ t, t = w.sq → w.deadline < m.d m.pos t := by
    intro t ht
    subst ht
    exact h w hw m hm
  have := far_not_bearing m.d hself hstep (fun t => t = w.sq) hfar hst hj
  exact this

/-- The deadline of a square the plan piece stands on from our move `arrive` until it leaves at our move `leave` (exclusive): their last chance to capture there is
their reply to our move `leave - 1`, so the deadline is `leave - 1`; a piece that still stands there at the end of the plan of length `n` has deadline `n`. -/
def standDeadline (leave : Option Nat) (n : Nat) : Nat :=
  match leave with
  | some l => l - 1
  | none => n

theorem standDeadline_le {leave : Option Nat} {n : Nat} (h : ∀ l, leave = some l → l ≤ n + 1) : standDeadline leave n ≤ n := by
  cases leave with
  | none => simp [standDeadline]
  | some l =>
    have := h l rfl
    simp only [standDeadline]
    omega

/-- A square the pawn pushes to at our move `t + 1` must be empty when they reply to our move `t`: the blockade deadline is `t`. -/
def blockadeDeadline (t : Nat) : Nat := t

end Ideas.Plan
