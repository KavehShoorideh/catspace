import Ideas.Race

/-!
# The plan's own weaknesses (ANALYST_PLAN 67; Kaveh 2026-10-22, dictated)

"Being able to ascertain the weakness in my own position resulting from my own actions, regardless of my opponent."

A PLAN is our move sequence to a goal with the opponent passing (the pawn walks; a piece of ours that blocks it steps aside: that step is part of the plan).
Rolling the plan forward WITHOUT the opponent gives the boards after each of our moves, and on those boards the plan creates WEAK MOMENTS by itself: a
square and a move number -- the pawn stands unprotected on `a7` after our move 3 while the rook steps aside; the square the pawn pushes to next must be empty;
a square from which a check would cost us a tempo. Each weak moment is THEIR goal with a DEADLINE: to matter they must reach that square by that move.

What is proved here is the arithmetic that makes a plan immune to the opponent: every piece of theirs has a lower bound `d` on the moves it needs to reach
a square (`kdist` for the king, `Passer2.pdist` for a pawn, `Race.lb` for the rest; each is 1-Lipschitz along the piece's real steps, `Race.lb_step`,
`Race.kdist_step`). If for EVERY weak moment EVERY piece of theirs needs more moves than the deadline allows, then however they move, no piece of theirs
stands on a weak square at its moment (`plan_sound`, from `Race.far_not_bearing`). The Python mirror (`discovery/plan67.py`) computes the rollout, the
weak moments and the reach; a plan with a REACHABLE weak moment is a DEFECT of the plan (a GAP for the chase, never a kill), and its fixes -- another
placement of the support with no weak moment (the rook BEHIND the pawn), a tempo gained -- are root questions of ours (67.4).

Over-approximation is the safe direction: more weak moments can only make `safe` harder (`safe_mono`), and a larger lower bound would be unsound, so the
mirror uses the proved bounds and lists every square that could matter (capture, blockade, check and pin squares), ignoring blockers (a blocker can only
make their piece slower, never faster).
-/

namespace Ideas.Plan

open Chess
open Ideas.Race

/-- A weak moment: a square that must be free of their pieces when they reply to our move `deadline` (counted from 1), and before. -/
structure Weak where
  sq : Square
  deadline : Nat

/-- A piece of theirs for the arithmetic: where it stands and the lower bound `d` on the moves it needs to any square. -/
structure Mover where
  pos : Square
  d : Square → Square → Nat

/-- THE CONDITION: every mover needs more moves to every weak square than the deadline allows. -/
def safe (W : List Weak) (M : List Mover) : Prop :=
  ∀ w ∈ W, ∀ m ∈ M, w.deadline < m.d m.pos w.sq

/-- More weak moments only make the condition harder: the mirror may list every square that could matter. -/
theorem safe_mono {W W' : List Weak} {M : List Mover} (hsub : ∀ w ∈ W', w ∈ W) (h : safe W M) : safe W' M := by
  intro w hw m hm
  exact h w (hsub w hw) m hm

/-- More movers only make the condition harder: every piece of theirs is listed. -/
theorem safe_mono_movers {W : List Weak} {M M' : List Mover} (hsub : ∀ m ∈ M', m ∈ M) (h : safe W M) : safe W M' := by
  intro w hw m hm
  exact h w hw m (hsub m hm)

/-- THE SOUNDNESS: with `safe`, a mover whose real steps respect its bound (1-Lipschitz: each step lowers the bound by at most one) stands on no weak square
after `j ≤ deadline` of its moves, whatever those moves are. The opponent's whole reply sequence is quantified over: nothing of theirs can capture on, block,
or check from a weak square in time. -/
theorem plan_sound {W : List Weak} {M : List Mover} (h : safe W M)
    {step : Square → Square → Prop} {m : Mover} (hm : m ∈ M) (hself : ∀ t, m.d t t = 0)
    (hstep : ∀ s s' t, step s s' → m.d s t ≤ m.d s' t + 1)
    {w : Weak} (hw : w ∈ W) {j : Nat} {s' : Square} (hst : Steps step j m.pos s') (hj : j ≤ w.deadline) : s' ≠ w.sq := by
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

namespace Ideas.Plan

open Chess

/-! ## The piece-specific lower bounds the mirror uses beside `Race.kdist` (king), `Passer2.pdist` (pawn) and `Race.lb` (queen, knight) -/

/-- A rook needs 0 moves to stay, 1 along its file or rank, else 2. -/
def rdist (s t : Square) : Nat := if s = t then 0 else if rookGeometryB s t then 1 else 2

/-- The colour of a square: the parity of file + rank. A bishop never changes it. -/
def colour (s : Square) : Nat := (s.1.val + s.2.val) % 2

/-- A bishop needs 0 moves to stay, 1 along its diagonal, 2 on its colour, and never reaches the other colour (64: more than any deadline). -/
def bdist (s t : Square) : Nat := if s = t then 0 else if bishopGeometryB s t then 1 else if colour s = colour t then 2 else 64

theorem rdist_self (t : Square) : rdist t t = 0 := by simp [rdist]

theorem bdist_self (t : Square) : bdist t t = 0 := by simp [bdist]

/-- A rook's step is 1-Lipschitz for `rdist` (the hypothesis of `plan_sound`). -/
theorem rdist_step {s s' t : Square} (h : rookGeometryB s s' = true) : rdist s t ≤ rdist s' t + 1 := by
  unfold rdist
  by_cases hst : s = t
  · simp [hst]
  · by_cases hs't : s' = t
    · subst hs't
      simp [hst, h]
    · split_ifs <;> omega

theorem bishop_keeps_colour {s s' : Square} (h : bishopGeometryB s s' = true) : colour s = colour s' := by
  unfold bishopGeometryB sameDiagonalB at h
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq] at h
  obtain ⟨_, hd⟩ := h
  unfold colour absDiff at *
  split_ifs at hd <;> omega

/-- A bishop's step is 1-Lipschitz for `bdist`. -/
theorem bdist_step {s s' t : Square} (h : bishopGeometryB s s' = true) : bdist s t ≤ bdist s' t + 1 := by
  have hc := bishop_keeps_colour h
  unfold bdist
  by_cases hst : s = t
  · simp [hst]
  · by_cases hs't : s' = t
    · subst hs't
      simp [hst, h]
    · by_cases hb : bishopGeometryB s t = true
      · simp only [hst, hb, if_false, if_true]
        split_ifs <;> omega
      · by_cases hb' : bishopGeometryB s' t = true
        · have : colour s = colour t := by
            rw [hc]
            exact bishop_keeps_colour hb'
          simp [hst, hb, hs't, hb', this]
        · simp [hst, hb, hs't, hb']
          rw [hc]
          split_ifs <;> omega

end Ideas.Plan
