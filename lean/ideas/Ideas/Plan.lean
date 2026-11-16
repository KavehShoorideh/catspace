import Ideas.Race
import Ideas.Core.Plan

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

/- `Weak`, `Mover`, `safe`, `safe_mono`, `safe_mono_movers`, `plan_sound`, `standDeadline`, `blockadeDeadline` are in `Ideas/Core/Plan.lean`, over an arbitrary type of places (PLAN 78.56). -/

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
