import Ideas.Fork

/-!
# The refuter: the reply that killed a route (ANALYST_PLAN 52.22 (3), Kaveh 2026-10-19 on 07Z0P)

"Bb3 fails to Qxb3": when a route of ours is killed, the kill is one reply of theirs, and the piece that makes it is the
REFUTER of the route. `Refutes` names it (a legal reply that leaves the goal unmet); `refuter_unforces` is why one such reply
is enough to kill the route (one reply of theirs suffices: they need only play it). The fix the agenda applies to it is the
one it applies to a defender or a blocker -- take the refuter (or replace it) with tempo, then ask the route again.
-/

namespace Ideas.Refuter

open Chess Ideas

/-- THE REFUTER: `r` is a legal reply of theirs in `q` (their move: `q.turn ≠ c`) after which the goal of `c` is not met. -/
def Refutes (c : Color) (goal : Position → Bool) (q : Position) (r : Move) : Prop :=
  q.turn ≠ c ∧ r ∈ legalMoves q ∧ goal (applyMove q r) = false

/-- One refuting reply kills the route: the goal is not forced one ply on from the position after our move. -/
theorem refuter_unforces (c : Color) (goal : Position → Bool) (q : Position) (r : Move) (h : Refutes c goal q r) :
    forcedAtLeaves c goal q 1 = false := by
  obtain ⟨ht, hr, hg⟩ := h
  have hne : (legalMoves q).isEmpty = false := by
    cases hl : legalMoves q with
    | nil => simp [hl] at hr
    | cons a t => rfl
  simp only [forcedAtLeaves, ht, if_false, hne]
  simp only [Bool.false_eq_true, if_false, List.all_eq_false]
  exact ⟨r, hr, by simpa [forcedAtLeaves] using hg⟩

end Ideas.Refuter
