import Ideas.Forcing
import Ideas.GeoLemmas

/-!
# The duty word: "pinned to a threat" (ANALYST_PLAN 63.2, Kaveh 2026-10-21: "we already have threat and pin; the bishop is pinned to a threat")

`value q r` is what we win BY RULE after their move `r` in `q` (a mate or a threat of mate -> 100, else the exchange; `_their_after_capture`'s reading,
mirrored in `regress.duty_after`), generic in that reading. The DUTY of a position is the LEAST of `value` over their legal moves; a piece is DUTY-BOUND
when that least is at least its worth (every move of it abandons a duty worth the piece). A frozen piece (no legal move) has duty 0 here (it is
frozen, not bound). The geometric pin is the special case where the threat behind is the exchange on the piece behind it.

* `duty_le`: the duty is at most the value of every legal move (it is a least).
* `dutyBound_of_all` / `pin_dutyBound`: if every legal move of the piece abandons a value >= its worth, it is duty-bound (and a pinned piece, whose every
  move opens the line to the piece behind -- for a knight `GeoLemmas.knight_leaves_every_line` -- is the case `value >= worth of the piece behind`).
-/

namespace Ideas.Duty

open Chess

/-- The duty: the least value over the legal moves (0 when there are none). -/
def duty (value : Position → Move → Nat) (q : Position) : Nat :=
  match ((legalMoves q).map (value q)).min? with
  | some v => v
  | none => 0

/-- Duty-bound: there is a legal move and every one of them abandons a value of at least `w`. -/
def dutyBound (value : Position → Move → Nat) (w : Nat) (q : Position) : Bool :=
  !(legalMoves q).isEmpty && decide (w ≤ duty value q)

/-- The duty is at least `w` exactly when every legal move abandons a value >= `w` (given there is a legal move). -/
theorem le_duty_iff {value : Position → Move → Nat} {w : Nat} {q : Position} (hne : legalMoves q ≠ []) :
    w ≤ duty value q ↔ ∀ r, r ∈ legalMoves q → w ≤ value q r := by
  have hn : ((legalMoves q).map (value q)).min? ≠ none := by
    intro h
    rw [List.min?_eq_none_iff, List.map_eq_nil_iff] at h
    exact hne h
  obtain ⟨v, hv⟩ := Option.ne_none_iff_exists'.mp hn
  have key := List.le_min?_iff hv (x := w)
  simp only [duty, hv]
  rw [key]
  constructor
  · intro h r hr; exact h _ (List.mem_map_of_mem hr)
  · intro h b hb
    obtain ⟨r, hr, rfl⟩ := List.mem_map.mp hb
    exact h r hr

/-- The duty is at most the value of each legal move. -/
theorem duty_le {value : Position → Move → Nat} {q : Position} {r : Move} (hr : r ∈ legalMoves q) : duty value q ≤ value q r := by
  have hne : legalMoves q ≠ [] := fun h => by rw [h] at hr; simp at hr
  exact (le_duty_iff hne).mp (Nat.le_refl _) r hr

/-- If every legal move abandons a value >= `w` (and there is one), the position is duty-bound. -/
theorem dutyBound_of_all {value : Position → Move → Nat} {w : Nat} {q : Position}
    (hne : legalMoves q ≠ []) (h : ∀ r, r ∈ legalMoves q → w ≤ value q r) : dutyBound value w q = true := by
  have : w ≤ duty value q := (le_duty_iff hne).mpr h
  have hne' : (legalMoves q).isEmpty = false := by
    cases hl : legalMoves q with
    | nil => exact absurd hl hne
    | cons _ _ => rfl
  simp [dutyBound, hne', this]

/-- Duty-bound means every legal move abandons a value >= `w`. -/
theorem dutyBound_abandons {value : Position → Move → Nat} {w : Nat} {q : Position}
    (h : dutyBound value w q = true) {r : Move} (hr : r ∈ legalMoves q) : w ≤ value q r := by
  simp only [dutyBound, Bool.and_eq_true, decide_eq_true_eq] at h
  exact Nat.le_trans h.2 (duty_le hr)

/-- THE PIN AS A DUTY: if every legal move of the piece opens the line to the piece behind it (the line was blocked by the piece; for a knight
`GeoLemmas.knight_leaves_every_line`: a knight on a line through `s` always opens it), and opening it wins a value of at least `w` (the exchange on the
piece behind, or the mate behind), the piece is duty-bound by `w`: the geometric pin is the special case of "pinned to a threat". -/
theorem pin_dutyBound {value : Position → Move → Nat} {opens : Position → Move → Bool} {w : Nat} {q : Position}
    (hne : legalMoves q ≠ [])
    (hopen : ∀ r, r ∈ legalMoves q → opens q r = true) (hwin : ∀ r, opens q r = true → w ≤ value q r) :
    dutyBound value w q = true :=
  dutyBound_of_all hne (fun r hr => hwin r (hopen r hr))

/-- The knight case of the pin: a knight standing on the line through `s` leaves that line with every move (re-export of `GeoLemmas.knight_leaves_every_line`). -/
theorem knight_pin_opens (s t : Square) (h : knightGeometryB s t = true) : s.1 ≠ t.1 ∧ s.2 ≠ t.2 :=
  let r := GeoLemmas.knight_leaves_every_line s t h
  ⟨r.1, r.2.1⟩

end Ideas.Duty
