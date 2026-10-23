import Ideas.Base.Vocabulary

/-!
# THE PRICED REACH (ANALYST_PLAN 72.1; Kaveh 2026-10-23: "attackable in, I love that ... nothing is illegal in chess, it just costs you something; the natural form of cost is material and tempo")

`Ideas.Vocabulary.attackableIn c s k p` says: side `c` can bear on square `s` within `k` of its own moves, the other side PASSING, every landing SAFE by the exchange. This file
generalises it to a PRICE: the last landing may be unsafe, and then the path costs what the other side wins by the exchange there (`seeOf`); every earlier landing must still be safe
(an unsafe landing before the end means the unit is taken and the roll ends). `reachPrice c s k p` is the least such material price, or `none` when no path exists within `k`;
the tempi are the depth at which it is found (the Python mirror `regress.reach_price` returns the pair and the path).

The semantics, said once: the other side's QUIET moves are passed, its PAYING CAPTURES are assumed (that is what "safe by the exchange" reads). So the word is a counterfactual over
the position, not a theorem about the game: a `none` kills the question "can this unit join" NOW, with its read set (the landings and the lines to `s`: a move of theirs that opens a line
re-asks it through the listener, as every word of the engine is re-asked -- `Ideas.FrameLemma`), and a `some v` is a CANDIDATE line of `k` moves with no reply read, never a proof.
What IS proved here is the link to the price-0 word: a reach at price 0 is `attackableIn`, and `attackableIn` is a reach at price 0 (`reachPrice_zero_iff`). The two knights of
Kaveh's example (one gives way for the other) are a path another unit must open: not read by this word (one unit rolls).
-/

namespace Ideas.Reach

open Chess Ideas.Vocabulary

/-- The PASS: the same board with `c` to move (the en-passant right dropped: the other side cannot use it). -/
def passTo (c : Color) (p : Position) : Position := { p with turn := c, enPassant := none }

/-- The least of two optional prices (`none` is "no path"). -/
def minOpt : Option Nat → Option Nat → Option Nat
  | none, b => b
  | a, none => a
  | some a, some b => some (min a b)

theorem minOpt_none_left (b : Option Nat) : minOpt none b = b := by cases b <;> rfl
theorem minOpt_none_right (a : Option Nat) : minOpt a none = a := by cases a <;> rfl

theorem minOpt_eq_some_zero {a b : Option Nat} (h : minOpt a b = some 0) : a = some 0 ∨ b = some 0 := by
  cases a with
  | none => right; simpa [minOpt] using h
  | some x =>
    cases b with
    | none => left; simpa [minOpt] using h
    | some y =>
      simp only [minOpt, Option.some.injEq] at h
      have hxy : x = 0 ∨ y = 0 := by omega
      rcases hxy with hx | hy
      · left; simp [hx]
      · right; simp [hy]

theorem minOpt_some_zero_left (b : Option Nat) : minOpt (some 0) b = some 0 := by cases b <;> simp [minOpt]
theorem minOpt_some_zero_right (a : Option Nat) : minOpt a (some 0) = some 0 := by cases a <;> simp [minOpt]

/-- THE PRICED REACH of side `c` on square `s` within `k` of its own moves (the other side passing): `some 0` when `c` bears on `s` already; else, over `c`'s legal moves on the
passed board, the least of (a) the price of landing where the unit bears on `s` (what the other side wins by the exchange there) and (b) a SAFE landing followed by a reach within
`k - 1`; `none` when no path. -/
def reachPrice (c : Color) (s : Square) : Nat → Position → Option Nat
  | 0, p => if 0 < attackerCount p c s then some 0 else none
  | k + 1, p =>
    if 0 < attackerCount p c s then some 0 else
      (legalMoves (passTo c p)).foldl
        (fun acc m =>
          let q := applyMove (passTo c p) m
          let here : Option Nat := if 0 < attackerCount q c s then some (seeOf q m.destination) else none
          let deeper : Option Nat := if landsSafe (passTo c p) m then reachPrice c s k q else none
          minOpt acc (minOpt here deeper))
        none

/-- The fold reaches `some 0` only through one of its steps. -/
private theorem foldl_minOpt_zero {α : Type} (f : α → Option Nat) :
    ∀ (l : List α) (acc : Option Nat), l.foldl (fun a x => minOpt a (f x)) acc = some 0 → acc = some 0 ∨ ∃ x ∈ l, f x = some 0 := by
  intro l
  induction l with
  | nil => intro acc h; left; simpa using h
  | cons x xs ih =>
    intro acc h
    simp only [List.foldl] at h
    rcases ih _ h with h1 | ⟨y, hy, hfy⟩
    · rcases minOpt_eq_some_zero h1 with ha | hx
      · left; exact ha
      · right; exact ⟨x, List.mem_cons_self, hx⟩
    · right; exact ⟨y, List.mem_cons_of_mem _ hy, hfy⟩

/-- And conversely a step at `some 0` pulls the whole fold to `some 0`. -/
private theorem foldl_minOpt_of_zero {α : Type} (f : α → Option Nat) :
    ∀ (l : List α) (acc : Option Nat), (acc = some 0 ∨ ∃ x ∈ l, f x = some 0) → l.foldl (fun a x => minOpt a (f x)) acc = some 0 := by
  intro l
  induction l with
  | nil =>
    intro acc h
    rcases h with h | ⟨_, hx, _⟩
    · simpa using h
    · simp at hx
  | cons x xs ih =>
    intro acc h
    simp only [List.foldl]
    apply ih
    rcases h with h | ⟨y, hy, hfy⟩
    · left; rw [h]; exact minOpt_some_zero_left _
    · rcases List.mem_cons.mp hy with rfl | hy'
      · left; rw [hfy]; exact minOpt_some_zero_right _
      · right; exact ⟨y, hy', hfy⟩

/-- A SAFE landing from which the unit bears on `s` is a price-0 reach of one move (`seeOf` there is 0 by `landsSafe`). -/
theorem landsSafe_see (p : Position) (m : Move) (h : landsSafe p m = true) : seeOf (applyMove p m) m.destination = 0 := by
  simpa [landsSafe] using h

/-- THE LINK (72.1): a reach at price 0 is exactly `attackableIn`. -/
theorem reachPrice_zero_iff (c : Color) (s : Square) : ∀ (k : Nat) (p : Position), reachPrice c s k p = some 0 ↔ attackableIn c s k p = true := by
  intro k
  induction k with
  | zero =>
    intro p
    simp only [reachPrice, attackableIn]
    by_cases h : 0 < attackerCount p c s <;> simp [h]
  | succ k ih =>
    intro p
    simp only [reachPrice, attackableIn]
    by_cases h : 0 < attackerCount p c s
    · simp [h]
    · have hd : decide (0 < attackerCount p c s) = false := by simp [h]
      rw [if_neg h, hd, Bool.false_or, List.any_eq_true]
      constructor
      · intro hf
        rcases foldl_minOpt_zero (fun m =>
            minOpt (if 0 < attackerCount (applyMove (passTo c p) m) c s then some (seeOf (applyMove (passTo c p) m) m.destination) else none)
                   (if landsSafe (passTo c p) m then reachPrice c s k (applyMove (passTo c p) m) else none)) _ none hf with h0 | ⟨m, hm, hm0⟩
        · cases h0
        · refine ⟨m, hm, ?_⟩
          show (landsSafe (passTo c p) m && attackableIn c s k (applyMove (passTo c p) m)) = true
          rcases minOpt_eq_some_zero hm0 with hh | hd'
          · -- the unit bears on s from a landing where the exchange wins nothing: that landing is safe, and attackableIn at depth k holds from its first disjunct
            by_cases ha : 0 < attackerCount (applyMove (passTo c p) m) c s
            · rw [if_pos ha, Option.some.injEq] at hh
              have hs : landsSafe (passTo c p) m = true := by simpa [landsSafe] using hh
              rw [Bool.and_eq_true]
              refine ⟨hs, ?_⟩
              cases k with
              | zero => simpa [attackableIn] using ha
              | succ k' => simp [attackableIn, ha]
            · rw [if_neg ha] at hh
              cases hh
          · by_cases hs : landsSafe (passTo c p) m = true
            · rw [if_pos hs] at hd'
              rw [Bool.and_eq_true]
              exact ⟨hs, (ih _).mp hd'⟩
            · rw [if_neg hs] at hd'
              cases hd'
      · rintro ⟨m, hm, hmk⟩
        simp only [Bool.and_eq_true] at hmk
        obtain ⟨hs0, hk0⟩ := hmk
        have hs : landsSafe (passTo c p) m = true := hs0
        have hk : attackableIn c s k (applyMove (passTo c p) m) = true := hk0
        apply foldl_minOpt_of_zero (fun m =>
            minOpt (if 0 < attackerCount (applyMove (passTo c p) m) c s then some (seeOf (applyMove (passTo c p) m) m.destination) else none)
                   (if landsSafe (passTo c p) m then reachPrice c s k (applyMove (passTo c p) m) else none))
        right
        refine ⟨m, hm, ?_⟩
        show minOpt (if 0 < attackerCount (applyMove (passTo c p) m) c s then some (seeOf (applyMove (passTo c p) m) m.destination) else none)
                    (if landsSafe (passTo c p) m then reachPrice c s k (applyMove (passTo c p) m) else none) = some 0
        rw [if_pos hs, (ih _).mpr hk]
        exact minOpt_some_zero_right _

end Ideas.Reach
