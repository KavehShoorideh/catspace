import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FinCases
import Mathlib.Topology.Sion
import Mathlib.Analysis.Convex.StdSimplex
import Mathlib.Analysis.Convex.Function
import Mathlib.Topology.Algebra.Module.FiniteDimension

/-!
# Layer 0, opposed choice (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 6)

Two laws about the finite trees of `cs_base::choice`:

1. In a finite tree of choices IN TURN (no chance) exactly one of "A can force P" and "B can force not-P" holds (`exactly_one`).
2. In a finite tree with choices in turn, choices AT ONCE (pure choices only) and choices by chance, the LOWER value is not above the UPPER value (`low_le_up`).

and the link between the two readings: in a tree of turns the value reaches `k` exactly when A can force "at least `k`" (`value_ge_iff_canForce`).
Plan 79.31 adds: determinacy for a SET of choosers (`coalition_exactly_one`), the certificate law of the mixed choice (`guarantees_meet_of_certificate`), the law of a choice against a belief (`no_mix_beats_best`), and worked examples whose numbers the Rust tests repeat.
Plan 79.32 adds the laws the walk of `cs_base::choice::walk` keeps its answers by: met within n steps is met within every depth from n on (`canIn_mono`), not met within m steps is not met within any fewer (`not_canIn_of_le`), and the set can force P exactly when it can within some number of steps (`canS_iff_canIn`).
THE MINIMAX THEOREM (that every table has spreads that meet, `minimax_exists`) IS PROVED, from Mathlib's Sion-von Neumann theorem (`Sion.exists_isSaddlePointOn`) on the two simplexes with the bilinear payoff; the Rust program that finds the spreads has the certificate (`guarantees_meet_of_certificate`) tested on every table it solves.
THE LEAN TREES ARE MODELS: the Rust `Tree` is a separate type; what ties them is the Rust tests on made cases and the worked examples, which the Rust tests repeat with the same numbers.
-/

namespace Ideas.Layer0.Choice

/-! ## Turns only: determinacy -/

/-- A tree of choices in turn; a result is true or false (the property P holds of it or not). -/
inductive TTree
  | leaf (b : Bool)
  | turnA (n : ℕ) (c : Fin n → TTree)
  | turnB (n : ℕ) (c : Fin n → TTree)

/-- A CAN FORCE P: at A's turn some option does; at B's turn every option does. -/
def canA : TTree → Prop
  | .leaf b => b = true
  | .turnA _ c => ∃ i, canA (c i)
  | .turnB _ c => ∀ i, canA (c i)

/-- B CAN FORCE NOT-P: at B's turn some option does; at A's turn every option does. -/
def canB : TTree → Prop
  | .leaf b => b = false
  | .turnA _ c => ∀ i, canB (c i)
  | .turnB _ c => ∃ i, canB (c i)

/-- EXACTLY ONE OF "A CAN FORCE P" AND "B CAN FORCE NOT-P" HOLDS in a finite tree of turns without chance
(Rust: `in_a_tree_of_turns_exactly_one_of_a_forces_p_and_b_forces_not_p`). -/
theorem exactly_one (t : TTree) : canA t ↔ ¬ canB t := by
  induction t with
  | leaf b => cases b <;> simp [canA, canB]
  | turnA n c ih => simp only [canA, canB, ih, not_forall]
  | turnB n c ih => simp only [canA, canB, ih, not_exists]

/-! ## Turns only: the value against the forcing reading -/

/-- A tree of turns with results that are real numbers. -/
inductive VT
  | leaf (v : ℝ)
  | turnA (n : ℕ) (hn : 0 < n) (c : Fin n → VT)
  | turnB (n : ℕ) (hn : 0 < n) (c : Fin n → VT)

private theorem fin_nonempty {n : ℕ} (hn : 0 < n) : (Finset.univ : Finset (Fin n)).Nonempty := ⟨⟨0, hn⟩, Finset.mem_univ _⟩

/-- The value for A: A takes the best option at its turn, B the worst. -/
noncomputable def val : VT → ℝ
  | .leaf v => v
  | .turnA _ hn c => (Finset.univ.sup' (fin_nonempty hn) fun i => val (c i))
  | .turnB _ hn c => (Finset.univ.inf' (fin_nonempty hn) fun i => val (c i))

/-- A can force "at least k". -/
def canForce (k : ℝ) : VT → Prop
  | .leaf v => k ≤ v
  | .turnA _ _ c => ∃ i, canForce k (c i)
  | .turnB _ _ c => ∀ i, canForce k (c i)

/-- THE VALUE OF A TREE OF TURNS AGAINST THE FORCING READING: the value reaches `k` exactly when A can force "at least k"
(Rust: `the_value_of_a_tree_of_turns_agrees_with_the_forcing_reading`). -/
theorem value_ge_iff_canForce (k : ℝ) (t : VT) : k ≤ val t ↔ canForce k t := by
  induction t with
  | leaf v => simp [val, canForce]
  | turnA n hn c ih => simp only [val, canForce, Finset.le_sup'_iff, Finset.mem_univ, true_and, ih]
  | turnB n hn c ih => simp only [val, canForce, Finset.le_inf'_iff, Finset.mem_univ, forall_true_left, ih]

/-! ## Turns, choices at once (pure) and chance: the lower value is not above the upper -/

/-- A tree with every kind of node. A table: A picks a row, B a column. A chance node: non-negative weights. -/
inductive CT
  | leaf (v : ℝ)
  | turnA (n : ℕ) (hn : 0 < n) (c : Fin n → CT)
  | turnB (n : ℕ) (hn : 0 < n) (c : Fin n → CT)
  | table (r k : ℕ) (hr : 0 < r) (hk : 0 < k) (cells : Fin r → Fin k → CT)
  | chance (n : ℕ) (w : Fin n → ℝ) (hw : ∀ i, 0 ≤ w i) (c : Fin n → CT)

/-- THE LOWER VALUE for A: what A can make sure of (at once, pure choices: the best row against the worst column). -/
noncomputable def low : CT → ℝ
  | .leaf v => v
  | .turnA _ hn c => Finset.univ.sup' (fin_nonempty hn) fun i => low (c i)
  | .turnB _ hn c => Finset.univ.inf' (fin_nonempty hn) fun i => low (c i)
  | .table _ _ hr hk cells => Finset.univ.sup' (fin_nonempty hr) fun i => Finset.univ.inf' (fin_nonempty hk) fun j => low (cells i j)
  | .chance n w _ c => ∑ i : Fin n, w i * low (c i)

/-- THE UPPER VALUE for A: what B can hold A down to (at once, pure choices: the best column for B against A's best row). -/
noncomputable def up : CT → ℝ
  | .leaf v => v
  | .turnA _ hn c => Finset.univ.sup' (fin_nonempty hn) fun i => up (c i)
  | .turnB _ hn c => Finset.univ.inf' (fin_nonempty hn) fun i => up (c i)
  | .table _ _ hr hk cells => Finset.univ.inf' (fin_nonempty hk) fun j => Finset.univ.sup' (fin_nonempty hr) fun i => up (cells i j)
  | .chance n w _ c => ∑ i : Fin n, w i * up (c i)

/-- FOR A CHOICE AT ONCE THE LOWER VALUE IS NOT ABOVE THE UPPER (Rust: `the_lower_value_is_not_above_the_upper_and_they_meet_without_a_choice_at_once`). -/
theorem low_le_up (t : CT) : low t ≤ up t := by
  induction t with
  | leaf v => simp [low, up]
  | turnA n hn c ih =>
    simp only [low, up]
    exact Finset.sup'_le _ _ fun i _ => (ih i).trans (Finset.le_sup' (fun i => up (c i)) (Finset.mem_univ i))
  | turnB n hn c ih =>
    simp only [low, up]
    exact Finset.le_inf' _ _ fun i _ => (Finset.inf'_le (fun i => low (c i)) (Finset.mem_univ i)).trans (ih i)
  | table r k hr hk cells ih =>
    simp only [low, up]
    apply Finset.sup'_le
    intro i _
    apply Finset.le_inf'
    intro j _
    exact (Finset.inf'_le (fun j => low (cells i j)) (Finset.mem_univ j)).trans
      ((ih i j).trans (Finset.le_sup' (fun i => up (cells i j)) (Finset.mem_univ i)))
  | chance n w hw c ih =>
    simp only [low, up]
    exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (ih i) (hw i)

/-- A tree with no choice at once. -/
def tableFree : CT → Prop
  | .leaf _ => True
  | .turnA _ _ c => ∀ i, tableFree (c i)
  | .turnB _ _ c => ∀ i, tableFree (c i)
  | .table .. => False
  | .chance _ _ _ c => ∀ i, tableFree (c i)

/-- WITHOUT A CHOICE AT ONCE THE TWO VALUES ARE ONE (Rust: the same test, `... they_meet_without_a_choice_at_once`). -/
theorem low_eq_up_of_tableFree : ∀ t : CT, tableFree t → low t = up t := by
  intro t
  induction t with
  | leaf v => intro _; simp [low, up]
  | turnA n hn c ih => intro h; simp only [low, up]; exact Finset.sup'_congr _ rfl fun i _ => ih i (h i)
  | turnB n hn c ih => intro h; simp only [low, up]; exact Finset.inf'_congr _ rfl fun i _ => ih i (h i)
  | table r k hr hk cells ih => intro h; exact h.elim
  | chance n w hw c ih => intro h; simp only [low, up]; exact Finset.sum_congr rfl fun i _ => by rw [ih i (h i)]

/-! ## A set of choosers (plan 79.31 part A): determinacy for a coalition -/

/-- A tree of choices in turn by any number of choosers (each identified by a number). -/
inductive GTree
  | leaf (b : Bool)
  | turn (who : ℕ) (n : ℕ) (c : Fin n → GTree)

open Classical in
/-- THE SET S CAN FORCE P: at the turn of a member some option does; at any other turn every option does. -/
noncomputable def canS (S : ℕ → Prop) : GTree → Prop
  | .leaf b => b = true
  | .turn w _ c => if S w then ∃ i, canS S (c i) else ∀ i, canS S (c i)

open Classical in
/-- THE REST CAN FORCE NOT-P: at the turn of a member of S every option does; at any other turn some option does. -/
noncomputable def canRest (S : ℕ → Prop) : GTree → Prop
  | .leaf b => b = false
  | .turn w _ c => if S w then ∀ i, canRest S (c i) else ∃ i, canRest S (c i)

/-- EXACTLY ONE OF "THE SET CAN FORCE P" AND "THE REST CAN FORCE NOT-P" HOLDS, for any set of choosers: the case of one chooser against one is `exactly_one`
(Rust: `a_set_of_one_is_the_old_reading_and_a_set_of_allied_choosers_can_force_what_none_can_alone`, and `the_set_and_the_rest_are_determined_with_any_number_of_choosers`). -/
theorem coalition_exactly_one (S : ℕ → Prop) (t : GTree) : canS S t ↔ ¬ canRest S t := by
  induction t with
  | leaf b => cases b <;> simp [canS, canRest]
  | turn w n c ih =>
    by_cases h : S w
    · simp only [canS, canRest, h, if_true, ih, not_forall]
    · simp only [canS, canRest, h, if_false, ih, not_exists]

/-! ## The walk of layer 0 (plan 79.32 part A): met within n steps, and what the walk keeps -/

open Classical in
/-- THE SET S CAN FORCE P WITHIN k STEPS, the reading of `cs_base::choice::walk::Walk`: a result decides; a node that is not a result and has no step left does not meet P; at the turn of a member some option does within one step fewer;
at any other turn every option does. (A choice in turn has at least one option in the Rust tree; a node with none is the same here as in `canS`.) -/
noncomputable def canIn (S : ℕ → Prop) : ℕ → GTree → Prop
  | _, .leaf b => b = true
  | 0, .turn _ _ _ => False
  | k + 1, .turn w _ c => if S w then ∃ i, canIn S k (c i) else ∀ i, canIn S k (c i)

/-- MET WITHIN k STEPS IS MET WITHIN k + 1. -/
theorem canIn_succ (S : ℕ → Prop) : ∀ (k : ℕ) (t : GTree), canIn S k t → canIn S (k + 1) t
  | _, .leaf b, h => by simpa [canIn] using h
  | 0, .turn _ _ _, h => by simp [canIn] at h
  | k + 1, .turn w _ c, h => by
      by_cases hw : S w
      · simp only [canIn, hw, if_true] at h ⊢
        obtain ⟨i, hi⟩ := h
        exact ⟨i, canIn_succ S k (c i) hi⟩
      · simp only [canIn, hw, if_false] at h ⊢
        exact fun i => canIn_succ S k (c i) (h i)

/-- WHAT THE WALK KEEPS WHEN A PROPERTY IS MET: met within n steps, it is met within every depth from n on (Rust: the kept entry `reach`, used where `reach <= depth`; `walk.rs`). -/
theorem canIn_mono (S : ℕ → Prop) {k m : ℕ} (hkm : k ≤ m) (t : GTree) (h : canIn S k t) : canIn S m t := by
  induction hkm with
  | refl => exact h
  | step _ ih => exact canIn_succ S _ t ih

/-- WHAT THE WALK KEEPS WHEN A PROPERTY IS NOT MET: not met within m steps, it is not met within any fewer (Rust: the kept entry `fail`, used where `fail >= depth`). -/
theorem not_canIn_of_le (S : ℕ → Prop) {k m : ℕ} (hkm : k ≤ m) (t : GTree) (h : ¬ canIn S m t) : ¬ canIn S k t :=
  fun hk => h (canIn_mono S hkm t hk)

/-- THE SET CAN FORCE P EXACTLY WHEN IT CAN WITHIN SOME NUMBER OF STEPS: a depth large enough reads the whole finite tree, so the walk with no limit is the reading `canS`
(Rust: `can_force_set` is the walk over the explicit tree; `the_set_and_the_rest_are_determined_with_any_number_of_choosers`). -/
theorem canS_iff_canIn (S : ℕ → Prop) (t : GTree) : canS S t ↔ ∃ k, canIn S k t := by
  induction t with
  | leaf b => simp [canS, canIn]
  | turn w n c ih =>
    by_cases hw : S w
    · simp only [canS, hw, if_true]
      constructor
      · rintro ⟨i, hi⟩
        obtain ⟨k, hk⟩ := (ih i).1 hi
        refine ⟨k + 1, ?_⟩
        simp only [canIn, hw, if_true]
        exact ⟨i, hk⟩
      · rintro ⟨k, hk⟩
        cases k with
        | zero => simp [canIn] at hk
        | succ k =>
          simp only [canIn, hw, if_true] at hk
          obtain ⟨i, hi⟩ := hk
          exact ⟨i, (ih i).2 ⟨k, hi⟩⟩
    · simp only [canS, hw, if_false]
      constructor
      · intro h
        choose f hf using fun i => (ih i).1 (h i)
        refine ⟨Finset.univ.sup f + 1, ?_⟩
        simp only [canIn, hw, if_false]
        exact fun i => canIn_mono S (Finset.le_sup (f := f) (Finset.mem_univ i)) _ (hf i)
      · rintro ⟨k, hk⟩
        cases k with
        | zero => simp [canIn] at hk
        | succ k =>
          simp only [canIn, hw, if_false] at hk
          exact fun i => (ih i).2 ⟨k, hk i⟩

/-! ## The mixed choice (plan 79.31 part B): the certificate -/

/-- THE GUARANTEES MEET. If the row chooser's spread `x` gets at least `v` against every column and the column chooser's spread `y` gives at most `v` against every row, then no spread of the row chooser
guarantees more than `v` (first half) and no spread of the column chooser holds the row chooser to less than `v` (second half): `v` is THE value (Rust: `the_guarantees_of_the_two_choosers_meet_on_made_tables` checks this
certificate on every table the program solves). That every table HAS such spreads is `minimax_exists` below. -/
theorem guarantees_meet_of_certificate {m n : ℕ} (A : Fin m → Fin n → ℝ) (x : Fin m → ℝ) (y : Fin n → ℝ) (v : ℝ)
    (hy0 : ∀ j, 0 ≤ y j) (hy1 : ∑ j, y j = 1) (hx0 : ∀ i, 0 ≤ x i) (hx1 : ∑ i, x i = 1)
    (hrow : ∀ j, v ≤ ∑ i, x i * A i j) (hcol : ∀ i, ∑ j, A i j * y j ≤ v) :
    (∀ x' : Fin m → ℝ, (∀ i, 0 ≤ x' i) → ∑ i, x' i = 1 → ∀ w, (∀ j, w ≤ ∑ i, x' i * A i j) → w ≤ v) ∧
    (∀ y' : Fin n → ℝ, (∀ j, 0 ≤ y' j) → ∑ j, y' j = 1 → ∀ w, (∀ i, ∑ j, A i j * y' j ≤ w) → v ≤ w) := by
  constructor
  · intro x' hx0' hx1' w hw
    calc w = ∑ j, y j * w := by rw [← Finset.sum_mul, hy1, one_mul]
      _ ≤ ∑ j, y j * ∑ i, x' i * A i j := Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hw j) (hy0 j)
      _ = ∑ i, x' i * ∑ j, A i j * y j := by
          simp_rw [Finset.mul_sum]
          rw [Finset.sum_comm]
          exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
      _ ≤ ∑ i, x' i * v := Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hcol i) (hx0' i)
      _ = v := by rw [← Finset.sum_mul, hx1', one_mul]
  · intro y' hy0' hy1' w hw
    calc v = ∑ j, y' j * v := by rw [← Finset.sum_mul, hy1', one_mul]
      _ ≤ ∑ j, y' j * ∑ i, x i * A i j := Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hrow j) (hy0' j)
      _ = ∑ i, x i * ∑ j, A i j * y' j := by
          simp_rw [Finset.mul_sum]
          rw [Finset.sum_comm]
          exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
      _ ≤ ∑ i, x i * w := Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hw i) (hx0 i)
      _ = w := by rw [← Finset.sum_mul, hx1, one_mul]

/-- TWO CERTIFICATES GIVE ONE VALUE: if two pairs of spreads each have the meeting property, with values `v` and `v'`, then `v = v'`. -/
theorem certificate_value_unique {m n : ℕ} (A : Fin m → Fin n → ℝ) (x x' : Fin m → ℝ) (y y' : Fin n → ℝ) (v v' : ℝ)
    (hy0 : ∀ j, 0 ≤ y j) (hy1 : ∑ j, y j = 1) (hx0 : ∀ i, 0 ≤ x i) (hx1 : ∑ i, x i = 1)
    (hrow : ∀ j, v ≤ ∑ i, x i * A i j) (hcol : ∀ i, ∑ j, A i j * y j ≤ v)
    (hy0' : ∀ j, 0 ≤ y' j) (hy1' : ∑ j, y' j = 1) (hx0' : ∀ i, 0 ≤ x' i) (hx1' : ∑ i, x' i = 1)
    (hrow' : ∀ j, v' ≤ ∑ i, x' i * A i j) (hcol' : ∀ i, ∑ j, A i j * y' j ≤ v') : v = v' := by
  have h := guarantees_meet_of_certificate A x y v hy0 hy1 hx0 hx1 hrow hcol
  have h' := guarantees_meet_of_certificate A x' y' v' hy0' hy1' hx0' hx1' hrow' hcol'
  exact le_antisymm (h.2 y' hy0' hy1' v' hcol') (h'.2 y hy0 hy1 v hcol)

/-! ## A choice against a belief (plan 79.31 part C): no spread of the options beats the best option -/

/-- AGAINST A FIXED BELIEF A SPREAD OF THE OPTIONS DOES NOT BEAT THE BEST OPTION: a weighted mean is not above its greatest part
(Rust: `against_a_fixed_belief_a_single_option_is_best_and_no_mixing_gains`). -/
theorem no_mix_beats_best {ι : Type*} (s : Finset ι) (hne : s.Nonempty) (w e : ι → ℝ) (hw0 : ∀ i ∈ s, 0 ≤ w i) (hw1 : ∑ i ∈ s, w i = 1) :
    ∑ i ∈ s, w i * e i ≤ s.sup' hne e := by
  calc ∑ i ∈ s, w i * e i ≤ ∑ i ∈ s, w i * s.sup' hne e :=
        Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_left (Finset.le_sup' e hi) (hw0 i hi)
    _ = s.sup' hne e := by rw [← Finset.sum_mul, hw1, one_mul]

/-- AND THE BEST OPTION IS ONE OF THE OPTIONS: some single option has the greatest expectation. -/
theorem best_is_an_option {ι : Type*} (s : Finset ι) (hne : s.Nonempty) (e : ι → ℝ) : ∃ i ∈ s, e i = s.sup' hne e :=
  (Finset.exists_mem_eq_sup' hne e).imp fun _ h => ⟨h.1, h.2.symm⟩

/-! ## Worked examples (plan 79.31 part H): the same trees and the same numbers as the Rust tests -/

private theorem sup'_two (f : Fin 2 → ℝ) (h : (Finset.univ : Finset (Fin 2)).Nonempty) : Finset.univ.sup' h f = max (f 0) (f 1) := by
  apply le_antisymm
  · apply Finset.sup'_le
    intro i _
    fin_cases i <;> simp
  · exact max_le (Finset.le_sup' f (Finset.mem_univ _)) (Finset.le_sup' f (Finset.mem_univ _))

private theorem inf'_two (f : Fin 2 → ℝ) (h : (Finset.univ : Finset (Fin 2)).Nonempty) : Finset.univ.inf' h f = min (f 0) (f 1) := by
  apply le_antisymm
  · exact le_min (Finset.inf'_le f (Finset.mem_univ _)) (Finset.inf'_le f (Finset.mem_univ _))
  · apply Finset.le_inf'
    intro i _
    fin_cases i <;> simp

/-- MATCHING PENNIES: the table [[1, -1], [-1, 1]] has the lower value -1 and the upper value 1
(Rust: `matching_pennies_has_no_pure_saddle_and_the_mixed_choice_is_worth_nought`). -/
theorem example_pennies_low :
    low (.table 2 2 (by norm_num) (by norm_num) ![![.leaf 1, .leaf (-1)], ![.leaf (-1), .leaf 1]]) = -1 := by
  simp [low, sup'_two, inf'_two]

theorem example_pennies_up :
    up (.table 2 2 (by norm_num) (by norm_num) ![![.leaf 1, .leaf (-1)], ![.leaf (-1), .leaf 1]]) = 1 := by
  simp [up, sup'_two, inf'_two]

/-- A TABLE WITH A SADDLE: [[3, 2], [1, 0]] has the value 2, both ways (the same Rust test). -/
theorem example_saddle_low :
    low (.table 2 2 (by norm_num) (by norm_num) ![![.leaf 3, .leaf 2], ![.leaf 1, .leaf 0]]) = 2 := by
  simp [low, sup'_two, inf'_two]; norm_num

theorem example_saddle_up :
    up (.table 2 2 (by norm_num) (by norm_num) ![![.leaf 3, .leaf 2], ![.leaf 1, .leaf 0]]) = 2 := by
  simp [up, sup'_two, inf'_two]; norm_num

/-- A COIN OR A SURE ONE: A picks between a coin (1/2 for 4, 1/2 for 0) and a sure 1: the value is 2
(Rust: `a_tree_with_chance_is_weighed_by_the_chances_and_forcing_says_the_least_chance`). -/
theorem example_coin_low :
    low (.turnA 2 (by norm_num) ![.chance 2 ![1/2, 1/2] (by intro i; fin_cases i <;> norm_num) ![.leaf 4, .leaf 0], .leaf 1]) = 2 := by
  simp [low, sup'_two, Fin.sum_univ_two]; norm_num

/-! ## The minimax theorem for a table (plan 79.31 part B) -/

/-- THE MINIMAX THEOREM FOR A TABLE: every table of real numbers with a row and a column has a spread of the rows and a spread of the columns and a number `v` such that the row spread gets at least `v` against every column
and the column spread gives at most `v` against every row (so, by `guarantees_meet_of_certificate`, `v` is the value). PROVED by Mathlib's Sion-von Neumann theorem: the two simplexes are convex and compact and the payoff is
bilinear, so continuous and linear in each spread; a saddle point read at the pure rows and columns gives the two guarantees. (Rust: the program `solve_matrix_game` finds such spreads exactly, and
`the_guarantees_of_the_two_choosers_meet_on_made_tables` tests the certificate on every table it solves.) -/
theorem minimax_exists {m n : ℕ} (hm : 0 < m) (hn : 0 < n) (A : Fin m → Fin n → ℝ) :
    ∃ (x : Fin m → ℝ) (y : Fin n → ℝ) (v : ℝ), (∀ i, 0 ≤ x i) ∧ ∑ i, x i = 1 ∧ (∀ j, 0 ≤ y j) ∧ ∑ j, y j = 1 ∧
      (∀ j, v ≤ ∑ i, x i * A i j) ∧ (∀ i, ∑ j, A i j * y j ≤ v) := by
  classical
  haveI : Nonempty (Fin m) := ⟨⟨0, hm⟩⟩
  haveI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
  let f : (Fin n → ℝ) → (Fin m → ℝ) → ℝ := fun c r => ∑ i, ∑ j, r i * (A i j * c j)
  have hfc : ∀ r : Fin m → ℝ, ∃ L : (Fin n → ℝ) →ₗ[ℝ] ℝ, ∀ c, L c = f c r := fun r =>
    ⟨∑ j, (∑ i, r i * A i j) • (LinearMap.proj j : (Fin n → ℝ) →ₗ[ℝ] ℝ), fun c => by
      simp only [f, LinearMap.sum_apply, LinearMap.smul_apply, LinearMap.proj_apply, smul_eq_mul]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [Finset.sum_mul]
      exact Finset.sum_congr rfl fun i _ => by ring⟩
  have hfr : ∀ c : Fin n → ℝ, ∃ L : (Fin m → ℝ) →ₗ[ℝ] ℝ, ∀ r, L r = f c r := fun c =>
    ⟨∑ i, (∑ j, A i j * c j) • (LinearMap.proj i : (Fin m → ℝ) →ₗ[ℝ] ℝ), fun r => by
      simp only [f, LinearMap.sum_apply, LinearMap.smul_apply, LinearMap.proj_apply, smul_eq_mul]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [Finset.sum_mul]
      exact Finset.sum_congr rfl fun j _ => by ring⟩
  have cont_c : ∀ r : Fin m → ℝ, Continuous fun c : Fin n → ℝ => f c r := fun r => by
    obtain ⟨L, hL⟩ := hfc r
    have : (fun c => f c r) = fun c => L c := funext fun c => (hL c).symm
    rw [this]
    exact L.continuous_of_finiteDimensional
  have cont_r : ∀ c : Fin n → ℝ, Continuous fun r : Fin m → ℝ => f c r := fun c => by
    obtain ⟨L, hL⟩ := hfr c
    have : (fun r => f c r) = fun r => L r := funext fun r => (hL r).symm
    rw [this]
    exact L.continuous_of_finiteDimensional
  have hX := convex_stdSimplex ℝ (Fin n)
  have hY := convex_stdSimplex ℝ (Fin m)
  obtain ⟨a, ha, b, hb, hab⟩ := Sion.exists_isSaddlePointOn (X := stdSimplex ℝ (Fin n)) (Y := stdSimplex ℝ (Fin m)) (f := f)
    ⟨_, single_mem_stdSimplex ℝ (⟨0, hn⟩ : Fin n)⟩ hX (isCompact_stdSimplex ℝ (Fin n))
    (fun r _ => (cont_c r).lowerSemicontinuous.lowerSemicontinuousOn _)
    (fun r _ => by
      obtain ⟨L, hL⟩ := hfc r
      have : (fun c => f c r) = fun c => L c := funext fun c => (hL c).symm
      rw [this]; exact (L.convexOn hX).quasiconvexOn)
    hY ⟨_, single_mem_stdSimplex ℝ (⟨0, hm⟩ : Fin m)⟩ (isCompact_stdSimplex ℝ (Fin m))
    (fun c _ => (cont_r c).upperSemicontinuous.upperSemicontinuousOn _)
    (fun c _ => by
      obtain ⟨L, hL⟩ := hfr c
      have : (fun r => f c r) = fun r => L r := funext fun r => (hL r).symm
      rw [this]; exact (L.concaveOn hY).quasiconcaveOn)
  have hrow : ∀ i : Fin m, ∑ j, A i j * a j ≤ f a b := by
    intro i
    have h := hab a ha (Pi.single i 1) (single_mem_stdSimplex ℝ i)
    simpa [f, Pi.single_apply] using h
  have hcol : ∀ j : Fin n, f a b ≤ ∑ i, b i * A i j := by
    intro j
    have h := hab (Pi.single j 1) (single_mem_stdSimplex ℝ j) b hb
    simpa [f, Pi.single_apply] using h
  exact ⟨b, a, f a b, hb.1, hb.2, ha.1, ha.2, hcol, hrow⟩

end Ideas.Layer0.Choice
