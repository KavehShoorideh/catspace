import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.BigOperators.Ring.Finset

/-!
# Layer 0, opposed choice (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 6)

Two laws about the finite trees of `cs_base::choice`:

1. In a finite tree of choices IN TURN (no chance) exactly one of "A can force P" and "B can force not-P" holds (`exactly_one`).
2. In a finite tree with choices in turn, choices AT ONCE (pure choices only) and choices by chance, the LOWER value is not above the UPPER value (`low_le_up`).

and the link between the two readings: in a tree of turns the value reaches `k` exactly when A can force "at least `k`" (`value_ge_iff_canForce`).
Nothing here is mixed choice: that is not built, and the laws say nothing of it.
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

end Ideas.Layer0.Choice
