import Mathlib.Data.Real.Basic
import Mathlib.Order.Basic
import Mathlib.Tactic.Linarith

/-!
# Layer 0, number and order (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 3)

The Rust `cs_base::number::Num` is one type for whole and real numbers with one total order. Here a number is a whole or a real,
and its order is the order of its real value; a real that is a whole number is the same number as the whole number (Rust stores it
as the whole number, so `==`, `Ord` and `Hash` agree). The laws below are the laws the Rust tests mirror, by name.
-/

namespace Ideas.Layer0.Number

/-- A number: a whole one or a real one. -/
inductive Num
  | whole (i : Int)
  | real (r : ℝ)

/-- The value of a number, as a real. -/
def Num.val : Num → ℝ
  | .whole i => (i : ℝ)
  | .real r => r

/-- The one order of numbers: by value. -/
def Num.le (a b : Num) : Prop := a.val ≤ b.val

/-- THE ORDER IS TOTAL (Rust: `the_order_is_total_antisymmetric_and_transitive`). -/
theorem order_total (a b : Num) : a.le b ∨ b.le a := le_total a.val b.val

/-- ANTISYMMETRIC: two numbers each at most the other have one value, so they are one number. -/
theorem order_antisymm {a b : Num} (h₁ : a.le b) (h₂ : b.le a) : a.val = b.val := le_antisymm h₁ h₂

/-- TRANSITIVE. -/
theorem order_trans {a b c : Num} (h₁ : a.le b) (h₂ : b.le c) : a.le c := le_trans h₁ h₂

/-- A real that is a whole number is the whole number: equal values are one number (why Rust stores it as the whole number). -/
theorem real_whole_same (i : Int) : (Num.real (i : ℝ)).val = (Num.whole i).val := rfl

/-- ADDITION IS MONOTONE IN THE ORDER (Rust: `addition_is_monotone_in_the_order`). -/
theorem add_monotone {a b : Num} (c : Num) (h : a.le b) : a.val + c.val ≤ b.val + c.val :=
  by unfold Num.le at h; linarith

/-- The least of two numbers is not above either (what `Ord::min` gives). -/
theorem min_le_left_val (a b : Num) : min a.val b.val ≤ a.val := min_le_left _ _

end Ideas.Layer0.Number
