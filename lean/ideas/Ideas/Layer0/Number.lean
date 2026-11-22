import Mathlib.Data.Real.Basic
import Mathlib.Order.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp

/-!
# Layer 0, number and order (plan 79.30, 79.31 part D; SYSTEM_SPEC 11 "Layer 0 holds six things" item 3)

The Rust `cs_base::number::Num` is one type for whole numbers, exact fractions (plan 79.31 part D) and real numbers with one total order. Here a number is a whole, a fraction or a real,
and its order is the order of its real value; a real that is a whole number is the same number as the whole number (Rust stores it
as the whole number, so `==`, `Ord` and `Hash` agree). The laws below are the laws the Rust tests mirror, by name.
THE LEAN NUMBER IS A MODEL: its value is a real number; the Rust order is exact on the stored forms. What ties the two is the Rust tests on made cases and the worked examples at the end of this file, which the Rust test of the same name repeats.
-/

namespace Ideas.Layer0.Number

/-- A number: a whole one, a fraction (a positive denominator) or a real one. -/
inductive Num
  | whole (i : Int)
  | frac (n : Int) (d : Nat) (hd : 0 < d)
  | real (r : ℝ)

/-- The value of a number, as a real. -/
noncomputable def Num.val : Num → ℝ
  | .whole i => (i : ℝ)
  | .frac n d _ => (n : ℝ) / (d : ℝ)
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

/-! ## Fractions (plan 79.31 part D) -/

/-- FRACTIONS ADD EXACTLY: a/b + c/d = (a d + c b) / (b d) (Rust: `fractions_are_exact_in_the_four_operations`; the Rust result is in lowest terms, which does not change the value). -/
theorem frac_add_val (a c : Int) (b d : Nat) (hb : 0 < b) (hd : 0 < d) :
    (Num.frac a b hb).val + (Num.frac c d hd).val = (((a * d + c * b : Int) : ℝ)) / ((b * d : Nat) : ℝ) := by
  have hb' : (b : ℝ) ≠ 0 := by positivity
  have hd' : (d : ℝ) ≠ 0 := by positivity
  simp only [Num.val]
  push_cast
  field_simp

/-- FRACTIONS MULTIPLY EXACTLY. -/
theorem frac_mul_val (a c : Int) (b d : Nat) (hb : 0 < b) (hd : 0 < d) :
    (Num.frac a b hb).val * (Num.frac c d hd).val = (((a * c : Int) : ℝ)) / ((b * d : Nat) : ℝ) := by
  simp only [Num.val]
  push_cast
  rw [div_mul_div_comm]

/-- THE ORDER OF TWO FRACTIONS IS DECIDED BY CROSS PRODUCTS, with whole numbers only (why the Rust order is exact with no rounding). -/
theorem frac_le_iff (a c : Int) (b d : Nat) (hb : 0 < b) (hd : 0 < d) :
    (Num.frac a b hb).le (Num.frac c d hd) ↔ a * d ≤ c * b := by
  unfold Num.le
  simp only [Num.val]
  have hb' : (0 : ℝ) < b := by exact_mod_cast hb
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  rw [div_le_div_iff₀ hb' hd']
  constructor
  · intro h; exact_mod_cast h
  · intro h; exact_mod_cast h

/-- A FRACTION WITH DENOMINATOR ONE IS THE WHOLE NUMBER (why the Rust stores it as the whole number, so equal numbers are one). -/
theorem frac_one_whole (n : Int) : (Num.frac n 1 Nat.one_pos).val = (Num.whole n).val := by
  simp [Num.val]

/-- A FRACTION IN NON-LOWEST TERMS IS THE FRACTION IN LOWEST TERMS: scaling both terms keeps the value (why the Rust reduces and `==` agrees). -/
theorem frac_scale (n : Int) (d k : Nat) (hd : 0 < d) (hk : 0 < k) :
    (Num.frac (n * k) (d * k) (Nat.mul_pos hd hk)).val = (Num.frac n d hd).val := by
  have hk' : (k : ℝ) ≠ 0 := by positivity
  simp only [Num.val]
  push_cast
  rw [mul_div_mul_right _ _ hk']

/-! ## Worked examples (plan 79.31 part H): the same numbers as the Rust test `fractions_are_exact_in_the_four_operations` -/

/-- One third and one sixth are one half. -/
theorem example_third_sixth : (Num.frac 1 3 (by norm_num)).val + (Num.frac 1 6 (by norm_num)).val = (Num.frac 1 2 (by norm_num)).val := by
  simp only [Num.val]; norm_num

/-- A tenth and two tenths are three tenths. -/
theorem example_tenths : (Num.frac 1 10 (by norm_num)).val + (Num.frac 2 10 (by norm_num)).val = (Num.frac 3 10 (by norm_num)).val := by
  simp only [Num.val]; norm_num

/-- Two thirds over four ninths are three halves. -/
theorem example_ninths : (Num.frac 2 3 (by norm_num)).val / (Num.frac 4 9 (by norm_num)).val = (Num.frac 3 2 (by norm_num)).val := by
  simp only [Num.val]; norm_num

end Ideas.Layer0.Number
