import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Ring
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Data.Real.Basic

/-!
# Layer 0, chance (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 5)

The Rust `cs_base::chance::Spread` is a finite spread of outcomes with chances between nought and one that sum to one. The laws the Rust tests mirror:
the product of two spreads sums to one, the spread given a condition sums to one, expectation is linear.
-/

open scoped BigOperators

namespace Ideas.Layer0.Chance

variable {α β : Type*}

/-- THE EXPECTATION of a number read off each outcome, with the chances `p`. -/
def expectation (s : Finset α) (p : α → ℝ) (f : α → ℝ) : ℝ := ∑ x ∈ s, p x * f x

/-- EXPECTATION IS ADDITIVE (Rust: `expectation_is_linear`). -/
theorem expectation_add (s : Finset α) (p f g : α → ℝ) :
    expectation s p (fun x => f x + g x) = expectation s p f + expectation s p g := by
  unfold expectation
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun x _ => mul_add _ _ _

/-- EXPECTATION SCALES. -/
theorem expectation_smul (s : Finset α) (p f : α → ℝ) (a : ℝ) :
    expectation s p (fun x => a * f x) = a * expectation s p f := by
  unfold expectation
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun x _ => by ring

/-- EXPECTATION IS LINEAR, in the form the Rust test uses: `a f + b g`. -/
theorem expectation_linear (s : Finset α) (p f g : α → ℝ) (a b : ℝ) :
    expectation s p (fun x => a * f x + b * g x) = a * expectation s p f + b * expectation s p g := by
  rw [expectation_add, expectation_smul, expectation_smul]

/-- EACH CHANCE IS AT MOST ONE when the chances are not negative and sum to one. -/
theorem chance_le_one (s : Finset α) (p : α → ℝ) (h0 : ∀ x ∈ s, 0 ≤ p x) (h1 : ∑ x ∈ s, p x = 1) {x : α} (hx : x ∈ s) : p x ≤ 1 := by
  rw [← h1]
  exact Finset.single_le_sum h0 hx

/-- THE PRODUCT OF TWO SPREADS THAT DO NOT DEPEND ON EACH OTHER sums to one (Rust: `a_product_and_a_spread_given_a_condition_sum_to_one`). -/
theorem product_sums_to_one (s : Finset α) (t : Finset β) (p : α → ℝ) (q : β → ℝ)
    (hp : ∑ x ∈ s, p x = 1) (hq : ∑ y ∈ t, q y = 1) : ∑ z ∈ s ×ˢ t, p z.1 * q z.2 = 1 := by
  rw [Finset.sum_product]
  simp_rw [← Finset.mul_sum, hq, mul_one]
  exact hp

/-- THE SPREAD GIVEN A CONDITION sums to one: the chances of the outcomes where the condition holds, each divided by their sum, when that sum is not nought. -/
theorem given_sums_to_one (s : Finset α) (p : α → ℝ) (P : α → Prop) [DecidablePred P]
    (hc : ∑ x ∈ s.filter P, p x ≠ 0) : ∑ x ∈ s.filter P, p x / (∑ y ∈ s.filter P, p y) = 1 := by
  rw [← Finset.sum_div]
  exact div_self hc

/-- THE CHANCES OF A SPREAD GIVEN A CONDITION are not negative when the chances are not (and the sum is positive). -/
theorem given_nonneg (s : Finset α) (p : α → ℝ) (P : α → Prop) [DecidablePred P]
    (h0 : ∀ x ∈ s, 0 ≤ p x) {x : α} (hx : x ∈ s.filter P) : 0 ≤ p x / (∑ y ∈ s.filter P, p y) := by
  apply div_nonneg (h0 x (Finset.mem_filter.mp hx).1)
  exact Finset.sum_nonneg fun y hy => h0 y (Finset.mem_filter.mp hy).1

end Ideas.Layer0.Chance
