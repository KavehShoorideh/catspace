import Mathlib.Data.Finset.Card
import Mathlib.Data.Finset.Filter

/-!
# Layer 0, logic (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 1)

The Rust `cs_base::logic::Formula` has `every` and `some` over a finite set and `at least so many`. The laws the Rust tests mirror, over a finite set `s`
and a property `P`: De Morgan for the two quantifiers, and "at least so many" as the size of the part where the property holds.
-/

namespace Ideas.Layer0.Logic

set_option linter.unusedSectionVars false

variable {α : Type*} (s : Finset α) (P : α → Prop) [DecidablePred P]

/-- NOT EVERY is SOME NOT (De Morgan over a finite set; Rust: `de_morgan_over_a_finite_set_and_at_least_is_monotone_in_the_count`). -/
theorem not_every_iff_some_not : ¬ (∀ x ∈ s, P x) ↔ ∃ x ∈ s, ¬ P x := by
  push Not; exact Iff.rfl

/-- NOT SOME is EVERY NOT. -/
theorem not_some_iff_every_not : ¬ (∃ x ∈ s, P x) ↔ ∀ x ∈ s, ¬ P x := by
  push Not; exact Iff.rfl

/-- AT LEAST n: the part of the set where the property holds has at least n members. -/
def AtLeast (n : ℕ) : Prop := n ≤ (s.filter P).card

/-- AT LEAST IS MONOTONE IN THE COUNT: at least n + 1 gives at least n. -/
theorem atLeast_mono {n : ℕ} (h : AtLeast s P (n + 1)) : AtLeast s P n := Nat.le_of_succ_le h

/-- AT LEAST ONE is SOME. -/
theorem atLeast_one_iff_some : AtLeast s P 1 ↔ ∃ x ∈ s, P x := by
  unfold AtLeast
  rw [Nat.one_le_iff_ne_zero, Ne, Finset.card_eq_zero, ← Ne, ← Finset.nonempty_iff_ne_empty]
  constructor
  · rintro ⟨x, hx⟩; exact ⟨x, (Finset.mem_filter.mp hx).1, (Finset.mem_filter.mp hx).2⟩
  · rintro ⟨x, hx, hp⟩; exact ⟨x, Finset.mem_filter.mpr ⟨hx, hp⟩⟩

/-- AT LEAST AS MANY AS THE SET HAS is EVERY. -/
theorem atLeast_card_iff_every : AtLeast s P s.card ↔ ∀ x ∈ s, P x := by
  unfold AtLeast
  constructor
  · intro h x hx
    have hsub : s.filter P ⊆ s := Finset.filter_subset _ _
    have heq : s.filter P = s := Finset.eq_of_subset_of_card_le hsub h
    have : x ∈ s.filter P := by rw [heq]; exact hx
    exact (Finset.mem_filter.mp this).2
  · intro h
    have : s.filter P = s := Finset.filter_true_of_mem h
    rw [this]

end Ideas.Layer0.Logic
