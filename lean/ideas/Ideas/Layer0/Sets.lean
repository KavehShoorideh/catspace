import Mathlib.Data.Finset.Card
import Mathlib.Data.Finset.Lattice.Fold

/-!
# Layer 0, sets and counting (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 2)

The Rust `cs_base::sets::Set` is a finite set; union, meet (intersection), part-of (subset) and how many (card). The Rust tests name the theorems below.
-/

namespace Ideas.Layer0.Sets

set_option linter.unusedSectionVars false

variable {α : Type*} [DecidableEq α]

theorem union_comm (a b : Finset α) : a ∪ b = b ∪ a := Finset.union_comm a b
theorem meet_comm (a b : Finset α) : a ∩ b = b ∩ a := Finset.inter_comm a b
theorem union_assoc (a b c : Finset α) : a ∪ b ∪ c = a ∪ (b ∪ c) := Finset.union_assoc a b c
theorem meet_assoc (a b c : Finset α) : a ∩ b ∩ c = a ∩ (b ∩ c) := Finset.inter_assoc a b c
theorem union_idem (a : Finset α) : a ∪ a = a := Finset.union_self a
theorem meet_idem (a : Finset α) : a ∩ a = a := Finset.inter_self a

/-- The meet distributes over the union. -/
theorem meet_union_distrib (a b c : Finset α) : a ∩ (b ∪ c) = a ∩ b ∪ a ∩ c := by
  ext x; simp only [Finset.mem_inter, Finset.mem_union]; tauto

/-- The union distributes over the meet. -/
theorem union_meet_distrib (a b c : Finset α) : a ∪ b ∩ c = (a ∪ b) ∩ (a ∪ c) := by
  ext x; simp only [Finset.mem_inter, Finset.mem_union]; tauto

/-- PART-OF IS THE ORDER OF THE UNION. -/
theorem part_iff_union (a b : Finset α) : a ⊆ b ↔ a ∪ b = b := by
  constructor
  · intro h; ext x; simp only [Finset.mem_union]; constructor
    · rintro (hx | hx); exact h hx; exact hx
    · intro hx; exact Or.inr hx
  · intro h x hx; rw [← h]; exact Finset.mem_union_left _ hx

/-- PART-OF IS THE ORDER OF THE MEET. -/
theorem part_iff_meet (a b : Finset α) : a ⊆ b ↔ a ∩ b = a := by
  constructor
  · intro h; ext x; simp only [Finset.mem_inter]; constructor
    · exact fun hx => hx.1
    · intro hx; exact ⟨hx, h hx⟩
  · intro h x hx; rw [← h] at hx; exact (Finset.mem_inter.mp hx).2

/-- HOW MANY OF A UNION is the sum less how many of the meet (stated without the subtraction). -/
theorem card_union_add_card_meet (a b : Finset α) : (a ∪ b).card + (a ∩ b).card = a.card + b.card :=
  Finset.card_union_add_card_inter a b

/-- A PART IS NO BIGGER THAN THE WHOLE. -/
theorem card_le_of_part {a b : Finset α} (h : a ⊆ b) : a.card ≤ b.card := Finset.card_le_card h

/-- The difference and the meet make the whole again (the Rust test `a.minus(b) ∪ a.meet(b) == a`). -/
theorem sdiff_union_meet (a b : Finset α) : a \ b ∪ a ∩ b = a := by
  ext x; simp only [Finset.mem_union, Finset.mem_sdiff, Finset.mem_inter]; tauto

end Ideas.Layer0.Sets
