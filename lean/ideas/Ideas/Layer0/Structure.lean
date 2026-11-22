import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Data.Nat.Find
import Mathlib.Order.Bounds.Basic

/-!
# Layer 0, structure (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 4)

The three forms of `cs_base::structure` and their one interface of distance:
* a GRAPH: distance in steps along links (a link goes one way; the relation `r` is the set of links); the triangle law, and nought only from a node to itself;
* the PLANE: the straight-line distance; the triangle law, and nought only from a point to itself;
* a SEQUENCE: the differences of neighbours add up to the last item less the first;
* REGIONS: a region is a set of points; the union and the meet of two are the union and the meet of the sets.
-/

namespace Ideas.Layer0.Structure

open scoped BigOperators

/-! ## The graph: distance in steps -/

section Graph

variable {N : Type*} (r : N → N → Prop)

/-- `StepsTo r a b n`: from `a` to `b` in `n` steps along the links `r`. -/
inductive StepsTo : N → N → ℕ → Prop
  | refl (a : N) : StepsTo a a 0
  | step {a b c : N} {n : ℕ} : r a b → StepsTo b c n → StepsTo a c (n + 1)

/-- STEPS ADD ALONG A PATH: `n` steps from `a` to `b` and `m` from `b` to `c` make `n + m` from `a` to `c`. -/
theorem stepsTo_trans {a b c : N} {n m : ℕ} (h₁ : StepsTo r a b n) (h₂ : StepsTo r b c m) : StepsTo r a c (n + m) := by
  induction h₁ with
  | refl a => simpa using h₂
  | @step a b' c' n' hab _ ih =>
    have := StepsTo.step hab (ih h₂)
    rwa [show n' + 1 + m = n' + m + 1 by omega]

/-- NOUGHT STEPS ONLY FROM A NODE TO ITSELF (Rust: `distance_in_steps_obeys_the_triangle_law_and_is_nought_only_from_a_node_to_itself`). -/
theorem stepsTo_zero_iff {a b : N} : StepsTo r a b 0 ↔ a = b := by
  constructor
  · intro h; cases h; rfl
  · rintro rfl; exact StepsTo.refl a

/-- `IsDist r a b d`: `d` is the distance in steps, the least number of steps from `a` to `b`. -/
def IsDist (a b : N) (d : ℕ) : Prop := IsLeast {n | StepsTo r a b n} d

/-- THE DISTANCE IS NOUGHT EXACTLY FROM A NODE TO ITSELF (Rust: `steps_zero_iff`, in `distance_in_steps_obeys_the_triangle_law...`). -/
theorem steps_zero_iff {a b : N} : IsDist r a b 0 ↔ a = b := by
  constructor
  · intro h; exact (stepsTo_zero_iff r).mp h.1
  · rintro rfl; exact ⟨StepsTo.refl a, fun n _ => Nat.zero_le n⟩

/-- A DISTANCE BETWEEN TWO DIFFERENT NODES IS NOT NOUGHT. -/
theorem steps_pos_of_ne {a b : N} {d : ℕ} (h : IsDist r a b d) (hne : a ≠ b) : 0 < d := by
  rcases Nat.eq_zero_or_pos d with rfl | hd
  · exact absurd ((steps_zero_iff r).mp h) hne
  · exact hd

/-- THE TRIANGLE LAW: if `a` reaches `b` at distance `d₁` and `b` reaches `c` at distance `d₂`, then `a` reaches `c` at some distance
no more than `d₁ + d₂` (Rust: `steps_triangle`). -/
theorem steps_triangle {a b c : N} {d₁ d₂ : ℕ} (h₁ : IsDist r a b d₁) (h₂ : IsDist r b c d₂) :
    ∃ d₃, IsDist r a c d₃ ∧ d₃ ≤ d₁ + d₂ := by
  classical
  have hne : ∃ n, StepsTo r a c n := ⟨d₁ + d₂, stepsTo_trans r h₁.1 h₂.1⟩
  refine ⟨Nat.find hne, ⟨Nat.find_spec hne, fun n hn => Nat.find_min' hne hn⟩, Nat.find_min' hne (stepsTo_trans r h₁.1 h₂.1)⟩

end Graph

/-! ## The plane: the straight-line distance -/

/-- The straight-line distance in the plane (Rust: `Plane`'s `distance`). -/
noncomputable def planeDist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

/-- A point of the plane as a point of the Euclidean space of two numbers. -/
noncomputable def toE (p : ℝ × ℝ) : EuclideanSpace ℝ (Fin 2) := !₂[p.1, p.2]

/-- The plane distance is the distance of Mathlib's Euclidean space. -/
theorem planeDist_eq_dist (p q : ℝ × ℝ) : planeDist p q = dist (toE p) (toE q) := by
  rw [EuclideanSpace.dist_eq, Fin.sum_univ_two]
  simp [planeDist, toE, Real.dist_eq, sq_abs]

/-- THE TRIANGLE LAW IN THE PLANE (Rust: `plane_triangle`). -/
theorem plane_triangle (p q s : ℝ × ℝ) : planeDist p s ≤ planeDist p q + planeDist q s := by
  rw [planeDist_eq_dist, planeDist_eq_dist, planeDist_eq_dist]
  exact dist_triangle _ _ _

/-- THE DISTANCE IN THE PLANE IS NOUGHT EXACTLY FROM A POINT TO ITSELF (Rust: `plane_dist_eq_zero`). -/
theorem plane_dist_eq_zero (p q : ℝ × ℝ) : planeDist p q = 0 ↔ p = q := by
  rw [planeDist_eq_dist, dist_eq_zero]
  constructor
  · intro h
    have h0 := congrFun (congrArg (fun v : EuclideanSpace ℝ (Fin 2) => (v : Fin 2 → ℝ)) h) 0
    have h1 := congrFun (congrArg (fun v : EuclideanSpace ℝ (Fin 2) => (v : Fin 2 → ℝ)) h) 1
    simp [toE] at h0 h1
    exact Prod.ext h0 h1
  · rintro rfl; rfl

/-- THE PLANE DISTANCE IS SYMMETRIC. -/
theorem plane_dist_comm (p q : ℝ × ℝ) : planeDist p q = planeDist q p := by
  rw [planeDist_eq_dist, planeDist_eq_dist]; exact dist_comm _ _

/-! ## A sequence: the differences of neighbours add up -/

/-- THE DIFFERENCES OF A SEQUENCE OF NUMBERS ADD UP TO THE LAST LESS THE FIRST (Rust: `the_differences_add_up_to_the_last_less_the_first`). -/
theorem differences_sum (f : ℕ → ℤ) (n : ℕ) : ∑ i ∈ Finset.range n, (f (i + 1) - f i) = f n - f 0 :=
  Finset.sum_range_sub f n

/-! ## Regions: sets of points -/

section Region

variable {P : Type*}

/-- THE UNION OF TWO REGIONS: a point is inside it exactly when it is inside one of the two (Rust: `the_union_and_the_meet_of_two_regions_are_or_and_and_of_inside`). -/
theorem inside_union (a b : Set P) (x : P) : x ∈ a ∪ b ↔ x ∈ a ∨ x ∈ b := Set.mem_union x a b

/-- THE MEET OF TWO REGIONS: inside it exactly when inside both. -/
theorem inside_meet (a b : Set P) (x : P) : x ∈ a ∩ b ↔ x ∈ a ∧ x ∈ b := Set.mem_inter_iff x a b

end Region

end Ideas.Layer0.Structure
