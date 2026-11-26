/-!
# The support lemma (plan 79.56 part A; SYSTEM_SPEC 11, "The support filter"; ledger rows 56, 57)

GENERAL machinery, no game: a computation that asks inputs one at a time and branches on the answers gives the same
result on two worlds that agree on every input it asked in the first. A decision tree whose nodes ask an input and
whose leaves give an answer is that computation; `asked` is the inputs on the path the first world takes.
-/
namespace Ideas.Support

/-- A test: it asks an input, and what it does next depends on the answer; it ends with a result. -/
inductive Test (I V A : Type) where
  | done : A → Test I V A
  | ask : I → (V → Test I V A) → Test I V A

variable {I V A : Type}

/-- The result of the test in a world (a world answers each input). -/
def Test.run : Test I V A → (I → V) → A
  | .done a, _ => a
  | .ask i k, w => (k (w i)).run w

/-- The inputs the test asks in a world, in order, along the path the answers make. -/
def Test.asked : Test I V A → (I → V) → List I
  | .done _, _ => []
  | .ask i k, w => i :: (k (w i)).asked w

/-- THE LEMMA: two worlds that agree on the inputs the test asked in the first give the same result, and the test
asks the same inputs in both. -/
theorem Test.run_congr (t : Test I V A) (w w' : I → V)
    (h : ∀ i ∈ t.asked w, w' i = w i) : t.run w' = t.run w ∧ t.asked w' = t.asked w := by
  induction t <;> simp_all [Test.run, Test.asked]

/-- The form the translator uses. `b` is the state; a key's world agrees with it except at the places it touches.
`w0` is the reference key's world, `w1` another key's; the reference's test asked `S0`, and no place of `S0` is touched
by either key. Then the other key's answer is the reference's. -/
theorem Test.run_of_disjoint (t : Test I V A) (b w0 w1 : I → V) (D0 D1 : I → Prop)
    (h0 : ∀ i, ¬ D0 i → w0 i = b i) (h1 : ∀ i, ¬ D1 i → w1 i = b i)
    (d0 : ∀ i ∈ t.asked w0, ¬ D0 i) (d1 : ∀ i ∈ t.asked w0, ¬ D1 i) :
    t.run w1 = t.run w0 := by
  apply (Test.run_congr t w0 w1 _).1
  intro i hi
  rw [h1 i (d1 i hi), h0 i (d0 i hi)]

#print axioms Test.run_congr
#print axioms Test.run_of_disjoint

end Ideas.Support
