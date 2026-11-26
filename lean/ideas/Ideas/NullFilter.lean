import Ideas.Support

/-!
# The null change (plan 79.59 part B1; SYSTEM_SPEC 11, "The lemma miner of layer 2"; ledger rows 59, 57)

GENERAL machinery, no game. The first lemma of the miner is the support lemma of `Ideas.Support` with the reference world the state itself:
a test is read once on the state as it stands (the null change, the hypothesis read as the state), the inputs it asked are its support, and
every key whose world differs from the state only outside that support has the answer of the test on the state.
`b` is the state, `w1` the world of one key, `D1` the places where the key's world may differ from the state.
-/
namespace Ideas.NullFilter

open Ideas.Support

variable {I V A : Type}

/-- THE NULL CHANGE: a world that agrees with the state outside `D1` gives the same result as the state when none of the inputs
the test asked on the state is a place of `D1`. -/
theorem Test.run_of_null (t : Test I V A) (b w1 : I → V) (D1 : I → Prop)
    (h1 : ∀ i, ¬ D1 i → w1 i = b i)
    (d1 : ∀ i ∈ t.asked b, ¬ D1 i) :
    t.run w1 = t.run b :=
  Test.run_of_disjoint t b b w1 (fun _ => False) D1 (fun _ _ => rfl) h1 (fun _ _ h => h) d1

#print axioms Test.run_of_null

end Ideas.NullFilter
