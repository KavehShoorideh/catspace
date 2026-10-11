import Ideas.Fork

/-!
# The refutation certificate (Kaveh 2026-10-11: "I like the refutation certificate")

A PROOF that a move wins is a tree where our moves are chosen and every reply of theirs is answered
(`Ideas.localCheck`, `forcedAtLeaves`). A REFUTATION is the dual: their replies are chosen -- one at each of their
nodes -- and every move of ours is shown to fall short. Checking a refutation costs our branching only: their
alternatives need no search, because fixing their replies can only make the claim EASIER for us, so a claim that fails
even then fails outright. That is the theorem here, `refuted_sound`: `refutedAtLeaves σ q n = true → forcedAtLeaves c
goal q n = false`, by induction on the plies.

In the planner (rust/analyst_core plan.rs, the memory): a question that FAILED on an earlier board stored their
refuting reply at every node (the hints); asked again on a changed board, the certificate is re-checked with their
replies fixed before any search is run -- the "harbour" of a chase is just their first stored reply. The language is
the search's: our moves at our nodes are its threat moves (the frame proof's filter), the leaf its leaf.
-/

namespace Ideas.Refutation

open Chess
open Ideas

/-- THE REFUTATION with their strategy `σ`: at the leaf the goal is not reached; at our nodes every legal move is refuted
one ply on; at theirs the strategy names a legal reply that refutes (a side with no legal move: we have not gained
unless they are mated -- the same conventions as `forcedAtLeaves`). -/
def refutedAtLeaves (c : Color) (goal : Position → Bool) (σ : Position → Option Move) : Position → Nat → Bool
  | q, 0 => !goal q
  | q, n + 1 =>
      if q.turn = c then
        (legalMoves q).all (fun m => refutedAtLeaves c goal σ (applyMove q m) n)
      else
        match σ q with
        | some r => (legalMoves q).contains r && refutedAtLeaves c goal σ (applyMove q r) n
        | none => !(legalMoves q).isEmpty && false

/-- A refutation under any strategy of theirs refutes the claim outright. -/
theorem refuted_sound (c : Color) (goal : Position → Bool) (σ : Position → Option Move) (q : Position) (n : Nat)
    (h : refutedAtLeaves c goal σ q n = true) : forcedAtLeaves c goal q n = false := by
  induction n generalizing q with
  | zero =>
      simp only [refutedAtLeaves, Bool.not_eq_true'] at h
      simp only [forcedAtLeaves]
      exact h
  | succ n ih =>
      by_cases hc : q.turn = c
      · simp only [refutedAtLeaves, hc, if_true, List.all_eq_true] at h
        simp only [forcedAtLeaves, hc, if_true]
        rw [Bool.eq_false_iff]
        intro hany
        rw [List.any_eq_true] at hany
        obtain ⟨m, hm, hf⟩ := hany
        have := ih (applyMove q m) (h m hm)
        rw [this] at hf
        exact absurd hf (by decide)
      · simp only [refutedAtLeaves, hc, if_false] at h
        simp only [forcedAtLeaves, hc, if_false]
        cases hσ : σ q with
        | none => simp [hσ] at h
        | some r =>
            simp only [hσ, Bool.and_eq_true, List.contains_iff_mem] at h
            obtain ⟨hmem, hr⟩ := h
            have hne : ¬ (legalMoves q).isEmpty = true := by
              intro he
              rw [List.isEmpty_iff] at he
              rw [he] at hmem
              exact absurd hmem (List.not_mem_nil)
            have hne' : (legalMoves q).isEmpty = false := Bool.eq_false_iff.mpr hne
            rw [if_neg (by rw [hne']; decide)]
            rw [Bool.eq_false_iff]
            intro hall
            rw [List.all_eq_true] at hall
            have := ih (applyMove q r) hr
            have h2 := hall r hmem
            rw [this] at h2
            exact absurd h2 (by decide)

end Ideas.Refutation
