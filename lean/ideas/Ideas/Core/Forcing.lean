import Ideas.Core.Prelude
import Ideas.Core.ForcedByMeasure

/-!
# The judge, the route cost, the carry principle and the status line, game-free (PLAN 78.56 stage B; moved out of `Ideas/Forcing.lean`, proofs unchanged)

`routeCost`, `judgeReads`, `judge_mono`, the repetition claim's value, `judgeNext` and its lemmas, `none_affordable_of_cheapest`, `claim_mono` (a claim forced with a route set is
forced with a wider one), `refutation_not_mono`, `routeLow`, `keepsStatus`, `relevant`. Numbers and relations only; no chess file is imported. The words over legal moves
(`parries`, `delayedThreat`, `noSafeMove`, `stillForcing`) are bound to the chess `Position` and stay in `Ideas/Forcing.lean`. The namespace is the one they had.
-/

namespace Ideas.Forcing

/-! ## Stage C (ANALYST_PLAN 61 C1, C2, C4): the cost of a route, the judge, the repetition claim -/

/-- C1 THE COST of an open route: one for the route, plus the cost of the unread child of each parry (`regress.route_cost`). -/
def routeCost (childCosts : List Nat) : Nat := 1 + childCosts.sum

/-- A route costs at least one. -/
theorem routeCost_pos (cs : List Nat) : 0 < routeCost cs := by
  unfold routeCost; omega

/-- A route costs at least as much as any one child it reads. -/
theorem routeCost_ge_child {cs : List Nat} {c : Nat} (h : c ∈ cs) : c ≤ routeCost cs := by
  unfold routeCost
  have key : ∀ l : List Nat, c ∈ l → c ≤ l.sum := by
    intro l
    induction l with
    | nil => intro h; simp at h
    | cons a t ih =>
      intro h
      simp only [List.mem_cons] at h
      simp only [List.sum_cons]
      rcases h with rfl | h
      · omega
      · have := ih h; omega
  have := key cs h
  omega

/-- C2 THE JUDGE: read on while the cost times the seconds per node fits in the time left (integers: milliseconds per node). -/
def judgeReads (cost msPerNode leftMs : Nat) : Bool := decide (cost * msPerNode ≤ leftMs)

/-- The judge is monotone in the cost: a chain that is cheaper than one that is read on is read on (re-asking it at each node
with the REMAINING cost never cuts a chain the first ask would have read). -/
theorem judge_mono {c c' m l : Nat} (hc : c' ≤ c) (h : judgeReads c m l = true) : judgeReads c' m l = true := by
  simp only [judgeReads, decide_eq_true_eq] at *
  exact Nat.le_trans (Nat.mul_le_mul_right m hc) h

/-- C4 THE REPETITION CLAIM: a repeated position where we are behind (balance `bal` < 0) is worth minus the balance. -/
def repetitionValue (bal : Int) : Int := -bal

/-- It brings the result exactly to level. -/
theorem repetition_level (bal : Int) : bal + repetitionValue bal = 0 := by
  unfold repetitionValue; omega

/-- Behind, the claim gains at least one: it clears the chase's bar (value + dbal >= 1). -/
theorem repetition_gains {bal : Int} (h : bal < 0) : 1 ≤ repetitionValue bal := by
  unfold repetitionValue; omega


/-! ## Plan 65: time is a priced choice across the tiers -/

/-- 65.1 THE NEXT TIER'S ROUTES ARE PRICED COMPETITORS: a route of tier k is opened only while it is priced no higher than the cheapest route tier k+1 would add
(`none`: tier k+1 adds no route, nothing competes, the route is read). A route cut here is a GAP (read in tier k+1 in cost order), never a kill. -/
def judgeNext (competitor : Option Nat) (cost : Nat) : Bool :=
  match competitor with
  | none => true
  | some c => decide (cost ≤ c)

/-- Nothing competes: the route is read. -/
theorem judgeNext_none (cost : Nat) : judgeNext none cost = true := by
  simp [judgeNext]

/-- A route priced above a cheaper competitor is never read in this tier. -/
theorem judgeNext_cut {c k : Nat} (h : c < k) : judgeNext (some c) k = false := by
  simp only [judgeNext, decide_eq_false_iff_not]; omega

/-- A route priced at or below the competitor is read (a tie is read: the order asked must not decide). -/
theorem judgeNext_read {c k : Nat} (h : k ≤ c) : judgeNext (some c) k = true := by
  simp only [judgeNext, decide_eq_true_eq]; exact h

/-- Monotone in the cost: a cheaper route than one that is read is read. -/
theorem judgeNext_mono {o : Option Nat} {k k' : Nat} (hk : k' ≤ k) (h : judgeNext o k = true) : judgeNext o k' = true := by
  cases o with
  | none => exact judgeNext_none k'
  | some c =>
    simp only [judgeNext, decide_eq_true_eq] at *
    omega

/-- Reading in tier k never reads a route priced above an unread cheaper one of tier k+1: if the competitor is `c` and a route of cost `k` is read, `k ≤ c`. -/
theorem judgeNext_read_le {c k : Nat} (h : judgeNext (some c) k = true) : k ≤ c := by
  simp only [judgeNext, decide_eq_true_eq] at h; exact h

/-- 65.2 STOP WHEN NOTHING AFFORDABLE IS LEFT: if the CHEAPEST open route (every open cost is at least `c`) does not fit in the time left, no open route does. -/
theorem none_affordable_of_cheapest {ps : List Nat} {c m l : Nat} (hmin : ∀ x ∈ ps, c ≤ x)
    (hno : judgeReads c m l = false) : ∀ x ∈ ps, judgeReads x m l = false := by
  intro x hx
  by_contra hne
  have hx' : judgeReads x m l = true := by
    simpa using hne
  have := judge_mono (hmin x hx) hx'
  rw [hno] at this
  exact absurd this (by decide)


/-! ## Plan 66: a proof carries across the tiers, a refutation does not; a chain stop is resumed only while the line is still forcing -/

section Carry

open Ideas.Forced

variable {α : Type}

/-- 66.1 THE CARRY PRINCIPLE. A claim forced with our route set `ours` (`Reach`: within `n` of our moves, against EVERY reply of theirs) is forced
with any wider route set `ours'` ⊇ `ours`: the proof names moves of ours that are still there, and their replies are the same (a wider tier
widens our vocabulary only). So a claim proved in tier k is a claim of tier k+1. -/
theorem claim_mono {ours ours' theirs : α → α → Prop} {goal target : α → Prop}
    (hsub : ∀ p q, ours p q → ours' p q) :
    ∀ {n : Nat} {p : α}, Reach ours theirs goal target n p → Reach ours' theirs goal target n p
  | 0, _, h => h
  | _ + 1, p, h => by
    rcases h with h | ⟨q, hq, hg⟩
    · exact Or.inl h
    · refine Or.inr ⟨q, hsub p q hq, ?_⟩
      rcases hg with hg | ⟨hex, hall⟩
      · exact Or.inl hg
      · exact Or.inr ⟨hex, fun r hr => claim_mono hsub (hall r hr)⟩

/-- ... and a REFUTATION DOES NOT CARRY: there is a game where no claim is forced with the narrower route set and one is forced with a wider one
(a kill under tier k is not a kill under tier k+1, so tier k+1 re-reads every kill and every GAP). -/
theorem refutation_not_mono : ∃ (ours ours' theirs : Bool → Bool → Prop) (goal : Bool → Prop),
    (∀ p q, ours p q → ours' p q) ∧ ¬ Reach ours theirs goal (fun _ => False) 1 false ∧ Reach ours' theirs goal (fun _ => False) 1 false := by
  refine ⟨fun _ _ => False, fun _ q => q = true, fun _ _ => False, fun q => q = true, fun _ _ h => h.elim, ?_, ?_⟩
  · simp [Reach]
  · exact Or.inr ⟨true, rfl, Or.inl rfl⟩

/-- 66.2 A CARRIED REPLY IS A FLOOR. The route's value is the least gain over the replies (`foldr min`, from a top value). Carried gains that are
each at most the gain a full read would find give a value at most the full read's: the carried value is a proved lower bound, never more. -/
def routeLow (gs : List Int) (top : Int) : Int := gs.foldr min top

theorem routeLow_mono {top : Int} : ∀ {gs ts : List Int}, List.Forall₂ (· ≤ ·) gs ts → routeLow gs top ≤ routeLow ts top
  | [], [], _ => le_refl _
  | _ :: _, _ :: _, List.Forall₂.cons h hs => by
    simp only [routeLow, List.foldr_cons]
    exact min_le_min h (routeLow_mono hs)

end Carry

/-! ## 68 Relevance to the status (ANALYST_PLAN 68; Kaveh 2026-10-22, eSJqH)

The status (plan 59) sets a LINE a claim must reach: `keepsStatus balance value line` is `line ≤ balance + value`. The NEED is `line - balance`. A root
question whose stake is below the need cannot keep the status by its own claim (`irrelevant_keeps_nothing`), so the driver asks it only after every
tier's relevant questions (`relevant`), and a claim that keeps the status came from a relevant question (`relevant_of_keeps`). -/

def keepsStatus (balance value line : Int) : Bool := decide (line ≤ balance + value)

def relevant (stake need : Int) : Bool := decide (need ≤ stake)

theorem irrelevant_keeps_nothing {balance value line : Int} (h : value < line - balance) : keepsStatus balance value line = false := by
  simp only [keepsStatus, decide_eq_false_iff_not, not_le]
  omega

theorem relevant_of_keeps {balance value line : Int} (h : keepsStatus balance value line = true) : relevant value (line - balance) = true := by
  simp only [keepsStatus, decide_eq_true_eq] at h
  simp only [relevant, decide_eq_true_eq]
  omega

theorem relevant_of_kept_need {need : Int} (h : need ≤ 0) (stake : Int) (hs : 0 ≤ stake) : relevant stake need = true := by
  simp only [relevant, decide_eq_true_eq]
  omega

end Ideas.Forcing
