import Ideas.ThreatLemma
import Ideas.Landing
import Ideas.ForcedByMeasure

/-!
# Forced chains: parries, the delayed threat, no safe move (ANALYST_PLAN 61 stage A, Kaveh GO 2026-10-21)

Three words over the legal moves, each generic in the CLAIM it reads (`bind`: a Bool on a position, e.g. "a mate in one stands",
"the exchange wins at least the stake", "the passer lemma binds"), so the Python mirror (regress.parries / delayed_threat /
no_safe_move) instantiates the claim with the rules' reading and the lemmas below hold for every instance.

* `parries bind counter q r` -- the reply `r` of the side to move in `q` PARRIES the threat `bind` when, after it, the claim no
  longer binds, OR the reply is itself forcing against the other side (`counter q r`: a check, a capture worth the stake, a threat
  of theirs by the same word with them on move -- Allis's counter-threat hole).
* `parryList` / `closedList` -- the legal replies that parry / do not.
  `closed_binds`: a reply outside `parryList` leaves the threat bound -- by the definition. This is why the chase may close it
  without opening an inner agenda (given the rules' other conditions, which the chase reads as before).
* `delayedThreat forcing q` -- after EVERY legal reply some legal move of ours is forcing (one bounded AND level: their replies x
  our forcing candidates, no recursion). `delayed_forcing`: for every legal reply there is a forcing move.
* `noSafeMove loses q` -- every legal reply loses (`loses q r`, a Bool: the exchange on the piece that moved or on the duty it left
  reaches the stake) and there is a reply at all (no stalemate). `noSafeMove_loses`: each legal reply loses.
-/

namespace Ideas.Forcing

open Chess

/-- The reply `r` parries: the claim no longer binds after it, or the reply is itself forcing against the mover. -/
def parries (bind : Position → Bool) (counter : Position → Move → Bool) (q : Position) (r : Move) : Bool :=
  !(bind (applyMove q r)) || counter q r

/-- The legal replies that parry. -/
def parryList (bind : Position → Bool) (counter : Position → Move → Bool) (q : Position) : List Move :=
  (legalMoves q).filter (parries bind counter q)

/-- The legal replies that do not. -/
def closedList (bind : Position → Bool) (counter : Position → Move → Bool) (q : Position) : List Move :=
  (legalMoves q).filter (fun r => !(parries bind counter q r))

/-- A reply that does not parry leaves the threat bound, and is no counter-threat. -/
theorem not_parries_binds {bind : Position → Bool} {counter : Position → Move → Bool} {q : Position} {r : Move}
    (h : parries bind counter q r = false) : bind (applyMove q r) = true ∧ counter q r = false := by
  simp only [parries, Bool.or_eq_false_iff, Bool.not_eq_false'] at h
  simpa using h

/-- THE LEMMA OF A1/A2: a legal reply outside `parryList` leaves the threat bound. -/
theorem closed_binds {bind : Position → Bool} {counter : Position → Move → Bool} {q : Position} {r : Move}
    (h : r ∈ closedList bind counter q) : r ∈ legalMoves q ∧ bind (applyMove q r) = true := by
  simp only [closedList, List.mem_filter, Bool.not_eq_true'] at h
  exact ⟨h.1, (not_parries_binds h.2).1⟩

/-- Every legal reply is a parry or closed: the two lists split the replies. -/
theorem parry_or_closed {bind : Position → Bool} {counter : Position → Move → Bool} {q : Position} {r : Move}
    (h : r ∈ legalMoves q) : r ∈ parryList bind counter q ∨ r ∈ closedList bind counter q := by
  by_cases hp : parries bind counter q r = true
  · left; simp [parryList, h, hp]
  · right; simp only [closedList, List.mem_filter, Bool.not_eq_true'] at *; simp [h, hp]

/-- THE DELAYED THREAT (lambda-2, Thomsen): after every legal reply of theirs, some legal move of ours is forcing. -/
def delayedThreat (forcing : Position → Move → Bool) (q : Position) : Bool :=
  (legalMoves q).all (fun r => (legalMoves (applyMove q r)).any (fun x => forcing (applyMove q r) x))

/-- A delayed threat has a forcing move after each legal reply. -/
theorem delayed_forcing {forcing : Position → Move → Bool} {q : Position} (h : delayedThreat forcing q = true) {r : Move}
    (hr : r ∈ legalMoves q) : ∃ x, x ∈ legalMoves (applyMove q r) ∧ forcing (applyMove q r) x = true := by
  simp only [delayedThreat, List.all_eq_true, List.any_eq_true] at h
  obtain ⟨x, hx, hf⟩ := h r hr
  exact ⟨x, hx, hf⟩

/-- NO SAFE MOVE (zugzwang as a word): the side to move has a legal reply, and every legal reply loses. -/
def noSafeMove (loses : Position → Move → Bool) (q : Position) : Bool :=
  !(legalMoves q).isEmpty && (legalMoves q).all (loses q)

/-- Each legal reply loses; and there is one (the position is not stalemate or mate). -/
theorem noSafeMove_loses {loses : Position → Move → Bool} {q : Position} (h : noSafeMove loses q = true) :
    (legalMoves q) ≠ [] ∧ ∀ r, r ∈ legalMoves q → loses q r = true := by
  simp only [noSafeMove, Bool.and_eq_true, Bool.not_eq_true', List.all_eq_true] at h
  refine ⟨?_, h.2⟩
  intro e; rw [e] at h; simp at h

/-- A free tempo move (a reply that loses nothing) is exactly what breaks it. -/
theorem free_move_breaks {loses : Position → Move → Bool} {q : Position} {r : Move} (hr : r ∈ legalMoves q)
    (hf : loses q r = false) : noSafeMove loses q = false := by
  by_contra hc
  have h : noSafeMove loses q = true := by simpa using hc
  have := (noSafeMove_loses h).2 r hr
  rw [hf] at this; exact absurd this (by simp)

/-! ### 63.1: a check that is also a threat is a follow-up

`checkThreat isCheck claim q x`: `x` is a check AND, on the board after it, the claim of ours binds (read with the move passed back to us, the
null-move device of `pass_claim`: the discovered check that uncovers a mate). `followUp` is the follow-up of a delayed threat: a threat / capture
worth the stake (`thr`), a threat of mate at any stake (`mateThr`), or a check that is also a threat. -/

/-- The check is also a threat: `x` gives check and the claim binds on the board after it. -/
def checkThreat (isCheck : Position → Move → Bool) (claim : Position → Bool) (q : Position) (x : Move) : Bool :=
  isCheck q x && claim (applyMove q x)

/-- THE FOLLOW-UP of a delayed threat (plan 63.1). -/
def followUp (thr mateThr isCheck : Position → Move → Bool) (claim : Position → Bool) (q : Position) (x : Move) : Bool :=
  thr q x || mateThr q x || checkThreat isCheck claim q x

/-- A check that is also a threat is a lambda-1 move: it is a check, and a pass (the claim read on the board after it) leaves a claim of ours. -/
theorem checkThreat_forcing {isCheck : Position → Move → Bool} {claim : Position → Bool} {q : Position} {x : Move}
    (h : checkThreat isCheck claim q x = true) : isCheck q x = true ∧ claim (applyMove q x) = true := by
  simpa [checkThreat] using h

/-- A bare check (no claim behind it) is not a check that is also a threat. -/
theorem bare_check_not_threat {isCheck : Position → Move → Bool} {claim : Position → Bool} {q : Position} {x : Move}
    (h : claim (applyMove q x) = false) : checkThreat isCheck claim q x = false := by
  simp [checkThreat, h]

/-- The follow-up includes the check that is also a threat. -/
theorem followUp_of_checkThreat {thr mateThr isCheck : Position → Move → Bool} {claim : Position → Bool} {q : Position} {x : Move}
    (h : checkThreat isCheck claim q x = true) : followUp thr mateThr isCheck claim q x = true := by
  simp [followUp, h]

/-- A delayed threat whose forcing word is `followUp` has, after every reply, a follow-up: a threat, a threat of mate, or a check with a claim behind it. -/
theorem delayed_followUp {thr mateThr isCheck : Position → Move → Bool} {claim : Position → Bool} {q : Position}
    (h : delayedThreat (followUp thr mateThr isCheck claim) q = true) {r : Move} (hr : r ∈ legalMoves q) :
    ∃ x, x ∈ legalMoves (applyMove q r) ∧ followUp thr mateThr isCheck claim (applyMove q r) x = true :=
  delayed_forcing h hr


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

/-- 66.4 ONE STEP OF A LINE: after our move (`q`, their turn) their reply `r`; `bind` is the claim our move threatened (a Bool on a position) and
`threatens` says our move named a threat at all (`regress.threat_of` is not None: a check, a capture, a threat of mate, a threat worth the stake). -/
structure Step where
  q : Position
  r : Move
  bind : Position → Bool
  threatens : Bool

/-- 66.4 STILL FORCING: at every step our move threatened, and their reply PARRIED it (`parries`: the threat no longer binds, or the reply is itself a
counter-threat). A chain-stopped line is resumed by one more level of the allowance only while this holds; a line that went quiet is not. -/
def stillForcing (counter : Position → Move → Bool) (line : List Step) : Bool :=
  line.all (fun s => s.threatens && parries s.bind counter s.q s.r)

/-- The empty line (the stop at the root's first reply) is still forcing. -/
theorem stillForcing_nil (counter : Position → Move → Bool) : stillForcing counter [] = true := rfl

/-- Extending a line by one step: still forcing iff the line was and the new step threatens and is parried. -/
theorem stillForcing_append (counter : Position → Move → Bool) (line : List Step) (s : Step) :
    stillForcing counter (line ++ [s]) = (stillForcing counter line && (s.threatens && parries s.bind counter s.q s.r)) := by
  simp [stillForcing, List.all_append]

/-- A QUIET STEP ENDS IT: a step where our move named no threat makes the line not forcing, whatever came before or after. -/
theorem quiet_breaks {counter : Position → Move → Bool} {line : List Step} {s : Step} (hs : s ∈ line) (hq : s.threatens = false) :
    stillForcing counter line = false := by
  by_contra h
  have h' : stillForcing counter line = true := by simpa using h
  simp only [stillForcing, List.all_eq_true] at h'
  have := h' s hs
  simp [hq] at this

/-- In a still-forcing line every legal reply read is in the parry list of its step (`parryList`): none of them is a reply the chase would have
closed by the threat (`closed_binds`). -/
theorem stillForcing_parryList {counter : Position → Move → Bool} {line : List Step} (h : stillForcing counter line = true) {s : Step}
    (hs : s ∈ line) (hl : s.r ∈ legalMoves s.q) : s.r ∈ parryList s.bind counter s.q := by
  simp only [stillForcing, List.all_eq_true] at h
  have := h s hs
  simp only [Bool.and_eq_true] at this
  simp [parryList, hl, this.2]

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
