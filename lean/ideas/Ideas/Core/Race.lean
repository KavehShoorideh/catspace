import Ideas.Core.Prelude
import Ideas.Core.ForcedByMeasure

/-!
# The game-free half of the race file (PLAN 78.56 stage B; moved out of `Ideas/Race.lean`, proofs unchanged)

This file imports no chess file. The declarations keep the namespace `Ideas.Race` they had, so every old name (`Ideas.Race.race_sound`, `Ideas.Race.frame_sound_pass`,
`Ideas.Race.far_stays_off`, ...) still resolves, now through `import Ideas.Race`, which imports this file. Three groups:

* the race arithmetic: `raceBinds`, `race_arith`, `race_sound`, `race_first` (an abstract game with a tempo measure of ours and of theirs);
* the frame lemma: `frame_sound`, `withPass`, `frame_sound_pass` (a claim in a projected game that holds against the pass is the whole-game claim);
* the walks of a unit over PLACES, an arbitrary type (they were stated over the chess square type, and nothing in them looked at a square): `Steps`, `far_stays_off`,
  `far_not_bearing`, `UWalk`, `real_is_uwalk`, `deny_sound`, the status of a lifted kill, and the overload of two goals over a complete list of places.
-/

namespace Ideas.Race

open Ideas.Forced

/-! ## 62.1 The tempo word -/

/-- The tempi a side needs to promote: the pushes of its passer plus the moves that the stopper (or a block to clear) costs it. -/
def promoTempi (pushes delay : Nat) : Nat := pushes + delay

/-- THE RACE WORD: with the move, our promotion takes no more of our moves than theirs takes of theirs -- they promote on their `theirs`-th move, after our `theirs`-th. -/
def raceBinds (ours theirs : Nat) : Bool := decide (ours ≤ theirs)

/-- The stricter reading (the plan's `<`): it implies the word. -/
theorem raceBinds_of_lt {a b : Nat} (h : a < b) : raceBinds a b = true := by
  simp only [raceBinds, decide_eq_true_eq]; omega

/-- A claim read when the word does NOT bind is not a claim: `b + 1 ≤ a` (theirs strictly faster) never binds. -/
theorem raceBinds_false {a b : Nat} (h : b < a) : raceBinds a b = false := by
  simp only [raceBinds, decide_eq_false_iff_not]; omega

/-- Counting fact behind the word: after `j` rounds (our move, then theirs) with `j + 1 ≤ ours ≤ theirs`, they have still made fewer than `theirs` moves when we make our
`(j+1)`-th: the first to finish is us. -/
theorem race_arith {a b j : Nat} (hab : a ≤ b) (hj : j + 1 ≤ a) : j < b := by omega

/-- THE RACE IS SOUND (the arithmetic): an abstract game whose positions carry our remaining tempi `μ` and theirs `ν`.
* `hprog`  our move lowers `μ` by one while there is one to make;
* `hgoal`  `μ = 0` is the goal (the promotion);
* `hheld`  THE STOPPER ANALYSIS: a reply of theirs never raises `μ` (nothing they can stand on or capture delays us) and never hands us the goal;
* `hlive`  they always have a reply (no stalemate);
* `hfirst` a reply lowers `ν` by at most one and our move never lowers it (we make them no faster).
From a position with the move, `0 < μ ≤ ν`, we force the goal within `μ` of our moves, against every reply. -/
theorem race_sound {α : Type} {ours theirs : α → α → Prop} {goal : α → Prop} (μ ν : α → Nat)
    (hprog : ∀ p, 0 < μ p → ∃ q, ours p q ∧ μ q + 1 = μ p)
    (hgoal : ∀ q, μ q = 0 → goal q)
    (hheld : ∀ q r, theirs q r → 0 < μ q → 0 < μ r ∧ μ r ≤ μ q)
    (hlive : ∀ q, 0 < μ q → ∃ r, theirs q r)
    (hfirst : ∀ q r, theirs q r → ν q ≤ ν r + 1)
    (hours : ∀ p q, ours p q → ν p ≤ ν q) :
    ∀ p, 0 < μ p → μ p ≤ ν p → Reach ours theirs goal (fun _ => False) (μ p) p := by
  suffices H : ∀ n p, 0 < μ p → μ p ≤ n → μ p ≤ ν p → Reach ours theirs goal (fun _ => False) (μ p) p from
    fun p h1 h2 => H (μ p) p h1 (Nat.le_refl _) h2
  intro n
  induction n with
  | zero => intro p h1 h2; omega
  | succ n ih =>
    intro p hpos hn hν
    obtain ⟨q, hq, hμ⟩ := hprog p hpos
    obtain ⟨m, hm⟩ : ∃ m, μ p = m + 1 := ⟨μ p - 1, by omega⟩
    rw [hm]
    refine Or.inr ⟨q, hq, ?_⟩
    by_cases h0 : μ q = 0
    · exact Or.inl (hgoal q h0)
    · refine Or.inr ⟨hlive q (by omega), fun r hr => ?_⟩
      obtain ⟨hrpos, hrle⟩ := hheld q r hr (by omega)
      have h1 := hfirst q r hr
      have h2 := hours p q hq
      have key : μ r ≤ ν r := by omega
      exact Reach.mono (ih r hrpos (by omega) key) (by omega)

/-- THEY ARE NOT FIRST: on the line of `race_sound` no reply of theirs has completed their race -- `ν` stays positive after every reply, so a goal of theirs
(`ν = 0`) is never on the board when we make our next move. -/
theorem race_first {α : Type} {ours theirs : α → α → Prop} (μ ν : α → Nat)
    (hfirst : ∀ q r, theirs q r → ν q ≤ ν r + 1)
    (hours : ∀ p q, ours p q → ν p ≤ ν q)
    {p q r : α} (hq : ours p q) (hr : theirs q r) (hμ : 1 < μ p) (hle : μ p ≤ ν p) : 0 < ν r := by
  have := hfirst q r hr
  have := hours p q hq
  omega

/-! ## 62.2 Tempo-immunity: the frame's soundness condition -/

/-- THE FRAME LEMMA. A frame claim (`Reach` in the FRAME game: the pieces the word reads) that is forced against every reply of the frame and against a PASS is the whole-board
claim, when (`hours`) each of our frame moves is a whole-board move, (`hgoal`) the goal is the same, (`htheirs`) every whole-board reply projects onto a frame reply or leaves the
projection as it was (an excluded piece's move: a pass), and (`hlive`) the whole board never stalemates them. A forced win is immune to spare tempo moves; a zugzwang win is not
(`Ideas.Forcing.free_move_breaks`), which is why the frame claim must hold against the pass. -/
theorem frame_sound {α β : Type} {oursW theirsW : α → α → Prop} {goalW : α → Prop}
    {oursF theirsF : β → β → Prop} {goalF : β → Prop} (proj : α → β)
    (hours : ∀ p q', oursF (proj p) q' → ∃ q, oursW p q ∧ proj q = q')
    (hgoal : ∀ q, goalF (proj q) → goalW q)
    (htheirs : ∀ q r, theirsW q r → theirsF (proj q) (proj r))
    (hlive : ∀ q, (∃ r', theirsF (proj q) r') → ∃ r, theirsW q r) :
    ∀ n p, Reach oursF theirsF goalF (fun _ => False) n (proj p) → Reach oursW theirsW goalW (fun _ => False) n p := by
  intro n
  induction n with
  | zero => intro p h; exact h
  | succ n ih =>
    intro p h
    rcases h with h | ⟨q', hq', hg⟩
    · exact absurd h id
    · obtain ⟨q, hq, rfl⟩ := hours p q' hq'
      refine Or.inr ⟨q, hq, ?_⟩
      rcases hg with hg | ⟨hex, hall⟩
      · exact Or.inl (hgoal q hg)
      · exact Or.inr ⟨hlive q hex, fun r hr => ih r (hall (proj r) (htheirs q r hr))⟩

/-- Their replies in the frame game WITH THE PASS: a frame reply, or standing still. -/
def withPass {β : Type} (theirsF : β → β → Prop) (q r : β) : Prop := theirsF q r ∨ r = q

/-- THE FRAME LEMMA, the form the mirror checks: the frame claim is forced against the frame replies AND the pass; each whole-board reply is a frame reply or leaves the projection
unchanged (an excluded piece's move); the whole board has a reply. -/
theorem frame_sound_pass {α β : Type} {oursW theirsW : α → α → Prop} {goalW : α → Prop}
    {oursF theirsF : β → β → Prop} {goalF : β → Prop} (proj : α → β)
    (hours : ∀ p q', oursF (proj p) q' → ∃ q, oursW p q ∧ proj q = q')
    (hgoal : ∀ q, goalF (proj q) → goalW q)
    (htheirs : ∀ q r, theirsW q r → theirsF (proj q) (proj r) ∨ proj r = proj q)
    (hlive : ∀ q, ∃ r, theirsW q r) :
    ∀ n p, Reach oursF (withPass theirsF) goalF (fun _ => False) n (proj p) → Reach oursW theirsW goalW (fun _ => False) n p :=
  frame_sound proj hours hgoal (fun q r h => by
    rcases htheirs q r h with h' | h'
    · exact Or.inl h'
    · exact Or.inr h') (fun q _ => hlive q)

/-! ## The walks of a unit over places (stated over an arbitrary type of places) -/

section Places

variable {Place : Type}

/-- Iterated steps of a unit over places. -/
def Steps (step : Place → Place → Prop) : Nat → Place → Place → Prop
  | 0, s, s' => s = s'
  | j + 1, s, s' => ∃ u, step s u ∧ Steps step j u s'

/-- (a) OF THE CONDITION, for a unit with a step-1-Lipschitz lower bound `d` on the moves it needs (kings: `kdist`; pawns: `Passer2.pdist`; the rest: `lb`): a unit more than `k` moves
from every place that bears on the frame is, after `j ≤ k` moves, still more than `k - j` from each of them -- and stands on none. -/
theorem far_stays_off {step : Place → Place → Prop} (d : Place → Place → Nat)
    (hstep : ∀ s s' t, step s s' → d s t ≤ d s' t + 1) (B : Place → Prop) :
    ∀ j k s s', (∀ t, B t → k < d s t) → Steps step j s s' → j ≤ k → ∀ t, B t → k - j < d s' t := by
  intro j
  induction j with
  | zero =>
    intro k s s' hfar hst _ t ht
    simp only [Steps] at hst
    subst hst
    simpa using hfar t ht
  | succ j ih =>
    intro k s s' hfar hst hjk t ht
    obtain ⟨u, hsu, hus⟩ := hst
    have hfar' : ∀ t, B t → k - 1 < d u t := by
      intro t ht
      have := hfar t ht
      have := hstep s u t hsu
      omega
    have := ih (k - 1) u s' hfar' hus (by omega) t ht
    omega

/-- ... so the unit stands on no place that bears on the frame during the word's `k` tempi. -/
theorem far_not_bearing {step : Place → Place → Prop} (d : Place → Place → Nat) (hself : ∀ t, d t t = 0)
    (hstep : ∀ s s' t, step s s' → d s t ≤ d s' t + 1) (B : Place → Prop) {k j : Nat} {s s' : Place}
    (hfar : ∀ t, B t → k < d s t) (hst : Steps step j s s') (hj : j ≤ k) : ¬ B s' := by
  intro hB
  have := far_stays_off d hstep B j k s s' hfar hst hj s' hB
  rw [hself] at this
  omega

/-- A walk over usable places only: every landing place is one the unit can stand on and survive on. -/
def UWalk (step : Place → Place → Prop) (usable : Place → Prop) : Nat → Place → Place → Prop
  | 0, s, s' => s = s'
  | j + 1, s, s' => ∃ u, step s u ∧ usable u ∧ UWalk step usable j u s'

/-- The real walks of a unit (its legal moves) are walks over usable places, when every legal move is a step that lands on a usable place. -/
theorem real_is_uwalk {step real : Place → Place → Prop} {usable : Place → Prop}
    (hleg : ∀ a b, real a b → step a b ∧ usable b) :
    ∀ j s s', Steps real j s s' → UWalk step usable j s s' := by
  intro j
  induction j with
  | zero => intro s s' h; exact h
  | succ j ih =>
    intro s s' h
    obtain ⟨u, hu, hrest⟩ := h
    obtain ⟨hs, hb⟩ := hleg s u hu
    exact ⟨u, hs, hb, ih u s' hrest⟩

/-- DENY SOUND: if no walk over usable places of at most `k` moves reaches a target, the unit never reaches a target in `k` real moves -- its path is denied, it does not stop the pawn
within `k`. (A route of ours is a denial exactly when it makes the hypothesis true; the mirror, `race.deny_moves`, searches the walks.) -/
theorem deny_sound {step real : Place → Place → Prop} {usable : Place → Prop} (T : Place → Prop) {k : Nat} {s : Place}
    (hleg : ∀ a b, real a b → step a b ∧ usable b)
    (hdeny : ∀ j, j ≤ k → ∀ t, T t → ¬ UWalk step usable j s t) :
    ∀ j t, j ≤ k → T t → ¬ Steps real j s t :=
  fun j t hj ht hw => hdeny j hj t ht (real_is_uwalk hleg j s t hw)

/-- OVERLOADED over a complete list of places: no place of the list is in both hold sets. (`Ideas.Race.overloaded` is this over the chess squares.) -/
def overloadedOver (places : List Place) (h₁ h₂ : Place → Bool) : Bool :=
  places.all fun s => !(h₁ s && h₂ s)

/-- A unit on a hold place of the first goal, when the list of places is complete, is on no hold place of the second. -/
theorem overloadedOver_exclusive {places : List Place} (hcomplete : ∀ s, s ∈ places) {h₁ h₂ : Place → Bool}
    (h : overloadedOver places h₁ h₂ = true) {s : Place} (hs : h₁ s = true) : h₂ s = false := by
  simp only [overloadedOver, List.all_eq_true, Bool.not_eq_true'] at h
  have := h s (hcomplete s)
  simp only [hs, Bool.true_and] at this
  exact this

/-- THE DEFLECTION IN TIME, over places: if a unit must stay on a hold place of the first goal from step `i` on, it is from that step on no hold place of the second. -/
theorem overload_sound_over {places : List Place} (hcomplete : ∀ s, s ∈ places) {h₁ h₂ : Place → Bool}
    (h : overloadedOver places h₁ h₂ = true) (pos : Nat → Place) {i : Nat}
    (hstay : ∀ j, i ≤ j → h₁ (pos j) = true) : ∀ j, i ≤ j → h₂ (pos j) = false :=
  fun j hj => overloadedOver_exclusive hcomplete h (hstay j hj)

end Places

/-! ### 62.1c The status of a lifted kill (Bool-valued; nothing of a game in it) -/

/-- The status of a promote question: killed by the verdict, claimed by a closed chase, or a gap (open). -/
inductive Status | killed | claim | gap
  deriving DecidableEq

/-- The status from the verdict's kill and the chase's closing: a kill is final; otherwise only a CLOSED chase claims. -/
def status (killed closed : Bool) : Status :=
  if killed then .killed else if closed then .claim else .gap

/-- The kill of the verdict that judges only the king: it kills when the king stops. -/
def killKingOnly (kingStops : Bool) : Bool := kingStops

/-- The kill of the old verdict: any stopper stops. -/
def killAny (kingStops otherStops : Bool) : Bool := kingStops || otherStops

/-- THE LIFTED KILL IS SOUND TO LIFT: the king-only verdict kills a subset of what the all-stoppers verdict killed, so no question that was alive is killed by the lift. -/
theorem stopper_king_only {k o : Bool} (h : killKingOnly k = true) : killAny k o = true := by
  simp [killKingOnly, killAny] at *; exact Or.inl h

/-- A lifted kill is a GAP until the chase closes it, never a claim. -/
theorem lifted_is_gap {closed : Bool} (hopen : closed = false) : status false closed = .gap := by
  simp [status, hopen]

/-- Only a closed chase claims: the lift itself never makes a claim. -/
theorem claim_only_if_closed {killed closed : Bool} (h : status killed closed = .claim) : closed = true ∧ killed = false := by
  cases killed <;> cases closed <;> simp [status] at h ⊢

end Ideas.Race
