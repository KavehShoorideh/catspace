import Ideas.Core.Game
import Ideas.Core.Plan

/-!
# THE BOARD STRUCTURE (PLAN 78.56 stage C): places, units, one static step relation, and the five properties of SYSTEM_SPEC 12.2

One structure over an ARBITRARY type of places and of units. It names what a game supplies so that the frame and path results of October (written over the chess squares)
become statements about any game; the proofs of those results used five properties and nothing else of the board. Each is stated ONCE, as a field of `Board.Laws`:

| property | field of `Laws` | says |
|---|---|---|
| touch | `touch` | a move changes the board only at the places it touches |
| line | `bears_occ`, `line` | a unit bears on a place only from where it stands; whether it bears depends on the unit on its place and on the places strictly between being empty -- emptying a place can only ADD bearings |
| footprint | `footprint_refl`, `bears_footprint`, `between_footprint` | a unit that bears on a place stands in that place's footprint, and so does every place strictly between them |
| goal unit | `goal_safe` | a legal move leaves the mover's goal unit on a place no opposing unit bears on |
| distance | `dist_self`, `dist_step` | a distance per kind, 0 on the place itself, that changes by at most one along each step of the kind's static step relation |

The data is `Board`: the moves and `apply` of a `Game` (the same `Ideas.Game.Game`), who is to move, what stands where (`occ`), the kind and side of a unit, the static step
relation of each kind (the places a unit of that kind can go to in one move on an EMPTY board), `bears` (a unit of side `c` on `s` bears on `t` in a state), `between`
(`u` strictly between `s` and `t` on a line), `touched` (the places a move changes), `footprint` (`footprint t x`: a unit on `x` can bear on `t`), `goalPlace` (where the goal
unit of a side stands: the king in chess) and `dist`.

`pathDist` and `pathDist_step` are the one new lemma: the SHORTEST PATH in a kind's step relation is a distance in the sense of the last row (it changes by at most one along a
step), so a game need not supply `dist` by hand when its step relation connects the places (`pathDist_lipschitz`). It replaces the separate chess lemmas `kdist_step`,
`rdist_step`, `bdist_step`, `lb_step` (those stay, as the instance's proofs of the field for the chess bounds).

Nothing here imports a chess file. The chess instance is in `Ideas/BoardInstance.lean` (it imports the chess kernel; this file does not).
-/

namespace Ideas.Core

open Ideas.Race (Steps)

/-! ## The shortest path in a step relation is a distance -/

section PathDist

variable {Place : Type}

/-- `n` is the length of a SHORTEST walk from `s` to `t` in `step`. -/
def IsShortest (step : Place → Place → Prop) (n : Nat) (s t : Place) : Prop :=
  Steps step n s t ∧ ∀ m, Steps step m s t → n ≤ m

/-- The length of a shortest walk from `s` to `t` (0 when `t` cannot be reached: use it with `pathDist_step`'s reachability hypothesis, or on a connected step relation). -/
noncomputable def pathDist (step : Place → Place → Prop) (s t : Place) : Nat :=
  open Classical in if h : ∃ n, Steps step n s t then Nat.find h else 0

theorem pathDist_isShortest {step : Place → Place → Prop} {s t : Place} (h : ∃ n, Steps step n s t) :
    IsShortest step (pathDist step s t) s t := by
  classical
  have e : pathDist step s t = Nat.find h := by simp [pathDist, h]
  rw [e]
  exact ⟨Nat.find_spec h, fun m hm => Nat.find_min' h hm⟩

theorem pathDist_self (step : Place → Place → Prop) (t : Place) : pathDist step t t = 0 := by
  have h : ∃ n, Steps step n t t := ⟨0, rfl⟩
  have := (pathDist_isShortest h).2 0 rfl
  omega

/-- THE LEMMA OF 12.2: the shortest path in a step relation changes by at most one per step. If `s` steps to `s'` and `t` can be reached from `s'`, the shortest walk from `s` to `t`
is at most one longer than the shortest from `s'`. -/
theorem pathDist_step {step : Place → Place → Prop} {s s' t : Place} (hst : step s s') (hr : ∃ n, Steps step n s' t) :
    pathDist step s t ≤ pathDist step s' t + 1 := by
  obtain ⟨hw, _⟩ := pathDist_isShortest hr
  have hs : Steps step (pathDist step s' t + 1) s t := ⟨s', hst, hw⟩
  exact (pathDist_isShortest ⟨_, hs⟩).2 _ hs

/-- On a step relation that connects every pair of places, the shortest path is a distance: the hypothesis `dist_step` of the board laws holds of it. -/
theorem pathDist_lipschitz {step : Place → Place → Prop} (hconn : ∀ s t, ∃ n, Steps step n s t) :
    ∀ s s' t, step s s' → pathDist step s t ≤ pathDist step s' t + 1 :=
  fun _ _ t hst => pathDist_step hst (hconn _ t)

/-- If steps run both ways (a symmetric relation), the shortest path changes by at most one in EITHER direction. -/
theorem pathDist_step_symm {step : Place → Place → Prop} (hsym : ∀ a b, step a b → step b a) (hconn : ∀ s t, ∃ n, Steps step n s t)
    {s s' : Place} (hst : step s s') (t : Place) :
    pathDist step s t ≤ pathDist step s' t + 1 ∧ pathDist step s' t ≤ pathDist step s t + 1 :=
  ⟨pathDist_step hst (hconn _ t), pathDist_step (hsym _ _ hst) (hconn _ t)⟩

end PathDist

/-! ## The board -/

/-- THE DATA a game supplies for the frame and path results: a `Game` over `State` and `Move` (`toGame`), places, units, sides, kinds. -/
structure Board (State Move Place Unit Side Kind : Type) extends Ideas.Game.Game State Move where
  /-- the side to move -/
  side : State → Side
  /-- the other side -/
  other : Side → Side
  /-- what stands on a place: a unit, or nothing -/
  occ : State → Place → Option Unit
  /-- the kind and the side of a unit -/
  kindOf : Unit → Kind
  sideOf : Unit → Side
  /-- the STATIC step relation of a kind: where a unit of that kind goes in one move on an empty board -/
  kstep : Kind → Place → Place → Prop
  /-- a unit of side `c` standing on `s` bears on `t` in the state -/
  bears : State → Side → Place → Place → Prop
  /-- `u` lies strictly between `s` and `t` on a line -/
  between : Place → Place → Place → Prop
  /-- the places a move changes -/
  touched : State → Move → Place → Prop
  /-- `footprint t x`: a unit standing on `x` can bear on `t` (the place, the lines through it, the leap sources) -/
  footprint : Place → Place → Prop
  /-- where the goal unit of a side stands (the king in chess), if it is on the board -/
  goalPlace : State → Side → Option Place
  /-- a distance per kind: a lower bound on the moves a unit of that kind needs -/
  dist : Kind → Place → Place → Nat

variable {State Move Place Unit Side Kind : Type}

/-- THE FIVE PROPERTIES, each once. A game satisfies the structure when these hold; whether they do is checked per game (in Lean for the games we have a kernel for, by the engine
on sampled positions otherwise). -/
structure Board.Laws (b : Board State Move Place Unit Side Kind) : Prop where
  /-- TOUCH: a move changes the board only at the places it touches. -/
  touch : ∀ p m u, ¬ b.touched p m u → b.occ (b.apply p m) u = b.occ p u
  /-- LINE (a): a unit bears on a place only from a place where one stands. -/
  bears_occ : ∀ p c s t, b.bears p c s t → b.occ p s ≠ none
  /-- LINE (b), for any two states: a bearing of `s` on `t` that holds in the second and not in the first needs the occupant of `s` changed, or a place strictly between them
  that went from occupied to empty. Emptying a place can only add bearings. -/
  line : ∀ p p' c s t, b.bears p' c s t → ¬ b.bears p c s t →
    b.occ p' s ≠ b.occ p s ∨ ∃ u, b.between s u t ∧ b.occ p u ≠ none ∧ b.occ p' u = none
  /-- FOOTPRINT (a): a place is in its own footprint. -/
  footprint_refl : ∀ t, b.footprint t t
  /-- FOOTPRINT (b): a unit that bears on `t` stands in `t`'s footprint. -/
  bears_footprint : ∀ p c s t, b.bears p c s t → b.footprint t s
  /-- FOOTPRINT (c): every place strictly between `s` and `t` is in `t`'s footprint. -/
  between_footprint : ∀ s u t, b.between s u t → b.footprint t u
  /-- GOAL UNIT: a legal move leaves the mover's goal unit on a place no unit of the other side bears on. -/
  goal_safe : ∀ p m, m ∈ b.moves p → ∀ t, b.goalPlace (b.apply p m) (b.side p) = some t →
    ∀ s, ¬ b.bears (b.apply p m) (b.other (b.side p)) s t
  /-- DISTANCE (a): a place is at distance 0 from itself. -/
  dist_self : ∀ k t, b.dist k t t = 0
  /-- DISTANCE (b): along a step of the kind's static step relation the distance to any place falls by at most one. -/
  dist_step : ∀ k s s' t, b.kstep k s s' → b.dist k s t ≤ b.dist k s' t + 1

/-! ## Stage D1: the walk results over the board's places -/

section D1

variable {b : Board State Move Place Unit Side Kind} (L : b.Laws)
include L

/-- `far_stays_off` for a kind of the board: a unit of kind `k` more than `K` moves from every place of `B` is, after `j ≤ K` steps, still more than `K - j` from each. -/
theorem Board.far_stays_off (kd : Kind) (B : Place → Prop) :
    ∀ j K s s', (∀ t, B t → K < b.dist kd s t) → Steps (b.kstep kd) j s s' → j ≤ K → ∀ t, B t → K - j < b.dist kd s' t :=
  Ideas.Race.far_stays_off (b.dist kd) (L.dist_step kd) B

/-- `far_not_bearing` for a kind of the board: it stands on no place of `B` during its `K` tempi. -/
theorem Board.far_not_bearing (kd : Kind) (B : Place → Prop) {K j : Nat} {s s' : Place}
    (hfar : ∀ t, B t → K < b.dist kd s t) (hst : Steps (b.kstep kd) j s s') (hj : j ≤ K) : ¬ B s' :=
  Ideas.Race.far_not_bearing (b.dist kd) (L.dist_self kd) (L.dist_step kd) B hfar hst hj

/-- `plan_sound` over the board: with `safe`, a unit of kind `kd` whose movers are measured by the board's distance stands on no weak place after `j ≤ deadline` of its steps. -/
theorem Board.plan_sound (kd : Kind) {W : List (Ideas.Plan.Weak Place)} {M : List (Ideas.Plan.Mover Place)}
    (h : Ideas.Plan.safe W M) {m : Ideas.Plan.Mover Place} (hm : m ∈ M) (hd : m.d = b.dist kd)
    {w : Ideas.Plan.Weak Place} (hw : w ∈ W) {j : Nat} {s' : Place} (hst : Steps (b.kstep kd) j m.pos s') (hj : j ≤ w.deadline) : s' ≠ w.sq :=
  Ideas.Plan.plan_sound h hm (by rw [hd]; exact L.dist_self kd) (by rw [hd]; exact L.dist_step kd) hw hst hj

/-- `deny_sound` over the board: a unit whose real moves are steps of its kind that land on usable places never reaches a target within `K` moves, when no walk over usable places does. -/
theorem Board.deny_sound (kd : Kind) {real : Place → Place → Prop} {usable : Place → Prop} (T : Place → Prop) {K : Nat} {s : Place}
    (hleg : ∀ a c, real a c → b.kstep kd a c ∧ usable c)
    (hdeny : ∀ j, j ≤ K → ∀ t, T t → ¬ Ideas.Race.UWalk (b.kstep kd) usable j s t) :
    ∀ j t, j ≤ K → T t → ¬ Steps real j s t :=
  Ideas.Race.deny_sound T hleg hdeny

/-- `overload_sound` over the board's places: two goals whose hold sets share no place of a complete list of places cannot both be held by one unit staying on its first. -/
theorem Board.overload_sound {places : List Place} (hcomplete : ∀ s, s ∈ places) {h₁ h₂ : Place → Bool}
    (h : Ideas.Race.overloadedOver places h₁ h₂ = true) (pos : Nat → Place) {i : Nat}
    (hstay : ∀ j, i ≤ j → h₁ (pos j) = true) : ∀ j, i ≤ j → h₂ (pos j) = false :=
  Ideas.Race.overload_sound_over hcomplete h pos hstay

/-- THE GOAL UNIT IS NEVER LEFT BORNE ON (the generic `no_stand_attacked`): after a legal move of the side to move whose goal unit stands on `t`, no unit of the other side bears on `t`. -/
theorem Board.no_stand_bears {p : State} {m : Move} (hm : m ∈ b.moves p) {t s : Place}
    (ht : b.goalPlace (b.apply p m) (b.side p) = some t) (hs : b.bears (b.apply p m) (b.other (b.side p)) s t) : False :=
  L.goal_safe p m hm t ht s hs

end D1

/-! ## Stage D2: the path lemma, the lost-attack path lemma, attack-unchanged and the frame lemma over the board -/

section D2

variable {b : Board State Move Place Unit Side Kind} (L : b.Laws)
include L

/-- THE PATH LEMMA over the board (touch and line): after a move, a NEW bearing of `s` on `t` comes from a unit that ARRIVED on a touched `s`, or through a touched place the
move VACATED strictly between `s` and `t` (a discovery). -/
theorem Board.path_lemma (p : State) (m : Move) (c : Side) (s t : Place)
    (h' : b.bears (b.apply p m) c s t) (h : ¬ b.bears p c s t) :
    (b.touched p m s ∧ b.occ (b.apply p m) s ≠ none) ∨
      ∃ u, b.between s u t ∧ b.touched p m u ∧ b.occ p u ≠ none ∧ b.occ (b.apply p m) u = none := by
  rcases L.line p (b.apply p m) c s t h' h with hs | ⟨u, hb, hne, h0⟩
  · left
    refine ⟨?_, fun h0 => L.bears_occ _ _ _ _ h' h0⟩
    by_contra hn
    exact hs (L.touch p m s hn)
  · right
    refine ⟨u, hb, ?_, hne, h0⟩
    by_contra hn
    have hoff := L.touch p m u hn
    rw [hoff] at h0
    exact hne h0

/-- THE LOST-BEARING PATH LEMMA over the board: a bearing is lost only if `s` is touched, or a touched place strictly between `s` and `t` became occupied (a unit arrived to block). -/
theorem Board.lost_path_lemma (p : State) (m : Move) (c : Side) (s t : Place)
    (h : b.bears p c s t) (h' : ¬ b.bears (b.apply p m) c s t) :
    b.touched p m s ∨ ∃ u, b.between s u t ∧ b.touched p m u ∧ b.occ (b.apply p m) u ≠ none ∧ b.occ p u = none := by
  rcases L.line (b.apply p m) p c s t h h' with hs | ⟨u, hb, hne, h0⟩
  · left
    by_contra hn
    exact hs (L.touch p m s hn).symm
  · right
    refine ⟨u, hb, ?_, hne, h0⟩
    by_contra hn
    have hoff := L.touch p m u hn
    rw [hoff] at hne
    exact hne h0

/-- BEARINGS ARE LOCAL TO THE TOUCHED PLACES: if `s` is not touched and no touched place lies strictly between `s` and `t`, whether `s` bears on `t` is the same before and after. -/
theorem Board.attack_unchanged (p : State) (m : Move) (c : Side) (s t : Place)
    (hs : ¬ b.touched p m s) (hb : ∀ u, b.between s u t → ¬ b.touched p m u) :
    b.bears (b.apply p m) c s t ↔ b.bears p c s t := by
  constructor
  · intro h'
    by_contra h
    rcases Board.path_lemma L p m c s t h' h with ⟨hs', _⟩ | ⟨u, hbu, hu, _, _⟩
    · exact hs hs'
    · exact hb u hbu hu
  · intro h
    by_contra h'
    rcases Board.lost_path_lemma L p m c s t h h' with hs' | ⟨u, hbu, hu, _, _⟩
    · exact hs hs'
    · exact hb u hbu hu

/-- THE FRAME-INDEPENDENCE LEMMA over the board (touch, line, footprint): if no touched place is in the footprint of any place of the frame `F`, what stands on `F` and every bearing
onto `F` are unchanged by the move. -/
theorem Board.frame_unchanged (p : State) (m : Move) (F : Place → Prop)
    (hF : ∀ t, F t → ∀ x, b.touched p m x → ¬ b.footprint t x) :
    (∀ t, F t → b.occ (b.apply p m) t = b.occ p t) ∧
    (∀ t, F t → ∀ c s, b.bears (b.apply p m) c s t ↔ b.bears p c s t) := by
  have off : ∀ t, F t → ¬ b.touched p m t := fun t ht hin => hF t ht t hin (L.footprint_refl t)
  refine ⟨fun t ht => L.touch p m t (off t ht), ?_⟩
  intro t ht c s
  by_cases hr : b.footprint t s
  · -- `s` is in the footprint, so it is untouched; so is every place between (each lies in the footprint of `t`)
    apply Board.attack_unchanged L p m c s t
    · intro hin; exact hF t ht s hin hr
    · intro u hb hin; exact hF t ht u hin (L.between_footprint s u t hb)
  · -- `s` cannot bear on `t`: it bears on neither state
    exact ⟨fun h => absurd (L.bears_footprint _ c s t h) hr, fun h => absurd (L.bears_footprint _ c s t h) hr⟩

end D2

end Ideas.Core
