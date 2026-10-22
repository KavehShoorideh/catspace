import Ideas.Passer
import Ideas.FrameLemma
import Ideas.ForcedByMeasure

/-!
# The passer RACE word, tempo-immunity, and the frame's soundness condition (ANALYST_PLAN 62.1, 62.2; GO 2026-10-22)

The case (RoR8t, `Kg4 Rf4 f5 h3 v Kg1 Rb8 a2 b3`, Black to move): the stopper defect says "their K on g1 reaches h1 in 1, the pawn needs 2" -- true by
its definition and too weak: a king that must be on h1 is not a stopper once our king covers h2 and a second passer deflects it. This file is the VOCABULARY for
that argument, in three layers; each theorem is proved by reasoning under the standard axioms (no `sorry`, no `decide` over the board).

**What is proved, and what is not (said once, plainly).** The words are DEFINED over squares and counts (`kdist`, `pushes`, `path` of `Ideas.Passer`); their soundness is proved in two kinds:

* *the arithmetic of the race* -- `race_sound` (an abstract game with a tempo measure `μ` of ours and `ν` of theirs: if our moves lower `μ`, their replies never raise it (the
  stopper analysis: `held`), and theirs is not faster, we force the goal within `μ` of our moves and they never complete first), `race_first`; and
* *the link to the rules where it is exact* -- `no_stand_adjacent`, `no_stand_attacked`: a LEGAL move never leaves the mover's king on a square attacked by the other side, so
  a square next to the other king, or attacked by one of its pawns, is DENIED to a king (`denied`). Whether a stopper is held by denial is therefore a theorem of the rules.

What is NOT proved: that every position satisfying the Python mirror's counts satisfies the hypotheses of `race_sound` for the full rules (a rook of theirs can give
checks). The mirror (`regress.race`) therefore serves the word ONLY on boards where it has no piece of theirs but the king and pawns (see its docstring), and the
test file checks it against the cases below.

* 62.1 `promoTempi`, `raceBinds`, `race_sound`, `race_first`; `holds` / `holdSet` / `overloaded` (the deflection: a king that must hold two promotion squares at once, and
  the two hold sets are disjoint once the squares we deny are taken out), `overloaded_exclusive`, `overloaded_far`.
* 62.2 `frame_sound` (a frame claim that is FORCED against every reply of the frame AND against a pass is the whole-board claim when every whole-board reply projects onto a
  frame reply or a pass), `excluded_is_pass` (an excluded piece's move touching no square that bears on the frame changes nothing the frame reads: `FrameLemma.frame_unchanged`),
  `far_stays_off` (a piece that needs more than `k` moves to reach any square that bears on the frame touches none in `k` moves), `lb_step` (the lower bound that serves
  rooks, bishops, queens and knights).
-/

namespace Ideas.Race

open Chess
open Ideas.Passer
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

/-! ### The rules link: a king never stands on a square the other side attacks -/

/-- A LEGAL move leaves the mover's king on a square the other side does not attack. -/
theorem legal_king_safe {p : Position} {m : Move} {t : Square} (hl : legalMove p m)
    (hk : kingSquare (applyMove p m).board p.turn = some t) :
    ¬ isAttacked (applyMove p m) (Color.opposite p.turn) t := by
  have h := legalMove_kingSafe hl
  rw [inCheck_iff hk] at h
  exact h

/-- DENIED BY ATTACK: after a legal move of `p.turn` whose king stands on `t`, no piece of the other side attacks `t`. -/
theorem no_stand_attacked {p : Position} {m : Move} {t s : Square} (hl : legalMove p m)
    (hk : kingSquare (applyMove p m).board p.turn = some t)
    (hs : pieceAttacks (applyMove p m) (Color.opposite p.turn) s t) : False :=
  legal_king_safe hl hk ((isAttacked_iff _ _ _).2 ⟨s, hs⟩)

/-- A step of the king is a king step: `kdist ≤ 1` between distinct squares. -/
theorem kingGeometryB_of_kdist {e t : Square} (hne : e ≠ t) (hd : kdist e t ≤ 1) : kingGeometryB e t = true := by
  have h := Nat.max_le.mp hd
  simp only [kingGeometryB, Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq]
  exact ⟨⟨hne, h.1⟩, h.2⟩

/-- DENIED BY THE OTHER KING: a king never stands next to the other king (`kdist ≤ 1`, distinct squares). This is why our king on g3 covers h2. -/
theorem no_stand_adjacent {p : Position} {m : Move} {t e : Square} (hl : legalMove p m)
    (hk : kingSquare (applyMove p m).board p.turn = some t)
    (he : (applyMove p m).board e = some ⟨Color.opposite p.turn, .king⟩) (hne : e ≠ t) (hd : kdist e t ≤ 1) : False :=
  no_stand_attacked hl hk (s := e) (by
    unfold pieceAttacks
    rw [Ideas.Passer.king_attack he]
    exact kingGeometryB_of_kdist hne hd)

/-- DENIED BY A PAWN: a king never stands on a square a pawn of the other side attacks (this is how the f-pawn on f2 takes g1 and e1 from the king). -/
theorem no_stand_pawn {p : Position} {m : Move} {t s : Square} (hl : legalMove p m)
    (hk : kingSquare (applyMove p m).board p.turn = some t)
    (hs : (applyMove p m).board s = some ⟨Color.opposite p.turn, .pawn⟩)
    (hg : pawnAttackGeometryB (Color.opposite p.turn) s t = true) : False :=
  no_stand_attacked hl hk (s := s) (by
    unfold pieceAttacks
    rw [Ideas.Passer.pawn_attack hs]
    exact hg)

/-! ### The deflection: a king cannot hold two promotion squares -/

/-- The king stops a pawn about to promote on `x` from `s` when it stands on `x` (the block) or next to it (it takes the queen). -/
def holds (x s : Square) : Bool := decide (kdist s x ≤ 1)

/-- The squares from which it holds `x` and may stand: not DENIED (next to our king, attacked by our pawns or pieces -- the theorems above). -/
def holdSet (denied : Square → Bool) (x s : Square) : Bool := holds x s && !denied s

/-- OVERLOADED: no square holds both `a` and `b`. -/
def overloaded (denied : Square → Bool) (a b : Square) : Bool :=
  allSquares.all fun s => !(holdSet denied a s && holdSet denied b s)

/-- A king that holds one of two overloaded promotion squares does not hold the other. -/
theorem overloaded_exclusive {d : Square → Bool} {a b : Square} (h : overloaded d a b = true) {s : Square}
    (hs : holdSet d a s = true) : holdSet d b s = false := by
  simp only [overloaded, List.all_eq_true, Bool.not_eq_true'] at h
  have := h s (mem_allSquares s)
  simp only [hs, Bool.true_and] at this
  exact this

/-- Promotion squares more than two apart are overloaded whatever is denied (the triangle inequality). -/
theorem overloaded_far {a b : Square} (h : 2 < kdist a b) (d : Square → Bool) : overloaded d a b = true := by
  simp only [overloaded, List.all_eq_true, Bool.not_eq_true']
  intro s _
  by_contra hc
  simp only [Bool.not_eq_false, Bool.and_eq_true, holdSet, holds, decide_eq_true_eq] at hc
  have t := kdist_triangle a s b
  have c1 := kdist_comm a s
  omega

/-- THE DEFLECTION IN TIME: if a king must stay on a hold square of `a` from step `i` on (else `a` promotes), then from that step it is on no hold square of `b`
(and `b` promotes when it arrives). -/
theorem overload_sound {d : Square → Bool} {a b : Square} (h : overloaded d a b = true) (kpos : Nat → Square) {i : Nat}
    (hstay : ∀ j, i ≤ j → holdSet d a (kpos j) = true) : ∀ j, i ≤ j → holdSet d b (kpos j) = false :=
  fun j hj => overloaded_exclusive h (hstay j hj)

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

/-- An EXCLUDED PIECE'S MOVE IS A PASS: if no square it touches bears on the frame, the board on the frame and every attack onto the frame are as they were
(`Ideas.FrameLemma.frame_unchanged`), so the frame's projection does not change. -/
theorem excluded_is_pass (p : Position) (m : Move) (F : Square → Prop)
    (hF : ∀ t, F t → ∀ x, x ∈ Ideas.PathLemma.touched p m → Ideas.FrameLemma.reachB t x = false) :
    (∀ t, F t → (applyMove p m).board t = p.board t) ∧
    (∀ t, F t → ∀ c s, pieceAttacksB (applyMove p m) c s t = pieceAttacksB p c s t) :=
  Ideas.FrameLemma.frame_unchanged p m F hF

/-- Iterated steps of a piece. -/
def Steps (step : Square → Square → Prop) : Nat → Square → Square → Prop
  | 0, s, s' => s = s'
  | j + 1, s, s' => ∃ u, step s u ∧ Steps step j u s'

/-- (a) OF THE CONDITION, for a piece with a step-1-Lipschitz lower bound `d` on the moves it needs (kings: `kdist`; pawns: `Passer2.pdist`; the rest: `lb`): a piece more than `k` moves
from every square that bears on the frame is, after `j ≤ k` moves, still more than `k - j` from each of them -- and stands on none. -/
theorem far_stays_off {step : Square → Square → Prop} (d : Square → Square → Nat)
    (hstep : ∀ s s' t, step s s' → d s t ≤ d s' t + 1) (B : Square → Prop) :
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

/-- ... so the piece stands on no square that bears on the frame during the word's `k` tempi. -/
theorem far_not_bearing {step : Square → Square → Prop} (d : Square → Square → Nat) (hself : ∀ t, d t t = 0)
    (hstep : ∀ s s' t, step s s' → d s t ≤ d s' t + 1) (B : Square → Prop) {k j : Nat} {s s' : Square}
    (hfar : ∀ t, B t → k < d s t) (hst : Steps step j s s') (hj : j ≤ k) : ¬ B s' := by
  intro hB
  have := far_stays_off d hstep B j k s s' hfar hst hj s' hB
  rw [hself] at this
  omega

/-- The king's bound: `kdist` is 1-Lipschitz along king steps. -/
theorem kdist_step {s s' t : Square} (h : kingGeometryB s s' = true) : kdist s t ≤ kdist s' t + 1 := by
  have := kdist_triangle s s' t
  have := kdist_king_step h
  omega

theorem kdist_self (t : Square) : kdist t t = 0 := by
  simp [kdist, absDiff]

/-- The lower bound for every other piece (rook, bishop, queen, knight): 0 on the square, 1 on a line or a knight's move, else 2 (no piece of these kinds needs more than two
moves except a knight, and a bound is all that is needed). -/
def lb (s t : Square) : Nat := if s = t then 0 else if queenGeometryB s t || knightGeometryB s t then 1 else 2

/-- A step of a rook, bishop, queen or knight. -/
def anyStep (s s' : Square) : Prop := queenGeometryB s s' = true ∨ knightGeometryB s s' = true

theorem lb_self (t : Square) : lb t t = 0 := by simp [lb]

/-- `lb` is 1-Lipschitz along such steps. -/
theorem lb_step {s s' t : Square} (h : anyStep s s') : lb s t ≤ lb s' t + 1 := by
  have hne : s ≠ s' := by
    intro e
    subst e
    rcases h with h | h <;>
      simp [queenGeometryB, rookGeometryB, bishopGeometryB, sameDiagonalB, knightGeometryB] at h
  have hg : (queenGeometryB s s' || knightGeometryB s s') = true := by
    rcases h with h | h <;> simp [h]
  unfold lb
  by_cases h1 : s = t
  · simp [h1]
  · by_cases h2 : s' = t
    · subst h2
      simp [h1, hg]
    · have : (if s' = t then 0 else if (queenGeometryB s' t || knightGeometryB s' t) = true then 1 else 2) ≥ 1 := by
        simp only [h2, if_false]; split_ifs <;> omega
      split_ifs <;> omega

/-! ### 62.1b Deny the stopper's path (Kaveh 2026-10-22)

My intent is to promote; their intent is to stop it; how -- which piece, along which path; can I deny those paths. The stopper's PATH is a walk of the piece (`Steps`) to a square of the
targets; a walk that lands only on squares it can STAND and SURVIVE on is a `UWalk`. A move of ours that leaves no `UWalk` of at most `k` moves to a target denies the path, and then the piece does
not stop the pawn within `k` moves: its real (legal) walks are `UWalk`s, because a legal move lands on a usable square (`legal_king_safe` for the king: a legal move never leaves it attacked). -/

/-- A walk over usable squares only: every landing square is one the piece can stand on and survive on. -/
def UWalk (step : Square → Square → Prop) (usable : Square → Prop) : Nat → Square → Square → Prop
  | 0, s, s' => s = s'
  | j + 1, s, s' => ∃ u, step s u ∧ usable u ∧ UWalk step usable j u s'

/-- The real walks of a piece (its legal moves) are walks over usable squares, when every legal move is a step that lands on a usable square. -/
theorem real_is_uwalk {step real : Square → Square → Prop} {usable : Square → Prop}
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

/-- DENY SOUND: if no walk over usable squares of at most `k` moves reaches a target, the piece never reaches a target in `k` real moves -- its path is denied, it does not stop the pawn
within `k`. (A route of ours is a denial exactly when it makes the hypothesis true; the mirror, `race.deny_moves`, searches the walks.) -/
theorem deny_sound {step real : Square → Square → Prop} {usable : Square → Prop} (T : Square → Prop) {k : Nat} {s : Square}
    (hleg : ∀ a b, real a b → step a b ∧ usable b)
    (hdeny : ∀ j, j ≤ k → ∀ t, T t → ¬ UWalk step usable j s t) :
    ∀ j t, j ≤ k → T t → ¬ Steps real j s t :=
  fun j t hj ht hw => hdeny j hj t ht (real_is_uwalk hleg j s t hw)

/-- The denial for a KING, tied to the rules: a king that lands by a legal move is on an unattacked square, so "usable" for it is "not attacked by the other side". -/
theorem king_usable {p : Position} {m : Move} {t : Square} (hl : legalMove p m)
    (hk : kingSquare (applyMove p m).board p.turn = some t) : ¬ isAttacked (applyMove p m) (Color.opposite p.turn) t :=
  legal_king_safe hl hk

/-! ### The RoR8t cases (checks of the definitions on the squares of the case, mirrored by tests/test_plan62.py; not lemmas)

Squares are (file, rank) from 0: g1 = (6,0), h1 = (7,0), f1 = (5,0), g2 = (6,1), e1 = (4,0), g3 = (6,2) -- Black's king on g3 denies h2, g2, f2 (the squares next to it). -/

/-- Their king on g1 or h1 holds the h-pawn's promotion square h1. -/
example : holds (7, 0) (6, 0) = true := by decide
example : holds (7, 0) (7, 0) = true := by decide

/-- With our king on g3 and the f-pawn on f2 the squares denied to their king are g2 (next to g3), g1 and e1 (attacked by the pawn on f2): the holds of h1 are {h1} and
the holds of f1 are not h1, so the two promotion squares are overloaded. -/
def deniedRoR8t (s : Square) : Bool :=
  decide (kdist s (6, 2) ≤ 1) || (s == (6, 0)) || (s == (4, 0))

example : overloaded deniedRoR8t (7, 0) (5, 0) = true := by decide

/-- ... and without the f-pawn's two squares (g1 free) one king covers both h1 and f1: not overloaded. -/
def deniedKingOnly (s : Square) : Bool := decide (kdist s (6, 2) ≤ 1)

example : overloaded deniedKingOnly (7, 0) (5, 0) = false := by decide

end Ideas.Race
