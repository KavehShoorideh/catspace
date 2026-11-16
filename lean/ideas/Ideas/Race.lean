import Ideas.Passer
import Ideas.FrameLemma
import Ideas.Core.Race

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

/-- OVERLOADED: no square holds both `a` and `b` (`Ideas.Race.overloadedOver` over the complete list of squares). -/
def overloaded (denied : Square → Bool) (a b : Square) : Bool :=
  overloadedOver allSquares (holdSet denied a) (holdSet denied b)

/-- A king that holds one of two overloaded promotion squares does not hold the other. -/
theorem overloaded_exclusive {d : Square → Bool} {a b : Square} (h : overloaded d a b = true) {s : Square}
    (hs : holdSet d a s = true) : holdSet d b s = false :=
  overloadedOver_exclusive mem_allSquares h hs

/-- Promotion squares more than two apart are overloaded whatever is denied (the triangle inequality). -/
theorem overloaded_far {a b : Square} (h : 2 < kdist a b) (d : Square → Bool) : overloaded d a b = true := by
  simp only [overloaded, overloadedOver, List.all_eq_true, Bool.not_eq_true']
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
  overload_sound_over mem_allSquares h kpos hstay

/-- An EXCLUDED PIECE'S MOVE IS A PASS: if no square it touches bears on the frame, the board on the frame and every attack onto the frame are as they were
(`Ideas.FrameLemma.frame_unchanged`), so the frame's projection does not change. -/
theorem excluded_is_pass (p : Position) (m : Move) (F : Square → Prop)
    (hF : ∀ t, F t → ∀ x, x ∈ Ideas.PathLemma.touched p m → Ideas.FrameLemma.reachB t x = false) :
    (∀ t, F t → (applyMove p m).board t = p.board t) ∧
    (∀ t, F t → ∀ c s, pieceAttacksB (applyMove p m) c s t = pieceAttacksB p c s t) :=
  Ideas.FrameLemma.frame_unchanged p m F hF

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

/-- The denial for a KING, tied to the rules: a king that lands by a legal move is on an unattacked square, so "usable" for it is "not attacked by the other side". -/
theorem king_usable {p : Position} {m : Move} {t : Square} (hl : legalMove p m)
    (hk : kingSquare (applyMove p m).board p.turn = some t) : ¬ isAttacked (applyMove p m) (Color.opposite p.turn) t :=
  legal_king_safe hl hk

/-! ### 62.1c Lifting the kill for the pieces that are not the king (Kaveh 2026-10-22)

With a rook or a knight of theirs on the board the stopper verdict ("it reaches the path in k, the pawn needs n") is a STATIC estimate (the other pieces stand still, no checks): as a KILL of the
promote question it is unsound for those pieces. Only their KING is judged by the proved analysis above (stand and survive, deny, deflect). The other pieces are not stoppers for the verdict at all: they are
REPLIES, and the chase reads every reply. A question whose kill is lifted is therefore a GAP until the chase closes it, and is claimed only by what the chase closes -- never by the lift itself. -/

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
