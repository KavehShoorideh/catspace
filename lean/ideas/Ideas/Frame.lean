import Ideas.Core.Frame
import Ideas.Game
import Ideas.PathLemma

/-!
# The neglect theorem: the frame lemma's contrapositive over `Game.step` (ANALYST_PLAN 72.10, Kaveh 2026-10-23)

"Qb7 threatens mate; who can be brought to the defence in one move? the queen cannot; a rook can, and is lost; can they delay? they
cannot." A player answers their move by looking where it changed something. The licence for looking nowhere else: every word carries a
READ SET, and a step that changes nothing the word reads leaves its value (`Ideas.FrameLemma.frame_unchanged`). The contrapositive is
the form the local inner agenda uses: anything NEW after a step -- a parry, a fresh threat, a delay -- is a word whose value differs,
so its reads MEET THE STEP'S DIFF. What is not a function of the squares (the fifty-move count, repetition, insufficient material) is a
terminal the Game reads, not a word of this kind.

* `ReadsOnly view w R`: two states that look the same on `R` give `w` the same value (the read set of `w`).
* `IsDiff g view D`: every location whose view a legal move changes is in `D` (the diff only has to be a SUPERSET of the change).
* `neglect` / `neglect_step`: GENERIC, over any `Game` -- a word whose value differs after a step has reads meeting the step's diff.
* Chess: `chessView` is the FrameLemma's pair of inputs for a square (the piece on it, every attack onto it); `chessDiff` is the touched
  squares, the squares a piece on a touched square attacks before or after the move, and the squares a line THROUGH a touched square
  reaches from an attacker before or after the move (the lines the move opened and closed: `Ideas.Battery`'s discoveries are the
  opened ones). `chessDiff_isDiff` proves it a diff (from `PathLemma.attack_unchanged`, which rests on the path lemma and the lost-attack
  path lemma), so `chess_neglect` holds.

The Python mirror is `regress.move_diff` (a superset of `chessDiff`: it takes each opened / closed line whole, up to and including its
first piece), tested against the brute-force "every square whose piece or attackers changed" on 50 random positions in
tests/test_plan72_10.py.
-/

namespace Ideas.Frame

open Chess
open Ideas.PathLemma

/- The generic half (`ReadsOnly`, `IsDiff`, `neglect`, `neglect_step`, `frame_of_diff`) is in `Ideas/Core/Frame.lean` (PLAN 78.56). -/


section Chess

/-- The chess view of a square: the piece on it and every attack onto it (`Ideas.FrameLemma`'s two inputs of a board-level proof). -/
def chessView (p : Position) (t : Square) : Option Piece × (Color → Square → Bool) :=
  (p.board t, fun c s => pieceAttacksB p c s t)

/-- THE DIFF of a chess move: `t` is touched, or some piece attacks `t` (before or after the move) from a touched square or along a line
through a touched square (a line the move opened or closed). -/
def chessDiff (p : Position) (m : Move) (t : Square) : Prop :=
  t ∈ touched p m ∨
    ∃ c s, (pieceAttacksB p c s t = true ∨ pieceAttacksB (applyMove p m) c s t = true) ∧
      (s ∈ touched p m ∨ ∃ u, between s u t ∧ u ∈ touched p m)

/-- An attack onto a square off the diff is the same before and after the move. -/
theorem attack_off_diff (p : Position) (m : Move) (t : Square) (h : ¬ chessDiff p m t) (c : Color) (s : Square) :
    pieceAttacksB (applyMove p m) c s t = pieceAttacksB p c s t := by
  by_cases hloc : s ∈ touched p m ∨ ∃ u, between s u t ∧ u ∈ touched p m
  · -- the attack could have changed, so it holds on neither board (else `t` is on the diff)
    have h1 : pieceAttacksB p c s t = false := by
      cases hb : pieceAttacksB p c s t with
      | false => rfl
      | true => exact absurd (Or.inr ⟨c, s, Or.inl hb, hloc⟩) h
    have h2 : pieceAttacksB (applyMove p m) c s t = false := by
      cases hb : pieceAttacksB (applyMove p m) c s t with
      | false => rfl
      | true => exact absurd (Or.inr ⟨c, s, Or.inr hb, hloc⟩) h
    rw [h1, h2]
  · -- neither `s` nor any square between it and `t` is touched: the path lemmas
    have hs : s ∉ touched p m := fun hin => hloc (Or.inl hin)
    have hb : ∀ u, between s u t → u ∉ touched p m := fun u hu hin => hloc (Or.inr ⟨u, hu, hin⟩)
    exact attack_unchanged p m c s t hs hb

/-- `chessDiff` IS A DIFF under `chessView`: a square whose piece or whose attacks changed lies on it. -/
theorem chessDiff_isDiff : IsDiff Ideas.Game.chess chessView chessDiff := by
  intro p m t _ hne
  by_contra h
  apply hne
  have ht : t ∉ touched p m := fun hin => h (Or.inl hin)
  show ((applyMove p m).board t, fun c s => pieceAttacksB (applyMove p m) c s t) = (p.board t, fun c s => pieceAttacksB p c s t)
  rw [applyMove_board, applyBoard_off p m t ht]
  congr 1
  funext c s
  exact attack_off_diff p m t h c s

/-- THE NEGLECT THEOREM FOR CHESS over `Game.step`: a word that reads the pieces on `R` and the attacks onto `R`, and whose value differs
after a step, reads a square of the step's diff. -/
theorem chess_neglect {α : Type} (w : Position → α) (R : Square → Prop) (hw : ReadsOnly chessView w R) {p q : Position}
    (hs : Ideas.Game.chess.step p q) (hne : w q ≠ w p) :
    ∃ m, m ∈ legalMoves p ∧ q = applyMove p m ∧ ∃ t, R t ∧ chessDiff p m t :=
  neglect_step Ideas.Game.chess chessView chessDiff chessDiff_isDiff w R hw hs hne

/-- The frame form for chess: a legal move whose diff misses every read of `w` leaves `w`'s value (the local inner agenda's licence to
neglect the rest of the board). -/
theorem chess_frame {α : Type} (w : Position → α) (R : Square → Prop) (hw : ReadsOnly chessView w R) {p : Position} {m : Move}
    (hm : m ∈ legalMoves p) (hmiss : ∀ t, R t → ¬ chessDiff p m t) : w (applyMove p m) = w p :=
  frame_of_diff Ideas.Game.chess chessView chessDiff chessDiff_isDiff w R hw hm hmiss

end Chess

end Ideas.Frame
