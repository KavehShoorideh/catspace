import Chess

/-!
# The path lemma (Kaveh 2026-10-09)

"If the line from the king is blocked, there's no piece that can attack from there -- that's a base lemma -- so we
should be able to see that we have accounted for all the threats, because there is no path to attack the king."

An attack of square `s` on square `t` (`pieceAttacksB`) reads only the piece on `s` and, for a slider, whether the
squares strictly between `s` and `t` are empty. So:

* `new_attack_local` (any two boards): if `s` attacks `t` on the second board and not on the first, then the piece on
  `s` changed, or some square strictly between `s` and `t` went from occupied to empty. Filling a square can only
  block.
* `applyBoard_off`: a move changes the board only on its `touched` squares -- its source and destination, the castling
  rook's two squares, en passant's captured square.
* `path_lemma`: so after a move, a NEW attack of `s` on `t` comes from a piece that ARRIVED on `s` (a touched square,
  occupied after the move), or through a touched square the move VACATED on the line between `s` and `t` (a
  discovery). This is the rule the Rust path lemma (rust/analyst_core/src/paths.rs) implements, audited there against
  brute force on 836,694 moves.
-/

namespace Ideas.PathLemma

open Chess

/-- The squares a move can change on the board. -/
def touched (p : Position) (m : Move) : List Square :=
  [m.source, m.destination] ++
    (match castleRookSource m with | some r => [r] | none => []) ++
    (match castleRookDestination m with | some r => [r] | none => []) ++
    (match m with
      | .enPassant _ t => (match epCapturedSquare p.turn t with | some cs => [cs] | none => [])
      | _ => [])

/-- A move leaves every square off its `touched` list as it was. -/
theorem applyBoard_off (p : Position) (m : Move) (u : Square) (hu : u ∉ touched p m) :
    applyBoard p m u = p.board u := by
  cases m with
  | normal s t =>
      simp_all [touched, applyBoard, boardMove, boardUpdate, Move.source, Move.destination,
        castleRookSource, castleRookDestination]
  | promotion s t k =>
      simp_all [touched, applyBoard, boardUpdate, Move.source, Move.destination,
        castleRookSource, castleRookDestination]
  | enPassant s t =>
      cases hcs : epCapturedSquare p.turn t with
      | none =>
          simp_all [touched, applyBoard, boardMove, boardUpdate, Move.source,
            Move.destination, castleRookSource, castleRookDestination]
      | some cs =>
          simp_all [touched, applyBoard, boardMove, boardUpdate, Move.source,
            Move.destination, castleRookSource, castleRookDestination]
  | whiteKingCastle | whiteQueenCastle | blackKingCastle | blackQueenCastle =>
      simp_all [touched, applyBoard, castleBoard, boardMove, boardUpdate, Move.source,
        Move.destination, castleRookSource, castleRookDestination]

/-- Emptying no square between `s` and `t` cannot clear a blocked line: if every square that is empty on `b'` was
already empty on `b`, a clear line on `b'` was clear on `b`. -/
theorem clear_mono (b b' : Board) (s t : Square)
    (hu : ∀ u, between s u t → b' u = none → b u = none)
    (h : clearBetweenB b' s t = true) : clearBetweenB b s t = true := by
  have h1 := (clearBetween_iff b' s t).mp h
  exact (clearBetween_iff b s t).mpr (fun u hm hb => hu u hb (h1 u hm hb))

/-- THE LOCALITY OF ATTACKS, for any two boards: a new attack of `s` on `t` needs a changed piece on `s`, or a square
strictly between them that went from occupied to empty. -/
theorem new_attack_local (p p' : Position) (c : Color) (s t : Square)
    (h' : pieceAttacksB p' c s t = true) (h : pieceAttacksB p c s t = false) :
    p'.board s ≠ p.board s ∨ ∃ u, between s u t ∧ p.board u ≠ none ∧ p'.board u = none := by
  by_contra hc
  push Not at hc
  obtain ⟨hs, hu⟩ := hc
  have hmono : clearBetweenB p'.board s t = true → clearBetweenB p.board s t = true :=
    clear_mono p.board p'.board s t (fun u hb h0 => by
      by_contra hne
      exact (hu u hb hne) h0)
  have key : pieceAttacksB p c s t = true := by
    unfold pieceAttacksB sliderAttackB at h' ⊢
    rw [hs] at h'
    cases hb : p.board s with
    | none => simp [hb] at h'
    | some q =>
        simp only [hb] at h' ⊢
        revert h'
        cases hA' : clearBetweenB p'.board s t <;> cases hA : clearBetweenB p.board s t <;> simp_all <;> tauto
  rw [key] at h
  exact absurd h (by decide)

/-- THE PATH LEMMA: after a move, a new attack of `s` on `t` comes from a piece that arrived on `s` (a touched square,
occupied after the move), or through a touched square the move vacated strictly between `s` and `t`. -/
theorem path_lemma (p : Position) (m : Move) (c : Color) (s t : Square)
    (h' : pieceAttacksB (applyMove p m) c s t = true) (h : pieceAttacksB p c s t = false) :
    (s ∈ touched p m ∧ (applyMove p m).board s ≠ none) ∨
      ∃ u, between s u t ∧ u ∈ touched p m ∧ p.board u ≠ none ∧ (applyMove p m).board u = none := by
  rcases new_attack_local p (applyMove p m) c s t h' h with hs | ⟨u, hb, hne, h0⟩
  · left
    refine ⟨?_, ?_⟩
    · by_contra hn
      exact hs (by rw [applyMove_board]; exact applyBoard_off p m s hn)
    · intro h0
      exact pieceAttacks_empty_source (applyMove p m) c s t h0 h'
  · right
    refine ⟨u, hb, ?_, hne, h0⟩
    by_contra hn
    have hoff := applyBoard_off p m u hn
    rw [applyMove_board, hoff] at h0
    exact hne h0

/-- THE SAFETY FORM (what the planner checks for "have I accounted for all the threats?"): if no piece that arrives on a
touched square attacks `t`, and no line to `t` opens through a touched square the move vacated, then a square `t` that
side `c` did not attack before is not attacked after the move. -/
theorem no_new_attack (p : Position) (m : Move) (c : Color) (t : Square)
    (harrive : ∀ s, s ∈ touched p m → pieceAttacksB (applyMove p m) c s t = false)
    (hline : ∀ s u, between s u t → u ∈ touched p m → p.board u ≠ none → (applyMove p m).board u = none →
      pieceAttacksB (applyMove p m) c s t = false)
    (hbefore : ∀ s, pieceAttacksB p c s t = false) :
    ∀ s, pieceAttacksB (applyMove p m) c s t = false := by
  intro s
  cases h' : pieceAttacksB (applyMove p m) c s t with
  | false => rfl
  | true =>
      rcases path_lemma p m c s t h' (hbefore s) with ⟨hs, _⟩ | ⟨u, hb, hu, hne, h0⟩
      · rw [harrive s hs] at h'; exact absurd h' (by decide)
      · rw [hline s u hb hu hne h0] at h'; exact absurd h' (by decide)

/-- THE LOST ATTACK, for any two boards (the path lemma's other direction; 2026-10-11, the incremental static exchange
rests on both): if `s` attacks `t` on the first board and not on the second, the piece on `s` changed, or a square
strictly between them went from empty to occupied (filling a square blocks). The new-attack lemma with the boards
swapped. -/
theorem lost_attack_local (p p' : Position) (c : Color) (s t : Square)
    (h : pieceAttacksB p c s t = true) (h' : pieceAttacksB p' c s t = false) :
    p.board s ≠ p'.board s ∨ ∃ u, between s u t ∧ p'.board u ≠ none ∧ p.board u = none :=
  new_attack_local p' p c s t h h'

/-- THE LOST-ATTACK PATH LEMMA: after a move, an attack of `s` on `t` is lost only if `s` is a touched square (the piece
left or was taken), or a touched square strictly between `s` and `t` became occupied (a piece arrived to block). -/
theorem lost_path_lemma (p : Position) (m : Move) (c : Color) (s t : Square)
    (h : pieceAttacksB p c s t = true) (h' : pieceAttacksB (applyMove p m) c s t = false) :
    s ∈ touched p m ∨ ∃ u, between s u t ∧ u ∈ touched p m ∧ (applyMove p m).board u ≠ none ∧ p.board u = none := by
  rcases lost_attack_local p (applyMove p m) c s t h h' with hs | ⟨u, hb, hne, h0⟩
  · left
    by_contra hn
    exact hs (by rw [applyMove_board]; exact (applyBoard_off p m s hn).symm)
  · right
    refine ⟨u, hb, ?_, hne, h0⟩
    by_contra hn
    have hoff := applyBoard_off p m u hn
    rw [applyMove_board, hoff] at hne
    exact hne h0

/-- ATTACKS ARE LOCAL TO THE TOUCHED SQUARES: if `s` is not touched and no touched square lies strictly between `s` and
`t`, then whether `s` attacks `t` is the same before and after the move. (What lets the static exchange on `t` be kept
when no touched square lies on a line through `t`, a knight's move from it or next to it: rules.rs exchange_affected.) -/
theorem attack_unchanged (p : Position) (m : Move) (c : Color) (s t : Square)
    (hs : s ∉ touched p m) (hb : ∀ u, between s u t → u ∉ touched p m) :
    pieceAttacksB (applyMove p m) c s t = pieceAttacksB p c s t := by
  cases h : pieceAttacksB p c s t <;> cases h' : pieceAttacksB (applyMove p m) c s t
  · rfl
  · rcases path_lemma p m c s t h' h with ⟨hs', _⟩ | ⟨u, hbu, hu, _, _⟩
    · exact absurd hs' hs
    · exact absurd hu (hb u hbu)
  · rcases lost_path_lemma p m c s t h h' with hs' | ⟨u, hbu, hu, _, _⟩
    · exact absurd hs' hs
    · exact absurd hu (hb u hbu)
  · rfl

end Ideas.PathLemma
