import Ideas.Passer
import Chess.Game

/-!
# Rule 2, THE DOUBLE THREAT (fork, discovered attack): a bounded local check, certified

Kaveh 2026-10-05: a rule in play may run "a bounded local check inside its own frame, like the frames' exchange
resolution", never a tree search over the position. The check: after our move `m`, for EVERY legal reply, either
our material is already up by `k` or some legal answer of ours makes it so -- two plies, the kernel's legal moves,
the kernel's `applyMove` (en passant, promotion, castling as the rules say). A node where we are to move and have no
legal move (mated or stalemated) never counts as gained.

`localCheck_sound`: the check implies `forcedWithinB` (Ideas.Passer: the goal holds within n plies whatever the
opponent does) for two plies after the move. The proof is a rewrite: the check IS the forced-within computation,
written in the detector's shape. What the theorem buys is the conventions: the Python prototype's bridge disagreed
with its own truth twice on 2026-10-05 (a stalemating reply, a mating capture) because the two sides counted
differently; here both are the kernel's.

Not covered, deliberately: counterplay beyond the second ply (their recapture or counter-threat on ply 3): the claim
is a fact about material two plies after the move, the ledger's own horizon kind.
-/

namespace Ideas

open Chess

def pieceValue : PieceKind → Int
  | .pawn => 1
  | .knight => 3
  | .bishop => 3
  | .rook => 5
  | .queen => 9
  | .king => 0

def squareValue (b : Board) (c : Color) (s : Square) : Int :=
  match b s with
  | none => 0
  | some pc => if pc.color = c then pieceValue pc.kind else -(pieceValue pc.kind)

/-- Material for `c`: its pieces less the opponent's, pawn 1, knight 3, bishop 3, rook 5, queen 9. -/
def material (p : Position) (c : Color) : Int :=
  (allSquares.map (squareValue p.board c)).sum

/-- Material for `c` has reached `target`. -/
def reached (c : Color) (target : Int) (q : Position) : Bool :=
  decide (target ≤ material q c)

/-- A capture: the destination holds a piece, or the move is en passant. -/
def isCaptureB (q : Position) (m : Move) : Bool :=
  match m with
  | .enPassant _ _ => true
  | _ => (q.board m.destination).isSome

/-- `c` reaches `goal` at the leaves `n` plies on, whatever the opponent does: at `c`'s nodes some legal move, at the
opponent's every legal move; the opponent with no legal move is mated (`c` has won) or stalemated (not a gain); `c`
with no legal move has not gained. Unlike `forcedWithinB`, the goal is read at the leaves only: material is
reversible, so "reached at some node" would count a capture before its recapture. -/
def forcedAtLeaves (c : Color) (goal : Position → Bool) : Position → Nat → Bool
  | q, 0 => goal q
  | q, n + 1 =>
      if q.turn = c then
        (legalMoves q).any (fun m => forcedAtLeaves c goal (applyMove q m) n)
      else
        let ms := legalMoves q
        if ms.isEmpty then inCheckB q q.turn
        else ms.all (fun r => forcedAtLeaves c goal (applyMove q r) n)

/-- THE LOCAL CHECK of move `m` for a gain of `k`: every legal reply is answered by one of our CAPTURES after which,
whatever they reply, material is up by `k` (their mate counts as won, their stalemate as not gained). Three plies
after the move, the kernel's legal moves and `applyMove` throughout. -/
def localCheck (p : Position) (m : Move) (k : Int) : Bool :=
  let c := p.turn
  let goal := reached c (material p c + k)
  let q := applyMove p m
  let replies := legalMoves q
  if replies.isEmpty then inCheckB q q.turn
  else replies.all (fun r =>
    let q2 := applyMove q r
    ((legalMoves q2).filter (isCaptureB q2)).any (fun m2 => forcedAtLeaves c goal (applyMove q2 m2) 1))

theorem opposite_ne (c : Color) : Color.opposite c ≠ c := by
  cases c <;> simp [Color.opposite]

theorem any_filter_imp {α : Type} (l : List α) (f : α → Bool) (g : α → Bool)
    (h : (l.filter f).any g = true) : l.any g = true := by
  simp only [List.any_eq_true, List.mem_filter] at h ⊢
  obtain ⟨x, ⟨hx, _⟩, hg⟩ := h
  exact ⟨x, hx, hg⟩

/-- The local check implies the forced gain three plies after `m`: our answers were a sublist of the legal moves. -/
theorem localCheck_sound (p : Position) (m : Move) (k : Int) (h : localCheck p m k = true) :
    forcedAtLeaves p.turn (reached p.turn (material p p.turn + k)) (applyMove p m) 3 = true := by
  have hq : (applyMove p m).turn ≠ p.turn := by
    rw [applyMove_turn]; exact opposite_ne p.turn
  have hr : ∀ r : Move, (applyMove (applyMove p m) r).turn = p.turn := by
    intro r
    rw [applyMove_turn, applyMove_turn, Color.opposite_opposite]
  simp only [localCheck] at h
  show forcedAtLeaves p.turn (reached p.turn (material p p.turn + k)) (applyMove p m) (2 + 1) = true
  simp only [forcedAtLeaves, if_neg hq]
  by_cases hemp : (legalMoves (applyMove p m)).isEmpty = true
  · simp only [hemp, if_true] at h ⊢
    exact h
  · simp only [hemp, if_false, Bool.false_eq_true] at h ⊢
    simp only [List.all_eq_true] at h ⊢
    intro r hrm
    have hh := h r hrm
    simp only [forcedAtLeaves, if_pos (hr r)]
    exact any_filter_imp _ _ _ hh

end Ideas
