/-!
# Where a king and a queen can mate (Kaveh 2026-10-12)

"In order to mate with a king and a queen, the lone king must be on the edge." Reached by REASONING, not by search --
the process, step by step:

1. THE DEMAND (the definition unfolded): mate means the king is attacked and so is every square it could step to --
   every square of its 3×3 box is covered (a square the queen stands on only if our king guards it, or the lone king
   simply takes her).
2. THE SUPPLY OF EACH PIECE, by its geometry: our king may never touch theirs, so it covers squares on ONE SIDE of the
   box only (`king_one_side`); the queen covers along her lines.
3. THE MOST CONSTRAINED PIECE CLAIMS FIRST: whatever side our king takes, the rest of the box -- a 2×3 rectangle holding
   the lone king -- falls to the queen alone, and no square lets one queen cover a 2×3 rectangle (`queen_rect_v`,
   `queen_rect_h`): from outside her lines meet it in at most four squares; inside, her own square stays uncovered.

So in the open -- a king with its whole box on the board -- there is no mate (`no_queen_mate_in_open`). The plane here has
no edge, so every position whose lone king stands off the edge is an instance of it: the box is on the board, and a line
between two squares of the board stays on the board. Blockers are ignored: that only adds to what the queen covers, so
the impossibility holds a fortiori. On the edge the demand drops to 6 squares (4 in a corner) and the argument no longer
closes -- which is where the mates are.
-/

namespace Ideas.QueenMate

/-- Our king on (a, b) attacks (x, y): one step in any direction. -/
def kingHits (a b x y : Int) : Prop := a - x ≤ 1 ∧ x - a ≤ 1 ∧ b - y ≤ 1 ∧ y - b ≤ 1 ∧ ¬ (a = x ∧ b = y)

/-- The queen on (a, b) has a line to (x, y): same file, rank or diagonal, another square (blockers ignored -- more
than she attacks, never less). -/
def queenLine (a b x y : Int) : Prop := ¬ (a = x ∧ b = y) ∧ (a = x ∨ b = y ∨ a - x = b - y ∨ a - x = y - b)

/-- The square (x, y) is covered by our king on (wa, wb) or our queen on (qa, qb). -/
def covered (wa wb qa qb x y : Int) : Prop := kingHits wa wb x y ∨ queenLine qa qb x y

/-- THE KING'S SUPPLY: two files to the right of the lone king (on (ka, kb)), our king touches no square of its box
left of the box's right column. -/
theorem king_one_side (ka kb wa wb i j : Int) (hfar : 2 ≤ wa - ka) (hi : i ≤ 0) : ¬ kingHits wa wb (ka + i) (kb + j) := by
  unfold kingHits; omega

/-- THE QUEEN AGAINST A 2×3 RECTANGLE (two files, three ranks; its lower-left square (x, y)): no square of the queen has a
line to all six. -/
theorem queen_rect_v (qa qb x y : Int)
    (h : ∀ i j : Int, 0 ≤ i → i ≤ 1 → 0 ≤ j → j ≤ 2 → queenLine qa qb (x + i) (y + j)) : False := by
  have h1 := h 0 0 (by omega) (by omega) (by omega) (by omega)
  have h2 := h 0 1 (by omega) (by omega) (by omega) (by omega)
  have h3 := h 0 2 (by omega) (by omega) (by omega) (by omega)
  have h4 := h 1 0 (by omega) (by omega) (by omega) (by omega)
  have h5 := h 1 1 (by omega) (by omega) (by omega) (by omega)
  have h6 := h 1 2 (by omega) (by omega) (by omega) (by omega)
  unfold queenLine at h1 h2 h3 h4 h5 h6
  omega

/-- The same, three files and two ranks. -/
theorem queen_rect_h (qa qb x y : Int)
    (h : ∀ i j : Int, 0 ≤ i → i ≤ 2 → 0 ≤ j → j ≤ 1 → queenLine qa qb (x + i) (y + j)) : False := by
  have h1 := h 0 0 (by omega) (by omega) (by omega) (by omega)
  have h2 := h 1 0 (by omega) (by omega) (by omega) (by omega)
  have h3 := h 2 0 (by omega) (by omega) (by omega) (by omega)
  have h4 := h 0 1 (by omega) (by omega) (by omega) (by omega)
  have h5 := h 1 1 (by omega) (by omega) (by omega) (by omega)
  have h6 := h 2 1 (by omega) (by omega) (by omega) (by omega)
  unfold queenLine at h1 h2 h3 h4 h5 h6
  omega

/-- NO MATE IN THE OPEN: with the kings apart (two or more files or ranks), our king and queen never cover the whole
3×3 box of the lone king on (ka, kb). -/
theorem no_queen_mate_in_open (ka kb wa wb qa qb : Int)
    (hfar : 2 ≤ wa - ka ∨ 2 ≤ ka - wa ∨ 2 ≤ wb - kb ∨ 2 ≤ kb - wb)
    (hbox : ∀ i j : Int, -1 ≤ i → i ≤ 1 → -1 ≤ j → j ≤ 1 → covered wa wb qa qb (ka + i) (kb + j)) : False := by
  -- the queen alone must cover the box away from our king's side
  have only_queen : ∀ i j : Int, -1 ≤ i → i ≤ 1 → -1 ≤ j → j ≤ 1 → ¬ kingHits wa wb (ka + i) (kb + j) →
      queenLine qa qb (ka + i) (kb + j) := fun i j a b c d hk => (hbox i j a b c d).resolve_left hk
  rcases hfar with h | h | h | h
  · -- our king on the right: the left two files of the box are the queen's
    exact queen_rect_v qa qb (ka - 1) (kb - 1) fun i j a b c d => by
      have := only_queen (i - 1) (j - 1) (by omega) (by omega) (by omega) (by omega) (by unfold kingHits; omega)
      rwa [show ka + (i - 1) = ka - 1 + i by omega, show kb + (j - 1) = kb - 1 + j by omega] at this
  · -- on the left: the right two files
    exact queen_rect_v qa qb ka (kb - 1) fun i j a b c d => by
      have := only_queen i (j - 1) (by omega) (by omega) (by omega) (by omega) (by unfold kingHits; omega)
      rwa [show kb + (j - 1) = kb - 1 + j by omega] at this
  · -- above: the lower two ranks
    exact queen_rect_h qa qb (ka - 1) (kb - 1) fun i j a b c d => by
      have := only_queen (i - 1) (j - 1) (by omega) (by omega) (by omega) (by omega) (by unfold kingHits; omega)
      rwa [show ka + (i - 1) = ka - 1 + i by omega, show kb + (j - 1) = kb - 1 + j by omega] at this
  · -- below: the upper two ranks
    exact queen_rect_h qa qb (ka - 1) kb fun i j a b c d => by
      have := only_queen (i - 1) j (by omega) (by omega) (by omega) (by omega) (by unfold kingHits; omega)
      rwa [show ka + (i - 1) = ka - 1 + i by omega] at this

end Ideas.QueenMate
