import Ideas.QueenMate

/-!
# The shapes of a king-and-queen mate (Kaveh 2026-10-12: the backward step -- from the goal to the target)

`no_queen_mate_in_open` says WHERE (the edge). Here the same demand on the edge says HOW the mate stands -- the target the
attacking king must reach. Coordinates relative to the lone king on (0, 0):

* ON AN EDGE (the board is the half-plane y ≥ 0; the box is the 2×3 rectangle on it): our king stands TWO RANKS IN
  FRONT, WITHIN A FILE (b = 2, |a| ≤ 1), and the queen stands directly in front (0, 1) or on the edge's own rank
  (`edge_shape`). Why: our king covers one side of the box; on the side it never touches, the queen would have to cover a
  2×3 rectangle (impossible, `queen_rect_h`) or a 2×2 block with the lone king in it; so it covers the front row, from
  the second rank; and the edge row is left to the queen -- along it, or from the one square that sees all three.
* IN A CORNER (the quadrant x, y ≥ 0; the box is 2×2): our king stands at distance exactly two (`corner_shape`): from
  further away it covers nothing, and no queen square covers the corner's box alone.

So in every king-and-queen mate our king stands two squares from theirs, on the inside -- THE TARGET of the approach: a
square the defender's king cannot move away from while it stays boxed on the edge (unlike "our king's distance to
theirs", which the defender undoes by stepping along the edge).
-/

namespace Ideas.QueenMate

/-- THE QUEEN AGAINST A 2×2 BLOCK (lower-left (x, y)): no queen square covers all four -- inside, her own square stays
uncovered; outside, her lines meet it in at most three. -/
theorem queen_block (qa qb x y : Int)
    (h : ∀ i j : Int, 0 ≤ i → i ≤ 1 → 0 ≤ j → j ≤ 1 → queenLine qa qb (x + i) (y + j)) : False := by
  have h1 := h 0 0 (by omega) (by omega) (by omega) (by omega)
  have h2 := h 1 0 (by omega) (by omega) (by omega) (by omega)
  have h3 := h 0 1 (by omega) (by omega) (by omega) (by omega)
  have h4 := h 1 1 (by omega) (by omega) (by omega) (by omega)
  unfold queenLine at h1 h2 h3 h4
  omega

/-- THE QUEEN ALONG THE EDGE ROW: a queen (on the board, d ≥ 0) with lines to the three squares (-1, 0), (0, 0), (1, 0)
stands on that row, or directly in front of its middle square. -/
theorem queen_row3 (c d : Int) (hd : 0 ≤ d) (h1 : queenLine c d (-1) 0) (h2 : queenLine c d 0 0) (h3 : queenLine c d 1 0) :
    d = 0 ∨ (c = 0 ∧ d = 1) := by
  unfold queenLine at h1 h2 h3
  omega

/-- THE EDGE SHAPE: the lone king on (0, 0), the board y ≥ 0; our king on (a, b) not touching it, our queen on (c, d). If
every square of the box on the board is covered, our king is two ranks in front within a file, and the queen directly in
front or on the edge's rank. -/
theorem edge_shape (a b c d : Int) (hb : 0 ≤ b) (hd : 0 ≤ d) (hfar : 2 ≤ a ∨ 2 ≤ -a ∨ 2 ≤ b)
    (hbox : ∀ i j : Int, -1 ≤ i → i ≤ 1 → 0 ≤ j → j ≤ 1 → covered a b c d i j) :
    b = 2 ∧ -1 ≤ a ∧ a ≤ 1 ∧ ((c = 0 ∧ d = 1) ∨ d = 0) := by
  have only_queen : ∀ i j : Int, -1 ≤ i → i ≤ 1 → 0 ≤ j → j ≤ 1 → ¬ kingHits a b i j → queenLine c d i j :=
    fun i j h1 h2 h3 h4 hk => (hbox i j h1 h2 h3 h4).resolve_left hk
  -- 1 our king is not to a side: the queen would cover the 2×2 block on the other side alone
  have hright : ¬ 2 ≤ a := fun h => queen_block c d (-1) 0 fun i j a1 a2 a3 a4 => by
    have := only_queen (i - 1) j (by omega) (by omega) (by omega) (by omega) (by unfold kingHits; omega)
    rwa [show i - 1 = -1 + i by omega, show j = 0 + j by omega] at this
  have hleft : ¬ 2 ≤ -a := fun h => queen_block c d 0 0 fun i j a1 a2 a3 a4 => by
    have := only_queen i j (by omega) (by omega) (by omega) (by omega) (by unfold kingHits; omega)
    rwa [show i = 0 + i by omega, show j = 0 + j by omega] at this
  -- 2 nor further than two ranks: the queen would cover the 2×3 rectangle alone
  have hnear : ¬ 3 ≤ b := fun h => queen_rect_h c d (-1) 0 fun i j a1 a2 a3 a4 => by
    have := only_queen (i - 1) j (by omega) (by omega) (by omega) (by omega) (by unfold kingHits; omega)
    rwa [show i - 1 = -1 + i by omega, show j = 0 + j by omega] at this
  have hb2 : b = 2 := by omega
  -- 3 the edge row is the queen's alone
  have q1 := only_queen (-1) 0 (by omega) (by omega) (by omega) (by omega) (by unfold kingHits; omega)
  have q2 := only_queen 0 0 (by omega) (by omega) (by omega) (by omega) (by unfold kingHits; omega)
  have q3 := only_queen 1 0 (by omega) (by omega) (by omega) (by omega) (by unfold kingHits; omega)
  have := queen_row3 c d hd q1 q2 q3
  omega

/-- THE CORNER SHAPE: the lone king on (0, 0), the board x, y ≥ 0; our king not touching it. If the corner's box is
covered, our king stands at distance exactly two. -/
theorem corner_shape (a b c d : Int) (hc : 0 ≤ c) (hd : 0 ≤ d) (hfar : 2 ≤ a ∨ 2 ≤ b)
    (hbox : ∀ i j : Int, 0 ≤ i → i ≤ 1 → 0 ≤ j → j ≤ 1 → covered a b c d i j) : (a = 2 ∨ b = 2) ∧ a ≤ 2 ∧ b ≤ 2 := by
  have h1 := hbox 0 0 (by omega) (by omega) (by omega) (by omega)
  have h2 := hbox 1 0 (by omega) (by omega) (by omega) (by omega)
  have h3 := hbox 0 1 (by omega) (by omega) (by omega) (by omega)
  have h4 := hbox 1 1 (by omega) (by omega) (by omega) (by omega)
  unfold covered kingHits queenLine at h1 h2 h3 h4
  omega

end Ideas.QueenMate
