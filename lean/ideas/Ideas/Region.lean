import Ideas.QueenMate

/-!
# The region of a piece, and what keeps a goal out of reach (ANALYST_PLAN 52.17 (1), Kaveh 2026-10-19)

"A piece's REGION is where it can get to": for a piece on `s`, the connected component of `s` in the graph whose vertices
are the squares where the piece may STAND (a king: not an attacked square; a slider: an empty square or one it may take)
and whose edges are the piece's single moves. The piece's move rule makes the walls differ: a king cannot enter an attacked
square, so a rook's line is a wall for it. A goal's ZONE is the set of squares the goal needs the piece on; a goal whose
piece has a region that meets no zone square is killed -- they need only keep the wall standing.

Two kills of a mating attempt, both "do not chase":
1. THE WALL (`king_region_wall_file`, `rook_wall_file`): a king whose way crosses a file the rook holds end to end never
   crosses it -- the region of the king stays on its side.
2. THE MEASURE THEY HOLD (`evasion_step`, `no_rook_mate_in_open`): a king at distance two or more from ours with a free
   square away keeps the distance after any one king move of ours -- a king never gains on a king with room; and a king and
   a rook never mate a king in the open (the lone king's whole 3x3 box on the board), so the mate needs the room shrunk,
   which is the rook's work.

The plane has no edge here (Int coordinates, as in `Ideas.QueenMate`); the board's edge enters through the `stand`
predicate of each use.
-/

namespace Ideas.Region

open Ideas.QueenMate

/-- A king's single step from (a, b) to (x, y). -/
def kingStep (a b x y : Int) : Prop := a - x ≤ 1 ∧ x - a ≤ 1 ∧ b - y ≤ 1 ∧ y - b ≤ 1 ∧ ¬ (a = x ∧ b = y)

/-- THE REGION of the piece on (a, b): the squares it reaches by single moves `mv` over squares where it may STAND
(`stand`). The start square is in it; every step lands on a standable square. -/
inductive Region (mv : Int → Int → Int → Int → Prop) (stand : Int → Int → Prop) (a b : Int) : Int → Int → Prop
  | refl : Region mv stand a b a b
  | step {x y u v : Int} : Region mv stand a b x y → mv x y u v → stand u v → Region mv stand a b u v

/-- A smaller standing set gives a smaller region (the wall can only grow with what they hold). -/
theorem Region.mono {mv : Int → Int → Int → Int → Prop} {s s' : Int → Int → Prop} (h : ∀ x y, s x y → s' x y) {a b x y : Int}
    (hr : Region mv s a b x y) : Region mv s' a b x y := by
  induction hr with
  | refl => exact Region.refl
  | step _ hm hs ih => exact Region.step ih hm (h _ _ hs)

/-- THE WALL, a file: when no square of file `c` is standable for the king, the region of a king that starts left of the
file stays left of it (a king moves one file at a time and never stands on the file). -/
theorem king_region_wall_file (stand : Int → Int → Prop) (c a b : Int) (hwall : ∀ y, ¬ stand c y) (ha : a < c) :
    ∀ x y, Region kingStep stand a b x y → x < c := by
  intro x y h
  induction h with
  | refl => exact ha
  | @step x' y' u v _ hm hs ih =>
    unfold kingStep at hm
    have hne : u ≠ c := by rintro rfl; exact hwall v hs
    omega

/-- The same, a rank. -/
theorem king_region_wall_rank (stand : Int → Int → Prop) (c a b : Int) (hwall : ∀ x, ¬ stand x c) (hb : b < c) :
    ∀ x y, Region kingStep stand a b x y → y < c := by
  intro x y h
  induction h with
  | refl => exact hb
  | @step x' y' u v _ hm hs ih =>
    unfold kingStep at hm
    have hne : v ≠ c := by rintro rfl; exact hwall u hs
    omega

/-- Occupancy-aware rook attack on a file: the rook on (c, r) attacks (x, y) up or down its file when nothing stands
between (`occ` = occupied). -/
def rookOnFile (occ : Int → Int → Prop) (c r x y : Int) : Prop :=
  x = c ∧ ((r < y ∧ ∀ t, r < t → t < y → ¬ occ c t) ∨ (y < r ∧ ∀ t, y < t → t < r → ¬ occ c t))

/-- A rook alone on its file (nothing else stands on it) attacks every other square of the file. -/
theorem rook_attacks_empty_file (occ : Int → Int → Prop) (c r : Int) (hempty : ∀ t, t ≠ r → ¬ occ c t) :
    ∀ y, y ≠ r → rookOnFile occ c r c y := by
  intro y hy
  unfold rookOnFile
  refine ⟨rfl, ?_⟩
  rcases Int.lt_or_gt_of_ne hy with h | h
  · exact Or.inr ⟨h, fun t h1 h2 => hempty t (by omega)⟩
  · exact Or.inl ⟨h, fun t h1 h2 => hempty t (by omega)⟩

/-- THE ROOK'S LINE IS A WALL FOR THE KING (0LRTF's c1 rook, were the c7 pawn not on the file): a rook on (c, r) with its
file otherwise empty, and itself protected (the king cannot take it), bars the file to a king -- a king never stands on an
attacked square -- so the region of a king that starts left of the file stays left of it. -/
theorem rook_wall_file (occ stand : Int → Int → Prop) (c r a b : Int) (hempty : ∀ t, t ≠ r → ¬ occ c t)
    (hnotatt : ∀ x y, stand x y → ¬ rookOnFile occ c r x y) (hself : ¬ stand c r) (ha : a < c) :
    ∀ x y, Region kingStep stand a b x y → x < c := by
  refine king_region_wall_file stand c a b ?_ ha
  intro y hs
  by_cases hy : y = r
  · subst hy; exact hself hs
  · exact hnotatt c y hs (rook_attacks_empty_file occ c r hempty y hy)

/-- A rook attacks along its file or rank, another square (blockers ignored: more than it attacks, never less). -/
def rookLine (a b x y : Int) : Prop := ¬ (a = x ∧ b = y) ∧ (a = x ∨ b = y)

/-- Whatever a rook attacks a queen on its square attacks. -/
theorem rookLine_queenLine (a b x y : Int) (h : rookLine a b x y) : queenLine a b x y := by
  unfold rookLine at h
  unfold queenLine
  omega

/-- THE ROOK ALONE, IN THE OPEN: with the kings apart (always, on a board), our king and a rook never cover the whole
3x3 box of the lone king on (ka, kb) -- the lone king, off the edge, is never mated by a king and a rook. A fortiori from
`no_queen_mate_in_open` (a rook attacks a subset of what a queen attacks from the same square). Pawns of ours are not in
this statement. -/
theorem no_rook_mate_in_open (ka kb wa wb ra rb : Int)
    (hfar : 2 ≤ wa - ka ∨ 2 ≤ ka - wa ∨ 2 ≤ wb - kb ∨ 2 ≤ kb - wb)
    (hbox : ∀ i j : Int, -1 ≤ i → i ≤ 1 → -1 ≤ j → j ≤ 1 → kingHits wa wb (ka + i) (kb + j) ∨ rookLine ra rb (ka + i) (kb + j)) :
    False :=
  no_queen_mate_in_open ka kb wa wb ra rb hfar fun i j h1 h2 h3 h4 =>
    (hbox i j h1 h2 h3 h4).imp_right (rookLine_queenLine ra rb _ _)

/-- Kings are within `d` of each other (Chebyshev distance at most `d`). -/
def chebLe (d a b x y : Int) : Prop := a - x ≤ d ∧ x - a ≤ d ∧ b - y ≤ d ∧ y - b ≤ d

/-- THE MEASURE THEY HOLD (a king never gains on a king with room): our king at distance two or more from theirs makes
one king move (to (wa', wb')); then one of the four orthogonal steps of their king, the one away along the axis on which
we are nearer, leaves the distance no smaller than it was. Whether that step is free (on the board, not attacked) is the
`room` the mating side must take away. -/
theorem evasion_step (wa wb wa' wb' ka kb : Int) (hmove : kingStep wa wb wa' wb') (hfar : ¬ chebLe 1 wa wb ka kb) :
    ∃ dx dy : Int, ((dx = 0 ∧ (dy = 1 ∨ dy = -1)) ∨ (dy = 0 ∧ (dx = 1 ∨ dx = -1))) ∧
      ∀ d, chebLe d wa' wb' (ka + dx) (kb + dy) → chebLe d wa wb ka kb := by
  unfold kingStep at hmove
  unfold chebLe at hfar ⊢
  by_cases c1 : wa' ≤ ka ∧ kb - wb' ≤ ka - wa' ∧ wb' - kb ≤ ka - wa'
  · exact ⟨1, 0, Or.inr ⟨rfl, Or.inl rfl⟩, fun d h => by omega⟩
  by_cases c2 : ka ≤ wa' ∧ kb - wb' ≤ wa' - ka ∧ wb' - kb ≤ wa' - ka
  · exact ⟨-1, 0, Or.inr ⟨rfl, Or.inr rfl⟩, fun d h => by omega⟩
  by_cases c3 : wb' ≤ kb ∧ ka - wa' ≤ kb - wb' ∧ wa' - ka ≤ kb - wb'
  · exact ⟨0, 1, Or.inl ⟨rfl, Or.inl rfl⟩, fun d h => by omega⟩
  · exact ⟨0, -1, Or.inl ⟨rfl, Or.inr rfl⟩, fun d h => by omega⟩

end Ideas.Region
