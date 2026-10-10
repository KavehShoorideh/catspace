import Chess

/-!
# Geometric lemmas of piece movement (discovered by catspace/incubator/analyst/geo_lemmas.py, 2026-10-09)

Stage 1 of the discovery solves, for each piece's move vectors, every linear form a*file + b*rank (mod m) that all
moves shift by the same constant; for the knight it finds (file + rank) mod 2 shifted by 1 -- the colour flip -- and
for the bishop the same form preserved. Stage 2 (Formal Concept Analysis over finite tables of configurations) derives
the consequences below as implications that hold on every row. Here they are proved for ALL squares, on ChessLean's
own geometry (`knightGeometryB`, `bishopGeometryB`, `absDiff`).

Kaveh's seed: "if a knight hops away from a square, it can never attack one of its adjacent squares -- the knight on d7
cannot hop and protect d8"; exactly: never an ORTHOGONAL neighbour (those are the other colour), while a diagonal
neighbour can be covered (d7 -> f6 attacks e8). And: "two pieces cannot be forked by a knight if they are on opposite
colours".
-/

namespace Ideas.GeoLemmas

open Chess

/-- The colour of a square: (file + rank) mod 2. -/
def colour (s : Square) : Nat := (s.1.val + s.2.val) % 2

/-- Two squares are orthogonal neighbours. -/
def orthAdj (s x : Square) : Prop := absDiff s.1 x.1 + absDiff s.2 x.2 = 1

theorem absDiff_def (a b : Fin 8) : absDiff a b = if b.val ≤ a.val then a.val - b.val else b.val - a.val := rfl

/-- STAGE 1, the knight: every knight move changes the colour of the square. -/
theorem knight_flips_colour (s t : Square) (h : knightGeometryB s t = true) : colour s ≠ colour t := by
  simp only [knightGeometryB, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
  obtain ⟨_, h⟩ := h
  simp only [colour, absDiff_def] at *
  split_ifs at h <;> omega

/-- STAGE 1, the bishop: every bishop move keeps the colour of the square. -/
theorem bishop_keeps_colour (s t : Square) (h : bishopGeometryB s t = true) : colour s = colour t := by
  simp only [bishopGeometryB, sameDiagonalB, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨_, h⟩ := h
  simp only [colour, absDiff_def] at *
  split_ifs at h <;> omega

/-- Orthogonal neighbours are of opposite colour. -/
theorem orthAdj_other_colour (s x : Square) (h : orthAdj s x) : colour s ≠ colour x := by
  simp only [orthAdj, colour, absDiff_def] at *
  split_ifs at h <;> omega

/-- STAGE 2, Kaveh's lemma: a knight that hops from `s` to `t` never attacks an orthogonal neighbour of `s` (the knight
on d7 cannot hop and still guard d8). -/
theorem knight_hop_never_covers_orth_neighbour (s t x : Square) (hm : knightGeometryB s t = true)
    (hx : orthAdj s x) : knightGeometryB t x = false := by
  cases hk : knightGeometryB t x with
  | false => rfl
  | true =>
      have h1 := knight_flips_colour s t hm
      have h2 := knight_flips_colour t x hk
      have h3 := orthAdj_other_colour s x hx
      simp only [colour] at h1 h2 h3
      omega

/-- STAGE 2: a knight that moves stops guarding EVERY square it guarded -- from `s` it attacks only the colour opposite
to `s`, from `t` only the colour of `s`. -/
theorem knight_drops_every_guard (s t x : Square) (hm : knightGeometryB s t = true)
    (hs : knightGeometryB s x = true) : knightGeometryB t x = false := by
  cases hk : knightGeometryB t x with
  | false => rfl
  | true =>
      have h1 := knight_flips_colour s t hm
      have h2 := knight_flips_colour t x hk
      have h3 := knight_flips_colour s x hs
      simp only [colour] at h1 h2 h3
      omega

/-- STAGE 2: a knight move leaves the file, the rank and both diagonals of its square -- so a knight standing on any
line through `s` always opens it (a pinned knight is frozen: no move of it keeps the line blocked). -/
theorem knight_leaves_every_line (s t : Square) (h : knightGeometryB s t = true) :
    s.1 ≠ t.1 ∧ s.2 ≠ t.2 ∧ absDiff s.1 t.1 ≠ absDiff s.2 t.2 := by
  simp only [knightGeometryB, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
  obtain ⟨_, h⟩ := h
  simp only [absDiff_def] at *
  refine ⟨?_, ?_, ?_⟩
  · intro he; rw [he] at h; split_ifs at h <;> omega
  · intro he; rw [he] at h; split_ifs at h <;> omega
  · split_ifs at h ⊢ <;> omega

/-- STAGE 2: a knight forks only two squares of the same colour -- two pieces on opposite colours cannot be forked. -/
theorem knight_fork_same_colour (f x y : Square) (hx : knightGeometryB f x = true) (hy : knightGeometryB f y = true) :
    colour x = colour y := by
  have h1 := knight_flips_colour f x hx
  have h2 := knight_flips_colour f y hy
  simp only [colour] at h1 h2 ⊢
  omega

end Ideas.GeoLemmas
