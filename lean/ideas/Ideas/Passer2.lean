import Ideas.Passer

/-!
# The unstoppable passer, second form: an enemy king with pawns

The first form (`Ideas/Passer.lean`) needs a bare enemy king and fires on no real position: the enemy has
pawns. Here the enemy has a king and pawns, and our side is still without rook, bishop or queen. Two
hypotheses replace `bareKing` and `kingHasMove`:

* `pawnsFar`  -- every enemy pawn needs more than `n - 1` moves (a lower bound: the larger of the file
                 distance and the rank distance less one for a possible double step) to reach any BAD
                 square: the path and our pawn's square, the squares from which it would check our king,
                 and its own promotion rank;
* `kingRoom`  -- every square the enemy king can reach in `n - 1` moves has more neighbours free of our
                 pieces, of our attacks (the passer's whole future attack set counted) and of the path
                 than the enemy has pawns: it can never run out of king moves (no self-stalemate).

Prototyped in Python first (`lean_passer_v2_proto.py`): 816 PROVED claims, all confirmed by an exhaustive
race search. Conclusion as before: our queen on the promotion square within `2n - 1` plies whatever the
opponent does.
-/

namespace Ideas

open Chess
open Passer

namespace Passer2

/-- The enemy's double-step rank. -/
def startRank : Color → Nat
  | .white => 1
  | .black => 6

/-- The enemy's own back rank: a pawn never stands there. -/
def backRank : Color → Nat
  | .white => 0
  | .black => 7

/-- The ranks of `s` and `t` in the pawn's direction of travel: `t` is ahead when `lo ≤ hi`. -/
def ranksAlong (e : Color) (s t : Square) : Nat × Nat :=
  match e with
  | .white => (s.2.val, t.2.val)
  | .black => (t.2.val, s.2.val)

/-- A lower bound on the moves an enemy pawn of colour `e` on `s` needs to reach `t`: none when `t` is
not ahead of it or too far aside; `some 0` on its own square. -/
def pdist (e : Color) (s t : Square) : Option Nat :=
  let lo := (ranksAlong e s t).1
  let hi := (ranksAlong e s t).2
  if lo ≤ hi ∧ absDiff s.1 t.1 ≤ hi - lo then
    some (max (absDiff s.1 t.1) (hi - lo - (if s.2.val = startRank e then 1 else 0)))
  else none

/-- More than `k` moves away (or unreachable). -/
def farB (e : Color) (s t : Square) (k : Nat) : Bool :=
  match pdist e s t with
  | none => true
  | some d => decide (k < d)

/-- The squares an enemy pawn must not reach during the race: the path, the squares from which it would
check our king on `ok`, and its promotion rank (our pawn's own square is not needed: we push first). -/
def badSquares (c : Color) (x ok : Square) : List Square :=
  path c x ++ allSquares.filter (fun s => pawnAttackGeometryB (Color.opposite c) s ok) ++
    allFiles.map (fun f => (f, promotionRank (Color.opposite c)))

/-- Squares our passer attacks now or later on its way: the adjacent files, strictly ahead. -/
def futureAttackB (c : Color) (x d : Square) : Bool :=
  (absDiff d.1 x.1 == 1) &&
    (match c with
      | .white => decide (x.2.val < d.2.val)
      | .black => decide (d.2.val < x.2.val))

/-- The squares the enemy king may never use: ours, attacked by us (the passer's future attacks counted),
the path, our pawn's square. Fixed at the start of the race. -/
def forbiddenB (p : Position) (x : Square) (d : Square) : Bool :=
  (match p.board d with
    | some pc => pc.color == p.turn
    | none => false) ||
  isAttackedB' p p.turn d || futureAttackB p.turn x d || (path p.turn x).contains d || d == x

/-- The neighbours of `x'` the enemy king may use. -/
def roomB (F : Square → Bool) (x' : Square) : Nat :=
  (allSquares.filter (fun d => kingGeometryB x' d && !F d)).length

def enemyPawnCount (p : Position) : Nat :=
  (allSquares.filter (fun s => p.board s == some ⟨Color.opposite p.turn, .pawn⟩)).length

/-! ## Hypotheses -/

/-- Every enemy piece is its king or a pawn (no pawn on its own back rank), and there is exactly one king. -/
def enemyKingPawnsB (p : Position) (_ : PasserParams) : Bool :=
  let e := Color.opposite p.turn
  allSquares.all (fun s => match p.board s with
    | some pc => (pc.color == p.turn) || (pc.kind == .king) ||
        ((pc.kind == .pawn) && (s.2.val != backRank e))
    | none => true) &&
  match kingSquare p.board e with
  | some k => allSquares.all fun s => (p.board s == some ⟨e, .king⟩) == decide (s = k)
  | none => false

def pawnsFarB (p : Position) (ps : PasserParams) : Bool :=
  match kingSquare p.board p.turn with
  | none => false
  | some ok =>
    let e := Color.opposite p.turn
    let n := pushes p.turn ps.sq
    allSquares.all fun s =>
      (p.board s != some ⟨e, .pawn⟩) ||
        (badSquares p.turn ps.sq ok).all (fun t => farB e s t (n - 1))

/-- Squares the enemy king can never stand on during the race: next to our king, or within one of the
promotion square (it is always more than one push away from it). -/
def neverKingB (c : Color) (x ok x' : Square) : Bool :=
  kingGeometryB ok x' || decide (kdist x' (promoSquare c x) ≤ 1)

def kingRoomB (p : Position) (ps : PasserParams) : Bool :=
  match kingSquare p.board (Color.opposite p.turn), kingSquare p.board p.turn with
  | some ek, some ok =>
    let n := pushes p.turn ps.sq
    let F := forbiddenB p ps.sq
    let np := enemyPawnCount p
    allSquares.all fun x' =>
      decide (n - 1 < kdist x' ek) || neverKingB p.turn ps.sq ok x' || decide (np < roomB F x')
  | _, _ => false

end Passer2

open Passer2 in
def unstoppablePasser2 : Idea PasserParams where
  id := "unstoppablePasser2"
  hyps :=
    [ ⟨"ownPawn", ownPawnB⟩, ⟨"pathClear", pathClearB⟩, ⟨"enemyKingPawns", enemyKingPawnsB⟩,
      ⟨"noSliders", noSlidersB⟩, ⟨"notInCheck", notInCheckB⟩, ⟨"kingOutside", kingOutsideB⟩,
      ⟨"pawnsFar", pawnsFarB⟩, ⟨"kingRoom", kingRoomB⟩ ]
  protagonist := fun ps => ps.sq
  tier := .unconditional
  horizon := 13

/-- Where to look when one hypothesis fails. -/
def Passer2.targets (p : Position) (ps : PasserParams) (hyp : String) : List Square :=
  if hyp = "pathClear" then (path p.turn ps.sq).filter (fun t => (p.board t).isSome)
  else if hyp = "kingOutside" then
    match kingSquare p.board (Color.opposite p.turn) with | some ek => [ek] | none => []
  else if hyp = "pawnsFar" then
    match kingSquare p.board p.turn with
    | none => []
    | some ok =>
      let e := Color.opposite p.turn
      let n := pushes p.turn ps.sq
      allSquares.filter fun s => (p.board s == some ⟨e, .pawn⟩) &&
        !(badSquares p.turn ps.sq ok).all (fun t => farB e s t (n - 1))
  else if hyp = "enemyKingPawns" then
    allSquares.filter (fun s => match p.board s with
      | some pc => pc.color != p.turn && pc.kind != .king && pc.kind != .pawn
      | none => false)
  else []

/-! ## The invariant of the race, second form -/

namespace Passer2

/-- Our side `c`, pawn on `x` with `r` pushes to go, the enemy king on `ek`, our king on `ok`, the forbidden
set `F` and the enemy's pawn count `np` fixed at the start, at our turn. -/
structure Inv2 (c : Color) (p : Position) (x : Square) (r : Nat) (ek ok : Square) (F : Square → Bool)
    (np : Nat) : Prop where
  turn : p.turn = c
  pawn : p.board x = some ⟨c, .pawn⟩
  notFirst : x.2 ≠ promotionRank (Color.opposite c)
  pushesEq : pushes c x = r
  pos : 1 ≤ r
  clear : ∀ t ∈ path c x, p.board t = none
  /-- every enemy piece is the king on `ek` or a pawn -/
  enemy : ∀ s pc, p.board s = some pc → pc.color = Color.opposite c →
    (s = ek ∧ pc.kind = .king) ∨ pc.kind = .pawn
  eking : p.board ek = some ⟨Color.opposite c, .king⟩
  /-- no enemy pawn on its own back rank (there its double step would shorten the distance by two) -/
  pawnsOffBack : ∀ s, p.board s = some ⟨Color.opposite c, .pawn⟩ → s.2.val ≠ backRank (Color.opposite c)
  outside : r < kdist ek (promoSquare c x)
  noSliders : ∀ s pc, p.board s = some pc → pc.color = c →
    pc.kind = .king ∨ pc.kind = .pawn ∨ pc.kind = .knight
  ourKing : kingSquare p.board c = some ok
  usSafe : ¬ inCheck p c
  themSafe : ¬ inCheck p (Color.opposite c)
  /-- enemy pawns are more than `r - 1` moves from every bad square -/
  pawnsFar : ∀ s, p.board s = some ⟨Color.opposite c, .pawn⟩ →
    ∀ t ∈ badSquares c x ok, farB (Color.opposite c) s t (r - 1) = true
  /-- our pieces, our attacks, the path and our pawn's square stay inside `F` -/
  oursInF : ∀ d pc, p.board d = some pc → pc.color = c → F d = true
  attacksInF : ∀ d, isAttackedB p c d = true → F d = true
  futureInF : ∀ d, futureAttackB c x d = true → F d = true
  pathInF : ∀ t ∈ path c x, F t = true
  /-- the enemy's pawn count is `np` -/
  count : (allSquares.filter (fun s => p.board s == some ⟨Color.opposite c, .pawn⟩)).length = np
  /-- every square the enemy king can still reach and stand on has room -/
  room : ∀ x', kdist x' ek ≤ r - 1 → neverKingB c x ok x' = true ∨ np < roomB F x'

/-! ### Facts about the bad squares and the distance -/

theorem pdist_self (e : Color) (s : Square) : pdist e s s = some 0 := by
  cases e <;> simp [pdist, ranksAlong, absDiff]

theorem farB_self (e : Color) (s : Square) (k : Nat) : farB e s s k = false := by
  simp [farB, pdist_self]

theorem mem_allFiles (f : Fin 8) : f ∈ allFiles := by fin_cases f <;> decide

theorem mem_badSquares {c : Color} {x ok t : Square} :
    t ∈ badSquares c x ok ↔ t ∈ path c x ∨ pawnAttackGeometryB (Color.opposite c) t ok = true ∨
      t.2 = promotionRank (Color.opposite c) := by
  simp only [badSquares, List.mem_append, List.mem_filter, mem_allSquares, true_and, List.mem_map]
  constructor
  · rintro ((h | h) | ⟨f, -, h⟩)
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr (by rw [← h]))
  · rintro (h | h | h)
    · exact Or.inl (Or.inl h)
    · exact Or.inl (Or.inr h)
    · refine Or.inr ⟨t.1, mem_allFiles t.1, ?_⟩
      obtain ⟨tf, tr⟩ := t
      simp only at h
      rw [h]

/-- An enemy pawn standing on a bad square is not far from it: it cannot stand there. -/
theorem enemy_pawn_not_on_bad {c : Color} {p : Position} {x : Square} {r : Nat} {ek ok : Square}
    {F : Square → Bool} {np : Nat} (h : Inv2 c p x r ek ok F np) {t : Square}
    (ht : t ∈ badSquares c x ok) : p.board t ≠ some ⟨Color.opposite c, .pawn⟩ := by
  intro hp
  have := h.pawnsFar t hp t ht
  rw [farB_self] at this
  cases this

/-- The enemy's pieces attack by geometry alone (king and pawns). -/
theorem enemy_noslider {c : Color} {p : Position} {x : Square} {r : Nat} {ek ok : Square}
    {F : Square → Bool} {np : Nat} (h : Inv2 c p x r ek ok F np) :
    ∀ s pc, p.board s = some pc → pc.color = Color.opposite c →
      pc.kind = .king ∨ pc.kind = .pawn ∨ pc.kind = .knight := by
  intro s pc hs hcol
  rcases h.enemy s pc hs hcol with ⟨_, hk⟩ | hk
  · exact Or.inl hk
  · exact Or.inr (Or.inl hk)

theorem enemy_kings_only {c : Color} {p : Position} {x : Square} {r : Nat} {ek ok : Square}
    {F : Square → Bool} {np : Nat} (h : Inv2 c p x r ek ok F np) :
    ∀ s, p.board s = some ⟨Color.opposite c, .king⟩ ↔ s = ek := by
  intro s
  constructor
  · intro hs
    rcases h.enemy s _ hs rfl with ⟨he, _⟩ | hk
    · exact he
    · cases hk
  · rintro rfl; exact h.eking

/-! ### The push -/

/-- What our push does to the board (any number of pushes): the pawn steps to `y`, as a pawn or a queen. -/
theorem push_board2 {c : Color} {p : Position} {x : Square} {r : Nat} {ek ok : Square} {F : Square → Bool}
    {np : Nat} (h : Inv2 c p x r ek ok F np) {m : Move} (hm : push c x = some m) :
    ∃ y v, ahead c x 1 = some y ∧ p.board y = none ∧ v.color = c ∧ (v.kind = .pawn ∨ v.kind = .queen) ∧
      (applyMove p m).board = boardUpdate (boardUpdate p.board x none) y (some v) := by
  have h1 : 1 ≤ pushes c x := h.pushesEq ▸ h.pos
  obtain ⟨y, hy, -, -, -⟩ := ahead_one h1
  have hy_empty : p.board y = none := h.clear y (ahead_mem_path hy)
  rw [push_eq hy] at hm
  split_ifs at hm with hpr
  · rw [Option.some.injEq] at hm; subst hm
    refine ⟨y, ⟨c, .queen⟩, hy, hy_empty, rfl, Or.inr rfl, ?_⟩
    rw [← h.turn]; rfl
  · rw [Option.some.injEq] at hm; subst hm
    refine ⟨y, ⟨c, .pawn⟩, hy, hy_empty, rfl, Or.inl rfl, ?_⟩
    simp [applyMove, applyBoard, boardMove, h.pawn]

theorem push_legal2 {c : Color} {p : Position} {x : Square} {r : Nat} {ek ok : Square} {F : Square → Bool}
    {np : Nat} (h : Inv2 c p x r ek ok F np) : ∃ m, push c x = some m ∧ legalMove p m := by
  have h1 : 1 ≤ pushes c x := h.pushesEq ▸ h.pos
  obtain ⟨y, hy, hyf, hyw, hyb⟩ := ahead_one h1
  have hy_empty : p.board y = none := h.clear y (ahead_mem_path hy)
  have hgeo := pawnMoveGeometryB_ahead hy
  have hpawn := h.pawn
  have hne : ∀ d : Color, d ≠ Color.opposite d := by intro d; cases d <;> decide
  -- the board after the pawn moves to `y` carrying `v`
  have kings_same : ∀ v : Piece, v.kind ≠ .king →
      ∀ s, p.board s = some ⟨c, .king⟩ ↔
        boardUpdate (boardUpdate p.board x none) y (some v) s = some ⟨c, .king⟩ := by
    intro v hv s
    simp only [boardUpdate, Function.update_apply]
    by_cases hsy : s = y
    · rw [if_pos hsy, hsy, hy_empty]
      constructor
      · intro hh; cases hh
      · intro hh; cases hh; exact absurd rfl hv
    · rw [if_neg hsy]
      by_cases hsx : s = x
      · rw [if_pos hsx, hsx, hpawn]
        constructor <;> intro hh <;> simp at hh
      · rw [if_neg hsx]
  -- the enemy's pieces are where they were
  have enemy_same : ∀ v : Piece, v.color = c → ∀ s pc, pc.color = Color.opposite c →
      (p.board s = some pc ↔ boardUpdate (boardUpdate p.board x none) y (some v) s = some pc) := by
    intro v hv s pc hcol
    simp only [boardUpdate, Function.update_apply]
    by_cases hsy : s = y
    · rw [if_pos hsy, hsy, hy_empty]
      constructor
      · intro hh; cases hh
      · intro hh; cases hh; rw [hv] at hcol; exact absurd hcol (hne c)
    · rw [if_neg hsy]
      by_cases hsx : s = x
      · rw [if_pos hsx, hsx, hpawn]
        constructor
        · intro hh; cases hh; exact absurd hcol (hne c)
        · intro hh; cases hh
      · rw [if_neg hsx]
  -- our king is not attacked after the move: the enemy's pieces did not move and attack by geometry
  have safe : ∀ v : Piece, v.color = c → v.kind ≠ .king →
      inCheckB { p with board := boardUpdate (boardUpdate p.board x none) y (some v) } c = false := by
    intro v hv hvk
    have hus := h.usSafe
    unfold inCheck inCheckB at hus
    unfold inCheckB
    simp only
    rw [← kingSquare_congr (kings_same v hvk)]
    rw [h.ourKing] at hus ⊢
    simp only [Bool.not_eq_true] at hus ⊢
    rw [← noslider_attack (enemy_noslider h) (enemy_same v hv)]
    exact hus
  refine ⟨_, push_eq hy, ?_⟩
  rw [legalMove_iff_legalMoveB]
  split_ifs with hpr
  · have hps : pseudoLegalB p (.promotion x y .queen) = true := by
      simp [pseudoLegalB, pseudoLegalPromotionB, hpawn, hpr, h.turn, hgeo, hy_empty]
    have hboard : (applyMove p (.promotion x y .queen)).board =
        boardUpdate (boardUpdate p.board x none) y (some ⟨c, .queen⟩) := by
      rw [← h.turn]; rfl
    have hnc : inCheckB (applyMove p (.promotion x y .queen)) p.turn = false := by
      rw [h.turn]
      have := safe ⟨c, .queen⟩ rfl (by simp)
      rw [← hboard] at this
      exact this
    simp [legalMoveB, Move.isCastle, hps, hnc]
  · have hps : pseudoLegalB p (.normal x y) = true := by
      simp [pseudoLegalB, normalPseudoLegalB, hpawn, h.turn, destinationFriendlyB, hy_empty,
        normalPieceGeometryB, hgeo, hpr]
    have hboard : (applyMove p (.normal x y)).board =
        boardUpdate (boardUpdate p.board x none) y (some ⟨c, .pawn⟩) := by
      simp [applyMove, applyBoard, boardMove, hpawn]
    have hnc : inCheckB (applyMove p (.normal x y)) p.turn = false := by
      rw [h.turn]
      have := safe ⟨c, .pawn⟩ rfl (by simp)
      rw [← hboard] at this
      exact this
    simp [legalMoveB, Move.isCastle, hps, hnc]

theorem promoted_after_last2 {c : Color} {p : Position} {x : Square} {ek ok : Square} {F : Square → Bool}
    {np : Nat} (h : Inv2 c p x 1 ek ok F np) {m : Move} (hm : push c x = some m) :
    promoted c x.1 (applyMove p m) = true := by
  have h1 : 1 ≤ pushes c x := h.pushesEq ▸ h.pos
  obtain ⟨y, hy, hyf, -, -⟩ := ahead_one h1
  have hpr : y.2 = promotionRank c := ahead_promo_of_one h.pushesEq hy
  rw [push_eq hy, if_pos hpr, Option.some.injEq] at hm
  subst hm
  have hyeq : (x.1, promotionRank c) = y := by
    obtain ⟨yf, yr⟩ := y
    simp only at hyf hpr
    rw [hyf, hpr]
  simp [promoted, hyeq, applyMove, applyBoard, boardUpdate, Function.update_apply, h.turn,
    PromotionKind.toPieceKind]

/-- With two or more pushes to go, our push is the plain step to `y`. -/
theorem push_board_two2 {c : Color} {p : Position} {x : Square} {r : Nat} {ek ok : Square} {F : Square → Bool}
    {np : Nat} (h : Inv2 c p x (r + 2) ek ok F np) {m : Move} (hm : push c x = some m) :
    ∃ y, ahead c x 1 = some y ∧ m = .normal x y ∧ p.board y = none ∧
      (applyMove p m).board = boardUpdate (boardUpdate p.board x none) y (some ⟨c, .pawn⟩) := by
  have h1 : 1 ≤ pushes c x := h.pushesEq ▸ (by omega)
  obtain ⟨y, hy, -, -, -⟩ := ahead_one h1
  have hnp : y.2 ≠ promotionRank c := ahead_not_promo_of_two (h.pushesEq ▸ (by omega)) hy
  rw [push_eq hy, if_neg hnp, Option.some.injEq] at hm
  subst hm
  refine ⟨y, hy, rfl, h.clear y (ahead_mem_path hy), ?_⟩
  simp [applyMove, applyBoard, boardMove, h.pawn]

/-- A pawn standing ahead of `x` on its file attacks only squares of the passer's future attack set. -/
theorem future_of_pawn_attack {c : Color} {x z d : Square} (hz : z ∈ path c x)
    (hd : pawnAttackGeometryB c z d = true) : futureAttackB c x d = true := by
  obtain ⟨hf, hw, hb⟩ := mem_path_iff.mp hz
  have hx := x.2.isLt
  have hz2 := z.2.isLt
  have hd2 := d.2.isLt
  rw [Fin.ext_iff] at hf
  cases c
  · have := hw rfl
    simp only [pawnAttackGeometryB, absDiff, Bool.and_eq_true, beq_iff_eq, if_true] at hd
    simp only [futureAttackB, absDiff, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
    split_ifs at hd ⊢ <;> omega
  · have := hb rfl
    simp only [pawnAttackGeometryB, absDiff, Bool.and_eq_true, beq_iff_eq, reduceCtorEq, if_false] at hd
    simp only [futureAttackB, absDiff, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
    split_ifs at hd ⊢ <;> omega

/-- The pigeonhole: more squares than enemy pawns leaves a square without one. -/
theorem exists_not_pawn {p : Position} {e : Color} {G : Square → Bool} {np : Nat}
    (hcount : (allSquares.filter (fun s => p.board s == some ⟨e, .pawn⟩)).length = np)
    (hL : np < (allSquares.filter G).length) : ∃ d, G d = true ∧ p.board d ≠ some ⟨e, .pawn⟩ := by
  by_contra hall
  push_neg at hall
  have hfilt : (allSquares.filter G).filter (fun s => p.board s == some ⟨e, .pawn⟩) = allSquares.filter G :=
    List.filter_eq_self.mpr (fun d hd => by
      have := hall d (List.mem_filter.mp hd).2
      simpa using this)
  have hsub : ((allSquares.filter G).filter (fun s => p.board s == some ⟨e, .pawn⟩)).length ≤
      (allSquares.filter (fun s => p.board s == some ⟨e, .pawn⟩)).length :=
    List.Sublist.length_le (List.Sublist.filter _ List.filter_sublist)
  rw [hfilt, hcount] at hsub
  omega

theorem enemy_has_move2 {c : Color} {p : Position} {x : Square} {r : Nat} {ek ok : Square}
    {F : Square → Bool} {np : Nat} (h : Inv2 c p x (r + 2) ek ok F np) {m : Move}
    (hm : push c x = some m) : (legalMoves (applyMove p m)) ≠ [] := by
  obtain ⟨y, hy, rfl, hy_empty, hboard⟩ := push_board_two2 h hm
  set q := applyMove p (.normal x y) with hq
  have hqturn : q.turn = Color.opposite c := by rw [hq, applyMove_turn, h.turn]
  have hne : ∀ d : Color, d ≠ Color.opposite d := by intro d; cases d <;> decide
  have hpawn := h.pawn
  have hekx : ek ≠ x := by intro e; have := h.eking; rw [e, hpawn] at this; simp at this
  have heky : ek ≠ y := by intro e; have := h.eking; rw [e, hy_empty] at this; cases this
  have hqother : ∀ s, s ≠ y → s ≠ x → q.board s = p.board s := by
    intro s hsy hsx; rw [hboard]; simp [boardUpdate, Function.update_apply, hsy, hsx]
  have hqy : q.board y = some ⟨c, .pawn⟩ := by rw [hboard]; simp [boardUpdate]
  have hxy : x ≠ y := by intro e; rw [← e, hpawn] at hy_empty; cases hy_empty
  have hqx : q.board x = none := by rw [hboard]; simp [boardUpdate, Function.update_apply, hxy]
  have hek_q : q.board ek = some ⟨Color.opposite c, .king⟩ := by rw [hqother ek heky hekx]; exact h.eking
  -- the room at the enemy king's own square
  have hroom := h.room ek (by simp [kdist, absDiff])
  have hnever : neverKingB c x ok ek = false := by
    simp only [neverKingB, Bool.or_eq_false_iff, decide_eq_false_iff_not, not_le]
    constructor
    · -- the kings are not adjacent: our king is not attacked
      have hus := h.usSafe
      unfold inCheck inCheckB at hus
      rw [h.ourKing] at hus
      simp only [isAttackedB, List.any_eq_true, not_exists, not_and, Bool.not_eq_true] at hus
      have := hus ek (mem_allSquares ek)
      rw [king_attack h.eking, kingGeometryB_comm] at this
      exact this
    · have := h.outside; omega
  rw [hnever] at hroom
  simp only [Bool.false_eq_true, false_or] at hroom
  obtain ⟨d, hGd, hnotpawn⟩ := exists_not_pawn h.count hroom
  simp only [Bool.and_eq_true, Bool.not_eq_true'] at hGd
  obtain ⟨hgeo, hFd⟩ := hGd
  have hdek : d ≠ ek := by intro e; rw [e] at hgeo; simp [kingGeometryB] at hgeo
  -- d is empty in q
  have hd : q.board d = none := by
    cases hqd : q.board d with
    | none => rfl
    | some pc =>
      exfalso
      by_cases hcol : pc.color = c
      · -- one of ours: inside F
        by_cases hdy : d = y
        · rw [hdy] at hFd; exact absurd (h.pathInF y (ahead_mem_path hy)) (by rw [hFd]; decide)
        · by_cases hdx : d = x
          · subst hdx; rw [hqx] at hqd; cases hqd
          · rw [hqother d hdy hdx] at hqd
            exact absurd (h.oursInF d pc hqd hcol) (by rw [hFd]; decide)
      · have hcol' : pc.color = Color.opposite c := by cases c <;> cases hc : pc.color <;> simp_all [Color.opposite]
        have hdy : d ≠ y := by intro e; rw [e, hqy] at hqd; cases hqd; exact hcol rfl
        have hdx : d ≠ x := by intro e; rw [e, hqx] at hqd; cases hqd
        rw [hqother d hdy hdx] at hqd
        rcases h.enemy d pc hqd hcol' with ⟨hde, _⟩ | hk
        · exact hdek hde
        · apply hnotpawn
          rw [hqd]
          obtain ⟨col, kind⟩ := pc
          simp only at hcol' hk
          rw [hcol', hk]
  -- the king step ek -> d is legal in q
  have hleg : legalMoveB q (.normal ek d) = true := by
    have hps : pseudoLegalB q (.normal ek d) = true := by
      simp [pseudoLegalB, normalPseudoLegalB, hek_q, hqturn, destinationFriendlyB, hd,
        normalPieceGeometryB, hgeo]
    have hnc : inCheckB (applyMove q (.normal ek d)) q.turn = false := by
      have hboard' : (applyMove q (.normal ek d)).board = boardMove q.board ek d := rfl
      have honly : ∀ s, (applyMove q (.normal ek d)).board s = some ⟨q.turn, .king⟩ ↔ s = d := by
        intro s
        rw [hboard', hqturn]
        simp only [boardMove, boardUpdate, Function.update_apply]
        by_cases hsd : s = d
        · subst hsd; simp [hek_q]
        · by_cases hse : s = ek
          · subst hse; simp [hsd]
          · simp only [hsd, hse, if_false, iff_false]
            intro hs
            have hs' : p.board s = some ⟨Color.opposite c, .king⟩ := by
              by_cases hsy : s = y
              · rw [hsy, hqy] at hs; simp at hs
              · by_cases hsx : s = x
                · subst hsx; rw [hqx] at hs; cases hs
                · rw [← hqother s hsy hsx]; exact hs
            exact hse ((enemy_kings_only h s).1 hs')
      unfold inCheckB
      rw [kingSquare_of_only honly, hqturn, Color.opposite_opposite]
      show isAttackedB (applyMove q (.normal ek d)) c d = false
      -- our pieces are the same before and after the king's step; none attacks d
      have hns : ∀ s pc, q.board s = some pc → pc.color = c →
          pc.kind = .king ∨ pc.kind = .pawn ∨ pc.kind = .knight := by
        intro s pc hs hcol
        by_cases hsy : s = y
        · subst hsy; rw [hqy] at hs; cases hs; simp
        · by_cases hsx : s = x
          · subst hsx; rw [hqx] at hs; cases hs
          · rw [hqother s hsy hsx] at hs; exact h.noSliders s pc hs hcol
      have hsame : ∀ s pc, pc.color = c →
          (q.board s = some pc ↔ (applyMove q (.normal ek d)).board s = some pc) := by
        intro s pc hcol
        rw [hboard']
        simp only [boardMove, boardUpdate, Function.update_apply]
        by_cases hsd : s = d
        · subst hsd
          rw [if_pos rfl, hd, hek_q]
          constructor
          · intro hs; cases hs
          · intro hs; cases hs; exact absurd hcol (by cases c <;> decide)
        · rw [if_neg hsd]
          by_cases hse : s = ek
          · subst hse
            rw [if_pos rfl, hek_q]
            constructor
            · intro hs; cases hs; exact absurd hcol (by cases c <;> decide)
            · intro hs; cases hs
          · rw [if_neg hse]
      rw [← noslider_attack hns hsame]
      -- not attacked in q: a static attacker would put d in F, the pawn's attack is a future attack
      apply Bool.eq_false_iff.mpr
      intro hatt
      simp only [isAttackedB, List.any_eq_true] at hatt
      obtain ⟨s, -, hs⟩ := hatt
      obtain ⟨pc, hpc, hcol⟩ := pieceAttacksB_source hs
      by_cases hsy : s = y
      · subst hsy
        rw [pawn_attack hqy] at hs
        have := h.futureInF d (future_of_pawn_attack (ahead_mem_path hy) hs)
        rw [hFd] at this; cases this
      · by_cases hsx : s = x
        · subst hsx; rw [hqx] at hpc; cases hpc
        · rw [hqother s hsy hsx] at hpc
          have hkind := h.noSliders s pc hpc hcol
          have hatt_p : pieceAttacksB p c s d = true := by
            rw [noslider_pieceAttacks hkind hpc (by rw [hqother s hsy hsx]; exact hpc)]
            exact hs
          have : isAttackedB p c d = true := by
            simp only [isAttackedB, List.any_eq_true]; exact ⟨s, mem_allSquares s, hatt_p⟩
          have := h.attacksInF d this
          rw [hFd] at this; cases this
    simp [legalMoveB, Move.isCastle, hps, hnc]
  intro hnil
  have := legalMoves_complete ((legalMove_iff_legalMoveB q _).2 hleg)
  rw [hnil] at this
  exact absurd this (List.not_mem_nil)

/-! ### Pawn moves and the distance bound -/

theorem farB_mono {e : Color} {s t : Square} {k k' : Nat} (hk : k' ≤ k) (h : farB e s t k = true) :
    farB e s t k' = true := by
  unfold farB at h ⊢
  revert h
  cases pdist e s t with
  | none => intro; rfl
  | some d => intro h; simp only [decide_eq_true_eq] at h ⊢; omega

/-- One pawn move brings an enemy pawn at most one move nearer any square. -/
theorem pdist_step {e : Color} {s t t' : Square} (hback : s.2.val ≠ backRank e)
    (hmove : pawnMoveGeometryB e s t = true ∨ pawnAttackGeometryB e s t = true ∨
      pawnDoubleGeometryB e s t = true) {k : Nat} (hfar : farB e s t' (k + 1) = true) :
    farB e t t' k = true := by
  have hs1 := s.1.isLt; have hs2 := s.2.isLt; have ht1 := t.1.isLt; have ht2 := t.2.isLt
  have ht'1 := t'.1.isLt; have ht'2 := t'.2.isLt
  unfold farB pdist ranksAlong at hfar ⊢
  -- the move as equations on the coordinates
  have hcoord : (t.1.val = s.1.val ∨ t.1.val + 1 = s.1.val ∨ s.1.val + 1 = t.1.val) ∧
      ((e = .white → t.2.val = s.2.val + 1 ∨ (t.1.val = s.1.val ∧ s.2.val = 1 ∧ t.2.val = s.2.val + 2)) ∧
       (e = .black → s.2.val = t.2.val + 1 ∨ (t.1.val = s.1.val ∧ s.2.val = 6 ∧ s.2.val = t.2.val + 2))) := by
    cases e <;> rcases hmove with hm | hm | hm <;>
      simp only [pawnMoveGeometryB, pawnAttackGeometryB, pawnDoubleGeometryB, absDiff, Bool.and_eq_true,
        beq_iff_eq, if_true, reduceCtorEq, if_false, Fin.ext_iff] at hm <;>
      (try split_ifs at hm) <;> refine ⟨?_, fun hc => ?_, fun hc => ?_⟩ <;>
      first | omega | exact absurd hc (by decide)
  clear hmove
  obtain ⟨hfile, hw, hb⟩ := hcoord
  cases e <;> simp only [true_implies, false_implies, reduceCtorEq, startRank, backRank, absDiff] at hw hb hfar hback ⊢ <;>
    [have hrank := hw; have hrank := hb] <;>
    split_ifs at hfar ⊢ <;> simp only [decide_eq_true_eq, Option.some.injEq] at hfar ⊢ <;> omega

/-- Our single push never sets an en-passant square. -/
theorem push_enPassant_none {c : Color} {p : Position} {x y : Square} (hturn : p.turn = c)
    (hy : ahead c x 1 = some y) : (applyMove p (.normal x y)).enPassant = none := by
  obtain ⟨hf, hw, hb⟩ := ahead_one_vals hy
  simp only [applyMove, newEnPassant, pawnDoubleTarget]
  split_ifs with hc <;> try rfl
  all_goals
    exfalso
    obtain ⟨-, -, (⟨hwt, h1, h3⟩ | ⟨hbt, h6, h4⟩)⟩ := hc
    · rw [hturn] at hwt; have := hw hwt; omega
    · rw [hturn] at hbt; have := hb hbt; omega

/-- The passer's future attack set shrinks as it advances. -/
theorem future_mono {c : Color} {x y d : Square} (hy : ahead c x 1 = some y)
    (hd : futureAttackB c y d = true) : futureAttackB c x d = true := by
  obtain ⟨hf, hw, hb⟩ := ahead_one_vals hy
  rw [Fin.ext_iff] at hf
  cases c <;> simp only [futureAttackB, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, absDiff] at hd ⊢ <;>
    [have := hw rfl; have := hb rfl] <;> split_ifs at hd ⊢ <;> omega

/-- The bad squares of the pawn one step ahead are bad squares of the pawn behind. -/
theorem bad_mono {c : Color} {x y ok t : Square} (hy : ahead c x 1 = some y) (ht : t ∈ badSquares c y ok) :
    t ∈ badSquares c x ok := by
  rw [mem_badSquares] at ht ⊢
  rcases ht with h | h | h
  · exact Or.inl (path_of_ahead hy h).1
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr h)

/-! ### The enemy's moves -/

/-- In a position where the side to move has only its king on `ek` and pawns, none of them one step
from promotion, with no en-passant square, a legal move is a king step or a pawn's normal move. -/
theorem enemy_move2 {q : Position} {m : Move} {ek : Square}
    (henemy : ∀ s pc, q.board s = some pc → pc.color = q.turn → (s = ek ∧ pc.kind = .king) ∨ pc.kind = .pawn)
    (hek : q.board ek = some ⟨q.turn, .king⟩)
    (hep : q.enPassant = none)
    (hnopromo : ∀ s, q.board s = some ⟨q.turn, .pawn⟩ → ∀ t, t.2 = promotionRank q.turn →
      ¬ (pawnMoveGeometryB q.turn s t = true ∨ pawnAttackGeometryB q.turn s t = true))
    (hl : legalMove q m) :
    (∃ t, m = .normal ek t ∧ kingGeometryB ek t = true ∧ destinationFriendlyB q q.turn t = true ∧
        ¬ inCheck (applyMove q m) q.turn) ∨
    (∃ s t, m = .normal s t ∧ q.board s = some ⟨q.turn, .pawn⟩ ∧
        (pawnMoveGeometryB q.turn s t = true ∨ pawnAttackGeometryB q.turn s t = true ∨
          pawnDoubleGeometryB q.turn s t = true) ∧
        t.2 ≠ promotionRank q.turn ∧ destinationFriendlyB q q.turn t = true ∧
        ¬ inCheck (applyMove q m) q.turn) := by
  obtain ⟨hps, hnc, _⟩ := hl
  cases m with
  | normal s t =>
      simp only [pseudoLegal, pseudoLegalB] at hps
      unfold normalPseudoLegalB at hps
      cases hs : q.board s with
      | none => simp [hs] at hps
      | some pc =>
          rw [hs] at hps
          simp only [Bool.and_eq_true, beq_iff_eq] at hps
          obtain ⟨⟨⟨hcol, hfr⟩, hpr⟩, hgeo⟩ := hps
          rcases henemy s pc hs hcol with ⟨rfl, hkind⟩ | hkind
          · left
            refine ⟨t, rfl, ?_, hfr, hnc⟩
            unfold normalPieceGeometryB at hgeo
            simp only [hkind] at hgeo
            exact hgeo
          · right
            have hpc : pc = ⟨q.turn, .pawn⟩ := by
              obtain ⟨col, kind⟩ := pc; simp only at hcol hkind; rw [hcol, hkind]
            subst hpc
            refine ⟨s, t, rfl, hs, ?_, ?_, hfr, hnc⟩
            · unfold normalPieceGeometryB at hgeo
              simp only [Bool.or_eq_true, Bool.and_eq_true] at hgeo
              rcases hgeo with (⟨h1, _⟩ | ⟨h1, _⟩) | ⟨⟨h1, _⟩, _⟩
              · exact Or.inl h1
              · exact Or.inr (Or.inl h1)
              · exact Or.inr (Or.inr h1)
            · simpa using hpr
  | promotion s t k =>
      exfalso
      simp only [pseudoLegal, pseudoLegalB] at hps
      unfold pseudoLegalPromotionB at hps
      cases hs : q.board s with
      | none => simp [hs] at hps
      | some pc =>
          simp only [hs, Bool.and_eq_true, beq_iff_eq] at hps
          have hcol : pc.color = q.turn := by tauto
          have hkind : pc.kind = .pawn := by tauto
          have hpr : t.2 = promotionRank q.turn := by tauto
          have hpc : pc = ⟨q.turn, .pawn⟩ := by
            obtain ⟨col, kind⟩ := pc; simp only at hcol hkind; rw [hcol, hkind]
          subst hpc
          apply hnopromo s hs t hpr
          obtain ⟨-, hg⟩ := hps
          simp only [Bool.or_eq_true, Bool.and_eq_true] at hg
          rcases hg with ⟨h1, _⟩ | ⟨h1, _⟩
          · exact Or.inl h1
          · exact Or.inr h1
  | enPassant s t =>
      exfalso
      simp only [pseudoLegal, pseudoLegalB] at hps
      unfold pseudoLegalEnPassantB at hps
      rw [hep] at hps
      simp at hps
  | whiteKingCastle | whiteQueenCastle | blackKingCastle | blackQueenCastle =>
      exfalso
      simp only [pseudoLegal, pseudoLegalB, pseudoLegalCastleB, castleColorOf, castleRightOf,
        castleRookSource, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hps
      obtain ⟨⟨⟨⟨hturn, _⟩, _⟩, hrook⟩, _⟩ := hps
      rcases henemy _ _ hrook (by rw [hturn]) with ⟨_, hk⟩ | hk <;> cases hk

-- chess-lean proves this by `native_decide` (an extra axiom); the kernel's own `decide` does it too.
set_option maxRecDepth 100000 in
theorem allSquares_nodup' : allSquares.Nodup := by decide

/-- Counting squares: one pawn moves from `s` to `t`, the count is unchanged. -/
theorem filter_length_swap {P P' : Square → Bool} {s t : Square} (hst : s ≠ t) (hPs : P s = true)
    (hPt : P t = false) (hP' : ∀ u, P' u = true ↔ (u = t ∨ (P u = true ∧ u ≠ s))) :
    (allSquares.filter P').length = (allSquares.filter P).length := by
  have huniv : allSquares.toFinset = (Finset.univ : Finset Square) := by
    ext u; simp [mem_allSquares]
  rw [← List.toFinset_card_of_nodup (allSquares_nodup'.filter _),
    ← List.toFinset_card_of_nodup (allSquares_nodup'.filter _), List.toFinset_filter, List.toFinset_filter, huniv]
  have hset : Finset.univ.filter (fun u => P' u = true) =
      insert t ((Finset.univ.filter (fun u => P u = true)).erase s) := by
    ext u
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Finset.mem_erase, hP']
    tauto
  have hs_mem : s ∈ Finset.univ.filter (fun u => P u = true) := by simp [hPs]
  have ht_not : t ∉ (Finset.univ.filter (fun u => P u = true)).erase s := by simp [hPt]
  have hpos : 0 < (Finset.univ.filter (fun u => P u = true)).card := Finset.card_pos.mpr ⟨s, hs_mem⟩
  rw [hset, Finset.card_insert_of_notMem ht_not, Finset.card_erase_of_mem hs_mem]
  omega

theorem filter_length_congr {P P' : Square → Bool} (h : ∀ u, P' u = P u) :
    (allSquares.filter P').length = (allSquares.filter P).length := by
  rw [List.filter_congr (fun u _ => h u)]

/-- After our push and any legal enemy reply, the invariant holds one push further. -/
theorem inv_step2 {c : Color} {p : Position} {x : Square} {r : Nat} {ek ok : Square} {F : Square → Bool}
    {np : Nat} (h : Inv2 c p x (r + 2) ek ok F np) {m : Move} (hm : push c x = some m) {e : Move}
    (he : legalMove (applyMove p m) e) :
    ∃ x' ek', ahead c x 1 = some x' ∧ Inv2 c (applyMove (applyMove p m) e) x' (r + 1) ek' ok F np := by
  obtain ⟨y, hy, rfl, hy_empty, hboard⟩ := push_board_two2 h hm
  set q := applyMove p (.normal x y) with hq
  have hqturn : q.turn = Color.opposite c := by rw [hq, applyMove_turn, h.turn]
  have hne : ∀ d : Color, d ≠ Color.opposite d := by intro d; cases d <;> decide
  have hcol_ne : ∀ k k' : PieceKind, (some (⟨c, k⟩ : Piece) : Option Piece) ≠ some ⟨Color.opposite c, k'⟩ := by
    intro k k' hh; simp only [Option.some.injEq, Piece.mk.injEq] at hh; exact hne c hh.1
  have hcol_ne' : ∀ k k' : PieceKind, (some (⟨Color.opposite c, k⟩ : Piece) : Option Piece) ≠ some ⟨c, k'⟩ := by
    intro k k' hh; simp only [Option.some.injEq, Piece.mk.injEq] at hh; exact hne c hh.1.symm
  have hpawn := h.pawn
  have hxy : x ≠ y := by intro e; rw [← e, hpawn] at hy_empty; cases hy_empty
  have hekx : ek ≠ x := by intro e; have := h.eking; rw [e, hpawn] at this; simp at this
  have heky : ek ≠ y := by intro e; have := h.eking; rw [e, hy_empty] at this; cases this
  have hqother : ∀ s, s ≠ y → s ≠ x → q.board s = p.board s := by
    intro s hsy hsx; rw [hboard]; simp [boardUpdate, Function.update_apply, hsy, hsx]
  have hqy : q.board y = some ⟨c, .pawn⟩ := by rw [hboard]; simp [boardUpdate]
  have hqx : q.board x = none := by rw [hboard]; simp [boardUpdate, Function.update_apply, hxy]
  have hek_q : q.board ek = some ⟨Color.opposite c, .king⟩ := by rw [hqother ek heky hekx]; exact h.eking
  -- geometry of the pawn's step
  have hyf := ahead_file hy
  have hpromo_y : promoSquare c y = promoSquare c x := promoSquare_of_file hyf
  have hpush_y : pushes c y = r + 1 := by have := pushes_ahead hy; rw [h.pushesEq] at this; omega
  have hy_near : kdist y (promoSquare c x) = r + 1 := by rw [← hpromo_y, kdist_promo, hpush_y]
  have hout := h.outside
  have hymem : y ∈ path c x := ahead_mem_path hy
  -- the enemy's pieces in q
  have hq_enemy : ∀ s pc, q.board s = some pc → pc.color = q.turn →
      (s = ek ∧ pc.kind = .king) ∨ pc.kind = .pawn := by
    rw [hqturn]; intro s pc hs hcol
    by_cases hsy : s = y
    · rw [hsy, hqy] at hs; cases hs; exact absurd hcol (hne c)
    · by_cases hsx : s = x
      · rw [hsx, hqx] at hs; cases hs
      · rw [hqother s hsy hsx] at hs; exact h.enemy s pc hs hcol
  have hq_pawn_eq : ∀ s, q.board s = some ⟨Color.opposite c, .pawn⟩ ↔ p.board s = some ⟨Color.opposite c, .pawn⟩ := by
    intro s
    by_cases hsy : s = y
    · rw [hsy, hqy, hy_empty]
      constructor
      · intro hh; exact absurd hh (hcol_ne _ _)
      · intro hh; cases hh
    · by_cases hsx : s = x
      · rw [hsx, hqx, hpawn]
        constructor
        · intro hh; cases hh
        · intro hh; exact absurd hh (hcol_ne _ _)
      · rw [hqother s hsy hsx]
  have hq_ep : q.enPassant = none := push_enPassant_none h.turn hy
  -- an enemy pawn's move never ends on a bad square (it is more than one move away)
  have hmoved_not_bad : ∀ s t, p.board s = some ⟨Color.opposite c, .pawn⟩ →
      (pawnMoveGeometryB (Color.opposite c) s t = true ∨ pawnAttackGeometryB (Color.opposite c) s t = true ∨
        pawnDoubleGeometryB (Color.opposite c) s t = true) →
      ∀ t', t' ∈ badSquares c x ok → t ≠ t' := by
    intro s t hs hmv t' ht' htt'
    subst htt'
    have hfar := h.pawnsFar s hs t ht'
    have hback := h.pawnsOffBack s hs
    have := pdist_step hback hmv (k := r) (by simpa using hfar)
    rw [farB_self] at this
    cases this
  have hq_nopromo : ∀ s, q.board s = some ⟨q.turn, .pawn⟩ → ∀ t, t.2 = promotionRank q.turn →
      ¬ (pawnMoveGeometryB q.turn s t = true ∨ pawnAttackGeometryB q.turn s t = true) := by
    rw [hqturn]; intro s hs t ht hmv
    have hp_s := (hq_pawn_eq s).1 hs
    have hbad : t ∈ badSquares c x ok := mem_badSquares.mpr (Or.inr (Or.inr ht))
    exact hmoved_not_bad s t hp_s (by rcases hmv with h1 | h1; exact Or.inl h1; exact Or.inr (Or.inl h1)) t hbad rfl
  -- our kings and pieces, from p to q
  have hq_ours : ∀ s pc, q.board s = some pc → pc.color = c → s = y ∨ (s ≠ x ∧ p.board s = some pc) := by
    intro s pc hs hcol
    by_cases hsy : s = y
    · exact Or.inl hsy
    · by_cases hsx : s = x
      · rw [hsx, hqx] at hs; cases hs
      · exact Or.inr ⟨hsx, by rw [← hqother s hsy hsx]; exact hs⟩
  have hq_kings : ∀ s, q.board s = some ⟨c, .king⟩ ↔ p.board s = some ⟨c, .king⟩ := by
    intro s
    by_cases hsy : s = y
    · rw [hsy, hqy, hy_empty]; constructor <;> intro hh <;> simp at hh
    · by_cases hsx : s = x
      · rw [hsx, hqx, hpawn]; constructor <;> intro hh <;> simp at hh
      · rw [hqother s hsy hsx]
  -- our attacks in q lie in F
  have hq_attacksF : ∀ d, isAttackedB q c d = true → F d = true := by
    intro d hatt
    simp only [isAttackedB, List.any_eq_true] at hatt
    obtain ⟨s, -, hs⟩ := hatt
    obtain ⟨pc, hpc, hcol⟩ := pieceAttacksB_source hs
    rcases hq_ours s pc hpc hcol with hsy | ⟨hsx, hp⟩
    · rw [hsy] at hs
      rw [pawn_attack hqy] at hs
      exact h.futureInF d (future_of_pawn_attack hymem hs)
    · have hkind := h.noSliders s pc hp hcol
      have : pieceAttacksB p c s d = true := by
        rw [noslider_pieceAttacks hkind hp hpc]; exact hs
      exact h.attacksInF d (by simp only [isAttackedB, List.any_eq_true]; exact ⟨s, mem_allSquares s, this⟩)
  have hq_oursF : ∀ s pc, q.board s = some pc → pc.color = c → F s = true := by
    intro s pc hs hcol
    rcases hq_ours s pc hs hcol with hsy | ⟨_, hp⟩
    · rw [hsy]; exact h.pathInF y hymem
    · exact h.oursInF s pc hp hcol
  have hq_noSliders : ∀ s pc, q.board s = some pc → pc.color = c →
      pc.kind = .king ∨ pc.kind = .pawn ∨ pc.kind = .knight := by
    intro s pc hs hcol
    rcases hq_ours s pc hs hcol with hsy | ⟨_, hp⟩
    · rw [hsy, hqy] at hs; cases hs; simp
    · exact h.noSliders s pc hp hcol
  have hq_count : (allSquares.filter (fun s => q.board s == some ⟨Color.opposite c, .pawn⟩)).length = np := by
    rw [← h.count]; apply filter_length_congr; intro u
    apply Bool.eq_iff_iff.mpr; simp only [beq_iff_eq]; exact hq_pawn_eq u
  -- our king is not attacked in q by the enemy's king: the kings are apart
  have hkings_apart : kingGeometryB ek ok = false := by
    have hus := h.usSafe
    unfold inCheck inCheckB at hus
    rw [h.ourKing] at hus
    simp only [isAttackedB, List.any_eq_true, not_exists, not_and, Bool.not_eq_true] at hus
    have := hus ek (mem_allSquares ek)
    rw [king_attack h.eking] at this
    exact this
  -- an enemy pawn in p stands on no check square
  have hno_check_pawn : ∀ s, p.board s = some ⟨Color.opposite c, .pawn⟩ →
      pawnAttackGeometryB (Color.opposite c) s ok = false := by
    intro s hs
    apply Bool.eq_false_iff.mpr
    intro hg
    exact enemy_pawn_not_on_bad h (mem_badSquares.mpr (Or.inr (Or.inl hg))) hs
  -- THE ENEMY'S MOVE
  rcases enemy_move2 hq_enemy (by rw [hqturn]; exact hek_q) hq_ep hq_nopromo he with
    ⟨d, rfl, hgeo, hfr, hnc⟩ | ⟨s, t, rfl, hs, hmv, hnpr, hfr, hnc⟩
  · -- ===== a king step ek -> d =====
    set q' := applyMove q (.normal ek d) with hq'
    have hboard' : q'.board = boardMove q.board ek d := rfl
    have hekd : ek ≠ d := by intro e; rw [e] at hgeo; simp [kingGeometryB] at hgeo
    have hstep := kdist_king_step hgeo
    have htri := kdist_triangle ek d (promoSquare c x)
    have hdist_d : r + 1 < kdist d (promoSquare c x) := by omega
    have hyd : y ≠ d := by intro e; rw [e] at hy_near; omega
    have hyek : y ≠ ek := heky.symm
    have hq'other : ∀ s, s ≠ d → s ≠ ek → q'.board s = q.board s := by
      intro s hsd hse; rw [hboard']; simp [boardMove, boardUpdate, Function.update_apply, hsd, hse]
    have hq'y : q'.board y = some ⟨c, .pawn⟩ := by rw [hq'other y hyd hyek]; exact hqy
    have hq'd : q'.board d = some ⟨Color.opposite c, .king⟩ := by
      rw [hboard']; simp [boardMove, boardUpdate, hek_q]
    have hq'ek : q'.board ek = none := by
      rw [hboard']; simp [boardMove, boardUpdate, Function.update_apply, hekd]
    have hd_not_our_king : q.board d ≠ some ⟨c, .king⟩ := by
      intro hqd; simp [destinationFriendlyB, hqd, hqturn] at hfr
    have hq'_ours : ∀ s pc, q'.board s = some pc → pc.color = c → q.board s = some pc := by
      intro s pc hs hcol
      by_cases hsd : s = d
      · rw [hsd, hq'd] at hs; cases hs; exact absurd hcol.symm (hne c)
      · by_cases hse : s = ek
        · rw [hse, hq'ek] at hs; cases hs
        · rw [← hq'other s hsd hse]; exact hs
    have hq'_pawn_eq : ∀ s, q'.board s = some ⟨Color.opposite c, .pawn⟩ ↔ q.board s = some ⟨Color.opposite c, .pawn⟩ := by
      intro s
      by_cases hsd : s = d
      · rw [hsd, hq'd]
        constructor
        · intro hh; cases hh
        · intro hh; simp [destinationFriendlyB, hh, hqturn] at hfr
      · by_cases hse : s = ek
        · rw [hse, hq'ek, hek_q]; constructor <;> intro hh <;> cases hh
        · rw [hq'other s hsd hse]
    have hk : ∀ s, q'.board s = some ⟨c, .king⟩ ↔ p.board s = some ⟨c, .king⟩ := by
      intro s
      by_cases hsd : s = d
      · rw [hsd, hq'd, ← hq_kings]
        constructor
        · intro hh; exact absurd hh (hcol_ne' _ _)
        · intro hh; exact absurd hh hd_not_our_king
      · by_cases hse : s = ek
        · rw [hse, hq'ek, ← hq_kings, hek_q]
          constructor
          · intro hh; cases hh
          · intro hh; exact absurd hh (hcol_ne' _ _)
        · rw [hq'other s hsd hse, hq_kings]
    have hbare' : ∀ s pc, q'.board s = some pc → pc.color = Color.opposite c → (s = d ∧ pc.kind = .king) ∨ pc.kind = .pawn := by
      intro s pc hs hcol
      by_cases hsd : s = d
      · rw [hsd, hq'd] at hs; cases hs; exact Or.inl ⟨hsd, rfl⟩
      · by_cases hse : s = ek
        · rw [hse, hq'ek] at hs; cases hs
        · rw [hq'other s hsd hse] at hs
          rcases hq_enemy s pc hs (by rw [hqturn]; exact hcol) with ⟨hse', _⟩ | hk
          · exact absurd hse' hse
          · exact Or.inr hk
    have honly : ∀ s, q'.board s = some ⟨Color.opposite c, .king⟩ ↔ s = d := by
      intro s
      constructor
      · intro hs; rcases hbare' s _ hs rfl with ⟨hd, _⟩ | hk; exact hd; cases hk
      · rintro rfl; exact hq'd
    -- the enemy king on d is not next to our king (its move was legal)
    have hd_ok : kingGeometryB d ok = false := by
      have hok : q'.board ok = some ⟨c, .king⟩ := (hk ok).2 (kingSquare_some h.ourKing).2
      unfold inCheck inCheckB at hnc
      rw [hqturn, kingSquare_of_only honly, Color.opposite_opposite] at hnc
      simp only [isAttackedB, List.any_eq_true, not_exists, not_and, Bool.not_eq_true] at hnc
      have := hnc ok (mem_allSquares ok)
      rw [king_attack hok, kingGeometryB_comm] at this
      exact this
    refine ⟨y, d, hy, ?_⟩
    exact
      { turn := by rw [hq', applyMove_turn, hqturn, Color.opposite_opposite]
        pawn := hq'y
        notFirst := ahead_notFirst hy
        pushesEq := hpush_y
        pos := by omega
        clear := by
          intro t ht
          obtain ⟨htx, hty⟩ := path_of_ahead hy ht
          have hnear := kdist_of_mem_path htx
          rw [h.pushesEq] at hnear
          have htd : t ≠ d := by intro e; rw [e] at hnear; omega
          have htek : t ≠ ek := by intro e; rw [e] at hnear; omega
          have htxne : t ≠ x := by intro e; rw [e] at htx; exact self_not_mem_path htx
          rw [hq'other t htd htek, hqother t hty htxne]
          exact h.clear t htx
        enemy := hbare'
        eking := hq'd
        pawnsOffBack := fun s hs => h.pawnsOffBack s ((hq_pawn_eq s).1 ((hq'_pawn_eq s).1 hs))
        outside := by rw [hpromo_y]; exact hdist_d
        noSliders := fun s pc hs hcol => hq_noSliders s pc (hq'_ours s pc hs hcol) hcol
        ourKing := by rw [kingSquare_congr hk]; exact h.ourKing
        usSafe := by
          intro hin
          unfold inCheck inCheckB at hin
          rw [kingSquare_congr hk, h.ourKing] at hin
          change isAttackedB q' (Color.opposite c) ok = true at hin
          simp only [isAttackedB, List.any_eq_true] at hin
          obtain ⟨s, -, hs⟩ := hin
          obtain ⟨pc, hpc, hcol⟩ := pieceAttacksB_source hs
          rcases hbare' s pc hpc hcol with ⟨hsd, hkd⟩ | hkp
          · rw [hsd] at hs
            rw [king_attack hq'd, hd_ok] at hs; cases hs
          · have hpc' : q'.board s = some ⟨Color.opposite c, .pawn⟩ := by
              obtain ⟨col, kind⟩ := pc; simp only at hcol hkp; rw [hcol, hkp] at hpc; exact hpc
            rw [pawn_attack hpc'] at hs
            have hp_s := (hq_pawn_eq s).1 ((hq'_pawn_eq s).1 hpc')
            rw [hno_check_pawn s hp_s] at hs; cases hs
        themSafe := by rw [← hqturn]; exact hnc
        pawnsFar := by
          intro s hs t ht
          have hp_s := (hq_pawn_eq s).1 ((hq'_pawn_eq s).1 hs)
          exact farB_mono (by omega) (h.pawnsFar s hp_s t (bad_mono hy ht))
        oursInF := fun s pc hs hcol => hq_oursF s pc (hq'_ours s pc hs hcol) hcol
        attacksInF := by
          intro d' hatt
          apply hq_attacksF d'
          -- our pieces in q' are our pieces in q, at most one fewer (a capture): any attacker in q' attacks in q
          simp only [isAttackedB, List.any_eq_true] at hatt ⊢
          obtain ⟨u, hu_mem, hu⟩ := hatt
          obtain ⟨pc, hpc, hcol⟩ := pieceAttacksB_source hu
          have hq_u := hq'_ours u pc hpc hcol
          refine ⟨u, hu_mem, ?_⟩
          rw [noslider_pieceAttacks (hq_noSliders u pc hq_u hcol) hq_u hpc]
          exact hu
        futureInF := fun d' hd' => h.futureInF d' (future_mono hy hd')
        pathInF := fun t ht => h.pathInF t (path_of_ahead hy ht).1
        count := by
          rw [← hq_count]; apply filter_length_congr; intro u
          apply Bool.eq_iff_iff.mpr; simp only [beq_iff_eq]; exact hq'_pawn_eq u
        room := by
          intro x' hx'
          have := h.room x' (by have := kdist_triangle x' d ek; rw [kdist_comm d ek] at this; omega)
          simp only [neverKingB, hpromo_y]
          exact this }
  · -- ===== a pawn move s -> t =====
    set q' := applyMove q (.normal s t) with hq'
    have hboard' : q'.board = boardMove q.board s t := rfl
    have hst : s ≠ t := by
      intro e; rw [e] at hmv
      cases c <;> simp [pawnMoveGeometryB, pawnAttackGeometryB, pawnDoubleGeometryB, absDiff, Color.opposite] at hmv <;> omega
    have hp_s : p.board s = some ⟨Color.opposite c, .pawn⟩ := (hq_pawn_eq s).1 (by rw [← hqturn]; exact hs)
    have hmv' : pawnMoveGeometryB (Color.opposite c) s t = true ∨ pawnAttackGeometryB (Color.opposite c) s t = true ∨
        pawnDoubleGeometryB (Color.opposite c) s t = true := by rw [← hqturn]; exact hmv
    have ht_not_bad : ∀ t', t' ∈ badSquares c x ok → t ≠ t' := hmoved_not_bad s t hp_s hmv'
    have hty : t ≠ y := ht_not_bad y (mem_badSquares.mpr (Or.inl hymem))
    have hsy : s ≠ y := by intro e; rw [e, hqy, hqturn] at hs; exact hcol_ne _ _ hs
    have hsek : s ≠ ek := by intro e; rw [e, hek_q] at hs; simp at hs
    have htek : t ≠ ek := by intro e; rw [e] at hfr; simp [destinationFriendlyB, hek_q, hqturn] at hfr
    have hq'other : ∀ u, u ≠ t → u ≠ s → q'.board u = q.board u := by
      intro u hut hus; rw [hboard']; simp [boardMove, boardUpdate, Function.update_apply, hut, hus]
    have hq't : q'.board t = some ⟨Color.opposite c, .pawn⟩ := by
      rw [hboard']; simp only [boardMove, boardUpdate, Function.update_apply, if_true]; rw [← hqturn]; exact hs
    have hq's : q'.board s = none := by
      rw [hboard']; simp [boardMove, boardUpdate, Function.update_apply, hst]
    have hq'y : q'.board y = some ⟨c, .pawn⟩ := by rw [hq'other y hty.symm hsy.symm]; exact hqy
    have hq'ek : q'.board ek = some ⟨Color.opposite c, .king⟩ := by rw [hq'other ek htek.symm hsek.symm]; exact hek_q
    have ht_not_our_king : q.board t ≠ some ⟨c, .king⟩ := by
      intro hqt; simp [destinationFriendlyB, hqt, hqturn] at hfr
    have hq'_ours : ∀ u pc, q'.board u = some pc → pc.color = c → q.board u = some pc := by
      intro u pc hu hcol
      by_cases hut : u = t
      · rw [hut, hq't] at hu; cases hu; exact absurd hcol.symm (hne c)
      · by_cases hus : u = s
        · rw [hus, hq's] at hu; cases hu
        · rw [← hq'other u hut hus]; exact hu
    have hk : ∀ u, q'.board u = some ⟨c, .king⟩ ↔ p.board u = some ⟨c, .king⟩ := by
      intro u
      by_cases hut : u = t
      · rw [hut, hq't, ← hq_kings]
        constructor
        · intro hh; exact absurd hh (hcol_ne' _ _)
        · intro hh; exact absurd hh ht_not_our_king
      · by_cases hus : u = s
        · rw [hus, hq's, ← hq_kings]
          constructor
          · intro hh; cases hh
          · intro hh; rw [hqturn] at hs; rw [hs] at hh; exact absurd hh (hcol_ne' _ _)
        · rw [hq'other u hut hus, hq_kings]
    have hbare' : ∀ u pc, q'.board u = some pc → pc.color = Color.opposite c → (u = ek ∧ pc.kind = .king) ∨ pc.kind = .pawn := by
      intro u pc hu hcol
      by_cases hut : u = t
      · rw [hut, hq't] at hu; cases hu; exact Or.inr rfl
      · by_cases hus : u = s
        · rw [hus, hq's] at hu; cases hu
        · rw [hq'other u hut hus] at hu
          exact hq_enemy u pc hu (by rw [hqturn]; exact hcol)
    have hq'_pawn : ∀ u, q'.board u = some ⟨Color.opposite c, .pawn⟩ ↔ (u = t ∨ (p.board u = some ⟨Color.opposite c, .pawn⟩ ∧ u ≠ s)) := by
      intro u
      by_cases hut : u = t
      · rw [hut, hq't]; simp
      · by_cases hus : u = s
        · rw [hus, hq's]; simp [hp_s, hst]
        · rw [hq'other u hut hus, hq_pawn_eq]; simp [hut, hus]
    refine ⟨y, ek, hy, ?_⟩
    exact
      { turn := by rw [hq', applyMove_turn, hqturn, Color.opposite_opposite]
        pawn := hq'y
        notFirst := ahead_notFirst hy
        pushesEq := hpush_y
        pos := by omega
        clear := by
          intro t' ht'
          obtain ⟨htx, hty'⟩ := path_of_ahead hy ht'
          have htt' : t ≠ t' := ht_not_bad t' (mem_badSquares.mpr (Or.inl htx))
          have hst' : s ≠ t' := by
            intro e; rw [e] at hp_s; rw [h.clear t' htx] at hp_s; cases hp_s
          have htxne : t' ≠ x := by intro e; rw [e] at htx; exact self_not_mem_path htx
          rw [hq'other t' htt'.symm hst'.symm, hqother t' hty' htxne]
          exact h.clear t' htx
        enemy := hbare'
        eking := hq'ek
        pawnsOffBack := by
          intro u hu
          rcases (hq'_pawn u).1 hu with rfl | ⟨hp_u, _⟩
          · -- the moved pawn went forward
            have hback := h.pawnsOffBack s hp_s
            have hs2 := s.2.isLt; have ht2 := u.2.isLt
            cases c <;> simp only [Color.opposite, backRank] at hback ⊢ <;>
              rcases hmv' with hm | hm | hm <;>
              simp only [Color.opposite, pawnMoveGeometryB, pawnAttackGeometryB, pawnDoubleGeometryB, absDiff,
                Bool.and_eq_true, beq_iff_eq, if_true, reduceCtorEq, if_false, Fin.ext_iff] at hm <;>
              (try split_ifs at hm) <;> omega
          · exact h.pawnsOffBack u hp_u
        outside := by rw [hpromo_y]; omega
        noSliders := fun u pc hu hcol => hq_noSliders u pc (hq'_ours u pc hu hcol) hcol
        ourKing := by rw [kingSquare_congr hk]; exact h.ourKing
        usSafe := by
          intro hin
          unfold inCheck inCheckB at hin
          rw [kingSquare_congr hk, h.ourKing] at hin
          change isAttackedB q' (Color.opposite c) ok = true at hin
          simp only [isAttackedB, List.any_eq_true] at hin
          obtain ⟨u, -, hu⟩ := hin
          obtain ⟨pc, hpc, hcol⟩ := pieceAttacksB_source hu
          rcases hbare' u pc hpc hcol with ⟨hue, hkd⟩ | hkp
          · rw [hue] at hu; rw [king_attack hq'ek, hkings_apart] at hu; cases hu
          · have hpc' : q'.board u = some ⟨Color.opposite c, .pawn⟩ := by
              obtain ⟨col, kind⟩ := pc; simp only at hcol hkp; rw [hcol, hkp] at hpc; exact hpc
            rw [pawn_attack hpc'] at hu
            rcases (hq'_pawn u).1 hpc' with rfl | ⟨hp_u, _⟩
            · exact ht_not_bad u (mem_badSquares.mpr (Or.inr (Or.inl hu))) rfl
            · rw [hno_check_pawn u hp_u] at hu; cases hu
        themSafe := by rw [← hqturn]; exact hnc
        pawnsFar := by
          intro u hu t' ht'
          have ht'x := bad_mono hy ht'
          rcases (hq'_pawn u).1 hu with rfl | ⟨hp_u, _⟩
          · exact pdist_step (h.pawnsOffBack s hp_s) hmv' (k := r) (by simpa using h.pawnsFar s hp_s t' ht'x)
          · exact farB_mono (by omega) (h.pawnsFar u hp_u t' ht'x)
        oursInF := fun u pc hu hcol => hq_oursF u pc (hq'_ours u pc hu hcol) hcol
        attacksInF := by
          intro d' hatt
          apply hq_attacksF d'
          -- our pieces in q' are our pieces in q, at most one fewer (a capture): any attacker in q' attacks in q
          simp only [isAttackedB, List.any_eq_true] at hatt ⊢
          obtain ⟨u, hu_mem, hu⟩ := hatt
          obtain ⟨pc, hpc, hcol⟩ := pieceAttacksB_source hu
          have hq_u := hq'_ours u pc hpc hcol
          refine ⟨u, hu_mem, ?_⟩
          rw [noslider_pieceAttacks (hq_noSliders u pc hq_u hcol) hq_u hpc]
          exact hu
        futureInF := fun d' hd' => h.futureInF d' (future_mono hy hd')
        pathInF := fun t' ht' => h.pathInF t' (path_of_ahead hy ht').1
        count := by
          rw [← hq_count]
          apply filter_length_swap (s := s) (t := t) hst
          · simp only [beq_iff_eq]; rw [← hqturn]; exact hs
          · apply Bool.eq_false_iff.mpr; intro hh
            simp only [beq_iff_eq] at hh
            simp [destinationFriendlyB, hh, hqturn] at hfr
          · intro u; simp only [beq_iff_eq]; rw [hq'_pawn u, hq_pawn_eq]
        room := by
          intro x' hx'
          have := h.room x' (by omega)
          simp only [neverKingB, hpromo_y]
          exact this }

/-- The race, second form. -/
theorem race2 (c : Color) (f : Fin 8) (ok : Square) (F : Square → Bool) (np : Nat) :
    ∀ r (p : Position) (x : Square) (ek : Square), x.1 = f → Inv2 c p x r ek ok F np →
      forcedWithinB c (promoted c f) p (2 * r - 1) = true := by
  intro r
  induction r with
  | zero => intro p x ek _ h; exact absurd h.pos (by omega)
  | succ r ih =>
    intro p x ek hf h
    obtain ⟨m, hm, hl⟩ := push_legal2 h
    cases r with
    | zero =>
        show forcedWithinB c (promoted c f) p (0 + 1) = true
        rw [forcedWithinB_ours h.turn]
        simp only [Bool.or_eq_true, List.any_eq_true]
        right
        exact ⟨m, legalMoves_complete hl, by subst hf; exact promoted_after_last2 h hm⟩
    | succ r =>
        have : 2 * (r + 1 + 1) - 1 = (2 * (r + 1) - 1 + 1) + 1 := by omega
        rw [this, forcedWithinB_ours h.turn]
        simp only [Bool.or_eq_true, List.any_eq_true]
        right
        refine ⟨m, legalMoves_complete hl, ?_⟩
        rw [forcedWithinB_theirs (after_turn_ne m h.turn)]
        simp only [Bool.or_eq_true, Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_eq_false_iff,
          List.all_eq_true]
        right
        refine ⟨enemy_has_move2 h hm, ?_⟩
        intro e he
        obtain ⟨x', ek', hx', h'⟩ := inv_step2 h hm (legalMoves_sound he)
        exact ih _ _ _ ((ahead_file hx').trans hf) h'

/-! ## From the Idea's hypotheses to the invariant -/

/-- **Soundness of the unstoppable passer, second form.** When every hypothesis holds, our pawn stands on
the promotion square as a queen within `2n - 1` plies whatever the opponent does. -/
theorem unstoppablePasser2_sound (p : Position) (ps : PasserParams)
    (h : unstoppablePasser2.holds p ps = true) :
    forcedWithinB p.turn (promoted p.turn ps.sq.1) p (2 * pushes p.turn ps.sq - 1) = true := by
  simp only [Idea.holds, unstoppablePasser2, List.all_cons, List.all_nil, Bool.and_true,
    Bool.and_eq_true] at h
  obtain ⟨hpawn, hclear, henemy, hns, hnc, hout, hfar, hroom⟩ := h
  have hne : ∀ d : Color, d ≠ Color.opposite d := by intro d; cases d <;> decide
  -- the pawn
  simp only [ownPawnB, Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at hpawn
  obtain ⟨⟨hpawn, hnotpromo⟩, hnotfirst⟩ := hpawn
  -- the enemy king and pawns
  simp only [enemyKingPawnsB, Bool.and_eq_true, List.all_eq_true] at henemy
  obtain ⟨hall, hone⟩ := henemy
  cases hek : kingSquare p.board (Color.opposite p.turn) with
  | none => rw [hek] at hone; simp at hone
  | some ek =>
    rw [hek] at hone
    simp only [List.all_eq_true] at hone
    have hek_board : p.board ek = some ⟨Color.opposite p.turn, .king⟩ := (kingSquare_some hek).2
    have hpiece : ∀ s pc, p.board s = some pc → pc.color = Color.opposite p.turn →
        pc.kind = .king ∨ (pc.kind = .pawn ∧ s.2.val ≠ backRank (Color.opposite p.turn)) := by
      intro s pc hs hcol
      have h1 := hall s (mem_allSquares s)
      rw [hs] at h1
      simp only [Bool.or_eq_true, beq_iff_eq, Bool.and_eq_true, bne_iff_ne, ne_eq] at h1
      rcases h1 with (h1 | h1) | h1
      · rw [hcol] at h1; exact absurd h1.symm (hne _)
      · exact Or.inl h1
      · exact Or.inr h1
    have henemy' : ∀ s pc, p.board s = some pc → pc.color = Color.opposite p.turn →
        (s = ek ∧ pc.kind = .king) ∨ pc.kind = .pawn := by
      intro s pc hs hcol
      rcases hpiece s pc hs hcol with hk | ⟨hk, _⟩
      · left
        refine ⟨?_, hk⟩
        have h2 := hone s (mem_allSquares s)
        have hpc : pc = ⟨Color.opposite p.turn, .king⟩ := by
          obtain ⟨col, kind⟩ := pc; simp only at hcol hk; rw [hcol, hk]
        rw [hs, hpc] at h2
        simp only [beq_self_eq_true, beq_iff_eq] at h2
        exact of_decide_eq_true h2.symm
      · exact Or.inr hk
    have hoffback : ∀ s, p.board s = some ⟨Color.opposite p.turn, .pawn⟩ →
        s.2.val ≠ backRank (Color.opposite p.turn) := by
      intro s hs
      rcases hpiece s _ hs rfl with hk | ⟨_, hb⟩
      · cases hk
      · exact hb
    -- not in check, our king
    simp only [notInCheckB, Bool.and_eq_true, Bool.not_eq_true', inCheckB'_eq] at hnc
    obtain ⟨⟨hour, hus⟩, hthem⟩ := hnc
    cases hok : kingSquare p.board p.turn with
    | none => rw [hok] at hour; simp at hour
    | some ok =>
      simp only [kingOutsideB, hek, decide_eq_true_eq] at hout
      simp only [pawnsFarB, hok, List.all_eq_true, Bool.or_eq_true, bne_iff_ne, ne_eq] at hfar
      simp only [kingRoomB, hek, hok, List.all_eq_true, Bool.or_eq_true, decide_eq_true_eq] at hroom
      have hpos : 1 ≤ pushes p.turn ps.sq := pushes_pos_of_not_promo hnotpromo
      apply race2 p.turn ps.sq.1 ok (forbiddenB p ps.sq) (enemyPawnCount p) (pushes p.turn ps.sq) p ps.sq ek rfl
      exact
        { turn := rfl
          pawn := hpawn
          notFirst := hnotfirst
          pushesEq := rfl
          pos := hpos
          clear := by
            intro t ht
            simp only [pathClearB, List.all_eq_true, beq_iff_eq] at hclear
            exact hclear t ht
          enemy := henemy'
          eking := hek_board
          pawnsOffBack := hoffback
          outside := hout
          noSliders := by
            intro s pc hs hcol
            simp only [noSlidersB, List.all_eq_true] at hns
            have := hns s (mem_allSquares s)
            rw [hs] at this
            simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq] at this
            rcases this with ((hh | hh) | hh) | hh
            · exact absurd hcol hh
            · exact Or.inl hh
            · exact Or.inr (Or.inl hh)
            · exact Or.inr (Or.inr hh)
          ourKing := hok
          usSafe := by unfold inCheck; rw [hus]; exact Bool.false_ne_true
          themSafe := by unfold inCheck; rw [hthem]; exact Bool.false_ne_true
          pawnsFar := by
            intro s hs t ht
            rcases hfar s (mem_allSquares s) with hh | hh
            · exact absurd hs hh
            · exact hh t ht
          oursInF := by
            intro d pc hd hcol
            simp only [forbiddenB, hd, hcol, beq_self_eq_true, Bool.true_or]
          attacksInF := by
            intro d hatt
            simp only [forbiddenB, Bool.or_eq_true, isAttackedB'_eq]
            exact Or.inl (Or.inl (Or.inl (Or.inr hatt)))
          futureInF := by
            intro d hd
            simp only [forbiddenB, Bool.or_eq_true]
            exact Or.inl (Or.inl (Or.inr hd))
          pathInF := by
            intro t ht
            simp only [forbiddenB, Bool.or_eq_true]
            exact Or.inl (Or.inr (List.contains_iff_mem.mpr ht))
          count := rfl
          room := by
            intro x' hx'
            rcases hroom x' (mem_allSquares x') with (hh | hh) | hh
            · omega
            · exact Or.inl hh
            · exact Or.inr hh }

end Passer2

end Ideas
