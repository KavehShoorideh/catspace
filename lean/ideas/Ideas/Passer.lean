import Ideas.Schema
import Ideas.Fast
import Ideas.Mate
import Chess.Game

/-!
# The unstoppable passer (the rule of the square), first form: a bare enemy king

Kaveh's example (2026-10-04): "I don't need a search to know my pawn can promote, and I don't need a
search to know that my focus should be getting rid of that bishop." Our pawn on `sq`, `n` single pushes
from promotion; we are to move. Hypotheses, each evaluable alone:

* `ownPawn`      -- our pawn stands on `sq`, not on either back rank;
* `pathClear`    -- the `n` squares ahead are empty (the blocking bishop fails this one: the target);
* `bareKing`     -- the opponent has exactly one piece, its king;
* `noSliders`    -- we have no rook, bishop or queen: a push can then never uncover a check, so the
                    enemy king is never in check and always has the square it came from;
* `notInCheck`   -- neither king is attacked (our king exists);
* `kingOutside`  -- the enemy king's king-distance to the promotion square exceeds `n`
                    (the rule of the square, with the move; it also keeps the new queen safe one ply);
* `kingHasMove`  -- after our first push the enemy king has an empty neighbour we do not attack
                    (no stalemate on its first turn; later it can always step back).

Conclusion: whatever the opponent does, our pawn stands on the promotion square as a queen within
`2n - 1` plies. Tier `unconditional`. The general lemma -- enemy pieces whose reach within the plies
they get misses the path (over-approximated), our sliders with "promotes or mates" -- comes after
this one is proved.
-/

namespace Ideas

open Chess

structure PasserParams where
  sq : Square
  deriving Repr, DecidableEq

namespace Passer

/-- Pushes from `sq` to the promotion rank. -/
def pushes (c : Color) (sq : Square) : Nat :=
  match c with
  | .white => 7 - sq.2.val
  | .black => sq.2.val

/-- The square `i` steps ahead of `sq` for colour `c` (none off the board). -/
def ahead (c : Color) (sq : Square) (i : Nat) : Option Square :=
  match c with
  | .white => if h : sq.2.val + i < 8 then some (sq.1, ⟨sq.2.val + i, h⟩) else none
  | .black => if h : i ≤ sq.2.val then some (sq.1, ⟨sq.2.val - i, by omega⟩) else none

/-- The squares ahead up to and including the promotion square. -/
def path (c : Color) (sq : Square) : List Square :=
  (List.range (pushes c sq)).filterMap fun i => ahead c sq (i + 1)

def promoSquare (c : Color) (sq : Square) : Square := (sq.1, promotionRank c)

/-- King distance (Chebyshev). -/
def kdist (s t : Square) : Nat := max (absDiff s.1 t.1) (absDiff s.2 t.2)

/-- Our next pawn move from `s`: a push, or the promotion to a queen from the seventh rank. -/
def push (c : Color) (s : Square) : Option Move :=
  match ahead c s 1 with
  | some t => some (if t.2 = promotionRank c then .promotion s t .queen else .normal s t)
  | none => none

/-- The goal: our queen stands on the promotion square. -/
def promoted (c : Color) (f : Fin 8) (q : Position) : Bool :=
  q.board (f, promotionRank c) == some ⟨c, .queen⟩

/-! ## Hypotheses -/

def ownPawnB (p : Position) (ps : PasserParams) : Bool :=
  (p.board ps.sq == some ⟨p.turn, .pawn⟩) && (ps.sq.2 != promotionRank p.turn) &&
    -- a pawn on its own first rank is illegal; the kernel's double-push geometry would also differ
    (ps.sq.2 != promotionRank (Color.opposite p.turn))

def pathClearB (p : Position) (ps : PasserParams) : Bool :=
  (path p.turn ps.sq).all fun t => p.board t == none

/-- Every enemy piece is a king, and there is exactly one. -/
def bareKingB (p : Position) (_ : PasserParams) : Bool :=
  let e := Color.opposite p.turn
  allSquares.all (fun s => match p.board s with
    | some pc => (pc.color == p.turn) || (pc.kind == .king)
    | none => true) &&
  match kingSquare p.board e with
  | some k => allSquares.all fun s => (p.board s == some ⟨e, .king⟩) == decide (s = k)
  | none => false

/-- We have no rook, bishop or queen. -/
def noSlidersB (p : Position) (_ : PasserParams) : Bool :=
  allSquares.all fun s => match p.board s with
    | some pc => (pc.color != p.turn) || (pc.kind == .king) || (pc.kind == .pawn) || (pc.kind == .knight)
    | none => true

/-- Neither king is attacked (ours exists). -/
def notInCheckB (p : Position) (_ : PasserParams) : Bool :=
  (kingSquare p.board p.turn).isSome && !inCheckB' p p.turn && !inCheckB' p (Color.opposite p.turn)

def kingOutsideB (p : Position) (ps : PasserParams) : Bool :=
  match kingSquare p.board (Color.opposite p.turn) with
  | some ek => decide (pushes p.turn ps.sq < kdist ek (promoSquare p.turn ps.sq))
  | none => false

/-- After our first push, the enemy king has an empty neighbour that we do not attack. -/
def kingHasMoveB (p : Position) (ps : PasserParams) : Bool :=
  match push p.turn ps.sq, kingSquare p.board (Color.opposite p.turn) with
  | some m, some ek =>
      let q := applyMove p m
      allSquares.any fun d => kingGeometryB ek d && (q.board d == none) && !isAttackedB' q p.turn d
  | _, _ => false

end Passer

open Passer in
def unstoppablePasser : Idea PasserParams where
  id := "unstoppablePasser"
  hyps :=
    [ ⟨"ownPawn", ownPawnB⟩, ⟨"pathClear", pathClearB⟩, ⟨"bareKing", bareKingB⟩,
      ⟨"noSliders", noSlidersB⟩, ⟨"notInCheck", notInCheckB⟩, ⟨"kingOutside", kingOutsideB⟩,
      ⟨"kingHasMove", kingHasMoveB⟩ ]
  protagonist := fun ps => ps.sq
  tier := .unconditional
  horizon := 13          -- at most 2 * 7 - 1 plies; per instance 2 * pushes - 1

/-- Where to look when one hypothesis fails. -/
def Passer.targets (p : Position) (ps : PasserParams) (hyp : String) : List Square :=
  if hyp = "pathClear" then (path p.turn ps.sq).filter (fun t => (p.board t).isSome)
  else if hyp = "bareKing" then
    allSquares.filter (fun s => match p.board s with
      | some pc => pc.color != p.turn && pc.kind != .king
      | none => false)
  else if hyp = "kingOutside" then
    match kingSquare p.board (Color.opposite p.turn) with | some ek => [ek] | none => []
  else []

/-- "Side `c` reaches `goal` within `n` plies whatever the opponent does": at `c`'s turn some legal move
reaches it, at the other side's turn every legal move does and there is at least one (a stalemated
opponent does not count as reaching). -/
def forcedWithinB (c : Color) (goal : Position → Bool) : Position → Nat → Bool
  | p, 0 => goal p
  | p, n + 1 =>
      goal p ||
      (if p.turn = c then
        (legalMoves p).any (fun m => forcedWithinB c goal (applyMove p m) n)
      else
        let ms := legalMoves p
        !ms.isEmpty && ms.all (fun m => forcedWithinB c goal (applyMove p m) n))

/-! ## First facts -/

namespace Passer

/-- Chebyshev distance obeys the triangle inequality. -/
theorem kdist_triangle (a b c : Square) : kdist a c ≤ kdist a b + kdist b c := by
  simp only [kdist, absDiff]
  split_ifs <;> omega

theorem kdist_comm (a b : Square) : kdist a b = kdist b a := by
  simp only [kdist, absDiff]
  split_ifs <;> omega

/-- A king step moves at most one. -/
theorem kdist_king_step {a b : Square} (h : kingGeometryB a b = true) : kdist a b ≤ 1 := by
  simp only [kingGeometryB, Bool.and_eq_true, decide_eq_true_eq] at h
  simp only [kdist]
  omega

set_option maxRecDepth 100000 in
theorem kingGeometryB_comm : ∀ a b : Square, kingGeometryB a b = kingGeometryB b a := by
  decide

/-- When the side to move has only its king, on `ek`, a legal move is a king step to a square free of
its own pieces that leaves the king unattacked. -/
theorem bare_king_move {p : Position} {m : Move} {ek : Square}
    (hbare : ∀ s pc, p.board s = some pc → pc.color = p.turn → s = ek ∧ pc.kind = .king)
    (hl : legalMove p m) :
    ∃ t, m = .normal ek t ∧ kingGeometryB ek t = true ∧ destinationFriendlyB p p.turn t = true ∧
      ¬ inCheck (applyMove p m) p.turn := by
  obtain ⟨hps, hnc, _⟩ := hl
  cases m with
  | normal s t =>
      simp only [pseudoLegal, pseudoLegalB] at hps
      unfold normalPseudoLegalB at hps
      cases hs : p.board s with
      | none => simp [hs] at hps
      | some pc =>
          rw [hs] at hps
          simp only [Bool.and_eq_true, beq_iff_eq] at hps
          obtain ⟨⟨⟨hcol, hfr⟩, _⟩, hgeo⟩ := hps
          obtain ⟨rfl, hkind⟩ := hbare s pc hs hcol
          refine ⟨t, rfl, ?_, hfr, hnc⟩
          unfold normalPieceGeometryB at hgeo
          simp only [hkind] at hgeo
          exact hgeo
  | promotion s t k =>
      exfalso
      simp only [pseudoLegal, pseudoLegalB] at hps
      unfold pseudoLegalPromotionB at hps
      cases hs : p.board s with
      | none => simp [hs] at hps
      | some pc =>
          simp only [hs, Bool.and_eq_true, beq_iff_eq] at hps
          obtain ⟨⟨⟨hcol, hkind⟩, _⟩, _⟩ := hps
          have := (hbare s pc hs hcol).2
          rw [hkind] at this
          cases this
  | enPassant s t =>
      exfalso
      simp only [pseudoLegal, pseudoLegalB] at hps
      unfold pseudoLegalEnPassantB at hps
      cases hep : p.enPassant with
      | none => simp [hep] at hps
      | some ep =>
          cases hs : p.board s with
          | none => simp [hep, hs] at hps
          | some pc =>
              simp only [hep, hs, Bool.and_eq_true, beq_iff_eq] at hps
              have hcol : pc.color = p.turn := by tauto
              have hkind : pc.kind = .pawn := by tauto
              have := (hbare s pc hs hcol).2
              rw [hkind] at this
              cases this
  | whiteKingCastle | whiteQueenCastle | blackKingCastle | blackQueenCastle =>
      exfalso
      simp only [pseudoLegal, pseudoLegalB, pseudoLegalCastleB, castleColorOf, castleRightOf,
        castleRookSource, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hps
      obtain ⟨⟨⟨⟨hturn, _⟩, _⟩, hrook⟩, _⟩ := hps
      have := (hbare _ _ hrook (by rw [hturn])).2
      cases this

/-! ## The invariant of the race -/

/-- Our side `c`, pawn on `x` with `r` pushes to go, the enemy king on `ek`, at our turn. -/
structure Inv (c : Color) (p : Position) (x : Square) (r : Nat) (ek : Square) : Prop where
  turn : p.turn = c
  pawn : p.board x = some ⟨c, .pawn⟩
  notFirst : x.2 ≠ promotionRank (Color.opposite c)
  pushesEq : pushes c x = r
  pos : 1 ≤ r
  clear : ∀ t ∈ path c x, p.board t = none
  bare : ∀ s pc, p.board s = some pc → pc.color = Color.opposite c → s = ek ∧ pc.kind = .king
  eking : p.board ek = some ⟨Color.opposite c, .king⟩
  outside : r < kdist ek (promoSquare c x)
  noSliders : ∀ s pc, p.board s = some pc → pc.color = c →
    pc.kind = .king ∨ pc.kind = .pawn ∨ pc.kind = .knight
  ourKing : (kingSquare p.board c).isSome
  usSafe : ¬ inCheck p c
  themSafe : ¬ inCheck p (Color.opposite c)
  /-- with two or more pushes to go, after our push the enemy king has an empty neighbour we do not
  attack (after the last push our new queen is a slider, and the clause is not needed) -/
  free : 2 ≤ r → ∀ m, push c x = some m →
    ∃ d, kingGeometryB ek d = true ∧ (applyMove p m).board d = none ∧
      ¬ isAttacked (applyMove p m) c d

/-! ### Geometry of the path -/

theorem ahead_file {c : Color} {x y : Square} {i : Nat} (h : ahead c x i = some y) : y.1 = x.1 := by
  cases c <;> simp only [ahead] at h <;> split_ifs at h <;> simp only [Option.some.injEq] at h <;>
    rw [← h]

/-- The square one step ahead, for a pawn with at least one push to go. -/
theorem ahead_one {c : Color} {x : Square} (h : 1 ≤ pushes c x) :
    ∃ y, ahead c x 1 = some y ∧ y.1 = x.1 ∧
      (c = .white → y.2.val = x.2.val + 1) ∧ (c = .black → y.2.val + 1 = x.2.val) := by
  have hx := x.2.isLt
  cases c with
  | white =>
      simp only [pushes] at h
      have hr : x.2.val + 1 < 8 := by omega
      refine ⟨(x.1, ⟨x.2.val + 1, hr⟩), ?_, rfl, fun _ => rfl, fun h => by cases h⟩
      simp [ahead, hr]
  | black =>
      simp only [pushes] at h
      refine ⟨(x.1, ⟨x.2.val - 1, by omega⟩), by simp [ahead, h], rfl,
        fun hc => absurd hc (by decide), fun _ => ?_⟩
      simp only
      omega

theorem pushes_pos_of_not_promo {c : Color} {x : Square} (h : x.2 ≠ promotionRank c) : 1 ≤ pushes c x := by
  have hx := x.2.isLt
  cases c <;> simp only [pushes, promotionRank, ne_eq, Fin.ext_iff] at h ⊢ <;> omega

/-- The coordinates one step ahead. -/
theorem ahead_one_vals {c : Color} {x y : Square} (hy : ahead c x 1 = some y) :
    y.1 = x.1 ∧ (c = .white → y.2.val = x.2.val + 1) ∧ (c = .black → y.2.val + 1 = x.2.val) := by
  have hx := x.2.isLt
  cases c <;> simp only [ahead] at hy <;> split_ifs at hy <;>
    simp only [Option.some.injEq] at hy <;> subst hy <;>
    refine ⟨rfl, fun hc => ?_, fun hc => ?_⟩ <;> first | exact absurd hc (by decide) | (dsimp only <;> omega)

/-- The path: the squares of the pawn's file strictly ahead of it. -/
theorem mem_path_iff {c : Color} {x t : Square} :
    t ∈ path c x ↔ t.1 = x.1 ∧ (c = .white → x.2.val < t.2.val) ∧ (c = .black → t.2.val < x.2.val) := by
  have hx := x.2.isLt
  have ht := t.2.isLt
  obtain ⟨tf, tr⟩ := t
  simp only [path, List.mem_filterMap, List.mem_range]
  cases c with
  | white =>
      simp only [pushes, ahead, reduceCtorEq, IsEmpty.forall_iff, and_true, true_implies]
      constructor
      · rintro ⟨i, hi, h⟩
        split_ifs at h with hb
        simp only [Option.some.injEq, Prod.mk.injEq, Fin.ext_iff] at h
        obtain ⟨h1, h2⟩ := h
        exact ⟨Fin.ext h1.symm, by omega⟩
      · rintro ⟨hf, hr⟩
        refine ⟨tr.val - x.2.val - 1, by omega, ?_⟩
        have hb : x.2.val + (tr.val - x.2.val - 1 + 1) < 8 := by omega
        rw [dif_pos hb]
        congr 1
        exact Prod.ext hf.symm (Fin.ext (by simp only; omega))
  | black =>
      simp only [pushes, ahead, reduceCtorEq, IsEmpty.forall_iff, true_and, true_implies]
      constructor
      · rintro ⟨i, hi, h⟩
        split_ifs at h with hb
        simp only [Option.some.injEq, Prod.mk.injEq, Fin.ext_iff] at h
        obtain ⟨h1, h2⟩ := h
        exact ⟨Fin.ext h1.symm, by omega⟩
      · rintro ⟨hf, hr⟩
        refine ⟨x.2.val - tr.val - 1, by omega, ?_⟩
        have hb : x.2.val - tr.val - 1 + 1 ≤ x.2.val := by omega
        rw [dif_pos hb]
        congr 1
        exact Prod.ext hf.symm (Fin.ext (by simp only; omega))

theorem pawnMoveGeometryB_ahead {c : Color} {x y : Square} (hy : ahead c x 1 = some y) :
    pawnMoveGeometryB c x y = true := by
  obtain ⟨hf, hw, hb⟩ := ahead_one_vals hy
  cases c
  · have := hw rfl; simp [pawnMoveGeometryB, hf, this]
  · have := hb rfl; simp [pawnMoveGeometryB, hf]; omega

theorem self_not_mem_path {c : Color} {x : Square} : x ∉ path c x := by
  rw [mem_path_iff]
  rintro ⟨_, hw, hb⟩
  cases c
  · exact absurd (hw rfl) (lt_irrefl _)
  · exact absurd (hb rfl) (lt_irrefl _)

/-- The distance of a square on the pawn's file to the promotion square is its number of pushes. -/
theorem kdist_promo (c : Color) (x : Square) : kdist x (promoSquare c x) = pushes c x := by
  have hx := x.2.isLt
  cases c <;> simp only [kdist, promoSquare, absDiff, pushes, promotionRank] <;> split_ifs <;> omega

theorem promoSquare_of_file {c : Color} {x y : Square} (h : y.1 = x.1) :
    promoSquare c y = promoSquare c x := by
  simp [promoSquare, h]

/-- A square on the path is nearer the promotion square than the pawn. -/
theorem kdist_of_mem_path {c : Color} {x t : Square} (h : t ∈ path c x) :
    kdist t (promoSquare c x) < pushes c x := by
  obtain ⟨hf, hw, hb⟩ := mem_path_iff.mp h
  rw [← promoSquare_of_file hf, kdist_promo]
  have hx := x.2.isLt
  have ht := t.2.isLt
  cases c
  · have := hw rfl; simp only [pushes]; omega
  · have := hb rfl; simp only [pushes]; omega

/-- One step ahead is one push nearer. -/
theorem pushes_ahead {c : Color} {x y : Square} (hy : ahead c x 1 = some y) : pushes c y + 1 = pushes c x := by
  obtain ⟨_, hw, hb⟩ := ahead_one_vals hy
  have hx := x.2.isLt
  cases c
  · have := hw rfl; simp only [pushes]; omega
  · have := hb rfl; simp only [pushes]; omega

theorem ahead_mem_path {c : Color} {x y : Square} (hy : ahead c x 1 = some y) : y ∈ path c x := by
  obtain ⟨hf, hw, hb⟩ := ahead_one_vals hy
  rw [mem_path_iff]
  refine ⟨hf, fun hc => ?_, fun hc => ?_⟩
  · have := hw hc; omega
  · have := hb hc; omega

/-- Ahead of one step ahead is ahead. -/
theorem path_of_ahead {c : Color} {x y t : Square} (hy : ahead c x 1 = some y) (ht : t ∈ path c y) :
    t ∈ path c x ∧ t ≠ y := by
  obtain ⟨hyf, hyw, hyb⟩ := ahead_one_vals hy
  obtain ⟨hf, hw, hb⟩ := mem_path_iff.mp ht
  constructor
  · rw [mem_path_iff]
    refine ⟨hf.trans hyf, fun hc => ?_, fun hc => ?_⟩
    · have := hw hc; have := hyw hc; omega
    · have := hb hc; have := hyb hc; omega
  · rintro rfl
    cases c
    · exact absurd (hw rfl) (lt_irrefl _)
    · exact absurd (hb rfl) (lt_irrefl _)

theorem ahead_not_promo_of_two {c : Color} {x y : Square} (h : 2 ≤ pushes c x) (hy : ahead c x 1 = some y) :
    y.2 ≠ promotionRank c := by
  obtain ⟨_, hw, hb⟩ := ahead_one_vals hy
  have hx := x.2.isLt
  intro he
  cases c
  · have := hw rfl; simp only [pushes] at h; rw [he] at this; simp [promotionRank] at this; omega
  · have := hb rfl; simp only [pushes] at h; rw [he] at this; simp [promotionRank] at this; omega

theorem ahead_promo_of_one {c : Color} {x y : Square} (h : pushes c x = 1) (hy : ahead c x 1 = some y) :
    y.2 = promotionRank c := by
  obtain ⟨_, hw, hb⟩ := ahead_one_vals hy
  have hx := x.2.isLt
  have hy2 := y.2.isLt
  cases c
  · have := hw rfl; simp only [pushes] at h; exact Fin.ext (by simp [promotionRank]; omega)
  · have := hb rfl; simp only [pushes] at h; exact Fin.ext (by simp [promotionRank]; omega)

theorem ahead_notFirst {c : Color} {x y : Square} (hy : ahead c x 1 = some y) :
    y.2 ≠ promotionRank (Color.opposite c) := by
  obtain ⟨_, hw, hb⟩ := ahead_one_vals hy
  have hx := x.2.isLt
  intro he
  cases c
  · have := hw rfl; rw [he] at this; simp [promotionRank, Color.opposite] at this
  · have := hb rfl; rw [he] at this; simp [promotionRank, Color.opposite] at this; omega

/-- A pawn standing on the path attacks only squares nearer the promotion square than the pawn behind
it was. -/
theorem pawn_attack_near {c : Color} {x z a : Square} (hz : z ∈ path c x)
    (ha : pawnAttackGeometryB c z a = true) : kdist a (promoSquare c x) < pushes c x := by
  obtain ⟨hf, hw, hb⟩ := mem_path_iff.mp hz
  have hx := x.2.isLt
  have hz2 := z.2.isLt
  have ha2 := a.2.isLt
  cases c
  · have := hw rfl
    simp only [pawnAttackGeometryB, absDiff, Bool.and_eq_true, beq_iff_eq, if_true] at ha
    simp only [kdist, promoSquare, absDiff, pushes, promotionRank]
    rw [Fin.ext_iff] at hf
    split_ifs at ha ⊢ <;> omega
  · have := hb rfl
    simp only [pawnAttackGeometryB, absDiff, Bool.and_eq_true, beq_iff_eq, reduceCtorEq, if_false] at ha
    simp only [kdist, promoSquare, absDiff, pushes, promotionRank]
    rw [Fin.ext_iff] at hf
    split_ifs at ha ⊢ <;> omega

/-! ### Kings and attacks when the enemy has only a king -/

/-- `kingSquare` depends only on where the colour's kings stand. -/
theorem kingSquare_congr {b b' : Board} {c : Color}
    (h : ∀ s, b s = some ⟨c, .king⟩ ↔ b' s = some ⟨c, .king⟩) : kingSquare b c = kingSquare b' c := by
  unfold kingSquare
  congr 1
  funext s
  simp only [h s]

/-- With one enemy piece, its king on `ek`, "attacked by the enemy" is "next to `ek`". -/
theorem bare_attack {q : Position} {e : Color} {ek t : Square}
    (hbare : ∀ s pc, q.board s = some pc → pc.color = e → s = ek ∧ pc.kind = .king)
    (hk : q.board ek = some ⟨e, .king⟩) : isAttackedB q e t = kingGeometryB ek t := by
  apply Bool.eq_iff_iff.mpr
  simp only [isAttackedB, List.any_eq_true]
  constructor
  · rintro ⟨s, _, hs⟩
    unfold pieceAttacksB at hs
    split at hs
    · rename_i pc hpc
      simp only [Bool.and_eq_true, beq_iff_eq] at hs
      obtain ⟨rfl, hkind⟩ := hbare s pc hpc hs.1
      obtain ⟨col, kind⟩ := pc
      simp only at hkind
      subst hkind
      simp only [Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, reduceCtorEq, false_and, false_or,
        or_false, true_and] at hs
      exact hs.2
    · simp at hs
  · intro hg
    refine ⟨ek, mem_allSquares ek, ?_⟩
    unfold pieceAttacksB
    simp [hk, hg]

/-- Attacks by a side without sliders do not depend on the rest of the board: only the attacker's
square and kind matter. -/
theorem noslider_attack {q q' : Position} {c : Color} {t : Square}
    (hns : ∀ s pc, q.board s = some pc → pc.color = c → pc.kind = .king ∨ pc.kind = .pawn ∨ pc.kind = .knight)
    (hsame : ∀ s pc, pc.color = c → (q.board s = some pc ↔ q'.board s = some pc)) :
    isAttackedB q c t = isAttackedB q' c t := by
  apply Bool.eq_iff_iff.mpr
  simp only [isAttackedB, List.any_eq_true]
  have key : ∀ s, pieceAttacksB q c s t = true ↔ pieceAttacksB q' c s t = true := by
    intro s
    constructor
    · intro hs
      unfold pieceAttacksB at hs ⊢
      split at hs
      · rename_i pc hpc
        simp only [Bool.and_eq_true, beq_iff_eq] at hs
        have hcol := hs.1
        rw [(hsame s pc hcol).1 hpc]
        rcases hns s pc hpc hcol with hk | hk | hk <;> simp_all
      · simp at hs
    · intro hs
      unfold pieceAttacksB at hs ⊢
      split at hs
      · rename_i pc hpc
        simp only [Bool.and_eq_true, beq_iff_eq] at hs
        have hcol := hs.1
        have hq := (hsame s pc hcol).2 hpc
        rw [hq]
        rcases hns s pc hq hcol with hk | hk | hk <;> simp_all
      · simp at hs
  constructor
  · rintro ⟨s, hs, h⟩; exact ⟨s, hs, (key s).1 h⟩
  · rintro ⟨s, hs, h⟩; exact ⟨s, hs, (key s).2 h⟩

/-! ### The push -/

/-- Our pawn's next move, with the square it goes to. -/
theorem push_eq {c : Color} {x y : Square} (hy : ahead c x 1 = some y) :
    push c x = some (if y.2 = promotionRank c then .promotion x y .queen else .normal x y) := by
  simp [push, hy]

/-- Our push exists and is legal. -/
theorem push_legal {c : Color} {p : Position} {x : Square} {r : Nat} {ek : Square}
    (h : Inv c p x r ek) : ∃ m, push c x = some m ∧ legalMove p m := by
  have h1 : 1 ≤ pushes c x := h.pushesEq ▸ h.pos
  obtain ⟨y, hy, hyf, hyw, hyb⟩ := ahead_one h1
  have hy_empty : p.board y = none := h.clear y (ahead_mem_path hy)
  have hgeo := pawnMoveGeometryB_ahead hy
  have hpawn := h.pawn
  have hek := h.eking
  have hekx : ek ≠ x := by intro e; rw [e, hpawn] at hek; simp at hek
  have heky : ek ≠ y := by intro e; rw [e, hy_empty] at hek; cases hek
  -- our king is not attacked after the pawn moves: the only enemy piece, the king, did not move
  have safe : ∀ q : Position, (∀ s, p.board s = some ⟨c, .king⟩ ↔ q.board s = some ⟨c, .king⟩) →
      q.board ek = some ⟨Color.opposite c, .king⟩ →
      (∀ s pc, q.board s = some pc → pc.color = Color.opposite c → s = ek ∧ pc.kind = .king) →
      inCheckB q c = false := by
    intro q hkings hqek hbare'
    have hus := h.usSafe
    unfold inCheck inCheckB at hus
    unfold inCheckB
    rw [← kingSquare_congr hkings]
    cases hks : kingSquare p.board c with
    | none => rw [hks] at hus; simp at hus
    | some ok =>
        rw [hks] at hus
        simp only [Bool.not_eq_true] at hus ⊢
        rw [bare_attack hbare' hqek]
        rw [bare_attack h.bare h.eking] at hus
        exact hus
  have hne : ∀ d : Color, d ≠ Color.opposite d := by intro d; cases d <;> decide
  -- the board after the pawn moves to `y` carrying piece `v` (a pawn, or the new queen)
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
  have ek_same : ∀ v : Piece, boardUpdate (boardUpdate p.board x none) y (some v) ek =
      some ⟨Color.opposite c, .king⟩ := by
    intro v
    simp [boardUpdate, Function.update_apply, hekx, heky, hek]
  have bare_same : ∀ v : Piece, v.color = c →
      ∀ s pc, boardUpdate (boardUpdate p.board x none) y (some v) s = some pc →
        pc.color = Color.opposite c → s = ek ∧ pc.kind = .king := by
    intro v hv s pc hs hcol
    simp only [boardUpdate, Function.update_apply] at hs
    by_cases hsy : s = y
    · rw [if_pos hsy] at hs
      cases hs
      rw [hv] at hcol
      exact absurd hcol (hne c)
    · rw [if_neg hsy] at hs
      by_cases hsx : s = x
      · rw [if_pos hsx] at hs; cases hs
      · rw [if_neg hsx] at hs
        exact h.bare s pc hs hcol
  refine ⟨_, push_eq hy, ?_⟩
  rw [legalMove_iff_legalMoveB]
  split_ifs with hpr
  · -- the promotion
    have hps : pseudoLegalB p (.promotion x y .queen) = true := by
      simp [pseudoLegalB, pseudoLegalPromotionB, hpawn, hpr, h.turn, hgeo, hy_empty]
    have hboard : (applyMove p (.promotion x y .queen)).board =
        boardUpdate (boardUpdate p.board x none) y (some ⟨c, .queen⟩) := by
      rw [← h.turn]; rfl
    have hnc : inCheckB (applyMove p (.promotion x y .queen)) p.turn = false := by
      rw [h.turn]
      apply safe
      · intro s; rw [hboard]; exact kings_same ⟨c, .queen⟩ (by simp) s
      · rw [hboard]; exact ek_same _
      · rw [hboard]; exact bare_same ⟨c, .queen⟩ rfl
    simp [legalMoveB, Move.isCastle, hps, hnc]
  · -- the ordinary push
    have hps : pseudoLegalB p (.normal x y) = true := by
      simp [pseudoLegalB, normalPseudoLegalB, hpawn, h.turn, destinationFriendlyB, hy_empty,
        normalPieceGeometryB, hgeo, hpr]
    have hboard : (applyMove p (.normal x y)).board =
        boardUpdate (boardUpdate p.board x none) y (some ⟨c, .pawn⟩) := by
      simp [applyMove, applyBoard, boardMove, hpawn]
    have hnc : inCheckB (applyMove p (.normal x y)) p.turn = false := by
      rw [h.turn]
      apply safe
      · intro s; rw [hboard]; exact kings_same ⟨c, .pawn⟩ (by simp) s
      · rw [hboard]; exact ek_same _
      · rw [hboard]; exact bare_same ⟨c, .pawn⟩ rfl
    simp [legalMoveB, Move.isCastle, hps, hnc]

/-- What our push does to the board, with two or more pushes to go: the pawn steps to `y`. -/
theorem push_board_two {c : Color} {p : Position} {x : Square} {r : Nat} {ek : Square}
    (h : Inv c p x (r + 2) ek) {m : Move} (hm : push c x = some m) :
    ∃ y, ahead c x 1 = some y ∧ m = .normal x y ∧ p.board y = none ∧
      (applyMove p m).board = boardUpdate (boardUpdate p.board x none) y (some ⟨c, .pawn⟩) := by
  have h1 : 1 ≤ pushes c x := h.pushesEq ▸ (by omega)
  obtain ⟨y, hy, -, -, -⟩ := ahead_one h1
  have hnp : y.2 ≠ promotionRank c := ahead_not_promo_of_two (h.pushesEq ▸ (by omega)) hy
  rw [push_eq hy, if_neg hnp, Option.some.injEq] at hm
  subst hm
  refine ⟨y, hy, rfl, h.clear y (ahead_mem_path hy), ?_⟩
  simp [applyMove, applyBoard, boardMove, h.pawn]

/-- The goal after the last push. -/
theorem promoted_after_last {c : Color} {p : Position} {x : Square} {ek : Square} (h : Inv c p x 1 ek)
    {m : Move} (hm : push c x = some m) : promoted c x.1 (applyMove p m) = true := by
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

/-- The enemy's king and pieces are untouched by our push. -/
theorem push_keeps_enemy {c : Color} {p : Position} {x y : Square} {ek : Square} {v : Piece}
    (hbare : ∀ s pc, p.board s = some pc → pc.color = Color.opposite c → s = ek ∧ pc.kind = .king)
    (hek : p.board ek = some ⟨Color.opposite c, .king⟩) (hpawn : p.board x = some ⟨c, .pawn⟩)
    (hy : p.board y = none) (hv : v.color = c) :
    (∀ s pc, boardUpdate (boardUpdate p.board x none) y (some v) s = some pc →
        pc.color = Color.opposite c → s = ek ∧ pc.kind = .king) ∧
      boardUpdate (boardUpdate p.board x none) y (some v) ek = some ⟨Color.opposite c, .king⟩ := by
  have hne : ∀ d : Color, d ≠ Color.opposite d := by intro d; cases d <;> decide
  have hekx : ek ≠ x := by intro e; rw [e, hpawn] at hek; simp at hek
  have heky : ek ≠ y := by intro e; rw [e, hy] at hek; cases hek
  constructor
  · intro s pc hs hcol
    simp only [boardUpdate, Function.update_apply] at hs
    by_cases hsy : s = y
    · rw [if_pos hsy] at hs; cases hs; rw [hv] at hcol; exact absurd hcol (hne c)
    · rw [if_neg hsy] at hs
      by_cases hsx : s = x
      · rw [if_pos hsx] at hs; cases hs
      · rw [if_neg hsx] at hs; exact hbare s pc hs hcol
  · simp [boardUpdate, Function.update_apply, hekx, heky, hek]

/-- After our push (more than one to go) the enemy has a legal move: the king steps to the free square. -/
theorem enemy_has_move {c : Color} {p : Position} {x : Square} {r : Nat} {ek : Square}
    (h : Inv c p x (r + 2) ek) {m : Move} (hm : push c x = some m) :
    (legalMoves (applyMove p m)) ≠ [] := by
  obtain ⟨y, hy, rfl, hy_empty, hboard⟩ := push_board_two h hm
  obtain ⟨d, hgeo, hd, hatt⟩ := h.free (by omega) _ hm
  set q := applyMove p (.normal x y) with hq
  have hqturn : q.turn = Color.opposite c := by rw [hq, applyMove_turn, h.turn]
  obtain ⟨hbare_q, hek_q⟩ := push_keeps_enemy h.bare h.eking h.pawn hy_empty (v := ⟨c, .pawn⟩) rfl
  rw [← hboard] at hbare_q hek_q
  -- the king step ek -> d is legal in q
  have hleg : legalMoveB q (.normal ek d) = true := by
    have hps : pseudoLegalB q (.normal ek d) = true := by
      simp [pseudoLegalB, normalPseudoLegalB, hek_q, hqturn, destinationFriendlyB, hd,
        normalPieceGeometryB, hgeo]
    have hnc : inCheckB (applyMove q (.normal ek d)) q.turn = false := by
      -- the enemy king now stands alone on d; our pieces (no sliders) attack d exactly as before
      have hboard' : (applyMove q (.normal ek d)).board = boardMove q.board ek d := rfl
      have hekd : ek ≠ d := by intro e; rw [e, hd] at hek_q; cases hek_q
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
            exact hse (hbare_q s _ hs rfl).1
      unfold inCheckB
      rw [kingSquare_of_only honly, hqturn, Color.opposite_opposite]
      show isAttackedB (applyMove q (.normal ek d)) c d = false
      -- our pieces are the same in q and after the king's step
      have hns : ∀ s pc, q.board s = some pc → pc.color = c →
          pc.kind = .king ∨ pc.kind = .pawn ∨ pc.kind = .knight := by
        intro s pc hs hcol
        rw [hboard] at hs
        simp only [boardUpdate, Function.update_apply] at hs
        by_cases hsy : s = y
        · rw [if_pos hsy] at hs; cases hs; simp
        · rw [if_neg hsy] at hs
          by_cases hsx : s = x
          · rw [if_pos hsx] at hs; cases hs
          · rw [if_neg hsx] at hs; exact h.noSliders s pc hs hcol
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
      have := hatt
      unfold isAttacked at this
      simpa using this
    simp [legalMoveB, Move.isCastle, hps, hnc]
  intro hnil
  have := legalMoves_complete ((legalMove_iff_legalMoveB q _).2 hleg)
  rw [hnil] at this
  exact absurd this (List.not_mem_nil)

/-- A piece of a side without sliders attacks by geometry alone: the board around it does not matter. -/
theorem noslider_pieceAttacks {q q' : Position} {c : Color} {s t : Square} {pc : Piece}
    (hk : pc.kind = .king ∨ pc.kind = .pawn ∨ pc.kind = .knight)
    (hs : q.board s = some pc) (hs' : q'.board s = some pc) :
    pieceAttacksB q c s t = pieceAttacksB q' c s t := by
  obtain ⟨col, kind⟩ := pc
  simp only at hk
  unfold pieceAttacksB
  rw [hs, hs']
  rcases hk with hk | hk | hk <;> subst hk <;> rfl

/-- A king attacks by king geometry. -/
theorem king_attack {q : Position} {c : Color} {s t : Square} (hs : q.board s = some ⟨c, .king⟩) :
    pieceAttacksB q c s t = kingGeometryB s t := by
  apply Bool.eq_iff_iff.mpr
  unfold pieceAttacksB; rw [hs]
  simp

/-- A pawn attacks by pawn geometry. -/
theorem pawn_attack {q : Position} {c : Color} {s t : Square} (hs : q.board s = some ⟨c, .pawn⟩) :
    pieceAttacksB q c s t = pawnAttackGeometryB c s t := by
  apply Bool.eq_iff_iff.mpr
  unfold pieceAttacksB; rw [hs]
  simp

/-- After our push and any legal enemy reply, the invariant holds one push further. -/
theorem inv_step {c : Color} {p : Position} {x : Square} {r : Nat} {ek : Square}
    (h : Inv c p x (r + 2) ek) {m : Move} (hm : push c x = some m) {e : Move}
    (he : legalMove (applyMove p m) e) :
    ∃ x' ek', ahead c x 1 = some x' ∧ Inv c (applyMove (applyMove p m) e) x' (r + 1) ek' := by
  obtain ⟨y, hy, rfl, hy_empty, hboard⟩ := push_board_two h hm
  set q := applyMove p (.normal x y) with hq
  have hqturn : q.turn = Color.opposite c := by rw [hq, applyMove_turn, h.turn]
  obtain ⟨hbare_q, hek_q⟩ := push_keeps_enemy h.bare h.eking h.pawn hy_empty (v := ⟨c, .pawn⟩) rfl
  rw [← hboard] at hbare_q hek_q
  have hbare_q' : ∀ s pc, q.board s = some pc → pc.color = q.turn → s = ek ∧ pc.kind = .king := by
    rw [hqturn]; exact hbare_q
  -- the enemy's move is a king step ek -> d
  obtain ⟨d, rfl, hgeo, hfr, hnc⟩ := bare_king_move hbare_q' he
  set q' := applyMove q (.normal ek d) with hq'
  have hboard' : q'.board = boardMove q.board ek d := rfl
  have hekd : ek ≠ d := by
    intro e; rw [e] at hgeo; simp [kingGeometryB] at hgeo
  have hne : ∀ d : Color, d ≠ Color.opposite d := by intro d; cases d <;> decide
  -- distances
  have hout := h.outside
  have hstep := kdist_king_step hgeo
  have htri := kdist_triangle ek d (promoSquare c x)
  have hdist_d : r + 1 < kdist d (promoSquare c x) := by omega
  have hyf := ahead_file hy
  have hpromo_y : promoSquare c y = promoSquare c x := promoSquare_of_file hyf
  have hpush_y : pushes c y = r + 1 := by have := pushes_ahead hy; rw [h.pushesEq] at this; omega
  have hy_near : kdist y (promoSquare c x) = r + 1 := by rw [← hpromo_y, kdist_promo, hpush_y]
  have hyd : y ≠ d := by intro e; rw [e] at hy_near; omega
  have hyek : y ≠ ek := by intro e; rw [e] at hy_near; omega
  have hxy : x ≠ y := by intro e; rw [← e, h.pawn] at hy_empty; cases hy_empty
  have hxek : ek ≠ x := by intro e; have := h.eking; rw [e, h.pawn] at this; simp at this
  -- the boards
  have hqother : ∀ s, s ≠ y → s ≠ x → q.board s = p.board s := by
    intro s hsy hsx; rw [hboard]; simp [boardUpdate, Function.update_apply, hsy, hsx]
  have hqy : q.board y = some ⟨c, .pawn⟩ := by rw [hboard]; simp [boardUpdate]
  have hqx : q.board x = none := by rw [hboard]; simp [boardUpdate, Function.update_apply, hxy]
  have hq'other : ∀ s, s ≠ d → s ≠ ek → q'.board s = q.board s := by
    intro s hsd hse; rw [hboard']; simp [boardMove, boardUpdate, Function.update_apply, hsd, hse]
  have hq'y : q'.board y = some ⟨c, .pawn⟩ := by rw [hq'other y hyd hyek]; exact hqy
  have hq'd : q'.board d = some ⟨Color.opposite c, .king⟩ := by
    rw [hboard']; simp [boardMove, boardUpdate, hek_q]
  have hq'ek : q'.board ek = none := by
    rw [hboard']; simp [boardMove, boardUpdate, Function.update_apply, hekd]
  -- the enemy king in q': alone on d
  have hbare' : ∀ s pc, q'.board s = some pc → pc.color = Color.opposite c → s = d ∧ pc.kind = .king := by
    intro s pc hs hcol
    by_cases hsd : s = d
    · subst hsd; rw [hq'd] at hs; cases hs; exact ⟨rfl, rfl⟩
    · by_cases hse : s = ek
      · subst hse; rw [hq'ek] at hs; cases hs
      · rw [hq'other s hsd hse] at hs; exact absurd (hbare_q s pc hs hcol).1 hse
  -- our kings: unchanged from p (the enemy cannot take a king)
  have hd_not_our_king : p.board d ≠ some ⟨c, .king⟩ := by
    intro hpd
    by_cases hdx : d = x
    · rw [hdx, h.pawn] at hpd; simp at hpd
    · have hqd : q.board d = some ⟨c, .king⟩ := by rw [hqother d (Ne.symm hyd) hdx]; exact hpd
      simp [destinationFriendlyB, hqd] at hfr
  have hk : ∀ s, p.board s = some ⟨c, .king⟩ ↔ q'.board s = some ⟨c, .king⟩ := by
    intro s
    by_cases hsd : s = d
    · subst hsd
      rw [hq'd]
      constructor
      · intro hs; exact absurd hs hd_not_our_king
      · intro hs; simp at hs; exact absurd hs (by cases c <;> decide)
    · by_cases hse : s = ek
      · subst hse
        rw [hq'ek, h.eking]
        constructor
        · intro hs; simp at hs; exact absurd hs (by cases c <;> decide)
        · intro hs; cases hs
      · rw [hq'other s hsd hse]
        by_cases hsy : s = y
        · subst hsy; rw [hqy, hy_empty]; constructor <;> intro hs <;> cases hs
        · by_cases hsx : s = x
          · subst hsx; rw [hqx, h.pawn]; constructor <;> intro hs <;> cases hs
          · rw [hqother s hsy hsx]
  have hks : kingSquare q'.board c = kingSquare p.board c := (kingSquare_congr hk).symm
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
        have htd : t ≠ d := by intro e; subst e; omega
        have htek : t ≠ ek := by intro e; subst e; omega
        have htxne : t ≠ x := by intro e; subst e; exact self_not_mem_path htx
        rw [hq'other t htd htek, hqother t hty htxne]
        exact h.clear t htx
      bare := hbare'
      eking := hq'd
      outside := by rw [hpromo_y]; exact hdist_d
      noSliders := by
        intro s pc hs hcol
        by_cases hsd : s = d
        · subst hsd; rw [hq'd] at hs; obtain rfl := Option.some.inj hs
          exact absurd hcol (by cases c <;> decide)
        · by_cases hse : s = ek
          · subst hse; rw [hq'ek] at hs; cases hs
          · rw [hq'other s hsd hse] at hs
            by_cases hsy : s = y
            · subst hsy; rw [hqy] at hs; obtain rfl := Option.some.inj hs; simp
            · by_cases hsx : s = x
              · subst hsx; rw [hqx] at hs; cases hs
              · rw [hqother s hsy hsx] at hs; exact h.noSliders s pc hs hcol
      ourKing := by rw [hks]; exact h.ourKing
      usSafe := by
        intro hin
        unfold inCheck inCheckB at hin
        rw [hks] at hin
        have hour := h.ourKing
        cases hksq : kingSquare p.board c with
        | none => rw [hksq] at hour; simp at hour
        | some ok =>
            rw [hksq] at hin
            change isAttackedB q' (Color.opposite c) ok = true at hin
            rw [bare_attack hbare' hq'd] at hin
            -- the enemy king on d is not attacked by our king on ok (its move was legal)
            have hok : q'.board ok = some ⟨c, .king⟩ := (hk ok).1 (kingSquare_some hksq).2
            have honly : ∀ s, q'.board s = some ⟨Color.opposite c, .king⟩ ↔ s = d := by
              intro s
              constructor
              · intro hs; exact (hbare' s _ hs rfl).1
              · intro hs; subst hs; exact hq'd
            unfold inCheck inCheckB at hnc
            rw [hqturn, kingSquare_of_only honly, Color.opposite_opposite] at hnc
            simp only [isAttackedB, List.any_eq_true, not_exists, not_and, Bool.not_eq_true] at hnc
            have := hnc ok (mem_allSquares ok)
            rw [king_attack hok, kingGeometryB_comm] at this
            rw [this] at hin
            cases hin
      themSafe := by rw [← hqturn]; exact hnc
      free := by
        intro _ m' hm'
        have h2 : 2 ≤ pushes c y := by omega
        obtain ⟨y2, hy2, -, -, -⟩ := ahead_one (by omega : 1 ≤ pushes c y)
        have hnp : y2.2 ≠ promotionRank c := ahead_not_promo_of_two h2 hy2
        rw [push_eq hy2, if_neg hnp, Option.some.injEq] at hm'
        subst hm'
        have hy2x : y2 ∈ path c x := (path_of_ahead hy (ahead_mem_path hy2)).1
        have hy2near := kdist_of_mem_path hy2x
        rw [h.pushesEq] at hy2near
        have hy2ek : y2 ≠ ek := by intro e; rw [e] at hy2near; omega
        have hy2d : y2 ≠ d := by intro e; rw [e] at hy2near; omega
        have hboard2 : (applyMove q' (.normal y y2)).board = boardMove q'.board y y2 := rfl
        refine ⟨ek, by rw [kingGeometryB_comm]; exact hgeo, ?_, ?_⟩
        · rw [hboard2]
          simp [boardMove, boardUpdate, Function.update_apply, hy2ek.symm, hyek.symm, hq'ek]
        · intro hatt
          unfold isAttacked at hatt
          simp only [isAttackedB, List.any_eq_true] at hatt
          obtain ⟨s, -, hs⟩ := hatt
          obtain ⟨pc, hpc, hcol⟩ := pieceAttacksB_source hs
          by_cases hs2 : s = y2
          · -- the pawn, now on y2, attacks only squares nearer the promotion square
            subst s
            have hpc' : (applyMove q' (.normal y y2)).board y2 = some ⟨c, .pawn⟩ := by
              rw [hboard2]; simp [boardMove, boardUpdate, hq'y]
            rw [pawn_attack hpc'] at hs
            have := pawn_attack_near hy2x hs
            rw [h.pushesEq] at this
            omega
          · by_cases hsy : s = y
            · subst s
              rw [hboard2] at hpc
              simp [boardMove, boardUpdate, Function.update_apply, hs2] at hpc
            · -- a piece of ours that did not move: it did not attack ek in p either
              have hpc2 : (applyMove q' (.normal y y2)).board s = q'.board s := by
                rw [hboard2]; simp [boardMove, boardUpdate, Function.update_apply, hs2, hsy]
              rw [hpc2] at hpc
              have hsd : s ≠ d := by
                intro e; subst e; rw [hq'd] at hpc; obtain rfl := Option.some.inj hpc
                exact absurd hcol (by cases c <;> decide)
              have hse : s ≠ ek := by intro e; subst e; rw [hq'ek] at hpc; cases hpc
              rw [hq'other s hsd hse] at hpc
              have hsx : s ≠ x := by intro e; subst e; rw [hqx] at hpc; cases hpc
              rw [hqother s hsy hsx] at hpc
              have hkind := h.noSliders s pc hpc hcol
              have hatt_p : pieceAttacksB p c s ek = true := by
                rw [noslider_pieceAttacks hkind hpc (by rw [hpc2, hq'other s hsd hse, hqother s hsy hsx]; exact hpc)]
                exact hs
              have hthem := h.themSafe
              unfold inCheck inCheckB at hthem
              have honly_p : ∀ t, p.board t = some ⟨Color.opposite c, .king⟩ ↔ t = ek := by
                intro t
                constructor
                · intro ht; exact (h.bare t _ ht rfl).1
                · intro ht; subst ht; exact h.eking
              rw [kingSquare_of_only honly_p, Color.opposite_opposite] at hthem
              simp only [isAttackedB, List.any_eq_true, not_exists, not_and, Bool.not_eq_true] at hthem
              have := hthem s (mem_allSquares s)
              rw [hatt_p] at this
              cases this }

theorem forcedWithinB_ours {c : Color} {goal : Position → Bool} {p : Position} {n : Nat}
    (hturn : p.turn = c) :
    forcedWithinB c goal p (n + 1) =
      (goal p || (legalMoves p).any (fun m => forcedWithinB c goal (applyMove p m) n)) := by
  simp [forcedWithinB, hturn]

theorem forcedWithinB_theirs {c : Color} {goal : Position → Bool} {p : Position} {n : Nat}
    (hturn : p.turn ≠ c) :
    forcedWithinB c goal p (n + 1) =
      (goal p || (!(legalMoves p).isEmpty &&
        (legalMoves p).all (fun m => forcedWithinB c goal (applyMove p m) n))) := by
  simp [forcedWithinB, hturn]

theorem after_turn_ne {c : Color} {p : Position} (m : Move) (hturn : p.turn = c) :
    (applyMove p m).turn ≠ c := by
  rw [applyMove_turn, hturn]
  cases c <;> simp [Color.opposite]

/-- The race: with the invariant, we promote within `2r - 1` plies whatever the enemy does. -/
theorem race (c : Color) (f : Fin 8) :
    ∀ r (p : Position) (x : Square) (ek : Square), x.1 = f → Inv c p x r ek →
      forcedWithinB c (promoted c f) p (2 * r - 1) = true := by
  intro r
  induction r with
  | zero => intro p x ek _ h; exact absurd h.pos (by omega)
  | succ r ih =>
    intro p x ek hf h
    obtain ⟨m, hm, hl⟩ := push_legal h
    cases r with
    | zero =>
        -- one push: the promotion
        show forcedWithinB c (promoted c f) p (0 + 1) = true
        rw [forcedWithinB_ours h.turn]
        simp only [Bool.or_eq_true, List.any_eq_true]
        right
        exact ⟨m, legalMoves_complete hl, by subst hf; exact promoted_after_last h hm⟩
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
        refine ⟨enemy_has_move h hm, ?_⟩
        intro e he
        obtain ⟨x', ek', hx', h'⟩ := inv_step h hm (legalMoves_sound he)
        exact ih _ _ _ ((ahead_file hx').trans hf) h'

/-! ## From the Idea's hypotheses to the invariant -/

/-- **Soundness of the unstoppable passer.** When every hypothesis holds, our pawn stands on the promotion
square as a queen within `2n - 1` plies whatever the opponent does. -/
theorem unstoppablePasser_sound' (p : Position) (ps : PasserParams)
    (h : unstoppablePasser.holds p ps = true) :
    forcedWithinB p.turn (promoted p.turn ps.sq.1) p (2 * pushes p.turn ps.sq - 1) = true := by
  simp only [Idea.holds, unstoppablePasser, List.all_cons, List.all_nil, Bool.and_true,
    Bool.and_eq_true] at h
  obtain ⟨hpawn, hclear, hbare, hns, hnc, hout, hfree⟩ := h
  have hne : ∀ d : Color, d ≠ Color.opposite d := by intro d; cases d <;> decide
  -- the pawn
  simp only [ownPawnB, Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at hpawn
  obtain ⟨⟨hpawn, hnotpromo⟩, hnotfirst⟩ := hpawn
  -- the enemy king
  simp only [bareKingB, Bool.and_eq_true, List.all_eq_true] at hbare
  obtain ⟨hall, hone⟩ := hbare
  cases hek : kingSquare p.board (Color.opposite p.turn) with
  | none => rw [hek] at hone; simp at hone
  | some ek =>
    rw [hek] at hone
    simp only [List.all_eq_true] at hone
    have hek_board : p.board ek = some ⟨Color.opposite p.turn, .king⟩ := (kingSquare_some hek).2
    have hbare' : ∀ s pc, p.board s = some pc → pc.color = Color.opposite p.turn →
        s = ek ∧ pc.kind = .king := by
      intro s pc hs hcol
      have h1 := hall s (mem_allSquares s)
      rw [hs] at h1
      simp only [Bool.or_eq_true, beq_iff_eq] at h1
      have hkind : pc.kind = .king := by
        rcases h1 with h1 | h1
        · rw [hcol] at h1; exact absurd h1.symm (hne _)
        · exact h1
      have h2 := hone s (mem_allSquares s)
      have hpc : pc = ⟨Color.opposite p.turn, .king⟩ := by
        obtain ⟨col, kind⟩ := pc; simp only at hcol hkind; rw [hcol, hkind]
      rw [hs, hpc] at h2
      simp only [beq_self_eq_true, beq_iff_eq] at h2
      exact ⟨of_decide_eq_true h2.symm, hkind⟩
    -- the rest
    simp only [notInCheckB, Bool.and_eq_true, Bool.not_eq_true', inCheckB'_eq] at hnc
    obtain ⟨⟨hour, hus⟩, hthem⟩ := hnc
    simp only [kingOutsideB, hek, decide_eq_true_eq] at hout
    have hpos : 1 ≤ pushes p.turn ps.sq := pushes_pos_of_not_promo hnotpromo
    apply race p.turn ps.sq.1 (pushes p.turn ps.sq) p ps.sq ek rfl
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
        bare := hbare'
        eking := hek_board
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
        ourKing := hour
        usSafe := by unfold inCheck; rw [hus]; exact Bool.false_ne_true
        themSafe := by unfold inCheck; rw [hthem]; exact Bool.false_ne_true
        free := by
          intro _ m hm
          simp only [kingHasMoveB, hm, hek, List.any_eq_true, Bool.and_eq_true, beq_iff_eq,
            Bool.not_eq_true', isAttackedB'_eq] at hfree
          obtain ⟨d, -, ⟨hg, hd⟩, hatt⟩ := hfree
          exact ⟨d, hg, hd, by unfold isAttacked; rw [hatt]; exact Bool.false_ne_true⟩ }

end Passer

end Ideas
