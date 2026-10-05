import Chess.Legality

/-!
# Fast primitives, proved equal to the kernel's

The kernel's definitions are transparent and slow: `squaresBetween` filters all 64 squares, `isAttackedB`
tries all 64 squares as attackers, and every slider test calls `squaresBetween`. Here each gets a fast
version -- a ray walk, a per-square table of the squares that can attack at all -- with a theorem that it
EQUALS the kernel's definition. Lemma hypotheses use the fast versions; their soundness theorems rewrite
to the kernel's and apply the proofs there. Nothing is trusted that is not proved equal.
-/

namespace Ideas

open Chess

/-! ## Squares between, by walking the ray -/

def dir (a b : Fin 8) : Int := if a.val < b.val then 1 else if b.val < a.val then -1 else 0

/-- `s` before `t` in the order of `allSquares` (file first, then rank). -/
def lexLt (s t : Square) : Bool := s.1.val < t.1.val || (s.1.val = t.1.val && s.2.val < t.2.val)

/-- The squares strictly between `s` and `t`, walking the ray from the lexicographically smaller end so
the list comes out in the order of `allSquares` (empty unless the squares are aligned). -/
def rayBetween (s t : Square) : List Square :=
  if rookGeometryB s t || bishopGeometryB s t then
    let (a, b) := if lexLt s t then (s, t) else (t, s)
    let df := dir a.1 b.1
    let dr := dir a.2 b.2
    let n := max (absDiff a.1 b.1) (absDiff a.2 b.2)
    (List.range (n - 1)).filterMap fun i =>
      let f : Int := a.1.val + df * (i + 1 : Int)
      let r : Int := a.2.val + dr * (i + 1 : Int)
      if h : 0 ≤ f ∧ f < 8 ∧ 0 ≤ r ∧ r < 8 then
        some (⟨f.toNat, by omega⟩, ⟨r.toNat, by omega⟩)
      else none
  else []

-- The kernel checks all 4,096 pairs (about 80 s; the module is rebuilt only when this file changes).
set_option maxHeartbeats 40000000 in
set_option maxRecDepth 200000 in
theorem rayBetween_eq : ∀ s t : Square, rayBetween s t = squaresBetween s t := by
  decide

def clearBetweenB' (b : Board) (s t : Square) : Bool :=
  (rayBetween s t).all (fun u => b u == none)

theorem clearBetweenB'_eq (b : Board) (s t : Square) : clearBetweenB' b s t = clearBetweenB b s t := by
  simp only [clearBetweenB', clearBetweenB, rayBetween_eq]

/-! ## Attacks -/

def sliderAttackB' (p : Position) (s t : Square) : Bool := clearBetweenB' p.board s t

theorem sliderAttackB'_eq (p : Position) (s t : Square) : sliderAttackB' p s t = sliderAttackB p s t := by
  simp only [sliderAttackB', sliderAttackB, clearBetweenB'_eq]

def pieceAttacksB' (p : Position) (c : Color) (s t : Square) : Bool :=
  match p.board s with
  | some q =>
      (q.color == c) &&
        ((q.kind == .rook && rookGeometryB s t && sliderAttackB' p s t) ||
          (q.kind == .bishop && bishopGeometryB s t && sliderAttackB' p s t) ||
          (q.kind == .queen && queenGeometryB s t && sliderAttackB' p s t) ||
          (q.kind == .knight && knightGeometryB s t) ||
          (q.kind == .king && kingGeometryB s t) ||
          (q.kind == .pawn && pawnAttackGeometryB c s t))
  | none => false

theorem pieceAttacksB'_eq (p : Position) (c : Color) (s t : Square) :
    pieceAttacksB' p c s t = pieceAttacksB p c s t := by
  unfold pieceAttacksB' pieceAttacksB
  cases p.board s <;> simp [sliderAttackB'_eq]

-- Any piece attacking `t` stands on a queen line or a knight jump from `t`.
set_option maxRecDepth 100000 in
theorem geometry_of_attack : ∀ s t : Square,
    (kingGeometryB s t || pawnAttackGeometryB .white s t || pawnAttackGeometryB .black s t) = true →
      (queenGeometryB s t || knightGeometryB s t) = true := by
  decide

/-- The squares from which `t` can be attacked at all: the queen lines and the knight jumps. -/
def attackerSquares (t : Square) : List Square :=
  allSquares.filter (fun s => queenGeometryB s t || knightGeometryB s t)

def sqIndex (s : Square) : Fin 64 := ⟨s.1.val * 8 + s.2.val, by have := s.1.isLt; have := s.2.isLt; omega⟩

/-- The table, computed once at start-up (a closed term). -/
def attackerTable : Array (List Square) :=
  Array.ofFn (fun i : Fin 64 => attackerSquares (sqOfIndexF i))
where
  sqOfIndexF (i : Fin 64) : Square := (⟨i.val / 8, by omega⟩, ⟨i.val % 8, by omega⟩)

theorem attackerTable_size : attackerTable.size = 64 := by simp [attackerTable]

theorem attackerTable_get (t : Square) :
    attackerTable[(sqIndex t).val]'(by rw [attackerTable_size]; exact (sqIndex t).isLt) = attackerSquares t := by
  simp only [attackerTable, Array.getElem_ofFn]
  congr 1
  obtain ⟨f, r⟩ := t
  have h1 := f.isLt
  have h2 := r.isLt
  simp only [attackerTable.sqOfIndexF, sqIndex]
  ext
  · simp only; omega
  · simp only; omega

theorem pieceAttacksB_geometry {p : Position} {c : Color} {s t : Square}
    (h : pieceAttacksB p c s t = true) : (queenGeometryB s t || knightGeometryB s t) = true := by
  unfold pieceAttacksB at h
  split at h
  · simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
    obtain ⟨_, h⟩ := h
    rcases h with (((((⟨⟨_, hg⟩, _⟩ | ⟨⟨_, hg⟩, _⟩) | ⟨⟨_, hg⟩, _⟩) | ⟨_, hg⟩) | ⟨_, hg⟩) | ⟨_, hg⟩)
    · simp [queenGeometryB, hg]
    · simp [queenGeometryB, hg]
    · simp [hg]
    · simp [hg]
    · exact geometry_of_attack s t (by simp [hg])
    · cases c <;> exact geometry_of_attack s t (by simp [hg])
  · simp at h

def isAttackedB' (p : Position) (c : Color) (t : Square) : Bool :=
  (attackerTable[(sqIndex t).val]'(by rw [attackerTable_size]; exact (sqIndex t).isLt)).any
    (fun s => pieceAttacksB' p c s t)

theorem isAttackedB'_eq (p : Position) (c : Color) (t : Square) : isAttackedB' p c t = isAttackedB p c t := by
  simp only [isAttackedB', isAttackedB, attackerTable_get, attackerSquares, pieceAttacksB'_eq]
  apply Bool.eq_iff_iff.mpr
  simp only [List.any_eq_true, List.mem_filter]
  constructor
  · rintro ⟨s, ⟨hs, _⟩, h⟩
    exact ⟨s, hs, h⟩
  · rintro ⟨s, hs, h⟩
    exact ⟨s, ⟨hs, pieceAttacksB_geometry h⟩, h⟩

-- A pawn's push squares lie on its file: a rook line, hence a queen line.
set_option maxRecDepth 100000 in
theorem geometry_of_push : ∀ s u : Square,
    (pawnMoveGeometryB .white s u || pawnDoubleGeometryB .white s u ||
      pawnMoveGeometryB .black s u || pawnDoubleGeometryB .black s u) = true →
      (queenGeometryB s u || knightGeometryB s u) = true := by
  decide

def inCheckB' (p : Position) (c : Color) : Bool :=
  match kingSquare p.board c with
  | some s => isAttackedB' p (Color.opposite c) s
  | none => true

theorem inCheckB'_eq (p : Position) (c : Color) : inCheckB' p c = inCheckB p c := by
  unfold inCheckB' inCheckB
  cases kingSquare p.board c <;> simp [isAttackedB'_eq]

/-! ## Legality of an ordinary move -/

def normalPieceGeometryB' (p : Position) (c : Color) (q : Piece) (s t : Square) : Bool :=
  match q.kind with
  | .rook => rookGeometryB s t && clearBetweenB' p.board s t
  | .bishop => bishopGeometryB s t && clearBetweenB' p.board s t
  | .queen => queenGeometryB s t && clearBetweenB' p.board s t
  | .knight => knightGeometryB s t
  | .king => kingGeometryB s t
  | .pawn =>
      (pawnMoveGeometryB c s t && (p.board t == none)) ||
        (pawnAttackGeometryB c s t &&
          match p.board t with
          | some q' => q'.color == Color.opposite c
          | none => false) ||
        (pawnDoubleGeometryB c s t && (p.board t == none) && clearBetweenB' p.board s t)

theorem normalPieceGeometryB'_eq (p : Position) (c : Color) (q : Piece) (s t : Square) :
    normalPieceGeometryB' p c q s t = normalPieceGeometryB p c q s t := by
  unfold normalPieceGeometryB' normalPieceGeometryB
  cases q.kind <;> cases p.board t <;> simp [clearBetweenB'_eq]

def normalPseudoLegalB' (p : Position) (s t : Square) : Bool :=
  match p.board s with
  | some q =>
      (q.color == p.turn) && destinationFriendlyB p p.turn t &&
        ((q.kind != .pawn) || (t.2 != promotionRank p.turn)) &&
        normalPieceGeometryB' p p.turn q s t
  | none => false

theorem normalPseudoLegalB'_eq (p : Position) (s t : Square) :
    normalPseudoLegalB' p s t = normalPseudoLegalB p s t := by
  unfold normalPseudoLegalB' normalPseudoLegalB
  cases p.board s <;> simp [normalPieceGeometryB'_eq]

/-- Legality of an ordinary (non-castling) move. -/
def legalNormalB' (p : Position) (s t : Square) : Bool :=
  normalPseudoLegalB' p s t && !inCheckB' (applyMove p (.normal s t)) p.turn

theorem legalNormalB'_eq (p : Position) (s t : Square) : legalNormalB' p s t = legalMoveB p (.normal s t) := by
  simp [legalNormalB', legalMoveB, pseudoLegalB, normalPseudoLegalB'_eq, inCheckB'_eq, Move.isCastle]

end Ideas
