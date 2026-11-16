import Ideas.Core.Board
import Ideas.Frame
import Ideas.FrameLemma
import Ideas.Race
import Ideas.Game

/-!
# CHESS AS AN INSTANCE OF THE BOARD STRUCTURE (PLAN 78.56 stage E)

`chessBoard : Board Position Move Square Piece Color ChessKind` -- the kernel's `legalMoves`/`applyMove` (through `Ideas.Game.chess`), the board's `Option Piece` on a square, the
kernel's `pieceAttacks`, `between`, the touched squares of `Ideas.PathLemma`, the footprint `Ideas.FrameLemma.reachB`, the king as the goal unit -- and `chessBoard_laws : chessBoard.Laws`,
each of the nine fields proved from a lemma that already existed:

| field | proved from |
|---|---|
| `touch` | `Ideas.PathLemma.applyBoard_off` (with `applyMove_board`) |
| `bears_occ` | the kernel's `pieceAttacks_empty_source` |
| `line` | `Ideas.PathLemma.new_attack_local` |
| `footprint_refl`, `bears_footprint`, `between_footprint` | `Ideas.FrameLemma.attack_reach`, `between_reach`, the definition of `reachB` |
| `goal_safe` | `Ideas.Race.no_stand_attacked` (the kernel's `legalMove_kingSafe`) |
| `dist_self`, `dist_step` | `kdist_self`, `kdist_step`, `lb_self`, `lb_step` |

Nothing is assumed and nothing is left out. The two kinds are the king and the rest: the king's static step is the king's geometry (`kdist` is its distance, 1-Lipschitz by
`kdist_step`); every other kind is given `anyStep` (a queen line or a knight's move) with the bound `lb`, which is a LOWER bound for a rook, bishop, queen, knight AND pawn (a
pawn's push or capture lies on a queen line), exactly as `Ideas.Race.lb` says. A distance for the pawn that is not a bound of `lb` (`Passer2.pdist`) and the rook's and
bishop's sharper ones (`Plan.rdist`, `Plan.bdist`) are further `Kind`s, not needed for the check.

THE RECOVERY (stage E's check that nothing was lost): `frame_unchanged_from_core` and `path_lemma_from_core` are the OLD chess statements (`Ideas.FrameLemma.frame_unchanged`,
`Ideas.PathLemma.path_lemma`, with `Bool` equalities and the kernel's types), re-derived from the generic `Board.frame_unchanged` and `Board.path_lemma` through the instance.
-/

namespace Ideas.BoardInstance

open Chess
open Ideas.Core

/-- The two kinds of the instance: the king (its own step and distance) and every other piece (a queen line or a knight's move, bounded by `lb`). -/
inductive ChessKind | king | other
  deriving DecidableEq

/-- The kind of a piece. -/
def kindOfPiece (q : Piece) : ChessKind := if q.kind = PieceKind.king then .king else .other

/-- The static step relation of a kind. -/
def chessStep : ChessKind → Square → Square → Prop
  | .king, s, s' => kingGeometryB s s' = true
  | .other, s, s' => Ideas.Race.anyStep s s'

/-- The distance (lower bound on the moves needed) of a kind. -/
def chessDist : ChessKind → Square → Square → Nat
  | .king, s, t => Ideas.Passer.kdist s t
  | .other, s, t => Ideas.Race.lb s t

/-- CHESS: the board structure over the rule kernel. -/
def chessBoard : Board Position Chess.Move Square Piece Color ChessKind where
  toGame := Ideas.Game.chess
  side := fun p => p.turn
  other := Color.opposite
  occ := fun p s => p.board s
  kindOf := kindOfPiece
  sideOf := fun q => q.color
  kstep := chessStep
  bears := fun p c s t => pieceAttacks p c s t
  between := fun s u t => Chess.between s u t
  touched := fun p m u => u ∈ Ideas.PathLemma.touched p m
  footprint := fun t x => Ideas.FrameLemma.reachB t x = true
  goalPlace := fun p c => kingSquare p.board c
  dist := chessDist

/-- THE INSTANCE: the kernel satisfies the five properties, every field from an existing lemma. -/
theorem chessBoard_laws : chessBoard.Laws where
  touch := by
    intro p m u hu
    show (applyMove p m).board u = p.board u
    rw [applyMove_board]
    exact Ideas.PathLemma.applyBoard_off p m u hu
  bears_occ := by
    intro p c s t h h0
    exact pieceAttacks_empty_source p c s t h0 h
  line := by
    intro p p' c s t h' h
    exact Ideas.PathLemma.new_attack_local p p' c s t h' (Bool.eq_false_iff.mpr h)
  footprint_refl := by
    intro t
    simp [chessBoard, Ideas.FrameLemma.reachB]
  bears_footprint := by
    intro p c s t h
    exact Ideas.FrameLemma.attack_reach p c s t h
  between_footprint := by
    intro s u t h
    exact Ideas.FrameLemma.between_reach s u t h
  goal_safe := by
    intro p m hm t ht s hs
    exact Ideas.Race.no_stand_attacked (legalMoves_sound hm) ht hs
  dist_self := by
    intro k t
    cases k
    · exact Ideas.Race.kdist_self t
    · exact Ideas.Race.lb_self t
  dist_step := by
    intro k s s' t h
    cases k
    · exact Ideas.Race.kdist_step h
    · exact Ideas.Race.lb_step h

/-! ## The recovery: the old chess statements from the generic ones -/

/-- `Ideas.FrameLemma.frame_unchanged`, re-derived from `Board.frame_unchanged` through `chessBoard_laws`. -/
theorem frame_unchanged_from_core (p : Position) (m : Move) (F : Square → Prop)
    (hF : ∀ t, F t → ∀ x, x ∈ Ideas.PathLemma.touched p m → Ideas.FrameLemma.reachB t x = false) :
    (∀ t, F t → (applyMove p m).board t = p.board t) ∧
    (∀ t, F t → ∀ c s, pieceAttacksB (applyMove p m) c s t = pieceAttacksB p c s t) := by
  have h := Board.frame_unchanged chessBoard_laws p m F (fun t ht x hx hfp => by
    have := hF t ht x hx
    simp only [chessBoard] at hfp
    rw [this] at hfp
    exact absurd hfp (by decide))
  refine ⟨h.1, fun t ht c s => ?_⟩
  exact Bool.eq_iff_iff.mpr (h.2 t ht c s)

/-- `Ideas.PathLemma.path_lemma`, re-derived from `Board.path_lemma` through `chessBoard_laws`. -/
theorem path_lemma_from_core (p : Position) (m : Move) (c : Color) (s t : Square)
    (h' : pieceAttacksB (applyMove p m) c s t = true) (h : pieceAttacksB p c s t = false) :
    (s ∈ Ideas.PathLemma.touched p m ∧ (applyMove p m).board s ≠ none) ∨
      ∃ u, between s u t ∧ u ∈ Ideas.PathLemma.touched p m ∧ p.board u ≠ none ∧ (applyMove p m).board u = none :=
  Board.path_lemma chessBoard_laws p m c s t h' (by simpa [chessBoard, pieceAttacks] using h)

/-- `Ideas.PathLemma.attack_unchanged`, re-derived from `Board.attack_unchanged`. -/
theorem attack_unchanged_from_core (p : Position) (m : Move) (c : Color) (s t : Square)
    (hs : s ∉ Ideas.PathLemma.touched p m) (hb : ∀ u, between s u t → u ∉ Ideas.PathLemma.touched p m) :
    pieceAttacksB (applyMove p m) c s t = pieceAttacksB p c s t :=
  Bool.eq_iff_iff.mpr (Board.attack_unchanged chessBoard_laws p m c s t hs hb)

end Ideas.BoardInstance
