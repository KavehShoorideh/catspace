import Ideas.Schema
import Ideas.Mate

/-!
# The back-rank mate

The defender's king on its home rank, boxed in by its own pieces or by squares we cover; our rook
or queen lands on that rank with check; nothing can take it, nothing can step between, the king
has nowhere to go. Every hypothesis is about squares and attacks, never about which piece does
the attacking or the defending ("who doesn't matter"). Mirrors `catspace/incubator/analyst/
backrank.py` (boxed / armed / strike / lure), where the strike was verified by a search; here
it is proved once.

Tier `unconditional`, horizon 1 ply: when every hypothesis holds, the move mates.
When exactly one fails, `targets` names the squares that broke it -- the defender of the
landing square, the piece in the rook's path, the flight square left open -- which is where the
search looks first.
-/

namespace Ideas

open Chess

/-- Our rook or queen on `a`, its landing square `c` on the defender's home rank, the
defender's king on `k`. -/
structure BackRankParams where
  a : Square
  c : Square
  k : Square
  deriving Repr, DecidableEq

namespace BackRank

/-- The home rank of colour `c` (its first rank). -/
def homeRank : Color → Fin 8
  | .white => 0
  | .black => 7

def move (ps : BackRankParams) : Move := .normal ps.a ps.c

/-- The position after our move: the defender to move. -/
def after (p : Position) (ps : BackRankParams) : Position := applyMove p (move ps)

/-- Our rook or queen stands on `a`. -/
def heavyB (p : Position) (ps : BackRankParams) : Bool :=
  match p.board ps.a with
  | some pc => (pc.color == p.turn) && (pc.kind == .rook || pc.kind == .queen)
  | none => false

/-- The king and the landing square are on the defender's home rank. -/
def onBackRankB (p : Position) (ps : BackRankParams) : Bool :=
  (ps.k.2 == homeRank (Color.opposite p.turn)) && (ps.c.2 == homeRank (Color.opposite p.turn))

/-- The move is legal (fails when a piece stands in the path, or the piece is pinned). -/
def legalB (p : Position) (ps : BackRankParams) : Bool :=
  legalMoveB p (move ps)

/-- The defender has one king, on `k`. -/
def oneKingB (p : Position) (ps : BackRankParams) : Bool :=
  kingOnlyAtB (after p ps).board (after p ps).turn ps.k

/-- The piece on `c` checks the king (fails when something stands between them on the rank). -/
def checkB (p : Position) (ps : BackRankParams) : Bool :=
  checksB (after p ps) ps.k ps.c

/-- Boxed: every flight off the home rank is held by the defender's own piece or covered. -/
def boxedB (p : Position) (ps : BackRankParams) : Bool :=
  allSquares.all (fun d => decide (d.2 = ps.k.2) || flightOkB (after p ps) ps.k d)

/-- Every flight along the home rank (taking the checker included) is held or covered. -/
def rankFlightsB (p : Position) (ps : BackRankParams) : Bool :=
  allSquares.all (fun d => decide (d.2 ≠ ps.k.2) || flightOkB (after p ps) ps.k d)

/-- No defender's piece but the king attacks the landing square. -/
def noDefenderB (p : Position) (ps : BackRankParams) : Bool :=
  noCaptureB (after p ps) ps.k ps.c

/-- No defender's piece can step between the landing square and the king. -/
def noInterposeB (p : Position) (ps : BackRankParams) : Bool :=
  noBlockB (after p ps) ps.k ps.c

/-- Where to look when one hypothesis fails: the squares that broke it. -/
def targets (p : Position) (ps : BackRankParams) (hyp : String) : List Square :=
  let q := after p ps
  if hyp = "legal" then
    (squaresBetween ps.a ps.c).filter (fun u => (p.board u).isSome)
  else if hyp = "check" then
    (squaresBetween ps.c ps.k).filter (fun u => (q.board u).isSome)
  else if hyp = "boxed" then
    allSquares.filter (fun d => decide (d.2 ≠ ps.k.2) && !flightOkB q ps.k d)
  else if hyp = "rankFlights" then
    allSquares.filter (fun d => decide (d.2 = ps.k.2) && !flightOkB q ps.k d)
  else if hyp = "noDefender" then
    allSquares.filter (fun s => decide (s ≠ ps.k) && pieceAttacksB q q.turn s ps.c)
  else if hyp = "noInterpose" then
    allSquares.filter (fun s => (squaresBetween ps.c ps.k).any (fun u => couldReachEmptyB q s u))
  else []

end BackRank

open BackRank in
/-- The back-rank mate, in the schema. -/
def backRankMate : Idea BackRankParams where
  id := "backRankMate"
  hyps :=
    [ ⟨"heavy", heavyB⟩, ⟨"onBackRank", onBackRankB⟩, ⟨"legal", legalB⟩,
      ⟨"oneKing", oneKingB⟩, ⟨"check", checkB⟩, ⟨"boxed", boxedB⟩,
      ⟨"rankFlights", rankFlightsB⟩, ⟨"noDefender", noDefenderB⟩,
      ⟨"noInterpose", noInterposeB⟩ ]
  protagonist := fun ps => ps.a
  tier := .unconditional
  horizon := 1

/-- A rook or queen move sets no en-passant square. -/
theorem BackRank.after_enPassant (p : Position) (ps : BackRankParams)
    (h : BackRank.heavyB p ps = true) : (BackRank.after p ps).enPassant = none := by
  unfold BackRank.heavyB at h
  cases ha : p.board ps.a with
  | none => simp [ha] at h
  | some pc =>
    rw [ha] at h
    have hnp : pc.kind ≠ .pawn := by
      intro hk
      simp [hk] at h
    simp [BackRank.after, BackRank.move, applyMove, newEnPassant, pawnDoubleTarget,
      moveIsPawn, movingPiece, Move.source, ha, hnp]

/-- **Soundness of the back-rank mate.** When every hypothesis holds, the move is legal and the
defender is checkmated -- whatever the rest of the board looks like. -/
theorem backRankMate_sound (p : Position) (ps : BackRankParams)
    (h : backRankMate.holds p ps = true) :
    legalMoveB p (BackRank.move ps) = true ∧ checkmateB (BackRank.after p ps) = true := by
  simp only [Idea.holds, backRankMate, List.all_cons, List.all_nil, Bool.and_true,
    Bool.and_eq_true] at h
  obtain ⟨hheavy, _, hlegal, hking, hcheck, hboxed, hrank, hdef, hint⟩ := h
  refine ⟨hlegal, ?_⟩
  have hflights : flightsCoveredB (BackRank.after p ps) ps.k = true := by
    simp only [flightsCoveredB, List.all_eq_true]
    intro d hd
    by_cases hdk : d.2 = ps.k.2
    · have := (List.all_eq_true.1 hrank) d hd
      simpa [hdk] using this
    · have := (List.all_eq_true.1 hboxed) d hd
      simpa [hdk] using this
  exact checkmate_of_unmeetable_check _ ps.k ps.c hking hcheck
    (BackRank.after_enPassant p ps hheavy) hflights hdef hint

end Ideas
