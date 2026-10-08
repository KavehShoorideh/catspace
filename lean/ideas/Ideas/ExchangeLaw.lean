import Ideas.Exchange
import Ideas.Fork

/-!
# The exchange law, part 2: when a square's game is the capture game -- the statement

Part 1 (`Ideas/Exchange.lean`) is the capture game on one square. This file says, in the kernel's own terms, when the
real game on a square of a chess position IS that game, and what is then claimed.

THE FRAME on square `s` of position `p` (side `c = p.turn` to move): the piece on `s` is the opponent's; `c`'s attackers
of `s` and the opponent's defenders, with their values (`pieceValue`, the king 0 -- excluded by hypothesis).

THE HONEST FRAME (`honestFrameB`), nine hypotheses, each a Bool on the position (Kaveh 2026-10-07: "the tactics are
basically exceptions to the exchange theorem" -- each failure is a named tactic):
  target        the piece on `s` is the opponent's and not a king; `c` attacks it; `c`'s cheapest capture is legal
  noKing        no king among the attackers or defenders
  noPin         every attacker's capture on `s` is legal now, and every defender's recapture is legal after it
  noXray        no slider of either side stands behind a single piece on a line to `s`
  noPromotion   no pawn participant on its seventh rank; `s` is not the en-passant square
  noOtherCapture  after `c`'s first capture, no capture of the opponent off `s` wins by the static exchange, and no
                attacker of `c` defends a piece of `c` the opponent attacks (an overloaded defender)
  noCheck       after `c`'s first capture the opponent has no checking move; `c`'s king shares no line and no knight
                relation with `s`; no opponent slider shares a line with `c`'s king
  noInterpose   every attacker of `c` is a knight or a pawn or stands next to `s` (no blocking a recapture)
  noArrival     no opponent piece off the frame can move to attack `s` in one move

THE CLAIM (`settledWithin`): after `c`'s cheapest capture on `s`, within the frame's size in plies and whatever the
opponent does, a SETTLED position is reached -- the opponent to move, `c`'s material at least its value before plus the
static exchange, and no recapture on `s` that pays the opponent (`forcedWithinB`: the goal read at every node, so the
path stops the moment the exchange has ended in `c`'s favour; no quiet move of ours is ever required).

THE LAW, to prove (`exchangeLaw`): `honestFrameB p s = true → settledWithin p s = true`. The induction: on the number
of participants left; at the opponent's node a recapture on `s` is answered by `c`'s next capture (legal: noPin,
noCheck), any other move leaves the position settled (noOtherCapture, noCheck, noArrival keep the material and the
frame fixed); the static value is the capture game's by part 1. Until it is proved, `settledWithin` is a per-position
CERTIFICATE the kernel computes (like the double threat's), and `honestFrameB` the kernel's own detector of an honest
frame, bridged against the Python prototype (`exchange_law.py`) on the census.
-/

namespace Ideas.ExchangeLaw

open Chess
open Ideas.Exchange

/-- The squares of `c`'s pieces attacking `s`. -/
def attackersOf (p : Position) (c : Color) (s : Square) : List Square :=
  allSquares.filter (fun q => pieceAttacksB p c q s)

/-- A piece's value in the exchange: the king counts 100 (it cannot be won, and as a capturer it is dearer than anything;
review 2026-10-07: at 0 it was a free recapturer). -/
def valueAt (p : Position) (q : Square) : Int :=
  match p.board q with
  | some pc => if pc.kind = .king then 100 else pieceValue pc.kind
  | none => 0

/-- The values of `c`'s attackers of `s`, ascending. -/
def valuesOf (p : Position) (c : Color) (s : Square) : List Int :=
  ((attackersOf p c s).map (valueAt p)).insertionSort (· ≤ ·)

/-- The static exchange for the side to move CAPTURING on `s` with its cheapest piece: what the capture itself is
worth, the sequence beyond it played out with each side free to stop (possibly negative: the law is a lower bound on
capturing, not a recommendation to capture; review 2026-10-07: flooring this at 0 made the claim false on two frames). -/
def seeB (p : Position) (s : Square) : Int :=
  match valuesOf p p.turn s with
  | [] => 0
  | a :: rest => valueAt p s - cheapest a (valuesOf p (Color.opposite p.turn) s) rest

/-- `c`'s cheapest legal capture on `s`, if any. -/
def cheapestCapture (p : Position) (s : Square) : Option Move :=
  let c := p.turn
  let atk := (attackersOf p c s).mergeSort (fun a b => valueAt p a ≤ valueAt p b)
  atk.findSome? fun q => (legalMoves p).find? (fun m => m.source == q && m.destination == s && !m.isCastle)

def isKingAt (p : Position) (q : Square) : Bool :=
  match p.board q with
  | some pc => pc.kind == .king
  | none => false

def isSliderAt (p : Position) (q : Square) : Bool :=
  match p.board q with
  | some pc => pc.kind == .bishop || pc.kind == .rook || pc.kind == .queen
  | none => false

def isPawnAt (p : Position) (q : Square) : Bool :=
  match p.board q with
  | some pc => pc.kind == .pawn
  | none => false

/-- A slider on `q` would attack `s` along a line were the squares between empty (geometry only). -/
def sliderLineB (p : Position) (q s : Square) : Bool :=
  match p.board q with
  | some pc => pieceGeometryB pc.kind pc.color q s
  | none => false

/-- Exactly one piece between `q` and `s`. -/
def oneBetweenB (p : Position) (q s : Square) : Bool :=
  (allSquares.filter (fun u => betweenB q u s && (p.board u).isSome)).length == 1

def seventhRankB (p : Position) (q : Square) : Bool :=
  match p.board q with
  | some pc => pc.kind == .pawn && (q.2.val == (if pc.color = .white then 6 else 1))
  | none => false

/-- The nine hypotheses, by name. -/
def hypotheses (p : Position) (s : Square) : List (String × Bool) :=
  let c := p.turn
  let o := Color.opposite c
  let A := attackersOf p c s
  let D := attackersOf p o s
  let parts := A ++ D
  let target := (match p.board s with
    | some pc => pc.color == o && pc.kind != .king
    | none => false) && !A.isEmpty && (cheapestCapture p s).isSome
  let noKing := parts.all (fun q => !isKingAt p q)
  let after := match cheapestCapture p s with
    | some m => applyMove p m
    | none => p
  let noPin := A.all (fun q => (legalMoves p).any (fun m => m.source == q && m.destination == s)) &&
    D.all (fun q => (legalMoves after).any (fun m => m.source == q && m.destination == s))
  let noXray := allSquares.all (fun q => !(isSliderAt p q && !parts.contains q && q != s && sliderLineB p q s && oneBetweenB p q s))
  let n := parts.length                                           -- the claim's horizon in plies; the opponent moves (n + 1) / 2 times
  let pawnNear := fun (q : Square) => match after.board q with    -- an enemy pawn that could promote inside the horizon (review
    | some pc => pc.color == o && pc.kind == .pawn &&              -- 2026-10-07: a non-participant pawn promoted and refuted the law)
        decide ((if pc.color = .white then 7 - q.2.val else q.2.val) ≤ (n + 1) / 2)
    | none => false
  let noPromotion := parts.all (fun q => !seventhRankB p q) && (p.enPassant != some s) && allSquares.all (fun q => !pawnNear q)
  let noOtherCapture := allSquares.all (fun q =>
    match after.board q with
    | some pc => !(pc.color == c && pc.kind != .king && q != s) ||
        (!(attackersOf after o q).isEmpty →
          (A.all (fun a => !pieceAttacksB after c a q)) &&
          decide (cheapest (valueAt after q) (valuesOf after o q) (valuesOf after c q) ≤ 0))
    | none => true) &&
    A.all (fun a => a == s || (after.board a).isNone || (attackersOf after o a).isEmpty)   -- our remaining attackers unattacked:
                                                                                              -- an even trade of one changes the frame
  let noCheck := (legalMoves after).all (fun m => !inCheckB (applyMove after m) c) &&
    (match kingSquare p.board c with
      | some k => !(queenGeometryB s k || knightGeometryB s k) &&
          allSquares.all (fun q => !(isSliderAt p q && (match p.board q with | some pc => pc.color == o | none => false) && queenGeometryB q k))
      | none => false)
  let noInterpose := A.all (fun q => isPawnAt p q || (match p.board q with | some pc => pc.kind == .knight | none => false) || kingGeometryB q s)
  let noArrival := (legalMoves after).all (fun m => D.contains m.source || m.destination == s || !pieceAttacksB (applyMove after m) o m.destination s)
  [("target", target), ("noKing", noKing), ("noPin", noPin), ("noXray", noXray), ("noPromotion", noPromotion),
   ("noOtherCapture", noOtherCapture), ("noCheck", noCheck), ("noInterpose", noInterpose), ("noArrival", noArrival)]

def honestFrameB (p : Position) (s : Square) : Bool :=
  (hypotheses p s).all (fun h => h.2)

/-- A settled position for `c` on `s`: the opponent to move, `c`'s material at least `target`, and no recapture on `s`
that pays the opponent by the static exchange. -/
def settledB (c : Color) (s : Square) (target : Int) (q : Position) : Bool :=
  q.turn != c &&
    (((legalMoves q).isEmpty && inCheckB q q.turn) ||                       -- the opponent is mated: won
     (decide (target ≤ material q c) &&
      allSquares.all (fun u =>                                               -- no capture anywhere that pays the opponent by the
        match q.board u with                                                 -- static exchange (review 2026-10-07: reading only `s`
        | some pc => !(pc.color == c) ||                                     -- certified a frame an even trade elsewhere refuted)
            decide (cheapest (valueAt q u) (valuesOf q q.turn u) (valuesOf q c u) ≤ 0)
        | none => true)))

/-- THE CLAIM: after `c`'s cheapest capture on `s`, a settled position within the frame's size in plies, whatever the
opponent does. A per-position certificate until `exchangeLaw` is proved. -/
def settledWithin (p : Position) (s : Square) : Bool :=
  match cheapestCapture p s with
  | none => false
  | some m =>
      let c := p.turn
      let n := (attackersOf p c s).length + (attackersOf p (Color.opposite c) s).length
      forcedWithinB c (settledB c s (material p c + seeB p s)) (applyMove p m) n

/-- The exchange law, as a proposition: every honest frame's claim holds. (The proof is the work of part 2.) -/
def exchangeLaw : Prop :=
  ∀ (p : Position) (s : Square), honestFrameB p s = true → settledWithin p s = true

end Ideas.ExchangeLaw
