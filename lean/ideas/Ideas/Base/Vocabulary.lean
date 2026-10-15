import Chess.Game
import Ideas.Mate
import Ideas.GeoLemmas
import Ideas.Passer
import Ideas.ExchangeLaw

/-!
# The base vocabulary (Kaveh 2026-10-12)

"The important thing to give it is the vocabulary because that's what maps to human understanding in the end, enabling
interpretability. You can build arbitrary lemmas on top of it, but the base has to be deterministic and interpretable."

The discovery system (catspace/incubator/analyst/proposer.py, its evaluator rust/analyst_core/src/vocab.rs) builds every
concept as a term over these primitives and nothing else; each is a deterministic definition with a plain gloss, so a
discovered concept explains itself by unfolding into these words. Three layers:

* THE RULES -- squares, pieces, moves, attacks: the chess library's own definitions (`pieceAttacksB`, `isAttackedB`,
  `legalMoves`, `inCheckB`), used as they are.
* PERCEPTION -- what a player sees at a glance (below): distance in king steps, the king's box, the edge and the corners,
  a rank and a file, the line through two squares and the squares between them, the squares nearer one square than
  another, the squares a side attacks (a king lifted off the board), the squares of a side's pieces, the colour of a
  square. Both sides have their words: our king W, their king L, our other piece P, their other piece O.
* COMBINATORS -- count, the set where something holds, union / intersection / complement, sums, and REACH: the squares a
  king walks to by king steps over a set.

Chess concepts (the wall, the opposition, the box a queen cuts, the front zone, the side room) are deliberately NOT here:
they must be built from this base, or the rediscovery tests are void.
-/

namespace Ideas.Vocabulary

open Chess

/-- A set of squares, as the board's squares where a test holds. -/
abbrev SquareSet := Square → Bool

/-- **distance**: how many king steps from `s` to `t`. -/
def dist (s t : Square) : Nat := Passer.kdist s t

/-- **box**: the squares a king on `s` touches, and `s` itself (its 3×3 box, on the board). -/
def box (s : Square) : SquareSet := fun t => decide (dist s t ≤ 1)

/-- **edge**: the squares on the board's edge. -/
def edge : SquareSet := fun t => t.1.val = 0 || t.1.val = 7 || t.2.val = 0 || t.2.val = 7

/-- **corner**: the four corner squares. -/
def corner : SquareSet := fun t => (t.1.val = 0 || t.1.val = 7) && (t.2.val = 0 || t.2.val = 7)

/-- **rank of**: the squares on `s`'s rank. -/
def rankOf (s : Square) : SquareSet := fun t => decide (t.2 = s.2)

/-- **file of**: the squares on `s`'s file. -/
def fileOf (s : Square) : SquareSet := fun t => decide (t.1 = s.1)

/-- **line through**: the squares on the line (rank, file or diagonal) through `s` and `t`, when they share one; none
otherwise. -/
def lineThrough (s t : Square) : SquareSet := fun u =>
  queenGeometryB s t && (u == s || u == t || (queenGeometryB s u && queenGeometryB t u &&
    ((sameFileB s t && sameFileB s u) || (sameRankB s t && sameRankB s u) ||
     (sameDiagonalB s t && sameDiagonalB s u && sameDiagonalB t u))))

/-- **between**: the squares strictly between `s` and `t` on their line. -/
def between (s t : Square) : SquareSet := fun u => betweenB s u t

/-- **nearer**: the squares nearer to `s` than to `t` (in king steps) -- `s`'s side of the board against `t`. -/
def nearer (s t : Square) : SquareSet := fun u => decide (dist u s < dist u t)

/-- **attacked by**: the squares side `c` attacks with the king on `k` lifted off the board (it cannot hide behind
itself). -/
def attackedBy (p : Position) (c : Color) (k : Square) : SquareSet := fun u => isAttackedB (withEmpty p k) c u

/-- **pieces of**: the squares holding a piece of side `c` other than its king (Kaveh 2026-10-12: "base vocab should be
complete with words for their pieces" -- `ours` is `piecesOf p c`, `theirs` is `piecesOf p c.opposite`; the squares
they attack, `hit`, is `attackedBy p c.opposite W`, our king lifted). -/
def piecesOf (p : Position) (c : Color) : SquareSet := fun u =>
  match p.board u with
  | some pc => decide (pc.color = c) && decide (pc.kind ≠ PieceKind.king)
  | none => false

-- GEOMETRY AND PIECES, a second batch (Kaveh 2026-10-12: "we want to be able to specify sets of squares ... along a
-- diagonal or along a rank or a file from here to here ... reachable by a knight, one hop or two hops ... the four squares
-- on the center ... any particular square should be able to be named, but ... the symmetry ... except for the last rank
-- where there's promotion ... some per piece vocabulary"; a time vocabulary later)

/-- **lines**: the squares on `s`'s rank or file (a rook's lines through `s`, blockers ignored). -/
def lines (s : Square) : SquareSet := fun t => rookGeometryB s t

/-- **diagonals**: the squares on the diagonals through `s` (a bishop's lines, blockers ignored). -/
def diagonals (s : Square) : SquareSet := fun t => bishopGeometryB s t

/-- **segment**: the squares from `s` to `t` along the rank, file or diagonal they share, both ends included (none when they
share no line). -/
def segment (s t : Square) : SquareSet := fun u => queenGeometryB s t && (u == s || u == t || betweenB s u t)

/-- **one knight hop**: the squares a knight on `s` reaches. -/
def knight1 (s : Square) : SquareSet := fun t => knightGeometryB s t

/-- **two knight hops**: the squares a knight on `s` reaches in two hops. -/
def knight2 (s : Square) : SquareSet := fun t => allSquares.any (fun u => knightGeometryB s u && knightGeometryB u t)

/-- **the centre**: d4, e4, d5, e5. -/
def center : SquareSet := fun t => (t.1.val == 3 || t.1.val == 4) && (t.2.val == 3 || t.2.val == 4)

-- a square's colour (Kaveh 2026-10-13): a1 is dark -- file + rank even
/-- the light squares -/
def light : SquareSet := fun t => (t.1.val + t.2.val) % 2 == 1
/-- the dark squares -/
def dark : SquareSet := fun t => (t.1.val + t.2.val) % 2 == 0
/-- the squares of `s`'s colour -/
def colour (s : Square) : SquareSet := fun t => (t.1.val + t.2.val) % 2 == (s.1.val + s.2.val) % 2

/-- **attacks from**: the squares the piece standing on `s` attacks, for its own colour, with the board's blockers (the chess
library's `pieceAttacksB`); none when `s` is empty. -/
def attacksFrom (p : Position) (s : Square) : SquareSet := fun t =>
  match p.board s with
  | some q => pieceAttacksB p q.color s t
  | none => false

/-- The board's eight symmetries of a square (rotations and reflections). -/
def symmetries (s : Square) : List Square :=
  let (f, r) := s
  [(f, r), (Fin.rev f, r), (f, Fin.rev r), (Fin.rev f, Fin.rev r), (r, f), (Fin.rev r, f), (r, Fin.rev f), (Fin.rev r, Fin.rev f)]

/-- **the class of a square**: the squares like `s` under the board's symmetries -- all eight when no pawn is on the board,
only the left-right mirror when one is (the promotion direction breaks the rest); `s` is named from side `c`'s view (its
first rank, rank 1). -/
def cls (p : Position) (c : Color) (s : Square) : SquareSet := fun u =>
  let pawns := allSquares.any (fun v => match p.board v with | some q => q.kind == .pawn | none => false)
  let abs : Square := if c == Color.white then s else (s.1, Fin.rev s.2)
  if pawns then u == abs || u == (Fin.rev abs.1, abs.2) else (symmetries abs).contains u

/-- **a named square**: the square `s` named from side `c`'s view -- its first rank is rank 1, so `a8` is its promotion
corner on the a-file (Kaveh 2026-10-13: "the squares individually available"; the board's symmetries are a lemma of their
own, not part of the words). -/
def named (c : Color) (s : Square) : Square := if c == Color.white then s else (s.1, Fin.rev s.2)

/-- **the square itself**, as a set. -/
def atSq (s : Square) : SquareSet := fun u => u == s

/-- **file number**: a square's file, a = 0 ... h = 7 (Kaveh 2026-10-13: the vocabulary complemented with relative geometry --
whatever it took to see a cycle lemma, as words). -/
def fileNum (s : Square) : Nat := s.1.val

/-- **rank number**: a square's rank counted from side `c`'s first rank (0 ... 7). -/
def rankNum (c : Color) (s : Square) : Nat := (named c s).2.val

/-- **files from t to s**: how many files `s` stands toward the h-file from `t`, plus 7 (7: the same file). -/
def dFile (s t : Square) : Nat := s.1.val + 7 - t.1.val

/-- **ranks from t to s**: how many ranks `s` stands ahead of `t` toward the other side, plus 7 (7: the same rank). -/
def dRank (c : Color) (s t : Square) : Nat := rankNum c s + 7 - rankNum c t

/-- **colour of a square**: light or dark. -/
def squareColour (s : Square) : Nat := GeoLemmas.colour s

/-- **count**: how many squares a set holds. -/
def count (A : SquareSet) : Nat := (allSquares.filter A).length

/-- **reach**: the squares a king on `s` walks to by king steps over the squares of `A` (and `s`), at most 64 steps. -/
def reach (s : Square) (A : SquareSet) : SquareSet :=
  let step (R : SquareSet) : SquareSet := fun u => R u || (A u && allSquares.any (fun v => R v && dist v u == 1))
  Nat.repeat step 64 (fun u => u == s)

/-! ## The vocabulary completed (Kaveh 2026-10-13: "Yes, please add all of these ... and also ... regular chess terms like
pin ... in terms of simpler vocabulary so that we can kind of have compositional vocab"). Each definition is the mirror of
the Rust evaluator (rust/analyst_core/src/vocab.rs); the chess terms are compositions of these, unfolded where they are
read (catspace/incubator/analyst/discovery/base_vocabulary.py CHESS_TERMS). -/

/-- An optional square: a word naming a piece that may not be on the board, or a square past the edge. -/
abbrev OSq := Option Square

/-- A set word over an optional square: empty when there is no square. -/
def oset (f : Square → SquareSet) : OSq → SquareSet
  | some s => f s
  | none => fun _ => false

def oset2 (f : Square → Square → SquareSet) : OSq → OSq → SquareSet
  | some s, some t => f s t
  | _, _ => fun _ => false

/-- A number word over an optional square: 0 when there is no square. -/
def onum (f : Square → Nat) : OSq → Nat
  | some s => f s
  | none => 0

def onum2 (f : Square → Square → Nat) : OSq → OSq → Nat
  | some s, some t => f s t
  | _, _ => 0

/-- reach from an optional square -/
def oreach (s : OSq) (A : SquareSet) : SquareSet := oset (fun x => reach x A) s

/-- A square's index rank by rank from a1 (the evaluator's order). -/
def sqIndex (s : Square) : Nat := s.2.val * 8 + s.1.val

/-- The strength order of the kinds: queen, rook, bishop, knight, pawn. -/
def kindRank : PieceKind → Nat
  | .queen => 5 | .rook => 4 | .bishop => 3 | .knight => 2 | .pawn => 1 | .king => 0

/-- `s` before `t` among side `c`'s pieces: the stronger kind, then (pawns) the more advanced, then the lower square. -/
def ranksBefore (p : Position) (c : Color) (s t : Square) : Bool :=
  let k : Square → Nat := fun u => match p.board u with | some q => kindRank q.kind | none => 0
  let a : Square → Nat := fun u => match p.board u with | some q => (if q.kind == PieceKind.pawn then rankNum c u else 0) | none => 0
  decide (k s > k t) || (k s == k t && (decide (a s > a t) || (a s == a t && decide (sqIndex s ≤ sqIndex t))))

/-- **every piece named**: side `c`'s pieces other than its king, ranked (P, P2, P3, ...; O, O2, ...). -/
def ranked (p : Position) (c : Color) : List Square :=
  (allSquares.filter (fun u => match p.board u with | some q => q.color == c && q.kind != .king | none => false)).mergeSort (ranksBefore p c)

/-- the `i`-th piece of side `c` (from 1) -/
def nthPiece (p : Position) (c : Color) (i : Nat) : OSq := (ranked p c)[i - 1]?

/-- **a kind as a set**: the squares of side `c`'s pieces of kind `k`. -/
def kindSet (p : Position) (c : Color) (k : PieceKind) : SquareSet := fun u =>
  match p.board u with | some q => q.color == c && q.kind == k | none => false

/-- The colour of the piece on `s`. -/
def ownerOf (p : Position) (s : Square) : Option Color := (p.board s).map (·.color)

/-- **ahead**: the squares in front of the piece on `s` on its file, toward its owner's far side. -/
def aheadOf (p : Position) (s : Square) : SquareSet := fun t =>
  match ownerOf p s with
  | some c => t.1 == s.1 && decide (rankNum c t > rankNum c s)
  | none => false

/-- **the passed-pawn zone**: in front of the piece on `s`, on its file and the two beside it. -/
def passerOf (p : Position) (s : Square) : SquareSet := fun t =>
  match ownerOf p s with
  | some c => decide (absDiff t.1 s.1 ≤ 1) && decide (rankNum c t > rankNum c s)
  | none => false

/-- **front**: the square in front of the piece on `s` (its owner's direction). -/
def frontOf (p : Position) : OSq → OSq
  | some s => match ownerOf p s with | some c => Passer.ahead c s 1 | none => none
  | none => none

/-- **promotion square**: the piece's file on its owner's last rank. -/
def promoOf (p : Position) : OSq → OSq
  | some s => (ownerOf p s).map (fun c => (s.1, promotionRank c))
  | none => none

/-- **pushes**: the ranks the piece on `s` has still to go to its owner's last rank. -/
def pushesOf (p : Position) (s : Square) : Nat := match ownerOf p s with | some c => Passer.pushes c s | none => 0

/-- **offset**: the square `df` files and `dr` ranks from `s`, from side `c`'s view (ranks toward the other side). -/
def offsetOf (c : Color) : OSq → Int → Int → OSq
  | some s, df, dr =>
    let f : Int := (s.1.val : Int) + df
    let r : Int := (s.2.val : Int) + (if c == Color.white then dr else -dr)
    if 0 ≤ f ∧ f < 8 ∧ 0 ≤ r ∧ r < 8 then some (⟨f.toNat % 8, Nat.mod_lt _ (by decide)⟩, ⟨r.toNat % 8, Nat.mod_lt _ (by decide)⟩)
    else none
  | none, _, _ => none

/-- **beyond**: the squares past `t` on the ray from `s` through `t` (none when they share no line). -/
def beyondOf (s t : Square) : SquareSet := fun u => queenGeometryB s t && betweenB s t u

/-- **next**: the first piece past `t` on the ray from `s` through `t`. -/
def nextOn (p : Position) : OSq → OSq → OSq
  | some s, some t => ((allSquares.filter (fun u => beyondOf s t u && (p.board u).isSome)).mergeSort
      (fun a b => decide (dist t a ≤ dist t b))).head?
  | _, _ => none

/-- **x-ray**: the squares the piece on `s` attacks through at most one piece in the way (a slider); else its attacks. -/
def xrayFrom (p : Position) (s : Square) : SquareSet := fun t =>
  match p.board s with
  | none => false
  | some q =>
    let geo := match q.kind with
      | .rook => rookGeometryB s t | .bishop => bishopGeometryB s t | .queen => queenGeometryB s t | _ => false
    if q.kind == .rook || q.kind == .bishop || q.kind == .queen then
      geo && decide ((allSquares.filter (fun u => betweenB s u t && (p.board u).isSome)).length ≤ 1)
    else pieceAttacksB p q.color s t

/-- **attackers**: how many of side `c`'s pieces attack `s`. -/
def attackerCount (p : Position) (c : Color) (s : Square) : Nat := (allSquares.filter (fun u => pieceAttacksB p c u s)).length

/-- **the exchange**: what the side not owning the piece on `s` wins by capturing there first (the x-ray static exchange of
`Ideas.ExchangeLaw`, never below 0). -/
def seeOf (p : Position) (s : Square) : Nat :=
  match ownerOf p s with
  | some c => (Ideas.ExchangeLaw.seeXB { p with turn := Color.opposite c } s).toNat
  | none => 0

/-- **moves**: the squares the piece on `s` can legally move to, its side to move (as if it were, when it is not and the side
to move is not in check). -/
def movesOf (p : Position) (s : Square) : SquareSet := fun t =>
  match p.board s with
  | none => false
  | some q =>
    let r : Option Position := if q.color == p.turn then some p
      else if inCheckB p p.turn then none else some { p with turn := q.color, enPassant := none }
    match r with
    | some r => (legalMoves r).any (fun m => m.source == s && m.destination == t)
    | none => false

/-- **material**: side `c`'s pieces at queen 9, rook 5, bishop 3, knight 3, pawn 1. -/
def materialOf (p : Position) (c : Color) : Nat :=
  (allSquares.map (fun u => match p.board u with | some q => (if q.color == c then (pieceValue q.kind).toNat else 0) | none => 0)).sum

/-- A dead draw: kings alone, or one knight or bishop beside them. -/
def deadDraw (q : Position) : Bool :=
  match allSquares.filter (fun u => match q.board u with | some x => x.kind != .king | none => false) with
  | [] => true
  | [u] => match q.board u with | some x => x.kind == .knight || x.kind == .bishop | none => false
  | _ => false

/-- A value no material reaches: only a mate counts. -/
def mateGain : Int := 100000

/-- The tactic's goal for side `c`: on its own turn, a gain of `v` over `base` held and the game alive -- or the other side
mated. -/
def gainGoal (c : Color) (base v : Int) (q : Position) : Bool :=
  (q.turn == c && decide ((materialOf q c : Int) - materialOf q (Color.opposite c) ≥ base + v) && !(deadDraw q)) ||
  (q.turn != c && (legalMoves q).isEmpty && inCheckB q q.turn)

/-- **the short window's tactic**: the fewest moves (1 to 3) in which side `c` forces a gain of `v` or more, or a mate, against
every defence (as if the other side passed when it is not `c`'s turn); 0 when none. The evaluator also stops at a claimable
draw, which a position without its game cannot see. -/
def forcingOf (p : Position) (c : Color) (v : Int) : Nat :=
  let q : Option Position := if p.turn == c then some p
    else if inCheckB p p.turn then none else some { p with turn := c, enPassant := none }
  match q with
  | none => 0
  | some q =>
    let base := (materialOf q c : Int) - materialOf q (Color.opposite c)
    match [1, 2, 3].find? (fun n => Ideas.forcedWithinB c (gainGoal c base v) q (2 * n)) with
    | some n => n
    | none => 0

/-- Two positions with the same board and the same side to move. -/
def sameSpot (a b : Position) : Bool := a.turn == b.turn && allSquares.all (fun u => a.board u == b.board u)

/-- **seen**: how many times the game's line holds this position. -/
def seenIn (hist : List Position) (p : Position) : Nat := (hist.filter (sameSpot p)).length

/-- **ago**: a number read on the game's position `k` moves of ours back (the line ends with the position now; 0 before
the game's start). -/
def agoIn (hist : List Position) (k : Nat) (f : Position → Nat) : Nat :=
  match hist.reverse[2 * k]? with | some q => f q | none => 0

/-- **without**: the board with every non-king piece on the set's squares taken off (kings are never removed); the rest of the
position as it is (the evaluator builds it relaxed -- a king left attacked is fine -- so there is no legality here either). -/
def dropPieces (p : Position) (X : SquareSet) : Position :=
  { p with board := fun u => match p.board u with
      | some q => if X u && decide (q.kind ≠ PieceKind.king) then none else some q
      | none => none }

/-- A number read on the board with the set's pieces gone (the set is read on the board as it is; the number's names are those
of the board that is left, the renderer rebinds them with `kingSq` and `nthPiece`). -/
def withoutIn (p : Position) (X : SquareSet) (f : Position → Nat) : Nat := f (dropPieces p X)

/-- Side `c`'s king, when there is one. -/
def kingSq (p : Position) (c : Color) : OSq :=
  (allSquares.filter (fun u => match p.board u with | some q => q.color == c && q.kind == PieceKind.king | none => false)).head?

/-- **after**: a number read on the board after the piece on `a` moves to `b` (whatever stood on `b` taken, one's own piece too; a pawn
reaching the last rank becomes a queen; the other side to move). Legality is not asked, as in the evaluator's relaxed board. -/
def afterMove (p : Position) (a b : OSq) (f : Position → Nat) : Nat :=
  match a, b with
  | some x, some y =>
    match p.board x with
    | none => 0
    | some q =>
      let q' : Piece := if q.kind == PieceKind.pawn && (y.2.val == 0 || y.2.val == 7) then { q with kind := PieceKind.queen } else q
      f { p with board := fun u => if u == y then some q' else if u == x then none else p.board u,
                 turn := Color.opposite q.color, enPassant := none }
  | _, _ => 0

/-- **delta**: a number now minus the same a ply ago, on an offset of 64 (64 when unchanged or when the line has no earlier position). -/
def deltaIn (hist : List Position) (p : Position) (f : Position → Nat) : Nat :=
  match hist.reverse[1]? with | some q => f p + 64 - f q | none => 64

/-- **gained**: the squares of a set now and not a ply ago. -/
def gainedIn (hist : List Position) (p : Position) (f : Position → SquareSet) : SquareSet :=
  match hist.reverse[1]? with | some q => fun u => f p u && !(f q u) | none => fun _ => false

/-- **lost**: the squares of a set a ply ago and not now. -/
def lostIn (hist : List Position) (p : Position) (f : Position → SquareSet) : SquareSet :=
  match hist.reverse[1]? with | some q => fun u => f q u && !(f p u) | none => fun _ => false

/-- **mobility** (Kaveh 2026-10-15, ANALYST_PLAN 40.2): the squares the piece on `s` can legally move to where it is safe by the
exchange -- after the move nothing is won by capturing on the square it lands on. "The escape squares of a piece." -/
def mobilityOf (p : Position) (s : Square) : SquareSet := fun t =>
  movesOf p s t && decide (afterMove p (some s) (some t) (fun q => seeOf q t) = 0)

/-- A move lands safely: after it, the exchange on its destination wins nothing for the other side. -/
def landsSafe (p : Position) (m : Move) : Bool := decide (seeOf (applyMove p m) m.destination = 0)

/-- Side `c` can attack square `s` within `k` of its own moves (the other side passes), every landing square safe by the
exchange; already attacked counts (0 moves). -/
def attackableIn (c : Color) (s : Square) : Nat → Position → Bool
  | 0, p => decide (0 < attackerCount p c s)
  | k + 1, p => decide (0 < attackerCount p c s) ||
      (legalMoves { p with turn := c, enPassant := none }).any (fun m =>
        landsSafe { p with turn := c, enPassant := none } m &&
        attackableIn c s k (applyMove { p with turn := c, enPassant := none } m))

/-- **attackable** (the REACH of ANALYST_PLAN 40.2; the name `reach` was taken by the king's walk): 1 when the side that does not own
the piece on `s` can attack `s` within `k` moves, every landing square safe by the exchange; else 0. The route and the blocking
squares are the evaluator's answer (vocab.rs `reach_route`); this is the yes/no. -/
def attackableOf (p : Position) (s : Square) (k : Nat) : Nat :=
  match ownerOf p s with
  | some o => if attackableIn (Color.opposite o) s k p then 1 else 0
  | none => 0

/-- **declined** (Kaveh 2026-10-15, "if they don't"): what the side to move can win by the exchange on a square OTHER than `s` --
its best capture if it declines to take on `s`. The split of a sacrifice: if they take, the exact solver's forced line; if they
don't, this. -/
def declinedOf (p : Position) (s : Square) : Nat :=
  (allSquares.filter (fun t => t != s && ownerOf p t == some (Color.opposite p.turn))).foldl (fun acc t => max acc (seeOf p t)) 0

/-- **how far apart** two numbers are. -/
def nDiff (a b : Nat) : Nat := if a ≤ b then b - a else a - b

/-- **least** (Kaveh 2026-10-15, the window as a tree): the least a number reads after ANY legal move of the side to move -- the
worst case over every reply, one ply (the forcing words are three-ply searches; this is the one-ply quantifier they are built
from). 0 when the side to move has no legal move (nothing is guaranteed then). `p` and the names are rebound inside, as for
`after`. -/
def leastOver (p : Position) (f : Position → Nat) : Nat :=
  match (legalMoves p).map (fun m => f (applyMove p m)) with
  | [] => 0
  | x :: xs => xs.foldl min x

/-- **most**: the most a number reads after any legal move of the side to move -- the best case over every reply; 0 when there
is no legal move. -/
def mostOver (p : Position) (f : Position → Nat) : Nat :=
  ((legalMoves p).map (fun m => f (applyMove p m))).foldl max 0

/-- The bridge from the word to the quantifier: whatever `least` reads, every reply reads at least that. -/
theorem leastOver_le (p : Position) (f : Position → Nat) (m : Move) (hm : m ∈ legalMoves p) :
    leastOver p f ≤ f (applyMove p m) := by
  unfold leastOver
  have hmem : f (applyMove p m) ∈ (legalMoves p).map (fun m => f (applyMove p m)) := List.mem_map_of_mem hm
  generalize (legalMoves p).map (fun m => f (applyMove p m)) = l at hmem
  match l, hmem with
  | x :: xs, hmem =>
    simp only
    have key : ∀ (ys : List Nat) (a : Nat), ys.foldl min a ≤ a ∧ ∀ y ∈ ys, ys.foldl min a ≤ y := by
      intro ys
      induction ys with
      | nil => intro a; simp
      | cons y ys ih =>
        intro a
        simp only [List.foldl_cons]
        obtain ⟨h1, h2⟩ := ih (min a y)
        refine ⟨le_trans h1 (Nat.min_le_left _ _), ?_⟩
        intro z hz
        simp only [List.mem_cons] at hz
        rcases hz with rfl | hz
        · exact le_trans h1 (Nat.min_le_right _ _)
        · exact h2 z hz
    simp only [List.mem_cons] at hmem
    rcases hmem with h | h
    · rw [h]; exact (key xs x).1
    · exact (key xs x).2 _ h

/-- Every reply reads at most what `most` reads. -/
theorem le_mostOver (p : Position) (f : Position → Nat) (m : Move) (hm : m ∈ legalMoves p) :
    f (applyMove p m) ≤ mostOver p f := by
  unfold mostOver
  have hmem : f (applyMove p m) ∈ (legalMoves p).map (fun m => f (applyMove p m)) := List.mem_map_of_mem hm
  generalize (legalMoves p).map (fun m => f (applyMove p m)) = l at hmem
  have key : ∀ (ys : List Nat) (a : Nat), a ≤ ys.foldl max a ∧ ∀ y ∈ ys, y ≤ ys.foldl max a := by
    intro ys
    induction ys with
    | nil => intro a; simp
    | cons y ys ih =>
      intro a
      simp only [List.foldl_cons]
      obtain ⟨h1, h2⟩ := ih (max a y)
      refine ⟨le_trans (Nat.le_max_left _ _) h1, ?_⟩
      intro z hz
      simp only [List.mem_cons] at hz
      rcases hz with rfl | hz
      · exact le_trans (Nat.le_max_right _ _) h1
      · exact h2 z hz
  exact (key l 0).2 _ hmem

end Ideas.Vocabulary
