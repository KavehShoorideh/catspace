import Ideas.Fork
import Ideas
import Chess.FENText

/-!
# `ideas` -- the detector binary

A line protocol over stdin/stdout for the Python search. It evaluates the compiled hypothesis
functions -- the same definitions the soundness theorems are about -- and never calls Lean's
kernel or searches.

    backrank <fen>   -> one `cand` line per rook/queen move onto the defender's home rank,
                        then `end`:
                        cand d1d8 MATE targets= heavy=1 onBackRank=1 legal=1 ...
                        cand d1d8 ONEAWAY noDefender targets=e8 heavy=1 ...
                        cand a1a8 FAIL 2 targets= heavy=1 ...
    legal <fen>      -> the legal moves in UCI, space separated (bridge check against python-chess)
    checkmate <fen>  -> 1 or 0 (bridge check; slow: the kernel's generator filters ~4,600 candidates)
-/

open Chess Ideas

def sqName (s : Square) : String :=
  String.ofList [Char.ofNat ('a'.toNat + s.1.val), Char.ofNat ('1'.toNat + s.2.val)]

def moveUci : Move → String
  | .normal s t => sqName s ++ sqName t
  | .enPassant s t => sqName s ++ sqName t
  | .promotion s t k =>
      sqName s ++ sqName t ++
        (match k with | .queen => "q" | .rook => "r" | .bishop => "b" | .knight => "n")
  | .whiteKingCastle => "e1g1"
  | .whiteQueenCastle => "e1c1"
  | .blackKingCastle => "e8g8"
  | .blackQueenCastle => "e8c8"

/-- Parse a FEN and read its board into an array once (`memoBoard`, equal to the parsed board by
`memoBoard_eq`; the array is bound here so every lookup shares it). -/
def hexDigit (c : Char) : Option Nat :=
  if '0' ≤ c ∧ c ≤ '9' then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c ∧ c ≤ 'f' then some (c.toNat - 'a'.toNat + 10)
  else if 'A' ≤ c ∧ c ≤ 'F' then some (c.toNat - 'A'.toNat + 10)
  else none

/-- A hex bitboard as a machine word (a `Nat` above 2^63 is a bignum; every bit test would allocate). -/
def parseHex (s : String) : Option UInt64 :=
  s.toList.foldlM (fun acc c => (hexDigit c).map fun d => acc * 16 + d.toUInt64) 0

def testBit (bb : UInt64) (i : Nat) : Bool := (bb >>> i.toUInt64) &&& 1 == 1

/-- The 64 squares from python-chess bitboards (bit `rank * 8 + file`): pawns, knights, bishops, rooks,
queens, kings, and the white occupancy. -/
@[noinline] def arrayOfBitboards (bbs : List UInt64) (white : UInt64) : Array (Option Piece) :=
  let kinds : List (UInt64 × PieceKind) :=
    bbs.zip [.pawn, .knight, .bishop, .rook, .queen, .king]
  Array.ofFn fun i : Fin 64 =>
    let sq := sqOfIndex i                      -- (file, rank)
    let bit := sq.2.val * 8 + sq.1.val
    match kinds.find? (fun (bb, _) => testBit bb bit) with
    | some (_, k) => some ⟨if testBit white bit then .white else .black, k⟩
    | none => none

theorem arrayOfBitboards_size (bbs : List UInt64) (white : UInt64) : (arrayOfBitboards bbs white).size = 64 := by
  simp [arrayOfBitboards]

/-- `bb <w|b> <castling|-> <ep|-> <pawns> <knights> <bishops> <rooks> <queens> <kings> <white>` in hex. -/
def readBitboards (fields : List String) : Option Position :=
  match fields with
  | [turn, castling, ep, p, n, b, r, q, k, w] => do
      let t ← parseTurn turn.toList
      let c ← parseCastling castling.toList
      let e ← parseEnPassant ep.toList
      let bbs ← [p, n, b, r, q, k].mapM parseHex
      let wh ← parseHex w
      let arr := arrayOfBitboards bbs wh
      pure { board := boardOfArray arr (arrayOfBitboards_size bbs wh), turn := t, castling := c,
             enPassant := e, halfmove := 0, fullmove := 1 }
  | _ => none

def readPosition (fen : String) : Option Position := do
  let p ← parseFen fen
  let arr := boardArray p.board
  pure { p with board := boardOfArray arr (boardArray_size p.board) }

/-- Candidate parameters: our rook or queen on `a`, a square `c` of the defender's home rank its
geometry reaches on an empty board, the defender's king on `k` (on that rank). -/
def candidates (p : Position) : List BackRankParams :=
  let d := Color.opposite p.turn
  let hr := BackRank.homeRank d
  match kingSquare p.board d with
  | none => []
  | some k =>
      if k.2 ≠ hr then [] else
      allSquares.flatMap fun a =>
        match p.board a with
        | some pc =>
            if pc.color = p.turn ∧ (pc.kind = .rook ∨ pc.kind = .queen) then
              allFiles.filterMap fun f =>
                let c : Square := (f, hr)
                let reach := if pc.kind = .rook then rookGeometryB a c else queenGeometryB a c
                if c ≠ k ∧ reach = true then some ⟨a, c, k⟩ else none
            else []
        | none => []

def report (p : Position) (ps : BackRankParams) : String :=
  let vals := backRankMate.evalUntil p ps
  let failing := (vals.filter (fun x => !x.2)).map (·.1)
  let status := match failing with
    | [] => "MATE"
    | [h] => s!"ONEAWAY {h}"
    | fs => s!"FAIL {fs.length}"
  let tg := match failing with
    | [h] => ",".intercalate ((BackRank.targets p ps h).map sqName)
    | _ => ""
  let hs := " ".intercalate (vals.map fun (n, b) => s!"{n}={if b then 1 else 0}")
  s!"cand {sqName ps.a}{sqName ps.c} {status} targets={tg} {hs}"

/-- Passer candidates: every pawn of the side to move. -/
def passerCandidates (p : Position) : List PasserParams :=
  allSquares.filterMap fun sq =>
    match p.board sq with
    | some pc => if pc.color = p.turn ∧ pc.kind = .pawn then some ⟨sq⟩ else none
    | none => none

def reportPasser (p : Position) (ps : PasserParams) : String :=
  let vals := unstoppablePasser.evalUntil p ps
  let failing := (vals.filter (fun x => !x.2)).map (·.1)
  let n := Passer.pushes p.turn ps.sq
  let status := match failing with
    | [] => s!"PROVED plies={2 * n - 1}"
    | [h] => s!"ONEAWAY {h}"
    | fs => s!"FAIL {fs.length}"
  let tg := match failing with
    | [h] => ",".intercalate ((Passer.targets p ps h).map sqName)
    | _ => ""
  let hs := " ".intercalate (vals.map fun (n, b) => s!"{n}={if b then 1 else 0}")
  s!"cand {sqName ps.sq} {status} targets={tg} {hs}"

/-- Both forms are theorems, so a pawn is PROVED when either form's hypotheses all hold (the second form's
room count over-approximates the passer's future attacks and loses some positions the first form keeps).
Reported: the second form's hypotheses; `form=1` marks a proof by the first form. -/
def reportPasser2 (p : Position) (ps : PasserParams) : String :=
  let vals := unstoppablePasser2.evalUntil p ps
  let failing := (vals.filter (fun x => !x.2)).map (·.1)
  let n := Passer.pushes p.turn ps.sq
  let byFirst := failing ≠ [] && unstoppablePasser.holds p ps
  let status :=
    if byFirst then s!"PROVED plies={2 * n - 1} form=1"
    else match failing with
    | [] => s!"PROVED plies={2 * n - 1}"
    | [h] => s!"ONEAWAY {h}"
    | fs => s!"FAIL {fs.length}"
  let tg := match failing with
    | [h] => if byFirst then "" else ",".intercalate ((Passer2.targets p ps h).map sqName)
    | _ => ""
  let hs := " ".intercalate (vals.map fun (n, b) => s!"{n}={if b then 1 else 0}")
  s!"cand {sqName ps.sq} {status} targets={tg} {hs}"

def handle (line : String) : List String :=
  match (line.trimAscii.toString.splitOn " ").filter (· ≠ "") with
  | ["ping"] => ["pong"]
  | "parse" :: rest =>
      match readPosition (" ".intercalate rest) with
      | some p => [toString (candidates p).length]
      | none => ["error bad fen"]
  | "backrank" :: rest =>
      match readPosition (" ".intercalate rest) with
      | some p => (candidates p).map (report p) ++ ["end"]
      | none => ["error bad fen", "end"]
  | "bb" :: rest =>                                  -- one line: candidates separated by " | " (no sentinel)
      match readBitboards rest with
      | some p => [" | ".intercalate ((candidates p).map (report p))]
      | none => ["error bad bitboards"]
  | "passer" :: rest =>                              -- one line: candidates separated by " | "
      match readBitboards rest with
      | some p => [" | ".intercalate ((passerCandidates p).map (reportPasser p))]
      | none => ["error bad bitboards"]
  | "passer2" :: rest =>
      match readBitboards rest with
      | some p => [" | ".intercalate ((passerCandidates p).map (reportPasser2 p))]
      | none => ["error bad bitboards"]
  | "passerfen" :: rest =>
      match readPosition (" ".intercalate rest) with
      | some p => [" | ".intercalate ((passerCandidates p).map (reportPasser p))]
      | none => ["error bad fen"]
  | "bblegal" :: rest =>
      match readBitboards rest with
      | some p => [" ".intercalate ((legalMoves p).map moveUci)]
      | none => ["error bad bitboards"]
  | "legal" :: rest =>
      match readPosition (" ".intercalate rest) with
      | some p => [" ".intercalate ((legalMoves p).map moveUci)]
      | none => ["error bad fen"]
  | "checkmate" :: rest =>
      match readPosition (" ".intercalate rest) with
      | some p => [if checkmateB p then "1" else "0"]
      | none => ["error bad fen"]
  | "fork" :: uci :: kstr :: rest =>                 -- fork <uci> <k> <fen>: is a gain of k forced three plies after the move (our capture, their reply)?
      match readPosition (" ".intercalate rest), kstr.toInt? with
      | some p, some k =>
          match (legalMoves p).find? (fun m => moveUci m == uci) with
          | some m => [if localCheck p m k then s!"CERTIFIED {k}" else s!"FAIL {k}"]
          | none => ["error illegal move"]
      | _, _ => ["error bad fen or k"]
  | _ => ["error unknown command"]

/-- Time `f` over `reps` runs; `f` gets the iteration index and the results are folded into a ref, so
nothing is hoisted or elided. -/
def benchOne (name : String) (reps : Nat) (f : Nat → Bool) : IO String := do
  let acc ← IO.mkRef false
  let t0 ← IO.monoNanosNow
  for i in [:reps] do
    let v := f i
    acc.modify (fun a => a != v)
  let t1 ← IO.monoNanosNow
  let _ ← acc.get
  pure s!"  {name}: {(t1 - t0) / 1000 / reps} us"

/-- `bench <reps> <fen>`: each hypothesis of the first candidate, and the primitives, timed alone. -/
def bench (reps : Nat) (fen : String) : IO (List String) := do
  match readPosition fen with
  | none => pure ["error bad fen"]
  | some p =>
    match candidates p with
    | [] => pure ["no candidates"]
    | ps :: _ =>
      let q := BackRank.after p ps
      -- a square that varies with the iteration, so nothing is hoisted
      let sq (i : Nat) : Square := (⟨i % 8, Nat.mod_lt _ (by decide)⟩, ⟨(i / 8) % 8, Nat.mod_lt _ (by decide)⟩)
      let mut out := [s!"candidate {sqName ps.a}{sqName ps.c} king {sqName ps.k}"]
      for h in backRankMate.hyps do
        out := out ++ [← benchOne h.name reps (fun i => h.test p { ps with k := if i % 2 = 0 then ps.k else ps.k })]
      out := out ++ [← benchOne "applyMove" reps (fun i => (applyMove p (.normal ps.a (if i % 2 = 0 then ps.c else ps.c))).turn == .white)]
      out := out ++ [← benchOne "kingSquare" reps (fun i => (kingSquare q.board (if i % 2 = 0 then q.turn else q.turn)).isSome)]
      out := out ++ [← benchOne "isAttackedB' (64 squares)" reps (fun i => isAttackedB' q (Color.opposite q.turn) (sq i))]
      out := out ++ [← benchOne "isAttackedB kernel (64 squares)" reps (fun i => isAttackedB q (Color.opposite q.turn) (sq i))]
      out := out ++ [← benchOne "pieceAttacksB' (64 sources)" reps (fun i => pieceAttacksB' q (Color.opposite q.turn) (sq i) ps.k)]
      out := out ++ [← benchOne "rayBetween (64)" reps (fun i => (rayBetween (sq i) ps.k).isEmpty)]
      out := out ++ [← benchOne "board lookup (64)" reps (fun i => (q.board (sq i)).isSome)]
      out := out ++ [← benchOne "kingGeometryB (64)" reps (fun i => kingGeometryB ps.k (sq i))]
      out := out ++ [← benchOne "candidates" reps (fun i => (candidates (if i % 2 = 0 then p else p)).isEmpty)]
      out := out ++ [← benchOne "readPosition(fen)" reps (fun i => (readPosition (if i % 2 = 0 then fen else fen)).isSome)]
      let bbf := ["w", "-", "-", "e310080c10e300", "240000240000", "410000000000020", "2100000000000081", "800000000000008", "4000000000000010", "ffff"]
      out := out ++ [← benchOne "readBitboards" reps (fun i => (readBitboards (if i % 2 = 0 then bbf else bbf)).isSome)]
      pure out

partial def loop (stdin : IO.FS.Stream) (stdout : IO.FS.Stream) (timing : Bool) : IO Unit := do
  let line ← stdin.getLine
  if line.isEmpty then return
  let outs ← match (line.trimAscii.toString.splitOn " ").filter (· ≠ "") with
    | "bench" :: reps :: rest => bench (reps.toNat?.getD 100) (" ".intercalate rest)
    | _ => pure (handle line)
  for out in outs do
    stdout.putStrLn out
  stdout.flush
  loop stdin stdout timing

def main (args : List String) : IO Unit := do
  loop (← IO.getStdin) (← IO.getStdout) (args.contains "--time")

