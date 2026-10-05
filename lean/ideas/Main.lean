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
  let vals := backRankMate.evalHyps p ps
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

def handle (line : String) : List String :=
  match (line.trimAscii.toString.splitOn " ").filter (· ≠ "") with
  | "backrank" :: rest =>
      match readPosition (" ".intercalate rest) with
      | some p => (candidates p).map (report p) ++ ["end"]
      | none => ["error bad fen", "end"]
  | "legal" :: rest =>
      match readPosition (" ".intercalate rest) with
      | some p => [" ".intercalate ((legalMoves p).map moveUci)]
      | none => ["error bad fen"]
  | "checkmate" :: rest =>
      match readPosition (" ".intercalate rest) with
      | some p => [if checkmateB p then "1" else "0"]
      | none => ["error bad fen"]
  | _ => ["error unknown command"]

partial def loop (stdin : IO.FS.Stream) (stdout : IO.FS.Stream) : IO Unit := do
  let line ← stdin.getLine
  if line.isEmpty then return
  for out in handle line do
    stdout.putStrLn out
  stdout.flush
  loop stdin stdout

def main : IO Unit := do
  loop (← IO.getStdin) (← IO.getStdout)
