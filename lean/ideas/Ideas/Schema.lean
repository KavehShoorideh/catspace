import Chess.Basic

/-!
# The `Idea` schema

Every chess idea -- a tactical lemma or a strategic concept -- is an instance of `Idea`:
a list of individually evaluable hypotheses, a protagonist, a tier and a horizon. The planner
never searches for what an idea already says: when every hypothesis holds, the conclusion is a
fact (proved once, offline); when exactly one fails, that hypothesis names the search target
(Kaveh 2026-10-04: "a bishop blocking my pawn's path -- I don't need a search to know my focus
should be getting rid of that bishop").
-/

namespace Ideas

open Chess

/-- `unconditional`: holds whatever the opponent does. `forcing`: every reply is bad for them,
but the proof branches on replies. `conjecture`: a statistical lift only, never proved. -/
inductive Tier where
  | unconditional
  | forcing
  | conjecture
  deriving DecidableEq, Repr

/-- One hypothesis: a name the planner and the renderer can point at, and a decidable test. -/
structure Hyp (P : Type) where
  name : String
  test : Position → P → Bool

/-- An idea: parameters `P`, hypotheses, the piece it is about, tier, horizon in plies. The
soundness theorem (all hypotheses ⇒ the conclusion) lives beside each instance. -/
structure Idea (P : Type) where
  id : String
  hyps : List (Hyp P)
  protagonist : P → Square
  tier : Tier
  horizon : Nat

namespace Idea

variable {P : Type}

/-- Every hypothesis holds. -/
def holds (I : Idea P) (p : Position) (ps : P) : Bool :=
  I.hyps.all (fun h => h.test p ps)

/-- The hypotheses that fail, by name. Empty iff `holds`. -/
def failing (I : Idea P) (p : Position) (ps : P) : List String :=
  (I.hyps.filter (fun h => !h.test p ps)).map (·.name)

/-- Each hypothesis with its value, in order (what the detector reports). -/
def evalHyps (I : Idea P) (p : Position) (ps : P) : List (String × Bool) :=
  I.hyps.map (fun h => (h.name, h.test p ps))

/-- The hypotheses in order, stopping after the second failure: the verdicts MATE (none fail) and
ONEAWAY (exactly one fails) need every hypothesis; FAIL needs only two, so the expensive tests at the
end of the list are skipped where the cheap ones already settle it. -/
def evalUntil (I : Idea P) (p : Position) (ps : P) : List (String × Bool) :=
  go I.hyps 0 []
where
  go : List (Hyp P) → Nat → List (String × Bool) → List (String × Bool)
    | [], _, acc => acc.reverse
    | h :: hs, fails, acc =>
        let v := h.test p ps
        let fails' := if v then fails else fails + 1
        if fails' ≥ 2 then ((h.name, v) :: acc).reverse
        else go hs fails' ((h.name, v) :: acc)

end Idea

/-- The board read once into an array: the same function, constant-time lookups for the
detector (FEN parsing builds it as 64 nested updates). -/
def sqOfIndex (i : Fin 64) : Square :=
  (⟨i.val / 8, by omega⟩, ⟨i.val % 8, by omega⟩)

/-- A board backed by a 64-entry array. `noinline`: inlined, the compiler floats the array's construction
into the returned closure and rebuilds it on every lookup (100-300 ms a query, found 2026-10-04). -/
@[noinline] def boardOfArray (arr : Array (Option Piece)) (h : arr.size = 64) : Board :=
  fun s => arr[s.1.val * 8 + s.2.val]'(by
    have h1 := s.1.isLt
    have h2 := s.2.isLt
    omega)

/-- The board's 64 squares, read once. -/
def boardArray (b : Board) : Array (Option Piece) :=
  Array.ofFn (fun i : Fin 64 => b (sqOfIndex i))

theorem boardArray_size (b : Board) : (boardArray b).size = 64 := by
  simp [boardArray]

def memoBoard (b : Board) : Board :=
  boardOfArray (boardArray b) (boardArray_size b)

theorem memoBoard_eq (b : Board) : memoBoard b = b := by
  funext s
  simp only [memoBoard, boardOfArray, boardArray, Array.getElem_ofFn, sqOfIndex]
  congr 1
  obtain ⟨f, r⟩ := s
  have h1 := f.isLt
  have h2 := r.isLt
  ext
  · simp only
    omega
  · simp only
    omega

end Ideas
