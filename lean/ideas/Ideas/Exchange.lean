import Mathlib.Tactic

/-!
# The exchange law, part 1: the capture game on one square

Kaveh 2026-10-07: "the tactics are basically exceptions to the exchange theorem". The theorem has two parts. This file is
the first: the MATHEMATICS of the exchange, with no chess in it -- a square holds a piece worth `onS`; the mover owns the
pieces `mine` (their values) that can capture there, the opponent owns `theirs`; the mover may capture with ANY of its
pieces (its piece then stands on the square, worth its own value, and the opponent is the mover) or STOP. The value is
the mover's best net material from here, the opponent playing the same game. Both sides free to choose the capturer, so
the value is the true minimax of the capture game -- no "cheapest first is optimal" lemma is needed, and the kernel can
compute it on the small lists a square carries. `cheapest` is the classical static exchange (cheapest piece first), the
Python prototype's read; where the two differ the order of capture matters.

Part 2 (`Ideas/ExchangeLaw.lean`, to come) states when a real chess position's game on a square IS this game -- the
honest-frame hypotheses: every capture legal, no king among the participants, no x-ray behind one, no promotion, no
capture elsewhere that pays, no in-between check, no blockable attacker, no arriving defender -- and proves that the
real material outcome against every legal reply is at least this value. Each hypothesis that fails in a real position
is a named tactic (the pin, the x-ray, the overload, the zwischenzug ...), measured in `exchange_law.py`.
-/

namespace Ideas.Exchange

/-- The value of the capture game to the mover, with `fuel` at least the number of pieces: `onS` the value of the piece
on the square, `mine` the mover's pieces able to capture there (values), `theirs` the opponent's. Stop (0) or take with
some piece and face the opponent's best answer in the game where that piece stands on the square. -/
def valueF : Nat → Int → List Int → List Int → Int
  | 0, _, _, _ => 0
  | _ + 1, _, [], _ => 0
  | n + 1, onS, a :: rest, theirs =>
      max 0 ((rest.map fun b => onS - valueF n b theirs (a :: rest.erase b)).foldl max (onS - valueF n a theirs rest))

def value (onS : Int) (mine theirs : List Int) : Int := valueF (mine.length + theirs.length) onS mine theirs

/-- The classical static exchange: both sides capture with their cheapest piece (the lists ascending). -/
def cheapestF : Nat → Int → List Int → List Int → Int
  | 0, _, _, _ => 0
  | _ + 1, _, [], _ => 0
  | n + 1, onS, a :: mine, theirs => max 0 (onS - cheapestF n a theirs mine)

def cheapest (onS : Int) (mine theirs : List Int) : Int := cheapestF (mine.length + theirs.length) onS mine theirs

theorem valueF_nonneg (n : Nat) (onS : Int) (mine theirs : List Int) : 0 ≤ valueF n onS mine theirs := by
  cases n with
  | zero => simp [valueF]
  | succ n =>
    cases mine with
    | nil => simp [valueF]
    | cons a rest => simp only [valueF]; exact le_max_left _ _

theorem value_nonneg (onS : Int) (mine theirs : List Int) : 0 ≤ value onS mine theirs := valueF_nonneg _ _ _ _

theorem cheapestF_nonneg (n : Nat) (onS : Int) (mine theirs : List Int) : 0 ≤ cheapestF n onS mine theirs := by
  cases n with
  | zero => simp [cheapestF]
  | succ n =>
    cases mine with
    | nil => simp [cheapestF]
    | cons a mine => simp only [cheapestF]; exact le_max_left _ _

theorem cheapest_nonneg (onS : Int) (mine theirs : List Int) : 0 ≤ cheapest onS mine theirs := cheapestF_nonneg _ _ _ _

/-- With nothing to capture with, nothing is won. -/
theorem value_nil (onS : Int) (theirs : List Int) : value onS [] theirs = 0 := by
  unfold value
  cases theirs <;> simp [valueF]

end Ideas.Exchange
