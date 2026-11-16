import Ideas.Core.Game

/-!
# THE DRAW RULES OVER A GAME RECORD, game-free (PLAN 78.56 stage B; moved out of `Ideas/Draws.lean`)

The record of a game's earlier positions with the count of each identity, its soundness invariant, the third occurrence, the clock at a limit (`fiftyAt`: the move count is a
PARAMETER, `fifty` is the limit 100), the draw by rule, and the strict form of the cycle claim (`forced_cycle_threefold`). Generic over a `Game`, an identity `key` and a `clock`;
no chess file is imported. The kernel's readings (`chess_threefold_iff`, `chess_fifty_iff`, `chess_clock`) stay in `Ideas/Draws.lean`; the story is there (ANALYST_PLAN 75.4).
-/

namespace Ideas.Game

/-- A GAME RECORD: the positions of the game before the current one (`before`, oldest first) and how often each identity occurred among them (`count`). -/
structure Record (State Key : Type) where
  before : List State
  count : Key → Nat

variable {State Move Key : Type} [DecidableEq Key]

/-- The occurrences of an identity among a list of positions. -/
def occurrences (key : State → Key) (k : Key) (hs : List State) : Nat :=
  (hs.filter (fun p => decide (key p = k))).length

/-- THE COUNT INVARIANT: the record's counts are the occurrences of each identity among the positions before. -/
def Record.Sound (key : State → Key) (r : Record State Key) : Prop :=
  ∀ k, r.count k = occurrences key k r.before

/-- The empty record: no game before the position (a puzzle). -/
def Record.empty : Record State Key := ⟨[], fun _ => 0⟩

theorem Record.empty_sound (key : State → Key) : (Record.empty : Record State Key).Sound key := by
  intro k
  simp [Record.empty, occurrences]

/-- One more position played past: the current position `p` joins the positions before, its identity counted once more. -/
def Record.push (key : State → Key) (r : Record State Key) (p : State) : Record State Key :=
  ⟨r.before ++ [p], fun k => r.count k + if key p = k then 1 else 0⟩

theorem occurrences_append_one (key : State → Key) (k : Key) (hs : List State) (p : State) :
    occurrences key k (hs ++ [p]) = occurrences key k hs + if key p = k then 1 else 0 := by
  unfold occurrences
  rw [List.filter_append, List.length_append]
  by_cases h : key p = k <;> simp [h]

/-- A push keeps the count invariant. -/
theorem Record.push_sound (key : State → Key) (r : Record State Key) (p : State) (h : r.Sound key) : (r.push key p).Sound key := by
  intro k
  show r.count k + (if key p = k then 1 else 0) = occurrences key k (r.before ++ [p])
  rw [occurrences_append_one, h k]

/-- THE RECORD ALONG A STEP OF THE GAME (`Game.step`): from `p` with the record `r`, a step to `q` leaves `q` the record `r.push key p`, and the count invariant
holds there -- the play server's record and the chase's line are both built this way. -/
theorem Record.step_sound (g : Game State Move) (key : State → Key) {r : Record State Key} {p q : State} (h : r.Sound key) (_hs : g.step p q) :
    (r.push key p).Sound key :=
  r.push_sound key p h

/-- THE THIRD OCCURRENCE: the current position's identity occurred at least twice before it. -/
def Record.threefold (key : State → Key) (r : Record State Key) (p : State) : Bool :=
  decide (3 ≤ r.count (key p) + 1)

/-- THE CLOCK AT A LIMIT, the limit a parameter (a game's own move-count rule): `limit` half-moves with no clock event. -/
def fiftyAt (limit : Nat) (clock : State → Nat) (p : State) : Bool :=
  decide (limit ≤ clock p)

/-- THE FIFTY-MOVE CLOCK at its limit: a hundred half-moves with no clock event. -/
def fifty (clock : State → Nat) (p : State) : Bool :=
  fiftyAt 100 clock p

/-- THE DRAW BY RULE with the clock's limit a parameter: the mover is not mated, and the position occurs a third time or the clock is at the limit. -/
def Record.drawByRuleAt (limit : Nat) (g : Game State Move) (key : State → Key) (clock : State → Nat) (r : Record State Key) (p : State) : Bool :=
  !(g.mated p) && (r.threefold key p || fiftyAt limit clock p)

/-- THE DRAW BY RULE, in the kernel's order (`Chess.gameStatus`: a checkmate on the move comes first): the mover is not mated, and the position occurs a third time
or the clock is at its limit (a hundred half-moves). -/
def Record.drawByRule (g : Game State Move) (key : State → Key) (clock : State → Nat) (r : Record State Key) (p : State) : Bool :=
  r.drawByRuleAt 100 g key clock p

omit [DecidableEq Key] in
/-- A mate on the move is never a draw by rule, whatever the limit. -/
theorem Record.mated_not_draw_at (limit : Nat) (g : Game State Move) (key : State → Key) (clock : State → Nat) (r : Record State Key) {p : State} (h : g.mated p = true) :
    r.drawByRuleAt limit g key clock p = false := by
  simp [Record.drawByRuleAt, h]

omit [DecidableEq Key] in
/-- A mate on the move is never a draw by rule. -/
theorem Record.mated_not_draw (g : Game State Move) (key : State → Key) (clock : State → Nat) (r : Record State Key) {p : State} (h : g.mated p = true) :
    r.drawByRule g key clock p = false := by
  simp [Record.drawByRule, Record.drawByRuleAt, h]

/-! ## The cycle claim's obligation, in its strict form (plan 75.4 (2))

The chase's CYCLE claim (plan 61 C4) reads ONE repeat of a position of its line as a draw. That is a derived lemma: the rule's draw is the third occurrence. In its
STRICT form it follows from the rule by reasoning -- if the cycle's replies were THEIR ONLY MOVES and the identity a congruence of the game, replaying the cycle from the
repeat reaches the third occurrence (`forced_cycle_threefold`). The chase checks neither: it reads every reply (an AND), it does not require one, and the congruence is not
proved for the chess kernel (`Congruent`: the kernel's moves and apply do not read the clocks nor an unusable en-passant square). So the store keeps the claim a CONJECTURE. -/

/-- A CYCLE's steps, read in pairs: our move, then their reply. `run2` plays them from `p`. -/
def run2 (g : Game State Move) : State → List (Move × Move) → State
  | p, [] => p
  | p, (m, r) :: cs => run2 g (g.apply (g.apply p m) r) cs

/-- Each pair is legal where it is played, and THEIR REPLY IS THEIR ONLY MOVE there (the replies are forced in the strict sense). -/
def ForcedLine (g : Game State Move) : State → List (Move × Move) → Prop
  | _, [] => True
  | p, (m, r) :: cs => m ∈ g.moves p ∧ g.moves (g.apply p m) = [r] ∧ ForcedLine g (g.apply (g.apply p m) r) cs

/-- The record after playing a line of pairs from `p`: `p` and every position the line passes through are pushed, the end position not. -/
def pushLine (g : Game State Move) (key : State → Key) : Record State Key → State → List (Move × Move) → Record State Key
  | r, _, [] => r
  | r, p, (m, rr) :: cs => pushLine g key ((r.push key p).push key (g.apply p m)) (g.apply (g.apply p m) rr) cs

/-- THE IDENTITY IS A CONGRUENCE of the game: two positions of one identity have the same moves, and a move leads them to one identity. (For chess: the kernel's
`applyMove` and `legalMoves` do not read the clocks, nor an en-passant square no capture can use -- a statement about the kernel, NOT proved here.) -/
def Congruent (g : Game State Move) (key : State → Key) : Prop :=
  ∀ p p', key p = key p' → g.moves p = g.moves p' ∧ ∀ m, key (g.apply p m) = key (g.apply p' m)

omit [DecidableEq Key] in
/-- A forced line replays from any position of the same identity, forced again, to the same identity. -/
theorem forcedLine_replays (g : Game State Move) (key : State → Key) (hc : Congruent g key) :
    ∀ (cs : List (Move × Move)) (a b : State), key a = key b → ForcedLine g a cs → ForcedLine g b cs ∧ key (run2 g a cs) = key (run2 g b cs)
  | [], a, b, h, _ => ⟨trivial, h⟩
  | (m, r) :: cs, a, b, h, ⟨hm, hr, hcs⟩ => by
    obtain ⟨hmv, hap⟩ := hc a b h
    have h1 : key (g.apply a m) = key (g.apply b m) := hap m
    obtain ⟨hmv1, hap1⟩ := hc _ _ h1
    have h2 : key (g.apply (g.apply a m) r) = key (g.apply (g.apply b m) r) := hap1 r
    obtain ⟨ih1, ih2⟩ := forcedLine_replays g key hc cs _ _ h2 hcs
    refine ⟨⟨hmv ▸ hm, hmv1 ▸ hr, ih1⟩, ih2⟩

/-- Pushing never lowers a count. -/
theorem Record.push_count_le (key : State → Key) (r : Record State Key) (p : State) (k : Key) : r.count k ≤ (r.push key p).count k := by
  show r.count k ≤ r.count k + (if key p = k then 1 else 0)
  omega

theorem pushLine_count_le (g : Game State Move) (key : State → Key) :
    ∀ (cs : List (Move × Move)) (r : Record State Key) (p : State) (k : Key), r.count k ≤ (pushLine g key r p cs).count k
  | [], _, _, _ => le_refl _
  | (m, rr) :: cs, r, p, k => by
    refine le_trans ?_ (pushLine_count_le g key cs _ _ k)
    exact le_trans (r.push_count_le key p k) ((r.push key p).push_count_le key (g.apply p m) k)

/-- THE FORCED CYCLE REACHES THE THIRD OCCURRENCE (generic; the obligation of the chase's cycle claim in its strict form): `b` repeats `a` (one identity; the record before
`b` holds `a` once at least), and a line of pairs that is legal from `a` with THEIR REPLIES THEIR ONLY MOVES brings `a` back to its identity. Then, under a congruent
identity, replaying the line from `b` is forced again, and its end is a third occurrence: the record before it (the record before `b`, then `b` and the line's positions)
reads the threefold. -/
theorem forced_cycle_threefold (g : Game State Move) (key : State → Key) (hc : Congruent g key) (r : Record State Key) (a b : State)
    (cs : List (Move × Move)) (hne : cs ≠ []) (hab : key a = key b) (hseen : 1 ≤ r.count (key b))
    (hf : ForcedLine g a cs) (hback : key (run2 g a cs) = key a) :
    ForcedLine g b cs ∧ (pushLine g key r b cs).threefold key (run2 g b cs) = true := by
  obtain ⟨hfb, hkey⟩ := forcedLine_replays g key hc cs a b hab hf
  refine ⟨hfb, ?_⟩
  have hend : key (run2 g b cs) = key b := by rw [← hkey, hback, hab]
  obtain ⟨⟨m, rr⟩, cs', rfl⟩ : ∃ x cs', cs = x :: cs' := List.exists_cons_of_ne_nil hne
  have hpush : r.count (key b) + 1 ≤ (r.push key b).count (key b) := by
    show r.count (key b) + 1 ≤ r.count (key b) + (if key b = key b then 1 else 0)
    simp
  have hle := pushLine_count_le g key cs' ((r.push key b).push key (g.apply b m)) (g.apply (g.apply b m) rr) (key b)
  have hle2 := (r.push key b).push_count_le key (g.apply b m) (key b)
  simp only [Record.threefold, decide_eq_true_eq, hend]
  show 3 ≤ (pushLine g key ((r.push key b).push key (g.apply b m)) (g.apply (g.apply b m) rr) cs').count (key b) + 1
  omega

end Ideas.Game
