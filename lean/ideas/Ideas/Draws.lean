import Ideas.Game
import Chess.Game

/-!
# THE DRAW RULES OVER A GAME RECORD (ANALYST_PLAN 75.4; Kaveh 2026-10-12: "in every frame that we analyze, we need to carry into that frame the move counter -- the
50-move clock -- and the triple position")

RULES, not knowledge. The rule kernel (chess-lean) already states the draws, over a game's HISTORY -- a list of positions, the current one last (`Chess.History`):
the position's identity `Chess.repetitionKey` (board, side to move, castling rights, and an en-passant square only when an en-passant capture is LEGAL:
`Chess.effectiveEnPassant`; the clocks omitted), `Chess.keyOccurrences`, the third occurrence `Chess.threefoldClaimable`; in `Chess.Game` the clock's
`Chess.fiftyMoveClaimable` (`100 ≤ halfmove`), `Chess.gameStatus` reading checkmate before either; and the clock's events in `Chess.applyMove_halfmove` (a pawn
move or a capture sets it to 0, any other move adds one).

This file names them for the machinery's GAME RECORD (discovery/game.py `Record`): the positions BEFORE the current one, with how often each identity occurred
among them. Generic over a `Game` with an identity `key` and a `clock`: the COUNT INVARIANT (`Record.Sound`), kept by every push (`Record.push_sound`) and so along
every step of the game (`Record.step_sound`); the third occurrence (`Record.threefold`), the clock at its limit (`fifty`) and the draw by rule in the kernel's order
(`Record.drawByRule`: a mate on the move comes first). For chess (the instance `Ideas.Game.chess`), the record's readings ARE the kernel's: `chess_threefold_iff`
(a sound record's third occurrence is `Chess.threefoldClaimable` of the history `before ++ [p]`), `chess_fifty_iff`, `chess_clock`. Each proof unfolds the kernel's
definitions; nothing about the board is proved here.

What this file does NOT prove: that a position of a line repeated ONCE (the chase's CYCLE claim, plan 61 C4) reaches the third occurrence as the chase reads it. That
is a derived lemma, not the rule; its strict form (the replies their only moves, the identity a congruence) is proved generically below (`forced_cycle_threefold`), the
chase's own reading is not -- the store keeps it a CONJECTURE (lemmas/repetition.json).
-/

namespace Ideas.Game

open Chess

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

/-- THE FIFTY-MOVE CLOCK at its limit: a hundred half-moves with no clock event. -/
def fifty (clock : State → Nat) (p : State) : Bool :=
  decide (100 ≤ clock p)

/-- THE DRAW BY RULE, in the kernel's order (`Chess.gameStatus`: a checkmate on the move comes first): the mover is not mated, and the position occurs a third time
or the clock is at its limit. -/
def Record.drawByRule (g : Game State Move) (key : State → Key) (clock : State → Nat) (r : Record State Key) (p : State) : Bool :=
  !(g.mated p) && (r.threefold key p || fifty clock p)

omit [DecidableEq Key] in
/-- A mate on the move is never a draw by rule. -/
theorem Record.mated_not_draw (g : Game State Move) (key : State → Key) (clock : State → Nat) (r : Record State Key) {p : State} (h : g.mated p = true) :
    r.drawByRule g key clock p = false := by
  simp [Record.drawByRule, h]

/-! ## Chess: the record's readings are the kernel's -/

/-- The third occurrence of a sound record is the kernel's `Chess.threefoldClaimable` of the history: the positions before, then the current one. -/
theorem chess_threefold_iff (r : Record Position RepetitionKey) (p : Position) (h : r.Sound repetitionKey) :
    r.threefold repetitionKey p = true ↔ threefoldClaimable (r.before ++ [p]) := by
  have hocc : keyOccurrences (repetitionKey p) (r.before ++ [p]) = r.count (repetitionKey p) + 1 := by
    have e := occurrences_append_one repetitionKey (repetitionKey p) r.before p
    rw [if_pos rfl] at e
    rw [h (repetitionKey p)]
    exact e
  simp only [Record.threefold, threefoldClaimable, threefoldClaimableB, List.getLast?_concat, decide_eq_true_eq]
  rw [hocc]

/-- The clock at its limit is the kernel's `Chess.fiftyMoveClaimable`. -/
theorem chess_fifty_iff (p : Position) : fifty Position.halfmove p = true ↔ fiftyMoveClaimable p := by
  simp [fifty, fiftyMoveClaimable]

/-- THE CLOCK'S EVENTS (the fifty-move rule's): a pawn move or a capture. -/
def clockResets (p : Position) (m : Chess.Move) : Bool :=
  moveIsPawn p m || moveIsCapture p m

/-- After a move the kernel's clock is 0 on a clock event, else one more (`Chess.applyMove_halfmove`). -/
theorem chess_clock (p : Position) (m : Chess.Move) :
    (chess.apply p m).halfmove = if clockResets p m then 0 else p.halfmove + 1 :=
  applyMove_halfmove p m

/-- THE DRAW BY RULE IN CHESS: the instance's mate (the kernel's legal moves and check), the kernel's identity, the kernel's clock. -/
def chessDrawByRule (r : Record Position RepetitionKey) (p : Position) : Bool :=
  r.drawByRule chess repetitionKey Position.halfmove p

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
