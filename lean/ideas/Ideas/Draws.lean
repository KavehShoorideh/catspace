import Ideas.Core.Draws
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

/- The generic record, its invariant, the third occurrence, the clock at a limit, the draw by rule and the cycle claim are in `Ideas/Core/Draws.lean` (PLAN 78.56). -/

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
  simp [fifty, fiftyAt, fiftyMoveClaimable]

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


end Ideas.Game
