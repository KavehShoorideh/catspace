# Piece-relational tactical encoding — the proposal, and does it specify the board? (2026-07-31)

## Origin

Kaveh's brainstorm, this session: instead of (or alongside) square-token encoding
(JEPA T1, Chessformer [arXiv:2605.19091]), give every PIECE its own persistent
token, and derive the board from piece-to-piece attack/defense relations rather
than square occupancy. Motivation: attack/defend status is a **low-order, locally-
updating** quantity (changes by a small, describable amount per ply, one capture
or one blocking move at a time) — plausibly a better substrate for tactics than a
representation that has to re-derive "who defends whom" from raw occupancy on
every position. The session's mechanism sketch: query = "am I capturable," key =
"I defend this square," value = protection contribution — literally what
attention already computes structurally, if the tokens carry the right content.

Empirical context gathered the same session (see JOURNAL.md 2026-07-31), relevant
to whether this is worth building:
- Real, engine-verified "hanging piece" status (Stockfish committor swing across
  an actual capture, not a hand-coded heuristic) turned out to be **linearly
  decodable at essentially every layer of every trunk tested** (JEPA T1, lc0-256x10,
  lc0-512x15x8h) once the positive-class sample was large enough (n=239, not 96) —
  not gated behind late-layer computation as first (wrongly, under-powered)
  concluded.
- One-step chess dynamics (phi(s) -> phi(s')) are close to **linear** in these
  same trunks' own embedding spaces (Koopman/DMDc-style closed-form fit gets ~96%
  of what a trained nonlinear predictor gets, replicated at 5x data scale on two
  independently-trained networks).

Read together: square-token trunks already carry attack/defense structure
linearly, and their move-to-move dynamics are already close to linear. That's
evidence FOR the "attack/defense is a low-order quantity" intuition motivating
this proposal, but it's evidence gathered from the EXISTING square-token
representation, not proof that a piece-relational token stream is necessary or
better — that's the open empirical question a redesign like this would need to
answer, not something to assume going in.

## The proposal, as stated

- **One token per piece.** Not one token per square.
- **Token count / lifecycle**: open question. Either (a) a fixed/static token
  budget with a "deleted" marker flag for captured pieces (never reused), or
  (b) a variable-length token sequence that shrinks as pieces are captured.
  Promotion is explicitly unresolved: does the pawn's token become a queen token
  in place (identity persists, type changes), or does the pawn token get deleted
  and a new queen token spawn (identity resets)? Both have real consequences for
  a persistent-identity design and aren't yet decided.
- **Piece placement is specified relationally, not by coordinate**: for each
  piece, list (a) which pieces it currently attacks, (b) which pieces it
  currently protects, (c) which pieces it could attack or protect at the end of
  next turn (one-ply lookahead over potential targets), and (d) for long-range
  pieces (Q, R, B) the linear distance along the relevant rank/file/diagonal to
  whatever they're aligned with.
- **Claim to evaluate**: this fully specifies the board.

## Does it fully specify the board? No — concrete gaps

Working through what information survives this encoding and what doesn't:

**1. It's translation-invariant, and the board isn't.** The scheme only encodes
*relations between pieces that already interact or nearly interact*. Two pieces
with no attack/defense relation and no shared rank/file/diagonal contribute
**zero** information about their relative position to each other. Concretely: a
lone king and a friendly rook sitting in a corner, attacking nothing, attacked by
nothing, unaligned with any other piece, produce an identical encoding no matter
*where* on the board that pair sits — but board-edge proximity changes real
things (king escape squares, rook mobility, whether a later long-range attack
becomes possible at all). The encoding has no notion of absolute square, and nothing
here reconstructs one.

**2. Empty-square control is invisible.** "Which pieces it attacks/protects" is
piece-to-piece. A piece controlling a *key empty square* (an outpost, an escape
square, a square a king can't safely move to) contributes nothing to this scheme
unless another piece happens to occupy that square. Space, weak squares, and
potential-outpost control — all tactically load-bearing — aren't represented
unless "could attack/protect at end of next turn" is read broadly enough to
include empty destination squares, which the proposal as stated doesn't say.

**3. Auxiliary state is missing entirely.** Castling rights, en passant target,
side to move, halfmove clock — none of these are piece-attack relations. JEPA T1's
`glob` vector already carries the first three; this is a cheap, uncontroversial
fix (concatenate the same global vector), not a fundamental gap, but worth stating
so "fully specifies the board" isn't silently assumed to include it.

**4. Consequence: distinct legal positions can collide.** Combining 1-3: take any
position, translate an unaligned, non-interacting subset of pieces to a different
empty region of the board that creates no new alignments — the relational
encoding is unchanged, but the position (and its future move options, especially
near edges) is not the same position. The encoding is lossy, and demonstrably not
invertible to the original board.

## What this means for the design (not a verdict, a framing)

None of the above is a reason to drop the idea — it's a reason to be precise about
**what kind of representation this is**. It is not a board-state encoding in the
sense JEPA's square tokens or lc0's 112 planes are (bijective-enough to reconstruct
the position). It's closer to a **tactics-specific relational summary**: a
projection of the board that keeps exactly the structure the "who attacks/defends
whom, now and one ply out" question needs, and discards everything else by
construction.

**Decided (Kaveh, 2026-07-31): augment, not replace.** This token stream is an
auxiliary tactical channel — concatenated to or cross-attended with the existing
square tokens, not a substitute for them. The lossiness identified above is
therefore a feature, not a defect: the channel's whole job is to force the
tactical signal into an explicit, inspectable, low-order form (the "changes one
step at a time" property motivating the idea in the first place) without also
being asked to carry full board reconstruction — the square tokens already do
that, and per this session's positive-control results, do it well (piece identity,
and by extension board geometry, is trivially linearly decodable from them).

This resolves gaps 1-3 above as non-issues by design (the square-token trunk
supplies absolute position/space/auxiliary-state; this channel supplies the
attack/defense relational structure on top) and settles open question 4 below.
It also weighs on open question 1 (token lifecycle): as an auxiliary channel
riding alongside a trunk that already processes fixed-shape square-token batches,
a variable-length piece-token sequence would be the more disruptive/expensive
choice architecturally, favoring static-with-delete-marker unless there's a
concrete reason variable-length is worth the batching complexity.

## Open questions (unresolved, for discussion)

1. ~~Replace-vs-augment~~ **RESOLVED 2026-07-31: augment.** See above.
2. Token lifecycle: static+delete-marker vs. variable-length. Leaning
   static+delete-marker given (1) — see reasoning above — but not yet decided.
3. Promotion: identity-persists-with-type-change vs. delete-and-spawn. This
   matters for anything that tracks a token's history/trajectory (Koopman modes,
   attention patterns keyed on token identity) — a persisting identity implies
   the pre- and post-promotion token should be "the same object" for those
   purposes; a delete-and-spawn implies it shouldn't. No strong argument either
   way yet.
4. Whether/how to fold in empty-square control (gap 2 above) matters less now
   that the square tokens are staying (they already cover space/empty-square
   information) — this channel can stay strictly piece-to-piece without leaving
   a hole in the combined representation. Still worth deciding explicitly rather
   than leaving implicit.
5. Fusion mechanism: how does this channel actually combine with the square
   tokens — concatenated as extra sequence positions the existing transformer
   attends over, a separate small encoder cross-attending into the square-token
   stream, or fused later (e.g. at the pooled/CLS level)? Not discussed yet.
