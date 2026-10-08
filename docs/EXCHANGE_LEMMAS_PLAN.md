# The exchange from lemmas — plan (2026-10-08, for Kaveh's go)

Kaveh 2026-10-08: "make it generic ... based upon the rest of the rules. If the king can never be put into a threat,
that's one of the lemmas, then we rely on that lemma and we don't have to do a legality check. If a piece is needed
somewhere else for a bigger job, the piece is not available, even though it's a legal move ... a Lean-based lemma for
every one of these failure modes ... then we cache the results of the lemmas so they become fast." And: keep every
variant's NAME (king exchange, intermezzo exchange, promotion exchange ...) even where no extra code computes it.

## The idea in one paragraph

Today the law is "the simple count is right IF ten hypotheses hold"; each hypothesis is hand-written code, and each
failure throws the frame away. The plan inverts it: ONE exchange game on the square -- each side captures with an
AVAILABLE piece or stops -- where *which pieces are available*, *what each is worth on the square*, and *who joins
later* come from lemmas about the rules, not from checks. The exceptions stop being exclusions and become what the
lemmas compute; each keeps its name as a label on the result.

## What we already have (measured, 2026-10-08)

- The kernel defines a legal move as pseudo-legal AND the mover's king not in check afterwards (`legalMove`,
  `legalMove_kingSafe` in chess-lean). `Ideas/Mate.lean` already proves attack monotonicity (`pieceAttacksB_mono`,
  `clearBetweenB_mono`) -- the raw material for the x-ray lemma.
- Part 1: the capture game, plain and over rays (`cheapestXF`; proved: never negative, equals the plain game when
  nothing is hidden).
- Measurements against the exact search (8,000 positions): intermezzo priced 97%, x-rays counted 97%, their king taking
  part 96%, our king 87%, a check in between 92% (29 of its 80 misses are mates).

## The lemmas — one per exception, layered

**Layer 0 — rule lemmas (about chess-lean's own definitions; no exchange in them)**

| Lemma | Statement | Replaces the hand check | Name it carries |
|---|---|---|---|
| R1 king safety | a capture is available iff our king is not attacked after it (`legalMove_kingSafe`, already in the kernel) | — | — |
| R1a king last | corollary: a king captures on `s` only when nothing of the other side can recapture there (revealed pieces included) -- so the king is always the last capturer | `noKing` | king exchange |
| R1b pin | corollary: a pinned piece captures on `s` only along its pin line | `noPin` | pinned exchange |
| R2 reveal | after a piece leaves `q`, the attackers of `s` are the old ones minus `q` plus the first piece behind `q` on the line `q`–`s` if it has the geometry; recursively, the x-ray chain | `noXray` | x-ray exchange |
| R2b block | the converse: a piece landing on a line between an attacker and `s` removes that attacker | `noInterpose` | interposition exchange |
| R2c arrival | a piece landing on a square with geometry to `s` (or uncovering one, by R2) adds a defender | `noArrival` | exchange with an arriving defender |
| R3 promotion | a pawn capturing onto its last rank stands on `s` as the promoted piece; a pawn reaching the last rank elsewhere is a threat worth the promotion | `noPromotion` | promotion exchange |

**Layer 1 — the game (part 1, extended):** the capture game over rays WITH availability: at each turn the side picks
an available capturer (R1) or stops; the attacker lists evolve by R2; values by R3. Proved: never negative, reduces to
the plain game (done for rays), and (measured 2,408/2,408) cheapest-first equals the full minimax on real frames -- a
lemma to prove or a certificate to keep.

**Layer 2 — frames that share pieces ("needed elsewhere for a bigger job")**

| Lemma | Statement | Replaces | Name |
|---|---|---|---|
| J1 overload | a piece that also guards square `u` costs, when it captures on `s`, what `u`'s exchange gains for the opponent without it (the same game on `u`, with and without the piece); outcome >= gain on `s` minus that cost | `noOverload` | overloaded exchange |
| J2 intermezzo | the opponent's best capture elsewhere costs at most its own frame's value when the frames are independent (J1 makes them so); outcome >= value - that cost | `noOtherCapture` (the composed tier today, certificate only) | intermezzo exchange |
| J3 check in between | a checking reply is answered by a capture on `s` when a participant can (the exchange goes on), otherwise the exchange is suspended and the check's own frame decides | `noCheck` | exchange with a check in between |

J3 needs mate lemmas (the misses are mating nets); back-rank mate is the only one proved. It goes LAST.

**Layer 3 — the law:** after our capture, within the game's length and whatever the opponent does, a settled position
is reached at the value Layers 1–2 compute (Kaveh's settled goal: no paying recapture on `s`, nothing elsewhere paying
more than the slack). Proof by induction on the participants, using R1–R3, J1–J2. Until proved, the kernel certificate
plus the exact bridge on the census stay the check.

## Caching

Every lemma's result is a function of a small frame (the square, the rays on it, the squares those pieces guard). The
binary returns the per-frame results; Python memoizes them by frame key, as the frame player already does (28% from
memory), with the verify mode (recompute and compare) on for every benchmark before a number is quoted.

## Order of work (each step: kernel lemma + certificate, prototype mirror, bridge on the census with 0 wrong, tests,
labels kept, journal)

1. **R1a / R1b** (king last, pin) -- corollaries of a lemma the kernel already has; `noKing` and `noPin` disappear as
   hypotheses and become availability. Smallest step; proves the pattern.
2. **R2 / R2b / R2c** (reveal, block, arrival) -- the hardest rule lemma (clear lines after a square empties or fills);
   `noXray`, `noInterpose`, `noArrival` disappear.
3. **R3** (promotion).
4. **J1, J2** (overload, intermezzo) as theorems over two frames.
5. **The law** (Layer 3), by induction.
6. **J3** (check in between) after more mate lemmas.

## Honest grading

- R1a/R1b: provable soon (corollaries of `legalMove_kingSafe`).
- R2 family: provable, real work (line geometry after a board change; the mono lemmas help).
- R3, J1, J2: provable as statements about the game; J1/J2 need the frames' independence stated exactly.
- The law's induction: the largest proof here; days, not hours. Until then every claim is "certified per position and
  bridged", never "proved".
- J3: not before mate machinery.
