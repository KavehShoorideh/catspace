# Explanation samples — the LLM naming layer, by hand (2026-10-02)

Kaveh: "the final product is important: to name why the normal human-looking move is not good, and why a different
move that humans don't see is good"; "it's ok to use you as a final LLM layer to manually hand-label and name things
that are chess-specific"; his pick: name each instance, with a hand-written sample first to judge the style.

How these were written: `explain.facts()` computes everything (Stockfish depth 14 lines, each move annotated by
python-chess — check, mate, captures, new pins, forks, uncovered lines — the family, the preventers), and the claims
below were checked against the board with python-chess (material along each line, which piece attacks which, the
king's flight squares). The prose is mine (Claude); nothing in it goes beyond those facts. The positions are six fresh
Lichess puzzles (rows i % 997 == 100), drawn at random among those where a family was named. The "human move" is the
real blunder the puzzle was made from. Evaluations are Stockfish's, for the side to move, in pawns.

---

**1V7KO** — White to move. `1Q3rk1/7p/3p4/1p1P4/4Pp1q/1P3P1R/P1r3P1/R5K1 w - - 3 35`

*Why not Qxd6?* It wins a pawn but ignores the mating net around White's own king: **Qf2+ Kh2 Qxg2#**. The black
queen checks from f2, the king has to step to h2, and the queen takes on g2 — protected by the rook on c2 along the
second rank. (Stockfish: from +6.9 to mated.)

*Instead: Qxf8+!* The black queen on h4 has been attacked by the rook on h3 all along; Black only got away with it
because of the mate threat. Qxf8+ removes the f8 rook **with check**, so Black has no time for Qf2+: after Kxf8,
Rxh4 takes the queen and White is a queen up (+8 in material).

---

**0obGu** — White to move. `1r2bk2/6p1/Q1p4r/3p1P2/P3p2p/2N3q1/1PP1B3/3R1R1K w - - 0 28`

*Why not Rg1?* The rook steps onto the king's last free square. With g1 blocked, **Qh3 is mate at once**: the queen
covers g2 and h2, and the king on h1 has nowhere else to go. (Stockfish: from level to mated.)

*Instead: Qa7* (Stockfish's choice) keeps g1 free. Black can still check on h3, but now the king simply steps to g1,
and the position stays level (0.00).

---

**0ETJB** — White to move. `5rk1/R5pp/1p2pqb1/1np5/4P3/1PN1Q3/P4PPP/6K1 w - - 0 23`

*Why not Nxb5?* It takes a knight, but White's king is boxed in by its own pawns on f2, g2 and h2 — its only free
squares are f1 and h1, both on the back rank. **Qa1+ Qe1 Qxe1#**: the queen checks along the first rank, White's
queen has to block on e1, and the blocker is taken with mate. This is the *back-rank mate through a blocker*.

*Instead: Rb7* (Stockfish's choice). White is already worse (−5.8), but the rook stays active and the back rank is no
longer the immediate problem — no mate.

---

**0Bxpq** — Black to move. `r2qr3/ppp2pk1/1bnpb2p/3N2pQ/2B1n3/1N3PB1/PPP3PP/2KR3R b - - 0 15`

*Why not Bxd5?* Taking the knight hands White a recapture with a fork: **Bxd5** (the bishop from c4) attacks both
black knights, on e4 and c6. Black's counter-fork **Nf2** (hitting both white rooks) looks like a rescue, but White
answers with **Qxf7+** — check, and the queen now also attacks the rook on e8 — and wins the race: after Kh8 Qg6 Re7
Bxc6, White has collected the material. (Stockfish: from +2.1 to −2.5.)

*Instead: Nf6!* The knight attacks the white knight on d5 and the queen on h5 at once. After Nxf6 Qxf6 Bxe6 Rxe6 the
pieces come off and Black keeps the extra pawn (+2.1).

---

**1Ldm2** — Black to move. `r1qr4/4bpBk/p3p3/3pP3/1p5P/1QP2b1R/PP3P2/2K3R1 b - - 1 22`

*Why not Be4?* The bishop on f3 was attacked by the rook on h3, but e4 is no safe home: after **Qd1 Bg6 h5** the
bishop is chased again, Black has to throw in **Bg5+**, and after **Rxg5** White comes out a piece ahead (material
from +2 to −1 for Black; Stockfish −2.2).

*Instead: Rg8!* Counter-attack: let the f3 bishop go and take White's bishop on g7. After **Rxf3 Rxg7** (and Qc2+ Kh8
Rxg7 Kxg7 Rg3+) the trades leave Black two points up (+2.2).

---

**0KrJZ** — Black to move. `8/6p1/2P4p/K3b1kP/6P1/1B6/8/8 b - - 0 61`

*Why not Kxg4?* It wins a pawn but takes the king away from White's h-pawn. After **Bf7 Kf4 Kb6 g5**, White takes
**en passant (hxg6)** and gets a new passed pawn on the g-file; Black's bishop must give itself up on g7, and the
ending is lost (Stockfish −4.5). The king was needed next to the h-pawn — the *endgame squeeze* family.

*Instead: Kf6* keeps the king beside the kingside pawns: after **Kb6 g6 hxg6 Kxg6** the king itself removes the
danger, and the bishop alone can watch the c-pawn. Level (Stockfish 0.0).

---

## What the samples show (for the style decision)

- The family names help where they fit (back rank through a blocker, endgame squeeze) and are left out where they do
  not (0Bxpq's family was "interference"; the real story is a fork after a recapture, and the facts say so).
- Every concrete claim comes from a computed fact: the lines, the annotations (fork, check, takes), material counts,
  flight squares. What the prose adds is the *why* in a coach's words: "ignores the mating net", "no time because it
  comes with check", "the king was needed next to the h-pawn".
- Weak spot: when Stockfish's better move has no computed point (0ETJB's Rb7, 0obGu's Qa7), the text can only say
  what it avoids, not what it does. That is the strategic side, not yet named.
