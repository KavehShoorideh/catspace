# Explanation samples — the LLM naming layer, by hand (v2, 2026-10-02)

Kaveh: "the final product is important: to name why the normal human-looking move is not good, and why a different
move that humans don't see is good"; "it's ok to use you as a final LLM layer to manually hand-label and name things
that are chess-specific"; his picks: name each instance — first a hand-written sample — then "more facts first".

How these were written: `explain.facts()` computes everything, and the prose (Claude) uses nothing beyond it —
- Stockfish lines (depth 14) annotated move by move by python-chess: check, mate, captures, new pins, forks, uncovered
  lines; material counted along each line;
- **what the human saw** (new): what the move grabs and attacks, what it would have threatened next (Stockfish's line
  with the opponent passing), and how *human* it is — Maia-1500's probability and rank;
- **the opponent's threat before the move** (the same pass-line for the opponent) — what the position demanded;
- **what the better move achieves** (new): its own threat, the opponent's best line after it, the what-if frames it
  completes, the moved piece's mobility and reach toward the enemy king;
- the family of the opponent's tactic, from what would have prevented it.
Extra claims were checked on the board with python-chess (which piece attacks which, the king's free squares).
Six fresh Lichess puzzles (rows i % 997 == 100), the human move being the real blunder the puzzle came from.
Evaluations are Stockfish's, for the side to move, in pawns.

---

**1V7KO** — White to move. `1Q3rk1/7p/3p4/1p1P4/4Pp1q/1P3P1R/P1r3P1/R5K1 w - - 3 35`

*What the human saw:* **Qxd6** — Maia's first choice at 1500 (36%). It grabs a pawn and attacks the rook on f8, and
the idea behind it is right: next comes Qxf8+ Kxf8 Rxh4, winning the queen (the black queen on h4 is attacked by the
rook on h3).
*Why it fails:* it is one move too slow. Black already threatened mate, and still does: **Qf2+ Kh2 Qxg2#** — the queen
takes on g2 protected by the rook on c2. (From +6.8 to mated.)
*Instead: Qxf8+!* (Maia: 7%, 4th) — the same idea, played first. It takes the rook **with check**, so Black has no
time for Qf2+: after Kxf8, Rxh4 wins the queen and White is a queen up.

---

**0obGu** — White to move. `1r2bk2/6p1/Q1p4r/3p1P2/P3p2p/2N3q1/1PP1B3/3R1R1K w - - 0 28`

*What the human saw:* **Rg1** — Maia's first choice (38%). It attacks the black queen on g3 and threatens to win it
(Rxg3).
*Why it fails:* the rook steps onto the king's last free square. **Qh3 is mate at once**: with g1 blocked, the queen
covers g2 and h2.
*Instead: Qa7* (Maia: 14%, 2nd). It keeps g1 free and brings the queen toward the black king (its mobility goes from 10
to 16 moves, and it now hits e7, f7 and g7, next to the king on f8). Black's best answer is to check forever —
Qh3+ Kg1 Qg3+ Kh1 Qh3+ — a draw (0.00) instead of a loss.

---

**0ETJB** — White to move. `5rk1/R5pp/1p2pqb1/1np5/4P3/1PN1Q3/P4PPP/6K1 w - - 0 23`

*What the human saw:* **Nxb5** — Maia's first choice (54%). Black's knight on b5 was attacking the rook on a7 (Black
threatened Nxa7), so taking it looks like the obvious defence — and it wins a piece.
*Why it fails:* the knight on c3 was the piece that could block a check on the back rank (it covers b1 and d1). White's
king is boxed in by its own pawns on f2, g2 and h2, so **Qa1+ Qc1 Qxc1#**: the queen is the only blocker left, and it
is taken with mate — the *back-rank mate through a blocker*.
*Instead: e5* (Maia: 7%, 3rd). The pawn attacks the black queen on f6, so Black must deal with that first, and the
knight stays on c3 guarding the back rank. White is still worse (−5.6), but there is no mate.

---

**0Bxpq** — Black to move. `r2qr3/ppp2pk1/1bnpb2p/3N2pQ/2B1n3/1N3PB1/PPP3PP/2KR3R b - - 0 15`

*What the human saw:* **Bxd5** (Maia: only 9%, 3rd — the human-looking move here was the right one, Nf6 at 44%). It
takes the knight and attacks the bishop on c4. But Black's knight on e4 was already hanging — the pawn on f3 attacks
it — and Bxd5 does nothing about it.
*Why it fails:* White recaptures **Bxd5**, and the bishop now forks both black knights, on e4 and c6. Black's
counter-fork **Nf2** (hitting both rooks) comes too late: **Qxf7+** is check and also attacks the rook on e8, and
White collects the material. (From +1.8 to −2.1.)
*Instead: Nf6!* — the knight steps out of the pawn's attack *with a double threat*: it hits the knight on d5 and the
queen on h5. After Nxf6 Qxf6 Bxe6 Rxe6 the pieces come off and Black keeps the extra pawn (Stockfish +1.8).

---

**1Ldm2** — Black to move. `r1qr4/4bpBk/p3p3/3pP3/1p5P/1QP2b1R/PP3P2/2K3R1 b - - 1 22`

*What the human saw:* **Be4** (Maia: 22%, 2nd; its top choice, Bh5, is similar). The bishop on f3 was attacked by the
rook on h3, so it retreats — and the plan was clear: next Rg8 and Rxg7, winning White's bishop.
*Why it fails:* there is no time for the plan. **Qd1 Bg6 h5** chases the bishop again, Black has to throw in Bg5+,
and after **Rxg5** White is a piece up. (From +2.8 to −3.0.)
*Instead: Rg8!* (Maia: 15%, 4th) — the plan, at once, and let the f3 bishop go: after **Rxf3 Rxg7 Qc2+ Kh8 Rxg7
Kxg7** the trades leave Black ahead (+2.7). The attack on g7 comes first; saving the bishop costs the time it needs.

---

**0KrJZ** — Black to move. `8/6p1/2P4p/K3b1kP/6P1/1B6/8/8 b - - 0 61`

*What the human saw:* **Kxg4** — Maia's first choice (60%). It grabs a pawn, and the king is heading for the next one
(Kxh5).
*Why it fails:* the king walks away from White's h-pawn. After **Bf7 Kf4 Kb6 g5**, White takes **en passant (hxg6)**
and has a new passed pawn on the g-file; Black's bishop is pulled back to stop it, and Stockfish judges the ending lost
(−4.5) — the king was needed next to the h-pawn (*endgame squeeze*).
*Instead: Kf6* (Maia: 10%, 3rd) — the king stays beside the kingside pawns: after **Kb6 g6 hxg6 Kxg6** it takes the
dangerous pawn itself, and Stockfish calls the ending level (0.0).

---

## What the new facts added

- **The human's reasoning becomes visible.** In four of six the human move was Maia's first choice — "human-looking"
  is measurable — and the facts show *why* it looked right: it grabs material (Qxd6, Nxb5, Kxg4), it answers a threat
  (Nxb5 vs Nxa7, Be4 vs Rxf3), or it starts a plan (Qxd6 → Qxf8+, Be4 → Rg8). The explanation can then say what that
  reasoning missed: the opponent's threat already on the board (1V7KO), the job of the piece that moved (0ETJB's
  knight guarding b1 and d1), a loose piece left hanging (0Bxpq), the tempo (1Ldm2).
- **The quiet better moves now have a point.** Qa7 brings the queen to the black king and turns a loss into a draw by
  perpetual check; e5 hits the queen while keeping the knight on guard; Kf6 keeps the king by the pawns.
- **Same idea, other order** is a recurring story (1V7KO, 1Ldm2): the human saw the right idea and played the
  wrong move first. The facts show it directly: the human move's own threat *is* Stockfish's better line.
- In one of six (0Bxpq) the best move was also the most human one — the blunder was the unusual move.
