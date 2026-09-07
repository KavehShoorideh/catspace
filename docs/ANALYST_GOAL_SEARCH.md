# Goal search: proof-number search toward a material goal, with a worked game

`catspace/incubator/analyst/goal_search.py`. Kaveh 2026-09-06: "explain this proof number thing and do an
example on my latest game; I want to see a transcript of the analysis." The game is tPVi7fZF (tevfikcikler vs
huvaxshtra, Colle, 0-1), `data/lichess/latest.pgn`. Reproduce any transcript below with

    .venv/bin/python -m catspace.incubator.analyst.goal_search --fen "<fen>" --target c5 --engine --trace --budget 600

## 1. The idea

A goal like "win the pawn on c5" is a yes/no question about a tree, not a number. At our move ONE good move is
enough (an OR node); at the opponent's move EVERY reply must be answered (an AND node). The question "is there a
move for us such that for every reply there is a move for us such that ... the pawn is ours" is exactly the
question a proof-number search (Allis 1994) answers; it was built for boolean goals (mate, connect-four), and it
is what we use instead of Monte Carlo, because a proof holds against every reply and nothing tactical is sampled
away.

Every node carries two numbers:

- **pn**, the proof number: how many leaves still have to turn out "goal reached" for this node to be proven.
  At an OR node pn is the MINIMUM over the children (one good move suffices); at an AND node it is the SUM (every
  reply must be answered).
- **dn**, the disproof number: how many leaves have to turn out "goal failed" for the node to be refuted. The
  mirror image: SUM at OR nodes, MINIMUM at AND nodes.

A proven leaf is (pn 0, dn infinity); a failed leaf (infinity, 0); an open leaf starts at (1, 1) in the textbook.
The search loop: from the root, walk down choosing the child with the smallest pn at OR nodes and the smallest dn
at AND nodes -- the **most-proving node**, the leaf whose resolution moves the root's numbers the most for the
least work -- expand it (generate every legal move, judge the new leaves), and push the numbers back up. Stop when
the root's pn hits 0 (proven), its dn hits 0 (refuted: for every move of ours there is a reply within the horizon
that fails), or the expansion budget runs out (given up). The last two are the "this, this, this... no; next goal"
Kaveh asked for; give-up is native to the algorithm, not a timer.

**Completeness.** Every legal move is generated on both sides. A prior only decides which leaf is looked at
first; it can never remove a reply from a proof. A wrong prior costs time, not the answer.

## 2. What is ours on top of the textbook

**The leaves are judged, not counted.** The goal predicate at OUR turn (the opponent has just replied):

1. the target piece is gone, captured by us (the search follows it if it runs: "win the knight on e5" becomes
   "win the knight on c4" when it moves there, and if they recapture on its square, "win what stands there now");
2. our material is up by the target's value on the raw count against the root (the material must be in hand:
   Nxc5 dxc5 is a knight for a pawn, not a pawn won, whatever the engine says);
3. Stockfish, asked to look to the SAME absolute horizon from every leaf (depth 8 at the search horizon, deeper at
   shallower leaves, 14 at the root), says the position is not worse than the root by more than 40 cp. The engine
   is the quiescence: recaptures, counter-attacks, horizon tricks are its job. It is not asked to have RISEN,
   because the root evaluation already anticipates a pawn that is falling. Its verdict is final -- it searched
   deeper than we will.

A leaf that has the material in hand only on the SETTLED count (their capture just happened and our recapture is
pending; or a piece of theirs is hanging by the ledger) is not judged yet: it may run up to four plies past the
horizon to cash it, so the horizon never falls between Qxd5 and cxd5. A leaf down a piece beyond the target's
value on the settled count is abandoned (a stated heuristic floor). With the target still standing at the
horizon the leaf fails.

**The prior is the planner and the ledger.** `plans.py` (backward chaining: what attacks the target, what
defends it, what can join in how many hops) supplies, at plies 0-3, our plan moves and the opponent's counter-goal
moves; after the target is taken it supplies the banking moves (save what of ours now hangs). Those moves are
expanded first. The recapture of whatever just captured is given the same standing.

**Initial numbers are priced, not (1, 1).** With textbook numbers a fresh sibling always looks cheaper than the
node just expanded (whose pn is now a sum over 40 replies), so the search runs breadth-first over all our moves
at the root and never goes deep. We price a fresh leaf by the work it will need: after our capture of the
target, few of the replies will need work (the engine settles most of them at once), so pn is small; after a plan
move or a move that attacks the target, a fraction of the replies; after an unrelated quiet move, most of them
(2.5 x the number of replies). The material deficit still to bank is added. This is what lets the search go
straight down the plan move and only come back to the siblings when the plan move is in real trouble.

## 3. Reading a trace line

    #2   AND bxc5: they to move, 41 moves -> 37 reach the goal, 0 fail, 4 open; open: Nxf7(2/20) Nxc6(2/20) Nxd5(2/19) dxc5(2/18) | node pn=8 dn=18

Expansion number 2; an AND node (they move) reached by 15...bxc5; the opponent has 41 legal replies; 37 of them
were settled at the leaf (the count is in hand and the engine sees no refutation); none failed; 4 need work, each
shown with its (pn/dn). The node's own numbers: pn 8 = the sum of the open children's pn (2+2+2+2), dn 18 = the
smallest child dn. "Refuted by: X (reason)" names the reply that failed a line and why: `down 5` (the settled
count fell past the floor), `engine +130 vs root +262` (the judge), `horizon` (the target still standing).

## 4. Transcript 1: the piece the game won -- after 12.a3, "win the bishop on b4"

Position `r2q1rk1/4bppp/npp1pn2/2Pp4/1B1P4/P3PN2/5PPP/RN1QK2R b KQ - 0 12`. The game went 12...Nxb4 and White
could not recapture. Thirteen expansions:

    #1   OR  (root): we to move, 29 moves -> 0 reach the goal, 0 fail, 29 open; open: Nxb4(5.8/1) Bxc5(32/1) Nxc5(80/1) bxc5(32/1) Kh8(77.5/1) ...
    #2   AND Nxb4: they to move, 29 moves -> 27 reach the goal, 0 fail, 2 open; open: axb4(6/15.5) cxb6(2/16.5) | node pn=8 dn=15.5
    #3   OR  Nxb4 axb4: we to move, 31 moves -> 0 reach the goal, 0 fail, 31 open; open: Rxa1(14/1) Bxc5(21/1) bxc5(21/1) ...
    #4   OR  Nxb4 cxb6: we to move, 33 moves -> 0 reach the goal, 0 fail, 33 open; open: Na6(6.6/1) Qxb6(5.6/1) Rxa3(14.5/1) Nd3+(3/1) Nc2+(3/1) ...
    #5   AND Nxb4 axb4 Rxa1: they to move, 28 moves -> 28 reach the goal, 0 fail, 0 open | node pn=0 dn=inf
    #6   AND Nxb4 cxb6 Nd3+: they to move, 4 moves -> 0 reach the goal, 0 fail, 4 open; ...
    #7   AND Nxb4 cxb6 Nc2+: they to move, 4 moves -> 0 reach the goal, 0 fail, 4 open; ...
    #8   AND Nxb4 cxb6 Qxb6: they to move, 28 moves -> 27 reach the goal, 0 fail, 1 open; open: axb4(6/21) | node pn=6 dn=21
    #9   OR  Nxb4 cxb6 Qxb6 axb4: we to move, 42 moves -> 0 reach the goal, 0 fail, 42 open; open: Rxa1(13.5/1) Bxb4+(4.2/1) Qxd4(21/1) Qxb4+(4.2/1) ...
    #10  AND Nxb4 cxb6 Qxb6 axb4 Bxb4+: they to move, 6 moves -> 0 reach the goal, 1 fail, 5 open; ... refuted by: Nbd2 (horizon) | node pn=inf dn=0
    #11  AND Nxb4 cxb6 Qxb6 axb4 Qxb4+: they to move, 6 moves -> 0 reach the goal, 1 fail, 5 open; ... refuted by: Nbd2 (horizon) | node pn=inf dn=0
    #12  AND Nxb4 cxb6 Na6: they to move, 28 moves -> 0 reach the goal, 0 fail, 28 open; ...
    #13  AND Nxb4 cxb6 Qxb6 axb4 Rxa1: they to move, 27 moves -> 27 reach the goal, 0 fail, 0 open | node pn=0 dn=inf
    win the bishop on b4: proven in 295 nodes, 109 engine checks: Nxb4 Nbd2; their longest try: Nxb4 cxb6 Qxb6 axb4 Rxa1

How to read it. The planner's prior puts Nxb4 first (pn 5.8 against 32 and more for the rest); after it, 27 of
White's 29 replies are settled at once: Black is a bishop up and the engine agrees. Two replies need work: the
recapture axb4 and the zwischenzug cxb6. Against axb4 the answer is Rxa1 (#5): the a-file opened when the knight
left a6 and the pawn left a3, and the b1 knight blocks the queen's defence of a1, so the rook is simply lost --
this is why 12...Nxb4 won a piece in the game. Against cxb6 the search first tries the two checks Nd3+ and
Nc2+ (four replies each look cheap; they lead nowhere), then Qxb6 (#8), after which only axb4 is open, and the
two bishop/queen checks on b4 fail at the horizon (#10, #11: White's knight interposes) before Rxa1 (#13) proves
the branch. The best defence by the judge is the quiet Nbd2 (White stays a piece down); the try that took the
most refuting was cxb6.

## 5. Transcript 2: the pawn the game won -- after 15.O-O, "win the pawn on c5"

Position `r4rk1/2q1bppp/npp1pn2/2PpN3/3P4/P1N1P3/5PPP/R2Q1RK1 b - - 5 15`. The game went 15...bxc5 16.dxc5 Nxc5.
Eighteen expansions:

    #1   OR  (root): we to move, 35 moves -> 0 reach the goal, 0 fail, 35 open; open: bxc5(8.2/1) Bxc5(20.5/1) Nxc5(20.5/1) Qxe5(85/1) Bd6(42/1) ...
    #2   AND bxc5: they to move, 41 moves -> 37 reach the goal, 0 fail, 4 open; open: Nxf7(2/20) Nxc6(2/20) Nxd5(2/19) dxc5(2/18) | node pn=8 dn=18
    #3   OR  bxc5 dxc5: we to move, 36 moves -> 0 reach the goal, 0 fail, 36 open; open: Qxe5(17/1) Bxc5(8.4/1) Nxc5(8.4/1) Kh8(22/1) ...
    #4   OR  bxc5 Nxd5: we to move, 38 moves -> 0 reach the goal, 0 fail, 38 open; open: Qd8(9.2/1) Qc8(9.2/1) cxd5(6.8/1) Qxe5(17/1) ...
    #5   OR  bxc5 Nxf7: we to move, 40 moves -> 0 reach the goal, 0 fail, 40 open; open: Kxf7(6.6/1) Rxf7(6.6/1) Qxh2+(1/1) ...
    #6   OR  bxc5 Nxc6: we to move, 40 moves -> 0 reach the goal, 0 fail, 40 open; open: Qxc6(6.6/1) Qxh2+(1/1) ...
    #7   AND bxc5 dxc5 Bxc5: they to move, 42 moves -> 38 reach the goal, 0 fail, 4 open; open: Nxf7(2/23.5) Nxc6(2/23) Nxd5(2/21.5) Qxd5(2/21.5) | node pn=8 dn=21.5
    #8   AND bxc5 Nxd5 cxd5: they to move, 34 moves -> 34 reach the goal, 0 fail, 0 open | node pn=0 dn=inf
    #9   AND bxc5 Nxf7 Qxh2+: they to move, 1 moves -> 0 reach the goal, 1 fail, 0 open; refuted by: Kxh2 (down 5) | node pn=inf dn=0
    #10  AND bxc5 Nxf7 Kxf7: they to move, 33 moves -> 33 reach the goal, 0 fail, 0 open | node pn=0 dn=inf
    #11  AND bxc5 Nxc6 Qxh2+: they to move, 1 moves -> 0 reach the goal, 1 fail, 0 open; refuted by: Kxh2 (down 8) | node pn=inf dn=0
    #12  AND bxc5 Nxc6 Qxc6: they to move, 33 moves -> 33 reach the goal, 0 fail, 0 open | node pn=0 dn=inf
    #13  OR  bxc5 dxc5 Bxc5 Nxd5: we to move, 43 moves -> 0 reach the goal, 0 fail, 43 open; ...
    #14  AND bxc5 dxc5 Nxc5: they to move, 42 moves -> 38 reach the goal, 0 fail, 4 open; open: Nxf7(2/24) Nxc6(2/24) Nxd5(2/22.5) Qxd5(2/22.5) | node pn=8 dn=22.5
    #15  OR  bxc5 dxc5 Nxc5 Nxd5: we to move, 45 moves -> 0 reach the goal, 0 fail, 45 open; ...
    #16  OR  bxc5 dxc5 Bxc5 Qxd5: we to move, 43 moves -> 0 reach the goal, 0 fail, 43 open; ...
    #17  OR  bxc5 dxc5 Nxc5 Qxd5: we to move, 45 moves -> 0 reach the goal, 0 fail, 45 open; ...
    #18  AND bxc5 dxc5 Qxe5: they to move, 34 moves -> 34 reach the goal, 0 fail, 0 open | node pn=0 dn=inf
    win the pawn on c5: proven in 626 nodes, 247 engine checks: bxc5 f4; their longest try: bxc5 dxc5 Qxe5

After 15...bxc5, 37 of White's 41 replies are settled at once; the four that need work are the four captures that
disturb the count: Nxf7, Nxc6, Nxd5, dxc5. Three are answered by the recapture (Kxf7, Qxc6, cxd5: #10, #12,
#8), each after one cheap detour into the queen sacrifice Qxh2+ (one reply, so it looks cheap; the floor refutes
it at once, #9, #11). The recapture dxc5 is the interesting one. The search first tries the two recaptures of
the pawn, Bxc5 and Nxc5 (#7, #14: each leaves four captures open, and their sub-lines #13, #15-17 are still
open), and then, at #18, the move the prior had priced at 17 against 8.4 for the recaptures because it does not
bank the pawn: **16...Qxe5** -- after dxc5 the d-pawn no longer guards e5 and the knight is undefended, and all 34
replies settle at once. The game played
16...Nxc5 and missed a knight. The judge's best defence for White is the quiet 16.f4; the branch that took the
most refuting was 16.dxc5.

## 6. Transcript 3: the knight the game missed -- after 16.dxc5, the goal loop

Position `r4rk1/2q1bppp/n1p1pn2/2PpN3/8/P1N1P3/5PPP/R2Q1RK1 b - - 0 16`. `pursue` orders the progress map's
targets (what falls first, most valuable first): the knight on e5 is first, and it is proven in eleven expansions:

    #1   OR  (root): we to move, 36 moves -> 0 reach the goal, 0 fail, 36 open; open: Qxe5(6.8/1) Bxc5(105/1) Nxc5(105/1) Bd6(43/1) ...
    #2   AND Qxe5: they to move, 34 moves -> 32 reach the goal, 0 fail, 2 open; open: Nxd5(2/22.5) Qxd5(2/21.5) | node pn=4 dn=21.5
    #3   OR  Qxe5 Qxd5: we to move, 43 moves -> ...; open: Ng4(8.8/1) Nb8(9/1) Qb8(9.6/1) cxd5(5.4/1) Nxd5(5.2/1) Qxd5(5.2/1) | node pn=1 dn=43
    #4   OR  Qxe5 Nxd5: we to move, 45 moves -> ...; open: Bd8(7.6/1) Bxc5(6.6/1) cxd5(5.6/1) Qxa1(15.5/1) Nxd5(5.4/1) Qxd5(5.4/1) | node pn=1 dn=45
    #5   AND Qxe5 Qxd5 Qxh2+: they to move, 1 moves -> 0 reach the goal, 0 fail, 1 open; open: Kxh2(18/15) | node pn=18 dn=15
    #6   AND Qxe5 Nxd5 Qxh2+: they to move, 1 moves -> 0 reach the goal, 0 fail, 1 open; open: Kxh2(18/15) | node pn=18 dn=15
    #7   AND Qxe5 Qxd5 Nxd5: they to move, 26 moves -> 26 reach the goal, 0 fail, 0 open | node pn=0 dn=inf
    #8   AND Qxe5 Nxd5 Nxd5: they to move, 27 moves -> 26 reach the goal, 0 fail, 1 open; open: Qxd5(2/22.5) | node pn=2 dn=22.5
    #9   OR  Qxe5 Nxd5 Nxd5 Qxd5: we to move, 45 moves -> 0 reach the goal, 0 fail, 45 open; open: Qxd5(3.6/1) exd5(3.6/1) cxd5(3.8/1) ...
    #10  AND Qxe5 Nxd5 Nxd5 Qxd5 Qxh2+: they to move, 1 moves -> 0 reach the goal, 0 fail, 1 open; open: Kxh2(18/14.5) | node pn=18 dn=14.5
    #11  AND Qxe5 Nxd5 Nxd5 Qxd5 Qxd5: they to move, 18 moves -> 18 reach the goal, 0 fail, 0 open | node pn=0 dn=inf
    win the knight on e5: proven in 277 nodes, 102 engine checks: Qxe5 Qd4; their longest try: Qxe5 Nxd5 Nxd5 Qxd5 Qxd5

The two desperado captures on d5 (the knight and the queen) are the only replies that need work; each is met by
the recapture, and the exchange Nxd5 Nxd5 Qxd5 Qxd5 is followed to its end past the count's first reading (the
settled-count extension: at "Qxd5" our raw count is briefly down and the ledger says the queen on d5 falls). The
Qxh2+ detours (#5, #6, #10) are the prior's cost: a check with one reply prices at pn 1 and is looked at once.

## 7. A goal the search gives up, and why that is right

AsXfL6OX before 18, White to move, `2kr1bnr/p1pp4/1p3p2/2P2qp1/8/4BNp1/PP2PP1P/2RQ1RK1 w - - 0 18`, goal "win the
pawn on b6". Stockfish's best move IS 18.cxb6 (+2.6 at depth 14, +2.9 at depth 20), and the search tries it
first, but it gives the goal up (budget 300, 8285 nodes, 11 s). The trace says why:

    #45  AND cxb6 gxh2+ Nxh2: they to move, 43 moves -> 32 reach the goal, 8 fail, 3 open; ... refuted by: Rh4 (engine +130 vs root +262), Bb4 (engine +216 vs root +262), Qh7 (engine +0 vs root +262)
    #47  AND cxb6 gxh2+ Kh1: they to move, 41 moves -> 0 reach the goal, 0 fail, 41 open; ...
    win the pawn on b6: given up after 8285 nodes (11.4s); tried cxb6, Nxg5, Bxg5, hxg3, fxg3, Qb3; hardest defence: cxb6 gxh2+ Kh1 Rh5 Rxc7+

Black's g3 pawn was already lost at the root, and 18...gxh2+ cashes it for the h2 pawn. Taking it back with the
knight (Nxh2) is refuted by the judge: after ...Qh7 the queen and rook line up on h2 and Stockfish calls it
equal at depth 10 and lost for White at depth 14. Stepping aside (Kh1) keeps the advantage (+3.2 by the engine),
but then the count is a pawn for a pawn --
b6 for h2 -- and never settles a pawn up within the horizon. That is the truth of the position: White does not
win a pawn; White converts a doomed pawn into b6 and a better position. An earlier version of this search, with
a flat shallow judge, "proved" this goal through Nxh2; the consistent-horizon judge is what removed the false
proof (the test was corrected the same day).

## 8. Costs, and what is heuristic

- Per goal: hundreds of expansions, thousands of nodes, a few hundred engine calls of depth 8-14. Under a second
  when the engine cache has seen the positions (a reviewed game), 10-20 s cold. The concept store (`memo.py`)
  serves half the ledger calls inside a search.
- Heuristics, stated: the material floor (down a piece beyond the target: give the line up); the horizon (6
  plies, plus up to 4 to cash a pending capture); the engine tolerance (40 cp); the initial pricing of leaves.
  None of them can make a proof unsound past the judge: a proof is checked by the engine at every leaf. They can
  make the search miss a proof within the budget (a give-up is not a refutation; a refutation, dn 0, is).
- Material goals only. Positional goals (open the file, occupy the outpost: the quiet 70% of Stockfish's moves
  that are neither captures nor checks) are the next layer, per docs/ANALYST_PLAN.md §22-23.
