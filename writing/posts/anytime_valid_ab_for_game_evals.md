# Anytime-valid A/B testing for game-playing agents

*One topic from the catspace project: how to compare two chess policies
honestly when games are slow, noisy, and you can't resist peeking. The
implementation described here is `catspace/abtest.py` + `experiments/playout_ab.py`.*

## The problem

You have two policies and a GPU that plays about one evaluation game per
second. You want to know which policy is better. Three things conspire
against you:

1. **Game outcomes are absurdly high-variance.** In our endgame testbed, most
   won positions end as draw-or-win for *both* policies, so the paired
   per-game difference is mostly 0 or ±1. At n = 200 games, our conversion-rate
   confidence interval was ±0.38 — wide enough to hide almost any real effect.
2. **You will peek.** Runs take hours. Nobody waits for a fixed n before
   looking, and with classical tests every look inflates your false-positive
   rate.
3. **You will run many variants.** The best-looking of eight sweeps is biased
   upward even if all eight are null (the winner's curse).

We got burned by all three before adopting the discipline below. The
canonical burn: an overnight sweep showed a variant at **0.575 vs the
incumbent's 0.517** on n = 60 point estimates. The properly powered, paired
re-run said **−0.005, e = 0.09** — the data actively favored "no
difference." The "win" was pure sampling noise.

## The design: pair everything, kill every noise source you can

- **Fixed frozen position sets.** Both policies play the same starting
  positions with matched seeds. The per-position paired difference is the
  unit of analysis; position difficulty cancels out.
- **A deterministic opponent.** We initially "paired" games against
  skill-limited Stockfish — and then discovered its skill-limiting draws from
  the engine's own internal RNG, which no external seed touches. Same
  position, same seed, different result: our paired design was secretly
  unpaired. The fix was a tablebase-optimal defender (deterministic by
  construction), which shrank the conversion-diff CI from ±0.38 to ±0.075 at
  comparable n. **Check that your opponent is actually deterministic; don't
  assume the seed you pass is the seed that matters.**
- **A continuous secondary metric.** Mate-or-not is binary; plies-to-mate on
  jointly-converted positions carries more information per game and has
  repeatedly shown effects (faster mates) that the rate metric couldn't
  resolve.

## The test: e-values instead of p-values

For the significance layer we use an **e-process** — an anytime-valid
sequential test. The intuition: an e-value is your accumulated winnings from
repeatedly betting against the null hypothesis at fair odds. Formally it's a
nonnegative supermartingale with expectation ≤ 1 under the null, which buys
you the property p-values don't have: **you may look at it after every single
game, stop whenever you like, or keep going, and the guarantee still holds.**

Usage rules, as we run them:

- **Reject when e ≥ 1/α.** At α = 0.05 that's e ≥ 20.
- **e ≪ 1 is informative too:** the data favor the null. An e of 0.09 after
  200 games isn't "not significant yet" — it's evidence of no effect.
- **Early stopping is free.** A search-method duel we expected to need 120
  paired games crossed e = 23.7 at game 47 and stopped, saving 60% of the
  compute. Symmetrically, a promising look at n = 120 was extended to n = 200
  mid-run with no correction needed.
- **Independent runs compose by multiplication.** Two independently-seeded
  head-to-heads gave e = 65.1 and e = 8.3; the second run alone wouldn't
  reject, but the composed evidence e = 539 is decisive. This is how you make
  replication quantitative instead of vibes.

We report the e-value **alongside** a bootstrap confidence interval on the
paired difference — the e-value answers "is there an effect," the CI answers
"how big could it plausibly be," and you want both in every verdict line.

## The selection layer: pre-registered confirmatories on single-use sets

Sequential validity does not protect you from *selection across experiments*.
Our rule: sweeps are declared exploratory up front; the selected winner gets
exactly **one confirmatory run on a fresh frozen position set that has never
been used and will never be used again.** A registry file records each set as
consumed, and the set generator refuses to remint a consumed seed — the
protocol is enforced by code, not by memory.

This protocol cut both ways within one week, which is exactly why we trust
it. A data-scaling sweep produced one significant-looking rung (+0.167,
e = 6.9 — one of four looks, so below the composed bar); its confirmatory
came back +0.050, not significant. Winner's curse, caught. Days later, the
same protocol *confirmed* a different claim on a different fresh set (+0.208,
CI [+0.108, +0.317], e = 184.7), and that one got promoted. A protocol that
only ever kills results, or only ever blesses them, isn't a filter — this one
demonstrably filters.

## The checklist

1. Pair on frozen position sets; analyze per-position differences.
2. Verify your opponent is deterministic (test: replay the same seed twice).
3. Report a bootstrap CI *and* an anytime-valid e-value on every comparison.
4. Never promote on a point estimate, whatever the n.
5. Declare sweeps exploratory; confirm the winner once, on a single-use set.
6. Add a continuous secondary metric; binary outcomes are power-starved.
7. Let replication compose: multiply e-values across independent runs.
