# Keeping oracles out of the training loop (without giving them up)

*One topic from the catspace project: chess hands you two perfect oracles —
engines and tablebases. Using them for everything except training, and
proving to yourself that you did, turns out to need actual machinery.*

## The tension

Our project's thesis requires that the planner learn to plan from play data
alone — no engine evaluations as labels, no tablebase distances in the loss.
But the same oracles are irreplaceable everywhere else: Stockfish is a
graded opponent ladder; Syzygy tablebases verify that test positions are
truly won, provide a *deterministic optimal defender* (which fixed a serious
variance problem in our paired evaluations), and grade how far from mate the
planner actually was. The discipline is a boundary, not abstinence:
**oracles may grade and oppose; they may never teach.**

Boundaries that live in convention decay. Ours is enforced three ways.

## 1. A source-inspecting audit at every evaluation

`catspace/audit.py` re-reads — at call time, via `inspect.getsource` — the
actual source of the training data path and the planner's readout path,
scanning for any oracle-derived identifier (`eval_cp`, `stockfish`, `wdl_*`,
…). The evaluation harness runs this as a hard gate: a dirty audit aborts
the run with no report written. The point of inspecting *live source* rather
than maintaining a checklist is that a future edit which starts reading
`eval_cp` into the loss fails the gate automatically; nobody has to remember
the invariant, and (verified by test) a synthetic offender IS caught.

Amusing implementation note: the first draft put the provenance check inside
the training script's `main()` — whose own source therefore contained the
string "stockfish", tripping the scanner on itself. The checker moved into
the audit module so the guarded code never needs the forbidden words. Any
self-referential guard will meet this bug.

## 2. Provenance stamped into every checkpoint

Every save embeds a provenance dict: script, args, git commit, and a
`stockfish_free` flag that is itself the *output* of the static audit against
the running code — not a hand-set boolean. Old checkpoints without stamps
are treated as "unknown," not "clean." When a checkpoint changes hands
(promotion, rollback, a branch trained weeks ago), its training conditions
travel with it.

## 3. Structural isolation, checked once, then trusted plus verified

The training loop's batch assembly reads only board/metadata fields; the
oracle-labeled columns present in the data (Lichess `[%eval]` annotations)
are simply never dereferenced on the training path — an isolation you can
confirm by reading one function, which the audit then keeps confirmed
forever. Where an oracle-consuming component legitimately exists (an
eval-head *probe* used for analysis), it writes to separate probe files and
has no code path that writes back into a planner-loadable checkpoint.

## Where the line actually sits (the subtle cases)

- **Oracle as opponent:** fine. Playing against Stockfish generates
  outcomes; outcomes are the world's feedback, not the oracle's opinion.
  (Games record moves and results only, never engine scores.)
- **Oracle as verifier of test data:** fine, and load-bearing — every
  evaluation start position is tablebase-verified as won *before* the
  planner ever sees it, so "conversion rate" measures the planner, not the
  test set.
- **Oracle as defender:** fine, and a methodological upgrade — a
  deterministic optimal defender removes opponent randomness from paired
  comparisons entirely.
- **Oracle as scaffold for estimator development:** the honest gray zone. We
  once needed conversion-probability estimates in a region where our own
  policy was too weak to produce signal (its P̂ ≈ 0.05 everywhere — no
  gradient). We used tablebase-guided rollouts *as White* to develop and
  validate the estimator, labeled every artifact as scaffolded, and then
  **de-scaffolded**: the production tables are regenerated from the
  planner's own play, and the scaffolded versions retired to reference. The
  rule we extracted: scaffolding is legitimate for building *instruments*,
  as long as the artifact that feeds training is regenerated oracle-free and
  the provenance of both is recorded.
- **Named concepts:** the same principle one level up. Detectors for "pin,"
  "double attack," "king cornered" exist in our verification tooling —
  they grade whether the planner's games exhibit the concepts — but play
  and search never consume them. Concepts must eventually be *discovered*
  structure in the embedding, named by us post-hoc.

## Why bother

Beyond the obvious scientific-validity reason, there's an engineering one:
the moment a "temporary" oracle signal enters training, every subsequent
improvement is confounded — you can no longer tell whether your mechanism
works or whether you've distilled the oracle. In a project explicitly about
*finding mechanisms* (what objective makes tactical safety emerge? what
geometry makes plans compose?), that confound is fatal. The audit gate costs
milliseconds per run. Untangling a leak after fifty experiments would cost
the fifty experiments.
