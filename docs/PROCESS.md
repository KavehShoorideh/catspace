# PROCESS — internal working rules (moved out of README 2026-07-28)

Standing rules for how the work is done. Canonical per-rule files live in the assistant's
session memory; the training subset is docs/TRAINING_STANDARDS.md. Not user-facing.


**Rigor & method**
- **Rigor over flattery** — no sycophancy; grade every claim (proven / provable / plausible) and retract loose ones explicitly.
- **Journal numbers must be verdicts** — no metric enters JOURNAL/commits without a printed script `VERDICT`; figures come from repo scripts, not heredocs.
- **Diagnose before concluding** — never call a plateau/failure "fundamental" before ruling out the loss and the data; **unit-test every loss** (canonical, tested terms live in `experiments/losses.py`); inspect the target distribution before choosing a loss.
- **Define identifications** — an "X is a Y" claim must state Y's criteria and verify X meets them.
- **Discuss before building** — exploratory questions get option-space + a recommendation, then wait; directional builds need Kaveh's call. **Notify his phone** for stop-for-a-question decisions (AskUserQuestion + PushNotification when away).
- **Search when stuck** (>15 min on something that should work) — stop tuning, read the paper / reference impl.
- **No one-lever rule** — bundle well-justified changes into one run; record what's bundled. **Conditional rejections**: A/B rejections are conditional on the field version — re-test the shelf after field promotions.

**Engineering**
- **Import, don't reinvent** — standard frameworks (PyTorch, MLflow, Ray, DVC); "PyTorch" in prose, `torch` as the package.
- **Short runs before big runs** — validate on a short run, commit when it works, then launch the full run.
- **Optimize before long runs** — can it be parallelized? can unnecessary compute be removed (fuse passes, precompute)? **Check long runs early and often** (watchdog on output growth; CPU-busy ≠ progress).
- **Always run latest** — kill stale runs on every engine/field update; resume makes it cheap.
- **Check representational collapse** — bootstrapped `eff_rank` is a health gate on every run; cure is repulsion, not width.
- **No concurrent disk-heavy jobs**; the tablebase probe cache is **`journal_mode=DELETE`** (not WAL — WAL grew unbounded and filled the disk twice).
- **Training standards** (`docs/TRAINING_STANDARDS.md`): checkpoint ladders + metadata, **no overwrites**, one richest input format, MLflow not hand-rolled.

**Workflow**
- **Keep JOURNAL.md current** as work happens; **time every run**. **No AI commit trailers** (no `Co-Authored-By: Claude` / `noreply@anthropic.com`). **Self-contained reports** — weekly-report style; numbers carry baselines.
