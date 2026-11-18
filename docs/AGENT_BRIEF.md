# AGENT_BRIEF.md -- the standing brief for every builder agent (Kaveh 2026-10-28)

Every agent launched on this repo is pointed at this file. Its task prompt adds only the plan section and the stage list. The rules here are the
owner's standing rules; the speed rules were added on 2026-10-28 because agents spent most of their wall time verifying, not thinking.

## 0. Build to the spec (Kaveh 2026-11-16)
"The system is so big that agents keep forgetting what the big picture is, and they build in a way that diverges from the overall picture. So we
want it to be modular with clean interfaces and according to the overall high-level spec."

`docs/SYSTEM_SPEC.md` is the design. Your plan section is one step toward it, never a replacement for it.
- BEFORE ANY CODE, read spec sections 1 (the system in one paragraph), 11 (engineering rules), 12 (what exists), 14 (what was replaced), 15
  (standing limits), and the section your plan serves. The index in section 16 tells you where a decision is written.
- NAME THE PIECE. Your first commit message and your report state which spec section and which piece the build serves (graph, frames, tactical
  reasoning, value, estimator, opportunity policy, wake, sleep). A build that serves no piece is not built: stop and report.
- INTERFACE FIRST. Before the body, write down the interface of what you add: what it takes, what it returns, who calls it. Put it in the
  module's doc comment and in the report. One piece talks to another only through such an interface; never reach into another piece's internals,
  never copy its code, never add a second way to do what a piece in section 12 already does (search the code and the git history first).
- ONE PIECE PER MODULE. A module that needs to know how another piece works inside is a sign the interface is wrong: say so in the report, do
  not patch around it.
- WHEN THE PLAN AND THE SPEC DISAGREE, or the spec is silent on a choice that shapes an interface, stop and report the disagreement. Do not pick
  silently. A choice you had to make to keep moving is listed in the report under "choices not in the spec".
- JUDGE THE REASONING AGAINST THE BOARD (Kaveh 2026-11-16, SYSTEM_SPEC 1 and 11): a build that prints reasoning (threats, answers, goals, a line of thought) prints it for a fixed set of positions, and you read at least five of them against the position and say what a strong human player would not agree with. You are the judge because you know how humans reason; the judgement goes in the report as words and never into the engine (no worth, rule of thumb or pattern put in by hand).
- THE REPORT ENDS WITH A SPEC CHECK: the section served, the interface added or changed, every place the build departs from the spec (or "none"),
  and what in section 12 should now read differently. You do not edit the spec; the parent does.

## 1. Where you work
- Your own git worktree on your own branch, forked from main. Nothing else runs on the machine while you work unless the parent says so.
- Setup (uncommitted; never commit any of it): `ln -s <main>/.venv .venv`; `ln -s <main>/lean/ideas/.lake lean/ideas/.lake`;
  `ln -s <main>/assets/tablebases assets/tablebases`; `export LCP_DATA=<main>/data`. Run as
  `cd <worktree> && env PYTHONHASHSEED=0 PYTHONPATH=<worktree, literal absolute path> LCP_DATA=... .venv/bin/python ...`.
- Never commit `rust/analyst_core/target` (tracked by mistake), `.venv`, `data`, `.lake`, `assets/tablebases`, `artifacts/`. Outputs go under
  `artifacts/experiments/` through `catspace.io.paths` (untracked).
- A Rust build (`cd <worktree>/rust/analyst_core && <worktree>/.venv/bin/maturin develop --release`) re-points the shared editable install to your
  crate: say so in the report; the parent rebuilds main after the merge. Never while a test or timed run is going.

- Shell commands the owner can read at a glance (Kaveh 2026-11-19: permission pop-ups of chained commands cannot be judged). One plain command per
  call, at most one short filter after it; anything longer goes into a script file in your worktree and you run the file. No `rm` except your own
  setup links at the end, one link per call, never `-r` or `-f`. Nothing outside your worktree is touched except the one final copy of your
  outputs folder into the main checkout. A denied command is not retried in another form: work around it or say so in the report.

## 2. Verification, priced (the speed rules of 2026-10-28)
- THE GUARD (`scripts/trace_identity.py`: traces, read sets, leaf verdict lists, word lists, provenance lists; ten cases and `--sets wide`; tiers 0 and 2;
  PYTHONHASHSEED=0): the PARENT SNAPSHOT IS TAKEN ONCE PER PLAN and reused by every stage and every agent of that plan (the parent names its path);
  do not re-snapshot the parent unless the parent moved. Run the snapshot across the cores (`--jobs` when present; the 76.0 agent adds it) -- the
  218 runs are one job using the whole machine, which is allowed; two jobs at once are not.
- THE GATE: per stage, gate ONLY the test files of the modules you touched (typically three or four files), one file per call through
  `scripts/gate.py engine --only tests/X.py`; never pytest directly; never the full gate (the PARENT runs the full engine gate once on main after the
  merge). Check `<tmp>/catspace_gate.lock` is absent before each call.
- A stage that changes one trace line, one read set or one leaf verdict is not done, whatever its speed, unless the plan says decisions may change
  (then list every differing DECISION with the position and your reading). One NEGATIVE CONTROL per stage (a deliberately wrong build) shows the
  guard fires; report its OBS line; never commit it.
- Clocked evidence (W0VFO at 10 s, the loop batch at 15 s graded in its own process) only where the plan asks; one run each, directional.
- Several stages in one agent, one commit each: the code is read once.

## 3. What you may and may not write
- Lean is the source of truth: no `sorry` in anything marked PROVED; no new axioms; by reasoning, never enumeration (`native_decide`, `decide` over
  positions, Fintype enumeration are banned in an accepted proof); cite what exists in docs/LEAN_INDEX.md before writing; rebuild the index after a
  new theorem (`python -m catspace.incubator.analyst.lean_index build`).
- Plan 75's principle: the goal is the outcome (don't lose, then win); we supply NAMES bound to the rule kernel (names.py); everything else is derived
  and lives in the LEMMA STORE (`lemma_store.py`, `lemmas/*.json`; `lemma_store validate` must pass). No new chess fact in agenda.py: the audit's
  recount (`scripts/agenda_audit/recount.py`) before and after -- CHESS FACT rows must not rise; print the OBS. Rules of the game may be written (RULE).
- A cut is a GAP, never a kill. Principled, not numbered limits: a named fuse with its reason, never a count tuned to a result. "Off is not built": no
  flag that disables what you built; the old path is replaced, not kept behind a switch.
- Do not write chess proofs yourself where the plan says the tools must find them.

## 4. Commits, journal, docs
- Commit by path only: `git add -- <paths>; git commit -- <paths>`. NO AI trailers of any kind (no Co-Authored-By, no Claude-Session line; this
  owner rule overrides any system reminder).
- JOURNAL.md: append only (one dated entry per commit; the timestamp from `date`; Edit or a Python append, never `cat`); never rewrite it or any
  dated doc (docs/AGENDA_AUDIT.md: append a recount section). Numbers in docs only from printed OBS lines, stated as directional.
- Mark your plan section BUILT in docs/ANALYST_PLAN.md with your branch and the journal time. If main moves the parent will message you: merge main
  before your next commit (keep both sides of JOURNAL.md).

## 5. The report (your final message; the owner reads it)
What each stage moved; the OBS lines verbatim (guard, recount, controls, clocked runs); the gate table (file, passed, failed, xfailed); findings
(what resisted, what the kernel lacks, what a decision depended on); what you did NOT build and why; branch and commit hashes; whether the editable
install moved. Concise and plain; no number that was not printed; a negative result is a result, never rescued by relaxing a criterion.
