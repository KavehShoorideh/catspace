# catspace — a reachability-based chess planner

**A live research project.** Claude (the AI) does the building and experiments;
Kaveh directs the research — the questions, the reframes, and most of the core
ideas are his, iterated on turn by turn. If you want to see *how* the work
actually happens — the hypotheses, the dead ends, the reversals, the reasoning
behind each decision — read **[JOURNAL.md](JOURNAL.md)**. It's the running lab
notebook, newest entry last, and it's the most honest picture of the project.

---

## The idea

Most engines (Stockfish, Leela) learn a **value** function — "how good is this
position" — and search over it. catspace instead learns a **reachability**
embedding: two encoders `F(s)` (a state) and `B(g)` (a goal) such that
`score(F(s), B(g))` tells you *how reachably a goal follows from a state* under
real play. Planning is then **navigation toward a goal region** (e.g. checkmate),
not position-scoring. The bet is that this is a distinct, viable way to build a
game-playing agent — one that *plans* rather than *evaluates*.

Two commitments follow from that bet:
- The distance is a **quasimetric** (respects the triangle inequality), so
  multi-step plans compose — "pin, then win the piece, then mate" — instead of
  only working when the exact sequence was seen in training.
- We stay reachability-native: no win/draw/loss value head (that would just be
  Leela). Where the engine needs to know *where to think harder*, it measures its
  own **reliability** (does searching deeper change its mind here?) rather than
  importing a value signal.

## The documents (and how to use them)

| file | what it's for |
|---|---|
| **[JOURNAL.md](JOURNAL.md)** | the running lab notebook — every experiment, result, and decision, with reasoning. **Start here.** |
| **[COMPONENTS.md](COMPONENTS.md)** | a map of the code: what each module/policy/probe does. Read when you're lost in `catspace/` or `experiments/`. |
| **[GLOSSARY.md](GLOSSARY.md)** | plain-language definitions of every term and metric (reachability, quasimetric, ACPL, ply-gap, e-value, ρ, …). For a chess enthusiast, not just an ML one. |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | the code layer diagram and invariants. |
| **[UNCERTAINTY_DESIGN.md](UNCERTAINTY_DESIGN.md)** | the current research direction: sharpness-as-self-reliability and the search↔training closed loop. |
| **[TWO_HORIZON_DESIGN.md](TWO_HORIZON_DESIGN.md)** | a superseded design, kept as a record. |

## Where the data lives

Everything heavy is under `data/` (git-ignored) and is produced/downloaded, not
committed:
- `data/shards/…` — human Lichess games as packed-bitboard position shards.
- `data/selfplay/…` — self-play shards (same schema; from `selfplay_generate.py`).
- `data/derived/…` — trained checkpoints (`*.pt`) and the competence map.
- `data/syzygy/…` — endgame tablebases (exact ground truth, download-on-demand).

Research **results** are split by durability:
- `artifacts/experiments/` — structured JSON records of every evaluation, plus
  fixed test sets. **Git-tracked** — this is the quantitative history.
- `artifacts/generated/` — HTML viewers and logs. Git-ignored (rebuildable).

## Running it

```bash
pip install -e .[nn]          # torch is an optional extra ([nn]); numpy core works without it
pytest                        # the test suite

# train an embedding (composable loss flags -- see COMPONENTS.md / train_lichess_fb.py --help)
python experiments/train_lichess_fb.py \
    --shards data/shards/<human-prefix> --ckpt data/derived/run.pt \
    --steps 90000 --quasimetric --ply-gap-weight 0.05 --ckpt-every 30000

# evaluate: arena vs Stockfish (enforces the no-leakage audit gate), blunder rate, endgame conversion
python experiments/experiment_report.py --ckpt data/derived/run.pt --opponent sf:skill=0 --games 40 --search-nodes 200
python experiments/acpl_probe.py    --ckpt data/derived/run.pt --n 400
python experiments/krrkbp_arena.py  --ckpt data/derived/run.pt --fixed-set artifacts/experiments/krrkbp_fixed_set_n60.json

# the self-improving loop (in progress): self-play -> competence -> reliability-gated search
python experiments/selfplay_generate.py    --ckpt data/derived/run.pt --out-dir data/selfplay/gen1
python experiments/build_competence_map.py --ckpt data/derived/run.pt --out data/derived/competence_map.npz
python experiments/adaptive_vs_uniform.py  --ckpt data/derived/run.pt --competence data/derived/competence_map.npz
```

## Inspecting results

- **Numbers:** read the JSON in `artifacts/experiments/`, or diff runs with
  `python experiments/experiment_leaderboard.py`. Every training run also prints
  `VERDICT` lines that get copied verbatim into JOURNAL.md.
- **Visually:** `python experiments/viz/build_*.py` renders self-contained HTML
  into `artifacts/generated/` (training curves, per-move decision viewer, reach
  maps, the fitness dashboard); `build_gallery.py` indexes them.
- **The story:** JOURNAL.md ties the numbers to the decisions.

## Metrics

- **Arena score** (0–1) vs a fixed Stockfish strength — the objective play metric
  (win=1, draw=0.5), reported with an anytime-valid **e-value** so early stopping
  doesn't inflate false positives.
- **ACPL** — average centipawn loss per move vs a strong Stockfish's judgment; a
  blunder-rate proxy (lower better; master <20).
- **KRRvKBP conversion** — a narrow, tablebase-verified winning endgame the
  planner must convert; the primary *planning* diagnostic (paired, matched-seed).
- **Fitness probe** — quasimetric health: Syzygy distance calibration, horizon
  retrieval, asymmetry, triangle-violation, degeneracy (`qm_fitness_probe.py`).
- **Reliability / competence** — the engine's self-measured unreliability, used to
  decide where to search harder (not a quality score).

## Choices worth knowing

- **Reachability, not value** — the thesis; we don't add a WDL head.
- **No Stockfish-eval leakage** — Stockfish is only ever an opponent or an offline
  grader; its evaluations never become a training label, enforced by a hard audit
  gate (`catspace/audit.py`) on every arena run.
- **Small search budget** (~200 nodes, ~10× below Leela's playing range) — any win
  should come from the *plan*, not from out-searching the opponent.
- **Validate by play, not by invented labels** — e.g. "sharpness" is a useful
  fiction for allocating effort, so it's judged by whether it improves play, not
  by matching a hand-defined target.
