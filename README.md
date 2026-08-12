# catspace

A chess engine built to plan like a human — committing to subgoals in a learned
concept space and searching only where its plan is uncertain — with
interpretability as the measured primary endpoint and strength-per-node as the
frontier.

**[docs/THESIS.md](docs/THESIS.md)** is the document of record: the hypothesis,
the formal frame, the current architecture (with diagram), findings, and failed
attempts. **[JOURNAL.md](JOURNAL.md)** is the lab notebook, written as the work
happened. **[MILESTONES.md](MILESTONES.md)** is the roadmap;
**[docs/SUBGOALFORMER.md](docs/SUBGOALFORMER.md)** specs the planner.

## Structure

| | |
|---|---|
| `catspace/research/components/encoder/` | trunk + quasimetric field + JQT concept layer (`reach_probability/`) |
| `catspace/research/components/planner/` | engine, search, planner, batteries (`quasimetric_nav/`) |
| `catspace/research/tools/training_infra/` | losses (unit-tested), training scaffold, data generators |
| `data/derived/` | DVC-tracked corpora (pointers in git, bytes outside) |
| `artifacts/experiments/` | checkpoints, sidecars, per-step metric logs (`*_steps.jsonl`, `*_gates.jsonl`) |
| `JOURNAL.md` | hypotheses, verdicts, retractions — every number is a printed script verdict |

Current layout details: [repo_structure.md](repo_structure.md).

## Reproduce

```bash
pip install -e .[nn]        # torch is the [nn] extra
dvc pull                    # corpora (incl. the stratified turn-fork set)
pytest                      # unit tests; losses have their own: python -m catspace.research.tools.training_infra.losses

# train the current stack (champion recipe + joint quantized training + streaming ingest)
python -m catspace.research.components.encoder.approaches.reach_probability.experiments.train_reach_vit \
  --games 4000 --sf-only --n-piecedown 27006 --split-head 1 --basin-pov white --w-basin 1000 \
  --mirror 1 --move-head 1 --w-hinge-a 10 --w-anchor-a 1 --w-pole-gas-a 5 \
  --w-qdistill 300 --qdistill-npz artifacts/experiments/reach_v2_latest_search_labels.npz \
  --jqt 1 --balance-npz data/derived/balance_weights_3256261.npz \
  --ingest-tsv data/derived/stratified_sfsf_moves.tsv \
  --steps 20000 --out artifacts/experiments/my_run

# gates: search battery, internal arena, corpus audit, race battery
python -m catspace.research.components.planner.approaches.quasimetric_nav.search_sanity --ckpt <ckpt>
python -m catspace.research.components.planner.approaches.quasimetric_nav.kittychess_arena --candidate <ckpt>
python -m catspace.research.components.encoder.approaches.reach_probability.experiments.audit_data_balance
python -m catspace.research.components.planner.approaches.quasimetric_nav.race_battery --ckpt <ckpt> --jqt <ckpt>_jqt.pt
```

## Visualize

```bash
# the analysis board (lichess-style: streaming lines, tricolor eval bar, concepts, play mode)
python -m catspace.research.components.planner.approaches.quasimetric_nav.kittychess_server --ckpt <ckpt> --port 8420

mlflow ui                   # every run: per-step losses + eval-cadence health gates
```

Training curves are also plain files next to each checkpoint: `*_steps.jsonl`
(every step: all loss terms, codebook perplexity, code flip-rate) and
`*_gates.jsonl` (every eval: effective ranks, basin spread, val metrics).
