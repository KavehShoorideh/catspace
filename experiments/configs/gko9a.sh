#!/bin/zsh
# gko9a -- LINEAR concept readouts (Kaveh 2026-08-25). Every head is a linear map of
# the binary concepts; WDL is one Linear(concepts->3)+softmax shared by root and rolled
# states. Trunk 4 layers. Sparsity by proximal soft-thresholding, lambda 0.5.
# The architecture contract this run must satisfy is docs/ARCHITECTURE.md.
# Kaveh 2026-08-24: "after this front finishes, run the joint training ... that
# includes the predictions."
#
# Fresh 8000 steps rather than resuming gko7a: the point of JOINT training is that the
# concepts are shaped by the future objective from step 0, not bent by it afterwards.
# Warm-starting the field from gcf4h is the same thing gko7a did, so gko7a is the
# directly comparable predecessor.
set -e
cd /Users/kav/code/remote/github/catspace
ARGS=(
  --batch 48 --chain-batch 16
  --chain-graphs data/derived/gko3_chain_graphs_mm.npz
  --chain-starts data/derived/gko1_chain_rows.npz
  --chain-values data/derived/gko1_chain_values.npz
  --ckpt-every 2000 --d-code 8 --d-phi 128 --ema-decay 0.999 --eval-every 1000
  --g-codes 8 --g-heads 32 --gen5 --iv --k-bins 64 --lift transformer --lift-layers 3 --trunk-layers 4
  --lr 0.0001 --max-rows 600000 --max-shards 200 --op6-after 1000 --ops
  --ops-after 300 --rescan-every 500
  --resume-field artifacts/experiments/gcf4h_latest.pt
  --row-elo data/derived/gko1_row_elo_2126316.npz --seed 0
  --shards data/derived/gko1_shards --sib-cap 16 --slow-heads 16 --sq-codes 64
  --threads 6 --vamp-batch 128 --vamp-every 4
  --w-av 1.0 --w-cons 0.05 --w-dec 0.5 --w-decor 0.3 --w-dyn2 1.0 --w-iv 1.0
  --w-maia 1.0 --w-move 2.0 --w-op1 1.0 --w-op3 0.5 --w-op6 0.3 --w-phi 0.0
  --w-sf 1.0 --w-slow 1.0 --w-spec 2.0 --w-v0 0.5 --w-vk 1.0
  --gen6 --binary --depth-cost --w-depth 0.1 --av-concepts --mv-concepts --linear-heads --w-l1 0.5 --l1-target 320 --resume-self artifacts/experiments/gko9a_step2000.pt
  --op-bases 128 --op-rank 32 --w-sib 1.0 --u-pen 0.0 --all-ply --w-opk 0.0
  --state-mix --sib-pl --lr-ops 1e-4 --steps 8000
  # --- the joint objective ---
  --w-fkind "${W_FKIND:-0.2}" --fkind-K "${FKIND_K:-8}"
  --fkind-dir data/derived/gko1_fkind
)
if [[ "$1" == "gates" ]]; then
  exec .venv/bin/python -u -m catspace.research.components.encoder.approaches\
.reach_probability.experiments.gko_train "${ARGS[@]}" --units \
    --out artifacts/experiments/gko9gate
fi
if [[ "$1" == "smoke" ]]; then
  exec .venv/bin/python -u -m catspace.research.components.encoder.approaches\
.reach_probability.experiments.gko_train "${ARGS[@]}" \
    --steps 60 --eval-every 50 --ckpt-every 1000 --out artifacts/experiments/gko9smoke
fi
exec .venv/bin/python -u -m catspace.research.components.encoder.approaches\
.reach_probability.experiments.gko_train "${ARGS[@]}" --out artifacts/experiments/gko9a
