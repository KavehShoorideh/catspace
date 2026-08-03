# infra/cloud — storage & compute scaffolding

**Storage:** datasets are DVC-tracked (pointers in git). To add a remote:
`dvc remote add -d store s3://...` (or gdrive/ssh) then `dvc push`. Checkpoints
+ metrics live under `artifacts/experiments/` — sync with the same remote when
runs move off the MacBook.

**Compute (spot workflow):** every trainer obeys the preemption contract, so
spot/preemptible instances are safe:
1. instance up → `git pull && dvc pull && pip install -e .[nn]`
2. `experiments/launch.sh <name> -- <trainer> --resume auto ...`
3. preemption sends SIGTERM → trainer checkpoints + exits 0
4. next instance repeats step 2 — `--resume auto` continues from `_latest.pt`.

Nothing here assumes a provider; it is deliberately just the contract.
