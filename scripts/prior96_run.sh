#!/bin/bash
# runs one module of plan 78.96 in this worktree with the standing environment: scripts/prior96_run.sh <module> <args...>; output to the scratchpad log named by $LOG
HERE=/Users/kav/code/remote/github/catspace/.claude/worktrees/agent-a8312bf7c6727b55f
cd "$HERE" || exit 1
mod=$1
shift
env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 PYTHONPATH="$HERE" LCP_DATA=/Users/kav/code/remote/github/catspace/data "$HERE/.venv/bin/python" -m "catspace.incubator.analyst.discovery.$mod" "$@"
