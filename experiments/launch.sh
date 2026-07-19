#!/usr/bin/env bash
# experiments/launch.sh — durable background launcher for long-running jobs
# (training, big evals). Detaches the job fully from the terminal AND from
# Claude Code, so it survives closing VSCode, the terminal, or the Claude Code
# session.
#
# Mechanism:
#   nohup                 -> ignore SIGHUP (terminal close can't kill it)
#   background + disown    -> drop from the shell job table; the process
#                             reparents to launchd (PID 1) = fully detached
#   caffeinate -i -w <pid> -> block idle sleep for EXACTLY the job's lifetime,
#                             then let go (auto-exits when the job exits)
#
# Output goes to a timestamped logfile; a stable "<name>.log" symlink always
# points at the newest run, so `tail` targets stay predictable across runs and
# sessions. The launched command's PID is written to "<name>.pid".
#
# Usage:
#   experiments/launch.sh <name> -- <command...>
#
# Example (the field run):
#   experiments/launch.sh qrl_iqe_sn_full -- \
#     .venv/bin/python -u experiments/train_lichess_fb.py \
#     --ckpt data/derived/sep/qrl_iqe_sn_full.pt --steps 40000 ...
#
# Monitor:  tail -f artifacts/experiments/qrl_iqe_sn_full.log
# Stop:     kill "$(cat artifacts/experiments/qrl_iqe_sn_full.pid)"
set -euo pipefail

if [[ $# -lt 3 || "$2" != "--" ]]; then
  echo "usage: experiments/launch.sh <name> -- <command...>" >&2
  exit 2
fi
name="$1"; shift 2   # consume <name> and the "--" separator

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="$repo_root/artifacts/experiments"
mkdir -p "$log_dir"

stamp="$(date +%Y%m%dT%H%M%S)"
log="$log_dir/${name}_${stamp}.log"
link="$log_dir/${name}.log"
pidfile="$log_dir/${name}.pid"

# Record the exact invocation at the top of the log (for replication).
{
  echo "# launched : $(date)"
  echo "# host     : $(hostname)"
  echo "# cwd      : $repo_root"
  echo "# command  : $*"
  echo "# ---"
} >"$log"

cd "$repo_root"
# Detach the job. nohup execs the command, so $! is the command's own PID.
nohup "$@" >>"$log" 2>&1 &
pid=$!
echo "$pid" >"$pidfile"
# Keep the Mac awake only while this job runs; -w makes caffeinate exit when
# the job does, so we never leave sleep disabled after a run finishes.
nohup caffeinate -i -w "$pid" >/dev/null 2>&1 &
disown -a
ln -sf "$(basename "$log")" "$link"   # stable pointer to the newest run

echo "launched '$name'"
echo "  pid  : $pid   (-> $pidfile)"
echo "  log  : $log"
echo "  tail : tail -f $link"
echo "  stop : kill $pid"
