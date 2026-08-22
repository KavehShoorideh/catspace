#!/bin/zsh
cd /Users/kav/code/remote/github/catspace/artifacts/experiments
while true; do
  ls -t gko2c_step*.pt 2>/dev/null | tail -n +3 | grep -v step2000 | while read f; do rm -f "$f"; done
  sleep 600
done
