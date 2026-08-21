#!/bin/zsh
while true; do
  for pid in $(pgrep -f "oracle_service loop") $(pgrep -f "gko_data") \
             $(pgrep -f "training_infra.acquisition") $(pgrep -f "gko_train") \
             $(pgrep -f "gko_chain_values") $(pgrep -x stockfish) $(pgrep -x lc0); do
    taskpolicy -B -p "$pid" 2>/dev/null
  done
  sleep 300
done
