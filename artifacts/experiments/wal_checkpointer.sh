#!/bin/zsh
cd /Users/kav/code/remote/github/catspace
while true; do
  sleep 60
  .venv/bin/python -c "
import sqlite3
try:
    c = sqlite3.connect('data/derived/gko1_oracle.sqlite', timeout=30)
    c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    c.close()
except Exception:
    pass" 2>/dev/null
done
