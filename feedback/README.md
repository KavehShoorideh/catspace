# Feedback queue

The "Something is not right" button in the local analyst (`python -m catspace.incubator.analyst.web`)
appends one JSON line per press to `feedback/flags.jsonl`. The flag's **id is its line number**.
Each line carries: `id`, `time` (UTC), `note` (what the user typed), `context` (what they were looking at:
game id, ply, move, FEN, the headline shown, the terse analysis text, the page URL).

`feedback/RESOLVED.md` is the coder's side: one entry per flag id, what was wrong, what changed, the commit.
A flag is open until it has an entry there. Both files are tracked in git so the queue travels with the code.

Reading the queue:

    python - <<'EOF'
    import json
    for l in open("feedback/flags.jsonl"):
        r = json.loads(l); print(r["id"], r["time"], r["note"], "|", r["context"].get("game"), r["context"].get("ply"), (r["context"].get("headline") or "")[:90])
    EOF
