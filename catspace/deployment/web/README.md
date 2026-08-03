# ui/ — interfacing with the engine (scaffold)

The engine speaks two protocols today, both of which this directory will grow a
front-end for:

- **UCI** (`catspace/uci.py`) — the engine as a standard chess engine; any chess
  GUI can already drive it.
- **HTTP viz server** (`experiments/viz/assistant_server.py`) — the existing
  research server (embedding maps, per-position readouts, MPS-backed).

Planned here: a thin server wrapping the engine's per-move **plan trace** —
target structure (exemplar boards), P_fall, horizon, priced concession — plus a
board UI that renders it. Under the central hypothesis the trace *is* the
product, so the UI's job is to show the explanation, not just the move.
