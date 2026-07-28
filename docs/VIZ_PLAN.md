# VIZ_PLAN — interactive visualization suite for catspace real-board results

This is an implementation plan for a set of **interactive, self-contained HTML viewers**
covering (a) *training-time* visualization (how the FB embedding improves) and
(b) *play-time* visualization (how the engine steps through its decisions).
It is written to be executed step-by-step without further design decisions.
Follow it exactly; every payload schema, file path, and API named here has been
verified against the current repo state (commit 58ffe0e + working tree, 2026-07-11).

---

## 0. Ground rules (read first)

1. **Python**: always run with the project venv: `.venv/bin/python`. Verified available:
   `torch`, `python-chess` (incl. `chess.svg`), `numpy`, `openTSNE`, `matplotlib`, `scipy`.
   **NOT installed**: `umap` — do not use it.
2. **The HTML pattern** (already established in this repo): every viewer is ONE
   self-contained HTML file. A template in `catspace/viz/templates/<name>.html`
   contains the exact line
   ```
   const DATA = /*__DATA__*/;
   ```
   A builder script computes a JSON-serializable `dict` and calls
   `catspace.viz.build_html.build_html(template_path, data, out_path)`, which
   replaces that placeholder. **No CDN imports, no external JS libraries, no
   fetch()** — all charts are hand-rolled inline SVG (see the existing templates
   for the style). All numpy values must survive JSON: `build_html` already uses
   `catspace.viz.payload.json_default`, so plain numpy scalars are fine, but
   convert arrays to lists yourself.
3. **File layout**:
   - builder scripts → `experiments/viz/build_<name>.py` (executable, argparse,
     `#!/usr/bin/env python` — copy the header style of `experiments/viz/build_krk_viewer.py`)
   - templates → `catspace/viz/templates/<name>.html`
   - reusable payload logic → `catspace/viz/realboard.py` (NEW module, §1)
   - outputs → `catspace.io.paths.generated_dir()` (= `artifacts/generated/`)
4. **Timing + journal** (project rule, non-negotiable): every builder prints
   per-stage wall-clock timings (`time.time()` deltas, like `decompose_demo.py`
   does) and after each successful build, append a short entry to `JOURNAL.md`
   (repo root, newest entry LAST): date, command run, timings, output path,
   one-line observation of what the viz shows.
5. **Dark style**: reuse the CSS palette of `catspace/viz/templates/fullboard_viewer.html`
   (`body #14161a`, panels `#1b1e24`, border `#2a2e36`, text `#cfd3da`,
   accents `#7aa2ff` blue / `#66d9a6` green / `#ffb86b` orange).
6. **Key data facts** (verified):
   - Shards: `catspace.io.paths.newest_shard_dir()` → dir of `shard_*.npz`, each with
     columns `packed (n,13) uint64? bit-packed`, `meta`, `ply`, `clock`, `eval_cp`
     (float32, **NaN when unannotated**), `result` (white-POV −1/0/+1), `white_elo`,
     `black_elo`, `game_id`. `game_id` is non-decreasing within a shard; a game's
     rows are contiguous. **Holdout = `game_id % 50 == 0`.**
   - Checkpoints: `derived_dir()/lichess_fb.pt` (step 30000) and
     `derived_dir()/lichess_fb_step2000.pt`. Load with
     `catspace.nn.fb.load_ckpt(path, device) -> (fb, payload)`;
     `payload["zgoals"]` has `MATE_W`, `MATE_B` (torch CPU tensors, shape (64,)).
     `MATE_DIFF` = `MATE_W − MATE_B`. **Always unit-normalize a z before dotting
     with F** (`z / np.linalg.norm(z)`) so reach is a cosine like everything else.
   - Eval heads: `derived_dir()/eval_heads.pt` (repr F), `eval_heads_B.pt`,
     `eval_heads_FB.pt`. Load with
     `catspace.nn.eval_head.load_heads(path, device) -> (desc, norm, meta)`;
     score with `head.expected_score(f) -> [0,1]` (white expected score).
     d_in is 64 for F/B, 128 for FB.
   - Embedding calls: `fb.embed_F(planes, om)`, `fb.embed_B(planes)` — planes from
     `catspace.nn.features.feature_planes(packed, meta)` (numpy → torch),
     omega rows from `catspace.nn.features.omega_ids(white_elo, black_elo, clock)`
     (numpy arrays in, (n,3) int out). Outputs are already L2-normalized.
     Copy the batched-embedding loop from `experiments/decompose_demo.py::embed_rows`.
   - Boards: `catspace.data.encode.board_from_packed(packed_row, meta_row) -> chess.Board`,
     inverse `encode_packed(board)`, `encode_meta(board)`.
   - Stockfish winprob: `catspace.nn.features.winprob_cp(cp)` (NaN passes through).
   - Training logs: `artifacts/generated/logs/train_30k.log` (and `.time`); line formats in §D2.
   - Projections: `catspace.viz.projection` — use `PCAProjection` / `TSNEProjection`
     directly (fit on a normalized subsample, `transform` the rest). The
     `fit_map()` convenience function is toy-specific (wants dtm/won) — do NOT use
     it for real boards; construct `Normalizer.fit` + projection by hand.
   - Play layer: `catspace.realboard.play_board_game(white, black, ...) -> BoardGameRecord`
     (`.moves` = UCI strings incl. opening, `.result`, `.termination`);
     `catspace.nn.policy_fb.FBBoardPolicy(fb, z, depth, elo, clock, device)`;
     opponents: `catspace.realboard.RandomBoardPolicy`, `catspace.uci.UCIBoardPolicy`.
   - Decomposer: `catspace.planner.decompose` — `WaypointPool(F, B, labels)`,
     `decompose(F_s, z_g, pool, tau_exec, tau_floor, ...) -> Decomposition` with
     `.root: HopNode` (fields `reach, depth, status, waypoint, bottleneck, left,
     right, detail`), `.executable`, `.block_rule`, `.waypoints` (pool indices in
     play order). Calibration recipe for `tau_exec`/`tau_floor` is in
     `experiments/decompose_demo.py` lines 117–136 — copy it verbatim.
7. **Board SVGs**: `import chess.svg; chess.svg.board(board, size=400, lastmove=mv,
   arrows=[...])` returns a full `<svg>` string. Embed the string in the payload;
   templates insert it with `innerHTML` (the existing `fullboard_viewer.html`
   already does `document.getElementById("board").innerHTML = p.svg`).
   Payload-size budget: an SVG is ~4–8 KB; keep each HTML file under ~15 MB
   (e.g. 12 games × ≤120 plies is fine).
8. **Performance traps**: bind every npz array ONCE (`data = {k: npz[k] for k in npz.files}`)
   — `NpzFile.__getitem__` re-reads the whole array each call. Use `--device cpu`
   as the default for builders (they're demo-sized; MPS may be busy training).
9. **Testing**: for each deliverable add one fast pytest in `tests/test_viz_builders.py`
   that exercises the payload function on tiny synthetic inputs (no shards, no
   checkpoints — construct 2–3 boards with `chess.Board()` and random unit
   embeddings) and asserts the payload is JSON-serializable
   (`json.dumps(payload, default=json_default)`) and has the required keys.
   Builders themselves are not unit-tested end-to-end.

**Build order** (each deliverable is independent; do them in this order —
earliest ones have the highest value/effort ratio): D2 → D1 → D3 → D4 → D5 → D6 → D7 → D8.

---

## 1. Shared module: `catspace/viz/realboard.py` (NEW)

Reusable payload helpers for all real-board viewers. Implement exactly these
functions (signatures given; keep them numpy/torch-light and unit-testable):

```python
def load_games_from_shard(shard_dir, n_games, seed=0, holdout_only=True,
                          min_plies=20, max_plies=160, want_results=(1, -1, 0)):
    """Scan ONE shard file (the first, sorted) and return up to n_games complete
    games as dicts of per-ply arrays sorted by ply:
      {packed, meta, ply, clock, eval_cp, result, white_elo, black_elo, game_id}.
    'Complete' = the game's first stored ply <= 1 (games can be split across
    shard boundaries; skip truncated ones). Balance across want_results
    round-robin (a win, a loss, a draw, a win, ...) so viewers get contrast.
    Holdout filter: game_id % 50 == 0."""

def games_from_pgn(path):
    """Parse a PGN file (e.g. artifacts/generated/arena_real.pgn, written by
    experiments/arena_real.py --save-pgn) with chess.pgn and return games as
    lists of (board_before_move, san, uci) plus headers (White, Black, Result).
    Include the final position with san=None."""

def infer_san(prev_board, packed_next, meta_next):
    """Recover the SAN of the move between two consecutive stored positions:
    try each legal move of prev_board, encode the child with encode_packed/
    encode_meta, and compare with np.array_equal to the stored next row.
    Return SAN string or None (gap / no match). O(legal moves) per ply — fine."""

def embed_positions(fb, packed, meta, white_elo, black_elo, clock, device, batch=2048):
    """Batched F and B for arbitrary rows using the rows' OWN omega
    (omega_ids of the true elos/clock). Returns (F, B) numpy, unit rows.
    Copy the loop shape of decompose_demo.embed_rows but with per-row omega."""

def board_svg(board, lastmove=None, arrows=(), size=400):
    """chess.svg.board wrapper; returns the SVG string."""

def fit_projection(F_bg, kind="pca", seed=0, perplexity=40.0):
    """Fit Normalizer + {PCAProjection|TSNEProjection} on F_bg; return an object
    with .transform(F) -> (n,2) float. (Thin wrapper so builders share one code
    path; do NOT use viz.projection.fit_map — it is toy-specific.)"""
```

Unit tests: `infer_san` on a 3-move game built by hand; `board_svg` returns a
string starting with `<svg`; `fit_projection(kind="pca")` round-trips shapes.

---

## D1 — Full-board trajectory viewer (eval heads on real games) **[template exists]**

**Goal**: watch reach + both eval heads evolve ply-by-ply through real games —
holdout human games AND the engine's own arena games — on a 2D map of the
embedding. This is the flagship "results so far" viewer.

**Template**: `catspace/viz/templates/fullboard_viewer.html` **already exists
and is final** — do not redesign it; only build the payload it expects.
(Verified expectations, from reading the template:)

```jsonc
{
  "meta":  { "title": "catspace — full-board cone viewer  ·  ckpt step 30000" },
  "map":   { "bg":    [[x, y], ...],          // background cloud, ~4000 pts
             "reach": [r, ...] },             // same length as bg; colors the cloud
  "games": [
    { "name": "win vs 1832 (gid 4150)",       // button label
      "result": "1-0",
      "plies": [
        { "ply": 12, "san": "Nf3",            // san may be null
          "xy": [x, y],                       // projected F(s)
          "reach": 0.113,                     // F(s|ω_row) · unit(z_MATE_DIFF)
          "e_desc": 0.61, "e_norm": 0.55,     // expected_score of each head; null ok
          "svg": "<svg ...>" }, ... ] }, ... ]
}
```

**Builder**: `experiments/viz/build_fullboard_viewer.py`

CLI: `--shards --ckpt --heads (default derived_dir()/eval_heads.pt) --pgn
(optional path; adds arena games) --n-games 9 --projection {pca,tsne} (default
pca) --n-bg 4000 --device cpu --seed 0 --out (default
generated_dir()/fullboard-viewer.html)`.

Algorithm:
1. Load ckpt + heads. `z = unit(MATE_W − MATE_B)`.
2. Background: `sample_shard_rows(shard_dir, n_bg, seed, holdout_only=True)` →
   `load_rows` (copy from decompose_demo) → `embed_positions` with true omega →
   fit projection on those F rows → `map.bg = round(xy, 2)`, `map.reach = F @ z`.
3. Shard games: `load_games_from_shard(..., n_games)`; per game, embed each ply
   (true omega), compute reach / e_desc / e_norm (`head.expected_score` on the
   F tensor; e_norm only from the F-repr heads — d_in 64), `xy = proj.transform`,
   SAN via `infer_san` (lastmove arrow: derive the move object the same way and
   pass to `board_svg`), SVG per ply.
4. PGN games (if `--pgn`): replay with `games_from_pgn`; encode each position with
   `encode_packed/encode_meta`; omega = elo 1800 / clock 300 both sides (matches
   the arena's `--elo-cond` default). Name buttons `"FB vs random #3"` etc.
5. `build_html(template, data, out)`; print timings; JOURNAL entry.

Acceptance: open the HTML — game buttons render, slider + arrow keys step plies,
three curves draw (blue reach, green e_desc, orange e_norm), map trajectory
follows the cursor. Estimated build time: ~1–3 min on CPU.

---

## D2 — Training dashboard (how we're improving) **[no model needed]**

**Goal**: interactive loss/metric curves parsed from the existing training logs.
Zero torch imports — pure log parsing, so it always works even mid-training
(re-run it to refresh).

**Builder**: `experiments/viz/build_training_dashboard.py`
CLI: `--logs (one or more paths or a glob; default artifacts/generated/logs/train_*.log)
--out (default generated_dir()/training-dashboard.html)`.

Parse per file (regexes, anchored):
- `^step (\d+)  loss ([\d.]+)  train_top1 ([\d.]+)  \(([\d.]+) it/s\)$`
- `^  VAL step (\d+)  loss ([\d.]+)  top1 ([\d.]+)  top8 ([\d.]+)$`
- final `^VERDICT .*$` lines (keep verbatim strings)
- `resumed .* at step (\d+)` → mark a resume boundary (vertical dashed line)

Payload:
```jsonc
{ "meta": {"title": "catspace — FB training dashboard", "built": "<iso date>"},
  "runs": [ { "name": "train_30k",
              "train": {"step": [...], "loss": [...], "top1": [...], "rate": [...]},
              "val":   {"step": [...], "loss": [...], "top1": [...], "top8": [...]},
              "chance": {"top1": 0.00195, "top8": 0.0156},   // 1/512, 8/512
              "resumes": [2000], "verdicts": ["VERDICT ..."] } ] }
```

**Template**: `catspace/viz/templates/training_dashboard.html` (NEW). Layout:
four SVG line-chart panels in a 2×2 grid (hand-rolled like `#curves` in
fullboard_viewer.html): (1) loss — train (thin, faint) + EMA-smoothed train
(bold) + val (dots+line); (2) top1 — train EMA + val + dashed chance line;
(3) top8 val + dashed chance; (4) it/s. Controls: run checkboxes (overlay
multiple runs in distinct hues), an EMA smoothing slider (α 0…0.99, applied in
JS), log-y toggle for the loss panel, and a hover crosshair that shows exact
values in a status line. Verdict strings printed under the charts. Resume steps
drawn as dashed vertical lines.

Acceptance: `.venv/bin/python experiments/viz/build_training_dashboard.py` runs
in <5 s and the chart shows the 30k run descending from ~4.5 to ~4.1 with VAL
points on top. Journal it.

---

## D3 — Decision viewer (how the engine steps through play)

**Goal**: per-ply inside view of `FBBoardPolicy`: every candidate move's score,
the feared opponent reply behind each depth-2 score, and what got chosen.

**Step 1 — instrument the policy** (small refactor of
`catspace/nn/policy_fb.py`, keep behavior identical): extract the scoring body
of `move()` into

```python
def move_scored(self, board, rng) -> tuple[chess.Move, list[dict]]:
    """As move(), but also returns per-candidate dicts:
    {uci, san, score, kind}  where kind ∈ {"mate","draw","mated","reach"} and,
    for depth-2 reach scores, {"feared_uci","feared_san","feared_score"} = the
    argmin opponent reply that produced the min. Terminal sentinels map to
    kind; score stays the raw float (±1e9/−2e9) for sorting but viewers display
    kind instead of the sentinel."""
```
`move()` becomes `return self.move_scored(board, rng)[0]`. Existing arena tests
must still pass (`python -m pytest tests -k policy or arena -q` — run the full
fast suite to be safe).

**Step 2 — builder**: `experiments/viz/build_decision_viewer.py`
CLI: `--opponent {random,sf:<elo>} (default random) --games 6 --depth 2
--opening-plies 6 --max-plies 200 --elo-cond 1800 --ckpt --device cpu --seed 0
--out (default generated_dir()/decision-viewer.html)`.

Play games exactly like `experiments/arena_real.py` (alternating colors, seeded
`np.random.default_rng([seed, i])`), but at each FB ply call `move_scored` and
record. Also record opponent plies (candidates=null). Per ply store:

```jsonc
{ "ply": 17, "mover": "fb" | "opp", "san": "Rd1", "fen": "...",
  "svg": "<svg with lastmove + arrows>",   // arrows: top-5 candidates colored by score rank
  "reach_after": 0.21,                     // F(board after chosen move) · unit(z_goal of FB's color)
  "cands": [ { "san": "Rd1", "score": 0.213, "kind": "reach",
               "feared_san": "Qxd1", "feared_svg": "<svg>", "chosen": true }, ... ] }
```
Cap `cands` at 8 (sorted desc, chosen always included); include `feared_svg`
only for the top 4 (size budget). Game-level: name, result, termination,
fb_color. Top-level: meta (ckpt step, opponent, depth), games list.

Arrows: use `chess.svg.Arrow(from_sq, to_sq, color=...)` — green for chosen,
blues fading by rank for the rest.

**Template**: `catspace/viz/templates/decision_viewer.html` (NEW). Layout: game
buttons; main board (left); candidate table (right) — columns move / kind or
score / feared reply, chosen row highlighted; click a candidate row → small
preview board below the table showing the feared-reply SVG; reach sparkline
across the whole game at the bottom (FB plies only), click-to-seek; slider +
arrow keys like fullboard_viewer.

Acceptance: build vs random (fast, no Stockfish dependency) — FB should win
most games; stepping through shows sensible mate-hunting near the end. ~1–2 min.

---

## D4 — Decomposition explorer (how the planner thinks)

**Goal**: interactive view of `Decomposition` trees on real middlegames: the
hop tree with reaches/bottlenecks/give-up badges, waypoint boards, and
population histograms.

**Builder**: `experiments/viz/build_decompose_viewer.py`
CLI mirrors `experiments/decompose_demo.py` (`--n-pool 20000 --n-starts 60
--start-ply-lo 20 --start-ply-hi 40 --max-depth 3 --dry-gain 0.02 --device cpu
--seed 0`) plus `--n-show 24` (how many starts get full trees in the payload)
and `--out (default generated_dir()/decompose-viewer.html)`.

Algorithm: reuse decompose_demo's pipeline verbatim (sample holdout rows, embed
under planner omega 1800/300, calibrate tau_exec/tau_floor from the first
shard, run `decompose` per start). Then:
1. Rank starts by gain (`plan_bottleneck − direct`); keep top `--n-show` plus
   the 3 worst (contrast).
2. Serialize each tree recursively:
   ```jsonc
   { "reach": 0.031, "depth": 0, "status": "decomposed",  // or executable / no_midpoint / ...
     "bottleneck": 0.46, "detail": "",
     "wp": { "fen": "...", "ply": 57, "svg": "<svg size=240>" },  // null on leaves
     "left": {...}, "right": {...} }
   ```
   (waypoint pool index → row → `board_from_packed` → fen/ply/svg; use
   `size=240` for waypoint boards, `size=320` for start boards.)
3. Per start also: start fen/ply/svg, direct reach, plan_bottleneck, gain,
   executable flag, block_rule, waypoint chain in play order (fens + plies).
4. Population stats over ALL `--n-starts` (not just shown ones):
   `hist_gain` (20 bins), `hist_waypoint_ply`, `hist_start_ply`, scalar verdicts
   (FRAC_IMPROVED, MEAN_GAIN, FRAC_EXECUTABLE, MEAN_WAYPOINTS, block-rule
   counts, tau_exec, tau_floor). Print the same VERDICT line the demo prints.

**Template**: `catspace/viz/templates/decompose_viewer.html` (NEW). Layout:
- left rail: start list (sorted by gain, showing `direct → bottleneck` and a
  colored status chip: green executable / orange give-up rule).
- center: the tree drawn as nested horizontal boxes (root spans full width;
  children split it; leaf boxes colored by status; each internal node labeled
  `min(legs)=<bottleneck>` and clickable → shows its waypoint board + fen in
  the right panel). Depth ≤ 3, so nesting is shallow — plain flexbox divs, no
  SVG layout needed.
- right: board panel (start board by default; clicked node's waypoint board
  otherwise) + the waypoint chain in play order as small thumbnails (click to
  enlarge).
- bottom: two histogram SVGs (gain, waypoint-ply-vs-start-ply overlay) + the
  verdict scalars as a stat strip.

Acceptance: matches decompose_demo's printed verdicts for the same seed; trees
render; every internal node's two children legs display; give-up chips only on
the worst starts (at 30k most starts are all-executable). ~1–2 min CPU.

---

## D5 — Embedding atlas (training-time geometry, before/after)

**Goal**: a 2D map of the F-embedding over holdout positions with switchable
coloring, hover boards, and a step-2000 vs step-30000 A/B toggle — the "what
did 15× training buy" picture.

**Builder**: `experiments/viz/build_embedding_atlas.py`
CLI: `--n 8000 --projection {pca,tsne} (default tsne) --ckpt-a
lichess_fb_step2000.pt --ckpt-b lichess_fb.pt --device cpu --seed 0 --out
(default generated_dir()/embedding-atlas.html)`.

Algorithm:
1. One shared holdout sample of `--n` rows (sample_shard_rows + load_rows,
   also loading `white_elo, black_elo, clock, eval_cp, result, ply`).
2. For EACH checkpoint: embed F with true omega; fit the projection on that
   checkpoint's own F (per-ckpt map — geometry is not comparable across ckpts,
   say so in the template legend); compute reach = F @ unit(z_MATE_DIFF of that
   ckpt).
3. Color channels per point (shared across ckpts): `result` (−1/0/1),
   `ply_bucket` (0–19, 20–39, 40–69, 70+), `reach` (per-ckpt, continuous),
   `winprob` (winprob_cp(eval_cp); NaN → gray), `white_elo_bin` (elo_bin()).
4. Hover boards: SVGs for all 8000 points would blow the budget. Instead store
   `fen` per point (≈80 B each — fine) and render the hovered board CLIENT-SIDE…
   **no js chess lib allowed**, so: precompute SVGs for a random 600-point
   subset (`has_svg` flag); on hover show the SVG if present, else show the FEN
   string + ply/result/elos in the tooltip.
5. Payload: `{meta, points: {fen:[...], result:[...], ply:[...], elo:[...],
   winprob:[...], svg_idx:[...]}, svgs:[...], ckpts: [{name, xy:[[x,y]...],
   reach:[...]}, ...]}` (xy rounded to 2 decimals).

**Template**: `catspace/viz/templates/embedding_atlas.html` (NEW): one large
SVG scatter (720px), ckpt toggle buttons (A/B), color-mode radio buttons with a
matching legend (W green / D gray / L red for result; viridis-ish 5-step ramp
for continuous channels — hardcode the hex stops), hover tooltip panel on the
right with board/FEN + metadata. Point radius 2, opacity 0.7.

Acceptance: at step 30000 with color=result there should be visibly more W/L
separation than at step 2000 (that is the point of the viewer); with
color=reach the map should show a smooth gradient. t-SNE on 8k×64 with
openTSNE takes ~1–2 min; print the timing.

---

## D6 — Divergence explorer (descriptive vs normative)

**Goal**: find and inspect trap positions — where human outcomes and Stockfish
eval disagree.

**Builder**: `experiments/viz/build_divergence_explorer.py`
CLI: `--n 6000 --ckpt --heads (F-repr eval_heads.pt) --device cpu --seed 0 --out
(default generated_dir()/divergence-explorer.html)`.

Algorithm: sample holdout rows and KEEP ONLY rows with finite `eval_cp`
(oversample ~8× then filter; annotation rate ≈ 8–10%). Embed F (true omega),
compute `e_desc`, `e_norm` (expected_score), `sf = winprob_cp(eval_cp)`,
`div = e_desc − e_norm`. Payload points: fen, ply, white_elo, black_elo,
e_desc, e_norm, sf, div; plus precomputed SVGs for the 80 largest-|div| points.

**Template**: `catspace/viz/templates/divergence_explorer.html` (NEW):
- main scatter: x = e_norm, y = e_desc, y=x diagonal drawn; color = white Elo
  bin (5-step ramp); click a point → right panel board (SVG if precomputed,
  else FEN text) + all numbers.
- filters: Elo-bin checkboxes, ply-range slider (double-ended: two range
  inputs), |div| threshold slider — all applied in JS.
- bottom: sortable table (click column headers) of the top-80 |div| positions:
  ply, elos, e_desc, e_norm, sf, div; clicking a row selects it in the scatter
  and shows its board.

Acceptance: diagonal-hugging cloud with off-diagonal tails; the top-|div| table
rows show plausible traps (sharp material-imbalance positions). ~1 min.

---

## D7 — Eval calibration & ablation dashboard

**Goal**: one page with the rigor plots for all three probe reprs + the
zero-label baseline: reliability, ROC, and where the signal lives (per-ply,
per-Elo AUC).

**Builder**: `experiments/viz/build_eval_dashboard.py`
CLI: `--n 20000 --ckpt --device cpu --seed 0 --out (default
generated_dir()/eval-dashboard.html)`.

Algorithm:
1. Sample `--n` holdout rows (need result; eval-annotated subset for normative
   panels). Embed F and B once (true omega); `FB = concat`.
2. Score five models on decisive positions (result ≠ 0): heads from
   `eval_heads.pt` (on F), `eval_heads_B.pt` (on B), `eval_heads_FB.pt` (on FB),
   plus baseline `F @ unit(z_MATE_DIFF)` min-max-rescaled to [0,1].
3. Compute IN PYTHON (numpy only, no sklearn — implement AUC via the
   rank-statistic formula and ROC by sweeping 101 thresholds):
   - ROC curves (fpr/tpr arrays, 101 pts) + AUC per model (won vs lost).
   - Reliability: 10 equal-width bins of predicted expected score → mean
     predicted vs empirical P(win among decisive) + bin counts.
   - Per-ply AUC: buckets [0,20), [20,40), [40,70), [70,∞) per model.
   - Per-white-Elo-bin AUC (elo_bin()) per model.
   - Normative panel: scatter sample (800 pts) of e_norm vs sf winprob +
     Spearman per model (implement Spearman as Pearson of ranks via argsort).
4. Payload: all curves/bars as plain arrays; meta records n, ckpt step, seeds.

**Template**: `catspace/viz/templates/eval_dashboard.html` (NEW): 2×3 panel
grid of hand-rolled SVG charts — (1) ROC overlay, (2) reliability with y=x
line, (3) per-ply AUC grouped bars, (4) per-Elo AUC grouped bars, (5) e_norm
vs sf scatter, (6) stat table (AUC / Spearman per model). Model
show/hide checkboxes with a fixed color per model (F blue, B red, FB purple,
baseline gray dashed). Hover values in a status line.

Acceptance: AUCs must reproduce the journaled 30k numbers within ±0.01
(F 0.625, B 0.596, FB 0.636, baseline 0.598) — if not, there is a bug; check
device placement and z normalization first. ~2–4 min.

---

## D8 — Gallery index (5 minutes, do last)

`experiments/viz/build_gallery.py`: scan `generated_dir()` for `*.html`, write
`generated_dir()/index.html` — a dark-styled list of links with file mtimes and
one-line descriptions (hardcode a `{filename: description}` dict for the known
viewers, fall back to the filename). No template needed; emit the HTML directly
(it has no data payload). Re-run after any build.

---

## Runbook (after implementing everything)

```bash
cd /Users/kav/code/remote/github/catspace
.venv/bin/python experiments/viz/build_training_dashboard.py
.venv/bin/python experiments/viz/build_fullboard_viewer.py --n-games 9
.venv/bin/python experiments/viz/build_decision_viewer.py --opponent random --games 6
.venv/bin/python experiments/viz/build_decompose_viewer.py --n-starts 60 --n-show 24
.venv/bin/python experiments/viz/build_embedding_atlas.py --projection tsne
.venv/bin/python experiments/viz/build_divergence_explorer.py
.venv/bin/python experiments/viz/build_eval_dashboard.py
.venv/bin/python experiments/viz/build_gallery.py
open artifacts/generated/index.html
.venv/bin/python -m pytest tests -q          # full fast suite must stay green
```

Then: one JOURNAL.md entry per builder run (timings + one observation each),
and a single commit: builders + templates + `catspace/viz/realboard.py` +
`tests/test_viz_builders.py` + JOURNAL update. The generated HTML is not
committed — `artifacts/generated/` is already in `.gitignore` (verified).
