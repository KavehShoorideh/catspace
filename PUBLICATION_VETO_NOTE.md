# PUBLICATION PLAN: "The Opponent's Veto, Learned" (conditional — gate below)

Kaveh 2026-07-23: if the multichannel veto works, ship (1) a self-contained note/post
(LinkedIn + repo), (2) an interactive HTML on GitHub with a viz highlighting the claim,
letting people play positions and analyze with our engine, reusing lichess code where
licensing allows.

## The gate (hard, printed-verdict only)

`experiments/measure_veto_channels.py` on the trained multichannel field (`lichess_mc`,
25k rung for the early read, 50k final): does the LEARNED channel gap
d(F(s; sf-optimal)) − d(F(s; random)) separate exactly-denied from exactly-forceable
target regions (tablebase ground truth)?
**PASS = AUC ≥ 0.65 and anchor-level spearman ≥ +0.4.** Below that: no publication;
back to training (more regime data / longer / sf-vs-weak channel).

## The claim (one sentence, for the note)

A chess engine's map can LEARN the opponent's veto — which winning futures an opponent
will simply never allow — as the divergence between two learned distances (purposeful-play
vs drift), with no tablebase at play time; mistakes appear as the veto gap COLLAPSING,
which is what a tactical opportunity is.

## Note structure (draft after the verdict; numbers only from printed VERDICTs)

1. Hook: 87% of the winning positions you might aim at, a good opponent never lets you
   reach — planning at exact positions is provably hopeless; regions are 99% forceable.
   (The measured veto.)
2. Two reachabilities: could-happen vs will-be-allowed; the gap IS the opponent.
3. The learned version: multichannel field, channels as play-regimes; the veto as metric
   divergence (the gate verdict + figure).
4. A tactic = a veto lapse: the gap collapsing after a mistake (tactic_events flip-rate
   ratio if available).
5. Interactive demo link; limitations (toy domain, support caveats), what's next.

## Interactive demo (GitHub)

- Reuse: the repo's play UI already runs chess.js client-side rules with our server
  endpoints (/engine_move /project /analyze). Add the VETO OVERLAY: for the current
  position, show candidate target regions colored by learned gap (denied=red …
  forceable=green), updating per move — a blunder visibly flips regions green (the
  spidey-sense, on screen).
- Two tiers: (a) STATIC GitHub Pages build — N curated positions with PRECOMPUTED
  engine analyses + veto maps (no server; works for everyone); (b) full local mode —
  `git clone && make demo` runs the Python server for free play.
- Licensing homework (use lichess code "as much as possible and allowed"):
  chessground (lichess board UI) = GPL-3.0 → fine if the demo frontend is GPL-compatible;
  chess.js = BSD-2 (already in use); lichess puzzle DB = CC0; cburnett piece SVGs =
  CC-BY-SA (attribute). Our repo licensing must be checked/aligned before bundling
  chessground; fallback = keep the current hand-rolled board.

## Sequencing

25k rung → early gate read (also per-channel health + d_step/d_rand per regime).
50k final → full gate + in-stratum cohesion + E1 flip-rate. Gate passes → draft note
(here), build veto overlay, curate demo positions (include one blunder→gap-collapse
story position), Kaveh reviews before anything goes public.

## Visualization option space (Kaveh 2026-07-23: "other ways to convey the point")

Design principle: EVERYTHING RENDERS ON THE BOARD, not in embedding space.

Evidence-carriers:
  1. GAP-COLLAPSE SEISMOGRAPH (top pick): replay a real tactic game; learned veto-gap of the
     tactic's basin plotted under the board; blunder ply = cliff; strike follows. "The map saw
     it before the player did."
  2. ON-BOARD VETO HEATMAP (demo core): per-square "reachable if he lets me" vs "if he fights"
     toggle; blunders re-light extinguished squares. Native lichess idiom.
  3. GHOST REFUTATIONS: hover a denied region -> ghost pieces play the refuting line (witness
     from the forceable() DFS). Every red square carries its proof.
Concept-carriers:
  4. TWO-CONE hero figure (could-reach cone vs will-be-allowed thread, 87% shaded).
  5. POINT-vs-REGION before/after (13% red vs 99% green).
  6. TREE-PRUNING animation (opponent moves grey out swaths; blunder re-lights a limb).
Differentiators:
  7. ELO SLIDER: regions flip denied->reachable with opponent strength; ORANGE TRAP regions
     (defense exists, this cohort won't see it) once the opponent model feeds in. v1 = two
     channels; full continuum needs channel 3 + opponent model.
  8. WATCHLIST FEED: per opponent move, "+2 newly forceable / -1 closed" notification panel.
  9. MULTIPLICITY BRAID (later): forcing paths as a braid; tactic = single thread.

Note package: two-cone + seismograph + strength-per-node chart. Demo package: heatmap toggle +
ghost refutations + watchlist feed; Elo slider as marquee when channel 3 lands.
