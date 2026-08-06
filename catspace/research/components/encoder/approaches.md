# encoder approaches

Registry for the **encoder** component: `encode(board, clock, ...) -> embedding`.

Every directory under `approaches/` must have an entry here, and every entry must have a
directory — `scripts/check_approaches.py` enforces both directions. Schema is defined in
`repo_structure.md` § "approaches.md schema".

`status` is one of `active`, `parked`, `superseded-by:<name>`.

---

## reachability_field

- **folder** — `approaches/reachability_field/`
- **status** — active
- **hypothesis** — A frozen community Leela distillate trunk plus our IQE head gives a
  play-time quasimetric `phi` good enough to price "how far to that goal" without training
  a trunk ourselves.
- **definition of done** — Held-out reach ordering beats the material-only baseline, and the
  head runs inside the per-move search budget on one box.
- **results** — JOURNAL.md (M1 field at play time)
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## jepa_tokenizer

- **folder** — `approaches/jepa_tokenizer/`
- **status** — active
- **hypothesis** — An anchored-JEPA stack trained jointly under three losses yields board
  tokens whose geometry carries hazard/reachability structure, and eliminates the lczerolens
  plane-encoding scalar sync that dominated traced decide time.
- **definition of done** — Held-out hazard NLL lift over the frozen-trunk baseline with no
  representation collapse (effective rank held).
- **results** — JOURNAL.md, "JEPA T1 landed": held-out hazard NLL +0.0751 nats (PASS),
  eff_rank 57 -> 77/256, no collapse.
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## cone_fb_embedding

- **folder** — `approaches/cone_fb_embedding/`
- **status** — active
- **hypothesis** — Reach factorizes as `F(s) @ zG` over the discounted successor measure, so
  the quasimetric can be built either tabularly (randomized SVD) or neurally (InfoNCE) behind
  one `QuasimetricEmbedding` seam.
- **definition of done** — On the exact toy domains, the tabular factorization recovers the
  ground-truth reach order, and the neural variant matches it before either is scaled to real
  boards.
- **results** — docs/ALTERNATIVES.md (quasimetric post-mortem and the three escapes)
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## concept_quantization

- **folder** — `approaches/concept_quantization/`
- **status** — active
- **hypothesis** — Plan tokens are better produced by one pluggable VQ/kmeans component with
  an explicit codebook size K than by the seven copy-pasted kmeans implementations it replaced.
- **definition of done** — All former call sites run through this module, and K is a swept
  hyperparameter rather than a constant buried per script.
- **results** — —
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## control_field_wdl

- **folder** — `approaches/control_field_wdl/`
- **status** — parked
- **hypothesis** — A hand-coded per-square control field (net force bearing on a square, SEE
  underneath) plus its directional derivative gives an ascent cone that identifies good moves
  without learning.
- **definition of done** — Ascent-cone membership predicts move quality better than chance on
  held-out positions.
- **why parked** — Kaveh's 2026-08-02 pivot: `C` (and SEE underneath it) is structurally blind
  to the cases that matter, so it is not used to judge moves. Kept for the WDL-decay analysis
  and the ground-truth labels, not on the engine path.
- **results** — `wdl_decay.py` module header; docs/ALTERNATIVES.md
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## reach_probability

- **folder** — `approaches/reach_probability/`
- **status** — active
- **hypothesis** — Trained ONLY on positions that really followed one another in lichess games,
  with no negatives and nothing chess-specific programmed, a JEPA that predicts the *region* of
  reachable futures will infer the irreversible stratification of chess (piece count never rises)
  on its own; and split conformal on held-out positives turns its score into a
  `probability_less_than(a, b, eps)` predicate with a bounded false-"impossible" rate.
- **definition of done** — (1) the PAIRED ratchet score (target held fixed, source varied) beats a
  random-init null, so a source that could reach the target outscores one that could not; (2)
  conformal validity holds at several eps, per bucket, against the multiple-comparison null.
- **results** — (1) **SUPERSEDED (trunk, 2026-08-05 am): NEGATIVE but inconclusive.** Paired ratchet
  0.570 vs a random-init null of 0.555, flat across the ladder — the material ratchet was a property
  of the frozen lc0 trunk, and a pretrained chess net's null is not zero, so the objective could not
  be shown to add anything. (2) **REBUILT from scratch (2026-08-05 pm): the ratchet DOES emerge, in
  trajectory-local geometry.** A randomly-initialised ViT over tokenized boards knows no chess, so
  its null is a true zero. On every-ply trajectories (18.9M positions, 419k observed repetitions),
  the capture-crossing vs quiet reversal differential — both groups unobserved, given byte-identical
  repulsion, ply-gap matched — grows monotonically with non-overlapping CIs:
  −0.057 (null) → +0.056 → +0.124 → +0.303 → +0.465 → +0.645 (12.5k steps), while the
  repetition-covered reversible control stays pinned at 0.95–0.96 and the retracted confounded
  metric stays flat. Population-invariant (human +0.294, SF +0.291), so it is a property of the
  rules, not of a playing population. (3) **The paired ratchet stays at ~0.500 throughout**, so the
  asymmetry does NOT become a transferable cross-game source ranking — a narrower and more specific
  finding than the plan anticipated. Single seed; per-rung CIs bootstrap over pairs, not runs.
