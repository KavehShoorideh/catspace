# search approaches

Registry for the **search** component: `find_move(state, goal, guidance_fn) -> move`.

Every directory under `approaches/` must have an entry here, and every entry must have a
directory — `scripts/check_approaches.py` enforces both directions. Schema is defined in
`repo_structure.md` § "approaches.md schema".

`status` is one of `active`, `parked`, `superseded-by:<name>`.

---

## puct_mcts

- **folder** — `approaches/puct_mcts/`
- **status** — active
- **hypothesis** — One production PUCT MCTS, taking its guidance function from outside
  (reach-guided, value-guided, or random), serves every search need in the engine; the
  guidance is the experiment, the tree is not.
- **definition of done** — The merged searcher reproduces the results of both predecessors it
  replaced (`nn/mcts.py` and `search/mcts.py`) at equal node budgets, and a new guidance
  function can be plugged in without editing `mcts.py`.
- **notes** — Merged from the two former MCTS implementations in the 2026-08-03 restructure.
  `repricing.py` is the fast-`MemoryField` re-pricing hook split out of the old top-level
  `two_field.py`; the scoring half went to `planner:two_perspective_scoring`. `layer.py` is
  the `MCTSSearch` layer with (value, prior) sockets; `nav.py` is the region navigator.
- **results** — JOURNAL.md (PUCT replaces beam-minimax as the search layer, 2026-07-14)
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## anytime_path

- **folder** — `approaches/anytime_path/`
- **status** — active
- **hypothesis** — Kaveh's search semantics — "we want a path to mate, then when we find one,
  we try for a better one" — is better served by a two-phase anytime search over the FB reach
  field than by a fixed-budget best-first search.
- **definition of done** — First path found no later than the fixed-budget baseline, and
  monotone improvement thereafter when given more time.
- **results** — —
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh
