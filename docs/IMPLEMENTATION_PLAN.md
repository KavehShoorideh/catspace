# Implementation Plan — Phases 5–8 (latentchess)

This is a step-by-step build plan for the remaining refactor phases. It is
written to be executed **task by task, in order**, by an implementer who
should NOT need to make design decisions — every module, class, function
signature, semantic rule, test, and verification command is specified. If
something is genuinely ambiguous or a gate fails in a way not covered here,
STOP and ask the user rather than improvising.

Phases 0–4 are already DONE and committed (see git log): package layout,
`TransitionChain` CSR unification, `TerminalScores`, readout (MEAN|MIN +
k-ply backup), Opponent/Policy protocols, game/arena, `CurriculumTrainer` +
checkpoints, `QuasimetricEmbedding` + `ConceptQuantizer` protocols,
`Projection2D`/`FittedMap`/`build_html` viz stack, 35+ passing tests.

---

## 0. Ground rules (read before every task)

### 0.1 Environment & verification loop

```bash
cd /Users/kav/code/remote/github/latent-chess-planner-toys
source .venv/bin/activate            # Python 3.14 venv, package installed -e
python -m pytest tests/ -m "not slow" -q    # after EVERY task — must be green
python -m pytest tests/ -m slow -q          # at phase ends only (~4 min)
```

- Commit after every task with the given message. One task = one commit.
- `git push` at the end of each phase.
- NEVER modify `tests/baselines/` (the regression reference), never commit
  anything under `data/` or `artifacts/generated/` (gitignored), never
  `git push -f`, never delete files outside `code/` without asking.

### 0.2 Code conventions (match the existing package style)

- `from __future__ import annotations` at the top of every new module.
- Absolute imports only: `from latentchess.chain import TransitionChain`.
  NEVER import from the legacy `code/` directory (it still exists until
  task 8.6; it is reference material only).
- numpy-only for everything except `latentchess/data/` (which may use
  `chess` (python-chess) and `zstandard`) and viz (matplotlib/openTSNE).
  No torch, no sklearn, no new dependencies beyond pyproject.toml.
- RNG: always `np.random.default_rng(seed)` passed explicitly. Never seed
  globally, never call `np.random.<fn>` module-level functions.
- Vectorize per-move operations with the `np.<op>.reduceat(x, chain.op0)` /
  `reduceat(x, chain.mp0)` idiom (see `planner/readout.py` for the pattern,
  including the first-argmax tie-break — copy it, don't reinvent it).
- Docstrings state constraints and semantics, not narration.
- Terminal outcome scoring goes through `latentchess.scoring.TerminalScores`
  ONLY. Never write a literal like `1e9` for a mate/draw score in new code.

### 0.3 Key existing APIs you will build on (do not re-implement)

| Thing | Where | Notes |
|---|---|---|
| `TransitionChain` | `latentchess/chain.py` | fields `n, n_live, move_ptr, move_kind, out_ptr, out_flat, terminals, move_names, strata`; derived `mp0, op0, move_counts, out_counts, n_moves, pos_idx`; methods `moves_of(s)`, `outs_of(mid)`. `move_kind`: 0=ongoing 1=mate 2=stalemate 3=white-terminal. |
| `Terminals` | `latentchess/chain.py` | `.mate, .draw, .bwin (or None), .indices` |
| domain builders | `latentchess/domains/{krk,krkn,krrk}.py` | `build_chain()`, `compute_dtm(...)`, `describe_state(chain, s)`. KRk: `compute_dtm(W, B)` takes lists from `enumerate_states()`; KRkn/KRRk: `compute_dtm(chain)`. Chains carry `chain.W` (and `chain.W1`/`chain.W2`) piece-tuple lists. |
| `TerminalScores` | `latentchess/scoring.py` | `.big()`, `.from_reach_quantiles(reach_live)`, `dtm_filled(dtm, n)`, `fill_terminal_state_scores(scores, chain, ts)`, `override_move_values(V, chain, ts)` |
| readout | `latentchess/planner/readout.py` | `ReplyAgg.MEAN/MIN`, `move_values(state_scores, chain, agg, ts)`, `policy_from_values(V, chain)`, `greedy_policy(...)`, `backup(state_scores, chain, agg, ts, k)`. `state_scores` is ALWAYS length `chain.n` with terminal entries pre-filled via `fill_terminal_state_scores`. |
| policies | `latentchess/planner/policy.py` | `Policy` protocol (`move_id(chain, s, rng) -> global mid`), `TablePolicy(local_moves)`, `RandomPolicy`, `EpsGreedy(base, eps)`, `DTMOraclePolicy(chain, dtm)` |
| opponents | `latentchess/opponents.py` | `Opponent` protocol (`reply_index(chain, mid, rng)`), `optimal_reply_table(chain, dtm)` (THE B_opt), `RandomOpponent`, `EpsOptimalDTM(table, eps)`, `TableOpponent(table)` |
| game/arena | `latentchess/game.py`, `latentchess/arena.py` | `play_game(chain, white, black, start, cap, rng) -> GameRecord(start, states, move_ids, result, final_kind)`; `rollout_transitions(...)`; `evaluate(...) -> ArenaResult` |
| embedding | `latentchess/cone/embedding.py` | `QuasimetricEmbedding` protocol (`d`, `reach(idx, goal)`), `GoalSpec(name, region, z)`, `make_goal(name, region, emb)`, `reach(emb, goal, idx)`, `EMBEDDING_METHODS` registry + `@register_embedding(name)` |
| tabular FB | `latentchess/cone/tabular.py` | `TabularFB.fit(P, gamma, d, n_oversample, seed)`, `sm_matvec`, `randomized_svd_sm`, `fb_from_svd`, `rank_error`; has `F_of/B_of/reach` |
| neural FB | `latentchess/cone/neural.py` | `MLP(din, dh, dout, seed)` (`.forward/.backward/.adam`), `NeuralFB(d, dh, seed, tau, din=77)` (`.train_step(Xs, Xg, lr)`, `.embed_F/.embed_B`), `one_hot_state`, `absorbing_vec`, `EncodedNeuralFB.from_encoded(net, X_all)` |
| trainer | `latentchess/train/curriculum.py` | `Round`, `CurriculumConfig`, `CurriculumTrainer`, `curriculum_starts` |
| checkpoints | `latentchess/train/checkpoints.py` | `TrainerState`, `save_ckpt`, `load_ckpt`, `ckpt_exists` (npz-based) |
| concepts | `latentchess/concepts.py` | `ConceptQuantizer` protocol, `KMeansVQ(n_tokens, iters, seed)`, `usage_perplexity(tokens, n_tokens)`, `QUANTIZERS` registry |
| viz | `latentchess/viz/` | `projection.py` (`Projection2D`, `PCAProjection`, `TSNEProjection`, `Normalizer`, `stratified_fit_index`, `FittedMap`, `fit_map`, `PROJECTIONS`), `payload.py` (KRkn viewer builder + `json_default`), `build_html.py`, `plots.py` (`style`, `kde_layer`) |
| paths | `latentchess/io/paths.py` | `data_dir()`, `derived_dir()`, `generated_dir()`, `lichess_dir()`, `shards_dir()`, `save_array(name, arr)`, `load_array(name)` |
| util | `latentchess/util.py` | `auc(pos, neg)` (tie-aware, NaN on empty class), `ridge_r2(X, y, folds, lam, seed)` |

### 0.4 Legacy scripts still in `code/` (reference until ported in Phase 8)

`experiment.py` (rung-1), `exp_search.py` (k-ply sweep), `exp_generalization.py`
(neural holdout), `diagnostics.py`, `atlas.py`, `region_map.py`, `krkn_map.py`,
`krrk_atlas.py`, `tsne_maps.py`, `tsne_cones.py`, `gen_ui_data.py`,
`gen_ui_data_pi.py`, `gen_ui_data_vs_optimal.py`, `gen_krkn_viewer.py`,
plus already-ported `domain/krkn/krrk/learn/neural/core/sklearn_free` and the
already-ported trainers (`exp_policy_iteration.py`, `exp_krkn.py`,
`exp_krkn2.py`, `exp_krrk.py`). Read them for exact behavior; never import.

### 0.5 Out of scope — DO NOT BUILD (later milestones)

- Learned plan selectors (bandit/REINFORCE), PlanMCTS, decomposition
  generators (enabling sets, region-graph pathfinding, geodesic midpoints),
  precondition/technique discovery, region-graph structures. (Phase M1.5.)
- Torch, MPS, full-board FB training, UCI Stockfish *opponent*, 8×8 viewer.
  (Real-data milestone.) NOTE: the eval-backfill UCI *labeler* in 6.6 IS in
  scope; the *opponent* is not.
- FSQ/SAE quantizers; UMAP beyond the existing optional registration.

---

## PHASE 5 — Plan memory + hierarchical-planning structures

New files: `latentchess/planner/move_identity.py`,
`latentchess/planner/plans.py`, `latentchess/planner/selector.py`,
`experiments/plan_memory_demo.py`, tests. One modification to
`latentchess/planner/policy.py` (add `PlanningPolicy`).

### Task 5.1 — MoveIdentity protocol + syntactic baseline

**Create `latentchess/planner/move_identity.py`:**

```python
class MoveIdentity(Protocol):
    name: str
    def key(self, chain: TransitionChain, s: int, mid: int) -> str:
        """Hashable identity of move mid played at live state s."""

class SyntacticIdentity:
    name = "syntactic"
    # key = chain.move_names[mid] with the state's stratum prefixed:
    # f"{stratum}:{chain.move_names[mid]}" where stratum is the name of the
    # chain.strata range containing s (fall back to "live" if none matches).

MOVE_IDENTITIES: dict[str, type] = {"syntactic": SyntacticIdentity}
```

Semantics: `SyntacticIdentity.key` must be pure/deterministic and never
raise for any valid `(s, mid)` with `chain.mp0[s] <= mid < chain.move_ptr[s+1]`.

**Tests (`tests/test_move_identity.py`):**
- `test_syntactic_key_deterministic`: build KRk chain, for 20 random (s, mid)
  pairs the key is a str, equal across two calls.
- `test_syntactic_key_stratum_prefix`: on KRkn chain (mark `@pytest.mark.slow`
  NOT needed — building KRkn takes ~15 s, acceptable in fast suite? NO:
  keep fast suite fast → use KRk only in fast tests; write one
  `@pytest.mark.slow` KRkn test asserting keys from the KRk stratum start
  with `"KRk:"` and from the KRkn stratum with `"KRkn:"`).
- `test_registry_contains_syntactic`.

**Commit:** `Add MoveIdentity protocol with syntactic baseline`

### Task 5.2 — Plan data model (dataclasses, JSON round-trip)

**Create `latentchess/planner/plans.py`** (this file grows over tasks
5.2–5.6; add pieces in order).

```python
class PlanStatus(Enum): ACTIVE="active"; ACHIEVED="achieved"; ABANDONED="abandoned"; INFEASIBLE="infeasible"
class PlanEvent(Enum):  PROGRESS="progress"; STALLED="stalled"; REPLAN="replan"; ACHIEVED="achieved"; LOST="lost"

@dataclass
class BlockReason:
    rule: str                      # "no_midpoint" | "unlikely_territory" | "dry_out" | "budget"
    bottleneck: tuple[int, int] | None   # (from_state_or_-1, hop_index) or None
    feasibility: float             # reach value at block time
    refutation_key: str | None     # MoveIdentity key of the refuting reply, if applicable
    delta: np.ndarray | None       # enabling direction Δ in F-space (d,) or None
    blocked_at_state: int          # live state index when blocked
    # to_json/from_json: delta serialized as list or None

@dataclass
class PlanStep:
    state: int
    move_id: int | None
    reach: float
    ply: int

@dataclass
class Plan:
    plan_id: str                   # e.g. f"{goal.name}-{origin_state}-{counter}"
    goal: GoalSpec
    origin_state: int
    feasibility0: float
    status: PlanStatus = PlanStatus.ACTIVE
    trace: list[PlanStep] = field(default_factory=list)
    block: BlockReason | None = None
    def to_json(self) -> dict      # goal serialized as {name, region: list} (z NOT serialized here)
    @classmethod
    def from_json(cls, d: dict, z_store: dict[str, np.ndarray]) -> "Plan"
    # z_store maps goal.name -> z vector; from_json rebuilds GoalSpec with z=z_store.get(name)
```

Rules:
- `to_json` must produce something `json.dumps(..., default=json_default)`
  can serialize (`json_default` from `latentchess.viz.payload`).
- `PlanNode` subgoal trees: add a minimal recursive structure now
  (decomposition itself is M1.5, but the tree type must exist):

```python
@dataclass
class PlanNode:
    goal: GoalSpec
    children: list["PlanNode"] = field(default_factory=list)   # ordered subgoal chain
    executable: bool | None = None      # None = not yet checked
    feasibility: float | None = None
```

**Tests (`tests/test_plan_memory.py`, part 1):**
- `test_plan_json_roundtrip`: build a Plan with 2 PlanSteps and a
  BlockReason with a small delta array; `to_json` → `json.dumps` →
  `json.loads` → `from_json` with a z_store; assert field equality
  (np.allclose for delta/z).

**Commit:** `Add plan data model: Plan/PlanStep/PlanNode/BlockReason with JSON round-trip`

### Task 5.3 — PlanMemory core (availability, propose, update)

Extend `latentchess/planner/plans.py`:

```python
def calibrate_tau(reach_live: np.ndarray, won_mask: np.ndarray) -> float:
    """Youden-J threshold separating won from not-won by reach score:
    scan 101 candidate thresholds = quantiles of reach_live at
    np.linspace(0,1,101); J(t) = TPR(t) - FPR(t) where positive class =
    won_mask; return the t maximizing J. If won_mask is all-True or
    all-False, return -inf (everything available)."""

class PlanMemory:
    def __init__(self, emb, goals: list[GoalSpec], tau: float,
                 drop_delta: float = 0.5, stall_plies: int = 6,
                 max_plans: int = 32):
    def available(self, s: int) -> list[tuple[GoalSpec, float, bool]]:
        """[(goal, reach_value, reach_value >= self.tau)] for every goal,
        sorted by reach_value descending. reach_value = float(emb.reach(np.array([s]), goal)[0])."""
    def propose(self, s: int) -> Plan:
        """Highest-reach goal with available=True -> new ACTIVE Plan
        (feasibility0 = its reach). If none available: create the Plan for
        the highest-reach goal anyway but with status=INFEASIBLE and
        block=BlockReason(rule="unlikely_territory", feasibility=<reach>,
        blocked_at_state=s, bottleneck=None, refutation_key=None,
        delta=<z-direction of that goal if emb exposes F-space, else None>).
        Newly created plans are stored in self.plans (evict lowest-
        feasibility non-ACTIVE plan if len > max_plans)."""
    def update(self, plan: Plan, s: int, move_id: int | None) -> PlanEvent:
        """Append PlanStep(state=s, move_id, reach=r_now, ply=len(trace)).
        Event logic, in priority order:
          1. if r_now >= running max of trace reaches so far -> PROGRESS
          2. if r_now < (1 - drop_delta) * running_max OR r_now < tau -> REPLAN
             (set plan.status = ABANDONED, record block=BlockReason(
              rule="unlikely_territory", feasibility=r_now, ...))
          3. if the last `stall_plies` steps' reaches are all within 1e-9
             of each other -> STALLED
          4. else PROGRESS
        Terminal handling is done by the caller (PlanningPolicy marks
        ACHIEVED/LOST); update() itself never sees terminal states."""
    active: Plan | None      # the plan currently being pursued (set by PlanningPolicy)
```

Notes: running_max computed over `[st.reach for st in plan.trace] + [r_now]`
EXCLUDING r_now for the comparison baseline (i.e. max of previous steps;
first call is always PROGRESS).

**Tests (`tests/test_plan_memory.py`, part 2)** — use a stub embedding, NOT
a real one:

```python
class StubEmb:      # scores keyed by (state, goal.name)
    d = 4
    def __init__(self, table): self.table = table
    def reach(self, idx, goal):
        return np.array([self.table[(int(i), goal.name)] for i in np.atleast_1d(idx)])
```

- `test_available_sorts_and_thresholds`
- `test_propose_picks_highest_available` and
  `test_propose_infeasible_when_none_available` (status INFEASIBLE, block set)
- `test_update_progress_then_replan_on_drop` (feed reaches 0.5, 0.6, 0.2
  with drop_delta=0.5 → third call returns REPLAN and plan ABANDONED)
- `test_update_stalled` (constant reach for stall_plies+1 steps → STALLED)
- `test_calibrate_tau_separates` (reach = won*1.0 + noise*0.01 → tau
  between the classes; all-won → -inf)

**Commit:** `Add PlanMemory core: availability, propose, update, tau calibration`

### Task 5.4 — Executability by rollout

Extend `plans.py`:

```python
def rollout_reaches(chain, policy, start: int, target: np.ndarray,
                    black, horizon: int = 12, n_rollouts: int = 20,
                    rng: np.random.Generator | None = None) -> float:
    """Fraction of n_rollouts games (play_game-style loop, cap=horizon white
    moves, using `policy` vs `black`) whose visited states enter `target`
    (a bool mask of length chain.n_live OR an int array of live-state
    indices — accept both, normalize to a bool mask). Entering the target
    counts at any visited state INCLUDING the start. Terminal endings
    without touching target count as failure. Reuse latentchess.game.play_game."""

def hop_executable(chain, policy, start, target, black,
                   horizon=12, n_rollouts=20, p_min=0.5, rng=None) -> tuple[bool, float]:
    frac = rollout_reaches(...); return frac >= p_min, frac
```

**Tests:** on KRk with the DTM oracle policy and optimal black:
- `test_rollout_reaches_mate_region`: target = states with dtm <= 3;
  start = a state with dtm == 9; expect fraction == 1.0 (oracle play is
  deterministic and must pass through dtm<=3 on the way to mate).
- `test_rollout_reaches_zero_for_unreachable`: target = a single random
  state far from the oracle line (pick target = {some state with dtm == 19}
  and start with dtm == 5 — oracle white never increases dtm, so fraction
  == 0.0).

**Commit:** `Add executability-by-rollout check (rollout_reaches/hop_executable)`

### Task 5.5 — Wake system (BlockReason listeners)

Extend `plans.py`:

```python
@dataclass
class WatchSpec:
    kind: str                      # "drift" | "event"
    plan_id: str
    key: str | None = None         # event key (MoveIdentity string), for kind="event"
    threshold: float = 0.15        # drift threshold on normalized dot product
    cooldown: int = 0              # plies remaining before this watch may fire again
    cooldown_base: int = 4         # doubles on each wake-and-fail (cap 32)

class PlanMemory:   # additions
    def register_block(self, plan: Plan, reason: BlockReason,
                       event_keys: list[str] = ()) -> None:
        """Set plan.status=INFEASIBLE, plan.block=reason; register a drift
        WatchSpec (if reason.delta is not None) and one event WatchSpec per
        key in event_keys. Store F(s_blocked) via emb.F_of if the embedding
        has F_of, else skip drift watch."""
    def on_ply(self, s_now: int, event_keys: set[str]) -> list[Plan]:
        """Called once per ply by PlanningPolicy. Decrement cooldowns.
        Fire checks:
          - event watches whose key ∈ event_keys and cooldown == 0
          - drift watches with cooldown == 0 where
            dot(F(s_now) - F(s_blocked), delta) / (|delta|^2 + 1e-12) >= threshold
        For each fired watch: re-check the plan's goal reach at s_now.
          If reach >= tau_wake (tau_wake = self.tau; hysteresis: plans were
          blocked when < tau, so requiring >= tau IS the hysteresis as long
          as blocking used < tau - see register semantics) -> plan.status =
          ACTIVE (woken), clear block, return it in the woken list.
          Else: wake-and-fail -> cooldown = min(cooldown_base * 2, 32),
          cooldown_base doubles (cap 32).
        Watches of ACHIEVED/ABANDONED plans are dropped lazily."""
```

Simplification that is REQUIRED (keep it this simple): τ_wake = τ used with
a strict `>=` while blocking uses `<`; the cooldown doubling is the
anti-thrash mechanism. Do not implement a separate τ_sleep.

**Tests:**
- `test_event_wake_on_matching_key_only`: block a plan with
  `event_keys=["KRk:Rxc3"]`; `on_ply(s, {"KRk:Ka2"})` wakes nothing;
  `on_ply(s, {"KRk:Rxc3"})` with stub reach above tau wakes exactly that
  plan (status ACTIVE, block None).
- `test_drift_wake`: block with delta=d0; stub embedding F table such that
  F(s_now)-F(s_blocked) has dot/|d0|² = 0.5 ≥ threshold and reach above
  tau → woken. With dot 0.01 → not woken.
- `test_wake_and_fail_cooldown`: reach stays below tau; fire the event key
  twice in consecutive plies → second fire must NOT re-check (cooldown);
  advance plies past cooldown → checks again.

**Commit:** `Add two-tier wake system: BlockReason registration, event index, drift watcher, cooldown`

### Task 5.6 — PlanStore persistence

Extend `plans.py`:

```python
class PlanStore:
    def __init__(self, dir: Path)        # creates dir
    def append(self, plan: Plan) -> None
        # appends plan.to_json() as one line to plans.jsonl;
        # upserts goal z into zs.npz (np.savez of {goal.name: z})
    def load_all(self) -> list[Plan]     # rebuilds z_store from zs.npz
```

**Test:** `test_plan_store_roundtrip` (tmp_path; two plans, one blocked
with delta; load back; field equality).

**Commit:** `Add PlanStore JSONL+npz persistence`

### Task 5.7 — PlanSelector protocol + GreedyReach baseline

**Create `latentchess/planner/selector.py`:**

```python
class PlanSelector(Protocol):
    name: str
    def select(self, s: int, memory: PlanMemory) -> Plan:
        """Return the plan to pursue at state s (may call memory.propose)."""

class GreedyReach:
    name = "greedy_reach"
    def select(self, s, memory):
        # if memory.active is an ACTIVE plan -> keep it;
        # else memory.propose(s) and set memory.active to the result.

PLAN_SELECTORS = {"greedy_reach": GreedyReach}
```

**Test:** `test_greedy_reach_keeps_active_plan`, `test_greedy_reach_proposes_when_none`.

**Commit:** `Add PlanSelector protocol with GreedyReach baseline`

### Task 5.8 — PlanningPolicy + parity gate + demo

**Modify `latentchess/planner/policy.py`** — add:

```python
class PlanningPolicy:
    """Policy that consults PlanMemory each move. With goals=[MATE-only],
    tau=-inf and no decomposition this MUST be move-for-move identical to
    greedy_policy(reach, MIN) — the parity anchor proving plan machinery
    adds no behavior until asked."""
    def __init__(self, chain, emb, memory: PlanMemory, ts: TerminalScores,
                 selector: PlanSelector | None = None,
                 agg: ReplyAgg = ReplyAgg.MIN, depth: int = 1,
                 identity: MoveIdentity | None = None):
        # selector default GreedyReach(); identity default SyntacticIdentity()
        # Precompute NOTHING per-state at init beyond storing args.
    def move_id(self, chain, s, rng) -> int:
        # 1. plan = self.selector.select(s, self.memory)
        # 2. scores = fill_terminal_state_scores(
        #        <length-n vector where [:n_live] = emb.reach(None, plan.goal)
        #         and terminal entries filled>, chain, self.ts)
        #    CACHE this scores vector per (plan.plan_id) — recompute only
        #    when the active plan changes (a dict self._scores_cache).
        # 3. if self.depth > 1: scores = backup(scores, chain, self.agg,
        #        self.ts, self.depth - 1)   (cache alongside)
        # 4. V = move_values(scores, chain, self.agg, self.ts)
        #    local = argmax over this state's segment with the SAME
        #    first-argmax tie-break as policy_from_values (extract the
        #    segment V[chain.mp0[s]:chain.move_ptr[s+1]] and use
        #    int(np.argmax(seg)) — np.argmax already returns first max).
        # 5. mid = chain.mp0[s] + local
        # 6. self.memory.update(plan, s, mid); emit events:
        #    self.memory.on_ply(s, {self.identity.key(chain, s, mid)})
        # 7. return mid
```

**Tests (`tests/test_planning_policy.py`):**
- `test_parity_with_min_readout_krk` (FAST, KRk): build chain, exact_P,
  `TabularFB.fit(P, 0.92, d=32)`, goal = MATE only
  (`make_goal("mate", np.array([chain.terminals.mate]), emb)`),
  memory = PlanMemory(emb, [goal], tau=-np.inf). Reference policy: local
  table `greedy_policy(fill_terminal_state_scores(reach(emb, goal), chain,
  ts), chain, ReplyAgg.MIN, ts)`. Play 50 seeded games (starts from
  `default_rng(3).integers(0, chain.n_live, 50)`, optimal black,
  cap 60) with both policies and assert **identical move_id sequences**
  per game.
- `test_parity_with_min_readout_krkn` — same on KRkn, `@pytest.mark.slow`.

**Create `experiments/plan_memory_demo.py`:** KRkn; two goals:
`mate_direct` (region=[MATE_S]) and `via_krk` (region = KRk-stratum states
with dtm <= 9, i.e. `n2 + np.where(dtm[n2:] <= 9)[0]`); tau from
`calibrate_tau(scores[:n2-ish]...)` — concretely: emb = TabularFB from the
saved training checkpoint if `data/derived/krkn_F.npy` exists, else train a
2-round quick curriculum first (reuse `experiments/train_krkn.py` pieces);
play one game from a deep-DTM start with PlanningPolicy; print per-ply
`available()` output; assert (in-script, not pytest) that `via_krk`
availability flips when the game crosses into the KRk stratum; persist all
plans to `PlanStore(data/derived/plan_demo)` and reload them, printing
counts. Keep it under ~120 lines.

**Verify phase:** `pytest -m "not slow"` green; `pytest -m slow` green;
`python experiments/plan_memory_demo.py` runs and shows the flip.

**Commit:** `Add PlanningPolicy with plan-memory parity anchor + KRkn availability demo`
Then: `git push`.

---

## PHASE 6 — Data layer: toy sources, Lichess streaming, shards, eval backfill, neural port

New deps already in pyproject: `python-chess` (import name `chess`),
`zstandard`. New files under `latentchess/data/` (+ `__init__.py`).

### Task 6.1 — PairSource protocol + ChainRolloutSource (toy)

**Create `latentchess/data/__init__.py`** (empty) and
**`latentchess/data/sources.py`:**

```python
@dataclass
class PairBatch:
    anchors: np.ndarray            # toy: int32 live-state indices (B,); lichess: uint64 packed boards (B,12)
    goals: np.ndarray              # same encoding family as anchors; may include absorbing codes (toy)
    meta: dict[str, np.ndarray] = field(default_factory=dict)

class PairSource(Protocol):
    def batches(self, batch_size: int, seed: int) -> Iterator[PairBatch]:
        """Bounded memory; fully deterministic given seed."""

class ChainRolloutSource:
    """Geometric-horizon (s, g) future pairs from uniform-random rollouts.
    Ports code/neural.py::sample_episodes + build_pairs semantics exactly:
      - episodes: starts uniform over live states; per step pick a uniform
        move, then a uniform outcome of that move; episode ends on absorbing.
        (Use uniform RandomPolicy + RandomOpponent via game.play_game? NO —
        the original picks a uniform OUTCOME of the move, which equals
        RandomOpponent. Use play_game with RandomPolicy/RandomOpponent, but
        the episode must record the FULL state sequence INCLUDING the final
        absorbing index: reconstruct it as rec.states + [terminal index
        implied by rec.result] — mate->terminals.mate, draw->terminals.draw,
        bwin->terminals.bwin; 'cap' episodes have no absorbing tail.)
      - pairs: for each i < len(ep)-1 with ep[i] live and not held out:
        k = 1 + rng.geometric(1 - gamma); j = min(i + k, len(ep) - 1);
        g = ep[j]; skip if g is live and held out. Pair = (ep[i], g).
    __init__(self, chain, gamma, n_games, max_plies=200,
             holdout_mask: np.ndarray | None = None)
    batches(batch_size, seed): generate all episodes with
      default_rng(seed), build the full pair list, then yield successive
      slices of batch_size as PairBatch (anchors int32, goals int32,
      meta={}). Memory note: the toy pair list fits easily; boundedness
      matters for the Lichess source, not here.
```

**Tests (`tests/test_data.py`):**
- `test_chain_rollout_deterministic`: same seed → identical first batch;
  different seed → different.
- `test_holdout_excluded_both_roles`: holdout 20% of KRk states; iterate
  all batches; assert no anchor and no LIVE goal is in the holdout set.
- `test_geometric_horizon_matches_reference`: reimplement the pairing rule
  inline on 3 fixed episodes `[[0,5,9,DRAW],[2,MATE],[1,3,4,6,8]]` with a
  seeded rng and assert ChainRolloutSource's internal pairing helper (expose
  as module function `pairs_from_episodes(episodes, gamma, rng, holdout,
  n_live)`) reproduces it exactly.

**Commit:** `Add PairSource protocol and ChainRolloutSource with reference-parity pairing`

### Task 6.2 — Board encoding (packed bitboards)

**Create `latentchess/data/encode.py`** (this is 8×8, python-chess):

```python
PIECE_PLANES = [(chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE), (chess.ROOK, chess.WHITE),
    (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK), ... same order BLACK]      # 12 planes

def encode_board(board: chess.Board) -> np.ndarray:
    """(12,) uint64 — plane i = board.pieces_mask(*PIECE_PLANES[i])."""

def encode_meta(board) -> np.ndarray:
    """(3,) uint8: [stm (0=white to move), castling bits (WK=1,WQ=2,BK=4,BQ=8),
    ep file 0-7 or 15]."""

def decode_batch(packed: np.ndarray) -> np.ndarray:
    """(B,12) uint64 -> (B,12,8,8) uint8 via ONE
    np.unpackbits(packed.view(np.uint8), bitorder="little") call + reshape.
    Square a1 = bit 0 = [row 0, col 0] of the plane."""

def board_from_planes(planes: np.ndarray) -> chess.Board:
    """Inverse of encode for TESTING ONLY (piece placement only, meta not
    restored). Build via chess.Board(None) + set_piece_at."""
```

**Tests:**
- `test_encode_decode_roundtrip_startpos` and
  `test_encode_decode_roundtrip_random`: for `chess.Board()` and 20 boards
  reached by random legal moves (seeded), `board_from_planes(decode_batch(
  encode_board(b)[None])[0])` has identical `piece_map()`.
- `test_decode_batch_vectorized`: decoding 50 boards at once equals
  one-at-a-time decoding.
- `test_meta_fields`: startpos → stm=0, castling=15, ep=15; after 1. e4 →
  stm=1, ep file = 4.

**Commit:** `Add packed-bitboard board encoding with vectorized batch decode`

### Task 6.3 — Shard writer/reader

**Create `latentchess/data/shards.py`:**

```python
def write_shards(pair_iter, out_dir: Path, shard_size: int = 250_000,
                 prefix: str = "shard") -> list[Path]:
    """pair_iter yields dicts of column-name -> 1D/2D np arrays of equal
    leading length (a 'block'). Accumulate blocks; every time the buffer
    reaches shard_size rows, np.savez_compressed(out_dir/f"{prefix}-{i:05d}.npz",
    **columns) and reset. Flush the remainder at the end. Also write a
    manifest.json: {"shards": [names], "rows": [counts], "columns": [names]}."""

class ShardReader:
    """PairSource over shards. batches(batch_size, seed):
    - shuffle shard ORDER with default_rng(seed)
    - maintain an in-memory shuffle buffer of `buffer_rows` (default 50_000)
      rows: fill from consecutive shards, yield uniformly-sampled batches
      (sample WITHOUT replacement from the buffer, refill as it drains —
      standard reservoir-ish shuffle; exact algorithm: keep a list; when
      emitting a batch, draw batch_size random indices, swap-remove them).
    - columns 'anchors' and 'goals' map to PairBatch fields; every other
      column goes into meta.
    One full pass over all shards = one epoch; batches() is a single-epoch
    iterator. Deterministic given seed."""
    def __init__(self, dir: Path, buffer_rows: int = 50_000)
```

**Tests:**
- `test_shard_roundtrip`: write 3 blocks totalling 1000 rows with
  shard_size 400 (→ 3 shards: 400/400/200); ShardReader with a huge buffer
  yields exactly the original multiset of rows (sort and compare).
- `test_shard_reader_deterministic_per_seed`.
- `test_reader_bounded_buffer`: buffer_rows=100, batch 32 — runs to
  completion and total rows match (boundedness by construction; assert the
  internal buffer length never exceeds buffer_rows + block remainder —
  expose a `_max_buffer_seen` counter for the test).

**Commit:** `Add npz shard writer and shuffling ShardReader PairSource`

### Task 6.4 — Committed Lichess PGN fixture

**Create `tests/fixtures/make_fixture.py`** (a dev script, committed) that
writes `tests/fixtures/lichess_fixture.pgn.zst` using `zstandard` from an
inline PGN string containing SIX tiny games (each ~10-20 plies, LEGAL move
sequences — write them by hand from real miniatures, e.g. Scholar's mate
variants; verify legality by replaying with chess.pgn while generating):

1. game A: `[WhiteElo "1500"] [BlackElo "1520"] [TimeControl "300+0"]`,
   `%eval` + `%clk` comments on every move, Result 1-0.
2. game B: same Elo band/TC, `%clk` only (NO evals — backfill target),
   Result 0-1.
3. game C: bullet `[TimeControl "60+0"]` (must be filtered out).
4. game D: `[WhiteElo "900"]` out-of-band (filtered out).
5. game E: `[WhiteTitle "BOT"]` (filtered out).
6. game F: in-band, `%clk` with some moves below 30s (those anchors must be
   dropped), one `%eval #-3` mate annotation, Result 1/2-1/2.

Run it once: `python tests/fixtures/make_fixture.py` → commit BOTH the
script and the ~2-5 KB `.pgn.zst`. (Exception to the no-data rule: test
fixtures are code.) Add `!tests/fixtures/*.pgn.zst` to `.gitignore` if the
`*.pkl`-style patterns would exclude it (check: current .gitignore has no
`*.zst` rule — nothing to do, but VERIFY `git status` shows the file).

**Commit:** `Add committed Lichess PGN fixture (6 games covering eval/clock/filter edge cases)`

### Task 6.5 — Lichess streaming source

**Create `latentchess/data/lichess.py`:**

```python
def open_pgn_stream(path: Path) -> io.TextIOWrapper:
    """zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader over
    the raw file handle, wrapped in TextIOWrapper(encoding='utf-8').
    NEVER reads the whole file; NEVER writes a decompressed copy."""

@dataclass
class GameFilter:
    min_elo: int = 1400
    max_elo: int = 1800
    min_base_seconds: int = 180          # TimeControl base; excludes bullet
    min_plies: int = 10
    skip_first_plies: int = 10           # Maia rule
    min_clock_seconds: int = 30          # drop anchors with less time (Maia rule)
    def headers_pass(self, headers) -> bool:
        """Both Elos parse as ints and lie in [min_elo, max_elo]; TimeControl
        parses 'B+I' with B >= min_base_seconds; Result in {1-0,0-1,1/2-1/2};
        neither WhiteTitle nor BlackTitle == 'BOT'."""

def stream_filtered_games(path, gf: GameFilter, max_games: int | None = None):
    """Iterator of chess.pgn.Game. Loop: offset = stream position is not
    seekable — use chess.pgn.read_headers to prefilter? IMPORTANT
    IMPLEMENTATION CONSTRAINT: chess.pgn.read_headers consumes the movetext
    of the game it reads headers for ONLY when followed by another
    read_headers; you cannot rewind a zstd stream. Therefore: read headers
    with chess.pgn.read_headers(stream); if they fail the filter, continue
    (read_headers already skipped to the next game); if they pass, you
    CANNOT re-read the moves. SOLUTION (do it this way): use
    chess.pgn.read_game(stream) for every game and filter on
    game.headers — simpler and correct; the fixture and --max-games guard
    keep runtime bounded. Yield games whose headers pass; stop after
    max_games yielded (not read)."""

def positions_with_labels(game, gf) -> list[tuple[chess.Board copy info...]]:
    """Walk game.mainline(); for ply index p (0-based, counting from the
    starting position BEFORE the move), collect a record AFTER pushing each
    move? NO — anchor = position BEFORE a move, at plies p >= skip_first_plies.
    For each anchor: parse the node's clock via node.clock() (python-chess
    returns seconds or None — the %clk of the move ABOUT to be played is on
    the NEXT node; concretely for node in game.mainline(): node.board() is
    the position AFTER node.move; use parent boards. SIMPLEST CORRECT LOOP:
      board = game.board()
      for ply, node in enumerate(game.mainline()):
          # board is the position BEFORE node.move
          record(board, eval=node.eval(), clock=node.clock(), ply=ply)
          board.push(node.move)
    node.eval() -> chess.engine.PovScore | None (the %eval comment on this
    move's node = evaluation AFTER the move per Lichess convention; ACCEPT
    this off-by-one: label the pre-move board with node.eval() — consistent
    across the whole dataset, which is what matters).
    Skip records with ply < skip_first_plies or (clock is not None and
    clock < min_clock_seconds).
    Returns list of (board_copy_or_encoded, eval_cp_or_None, is_mate_score,
    clock, ply)."""

class LichessPairSource:
    """PairSource over one .pgn.zst. __init__(path, gf, gamma,
    max_games=None, backfiller=None).
    batches(batch_size, seed): stream games; per game build the anchor
    list via positions_with_labels; geometric-horizon pairing over anchor
    indices within the game (k = 1 + rng.geometric(1-gamma), j = min(i+k,
    last)); encode anchor and goal boards with encode.encode_board; meta
    columns: eval (float32, tanh-squashed win-ish scale — see 6.6 —
    np.nan when absent and no backfiller), eval_is_real (uint8),
    white_elo, black_elo (uint16), result (int8: +1/0/-1 white POV),
    clock (float32 seconds, np.nan if None), stm (uint8).
    If backfiller is not None, missing evals are filled via
    backfiller.eval_board(board) at pair-emission time.
    Accumulate into blocks of batch_size and yield PairBatch. Bounded
    memory: never hold more than one game's records + one block."""
```

**Tests (fixture-driven, all fast):**
- `test_stream_never_materializes`: open fixture via `open_pgn_stream`,
  read 1 game, assert the returned object is a TextIOWrapper and no
  `.pgn` file appeared next to the fixture.
- `test_filter_correctness`: stream_filtered_games over the fixture with
  the default GameFilter(min_elo=1400, max_elo=1800, min_base_seconds=180)
  yields EXACTLY games A, B, F (identify by Result+White header).
- `test_anchor_rules`: for game F, no anchor has ply < 10; no anchor has
  clock < 30 where clock is present.
- `test_pair_source_batches`: LichessPairSource on the fixture,
  batch_size=8: yields ≥ 1 batch; anchors dtype uint64 shape (B,12);
  meta['eval'] has NaN exactly where eval_is_real == 0 (no backfiller);
  deterministic given seed.

**Commit:** `Add streaming Lichess source: zstd stream, GameFilter, anchor rules, LichessPairSource`

### Task 6.6 — Eval backfill (UCI Stockfish labeler, injectable engine)

**Create `latentchess/data/eval_backfill.py`:**

```python
CP_SCALE = 400.0
def squash_cp(cp: float) -> float: return float(np.tanh(cp / CP_SCALE))
def squash_mate(mate_in: int) -> float: return 1.0 if mate_in > 0 else -1.0
def squash_pov_score(score: "chess.engine.PovScore") -> float:
    """White-POV: score.white().score(mate_score=None); if it's a mate,
    use squash_mate(score.white().mate()); else squash_cp(cp)."""

class UCIEvaluator:
    """Real engine wrapper. __init__(engine_path='stockfish', depth=14).
    Uses chess.engine.SimpleEngine.popen_uci (python-chess handles the
    protocol; do NOT hand-roll pipes). eval_board(board) -> float in [-1,1]
    via engine.analyse(board, chess.engine.Limit(depth=self.depth))
    ['score'] -> squash_pov_score. close() quits the engine.
    Wrap analyse in try/except returning np.nan on engine errors."""

class Backfiller:
    """Cache + never-overwrite policy. __init__(evaluator, cache: dict |
    None = None). eval_board(board): key = board.fen()... ACTUALLY use
    key = (board.board_fen(), board.turn) to ignore clocks; if key in
    cache return it; else call evaluator.eval_board, store, return.
    NOTE: the never-overwrite rule lives in LichessPairSource — it only
    calls the backfiller when the PGN eval is absent. Add an assertion
    helper `assert_not_overwriting(has_real_eval)` that raises if called
    with has_real_eval=True (LichessPairSource passes this flag)."""
```

Wire into `LichessPairSource` (already parameterized in 6.5): when a
record's eval is None and backfiller is set → `backfiller.eval_board(board)`
and `eval_is_real=0`.

**Tests (NO real Stockfish — a fake evaluator):**

```python
class FakeEvaluator:
    def __init__(self): self.calls = []
    def eval_board(self, board): self.calls.append(board.fen()); return 0.25
```

- `test_squash_bounds_and_monotone`: squash_cp monotone over
  [-2000..2000], within (-1,1); squash_mate(+3)=1.0, squash_mate(-2)=-1.0.
- `test_backfiller_cache`: two calls with the same position → evaluator
  called once.
- `test_backfill_only_missing`: LichessPairSource on fixture with
  Backfiller(FakeEvaluator()): after one full epoch, every FEN in
  `fake.calls` corresponds to a record WITHOUT a PGN eval (game B / F's
  unannotated moves), and meta['eval'] has no NaNs; rows with
  eval_is_real==1 keep their PGN-derived values (spot-check one known
  value from game A's first annotated move).

**Commit:** `Add eval backfill: cp/mate squash, UCI evaluator wrapper, cached never-overwrite Backfiller`

### Task 6.7 — NeuralFB eval head (auxiliary loss + frozen probe)

**Modify `latentchess/cone/neural.py`:**

1. `NeuralFB.__init__` gains `eval_dh: int = 64`; construct
   `self.E = MLP(d, eval_dh, 1, seed + 2)` — input is the F-EMBEDDING
   (dimension d), not the board encoding.
2. New method:

```python
def train_step_with_eval(self, Xs, Xg, lr, eval_targets=None, eval_weight=0.5):
    """InfoNCE step exactly as train_step, PLUS if eval_targets is not None:
    mask = ~np.isnan(eval_targets); if mask.any():
      f = <the F.forward(Xs) activations ALREADY computed for InfoNCE — reuse>
      pred = self.E.forward(f[mask])            # (m,1)
      err = (pred[:,0] - eval_targets[mask])    # MSE
      loss_eval = (err**2).mean()
      dpred = (2*err/len(err))[:,None] * eval_weight
      dF_extra = self.E.backward(dpred) ... CAREFUL: joint mode requires
      adding the eval gradient w.r.t. f into the InfoNCE df BEFORE calling
      self.F.adam. Restructure: compute df_infonce and db as in train_step;
      df_total = df_infonce.copy(); df_total[mask] += self.E.backward(dpred)
      is WRONG because E.backward returns grad w.r.t. its INPUT rows (m,d) —
      scatter-add those rows into df_total at mask positions. Then
      self.E.adam(<E's own param grads — capture them: MLP.backward returns
      param grads dict, and grad-w.r.t-input must be derived separately.
      MODIFY MLP: add method `backward_with_input_grad(dout) ->
      (param_grads, dinput)` where dinput = dz1 @ W1.T computed at the end
      of the existing backward chain. Use it here.>, lr)
      self.F.adam(self.F.backward(df_total), lr); self.B.adam(...)
    Returns (loss_infonce, loss_eval or nan)."""
```

   Keep the original `train_step` untouched (used elsewhere).
3. New module-level function:

```python
def fit_eval_probe(F: np.ndarray, targets: np.ndarray, lam: float = 1.0)
    -> tuple[np.ndarray, float]:
    """FROZEN probe: closed-form ridge weights w (d+1 incl. bias) on rows
    where targets is finite; returns (w, in-sample R^2). Predict helper:
    eval_probe_predict(F, w) -> np.ndarray."""
```

4. `EncodedNeuralFB` gains `evaluate(self, idx=None) -> np.ndarray` using
   `self.net.E.forward(self.F_of(idx))[:, 0]` (returns squashed scale).

**Tests (`tests/test_eval_head.py`):**
- `test_mlp_input_grad_matches_numeric`: finite-difference check of
  `backward_with_input_grad`'s dinput on a tiny MLP (3→4→4→1, 5 samples,
  tolerance 1e-3 relative).
- `test_joint_eval_loss_decreases`: synthetic task — X = one-hot of 40
  states (din=77 padding ok), targets = a fixed random linear function of a
  hidden embedding; run 300 train_step_with_eval iterations; final
  loss_eval < first loss_eval * 0.5.
- `test_frozen_probe_r2`: F = random (500, 32), targets = F @ w_true +
  0.01 noise → probe R² > 0.95; NaN rows are ignored.
- `test_evaluate_shape`: EncodedNeuralFB.evaluate(None) shape (n_states,).

**Commit:** `Add eval head to NeuralFB: joint auxiliary MSE + frozen ridge probe + evaluate()`

### Task 6.8 — Shard-building CLI with resource guards

**Create `experiments/build_lichess_shards.py`:**

```
args: --pgn PATH (required; a .pgn.zst), --out DIR (default
      latentchess.io.paths.shards_dir()/<pgn stem>),
      --min-elo/--max-elo/--min-base-seconds (GameFilter fields),
      --gamma (default 0.98), --max-games (default 5000),
      --max-gb (default 2.0; abort writing when total shard bytes exceed),
      --shard-size (default 250_000),
      --backfill-engine PATH (optional; when set, wrap UCIEvaluator in
      Backfiller; when unset, missing evals stay NaN),
      --depth (default 14).
Behavior: LichessPairSource -> adapt its batches into write_shards blocks
(columns: anchors, goals, eval, eval_is_real, white_elo, black_elo, result,
clock, stm). After each shard flush, check cumulative bytes vs --max-gb and
stop cleanly with a message. Print a final manifest summary. MUST be
runnable against the test fixture:
  python experiments/build_lichess_shards.py --pgn tests/fixtures/lichess_fixture.pgn.zst --max-games 10
```

**Test:** `test_build_shards_cli_on_fixture` — invoke `main([...])` (design
main to accept argv) writing into tmp_path; assert manifest.json exists,
ShardReader round-trips ≥ 1 batch, and no file outside tmp_path was created.

**Commit:** `Add build_lichess_shards CLI with max-games/max-gb guards, fixture-tested`

### Task 6.9 — Port the generalization experiment (neural holdout)

**Create `experiments/generalization.py`** reproducing
`code/exp_generalization.py` on the new stack: KRk chain; 15% holdout
(`default_rng(0).random(n_live) < 0.15`); episodes/pairs via
`ChainRolloutSource(chain, gamma=0.92, n_games=32000)` with holdout;
tabular baseline via counts on the same filtered transitions (reuse
`chain.empirical_P`); neural: `NeuralFB(d=32, dh=256, seed=0, tau=0.1)`,
12000 steps, batch 256, lr schedule [(0,1e-3),(8000,3e-4)] — mirror the
original loop but consume ChainRolloutSource pairs; evaluations E1
(spearman at holdout vs train against exact reach from
`sm_matvec(exact_P(chain), e_region, gamma)`), E2/E3 engines from holdout
starts (reuse arena.evaluate with a TablePolicy built from each score
vector; opponents: EpsOptimalDTM(0) and RandomOpponent). Print the same
style of table; save `Fn_neural`, `holdout_mask`, `reach_neural` via
`io.paths.save_array`.

**Gate (run manually, ~3-5 min):** holdout spearman ≥ 0.40 and
|train − holdout| ≤ 0.10 (the documented zero-gap result was 0.46/0.41).
Also add `test_generalization_smoke` (`@pytest.mark.slow`): same script
with n_games=4000, steps=2000 → holdout spearman > 0.2 (weak but nonzero).

**Verify phase 6:** fast suite green; slow suite green;
`python experiments/generalization.py` hits the gate;
`python experiments/build_lichess_shards.py --pgn tests/fixtures/lichess_fixture.pgn.zst --max-games 10` works.

**Commit:** `Port generalization experiment to ChainRolloutSource/EncodedNeuralFB` → `git push`.

---

## PHASE 7 — A/B + e-test comparison harness

### Task 7.1 — E-process (anytime-valid paired test)

**Create `latentchess/abtest.py`:**

```python
LAMBDA_GRID = (0.25, 0.5, 1.0, 1.5)

class EValueTest:
    """Anytime-valid test for paired bounded outcomes.
    Outcomes x_a, x_b in [0,1]. y = (x_a - x_b + 1)/2 in [0,1].
    H0_upper: E[y] <= 1/2  (A not better).  For each lambda in LAMBDA_GRID
    keep the product E_lam *= 1 + lam*(y - 1/2)  (>= 0 for lam <= 2).
    e_upper = mean over the grid (mixture of supermartingales is a
    supermartingale). Symmetrically e_lower for H0_lower: E[y] >= 1/2 with
    factors 1 - lam*(y - 1/2). Track running max of each (for reporting).
    API:
      update(x_a: float, x_b: float) -> None
      @property e_a_better (=e_upper), e_b_better (=e_lower)
      def decision(self, alpha=0.05) -> str   # "a_better"|"b_better"|"continue"
        (reject H0_upper when e_a_better >= 1/alpha, i.e. A better)
      n: int   # updates so far
    """
```

**Tests (`tests/test_abtest.py`):**
- `test_e_process_null_safe`: 200 seeds × 500 paired Bernoulli(0.5,0.5)
  outcomes; fraction of seeds where max(e_a_better) ever ≥ 20 must be
  ≤ 0.05 + 0.02 slack.
- `test_e_process_detects_effect`: x_a~Bern(0.65), x_b~Bern(0.5), 50 seeds:
  ≥ 90% of seeds reach decision "a_better" within 800 updates; record and
  assert median stopping time < 500.
- `test_symmetry`: feeding (x_b, x_a) swaps the decision.

**Commit:** `Add anytime-valid paired e-process (mixture over lambda grid)`

### Task 7.2 — MethodSpec + paired_eval + ComparisonReport

Extend `abtest.py`:

```python
@dataclass
class MethodSpec:
    name: str
    build: Callable[[], Policy]      # closures capture chain/emb/etc.

@dataclass
class ComparisonReport:
    method_a: str; method_b: str
    n_pairs: int
    mean_a: float; mean_b: float
    e_a_better: float; e_b_better: float
    decision: str
    stopped_early: bool
    def to_json(self) -> dict

def paired_compare(chain, dtm, spec_a, spec_b, black_builder,
                   starts: np.ndarray, cap=70, alpha=0.05, seed=0,
                   outcome=lambda rec: 1.0 if rec.result == "mate" else 0.0,
                   min_pairs=50) -> ComparisonReport:
    """For each start s (MATCHED across arms) play one game per arm with
    play_game using rng=default_rng(seed + i) RECREATED IDENTICALLY for
    each arm (same seed for A's game and B's game at pair i — matched
    randomness). Feed outcomes to EValueTest; early-stop when decision !=
    'continue' AND i+1 >= min_pairs; else run out the starts."""
```

**Tests:**
- `test_paired_compare_oracle_beats_noisy` (KRk, fast): spec_a =
  DTMOraclePolicy, spec_b = EpsGreedy(DTMOraclePolicy, 0.5), black =
  EpsOptimalDTM(0), 400 starts → decision "a_better", stopped_early True.
- `test_paired_matched_seeds`: identical specs → decision "continue",
  mean_a == mean_b exactly (matched rng ⇒ identical games).

**Commit:** `Add MethodSpec/paired_compare/ComparisonReport with matched-seed arms and early stop`

### Task 7.3 — Deflated-SVD ablation registered

**Modify `latentchess/cone/tabular.py`:** add

```python
@classmethod
def fit_deflated(cls, P, gamma, d, deflate_index: int, n_oversample=10, seed=0):
    """Port of code/diagnostics.py rsvd_defl: right-multiply the successor
    matvec by a mask zeroing column `deflate_index` (the DRAW column):
    Y = sm_matvec(P, mask*Omega, gamma); Z = mask * sm_matvec(P.T, Q, gamma).
    Returns TabularFB."""
```

Register in `EMBEDDING_METHODS` under `"fb_svd_deflated"` via a small
factory (the registry stores callables; a `functools.partial`-style wrapper
class is fine — keep `TabularFB` itself registered as `"fb_svd"`).

**Test:** `test_deflated_fit_runs_and_differs`: on KRk exact_P, deflated
(deflate_index=chain.terminals.draw) vs plain fit at d=32 produce reach
vectors with spearman < 0.999 (they must differ) and both correlate > 0.3
with exact reach.

**Commit:** `Register deflated-SVD ablation as fb_svd_deflated`

### Task 7.4 — compare_methods CLI + the MIN-vs-MEAN validation gate

**Create `experiments/compare_methods.py`:**

```
args: --domain {krk,krkn} (default krk), --compare {agg,embedding}
      (what varies), --a / --b (values: for agg: mean|min; for embedding:
      fb_svd|fb_svd_deflated|fb_neural-NOT-implemented-here -> error),
      --n-starts (default 600), --alpha 0.05, --d 48, --gamma (0.92 krk /
      0.93 krkn), --seed 0.
Behavior: build chain+dtm (+ trained scores: for krkn REQUIRE
data/derived/krkn_F.npy etc. — error with instructions to run
experiments/train_krkn.py first; for krk train a quick 2-round curriculum
inline, ~10 s); build two TablePolicy specs differing ONLY in the compared
axis (readout agg MEAN vs MIN over the same scores, or embeddings);
paired_compare vs optimal black from won starts; print the
ComparisonReport JSON.
```

**Validation gate (manual, KRkn, ~2 min):**
`python experiments/compare_methods.py --domain krkn --compare agg --a min --b mean`
must output decision `"a_better"` (MIN beats MEAN — the documented
+20-point effect) with early stop. Add
`test_compare_min_beats_mean_krkn` as `@pytest.mark.slow` wrapping this.
Fast test: `test_compare_cli_krk_runs` — krk domain, agg comparison,
n-starts 150; asserts a valid decision string is produced (any value).

**Verify phase 7:** suites green; gate passes.
**Commit:** `Add compare_methods CLI; e-test rediscovers the MIN-vs-MEAN readout effect` → `git push`.

---

## PHASE 8 — Port remaining scripts, delete legacy, docs, final repro

General porting rules for 8.1–8.5: each new script goes in `experiments/`
(viz ones in `experiments/viz/`), uses `io.paths` for ALL outputs
(`derived_dir()` for arrays, `generated_dir()` for png/json/html), absolute
latentchess imports, argparse with sane defaults, ≤ ~150 lines each. After
each port, run the script, confirm output exists and prints sane numbers,
then `git rm` the corresponding legacy file(s) in the same commit.

### Task 8.1 — Rung-1 experiment
Port `code/experiment.py` → `experiments/krk_rung1.py` using: `build_chain`,
`exact_P`, `TabularFB`, `rank_error`, `concepts.KMeansVQ(n_tokens=32)` +
`usage_perplexity`, `arena.evaluate`-style engine eval (uniform-random
black = `RandomOpponent`), `sm_matvec` for exact reach, and matplotlib
figures via `viz.plots.style` where trivial (figures go to
`generated_dir()`). Keep the five sections and the G-M1 gate printout.
Preserve the empirical-P learning curve using `empirical_P` on transitions
from `rollout_transitions(chain, RandomPolicy(), RandomOpponent(), ...)`.
**Gate:** mate-rate at 32k games within ±0.02 of 0.955
(tests/baselines/experiment.log); rank-64 reach_rel_err within ±0.02 of
0.4059. Delete `code/experiment.py`.
**Commit:** `Port rung-1 KRk experiment; reproduces 95.5% / rank-probe baselines`

### Task 8.2 — Search sweep
Port `code/exp_search.py` → `experiments/krkn_search_sweep.py`: load
derived krkn arrays (error if missing, point at train_krkn.py); for each
goal in {oracle region, B[MATE] alone}: `scores = F @ z`;
`fill_terminal_state_scores`; k = 0..6: `V = backup(scores, chain,
ReplyAgg.MIN, ts, k)`; policy = `greedy_policy(V, chain, MIN, ts)`;
evaluate vs optimal black from won starts (n=400, cap 70, seed 99) printing
conversion / exact-DTM rate / tempo / rook-loss. Also produce the
goal-region PCA scatter via `PCAProjection` into `generated_dir()`.
**Gate:** with the CURRENT derived arrays, k=1 conversion ≥ k=0 − 0.02 and
conversion collapses (< 0.1) for k ≥ 2 (the documented pessimistic-collapse
regime). Delete `code/exp_search.py`.
**Commit:** `Port k-ply search sweep onto shared readout.backup`

### Task 8.3 — Diagnostics
Port `code/diagnostics.py` → `experiments/diagnostics_krk.py` (D1 fixed
DTM ceiling via a mean-readout on dtm with draws scored bad — reuse
DTMOraclePolicy is NOT the same: D1 historically used mean-over-replies;
implement with `move_values(dtm-based scores...)` MEAN and draw=90 —
simplest faithful port: reuse original logic with chain accessors; D2
deflated rank probe via `TabularFB.fit_deflated`; D3 `util.ridge_r2`
probes). Delete `code/diagnostics.py` and `code/sklearn_free.py`.
**Commit:** `Port diagnostics (fixed ceiling, deflated probe, ridge concept audit)`

### Task 8.4 — Static atlases and t-SNE maps
Port in one task, four scripts → `experiments/viz/`:
- `atlas.py` → `atlas_krk.py` (loads `Fn_neural`/`reach_neural`/`holdout_mask`
  from derived_dir — produced by generalization.py; PCAProjection).
- `region_map.py` → `region_map_krk.py` (rebuilds a KRk PI cone via a
  short CurriculumTrainer run (3 rounds, 8000 games) instead of the
  copy-pasted PI block; KMeansVQ(12); kde fields via plots.kde_layer).
- `krkn_map.py` + `tsne_maps.py` + `tsne_cones.py` → `maps_krkn.py` with
  `--projection {pca,tsne}` producing region/goal maps, and `cones_krkn.py`
  producing the oracle-vs-planner + cone filmstrip + width figures (reuse
  `viz.payload.KrknViewerBuilder` pickers for policies; FittedMap for
  coords; width measured in F-space as in the original).
- `krrk_atlas.py` → `atlas_krrk.py` (needs `dtm_union` + a trained KRRk
  field: run `experiments/train_krrk.py` first — script should error
  helpfully if arrays missing; add `--quick` flag training 3 rounds inline).
**Gate:** every script runs end-to-end producing PNGs in
`generated_dir()`; no pixel-parity requirement. Delete the four legacy
files (+ `tsne_maps.py`, `tsne_cones.py` = six files total).
**Commit:** `Port static atlases and t-SNE/PCA maps onto FittedMap/plots`

### Task 8.5 — KRk viewers
Port `gen_ui_data.py` / `gen_ui_data_pi.py` / `gen_ui_data_vs_optimal.py`
→ ONE script `experiments/viz/build_krk_viewer.py` with
`--engine {random-data,pi} --opponent {random,optimal}` covering the three
legacy variants (they differ only in field construction + opponent). Build
payloads matching the krk_viewer template schema (READ
`latentchess/viz/templates/krk_viewer.html` lines ~140-160 for the DATA
fields; the legacy generators are the schema reference), then `build_html`
into `generated_dir()`. Reuse `concepts.KMeansVQ` for tokens and
`krk.concept_features` for the concept bars. Delete the three legacy
generators and `code/gen_krkn_viewer.py` (superseded in Phase 4), and
`code/krkn_viewer_template.html` + `code/viewer_template.html` (now living
in `latentchess/viz/templates/`).
**Gate:** the generated `thought-viewer-*.html` opens (check: file contains
`const DATA = {` and no `__DATA__`), and `test_viewer.py` (NEW small test)
asserts one `played=True` per node's cands in a generated payload.
**Commit:** `Port KRk viewer generators into one parameterized builder; retire legacy templates`

### Task 8.6 — Delete legacy, final sweep
- `git rm -r code/` (everything remaining: `domain.py krkn.py krrk.py
  learn.py neural.py core.py` and the already-ported trainer scripts
  `exp_policy_iteration.py exp_krkn.py exp_krkn2.py exp_krrk.py`,
  `exp_generalization.py`). BEFORE deleting, grep the repo for any
  remaining `from code.` / `sys.path` references (there must be none
  outside tests/baselines logs).
- Delete `code/out/` leftovers; confirm `.gitignore` still covers
  data/ and artifacts/generated/.
- Run: fast suite, slow suite, `experiments/repro_check.py` (NEW script):
  runs krk_rung1 + train_krkn (fresh ckpt) + krkn_search_sweep and diffs
  key numbers against `tests/baselines/expected.json` with bands:
  krk mate-rate 0.955±0.03; krkn final conversion ≥ 0.45; final AUC ≥ 0.65;
  sweep k=1 conversion ≥ 0.60. Prints PASS/FAIL per item, exit 1 on FAIL.
**Commit:** `Remove legacy code/ tree; add repro_check against captured baselines`

### Task 8.7 — Docs + final push
- Rewrite `README.md` §5 (file map & how to run) for the new layout:
  install (`pip install -e ".[dev]"`), test commands, the pipeline:
  `python -m latentchess.domains.krk` → `experiments/krk_rung1.py` →
  `python -m latentchess.domains.krkn` → `experiments/train_krkn.py` →
  `experiments/krkn_search_sweep.py` → `experiments/viz/maps_krkn.py` →
  `experiments/viz/build_krkn_viewer.py --projection tsne`; plus the
  Lichess fixture demo command. Keep §§1-4, 6, 7 intact (history/results).
- Write `ARCHITECTURE.md`: one-screen layer diagram (board → domains →
  chain → {cone, planner, opponents} → {game, arena, train, abtest} →
  viz/data/io), the six extension points with 3-line "how to add a new X"
  recipes (embedding method, quantizer, opponent, projection, move
  identity, plan selector), and the invariants (TerminalScores only;
  state_scores length-n convention; reduceat tie-break parity; seeds).
- Final: full test suites, `git push`.
**Commit:** `Rewrite README run instructions; add ARCHITECTURE.md`

---

## Final acceptance checklist

- [ ] `pytest -m "not slow"` — all green, < 60 s
- [ ] `pytest -m slow` — all green, < 6 min
- [ ] `experiments/repro_check.py` — PASS on all bands
- [ ] `experiments/plan_memory_demo.py` — availability flip demonstrated
- [ ] `experiments/compare_methods.py --domain krkn --compare agg --a min --b mean` — "a_better", early stop
- [ ] `experiments/build_lichess_shards.py --pgn tests/fixtures/lichess_fixture.pgn.zst --max-games 10` — shards + manifest written
- [ ] `experiments/generalization.py` — holdout spearman ≥ 0.40, gap ≤ 0.10
- [ ] `code/` directory gone; `git status` clean; branch pushed
- [ ] No file under `data/` or `artifacts/generated/` tracked by git

## Deferred (do not start): M1.5 research phase
Precondition discovery (enabling sets/contrast sets/technique quantization),
region graph + decomposition generators + give-up rules, LearnedSelector /
PlanMCTS, region-pair & displacement MoveIdentity implementations, and the
two pre-registered gates (KRkn conditional-capture audit; deep-conversion
frontier). Also full-board: torch encoder + FB heads, descriptive
results-head, deep-audit protocol (SF depth-30 / lc0-WDL / arena MC),
Stockfish UCI opponent, 8×8 viewer. These are DESIGNED (see
docs/DESIGN_NOTES or the plan history) but explicitly out of scope here.
