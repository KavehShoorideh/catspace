"""PLAN 78.14 SLEEP: from the wake's logs to verified lemmas and the next snapshot (the architecture document's surprise loop). Python orchestrates, the engine does every
reading (the cards, the matching, the verdicts).

  1 DETECT   a surprise is a pair of consecutive decisions by the two sides where the second side sees a loss of the first side's units (along its best line) that the
             first side's best line did not foresee: the units lost along the second line exceed the first's by at least `min_units`. Read from the log's ledger events.
  2 REPLAY   the position before the first side's move becomes a CARD: every legal move's effect on the facts and its exact value at `depth`, with the units each side
             loses along its best line. The surprise is CONFIRMED when the exact reading reproduces the loss for the move that was played. The SAVING moves are those that
             lose fewer units and are not worth less. (A teacher, if one is given, may label: it is flagged training-only and recorded on what it touched.)
  3 EXPLAIN  each saving move's effect on the facts, with the facts around it, is a ground lemma (abstraction.seed_from).
  4 ABSTRACT the climb of abstraction.generalize on the DEVELOPMENT games' cards: lift constants, drop literals, every step exact-checked.
  5 VERIFY   the lemma that survives is read on the HELD-OUT games' cards -- games the seed and the climb never saw: precision, excess over chance, selectivity, and
             enough firings in enough independent games. Only a lemma that passes is stored. Its status is SAMPLED: exact on a sample, not proved.
  6 PUBLISH  with --write the lemmas go to rules/learned/<game>/lemmas and the next snapshot is published (game_store.publish).

The funnel is reported at every stage; nothing is hidden when a stage empties.

  .venv/bin/python -m catspace.incubator.analyst.discovery.sleep chess --log <wake.jsonl> [--depth 4] [--write]
  .venv/bin/python -m catspace.incubator.analyst.discovery.sleep chess --cycles 2 --games 16 --write      # wake, sleep, publish, again
  .venv/bin/python -m catspace.incubator.analyst.discovery.sleep chess --games 48 --judge "material (mined)"   # play and value the cards by a named store record

THE JUDGE IS PART OF THE CLAIM: a lemma says "an optimal move of this kind exists", and optimal is by the judge, so the wake plays and the cards are valued by ONE
named store record, recorded on every lemma. In quiet self-play positions many legal moves tie for optimal under any judge of units (measured 2026-11-08 on the same
40 cards at depth 5: 0.22 of the legal moves under `units (no values)`, 0.47 under `material (mined)`), which is why the `expected` figure of a lemma is high and
its excess over chance is the gate that does the work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field

import cs_engine as CE

from catspace.incubator.analyst.discovery import abstraction as AB
from catspace.incubator.analyst.discovery import detectors as D
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import judges as JG
from catspace.incubator.analyst.discovery import evidence as EV


# ---------------------------------------------------------------------------------------------------------------- the log

def read_log(path: str) -> tuple:
    """-> (decisions {(run, game, ply): decision}, games {(run, game): game row}) from a wake log. A decision: role, move, proved, the ledger step's value / depth /
    line / events, the row."""
    decisions, games = {}, {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "game":
            r.setdefault("start", None)                      # (a log from before plan 78.22 has no start: the game's own)
            games[(r["run"], r["game"])] = r
        elif r.get("kind") == "position" and r.get("label", "").count(":") >= 3 and not r.get("error"):
            run, g, ply, role = r["label"].rsplit(":", 3)
            steps = r["steps"]
            led = next((s for s in steps if s["operator"] == "ledger" and s["status"] == "Read"), None)
            proved = any(s["operator"] == "forcing" and s["status"] == "Proved" for s in steps)
            decisions[(run, int(g), int(ply))] = {"role": int(role), "move": r["move"], "proved": proved, "ledger": led, "row": r}
    games = {(k[0], int(k[1])): v for k, v in games.items()}
    return decisions, games


def losses(step: "dict | None", role: int) -> int:
    """The units `role` loses along a ledger step's best line."""
    return 0 if step is None else sum(1 for ro, e in step["events"] if ro == role and e.startswith("lose "))


def surprises(decisions: dict, games: dict, min_units: int = 1) -> list:
    """The surprises of the decisions (see the module)."""
    out = []
    for (run, g, ply), a in sorted(decisions.items()):
        b = decisions.get((run, g, ply + 1))
        game = games.get((run, g))
        if b is None or game is None or a["role"] == b["role"] or a["proved"] or a["ledger"] is None:
            continue
        predicted = losses(a["ledger"], a["role"])
        seen = predicted + min_units if b["proved"] else (losses(b["ledger"], a["role"]) if b["ledger"] is not None else None)
        if seen is None or seen - predicted < min_units:
            continue
        bd = (b["ledger"] or {}).get("depth")
        if b["proved"]:                                     # a forced line of n of B's moves: 2n - 1 plies from B's position
            n = next((st["depth"] for st in b["row"]["steps"] if st["operator"] == "forcing" and st["status"] == "Proved"), None)
            need = None if n is None else 2 * n
        else:
            need = None if bd is None else bd + 1           # the first side's move, then what the second side read
        out.append({"run": run, "game": g, "ply": ply, "role": a["role"], "predicted": predicted, "seen": seen, "played": a["move"], "reply": b["move"],
                    "b_depth": bd, "need": need, "moves_before": list(game["moves"][:ply]), "start": game.get("start"), "group": f"{run}:{g}", "id": f"{run}:{g}:{ply}"})
    return out


# ---------------------------------------------------------------------------------------------------------------- rankings from outside sources

MATE_SCORE = 100_000                 # a mate stands for a score of this magnitude (less the distance): only the ORDER of a ranking is ever used, never its scale


def parse_multipv(text: str) -> list:
    """-> [(move, score)] best first, from the `info ... multipv k ... score cp|mate n ... pv <move> ...` lines of a UCI search (pure; the later line for a rank is the
    deeper one and wins). A mate in n scores +-(MATE_SCORE - n)."""
    best: dict = {}
    for line in text.splitlines():
        t = line.split()
        if not t or t[0] != "info" or "pv" not in t or "score" not in t:
            continue
        k = int(t[t.index("multipv") + 1]) if "multipv" in t else 1
        i = t.index("score")
        kind, val = t[i + 1], int(t[i + 2])
        score = val if kind == "cp" else (MATE_SCORE - abs(val)) * (1 if val > 0 else -1)
        j = t.index("pv")
        if j + 1 < len(t):
            best[k] = (t[j + 1], score)
    return [best[k] for k in sorted(best)]


def parse_multipv_lines(text: str) -> list:
    """-> [(move, score, line)] best first: `parse_multipv` with each rank's whole principal line (`line[0]` is the move). Pure; the later `info` line for a rank wins."""
    best: dict = {}
    for row in text.splitlines():
        t = row.split()
        if not t or t[0] != "info" or "pv" not in t or "score" not in t:
            continue
        k = int(t[t.index("multipv") + 1]) if "multipv" in t else 1
        i = t.index("score")
        kind, val = t[i + 1], int(t[i + 2])
        score = val if kind == "cp" else (MATE_SCORE - abs(val)) * (1 if val > 0 else -1)
        line = t[t.index("pv") + 1:]
        if line:
            best[k] = (line[0], score, line)
    return [best[k] for k in sorted(best)]


def disagreement(card: dict, ranking: list, source: str = "teacher", top_k: int = 3) -> "dict | None":
    """THE SECOND KIND OF SURPRISE (rank-based, scale-free, game-free). `ranking`: [(move name, score)] by an outside source, best first by score; `card`: the engine's
    card (moves with name and exact value). A DISAGREEMENT is when (a) the source's top move has a card value below the card's best -- we say it loses something, the source says
    it is best -- or (b) none of the card's optimal moves is among the source's top `top_k`. Moves the card does not have are ignored. -> None when they agree, else
    {kind: "teacher", source, reason, source_order, our_order, source_top, source_top_value, best}."""
    value = {m["name"]: m["value"] for m in card["moves"]}
    order = [m for m, _ in sorted(ranking, key=lambda x: -x[1]) if m in value]
    if not order:
        return None
    best = max(value.values())
    optimal = {m for m, v in value.items() if v == best}
    reason = "top move below our best" if value[order[0]] < best else ("no optimal move of ours in the top %d" % top_k if not optimal & set(order[:top_k]) else None)
    if reason is None:
        return None
    return {"kind": "teacher", "source": source, "reason": reason, "source_order": order, "our_order": [m for m, _ in sorted(value.items(), key=lambda x: (-x[1], x[0]))],
            "source_top": order[0], "source_top_value": value[order[0]], "best": best}


def break_ties(moves: list, ranking: list) -> list:
    """The card moves `moves` (the judge-tied saving moves) ordered by the source's ranking first (a move it does not rank goes last), then by the fewer units lost, then name."""
    rank = {m: i for i, (m, _) in enumerate(sorted(ranking, key=lambda x: -x[1]))}
    return sorted(moves, key=lambda m: (rank.get(m["name"], len(rank)), m["lost"], m["name"]))


# ---------------------------------------------------------------------------------------------------------------- the tools

def plan_tools(diag: dict, source, s: "Settings") -> dict:
    """THE SLEEP CHOOSES ITS TOOLS FROM WHAT THE ENGINE'S OWN LOG SAYS IT LACKS (the training-time controller, in its first and simplest form). `diag`: diagnose.diagnose's
    report of the wake log; `source`: a rank source (a teacher or an estimate), a LIST of them, or None. The one rule so far: when the judge ties often -- the tie rate is above
    `tie_threshold`, or the wake never asked for ties so the rate is unknown -- a rank source is worth its cost: it breaks the ties among saving moves and registers
    the rank-based surprises. With several sources the human games (`source == "human"`, plan 78.19 stage E: "in the beginning, if it doesn't know what to do, it queries the
    human games") come FIRST when coarse, the others after in the order given; `sources` names the order and `source` stays the first. Everything chosen and why is returned."""
    ties = diag.get("ties", {})
    asked, rate = bool(ties.get("asked")), ties.get("rate")
    coarse = (not asked) or (rate is not None and rate >= s.tie_threshold)
    why = ("the wake did not read ties: the judge's coarseness is unknown" if not asked else f"tie rate {rate} against the threshold {s.tie_threshold}")
    chosen = {"teacher_mode": s.teacher_mode, "teacher_surprises": s.teacher_surprises}
    many = [x for x in (source if isinstance(source, (list, tuple)) else [source]) if x is not None]
    if not many:
        return {"diagnosis": ties, "coarse": coarse, "chosen": chosen, "why": "no rank source given; " + why}
    if coarse:
        chosen = {"teacher_mode": "break", "teacher_surprises": True}
        many = sorted(many, key=lambda x: getattr(x, "source", "teacher") != "human")          # stable: the human source first, the rest as given
        if getattr(many[0], "source", "") == "human":
            why += "; the human games are queried first"
    first = many[0]
    out = {"diagnosis": ties, "coarse": coarse, "source": getattr(first, "source", "teacher"), "training_only": bool(getattr(first, "training_only", True)), "chosen": chosen, "why": why}
    if isinstance(source, (list, tuple)):
        out["sources"] = [getattr(x, "source", "teacher") for x in many]
    return out


class EstimateSource:
    """A RANK SOURCE from the engine's own Monte Carlo tree search operator (cs_engine.estimate): the legal moves in visit order with their mean outcome. It belongs to the
    INFERENCE kit (nothing outside the rules is consulted), so a lemma it touched is not a teacher's -- but its figures are sampled, not exact, which is why a seed from it
    still needs the deeper exact replay like any other source's. The same shape as the teacher's `rank`, so the detector and the tie-breaker read either."""
    source = "mcts"
    training_only = False

    def __init__(self, seconds: float = 0.2, seed: int = 1, max_nodes: int = 10_000_000, **options):
        """`options` are `cs_engine.estimate`'s (plan 78.18: rollout, leaf, eps, scale, temperature, store, prior, floor); the source's name records the policy chosen
        (`mcts` for the plain uniform search, else `mcts:<rollout>/<leaf>`)."""
        self.seconds, self.seed, self.max_nodes, self.options = seconds, seed, max_nodes, dict(options)
        self.specs = self.options.pop("specs", None) or {}          # {variant: description text}: the lemma prior's fact view, per variant ranked
        rollout, leaf = self.options.get("rollout", "uniform"), self.options.get("leaf", "draw")
        if (rollout, leaf) != ("uniform", "draw"):
            self.source = f"mcts:{rollout}/{leaf}"

    def rank(self, variant: str, moves: list, cap: "int | None" = None, start: "str | None" = None) -> list:
        opts = dict(self.options)
        if variant in self.specs:
            opts["spec"] = self.specs[variant]
        r = CE.estimate(variant, start, list(moves), seconds=self.seconds, max_nodes=self.max_nodes, seed=self.seed, **opts)
        out = sorted(((name, visits) for name, visits, _mean in r["visits"]), key=lambda x: -x[1])
        return out[:cap] if cap else out

    def __call__(self, variant: str, moves: list, start: "str | None" = None) -> "str | None":
        r = self.rank(variant, moves, start=start)
        return r[0][0] if r else None

    def close(self) -> None:
        pass


def source_of(name: str, seconds: float = 0.2, seed: int = 1, store: "str | None" = None):
    """`gnn` or `gnn:<model path>`: the GraphSource (graph_prior.py; inference kit, all in Rust; default weights file paths.derived("graph_prior_chess.json")). `human` or `human:<path>`: the HumanSource over the human-games index (human_index.py; training only; default path human_index.default_path("chess")). Else an EstimateSource from a name: `mcts` (uniform playouts scored at the game's end: the plain search) or `mcts:<rollout>/<leaf>` (plan 78.18); `store` (a snapshot
    path) gives the balance leaf its judge and the lemma prior its lemmas."""
    if name == "gnn" or name.startswith("gnn:"):
        # plan 78.20: the graph net over the typed fact graph (a PRIOR of the inference kit); default weights file paths.derived("graph_prior_chess.json")
        from catspace.incubator.analyst.discovery import graph_prior as GP
        return GP.GraphSource(name.partition(":")[2] or GP.default_path("chess"))
    if name == "human" or name.startswith("human:"):
        from catspace.incubator.analyst.discovery import human_index as HI
        return HI.HumanSource(name.partition(":")[2] or None)
    opts = {}
    if ":" in name:
        rollout, _, leaf = name.split(":", 1)[1].partition("/")
        opts["rollout"] = rollout or "uniform"
        if leaf:
            opts["leaf"] = leaf
        if rollout == "lemma":
            opts["prior"] = True
    if store and opts:
        opts["store"] = str(store)
        if opts.get("prior"):
            # the lemma prior reads a position's facts through the variant's description: the store's game has one per variant, chosen when a position is ranked
            opts["specs"] = GS.spec_texts(CE.snapshot_info(str(store))["game"])
    return EstimateSource(seconds, seed, **opts)


# ---------------------------------------------------------------------------------------------------------------- the sleep

@dataclass
class Settings:
    depth: int = 4
    fuse: int = 20_000_000
    min_units: int = 1
    stride: int = 2                  # one position in `stride` of each game's decisions goes into the corpus
    max_cards: int = 3000
    max_saving: int = 3              # saving moves seeded per surprise
    max_seeds: int = 60
    dev_share: float = 0.5
    seed: int = 1
    threads: int = 0
    teacher_mode: str = "off"        # off | label | filter | break (among tied saving moves prefer the source's highest-ranked)
    teacher_surprises: bool = False  # register the rank-based surprises (the source's ordering against the card's) on development cards
    teacher_positions: int = 200     # at most this many development cards are ranked by the source (it costs its movetime each)
    auto_tools: bool = True          # let the diagnosis of the log choose the tools (a rank source breaks ties and registers surprises when the judge ties often)
    tie_threshold: float = 0.2       # the tie rate (decisions with more than one move as good as the best) above which the judge is called too coarse
    teacher_top_k: int = 3
    regret: bool = True              # plan 78.22: whenever a rank source is given, the move PLAYED is read against its ranking (detectors.Regret), whatever the tie diagnosis says
    judge_by: str = "store"          # plan 78.23: "store" (the cards are valued by the store's record) | "source" (the rank source given is the judge: judges.revalue)
    judge_positions: "int | None" = None   # with judge_by == "source": at most this many cards (the first ones) are revalued by the source (None: all); the rest are left out
    vocabulary: "str | None" = None  # plan 78.24: the NAME of the lemma vocabulary (rules/learned/<game>/vocabulary.<name>.json); None is the default one
    judge: "str | None" = None       # the store's VALUATION record the cards are valued by (None: the record in use); never values written in code
    standard: str = "score"          # plan 78.31: "score" (a faster mate is better, any named move best counts) | "result" (the rank source grades by RESULT class, every named move must be best)
    evidence_per_candidate: int = 60     # plan 78.30: positions of a corpus the rank source is asked about per candidate (only with a corpus, only for forced seeds)
    evidence_total: int = 600            # ... and in all (new questions) per sleep
    evidence_corpus_cap: int = 30000     # ... corpus positions read into cards (a stride beyond it)
    gates: AB.Gates = field(default_factory=AB.Gates)
    held: AB.Gates = field(default_factory=lambda: AB.Gates(min_fires=3, min_groups=2))


def apply_standard(s: Settings, teacher=None) -> Settings:
    """The settings with the gates' quantifier of the standard (result: every move the rule names must be in the best class). A rank source that grades by score under the result standard (or the
    other way round) is refused: results would be mixed."""
    if s.standard not in ("score", "result"):
        raise ValueError(f"standard is 'score' or 'result', not {s.standard!r}")
    if teacher is not None and getattr(teacher, "grade_mode", "score") != s.standard:
        raise ValueError(f"standard {s.standard!r} but the rank source grades by {getattr(teacher, 'grade_mode', 'score')!r}: the two must agree")
    import dataclasses
    on = s.standard == "result"
    return Settings(**{**s.__dict__, "gates": dataclasses.replace(s.gates, all_named=on), "held": dataclasses.replace(s.held, all_named=on)})


def standard_text(s: Settings, teacher=None) -> str:
    g = teacher
    mode = getattr(g, "grade_mode", "score") if g is not None else s.standard
    if mode == "result":
        return (f"RESULT: the rank source grades by result class (win / draw / loss; decisive at {getattr(g, 'decisive', None)} cp or a mate), every move the rule names must be in the best class, "
                f"a card where every move is in it is trivial")
    return "SCORE: the rank source grades by score (mate_bucket " + str(getattr(g, 'mate_bucket', 1)) + ", gap " + str(getattr(g, 'gap', None)) + "), a firing holds when ANY move the rule names is best"


def is_dev(group: str, s: Settings) -> bool:
    h = int(hashlib.sha1(f"{s.seed}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < s.dev_share


def summary(ev: dict, gates: "AB.Gates | None" = None) -> dict:
    out = {k: (round(ev[k], 4) if isinstance(ev[k], float) else ev[k]) for k in ("fires", "holds", "refuted", "unknown", "groups", "groups_held", "precision", "expected", "excess", "selectivity")}
    out["lower"] = round((gates or AB.Gates()).lower(ev), 4)
    return out


def spread(seeds: list, key) -> list:
    """The seeds interleaved by `key` (their game): a cap on the seeds then reads every game, not the first few."""
    by: dict = {}
    for x in seeds:
        by.setdefault(key(x), []).append(x)
    out, queues = [], list(by.values())
    while queues:
        out.extend(q.pop(0) for q in queues)
        queues = [q for q in queues if q]
    return out


def teacher_seeds(cards, dev_idx: list, positions: dict, teacher, variant: str, spec: dict, vocab: str, judge: dict, s: "Settings", rep: dict, played: "dict | None" = None,
                  detector: str = "disagreement") -> tuple:
    """The rank-based surprises on the DEVELOPMENT cards (plan 78.22: through detectors.py, which this wraps; `detector` is `disagreement` or `regret`). `positions`:
    {card id: (start, moves)}; `played`: {card id: the move played} (the regret's). A seed only where our own deeper reading confirms (`detectors.confirm`).
    -> (seeds [(x, card, move, source move, played, dev indices)], the deeper CardSet or None). Records rep["teacher"] (disagreement) or rep["regret"]."""
    ctx = D.Context(variant, s, {}, {}, cards=cards, dev_idx=dev_idx, positions=positions, played=played or {}, spec=spec, vocab=vocab, judge=judge, rep=rep)
    return D.run(ctx, [D.detector_of(detector, teacher)])


def sleep(game: str, log: str, *, variant: "str | None" = None, settings: "Settings | None" = None, teacher=None, write: bool = False, say=print, extra_positions: "list | None" = None, corpus: "list | None" = None, prefer=None) -> dict:
    """One sleep over a wake log -> the report (the funnel, the lemmas admitted, what was rejected and why). `teacher(variant, moves) -> uci or None` is optional
    and training-only; `write` files the lemmas and publishes the next snapshot. `extra_positions` (plan 78.28): positions beside the log's, each {id, start, moves (the prefix),
    played (the move made there), group}; they join the corpus as cards (and so the rank source's regret read) exactly like a logged decision; absent, the sleep is unchanged.
    Plan 78.29: an entry with `"force": True` is a FORCED SEED (an open problem): its group is always in the DEVELOPMENT half (never held out), the rank source is always asked about it (not
    only one development card in `stride`), a regret is raised whenever the source grades the move played below its best (no gap or top-k), and its seeds are climbed FIRST (not cut by
    `max_seeds`). Admission is unchanged: the same gates on the held-out cards. Prints `OBS sleep forced seeds: ...` and, per forced seed that does not end as a lemma, the stage where it
    died and the best candidate's development / held figures; the report gains `forced`.
    Plan 78.30: `corpus` ([{start, moves, group}], default None: nothing changes) lets a FORCED seed's candidate that has no refutation but too little support (development or held-out) gather
    EVIDENCE (evidence.py): corpus positions where its body binds and the rule names a move, the rank source asked at up to `evidence_per_candidate` of them (`evidence_total` in all, spread over
    groups), split dev / held by group as the games are, added to the candidate's cards; the SAME gates are re-applied. A refutation there is a counter-example (the climb is tried once more on
    the extended development cards, else the candidate dies). Prints `OBS sleep evidence <seed>: ...` and `OBS sleep evidence total: ...`. `prefer(lemma, x) -> {ok, named, lose, ...}` (optional):
    among the candidates of one forced seed that pass the gates, those whose named moves at the seed problem all keep the verdict are admitted; when none does, all are, and it says so."""
    s = apply_standard(settings or Settings(), teacher)
    variant = variant or game
    t0 = time.perf_counter()
    say(f"OBS sleep standard: {standard_text(s, teacher)}")
    decisions, games = read_log(log)
    spec, vocab = GS.spec_texts(game, s.vocabulary), json.dumps(GS.vocabulary(game, s.vocabulary))
    if s.judge_by not in ("store", "source"):
        raise ValueError(f"judge_by is 'store' or 'source', not {s.judge_by!r}")
    if s.judge_by == "source" and teacher is None:
        raise ValueError("judge_by='source' needs a rank source (a teacher or --source): it is the judge")
    judge = GS.judge_of(game, s.judge) or {}
    rep = {"game": game, "log": log, "settings": {**s.__dict__, "gates": s.gates.as_dict(), "held": s.held.as_dict()}, "judge": judge.get("name", "count"), "funnel": {}, "lemmas": [], "rejected": []}
    # the sleep reads the engine's own account of the wake and chooses its tools by it (plan 78.17); what the caller set explicitly stands
    from catspace.incubator.analyst.discovery import diagnose as DG
    diag = DG.diagnose(log, min_units=s.min_units, say=lambda *_a, **_k: None)
    rep["tools"] = plan_tools(diag, teacher, s)
    if s.auto_tools and teacher is not None and s.teacher_mode == "off" and not s.teacher_surprises:
        s = Settings(**{**s.__dict__, **rep["tools"]["chosen"]})
    if s.judge_by == "source":
        rep["judge"] = "source:" + getattr(teacher, "source", "teacher")
    rep["tools"]["applied"] = {"teacher_mode": s.teacher_mode, "teacher_surprises": s.teacher_surprises}
    f = rep["funnel"]
    f["games"], f["decisions"] = len(games), len(decisions)
    forced_ids = {x["id"] for x in (extra_positions or []) if x.get("force")}
    forced_groups = {x["group"] for x in (extra_positions or []) if x.get("force")}
    ctx = D.Context(variant, s, decisions, games, spec=spec, vocab=vocab, judge=judge, rep=rep, force_ids=forced_ids)
    sur_all = D.Hindsight().detect(ctx)
    f["surprises"] = len(sur_all)
    sur = [x for x in sur_all if x["need"] is not None and x["need"] <= s.depth]          # a replay of `depth` plies can only reproduce a loss it can reach
    f["replayable"] = len(sur)
    # the corpus: every game's decision positions at the stride (capped), then the surprises' positions
    items, seen, moves_of = [], set(), {}
    positions, played_of = {}, {}                          # {card id: (start, moves)}, {card id: the move the engine played}
    for (run, g, ply), d in sorted(decisions.items()):
        game_row = games.get((run, g))
        if game_row is None or ply % s.stride != 0 or len(items) >= s.max_cards:
            continue
        cid = f"{run}:{g}:{ply}"
        items.append((variant, game_row.get("start"), list(game_row["moves"][:ply]), cid, f"{run}:{g}"))
        seen.add(cid)
        moves_of[cid] = list(game_row["moves"][:ply])
        positions[cid], played_of[cid] = (game_row.get("start"), moves_of[cid]), d["move"]
    for x in sur:
        if x["id"] not in seen:
            moves_of[x["id"]] = x["moves_before"]
            items.append((variant, x["start"], x["moves_before"], x["id"], x["group"]))
            seen.add(x["id"])
            positions[x["id"]], played_of[x["id"]] = (x["start"], x["moves_before"]), x["played"]
    for x in (extra_positions or []):
        if x["id"] not in seen:
            items.append((variant, x.get("start"), list(x["moves"]), x["id"], x["group"]))
            seen.add(x["id"])
            moves_of[x["id"]] = list(x["moves"])
            positions[x["id"]], played_of[x["id"]] = (x.get("start"), list(x["moves"])), x["played"]
    if extra_positions:
        f["extra_positions"] = len(extra_positions)
    t_cards = time.perf_counter()
    cards = CE.build_cards(items, spec, vocab, s.depth, s.fuse, s.threads, judge.get("worth", {}), bool(judge.get("ours_first", False)))
    timing = {"cards": round(time.perf_counter() - t_cards, 2), "climb": 0.0, "verify": 0.0}
    f["cards"] = len(cards)
    ids = [json.loads(cards.card(i))["id"] for i in range(len(cards))]
    groups = [json.loads(cards.card(i))["group"] for i in range(len(cards))]
    index = {cid: i for i, cid in enumerate(ids)}
    dev_idx = [i for i, g in enumerate(groups) if is_dev(g, s) or g in forced_groups]
    held_idx = [i for i, g in enumerate(groups) if not is_dev(g, s) and g not in forced_groups]
    by_source = s.judge_by == "source"
    if by_source:
        # plan 78.23: the source is the judge. The cards keep the effects of each move on the facts; every move's value is the source's grade, dev and held alike
        scope = list(range(len(cards)))[:s.judge_positions]
        cards, jrep = JG.revalue(cards, positions, teacher, variant, scope, context=ctx, say=say)
        jrep["disagreement"] = "skipped: the source is the judge, so it cannot disagree with itself"
        jrep["describe"] = JG.describe(teacher)
        rep["judge_report"] = jrep
        keep = set(scope)
        dev_idx, held_idx = [i for i in dev_idx if i in keep], [i for i in held_idx if i in keep]
    ctx.cards, ctx.dev_idx, ctx.index, ctx.positions, ctx.played = cards, dev_idx, index, positions, played_of
    f["cards_dev"], f["cards_held"] = len(dev_idx), len(held_idx)
    # replay: confirm the surprises on the development games and find the saving moves
    seeds, confirmed, with_saving = [], 0, 0
    for x in sur:
        if not is_dev(x["group"], s) or x["id"] not in index:
            continue
        card = json.loads(cards.card(index[x["id"]]))
        played = next((m for m in card["moves"] if m["name"] == x["played"]), None)
        if played is None or card["cut"] or played["lost"] < x["seen"]:
            continue
        confirmed += 1
        best = max(m["value"] for m in card["moves"])
        saving = [m for m in card["moves"] if m["lost"] < played["lost"] and m["value"] == best]          # the lemma's claim is "an optimal move of this kind exists"
        saving.sort(key=lambda m: (-m["value"], m["lost"], m["name"]))
        tm = None
        if teacher is not None and s.teacher_mode == "break":
            if len(saving) > 1:                                        # only a tie is the teacher's to break
                own = saving[0]                                        # the engine's own first choice stays among the seeds: a noisy source may order it out (found 2026-11-08: the
                ranking = D.ask(teacher, variant, x["moves_before"], x["start"])     #  estimate displaced the one seed that verifies)
                saving = break_ties(saving, ranking)
                if own not in saving[:s.max_saving]:
                    saving = saving[:max(0, s.max_saving - 1)] + [own] + [m for m in saving[max(0, s.max_saving - 1):] if m is not own]
                tm = saving[0]["name"] if any(m == saving[0]["name"] for m, _ in ranking) else None
        elif teacher is not None and s.teacher_mode != "off":
            tm = teacher(variant, x["moves_before"]) if x["start"] is None else teacher(variant, x["moves_before"], start=x["start"])
            if s.teacher_mode == "filter":
                saving = [m for m in saving if m["name"] == tm]
        if not saving:
            continue
        with_saving += 1
        for m in saving[:s.max_saving]:
            seeds.append((x, card, m, tm, played, None))
    f["confirmed"], f["with_saving"], f["seeds"] = confirmed, with_saving, len(seeds)
    # the rank-based surprises (development cards only), through the detectors: the source's ordering against the card's (Disagreement, when the tools say so) and the move
    # played against the source's ranking (Regret, whenever a source is given); a seed only where our own deeper reading confirms the source's move (detectors.confirm)
    f["teacher_surprises"] = f["teacher_confirmed"] = 0
    f["regret_surprises"] = f["regret_confirmed"] = f["regret_tied"] = f["regret_unconfirmed"] = 0
    deep_cards, cards_all = None, cards
    dets = []
    if teacher is not None and s.teacher_surprises and not by_source:
        dets.append(D.detector_of("disagreement", teacher))
    if teacher is not None and s.regret and not by_source:
        dets.append(D.detector_of("regret", teacher))
    if by_source and s.regret:
        # the source is the judge: the regret is confirmed directly on the revalued card, no deeper card (judges.regret_seeds)
        seeds += JG.regret_seeds(ctx, teacher, rep)
        r = rep["regret"]
        f["regret_surprises"], f["regret_confirmed"], f["regret_tied"], f["regret_unconfirmed"] = r["surprises"], len(r["confirmed"]), len(r["tied"]), len(r["unconfirmed"])
        f["seeds"] = len(seeds)
        for w in r.get("confirmed_rows", [])[:5]:
            say(f"OBS sleep why {w['id']}: played {w['played']} leads to {' '.join(w['refutation'][:6])} ({JG.say_consequence(w['refutation_end'])}); "
                f"best {w['best']} leads to {' '.join(w['best_line'][:6])} ({JG.say_consequence(w['best_end'])})")
    if dets:
        tseeds, deep_cards = D.run(ctx, dets)
        seeds += tseeds
        if "teacher" in rep:
            f["teacher_surprises"], f["teacher_confirmed"] = rep["teacher"]["surprises"], rep["teacher"]["confirmed"]
        if "regret" in rep:
            r = rep["regret"]
            f["regret_surprises"], f["regret_confirmed"], f["regret_tied"], f["regret_unconfirmed"] = r["surprises"], len(r["confirmed"]), len(r["tied"]), len(r["unconfirmed"])
        f["seeds"] = len(seeds)
        if deep_cards is not None:
            cards_all = CE.CardSet.merge([cards, deep_cards])
    forced_seeds = [t for t in seeds if t[0].get("forced")]
    seeds = forced_seeds + spread([t for t in seeds if not t[0].get("forced")], lambda t: t[0]["group"])[:s.max_seeds]          # (forced seeds first, and not cut by the cap)
    # abstract on the development cards, verify on the held-out cards
    known = {AB.canonical(l) for l in GS.read_lemmas(game, s.vocabulary)}
    admitted, climbed, dev_ok = {}, 0, 0
    evid = None
    if corpus and teacher is not None and forced_ids:
        known_keys = {(st, tuple(mv)) for st, mv in positions.values()}
        evid = EV.Evidence(corpus, variant=variant, spec=spec, vocab=vocab, teacher=teacher, is_dev=lambda g: is_dev(g, s) or g in forced_groups, per_candidate=s.evidence_per_candidate,
                           total=s.evidence_total, cap=s.evidence_corpus_cap, known=known_keys)
        say(f"OBS sleep evidence bounds: corpus {len(corpus)} positions ({len(evid.rows)} read into cards after the cap {s.evidence_corpus_cap}, the sleep's own cards left out), at most "
            f"{s.evidence_per_candidate} asked per candidate and {s.evidence_total} new questions in all, spread over groups")
    ev_log = []
    pending: dict = {}
    for x, card, m, tm, played, dev_override in seeds:
        seed = AB.seed_from(card, m, s.gates, played)
        t1 = time.perf_counter()
        dev_here = dev_idx if dev_override is None else dev_override          # (a teacher seed's own position is read at the deeper depth that confirmed it)
        c = AB.generalize(seed, cards_all, dev_here, s.gates)
        timing["climb"] += time.perf_counter() - t1
        tag = f"{x['id']} {m['name']}"
        if c is None:
            ev0 = AB.evaluate(cards_all, seed, dev_here, s.gates)
            rep["rejected"].append({"seed": tag, "why": "the ground seed is not admissible on the development cards", "dev": summary(ev0, s.gates),
                                    "truncated": ev0["truncated"], "body": len(seed["body"])})
            continue
        climbed += 1
        want = evid is not None and bool(x.get("forced"))
        dev_sup = s.gates.supported(c.ev)
        if not dev_sup and not (want and c.ev["refuted"] == 0):
            rep["rejected"].append({"seed": tag, "why": "too little support on the development cards", "dev": summary(c.ev, s.gates)})
            continue
        h, info, died = None, None, None
        if dev_sup:
            canon = AB.canonical(c.lemma)
            if canon in admitted or canon in known:
                rep["rejected"].append({"seed": tag, "why": "already known" if canon in known else "the same lemma as another seed's"})
                continue
            t2 = time.perf_counter()
            h = AB.evaluate(cards, c.lemma, held_idx, s.held)          # exact, by our judge, on cards the teacher never touched
            timing["verify"] += time.perf_counter() - t2
            if not s.held.admissible(h) and not (want and h["refuted"] == 0 and not h["truncated"] and (h["fires"] == 0 or h["excess"] >= s.held.min_excess)):
                rep["rejected"].append({"seed": tag, "why": "refuted or trivial on the held-out games", "dev": summary(c.ev, s.gates), "held": summary(h, s.held)})
                continue
        if want and (not dev_sup or not s.held.supported(h)):
            # EVIDENCE ON DEMAND: no refutation, too little support: look in the corpus (the same gates, afterwards)
            for rnd in range(2):
                g = evid.gather(c.lemma)
                if g is None:
                    died = "no corpus position where the body binds and the rule names a move, or the bounds are spent"
                    info = {"corpus": len(evid.rows), "binds": 0, "asked": 0, "holds": 0, "refuted": 0}
                    break
                nd, nh = len(cards_all), len(cards)
                big = CE.CardSet.merge([cards_all, g.cards])
                ev_d = AB.evaluate(big, c.lemma, list(dev_here) + [nd + j for j in g.dev], s.gates)
                bigh = CE.CardSet.merge([cards, g.cards])
                h = AB.evaluate(bigh, c.lemma, list(held_idx) + [nh + j for j in g.held], s.held)
                new_ev = AB.evaluate(g.cards, c.lemma, None, s.gates)
                info = {"corpus": len(evid.rows), "binds": g.binds, "asked": g.asked, "holds": new_ev["holds"], "refuted": new_ev["refuted"]}
                if ev_d["refuted"] > 0 and rnd == 0:
                    c2 = AB.generalize(seed, big, list(dev_here) + [nd + j for j in g.dev], s.gates)          # the counter-example joins the development cards: the climb again
                    if c2 is not None:
                        c = c2
                        continue
                c.ev = ev_d
                if ev_d["refuted"] > 0 or h["refuted"] > 0:
                    died = f"a counter-example: {new_ev['refuted']} refutation(s) among the {g.asked} positions asked"
                elif not s.gates.admissible(ev_d):
                    died = "not admissible on the extended development cards"
                elif not s.gates.supported(ev_d):
                    died = "still too little support on the development cards"
                elif not s.held.admissible(h):
                    died = "refuted or trivial on the extended held-out cards"
                elif not s.held.supported(h):
                    died = "still too little support on the held-out cards"
                break
            ev_log.append({"seed": tag, "died": died, **(info or {})})
            say(f"OBS sleep evidence {tag}: corpus {info['corpus']}, body binds at {info['binds']}, asked {info['asked']}, holds {info['holds']}, refuted {info['refuted']}; after: dev "
                f"{summary(c.ev, s.gates)}, held {summary(h, s.held) if h is not None else None}; " + (f"died: {died}" if died else "admitted"))
            if died:
                rep["rejected"].append({"seed": tag, "why": "died after evidence: " + died, "dev": summary(c.ev, s.gates), **({"held": summary(h, s.held)} if h is not None else {})})
                continue
            canon = AB.canonical(c.lemma)
            if canon in admitted or canon in known:
                rep["rejected"].append({"seed": tag, "why": "already known" if canon in known else "the same lemma as another seed's"})
                continue
        dev_ok += 1
        if not s.held.admissible(h):
            rep["rejected"].append({"seed": tag, "why": "refuted or trivial on the held-out games", "dev": summary(c.ev, s.gates), "held": summary(h, s.held)})
            continue
        if not s.held.supported(h):
            rep["rejected"].append({"seed": tag, "why": "too little support on the held-out games", "dev": summary(c.ev, s.gates), "held": summary(h, s.held)})
            continue
        name = AB.name_of(c.lemma)
        lemma = {**c.lemma, "name": name,
                 "stats": {"status": "SAMPLED", "dev": summary(c.ev, s.gates), "held": summary(h, s.held), "depth": s.depth, "judge": rep["judge"]},
                 "meta": {"origin": "sleep", "plan": "78.14", "game": variant, "surprise": {k: x[k] for k in ("id", "played", "reply", "predicted", "seen")} | {"kind": x.get("kind", "loss")},
                          "seed_move": m["name"], "steps": c.steps, "teacher": (bool(tm) or by_source) and bool(getattr(teacher, "training_only", True)),
                          "source": getattr(teacher, "source", "teacher") if (tm or by_source) else None, "gates": s.gates.as_dict(), "held_gates": s.held.as_dict(), **({"vocabulary": s.vocabulary} if s.vocabulary else {}),
                          **({"judge_source": JG.describe(teacher)} if by_source else {}), **({"why": x["why"]} if x.get("why") else {}), **({"evidence": {"asked": info["asked"]}} if info else {})}}
        GS.check_lemma(lemma)
        canon = AB.canonical(c.lemma)
        if prefer is not None and x.get("forced"):
            pending.setdefault(x["id"], []).append((canon, lemma, x, tag))
        else:
            admitted[canon] = lemma
    for fid, lst in pending.items():
        verdicts = [prefer(l, x) for _c, l, x, _t in lst]
        good = [i for i, v in enumerate(verdicts) if v["ok"]]
        for i, ((canon, l, x, tag), v) in enumerate(zip(lst, verdicts)):
            say(f"OBS sleep forced rule {l['name']} (seed {tag}): fires at its seed problem {v['fires']}, names {v['named']}, of which lose the verdict {v['lose']}")
        if good:
            say(f"OBS sleep forced rule choice {fid}: {len(lst)} candidate(s) pass the gates, {len(good)} name only moves that keep the verdict at the problem: admitted those only")
            keep_i = good
        else:
            say(f"OBS sleep forced rule choice {fid}: {len(lst)} candidate(s) pass the gates and NONE names only moves that keep the verdict at the problem: all admitted")
            keep_i = range(len(lst))
        for i in keep_i:
            admitted[lst[i][0]] = lst[i][1]
    if ev_log:
        say(f"OBS sleep evidence total: candidates {len(ev_log)}, asked {evid.new_asks}, admitted {sum(1 for e in ev_log if not e['died'])}, refuted {sum(e['refuted'] for e in ev_log)}")
        rep["evidence"] = ev_log
    f["climbed"], f["supported_dev"], f["verified_held"] = climbed, dev_ok, len(admitted)
    rep["lemmas"] = list(admitted.values())
    if forced_ids:
        reg = rep.get("regret", {})
        fl = []
        for fid in sorted(forced_ids):
            rej = [r for r in rep["rejected"] if r["seed"].startswith(fid + " ")]
            adm = [l["name"] for l in admitted.values() if (l["meta"].get("surprise") or {}).get("id") == fid]
            n_seed = sum(1 for t in seeds if t[0]["id"] == fid)
            if fid not in index:
                stage = "no card (the position is over or gives no facts)"
            elif adm:
                stage = "admitted"
            elif fid in reg.get("tied", []):
                stage = "tied: the exact reading at the card's depth values the move played as best"
            elif fid in reg.get("unconfirmed", []):
                stage = "unconfirmed: the move played is not on the card"
            elif fid in reg.get("cut", []):
                stage = "cut: the card's reading was cut"
            elif n_seed == 0:
                stage = "no regret raised: the rank source grades the move played as its best"
            elif rej:
                stage = "; ".join(sorted({r["why"] for r in rej}))
            else:
                stage = "climbed, not admitted"
            best = max(rej, key=lambda r: ((r.get("held") or {}).get("fires", -1), (r.get("dev") or {}).get("fires", -1)), default=None)
            fl.append({"id": fid, "seeds": n_seed, "confirmed": fid in reg.get("confirmed", []), "admitted": adm, "stage": stage, "best": ({k: best.get(k) for k in ("seed", "dev", "held")} if best else None),
                       "climbed": n_seed - sum(1 for r in rej if r["why"].startswith("the ground seed")), "supported": n_seed - sum(1 for r in rej if r["why"].startswith(("the ground seed", "too little support on the dev")))})
        rep["forced"] = fl
        f["forced"] = len(fl)
        say(f"OBS sleep forced seeds: given {len(fl)}, confirmed {sum(1 for x in fl if x['confirmed'])}, climbed {sum(1 for x in fl if x['climbed'] > 0)}, supported on dev "
            f"{sum(1 for x in fl if x['supported'] > 0)}, verified held {sum(1 for x in fl if x['admitted'])}")
        for x in fl:
            if not x["admitted"]:
                b = x["best"]
                say(f"OBS sleep forced seed {x['id']} died: {x['stage']}" + (f"; best candidate {b['seed'].split(' ', 1)[1]}: dev {b['dev']}, held {b['held']}" if b else ""))
    if write and admitted:
        for l in admitted.values():
            GS.write_lemma(game, l)
        rep["snapshot"] = GS.publish(game, teacher_used=any(l["meta"]["teacher"] for l in admitted.values()), vocabulary=s.vocabulary,
                                     judge=GS.judge_of(game, s.judge) if s.judge else "store")      # (plan 78.32: the next wake plays by the record the sleep learned by)
    rep["seconds"] = round(time.perf_counter() - t0, 2)
    rep["timing"] = {k: round(v, 2) for k, v in timing.items()}
    say("OBS sleep time " + game + ": " + ", ".join(f"{k} {v} s" for k, v in rep["timing"].items()) + f", total {rep['seconds']} s")
    say("OBS sleep funnel " + game + f" (judge {rep['judge']}, depth {s.depth}): " + ", ".join(f"{k} {v}" for k, v in f.items()) + f", {rep['seconds']} s")
    for l in rep["lemmas"]:
        say(f"OBS lemma {l['name']}: dev {l['stats']['dev']['fires']} fires / precision {l['stats']['dev']['precision']} (lower {l['stats']['dev']['lower']}); held {l['stats']['held']['fires']} fires in "
            f"{l['stats']['held']['groups']} games / precision {l['stats']['held']['precision']} (lower {l['stats']['held']['lower']}) excess {l['stats']['held']['excess']}")
    return rep


def credit(game: str, store: str, positions: list, *, seconds: float = 0.2, threads: int = 0, say=print) -> dict:
    """CREDIT BY REMOVAL (the architecture document's ablation): every lemma of the snapshot at `store` is left out in turn and the same positions are decided again.
    -> {lemma: {fired, moved, depth_with, depth_without, nodes_with, nodes_without}}: on the positions where the lemma FIRED, how often the decision changed, and the mean
    depth the ledger search reached and the nodes it spent with the lemma and without. The clock is wall time and the machine is shared, so a difference is evidence
    only over many positions: the standard error of the depth difference is reported with it. `positions` is [(variant, fen or None, moves)]."""
    import math
    import os
    import tempfile
    specs = GS.spec_texts(game)
    names = CE.snapshot_info(store)["lemmas"]

    def run(ablate: list) -> list:
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "credit.jsonl")
            CE.solve_batch(positions, seconds, store=store, specs=specs, trace_path=log, threads=threads, ablate=ablate)
            return [json.loads(x) for x in open(log) if x.strip()]

    def facts(r) -> dict:
        return {f["args"][0]: (int(f["args"][1]), int(f["args"][2])) for f in r["facts"] if f["relation"] == "fired"}

    def ledger(r) -> "dict | None":
        return next((s for s in r["steps"] if s["operator"] == "ledger"), None)

    base = run([])
    out = {}
    for name in names:
        without = run([name])
        fired = [i for i, r in enumerate(base) if name in facts(r)]
        moved = sum(1 for i in fired if base[i]["move"] != without[i]["move"])
        dw, do, nw, no, diffs = [], [], [], [], []
        for i in fired:
            a, b = ledger(base[i]), ledger(without[i])
            if a is None or b is None or a["depth"] is None or b["depth"] is None:
                continue
            dw.append(a["depth"]); do.append(b["depth"]); nw.append(a["nodes"]); no.append(b["nodes"]); diffs.append(a["depth"] - b["depth"])
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
        se = None
        if len(diffs) > 1:
            m = sum(diffs) / len(diffs)
            se = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs))
        out[name] = {"fired": len(fired), "moved": moved, "depth_with": mean(dw), "depth_without": mean(do), "nodes_with": mean(nw), "nodes_without": mean(no),
                     "depth_difference": mean(diffs), "depth_difference_se": se, "positions_compared": len(diffs)}
        say(f"OBS credit {name}: fired on {len(fired)} of {len(positions)} positions, the decision changed on {moved}, depth {out[name]['depth_with']} with / {out[name]['depth_without']} without "
            f"(difference {out[name]['depth_difference']}, se {se}, over {len(diffs)})")
    return out


class UciTeacher:
    """A teacher over an outside UCI engine (Stockfish; Fairy-Stockfish for the variants): TRAINING ONLY -- it labels positions in sleep, nothing of a decision calls it.
    `__call__(variant, moves)` -> the engine's move at the position after `moves` from the start, or None."""
    training_only = True
    source = "uci"
    MULTIPV_CAP = 64
    mate_bucket = 1                   # plan 78.24: plies; a mate's distance is floored to a multiple of this, so mates within one bucket tie (1: exact)
    gap = 100                         # the regret detector's threshold, in THIS source's units (centipawns: a pawn); a mate is +-MATE_SCORE so a missed win always counts

    grade_mode = "score"              # plan 78.31: "score" (below) or "result": a score is a RESULT class for the mover, 1 win / 0 draw / -1 loss
    decisive = 300                    # (result mode) a score of at least this many centipawns is a win, at most minus this a loss; a mate is one too. 300 is read off Stockfish with SyzygyPath at
                                      # three units (a tablebase win comes back as ~+500 for its best move, drawn moves 0, a lost one -300 or less; a mate as +-99990): stated, not fitted to outcomes

    @property
    def regret_by_grade(self) -> bool:
        """In result mode a regret is a move whose class is below the best class (detectors.Regret), not a score gap."""
        return self.grade_mode == "result"

    def grade(self, score: int) -> int:
        """Plan 78.23: how a score becomes a card value when this source is the judge. In result mode (plan 78.31) the class of the result: 1, 0 or -1 (a mate, or a score beyond `decisive`,
        is a win or a loss; every other move is a draw), so all winning moves tie as best and all drawing moves tie. Otherwise: a mate keeps its exact value (a faster win is strictly better, a slower loss strictly better);
        any other score is `score // gap`, so moves within a gap share a value (and the optimal moves are those within the gap of the best bucket)."""
        if self.grade_mode == "result":
            return 1 if score >= self.decisive or score > MATE_SCORE - 1000 else -1 if score <= -self.decisive or score < -(MATE_SCORE - 1000) else 0
        if abs(score) > MATE_SCORE - 1000:
            b = max(1, int(self.mate_bucket))
            d = (MATE_SCORE - abs(int(score))) // b * b          # (flooring is monotone: a faster win is never below a slower one, a slower loss never below a faster one)
            return (MATE_SCORE - d) * (1 if score > 0 else -1)
        return int(score) // self.gap

    def rank(self, variant: str, moves: list, cap: "int | None" = None, start: "str | None" = None) -> list:
        """-> [(move, score)] best first for EVERY legal move (see `_analyse`, which does the one engine call `lines` shares)."""
        return [(m, sc) for m, sc, _ in self._analyse(variant, moves, cap, start)]

    def lines(self, variant: str, moves: list, cap: "int | None" = None, start: "str | None" = None) -> dict:
        """Plan 78.23 D: -> {move: [move, m2, m3, ...]}: the engine's own principal line after each legal move (starting with it), from the SAME analysis `rank` ran
        (the last analysis is kept on the instance, `_last`, keyed by position, cap and start: one engine call serves both)."""
        return {m: line for m, _sc, line in self._analyse(variant, moves, cap, start)}

    def _analyse(self, variant: str, moves: list, cap: "int | None" = None, start: "str | None" = None) -> list:
        """-> [(move, score, principal line)] best first for EVERY legal move (MultiPV up to `cap`, default MULTIPV_CAP), scores from the mover's view; a mate is +-MATE_SCORE less its
        distance. Through python-chess's engine wrapper when it imports and knows the variant (the one place chess appears: the boundary returns the generic shape);
        else through the raw UCI pipe and `parse_multipv`. `start` (plan 78.22) is a position (a FEN) the moves are played from, None for the game's own start. Only the order
        is used by the callers, except the regret detector's `gap`."""
        cap = cap or self.MULTIPV_CAP
        key = (variant, tuple(moves), start, cap)
        last = getattr(self, "_last", None)
        if last is not None and last[0] == key:
            return last[1]
        out = self._analyse_uncached(variant, moves, cap, start)
        self._last = (key, out)
        return out

    def _analyse_uncached(self, variant: str, moves: list, cap: int, start: "str | None") -> list:
        try:
            import chess
            import chess.engine
            import chess.variant
        except ImportError:
            return self._rank_raw(variant, moves, cap, start)
        try:
            board = (chess.variant.find_variant(variant)(start) if start else chess.variant.find_variant(variant)()) if variant != "chess" else (chess.Board(start) if start else chess.Board())
            for m in moves:
                board.push_uci(m)
        except ValueError:
            return self._rank_raw(variant, moves, cap, start)
        n = min(cap, board.legal_moves.count())
        if n == 0:
            return []
        if self._engine is None:
            import chess.engine as ce
            self._engine = ce.SimpleEngine.popen_uci(self.path)
        infos = self._engine.analyse(board, chess.engine.Limit(time=self.movetime / 1000.0), multipv=n)
        out = []
        for info in infos if isinstance(infos, list) else [infos]:
            pv = info.get("pv")
            if pv:
                out.append((pv[0].uci(), info["score"].relative.score(mate_score=MATE_SCORE), [x.uci() for x in pv]))
        return out

    def _rank_raw(self, variant: str, moves: list, cap: int, start: "str | None" = None) -> list:
        """The fallback: the engine's own pipe, MultiPV set, `info` lines parsed by `parse_multipv`."""
        self._send(f"setoption name MultiPV value {cap}")
        self._send(self._position(moves, start))
        self._send(f"go movetime {self.movetime}")
        lines = []
        for line in self.p.stdout:
            lines.append(line)
            if line.startswith("bestmove"):
                break
        self._send("setoption name MultiPV value 1")
        return parse_multipv_lines("".join(lines))

    def __init__(self, path: str, movetime_ms: int = 100, options: "dict | None" = None, mate_bucket: int = 1, grade_mode: str = "score", decisive: "int | None" = None):
        import subprocess
        if grade_mode not in ("score", "result"):
            raise ValueError(f"grade_mode is 'score' or 'result', not {grade_mode!r}")
        self.grade_mode = grade_mode
        if decisive is not None:
            self.decisive = int(decisive)
        self.movetime = movetime_ms
        self.mate_bucket = mate_bucket
        self.path = path
        self._engine = None
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._send("uci")
        self._until("uciok")
        for k, v in (options or {}).items():
            self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._until("readyok")

    def _send(self, s: str) -> None:
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _until(self, word: str) -> str:
        for line in self.p.stdout:
            if line.startswith(word):
                return line.strip()
        raise RuntimeError("the engine closed its output")

    @staticmethod
    def _position(moves: list, start: "str | None" = None) -> str:
        return ("position fen " + start if start else "position startpos") + (" moves " + " ".join(moves) if moves else "")

    def __call__(self, variant: str, moves: list, start: "str | None" = None) -> "str | None":
        self._send(self._position(moves, start))
        self._send(f"go movetime {self.movetime}")
        m = self._until("bestmove").split()
        return m[1] if len(m) > 1 and m[1] != "(none)" else None

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
        try:
            self._send("quit")
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


def main(argv: list) -> int:
    from catspace.incubator.analyst.discovery import wake
    from catspace.io import paths
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--log", default=None, help="a wake log; without it, --cycles of wake then sleep")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=0.2)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--judge", default=None, help="the store's VALUATION record the engine plays and the cards are valued by (default: the record in use)")
    ap.add_argument("--teacher", default=None, help="a UCI engine to label with (training only)")
    ap.add_argument("--teacher-mode", default="label", choices=["label", "filter", "break"])
    ap.add_argument("--teacher-surprises", action="store_true", help="register the rank-based surprises (training only)")
    ap.add_argument("--source", default=None, help="a rank source from the inference kit in place of a teacher: `mcts` (the plain estimate) or `mcts:<rollout>/<leaf>` "
                    "or `human[:<path>]` (the human-games index, training only, plan 78.19 E); (plan 78.18: rollout uniform | greedy | gain | lemma, leaf draw | balance; the store the sleep reads gives the judge and the lemmas)")
    ap.add_argument("--teacher-ms", type=int, default=100, help="the UCI teacher's time per position, in milliseconds")
    ap.add_argument("--teacher-option", action="append", default=[], metavar="NAME=VALUE", help="a UCI option for the teacher (repeatable), e.g. SyzygyPath=/path")
    ap.add_argument("--source-seconds", type=float, default=0.2)
    ap.add_argument("--judge-by", default="store", choices=["store", "source"], help="plan 78.23: `store` values the cards by the store's record; `source` revalues them by the rank source given "
                    "(--teacher or --source), which is then the judge of the lemmas (training only, recorded as teacher-judged)")
    ap.add_argument("--judge-positions", type=int, default=None, help="with --judge-by source: revalue at most this many cards")
    ap.add_argument("--vocabulary", default=None, help="plan 78.24: the lemma vocabulary's name (rules/learned/<game>/vocabulary.<name>.json); default: the game's own")
    ap.add_argument("--gate", action="append", default=[], metavar="NAME=VALUE", help="override a field of BOTH the development and the held-out gates (repeatable), e.g. min_fires=2")
    ap.add_argument("--held-gate", action="append", default=[], metavar="NAME=VALUE", help="override a field of the held-out gates only (repeatable)")
    ap.add_argument("--teacher-mate-bucket", type=int, default=1, help="plies: the UCI teacher's mate distances are floored to a multiple of this, so mates within a bucket tie (1: exact)")
    ap.add_argument("--no-auto", action="store_true", help="do not let the diagnosis choose the tools")
    ap.add_argument("--ties", action="store_true", help="the wake reads the ledger's ties (so the sleep can measure the tie rate)")
    ap.add_argument("--write", action="store_true", help="file the lemmas and publish the next snapshot")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    variant = a.variant or a.game
    if a.judge_by == "source" and not (a.teacher or a.source):
        ap.error("--judge-by source needs a rank source: --teacher or --source")
    explicit = a.teacher_mode != "label" or a.teacher_surprises                      # (the CLI's default mode is the chooser's to override)
    st = Settings(depth=a.depth, stride=a.stride, seed=a.seed, teacher_mode=a.teacher_mode if (a.teacher or a.source) and explicit else "off",
                  teacher_surprises=bool((a.teacher or a.source) and a.teacher_surprises), judge=a.judge, auto_tools=not a.no_auto, judge_by=a.judge_by, judge_positions=a.judge_positions,
                  vocabulary=a.vocabulary, gates=AB.Gates().override(a.gate), held=AB.Gates(min_fires=3, min_groups=2).override(a.gate).override(a.held_gate))
    if (a.judge or a.vocabulary) and not a.log:
        # the wake must play by the judge the sleep learns by: publish the store state with that judge (the same content is the same snapshot; nothing is written twice)
        print(f"OBS snapshot {a.game} {GS.publish(a.game, judge=GS.judge_of(a.game, a.judge) if a.judge else 'store', vocabulary=a.vocabulary)}: judge {a.judge}, vocabulary {a.vocabulary}")
    options = dict(o.partition("=")[::2] for o in a.teacher_option)
    teacher = UciTeacher(a.teacher, a.teacher_ms, options, a.teacher_mate_bucket) if a.teacher else (source_of(a.source, a.source_seconds, a.seed, GS.path_of(a.game, GS.head(a.game)) if GS.head(a.game) else None) if a.source else None)
    try:
        for c in range(a.cycles if not a.log else 1):
            log = a.log
            if not log:
                tag = f"{a.tag or time.strftime('%Y%m%d-%H%M%S')}_c{c}"
                log = paths.experiment(f"wake_{a.game}_{tag}.jsonl")
                head = GS.head(a.game)
                store = str(GS.path_of(a.game, head)) if head else None
                wake.play_self(variant, a.games, seconds=a.seconds, seed=a.seed + c, store=store, specs=GS.spec_texts(a.game, a.vocabulary) if store else None, log=log, run=tag, ties=a.ties)
            rep = sleep(a.game, log, variant=variant, settings=st, teacher=teacher, write=a.write)
            out = paths.experiment(f"sleep_{a.game}_{time.strftime('%Y%m%d-%H%M%S')}_{c}.json")
            json.dump(rep, open(out, "w"), indent=1, default=str)
            print(f"OBS sleep report -> {out}")
    finally:
        if teacher:
            teacher.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
