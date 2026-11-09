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
from catspace.incubator.analyst.discovery import game_store as GS


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
                    "b_depth": bd, "need": need, "moves_before": list(game["moves"][:ply]), "group": f"{run}:{g}", "id": f"{run}:{g}:{ply}"})
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
    report of the wake log; `source`: a rank source (a teacher or an estimate) or None. The one rule so far: when the judge ties often -- the tie rate is above
    `tie_threshold`, or the wake never asked for ties so the rate is unknown -- a rank source is worth its cost: it breaks the ties among saving moves and registers
    the rank-based surprises. Everything chosen and why is returned, so the report says it."""
    ties = diag.get("ties", {})
    asked, rate = bool(ties.get("asked")), ties.get("rate")
    coarse = (not asked) or (rate is not None and rate >= s.tie_threshold)
    why = ("the wake did not read ties: the judge's coarseness is unknown" if not asked else f"tie rate {rate} against the threshold {s.tie_threshold}")
    chosen = {"teacher_mode": s.teacher_mode, "teacher_surprises": s.teacher_surprises}
    if source is None:
        return {"diagnosis": ties, "coarse": coarse, "chosen": chosen, "why": "no rank source given; " + why}
    if coarse:
        chosen = {"teacher_mode": "break", "teacher_surprises": True}
    return {"diagnosis": ties, "coarse": coarse, "source": getattr(source, "source", "teacher"), "training_only": bool(getattr(source, "training_only", True)), "chosen": chosen, "why": why}


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

    def rank(self, variant: str, moves: list, cap: "int | None" = None) -> list:
        opts = dict(self.options)
        if variant in self.specs:
            opts["spec"] = self.specs[variant]
        r = CE.estimate(variant, None, list(moves), seconds=self.seconds, max_nodes=self.max_nodes, seed=self.seed, **opts)
        out = sorted(((name, visits) for name, visits, _mean in r["visits"]), key=lambda x: -x[1])
        return out[:cap] if cap else out

    def __call__(self, variant: str, moves: list) -> "str | None":
        r = self.rank(variant, moves)
        return r[0][0] if r else None

    def close(self) -> None:
        pass


def source_of(name: str, seconds: float = 0.2, seed: int = 1, store: "str | None" = None) -> EstimateSource:
    """An EstimateSource from a name: `mcts` (uniform playouts scored at the game's end: the plain search) or `mcts:<rollout>/<leaf>` (plan 78.18); `store` (a snapshot
    path) gives the balance leaf its judge and the lemma prior its lemmas."""
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
    judge: "str | None" = None       # the store's VALUATION record the cards are valued by (None: the record in use); never values written in code
    gates: AB.Gates = field(default_factory=AB.Gates)
    held: AB.Gates = field(default_factory=lambda: AB.Gates(min_fires=3, min_groups=2))


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


def teacher_seeds(cards, dev_idx: list, moves_of: dict, teacher, variant: str, spec: dict, vocab: str, judge: dict, s: "Settings", rep: dict) -> tuple:
    """The rank-based surprises on the DEVELOPMENT cards: each position the source ranks whose ordering disagrees with the card's is a surprise (`disagreement`); one is
    CONFIRMED when a card of the same position one `+2` deeper (ours, exact) finds the source's top move optimal, and only then is a seed made of it (a move our own deeper
    reading cannot confirm is not seeded). -> (seeds [(x, card, move, tm, played, dev indices)], the deeper CardSet or None). Records rep["teacher"]."""
    t = rep["teacher"] = {"source": getattr(teacher, "source", "teacher"), "ranked": 0, "surprises": 0, "confirmed": 0, "unconfirmed": [], "cut": [], "deeper_depth": s.depth + 2}
    stride = max(1, len(dev_idx) // max(1, s.teacher_positions))
    found = []
    for i in dev_idx[::stride][:s.teacher_positions]:
        card = json.loads(cards.card(i))
        if card["cut"] or card["id"] not in moves_of:
            continue
        ranking = teacher.rank(variant, moves_of[card["id"]])
        t["ranked"] += 1
        d = disagreement(card, ranking, t["source"], s.teacher_top_k)
        if d is not None:
            found.append((i, card, d))
    t["surprises"] = len(found)
    if not found:
        return [], None
    items = [(variant, None, moves_of[c["id"]], c["id"], c["group"]) for _, c, _ in found]
    deep = CE.build_cards(items, spec, vocab, s.depth + 2, s.fuse, s.threads, judge.get("worth", {}), bool(judge.get("ours_first", False)))
    seeds = []
    for k, (i, card, d) in enumerate(found):
        dc = json.loads(deep.card(k))
        top = next((m for m in dc["moves"] if m["name"] == d["source_top"]), None)
        if top is None or dc["cut"]:
            t["cut" if dc["cut"] else "unconfirmed"].append(card["id"])
            continue
        if top["value"] != max(m["value"] for m in dc["moves"]):
            t["unconfirmed"].append(card["id"])
            continue
        t["confirmed"] += 1
        ours = max(dc["moves"], key=lambda m: (m["value"], -m["lost"]))
        x = {"id": card["id"], "group": card["group"], "played": ours["name"], "reply": d["source_top"], "predicted": 0, "seen": 0, "kind": "teacher", "disagreement": d}
        seeds.append((x, dc, top, d["source_top"], ours, ("deep", k, i)))
    # the dev indices of a seed: the development cards with its own position's card replaced by the deeper one (appended after the depth-`depth` cards)
    n = len(cards)
    seeds = [(x, dc, top, tm, ours, [j for j in dev_idx if j != i] + [n + k]) for (x, dc, top, tm, ours, (_, k, i)) in seeds]
    return seeds, deep


def sleep(game: str, log: str, *, variant: "str | None" = None, settings: "Settings | None" = None, teacher=None, write: bool = False, say=print) -> dict:
    """One sleep over a wake log -> the report (the funnel, the lemmas admitted, what was rejected and why). `teacher(variant, moves) -> uci or None` is optional
    and training-only; `write` files the lemmas and publishes the next snapshot."""
    s = settings or Settings()
    variant = variant or game
    t0 = time.perf_counter()
    decisions, games = read_log(log)
    spec, vocab = GS.spec_texts(game), json.dumps(GS.vocabulary(game))
    judge = GS.judge_of(game, s.judge) or {}
    rep = {"game": game, "log": log, "settings": {**s.__dict__, "gates": s.gates.as_dict(), "held": s.held.as_dict()}, "judge": judge.get("name", "count"), "funnel": {}, "lemmas": [], "rejected": []}
    # the sleep reads the engine's own account of the wake and chooses its tools by it (plan 78.17); what the caller set explicitly stands
    from catspace.incubator.analyst.discovery import diagnose as DG
    diag = DG.diagnose(log, min_units=s.min_units, say=lambda *_a, **_k: None)
    rep["tools"] = plan_tools(diag, teacher, s)
    if s.auto_tools and teacher is not None and s.teacher_mode == "off" and not s.teacher_surprises:
        s = Settings(**{**s.__dict__, **rep["tools"]["chosen"]})
    rep["tools"]["applied"] = {"teacher_mode": s.teacher_mode, "teacher_surprises": s.teacher_surprises}
    f = rep["funnel"]
    f["games"], f["decisions"] = len(games), len(decisions)
    sur_all = surprises(decisions, games, s.min_units)
    f["surprises"] = len(sur_all)
    sur = [x for x in sur_all if x["need"] is not None and x["need"] <= s.depth]          # a replay of `depth` plies can only reproduce a loss it can reach
    f["replayable"] = len(sur)
    # the corpus: every game's decision positions at the stride (capped), then the surprises' positions
    items, seen, moves_of = [], set(), {}
    for (run, g, ply), d in sorted(decisions.items()):
        game_row = games.get((run, g))
        if game_row is None or ply % s.stride != 0 or len(items) >= s.max_cards:
            continue
        cid = f"{run}:{g}:{ply}"
        items.append((variant, None, list(game_row["moves"][:ply]), cid, f"{run}:{g}"))
        seen.add(cid)
        moves_of[cid] = list(game_row["moves"][:ply])
    for x in sur:
        if x["id"] not in seen:
            moves_of[x["id"]] = x["moves_before"]
            items.append((variant, None, x["moves_before"], x["id"], x["group"]))
            seen.add(x["id"])
    t_cards = time.perf_counter()
    cards = CE.build_cards(items, spec, vocab, s.depth, s.fuse, s.threads, judge.get("worth", {}), bool(judge.get("ours_first", False)))
    timing = {"cards": round(time.perf_counter() - t_cards, 2), "climb": 0.0, "verify": 0.0}
    f["cards"] = len(cards)
    ids = [json.loads(cards.card(i))["id"] for i in range(len(cards))]
    groups = [json.loads(cards.card(i))["group"] for i in range(len(cards))]
    index = {cid: i for i, cid in enumerate(ids)}
    dev_idx = [i for i, g in enumerate(groups) if is_dev(g, s)]
    held_idx = [i for i, g in enumerate(groups) if not is_dev(g, s)]
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
                ranking = teacher.rank(variant, x["moves_before"])     #  estimate displaced the one seed that verifies)
                saving = break_ties(saving, ranking)
                if own not in saving[:s.max_saving]:
                    saving = saving[:max(0, s.max_saving - 1)] + [own] + [m for m in saving[max(0, s.max_saving - 1):] if m is not own]
                tm = saving[0]["name"] if any(m == saving[0]["name"] for m, _ in ranking) else None
        elif teacher is not None and s.teacher_mode != "off":
            tm = teacher(variant, x["moves_before"])
            if s.teacher_mode == "filter":
                saving = [m for m in saving if m["name"] == tm]
        if not saving:
            continue
        with_saving += 1
        for m in saving[:s.max_saving]:
            seeds.append((x, card, m, tm, played, None))
    f["confirmed"], f["with_saving"], f["seeds"] = confirmed, with_saving, len(seeds)
    # the rank-based surprises (development cards only): the source's ordering against the card; a seed only where our own deeper reading confirms the source's move
    f["teacher_surprises"] = f["teacher_confirmed"] = 0
    deep_cards, cards_all = None, cards
    if teacher is not None and s.teacher_surprises:
        tseeds, deep_cards = teacher_seeds(cards, dev_idx, moves_of, teacher, variant, spec, vocab, judge, s, rep)
        seeds += tseeds
        f["teacher_surprises"], f["teacher_confirmed"] = rep["teacher"]["surprises"], rep["teacher"]["confirmed"]
        f["seeds"] = len(seeds)
        if deep_cards is not None:
            cards_all = CE.CardSet.merge([cards, deep_cards])
    seeds = spread(seeds, lambda t: t[0]["group"])[:s.max_seeds]
    # abstract on the development cards, verify on the held-out cards
    known = {AB.canonical(l) for l in GS.read_lemmas(game)}
    admitted, climbed, dev_ok = {}, 0, 0
    for x, card, m, tm, played, dev_override in seeds:
        seed = AB.seed_from(card, m, s.gates, played)
        t1 = time.perf_counter()
        dev_here = dev_idx if dev_override is None else dev_override          # (a teacher seed's own position is read at the deeper depth that confirmed it)
        c = AB.generalize(seed, cards_all, dev_here, s.gates)
        timing["climb"] += time.perf_counter() - t1
        if c is None:
            ev0 = AB.evaluate(cards_all, seed, dev_here, s.gates)
            rep["rejected"].append({"seed": f"{x['id']} {m['name']}", "why": "the ground seed is not admissible on the development cards", "dev": summary(ev0, s.gates),
                                    "truncated": ev0["truncated"], "body": len(seed["body"])})
            continue
        climbed += 1
        if not s.gates.supported(c.ev):
            rep["rejected"].append({"seed": f"{x['id']} {m['name']}", "why": "too little support on the development cards", "dev": summary(c.ev, s.gates)})
            continue
        dev_ok += 1
        canon = AB.canonical(c.lemma)
        if canon in admitted or canon in known:
            rep["rejected"].append({"seed": f"{x['id']} {m['name']}", "why": "already known" if canon in known else "the same lemma as another seed's"})
            continue
        t2 = time.perf_counter()
        h = AB.evaluate(cards, c.lemma, held_idx, s.held)          # exact, by our judge, on cards the teacher never touched
        timing["verify"] += time.perf_counter() - t2
        if not s.held.admissible(h):
            rep["rejected"].append({"seed": f"{x['id']} {m['name']}", "why": "refuted or trivial on the held-out games", "dev": summary(c.ev, s.gates), "held": summary(h, s.held)})
            continue
        if not s.held.supported(h):
            rep["rejected"].append({"seed": f"{x['id']} {m['name']}", "why": "too little support on the held-out games", "dev": summary(c.ev, s.gates), "held": summary(h, s.held)})
            continue
        name = AB.name_of(c.lemma)
        lemma = {**c.lemma, "name": name,
                 "stats": {"status": "SAMPLED", "dev": summary(c.ev, s.gates), "held": summary(h, s.held), "depth": s.depth, "judge": rep["judge"]},
                 "meta": {"origin": "sleep", "plan": "78.14", "game": variant, "surprise": {k: x[k] for k in ("id", "played", "reply", "predicted", "seen")} | {"kind": x.get("kind", "loss")},
                          "seed_move": m["name"], "steps": c.steps, "teacher": bool(tm) and bool(getattr(teacher, "training_only", True)),
                          "source": getattr(teacher, "source", "teacher") if tm else None, "gates": s.gates.as_dict(), "held_gates": s.held.as_dict()}}
        GS.check_lemma(lemma)
        admitted[canon] = lemma
    f["climbed"], f["supported_dev"], f["verified_held"] = climbed, dev_ok, len(admitted)
    rep["lemmas"] = list(admitted.values())
    if write and admitted:
        for l in admitted.values():
            GS.write_lemma(game, l)
        rep["snapshot"] = GS.publish(game, teacher_used=any(l["meta"]["teacher"] for l in admitted.values()))
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

    def rank(self, variant: str, moves: list, cap: "int | None" = None) -> list:
        """-> [(move, score)] best first for EVERY legal move (MultiPV up to `cap`, default MULTIPV_CAP), scores from the mover's view; a mate is +-MATE_SCORE less its
        distance. Through python-chess's engine wrapper when it imports and knows the variant (the one place chess appears: the boundary returns the generic shape);
        else through the raw UCI pipe and `parse_multipv`. Only the order is used by the callers."""
        cap = cap or self.MULTIPV_CAP
        try:
            import chess
            import chess.engine
            import chess.variant
        except ImportError:
            return self._rank_raw(variant, moves, cap)
        board = chess.variant.find_variant(variant)() if variant != "chess" else chess.Board()
        try:
            for m in moves:
                board.push_uci(m)
        except ValueError:
            return self._rank_raw(variant, moves, cap)
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
                out.append((pv[0].uci(), info["score"].relative.score(mate_score=MATE_SCORE)))
        return out

    def _rank_raw(self, variant: str, moves: list, cap: int) -> list:
        """The fallback: the engine's own pipe, MultiPV set, `info` lines parsed by `parse_multipv`."""
        self._send(f"setoption name MultiPV value {cap}")
        self._send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        self._send(f"go movetime {self.movetime}")
        lines = []
        for line in self.p.stdout:
            lines.append(line)
            if line.startswith("bestmove"):
                break
        self._send("setoption name MultiPV value 1")
        return parse_multipv("".join(lines))

    def __init__(self, path: str, movetime_ms: int = 100, options: "dict | None" = None):
        import subprocess
        self.movetime = movetime_ms
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

    def __call__(self, variant: str, moves: list) -> "str | None":
        self._send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
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
                    "(plan 78.18: rollout uniform | greedy | gain | lemma, leaf draw | balance; the store the sleep reads gives the judge and the lemmas)")
    ap.add_argument("--source-seconds", type=float, default=0.2)
    ap.add_argument("--no-auto", action="store_true", help="do not let the diagnosis choose the tools")
    ap.add_argument("--ties", action="store_true", help="the wake reads the ledger's ties (so the sleep can measure the tie rate)")
    ap.add_argument("--write", action="store_true", help="file the lemmas and publish the next snapshot")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    variant = a.variant or a.game
    explicit = a.teacher_mode != "label" or a.teacher_surprises                      # (the CLI's default mode is the chooser's to override)
    st = Settings(depth=a.depth, stride=a.stride, seed=a.seed, teacher_mode=a.teacher_mode if (a.teacher or a.source) and explicit else "off",
                  teacher_surprises=bool((a.teacher or a.source) and a.teacher_surprises), judge=a.judge, auto_tools=not a.no_auto)
    if a.judge and not a.log:
        # the wake must play by the judge the sleep learns by: publish the store state with that judge (the same content is the same snapshot; nothing is written twice)
        print(f"OBS snapshot {a.game} {GS.publish(a.game, judge=GS.judge_of(a.game, a.judge))}: judge {a.judge}")
    teacher = UciTeacher(a.teacher) if a.teacher else (source_of(a.source, a.source_seconds, a.seed, GS.path_of(a.game, GS.head(a.game)) if GS.head(a.game) else None) if a.source else None)
    try:
        for c in range(a.cycles if not a.log else 1):
            log = a.log
            if not log:
                tag = f"{a.tag or time.strftime('%Y%m%d-%H%M%S')}_c{c}"
                log = paths.experiment(f"wake_{a.game}_{tag}.jsonl")
                head = GS.head(a.game)
                store = str(GS.path_of(a.game, head)) if head else None
                wake.play_self(variant, a.games, seconds=a.seconds, seed=a.seed + c, store=store, specs=GS.spec_texts(a.game) if store else None, log=log, run=tag, ties=a.ties)
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
