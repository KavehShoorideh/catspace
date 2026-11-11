"""PLAN 78.22 SURPRISE DETECTORS AS MODULES. A surprise is where something the engine did or believed is contradicted; HOW it is found is one detector among several, all of one shape:

    name           recorded on each surprise as its `kind`
    training_only  True when the detector consults anything outside the rules (a teacher, the human games); the sleep marks the lemmas it seeds
    detect(ctx)    -> [surprise dict]; one common shape: id, group, start, moves_before, played, kind, plus the detector's own evidence

THREE DETECTORS. `Hindsight`: the engine's own hindsight from the ledger events (no outside source; sleep.surprises). `Disagreement(source)`: a rank source's ordering
against the card's (sleep.disagreement). `Regret(source)`: the move the engine PLAYED against the source's ranking of all legal moves: a surprise when the source scores the
played move worse than its best by at least the source's OWN declared `gap` (its own units; it never leaves the training kit), else, when it declares none, when the played
move is not among its top `teacher_top_k`. A teacher is one way to get surprise; a game without an oracle uses `Hindsight` or a source of its own (a rank source is anything
with `rank(variant, moves, cap=None, start=None)`).

A rank-based surprise becomes a SEED only after our own exact card two plies deeper (`confirm`): `Disagreement` needs the source's top move optimal there; `Regret` also needs
the played move strictly worse there. Source's top optimal but the played move tied under our judge is OPEN: the judge cannot see what the source sees.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import cs_engine as CE


@dataclass
class Context:
    """What the detectors read. `cards`/`index`/`dev_idx` are filled once the cards are built (Hindsight needs none of them); `positions` {card id: (start, moves)};
    `played` {card id: the move the engine played there}; `rankings` is the shared cache (one source call serves every detector)."""
    variant: str
    settings: object
    decisions: dict
    games: dict
    cards: object = None
    dev_idx: list = field(default_factory=list)
    index: dict = field(default_factory=dict)
    positions: dict = field(default_factory=dict)
    played: dict = field(default_factory=dict)
    rankings: dict = field(default_factory=dict)
    continuations: dict = field(default_factory=dict)       # plan 78.23 D: {(id(source), card id): {move: principal line}} from a source's optional `lines`
    spec: str = ""
    vocab: str = ""
    judge: dict = field(default_factory=dict)
    rep: dict = field(default_factory=dict)
    force_ids: set = field(default_factory=set)             # plan 78.29: card ids of FORCED positions (problems): always shown to the rank source, a regret raised whenever the played move is not best

    def card(self, i: int) -> dict:
        return json.loads(self.cards.card(i))

    def ranking(self, source, cid: str) -> list:
        """The source's ranking of the position of card `cid` (cached by source and id)."""
        key = (id(source), cid)
        if key not in self.rankings:
            start, moves = self.positions[cid]
            self.rankings[key] = ask(source, self.variant, moves, start)
        return self.rankings[key]


def ask(source, variant: str, moves: list, start: "str | None" = None, cap: "int | None" = None) -> list:
    """`source.rank(...)`, with `start=` only when it is not None and `cap=` only when given (a source without the keywords keeps working on ordinary games)."""
    kw = {} if start is None else {"start": start}
    if cap is not None:
        kw["cap"] = cap
    return source.rank(variant, list(moves), **kw)


def _sampled(ctx: Context):
    """The development cards a rank source is asked about (at most `teacher_positions`, evenly spread): (index, card) with a usable position."""
    s = ctx.settings
    stride = max(1, len(ctx.dev_idx) // max(1, s.teacher_positions))
    pick = list(ctx.dev_idx[::stride][:s.teacher_positions])
    if ctx.force_ids:
        have = set(pick)
        pick += [i for i in ctx.dev_idx if i not in have and ctx.card(i)["id"] in ctx.force_ids]          # (a forced position is always shown)
    for i in pick:
        card = ctx.card(i)
        if card["cut"] or card["id"] not in ctx.positions:
            continue
        yield i, card


def _surprise(card: dict, i: int, ctx: Context, kind: str, **evidence) -> dict:
    start, moves = ctx.positions[card["id"]]
    return {"id": card["id"], "group": card["group"], "start": start, "moves_before": list(moves), "played": ctx.played.get(card["id"]), "kind": kind, "card": i, **evidence}


def _deeper_top(sur: dict, dc: dict) -> tuple:
    """-> ("cut" | "unconfirmed" | "optimal", the deeper card's row of the source's top move): the shared half of every rank-based confirmation."""
    top = next((m for m in dc["moves"] if m["name"] == sur["source_top"]), None)
    if top is None or dc["cut"]:
        return ("cut" if dc["cut"] else "unconfirmed"), None
    return ("optimal" if top["value"] == max(m["value"] for m in dc["moves"]) else "unconfirmed"), top


class Hindsight:
    """THE NO-ORACLE METHOD: the engine's own hindsight from the ledger events (sleep.surprises, unchanged)."""
    name = "hindsight"
    training_only = False

    def detect(self, ctx: Context) -> list:
        from catspace.incubator.analyst.discovery import sleep as SL
        return SL.surprises(ctx.decisions, ctx.games, ctx.settings.min_units)


class Disagreement:
    """The source's ordering against the card's (sleep.disagreement) on the development cards. `kind` is "teacher" (the name this kind of surprise always had)."""
    name = "teacher"
    record = "teacher"

    def __init__(self, source):
        self.source = source
        self.training_only = bool(getattr(source, "training_only", True))

    def detect(self, ctx: Context) -> list:
        from catspace.incubator.analyst.discovery import sleep as SL
        s, src = ctx.settings, getattr(self.source, "source", "teacher")
        t = ctx.rep["teacher"] = {"source": src, "ranked": 0, "surprises": 0, "confirmed": 0, "unconfirmed": [], "cut": [], "deeper_depth": s.depth + 2}
        out = []
        for i, card in _sampled(ctx):
            ranking = ctx.ranking(self.source, card["id"])
            t["ranked"] += 1
            d = SL.disagreement(card, ranking, src, s.teacher_top_k)
            if d is not None:
                out.append(_surprise(card, i, ctx, self.name, source=src, disagreement=d, source_top=d["source_top"]))
        t["surprises"] = len(out)
        return out

    def verdict(self, sur: dict, dc: dict, ctx: Context) -> tuple:
        """-> (status, seed parts or None); records the outcome on rep["teacher"]."""
        t = ctx.rep["teacher"]
        status, top = _deeper_top(sur, dc)
        if status != "optimal":
            t[status].append(sur["id"])
            return status, None
        t["confirmed"] += 1
        ours = max(dc["moves"], key=lambda m: (m["value"], -m["lost"]))
        x = {"id": sur["id"], "group": sur["group"], "played": ours["name"], "reply": sur["source_top"], "predicted": 0, "seen": 0, "kind": self.name, "disagreement": sur["disagreement"]}
        return "confirmed", (x, dc, top, sur["source_top"], ours)


class Regret:
    """THE MOVE THE ENGINE PLAYED against the source's ranking of all legal moves (see the module): a surprise when the source scores it worse than its best by at least the
    source's own `gap` (`getattr(source, "gap", None)`, its own units; mates are +-MATE_SCORE so a missed win always counts); a source with no gap: when the move is not among
    its top `teacher_top_k`."""
    name = "regret"
    record = "regret"

    def __init__(self, source):
        self.source = source
        self.training_only = bool(getattr(source, "training_only", True))

    def rule(self, ranking: list, played: str, top_k: int) -> "dict | None":
        """-> None, or the evidence {source_top, top_score, played_score, gap, rule} when `played` is a regret against `ranking` ([(move, score)], best first by score)."""
        order = sorted(ranking, key=lambda x: -x[1])
        if not order or played is None or order[0][0] == played:
            return None
        score = dict(order)
        gap = getattr(self.source, "gap", None)
        ev = {"source_top": order[0][0], "top_score": order[0][1], "played_score": score.get(played), "gap": gap}
        if getattr(self.source, "regret_by_grade", False):
            # plan 78.31, the result standard: a regret is a move whose RESULT class is below the best class
            if played not in score or int(self.source.grade(score[played])) >= int(self.source.grade(order[0][1])):
                return None
            return {**ev, "rule": "result"}
        if gap is not None:
            if played not in score or order[0][1] - score[played] < gap:
                return None                                               # (a move the source did not score cannot be measured in its units)
            return {**ev, "rule": "gap"}
        if played in [m for m, _ in order[:top_k]]:
            return None
        return {**ev, "rule": "order"}

    def detect(self, ctx: Context) -> list:
        s = ctx.settings
        src = getattr(self.source, "source", "teacher")
        r = ctx.rep["regret"] = {"source": src, "ranked": 0, "surprises": 0, "confirmed": [], "tied": [], "unconfirmed": [], "cut": [], "open": [], "deeper_depth": s.depth + 2}
        out = []
        for i, card in _sampled(ctx):
            played = ctx.played.get(card["id"])
            legal = {m["name"] for m in card["moves"]}
            ranking = [(m, sc) for m, sc in ctx.ranking(self.source, card["id"]) if m in legal]
            if not ranking or played not in legal:
                continue
            r["ranked"] += 1
            ev = self.rule(ranking, played, s.teacher_top_k)
            if ev is None and card["id"] in ctx.force_ids:
                # a forced position: a regret whenever the source grades the move played below its best (no gap, no top-k: the position is a problem already)
                grade = getattr(self.source, "grade", None)
                grade = grade if callable(grade) else (lambda x: int(round(x)))
                order = sorted(ranking, key=lambda x: -x[1])
                sc = dict(order)
                if order[0][0] != played and grade(sc[played]) < grade(order[0][1]):
                    ev = {"source_top": order[0][0], "top_score": order[0][1], "played_score": sc[played], "gap": getattr(self.source, "gap", None), "rule": "forced"}
            if ev is not None:
                out.append(_surprise(card, i, ctx, self.name, source=src, regret=ev, source_top=ev["source_top"]))
        r["surprises"] = len(out)
        return out

    def verdict(self, sur: dict, dc: dict, ctx: Context) -> tuple:
        r = ctx.rep["regret"]
        status, top = _deeper_top(sur, dc)
        if status != "optimal":
            r[status].append(sur["id"])
            return status, None
        p = next((m for m in dc["moves"] if m["name"] == sur["played"]), None)
        if p is None:
            r["unconfirmed"].append(sur["id"])
            return "unconfirmed", None
        ev = sur["regret"]
        if p["value"] < top["value"]:
            r["confirmed"].append(sur["id"])
            x = {"id": sur["id"], "group": sur["group"], "played": sur["played"], "reply": sur["source_top"], "predicted": 0, "seen": 0, "kind": self.name, "regret": ev}
            return "confirmed", (x, dc, top, sur["source_top"], p)
        r["tied"].append(sur["id"])
        r["open"].append({"id": sur["id"], "played": sur["played"], "source_top": sur["source_top"], "played_score": ev["played_score"], "top_score": ev["top_score"],
                          "gap": ev["gap"], "rule": ev["rule"], "source": sur["source"]})
        return "tied", None


def confirm(ctx: Context, found: list) -> tuple:
    """THE SHARED CONFIRMATION of the rank-based surprises. `found` [(detector, surprise)]: one card per distinct position, `depth + 2` (ours, exact), then each detector's
    `verdict`. -> (seeds [(x, card, move, source move, played, dev indices)], the deeper CardSet or None). A seed's dev indices are the development cards with its own
    position's card replaced by the deeper one (appended after the depth-`depth` cards)."""
    if not found:
        return [], None
    s = ctx.settings
    slot, items = {}, []
    for _det, sur in found:
        if sur["id"] not in slot:
            slot[sur["id"]] = len(items)
            items.append((ctx.variant, sur["start"], sur["moves_before"], sur["id"], sur["group"]))
    deep = CE.build_cards(items, ctx.spec, ctx.vocab, s.depth + 2, s.fuse, s.threads, ctx.judge.get("worth", {}), bool(ctx.judge.get("ours_first", False)))
    n, seeds = len(ctx.cards), []
    for det, sur in found:
        k = slot[sur["id"]]
        status, parts = det.verdict(sur, json.loads(deep.card(k)), ctx)
        if parts is not None:
            x, dc, top, tm, played = parts
            seeds.append((x, dc, top, tm, played, [j for j in ctx.dev_idx if j != sur["card"]] + [n + k]))
    return seeds, deep


def run(ctx: Context, detectors: list) -> tuple:
    """Run the rank-based detectors of the list (not `Hindsight`, which needs no cards) -> (seeds, deeper CardSet or None); the reports are on ctx.rep."""
    found = [(d, sur) for d in detectors if hasattr(d, "verdict") for sur in d.detect(ctx)]
    return confirm(ctx, found)


REGISTRY = {"hindsight": Hindsight, "disagreement": Disagreement, "teacher": Disagreement, "regret": Regret}


def detector_of(name: str, source=None):
    """The detector called `name` (hindsight | disagreement | regret); the rank-based ones need a `source`."""
    if name not in REGISTRY:
        raise ValueError(f"no detector {name!r}; known: {sorted(REGISTRY)}")
    cls = REGISTRY[name]
    if cls is Hindsight:
        return cls()
    if source is None:
        raise ValueError(f"the {name} detector needs a rank source")
    return cls(source)
