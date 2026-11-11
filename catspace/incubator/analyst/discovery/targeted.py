"""PLAN 78.29 THE TARGETED MINER: ONE PROBLEM AT A TIME, THE SIMPLEST RULE, DRILLED, ADMITTED ONLY IF THE PROBLEM GOES AWAY.

A PROBLEM (playout_sleep) is a position where the engine's own decision plays a move after which the outside engine's verdict class for the mover is lower. The statistical lemma
sleep (sleep.py) takes the turning positions as a few seeds among dozens and asks a held-out bar of many firings; nothing in it is aimed at one problem. This module is: (October's
way: seed from ONE position, the simplest rule, diagnose where it is wrong, repair, drill with counter-positions; selfmine.diagnose / repair / drill)

  a READ     the facts at the problem, and for every legal move the facts it adds / removes (the cards machinery) and whether it KEEPS the verdict (a judge object: training only).
  b HEAD     the response of a rule in the store's own lemma format: a pattern over what a move adds / removes that moves which keep satisfy and no move which loses the verdict
             satisfies at the problem; the heads that single out the fewest moves first. Constants may be lifted to variables.
  c DRILL    counter-positions: the problem's own game one to three moves before it, the playouts from it (the moves that keep and the moves that lose), random neighbours, the other
             problems, and positions of the wake logs whose facts have the same make-up. The judge says at each whether every move the rule names keeps the verdict.
  d REPAIR   the body starts as one literal true at the problem; where the rule fires and names a move that loses the verdict (a WRONG firing), the fact of the problem (lifted) that
             removes most wrong firings and keeps the problem and most right firings is added to the body; bounded rounds; then literals not needed are dropped.
  e ADMIT    zero wrong firings on the drill AND on a FRESH drill the repair never saw (a wrong firing there goes into the drill and the repair starts again), at least `min_right`
             right firings beyond the problem (a firing at a position where every move keeps says nothing and is not counted), at least `min_fresh` of them on fresh positions, AND the problem's own check (playout_sleep.check: the wake's decision, with a snapshot that holds the rule) passes, AND no problem solved before is broken. The rule is
             written by the same writer as the lemma sleep's (game_store.write_lemma + publish); its meta says it was mined from a problem. No verdict or number of the judge is in it.
  f solve_problems  every open problem, the most frequent family first (the facts' make-up and the effect of the move played), prints the OBS lines and which OTHER open problems
             each rule solved.

WHAT A RULE CAN DO IN THE WAKE (read from rust/engine/decision: `Fixed::ledger` hands the lemma operator's candidates to the ledger search as `prefer`, `Search::root_first`): the proposals
are READ FIRST at the root, in the order of the moves' index. An order, never a verdict: the value the search finds is the value it finds without them. So a rule changes the move
played only among moves the ledger's value leaves tied, and among several named moves the first in the move list is the one that is played. `diagnose_problems` measures exactly that
(the ledger's value of the played move against the keeping moves; an oracle rule that names only the keeping moves; what the decision then does).

Game-free: the game is the variant string, the store's fact vocabulary and the judge / teacher objects passed in; no game object is named here.

  .venv/bin/python -m catspace.incubator.analyst.discovery.targeted chess --store-root DIR --judge-engine UCI [--judge-depth 12] [--judge-cache F] [--judge-option NAME=VALUE ...]
         [--vocabulary NAME] [--seconds 0.2] [--log LOG ...] [--diagnose | --solve] [--teacher UCI --teacher-ms 100 --teacher-cache F] [--deep-depth 20 --deep-option NAME=VALUE ...]
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import itertools
import json
import random
import re
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import abstraction as AB
from catspace.incubator.analyst.discovery import apply_audit as AU
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import judges as JG
from catspace.incubator.analyst.discovery import playout_sleep as PS
from catspace.incubator.analyst.discovery import sleep as SL

PROTECTED = AB.PROTECTED


# ------------------------------------------------------------------------------------------------------------------------------------ a: the read of one position

@dataclass
class Reading:
    """One position read: its start / moves, where it came from (`label`), its group, the class the judge gives it (`before`, for the side to move) and, per legal move, whether
    the move keeps that class (`keep`)."""
    pos: dict
    label: str
    group: str
    before: "str | None"
    keep: dict = field(default_factory=dict)

    @property
    def bad(self) -> list:
        return [m for m, k in self.keep.items() if not k]

    @property
    def good(self) -> list:
        return [m for m, k in self.keep.items() if k]


def pos_key(p: dict) -> tuple:
    return (p.get("start"), tuple(p["moves"]))


class Reader:
    """Reads positions: the facts and every move's effect (cards of depth `depth`, values are never used), and the judge's keep / lose of every legal move (`Verdicts`, remembered)."""

    def __init__(self, variant: str, spec: dict, vocabulary: str, verdicts: "PS.Verdicts", *, depth: int = 1, fuse: int = 2_000_000, threads: int = 0, worth: "dict | None" = None,
                 ours_first: bool = False):
        self.variant, self.spec, self.vocab, self.v = variant, spec, vocabulary, verdicts
        self.depth, self.fuse, self.threads, self.worth, self.ours_first = depth, fuse, threads, worth or {}, ours_first

    def legal(self, p: dict) -> "tuple | None":
        """(mover role index or None, legal moves) -- None for a position whose game is over."""
        _fen, mover, legal, outcome, _key = CE.state(self.variant, p.get("start"), list(p["moves"]))
        if mover is None or outcome is not None or not legal:
            return None
        return mover, list(legal)

    def read(self, p: dict, label: str = "", group: str = "") -> "Reading | None":
        lg = self.legal(p)
        if lg is None:
            return None
        role, _ = self.v.seat(p.get("start"), p["moves"])
        before = self.v.of(p.get("start"), p["moves"])
        keep = {}
        for m in lg[1]:
            after = self.v.after(p.get("start"), list(p["moves"]) + [m], role)
            keep[m] = PS.rank_of(after) >= PS.rank_of(before)
        return Reading(pos={"start": p.get("start"), "moves": list(p["moves"])}, label=label, group=group or label, before=before, keep=keep)

    def cards(self, items: list):
        """items [(start, moves, id, group)] -> the CardSet (cards of positions with no facts or no move are left out: look the ids up with `ids_of`)."""
        return CE.build_cards([(self.variant, s, list(m), i, g) for s, m, i, g in items], self.spec, self.vocab, self.depth, self.fuse, self.threads, self.worth, self.ours_first)


def ids_of(cards) -> list:
    return [json.loads(cards.card(i))["id"] for i in range(len(cards))]


class Drill:
    """The counter-positions of a problem as a card corpus whose move values are the judge's: a LOSING move (one that drops the class) is worth 1, a keeping move 0. Read with the
    engine's own matcher (CardSet.eval): on a card with at least one losing move the lemma's claim 'holds' exactly when a move the rule names LOSES -- a WRONG firing; a card where
    every move keeps cannot be wrong and is TRIVIAL (not counted as a right firing). RIGHT: the rule names at least one move and every move it names keeps the verdict."""

    def __init__(self, reader: Reader, readings: list):
        items = [(r.pos.get("start"), r.pos["moves"], f"d{i}", r.group) for i, r in enumerate(readings)]
        cs = reader.cards(items)
        by = {r_i: r for r_i, r in enumerate(readings)}
        self.readings, vals = [], []
        for i in range(len(cs)):
            c = json.loads(cs.card(i))
            r = by[int(c["id"][1:])]
            self.readings.append(r)
            vals.append({m: (0 if r.keep.get(m, True) else 1) for m in (x["name"] for x in c["moves"])})
        self.cards = cs.revalued(vals) if len(cs) else cs
        self.card_json = [json.loads(self.cards.card(i)) for i in range(len(self.cards))]
        self.has_bad = [any(not r.keep.get(m["name"], True) for m in c["moves"]) for r, c in zip(self.readings, self.card_json)]
        self.index = {pos_key(r.pos): i for i, r in enumerate(self.readings)}

    def __len__(self) -> int:
        return len(self.readings)

    def tally(self, ev: dict) -> dict:
        fires = wrong = right = trivial = 0
        w, r_, t = [], [], []
        for ci, holds, k in ev["cases"]:
            if k == 0:
                continue
            fires += 1
            if not self.has_bad[ci]:
                trivial += 1
                t.append(ci)
            elif holds:
                wrong += 1
                w.append(ci)
            else:
                right += 1
                r_.append(ci)
        return {"fires": fires, "wrong": wrong, "right": right, "trivial": trivial, "wrong_at": w, "right_at": r_, "trivial_at": t, "truncated": ev["truncated"]}

    def read(self, lemma: dict, max_bindings: int = 64) -> dict:
        return self.tally(self.cards.eval(json.dumps(lemma), None, max_bindings, None))

    def read_many(self, lemmas: list, max_bindings: int = 64) -> list:
        if not lemmas:
            return []
        return [self.tally(e) for e in self.cards.eval_many([json.dumps(l) for l in lemmas], None, max_bindings, None)]


# ------------------------------------------------------------------------------------------------------------------------------------ b: the head

def is_const(a: str) -> bool:
    return not a.startswith("?") and not PROTECTED.match(a)


def head_matches(head: dict, removed: list, added: list) -> bool:
    """Does a move (its removed / added facts as [relation, args...]) satisfy the head ({removed: [(rel, args)], added: [...]}): every literal matches a fact of its side, variables
    consistent (the response's variables are free; the body's binding is not read here, so this is the superset of what the engine matches)."""
    lits = [("r", rel, args) for rel, args in head["removed"]] + [("a", rel, args) for rel, args in head["added"]]

    def go(i: int, b: dict) -> bool:
        if i == len(lits):
            return True
        side, rel, args = lits[i]
        for f in (removed if side == "r" else added):
            if f[0] != rel or len(f) - 1 != len(args):
                continue
            nb, ok = dict(b), True
            for a, v in zip(args, f[1:]):
                if a.startswith("?"):
                    if nb.setdefault(a, v) != v:
                        ok = False
                        break
                elif a != v:
                    ok = False
                    break
            if ok and go(i + 1, nb):
                return True
        return False
    return go(0, {})


def lift_head(head: dict, keep: frozenset = frozenset()) -> "tuple | None":
    """The head with every constant (not us / them / r<n>) except those in `keep` lifted to a variable ?h<n> (the same constant, the same variable). -> (head, {constant: variable})."""
    names: dict = {}

    def f(a: str) -> str:
        if not is_const(a) or a in keep:
            return a
        return names.setdefault(a, f"?h{len(names)}")
    return {"removed": [(rel, [f(a) for a in args]) for rel, args in head["removed"]], "added": [(rel, [f(a) for a in args]) for rel, args in head["added"]]}, names


def head_text(head: dict) -> str:
    lit = lambda l: f"{l[0]}({', '.join(l[1])})"  # noqa: E731
    return f"removes [{', '.join(lit(l) for l in head['removed'])}] adds [{', '.join(lit(l) for l in head['added'])}]"


def heads_for(card: dict, keep: dict, *, max_lits: int = 2, limit: int = 8, variety: bool = True) -> list:
    """The candidate heads at one card (a position as the card builder writes it) and the judge's `keep` {move: bool}: patterns over a move's effect that at least one KEEPING move
    has and no losing move has. Built from the keeping moves' effects: every set of one or two literals, as it stands (ground) and with constants lifted (all of them, or all but one).
    -> [{head, matched (the moves at this position), lits, consts, names}], the fewest matched moves first, then the fewest literals, then the fewest constants. At most `limit`."""
    moves = {m["name"]: m for m in card["moves"]}
    good = {n for n in moves if keep.get(n, True)}
    if not good or len(good) == len(moves):
        return []
    out, seen = [], set()
    for n in sorted(good):
        m = moves[n]
        eff = [("r", f) for f in m["removed"]] + [("a", f) for f in m["added"]]
        for k in range(1, max_lits + 1):
            for sub in itertools.combinations(eff, k):
                ground = {"removed": [(f[0], list(f[1:])) for s, f in sub if s == "r"], "added": [(f[0], list(f[1:])) for s, f in sub if s == "a"]}
                consts = sorted({a for _s, lits in (("r", ground["removed"]), ("a", ground["added"])) for _rel, args in lits for a in args if is_const(a)})
                variants = [(ground, {})]
                full, names = lift_head(ground)
                variants.append((full, names))
                for c in consts:
                    h, nm = lift_head(ground, frozenset({c}))
                    variants.append((h, nm))
                for h, nm in variants:
                    key = json.dumps(h, sort_keys=True)
                    if key in seen:
                        continue
                    seen.add(key)
                    matched = [x for x, mv in moves.items() if head_matches(h, mv["removed"], mv["added"])]
                    if not matched or any(not keep.get(x, True) for x in matched):
                        continue
                    nconst = len({a for lits in (h["removed"], h["added"]) for _rel, args in lits for a in args if is_const(a)})
                    out.append({"head": h, "matched": sorted(matched), "lits": k, "consts": nconst, "names": nm})
    out.sort(key=lambda c: (len(c["matched"]), c["lits"], c["consts"], json.dumps(c["head"], sort_keys=True)))
    # variety: at most two heads of the same literal-relation signature (so one family of heads does not fill the list)
    res, per = [], Counter()
    for c in out:
        sig = tuple(sorted([rel for rel, _a in c["head"]["removed"]] + ["+" + rel for rel, _a in c["head"]["added"]]))
        if variety and per[sig] >= 2:
            continue
        per[sig] += 1
        res.append(c)
        if len(res) >= limit:
            break
    return res


# ------------------------------------------------------------------------------------------------------------------------------------ the lemma and its body

def lemma_of(body: list, head: dict, name: str = "targeted") -> dict:
    """The lemma dict (the store's format) for a body (literals as dicts) and a head."""
    return {"name": name, "body": body, "response": {"removed": [AB.lit(rel, args) for rel, args in head["removed"]], "added": [AB.lit(rel, args) for rel, args in head["added"]]},
            "stats": {}, "meta": {}}


def is_counter(fact: list) -> bool:
    """A move-counter artefact: a fact of one numeric argument (relation(N)), like a half-move clock. It differs from position to position by nothing but the game's age, so it is never
    a body literal."""
    return len(fact) == 2 and fact[1].isdigit()


def body_candidates(facts: list, names: dict, have: set) -> list:
    """Body literals from the facts of the problem position: each fact once with its constants LIFTED -- every constant of the position is one variable everywhere (the head's own
    constants are the head's variables), so two body literals that share a square at the problem share a variable -- and once with the constants kept (the head's still lifted).
    -> [(lit dict, description)], not those already in the body (`have`: json texts)."""
    allmap = dict(names)
    facts = [f for f in facts if not is_counter(f)]
    for f in facts:
        for a in f[1:]:
            if is_const(a) and a not in allmap:
                allmap[a] = f"?k{len(allmap)}"
    out, seen = [], set()
    for f in facts:
        rel, args = f[0], list(f[1:])
        for kind in ("lift", "keep"):
            lit = AB.lit(rel, [(allmap[a] if (kind == "lift" or a in names) else a) if is_const(a) else a for a in args])
            key = json.dumps(lit, sort_keys=True)
            if key in seen or key in have:
                continue
            seen.add(key)
            out.append((lit, f"{rel}({', '.join(lit['args'])})"))
    return out


@dataclass
class Mined:
    ok: bool
    lemma: "dict | None" = None
    reading: "dict | None" = None
    rounds: list = field(default_factory=list)
    why: str = ""
    head: "dict | None" = None


def _score(t: dict, seed_ci: int) -> "tuple | None":
    """Rank of a candidate by its drill tally: it must keep the problem's own firing right; fewer wrong firings first, then more right ones."""
    if t["truncated"] or seed_ci not in t["right_at"]:
        return None
    return (t["wrong"], -t["right"])


def mine_body(drill: Drill, seed_ci: int, facts: list, head: dict, names: dict, *, max_rounds: int = 8, say=print, tag: str = "", wild: "Wild | None" = None) -> Mined:
    """The REPAIR loop for one head: the body starts empty (invalid in the store, so the first step is the best single literal of the problem's facts), then each round adds the fact of
    the problem that, lifted, leaves the problem's own firing right and removes the most wrong firings of the local drill, then the most refutations on the DEVELOPMENT half of the wild
    drill (the audit's standard), preferring literals with fewer ground constants, then more right firings. Stops when both are zero; then drops the literals the zero does not need. Prints
    each round as an OBS line. -> Mined(ok, lemma, reading, rounds)."""
    body: list = []
    rounds: list = []
    cur, wcur = None, None
    for r in range(1, max_rounds + 1):
        have = {json.dumps(l, sort_keys=True) for l in body}
        cands = body_candidates(facts, names, have)
        lemmas = [lemma_of(body + [l], head) for l, _d in cands]
        tallies = drill.read_many(lemmas)
        wts = wild.read_many(lemmas, wild.dev_idx) if wild is not None else [None] * len(lemmas)
        best = None
        for (lit, desc), t, wt in zip(cands, tallies, wts):
            if _score(t, seed_ci) is None or (wt is not None and wt["truncated"]):
                continue
            nconst = len(lit["args"]) - sum(1 for a in lit["args"] if a.startswith("?"))
            key = (t["wrong"], wt["refuted"] if wt else 0, nconst, -(t["right"] + (wt["holds"] if wt else 0)), desc)
            if best is None or key < best[0]:
                best = (key, lit, desc, t, wt)
        if best is None:
            return Mined(False, rounds=rounds, why="no fact of the problem can be added without losing the problem's own firing", head=head)
        _k, lit, desc, t, wt = best
        body = body + [lit]
        cur, wcur = t, wt
        rounds.append({"round": r, "repair": desc, "fires": t["fires"], "right": t["right"], "wrong": t["wrong"], "trivial": t["trivial"], "body": len(body)})
        lm = lemma_of(body, head)
        say(f"OBS targeted {tag} round {r}: rule {AU.render(lm).splitlines()[0]}; local drill: fires {t['fires']}, right {t['right']} (the problem and {max(0, t['right'] - 1)} more), wrong {t['wrong']}, "
            f"trivial {t['trivial']}" + (f"; wild development half: body binds {wt['fires']}, names a move at {wt['named']}, holds {wt['holds']}, refuted {wt['refuted']}, chance {wt['expected']:.3f}" if wt else "")
            + f"; repair: add {desc}")
        if t["wrong"] == 0 and (wt is None or wt["refuted"] == 0):
            break
    if cur is None or cur["wrong"] > 0 or (wcur is not None and wcur["refuted"] > 0):
        return Mined(False, rounds=rounds, why=f"after {len(rounds)} round(s) {cur['wrong'] if cur else '?'} wrong firing(s) remain on the local drill and "
                     f"{wcur['refuted'] if wcur else 0} refutation(s) on the wild development half", head=head)
    # the smallest body: drop every literal the zero does not need (the problem's firing stays right, right firings do not fall)
    changed = True
    while changed and len(body) > 1:
        changed = False
        for i in range(len(body)):
            tr = body[:i] + body[i + 1:]
            if not any(not l["neg"] for l in tr):
                continue
            lm = lemma_of(tr, head)
            t = drill.read(lm)
            wt = wild.read_many([lm], wild.dev_idx)[0] if wild is not None else None
            if t["wrong"] == 0 and seed_ci in t["right_at"] and t["right"] >= cur["right"] and (wt is None or (wt["refuted"] == 0 and wt["holds"] >= wcur["holds"])):
                say(f"OBS targeted {tag} drop: {AU.literal(body[i])} is not needed (wrong {t['wrong']}, right {t['right']})")
                body, cur, wcur, changed = tr, t, wt, True
                break
    return Mined(True, lemma_of(body, head), cur, rounds, head=head)


# ------------------------------------------------------------------------------------------------------------------------------------ c: the drill positions

def rel_profile(facts: list) -> Counter:
    return Counter(f[0] for f in facts)


def drill_positions(reader: Reader, problem: dict, reading: Reading, *, others: list, log_pool: list, seed: int = 1, before_plies: int = 3, playouts: int = 3, playout_plies: int = 6,
                    neighbours: int = 16, similar: int = 24, cap: int = 120, profile: "Counter | None" = None, exclude: "set | None" = None) -> list:
    """The counter-positions of a problem as [(position, label)], the problem first: its own game 1..`before_plies` moves before it; `playouts` careful playouts from it, from the
    positions after each move that keeps and after the move that loses (`playout_plies` plies, every position along them); `neighbours` positions after one to three random legal moves
    from it; the other problems (`others`: {start, moves} rows) and the position before each; the `similar` positions of `log_pool` ({start, moves}) whose relation profile is nearest
    to the problem's (`profile`). Deduplicated, capped at `cap` (the labels in the order above, so the cap cuts the last kind first). Positions whose key is in `exclude` are skipped
    (a fresh drill for a check that the repair never saw)."""
    rng = random.Random(seed)
    start, moves = problem.get("start"), list(problem["moves"])
    out, seen = [], set(exclude or ())

    def add(p: dict, label: str) -> None:
        k = pos_key(p)
        if k not in seen and len(out) < cap:
            seen.add(k)
            out.append(({"start": start if p.get("start") is None else p.get("start"), "moves": list(p["moves"])}, label))
    add({"start": start, "moves": moves}, "problem")
    for k in range(1, before_plies + 1):
        if len(moves) - k >= 0:
            add({"start": start, "moves": moves[:len(moves) - k]}, f"before {k}")
    lead = [m for m in reading.good[:3]] + [m for m in reading.bad[:2]]
    for mv in lead:
        add({"start": start, "moves": moves + [mv]}, "after keeping move" if reading.keep[mv] else "after losing move")
    for j, mv in enumerate([None] + lead):
        base = moves if mv is None else moves + [mv]
        for n in range(playouts if mv is None else max(1, playouts - 1)):
            try:
                g = CE.playout_game(reader.variant, start, base, seed=seed + 31 * j + n, max_plies=playout_plies, rollout="careful", store=None, spec=None, worth={}, valuation="")
            except Exception:
                continue
            for i in range(1, len(g["moves"]) + 1):
                add({"start": start, "moves": base + g["moves"][:i]}, "playout")
    for n in range(neighbours):
        cur = list(moves)
        for _ in range(rng.randint(1, 3)):
            lg = reader.legal({"start": start, "moves": cur})
            if lg is None:
                break
            cur.append(rng.choice(lg[1]))
        add({"start": start, "moves": cur}, "neighbour")
    for o in others:
        add(o, "other problem")
        if len(o["moves"]) >= 1:
            add({"start": o.get("start"), "moves": list(o["moves"])[:-1]}, "other problem, a move before")
    if log_pool and profile is not None and similar:
        scored = []
        for p in log_pool:
            try:
                f = CE.facts(reader.variant, reader.spec[reader.variant] if isinstance(reader.spec, dict) else reader.spec, reader.vocab, p.get("start"), list(p["moves"]), True, False)
            except Exception:
                continue
            pr = rel_profile(f)
            scored.append((sum(abs(pr[k] - profile[k]) for k in set(pr) | set(profile)), pos_key(p), p))
        scored.sort(key=lambda t: (t[0], str(t[1])))
        taken = 0
        for _d, _k, p in scored:
            if taken >= similar:
                break
            if pos_key(p) not in seen:
                taken += 1
                add(p, "log position of the same make-up")
    return out


# ------------------------------------------------------------------------------------------------------------------------------------ the WILD drill

STANDARD = ("the drill's RIGHT: every move the rule names keeps the outside engine's win / draw / loss class at THAT position (judge, all named moves). The audit's CLAIM HELD "
            "(apply_audit.audit) and the lemma sleep's gates (abstraction.Gates on cards valued by the rank source): at a firing the moves the rule names include one of the rank source's "
            "best-graded moves (ANY named move, by the source's graded ranking, not the class). They differ in the source of the verdict (class of the judge / grade of the rank source) and in "
            "the quantifier (all named moves keep / one named move is best). The wild drill below uses the audit's and the gates' own standard, by the same code (CardSet.eval on cards revalued "
            "by the rank source, abstraction.Gates), so a rule admitted here cannot fail the audit by a mismatch of definition.")


class Wild:
    """THE WILD DRILL: positions of the cycle's wake logs, unrelated to any problem (a stride over ALL the games' decisions, `n` of them), as cards valued by the RANK SOURCE (judges.revalue:
    every legal move's value is the source's grade; a card the source cannot rank completely is cut and never counts), split by game into a DEVELOPMENT half the repair reads and a
    HELD-OUT half it never sees (sleep.is_dev, the lemma sleep's own split). A rule is read by the engine's matcher (CardSet.eval): holds when a move it names is among the source's best,
    refuted when not -- the audit's 'claim held'. The gates are the lemma sleep's (Settings().held: abstraction.Gates)."""

    def __init__(self, reader: Reader, logs: list, teacher, *, n: int = 1200, settings: "SL.Settings | None" = None, say=print):
        self.settings = settings or SL.Settings()
        self.gates = self.settings.held
        items, positions = [], {}
        per_log = max(1, n // max(1, len(logs)))
        for lg in logs:
            decisions, games = SL.read_log(lg)
            keys = [k for k in sorted(decisions) if (k[0], k[1]) in games]
            stride = max(1, -(-len(keys) // per_log))          # (rounded up: the last games are read too)
            for run, g, ply in keys[::stride][:per_log]:
                row = games[(run, g)]
                cid = f"w{len(items)}"
                moves = list(row["moves"][:ply])
                items.append((row.get("start"), moves, cid, f"{run}:{g}"))
                positions[cid] = (row.get("start"), moves)
        cards = reader.cards(items)
        self.cards, rep = JG.revalue(cards, positions, teacher, reader.variant)
        self.groups = [json.loads(self.cards.card(i))["group"] for i in range(len(self.cards))]
        self.dev_idx = [i for i, g in enumerate(self.groups) if SL.is_dev(g, self.settings)]
        self.held_idx = [i for i, g in enumerate(self.groups) if not SL.is_dev(g, self.settings)]
        self.rep = rep
        say(f"OBS targeted wild: {len(items)} log positions in {len(set(self.groups))} games (a stride over every game), cards {len(self.cards)}, revalued by the rank source {rep['revalued']} "
            f"(cut {rep['cut']}, asked {rep['asked']}), development {len(self.dev_idx)} / held-out {len(self.held_idx)} cards; chance of a best move among all moves "
            f"{rep['optimal_share']}; gates (the lemma sleep's held gates): {self.gates.as_dict()}")

    def read_many(self, lemmas: list, idx: "list | None" = None) -> list:
        evs = self.cards.eval_many([json.dumps(l) for l in lemmas], idx, self.gates.max_bindings, None) if lemmas else []
        return [{**e, "named": sum(1 for _c, _h, k in e["cases"] if k > 0)} for e in evs]

    def held(self, lemma: dict) -> dict:
        """The lemma on the held-out half: the engine's eval, its summary with the lower bound, the chance level and excess, the share of positions where it names a move, and the
        gates it fails (empty when admitted)."""
        ev = AB.evaluate(self.cards, lemma, self.held_idx, self.gates)
        named = sum(1 for _c, _h, k in ev["cases"] if k > 0)
        sm = SL.summary(ev, self.gates)
        fails = []
        g = self.gates
        if ev["truncated"]:
            fails.append("the match was cut (truncated)")
        if ev["fires"] < g.min_fires:
            fails.append(f"fires {ev['fires']} < {g.min_fires}")
        if ev["groups"] < g.min_groups:
            fails.append(f"in {ev['groups']} game(s) < {g.min_groups}")
        if ev["fires"] and ev["precision"] < g.min_precision:
            fails.append(f"precision {ev['precision']:.3f} < {g.min_precision} ({ev['refuted']} refuted of {ev['holds'] + ev['refuted']})")
        if ev["fires"] and ev["excess"] < g.min_excess:
            fails.append(f"excess over chance {ev['excess']:.3f} < {g.min_excess}")
        if ev["fires"] and ev["selectivity"] > g.max_selectivity:
            fails.append(f"names {ev['selectivity']:.2f} of the legal moves on average > {g.max_selectivity}")
        if ev["fires"] and g.lower(ev) < g.min_lower:
            fails.append(f"precision lower bound {g.lower(ev):.3f} < {g.min_lower}")
        if not ev["fires"]:
            fails.append("it fires on no held-out position")
        n = max(1, len(self.held_idx))
        share = round(named / n, 4)
        if share > 0.3 and fails:
            fails.append(f"it names a move at {share:.0%} of ALL held-out decisions with that precision")
        return {"ev": ev, "summary": sm, "named": named, "share": share, "positions": len(self.held_idx), "fails": fails, "ok": not fails and g.admissible(ev) and g.supported(ev)}


def say_wild(tag: str, w: dict, say) -> None:
    sm = w["summary"]
    say(f"OBS targeted {tag} wild held-out: {w['positions']} positions, the body binds at {sm['fires']} of them in {sm['groups']} games, the rule names a move at {w['named']} ({w['share']:.1%}), holds "
        f"{sm['holds']}, refuted {sm['refuted']}, precision {sm['precision']} (lower bound {sm['lower']}), chance level {sm['expected']}, excess {sm['excess']}, selectivity {sm['selectivity']}; "
        + ("passes the lemma sleep's gates" if w["ok"] else "REJECTED: " + "; ".join(w["fails"])))


# ------------------------------------------------------------------------------------------------------------------------------------ the trial snapshot and the writer

def trial_snapshot(game: str, lemma: dict, directory, vocabulary: "str | None" = None) -> str:
    """The snapshot the store would publish if `lemma` were written now (the head's lemmas plus it, in the writer's order), written under `directory` as <id>.json. -> its path. The
    published snapshot of the same lemma has the same content and so the same id."""
    data = json.loads(GS.build(game, parent=GS.head(game), teacher_used=True, vocabulary=vocabulary))
    data["lemmas"] = sorted([l for l in data["lemmas"] if l["name"] != lemma["name"]] + [lemma], key=lambda l: GS.slug(l["name"]))
    data["teacher_used"] = True
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    p = Path(directory) / f"{CE.content_id(raw)}.json"
    p.write_bytes(raw)
    CE.snapshot_info(str(p))                    # (the engine reads it back; it must hash to its name)
    return str(p)


def provenance(problem: dict, head: dict, rounds: list, drill_n: int, reading: dict, judge, variant: str, vocabulary: "str | None") -> dict:
    """The lemma's meta: where it came from. Counts of the drill only; no verdict and no number of the outside engine."""
    return {"origin": "targeted", "plan": "78.29", "game": variant, "mined_from": "problem", "problem": {"start": problem.get("start"), "moves": len(problem["moves"]), "played": problem.get("played")},
            "teacher": True, "source": getattr(judge, "source", "judge"), "rounds": len(rounds), **({"vocabulary": vocabulary} if vocabulary else {}),
            "drill": {"positions": drill_n, "firings": reading["fires"], "right": reading["right"], "wrong": reading["wrong"]}}


@contextlib.contextmanager
def sandbox(root):
    """Point the store at the sandbox `root` (learned/ and snapshots/ inside it) for the block, restoring the real paths after."""
    from catspace.incubator.analyst.discovery import initialize as IN
    root = Path(root)
    old = GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT
    GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT = IN.inside(root / "learned", root), IN.inside(root / "snapshots", root)
    try:
        yield root
    finally:
        GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT = old


# ------------------------------------------------------------------------------------------------------------------------------------ one problem

def family_key(card: dict, played: "str | None") -> tuple:
    """A cheap family of a problem: the make-up of its facts (relation counts of the relations that are not about distances) and the relations the move played touches."""
    pr = rel_profile(card["facts"])
    mv = next((m for m in card["moves"] if m["name"] == played), None)
    eff = tuple(sorted({f[0] for f in (mv["removed"] + mv["added"])})) if mv else ()
    return (tuple(sorted((k, v) for k, v in pr.items() if k not in ("closer", "halfmove"))), eff)


@dataclass
class Ctx:
    """What solving needs, once per run."""
    game: str
    variant: str
    spec: dict
    vocabulary: "str | None"
    vocab_json: str
    judge: object
    teacher: object
    seconds: float
    wake_kw: dict
    verdicts: "PS.Verdicts"
    reader: Reader
    log_pool: list
    wild: "Wild | None"
    settings: dict
    say: object
    tmp: str


def try_problem(ctx: Ctx, rows: list, i: int, chosen: str, solved_rows: list, tag: str) -> dict:
    """The targeted miner on problem `rows[i]` whose decision chose `chosen`. -> {status 'admitted' | 'failed', why, lemma (admitted), heads_tried, ...}."""
    s, say = ctx.settings, ctx.say
    row = rows[i]
    p = {"start": row.get("start"), "moves": list(row["moves"])}
    rd = ctx.reader.read(p, "problem", "problem")
    if rd is None:
        return {"status": "failed", "why": "the position is over"}
    good, bad = rd.good, rd.bad
    say(f"OBS targeted {tag}: position {len(p['moves'])} plies, class {rd.before}, {len(rd.keep)} legal moves, keep the verdict {sorted(good)}, lose it {sorted(bad)}; the decision chose {chosen}")
    if not good or not bad:
        return {"status": "failed", "why": "no move keeps the verdict" if not good else "every move keeps the verdict"}
    others = [{"start": r.get("start"), "moves": r["moves"]} for j, r in enumerate(rows) if j != i]
    cs = ctx.reader.cards([(p["start"], p["moves"], "p", "problem")])
    card = json.loads(cs.card(0))
    positions = drill_positions(ctx.reader, p, rd, others=others, log_pool=ctx.log_pool, seed=s["seed"], cap=s["drill_cap"], similar=s["similar"], neighbours=s["neighbours"],
                                profile=rel_profile(card["facts"]))
    t0 = time.perf_counter()
    readings = []
    for q, label in positions:
        r = rd if label == "problem" else ctx.reader.read(q, label, f"{label}:{len(readings)}")
        if r is not None:
            readings.append(r)
    drill = Drill(ctx.reader, readings)
    seed_ci = drill.index[pos_key(p)]
    nb = sum(drill.has_bad)
    kinds = Counter(r.label for r in readings)
    say(f"OBS targeted {tag}: drill {len(drill)} positions ({', '.join(f'{k} {v}' for k, v in kinds.items())}), {nb} with a move that loses the verdict, {ctx.verdicts.calls} judge calls so far, {time.perf_counter() - t0:.1f} s")
    heads = heads_for(card, rd.keep, limit=s["max_heads"])
    if not heads:
        return {"status": "failed", "why": "no pattern over the moves' effects separates the keeping moves from the losing ones"}
    say(f"OBS targeted {tag}: {len(heads)} head(s) tried in order, fewest named moves first: " + "; ".join(f"{head_text(h['head'])} names {h['matched']}" for h in heads))
    used = {pos_key(r.pos) for r in readings}
    last_why = ""
    for hi, h in enumerate(heads, 1):
        dev = list(readings)
        m, ver, fresh_used = None, None, set(used)
        for attempt in range(1, s["verify_rounds"] + 1):
            drill = Drill(ctx.reader, dev)
            m = mine_body(drill, drill.index[pos_key(p)], card["facts"], h["head"], h["names"], max_rounds=s["max_rounds"], say=say, tag=f"{tag} head {hi}", wild=ctx.wild)
            if not m.ok:
                break
            # the rule is read on positions the repair never saw (the same kinds, other seeds); a wrong firing there is a counter-example added to the drill, then the repair starts again
            vpos = drill_positions(ctx.reader, p, rd, others=[], log_pool=ctx.log_pool, seed=s["seed"] + 1000 * attempt, cap=s["verify_cap"], similar=s["verify_similar"],
                                   neighbours=s["verify_neighbours"], profile=rel_profile(card["facts"]), exclude=fresh_used, before_plies=0)
            vread = [rd] + [r for r in (ctx.reader.read(q, "fresh " + label, f"fresh:{attempt}:{n}") for n, (q, label) in enumerate(vpos) if label != "problem") if r is not None]
            fresh_used |= {pos_key(r.pos) for r in vread}
            ver_drill = Drill(ctx.reader, vread)
            ver = ver_drill.read(m.lemma)
            say(f"OBS targeted {tag} head {hi} fresh drill {attempt}: {len(ver_drill)} positions never seen by the repair, the rule fires {ver['fires'] - 1 if ver_drill.index.get(pos_key(p)) in ver['right_at'] else ver['fires']} time(s) there, "
                f"right {max(0, ver['right'] - 1)}, wrong {ver['wrong']}, trivial {ver['trivial']}")
            if ver["wrong"] == 0:
                break
            dev += [ver_drill.readings[ci] for ci in ver["wrong_at"]]
            say(f"OBS targeted {tag} head {hi}: {ver['wrong']} counter-example(s) from the fresh drill added; the repair starts again")
            m = None
        if m is None or not m.ok or ver is None or ver["wrong"] > 0:
            last_why = (m.why if (m is not None and not m.ok) else f"wrong firings remain on fresh positions after {s['verify_rounds']} counter-example round(s)")
            say(f"OBS targeted {tag} head {hi}: rejected, {last_why}")
            continue
        wl = None
        if ctx.wild is not None:
            wl = ctx.wild.held(m.lemma)
            say_wild(f"{tag} head {hi}", wl, say)
            if not wl["ok"]:
                last_why = "on the wild held-out half of the log positions: " + "; ".join(wl["fails"])
                say(f"OBS targeted {tag} head {hi}: rejected, {last_why}")
                continue
        beyond = m.reading["right"] - 1
        beyond_ver = max(0, ver["right"] - 1)
        if beyond + beyond_ver < s["min_right"] or beyond_ver < s["min_fresh"]:
            last_why = (f"only {beyond} right firing(s) beyond the problem on the drill and {beyond_ver} on fresh positions (need {s['min_right']} in all and {s['min_fresh']} on fresh positions)")
            say(f"OBS targeted {tag} head {hi}: rejected, {last_why}")
            continue
        lemma = m.lemma
        lemma["name"] = AB.name_of(lemma)
        counts = {"fires": m.reading["fires"] + ver["fires"], "right": m.reading["right"] + beyond_ver, "wrong": 0}
        lemma["meta"] = provenance(row, h["head"], m.rounds, len(dev) + len(vread), counts, ctx.judge, ctx.variant, ctx.vocabulary)
        lemma["stats"] = {"status": "TARGETED", "drill": {"positions": len(dev), "right": m.reading["right"], "wrong": m.reading["wrong"], "trivial": m.reading["trivial"]},
                          "fresh": {"positions": len(vread), "right": beyond_ver, "wrong": ver["wrong"]}}
        if wl is not None:
            lemma["stats"]["wild_held"] = {k: wl["summary"][k] for k in ("fires", "holds", "refuted", "groups", "precision", "lower", "expected", "excess", "selectivity")}
        GS.check_lemma(lemma)
        snap = trial_snapshot(ctx.game, lemma, ctx.tmp, ctx.vocabulary)
        check_rows = [rows[i]] + [r for r in solved_rows if r is not rows[i]]
        res = PS.check([{"start": r.get("start"), "moves": r["moves"]} for r in check_rows], variant=ctx.variant, store=snap, spec=ctx.spec, judge=ctx.judge, seconds=ctx.seconds,
                       wake_kw=ctx.wake_kw, verdicts=ctx.verdicts)
        own = res[0]
        broken = [k for k, c in zip(check_rows[1:], res[1:]) if not c["keeps"]]
        say(f"OBS targeted {tag} head {hi}: check with the rule: the decision chooses {own['move']} (keeps the verdict: {own['keeps']}; {own['before']} -> {own['after']}), steps {own['steps']}; "
            f"problems solved before still kept: {len(check_rows) - 1 - len(broken)} of {len(check_rows) - 1}")
        if not own["keeps"]:
            last_why = f"the problem's own check fails with the rule: the decision still chooses {own['move']} (the rule names {h['matched']} at the problem)"
            continue
        if broken:
            last_why = f"the rule breaks {len(broken)} problem(s) solved before"
            continue
        return {"status": "admitted", "lemma": lemma, "head": h, "reading": {**m.reading, "fires": counts["fires"], "right": counts["right"]}, "drill": len(dev) + len(vread), "heads_tried": hi, "wild": wl}
    return {"status": "failed", "why": last_why or "no head", "heads_tried": len(heads)}


# ------------------------------------------------------------------------------------------------------------------------------------ the loop over the problems

SETTINGS = {"min_right": 3, "min_fresh": 1, "verify_rounds": 3, "verify_cap": 80, "verify_similar": 24, "verify_neighbours": 16, "max_rounds": 8, "max_heads": 6, "drill_cap": 120,
            "similar": 24, "neighbours": 16, "seed": 1, "pool": 400, "wild_n": 1200}


def solve_problems(game: str, problems_path, *, variant: "str | None" = None, store_root=None, judge, teacher=None, logs=(), seconds: float = 0.2, max_rounds: int = 8,
                   vocabulary: "str | None" = None, spec: "dict | None" = None, wake_kw: "dict | None" = None, settings: "dict | None" = None, say=print) -> dict:
    """Solve the open problems of the problem file one at a time with the targeted miner. `store_root`: the sandbox (learned/ and snapshots/ inside it), or None when the store paths are
    already set (initialize.run_size does that). `judge`: the outside engine's verdict object (training only; `wdl(variant, moves, start)`); `teacher` is kept for symmetry with the lemma
    sleep (unused: the keep / lose of a move is the judge's); `logs`: wake logs for positions of the same make-up; `seconds`, `wake_kw`: the wake's own decision settings; `max_rounds`: the
    repair rounds per head. `settings`: wild_n (log positions of the wild drill; 0 goes without), min_right (right firings beyond the problem a rule needs, drill and fresh positions together), min_fresh (of them, on positions the repair never saw), verify_rounds, verify_cap,
    verify_similar, verify_neighbours, max_heads, drill_cap, similar, neighbours, seed, pool.
    `problems_path` may also be the list of rows itself (the cycle's, not yet saved). The problem file is READ, not written. A rule is written to the store (write_lemma + publish) only when admitted. -> {total, attempted, solved, still_open, admitted: [{name, text,
    problem, solved_others, snapshot}], rows: [{index, status, why}], snapshot}."""
    cm = sandbox(store_root) if store_root is not None else contextlib.nullcontext()
    with cm:
        return _solve(game, problems_path, variant or game, judge, teacher, list(logs or []), seconds, max_rounds, vocabulary, spec, wake_kw or {}, {**SETTINGS, **(settings or {}), "max_rounds": max_rounds}, say)


def _solve(game, problems_path, variant, judge, teacher, logs, seconds, max_rounds, vocabulary, spec, wake_kw, settings, say) -> dict:
    t0 = time.perf_counter()
    rows = list(problems_path) if isinstance(problems_path, (list, tuple)) else PS.load_problems(problems_path)
    spec = spec if spec is not None else GS.spec_texts(game, vocabulary)
    vocab_json = json.dumps(GS.vocabulary(game, vocabulary))
    verdicts = PS.Verdicts(judge, variant)
    reader = Reader(variant, spec, vocab_json, verdicts)
    pool = []
    if logs:
        try:
            pool = PS.sample_positions(logs, int(settings["pool"]), seed=int(settings["seed"]))
        except Exception as e:                                     # (a log without positions: the drill has no log positions)
            say(f"OBS targeted: no log positions for the drill ({e})")
    with tempfile.TemporaryDirectory() as tmp:
        wild = None
        if int(settings["wild_n"]) > 0:
            if teacher is None or not logs:
                raise ValueError("the wild drill needs a rank source (teacher) and wake logs; set wild_n=0 to go without (no rule is then held to the audit's standard)")
            say("OBS targeted standard: " + STANDARD)
            wild = Wild(reader, logs, teacher, n=int(settings["wild_n"]), settings=settings.get("sleep_settings"), say=say)
        ctx = Ctx(game, variant, spec, vocabulary, vocab_json, judge, teacher, seconds, wake_kw, verdicts, reader, pool, wild, settings, say, tmp)

        def check_all(snapshot_path: str) -> list:
            return PS.check([{"start": r.get("start"), "moves": r["moves"]} for r in rows], variant=variant, store=snapshot_path, spec=spec, judge=judge, seconds=seconds, wake_kw=wake_kw,
                            verdicts=verdicts)
        head = GS.head(game)
        snap_path = str(GS.path_of(game, head)) if head else None
        res = check_all(snap_path)
        status = ["solved" if c["keeps"] else "open" for c in res]
        chosen = [c["move"] for c in res]
        already = status.count("solved")
        open_idx = [i for i, s in enumerate(status) if s == "open"]
        say(f"OBS targeted: snapshot {head}, {len(rows)} problem(s), the decision already keeps the verdict at {already}, open {len(open_idx)}")
        # families, the most frequent first
        fam = {}
        for i in open_idx:
            cs = reader.cards([(rows[i].get("start"), rows[i]["moves"], "p", "p")])
            fam[i] = family_key(json.loads(cs.card(0)), chosen[i]) if len(cs) else ("none",)
        count = Counter(fam.values())
        order = sorted(open_idx, key=lambda i: (-count[fam[i]], i))
        if open_idx:
            say("OBS targeted families: " + ", ".join(f"family {n} ({c} problem(s))" for n, (_k, c) in enumerate(count.most_common(), 1)))
        attempted = {i: "" for i in open_idx}
        info = {i: {"index": i, "status": status[i], "why": ""} for i in range(len(rows))}
        admitted = []
        for i in order:
            if status[i] != "open":
                continue
            tag = f"problem {i + 1}/{len(rows)}"
            solved_rows = [rows[j] for j in range(len(rows)) if status[j] == "solved"]
            out = try_problem(ctx, rows, i, chosen[i], solved_rows, tag)
            getattr(judge, "save", lambda: None)()                 # (a judge that remembers its answers on disk keeps them after each problem)
            if out["status"] != "admitted":
                info[i]["why"] = out["why"]
                say(f"OBS targeted {tag}: still open, because {out['why']}")
                continue
            lemma = out["lemma"]
            GS.write_lemma(game, lemma)
            sid = GS.publish(game, teacher_used=True, vocabulary=vocabulary)
            snap_path = str(GS.path_of(game, sid))
            res = check_all(snap_path)
            now = ["solved" if c["keeps"] else "open" for c in res]
            newly = [j for j in range(len(rows)) if status[j] == "open" and now[j] == "solved"]
            lost = [j for j in range(len(rows)) if status[j] == "solved" and now[j] == "open"]
            status = now
            chosen = [c["move"] for c in res]
            others = [j + 1 for j in newly if j != i]
            admitted.append({"name": lemma["name"], "text": AU.render(lemma).splitlines()[0], "problem": i, "solved_others": [j for j in newly if j != i], "broke": lost, "snapshot": sid,
                             "drill": out["drill"], "right": out["reading"]["right"], "heads_tried": out["heads_tried"],
                             "wild_held": ({k: out["wild"]["summary"][k] for k in ("fires", "holds", "refuted", "precision", "expected", "excess")} if out.get("wild") else None)})
            for j in newly:
                info[j]["status"] = "solved"
                info[j]["why"] = f"by rule {lemma['name']}"
            say(f"OBS targeted {tag}: admitted {lemma['name']} (snapshot {sid}): {AU.render(lemma).splitlines()[0]}; fires {out['reading']['fires']} on the drill of {out['drill']}, right "
                f"{out['reading']['right']}, wrong {out['reading']['wrong']}; solved {len(newly)} problem(s) in all, the others: {others}; broke {len(lost)}")
        total, solved_n, open_n = len(rows), 0, 0
        solved_n = sum(1 for j in open_idx if status[j] == "solved")
        open_n = sum(1 for s in status if s == "open")
        say(f"OBS targeted problems: total {total}, attempted {len(open_idx)}, solved {solved_n}, still open {open_n}, rules admitted {len(admitted)}")
        for a in admitted:
            w = a.get("wild_held")
            say(f"OBS targeted rule {a['name']}: mined from problem {a['problem'] + 1}, also solved {len(a['solved_others'])} other open problem(s) {[j + 1 for j in a['solved_others']]}"
                + (f"; wild held-out: fires {w['fires']}, right {w['holds']}, precision {w['precision']}, chance {w['expected']}, excess {w['excess']}" if w else ""))
        say(f"OBS targeted time: {time.perf_counter() - t0:.1f} s, judge calls {verdicts.calls}")
    return {"total": len(rows), "attempted": len(open_idx), "solved": solved_n, "still_open": open_n, "admitted": admitted, "rows": [info[i] for i in range(len(rows))],
            "snapshot": GS.head(game), "already_kept": already}


# ------------------------------------------------------------------------------------------------------------------------------------ step 1: the diagnosis

def ground_oracle(card: dict, keep: dict) -> "dict | None":
    """A rule that can fire only at ONE position (its body every fact of it, ground) and names only keeping moves (the head with the fewest named moves, constants as they stand): the
    ceiling of what a rule can do at that position. None if no head."""
    hs = heads_for(card, keep, limit=500, variety=False)
    if not hs:
        return None
    h = next((c for c in hs if not c["names"]), hs[0])
    body = [AB.lit(f[0], list(f[1:])) for f in card["facts"]]
    return {"lemma": lemma_of(body, h["head"], "oracle"), "matched": h["matched"], "head": h["head"]}


def diagnose_problems(game: str, problems_path, *, variant: "str | None" = None, store_root=None, judge, deep_judge=None, vocabulary: "str | None" = None, spec=None,
                      seconds: float = 0.2, wake_kw: "dict | None" = None, ledger_depth: int = 6, say=print) -> dict:
    """STEP 1 of the plan: per problem, why the lemma sleep's rules did not help, from what the decision does with a proposal. Prints `OBS targeted diagnose` lines and returns the rows.
    Per problem: the decision of the current snapshot (move, steps: is the lemma operator silent?), the judge's keep / lose of every move, the ledger's own value of every move at
    `ledger_depth` (cards): does the played move rank above every keeping move (an order of the proposals can then never change the choice) or tie with one; an ORACLE rule (body: every
    fact of the position, head: a pattern of the keeping moves only) in a trial snapshot, and what the decision does with it; and, with `deep_judge` (a deeper judge object), whether the
    played move still drops the class under it (the problem survives) or the position is verdict noise."""
    variant = variant or game
    cm = sandbox(store_root) if store_root is not None else contextlib.nullcontext()
    wake_kw = wake_kw or {}
    with cm:
        rows = PS.load_problems(problems_path)
        spec = spec if spec is not None else GS.spec_texts(game, vocabulary)
        vocab_json = json.dumps(GS.vocabulary(game, vocabulary))
        v = PS.Verdicts(judge, variant)
        dv = PS.Verdicts(deep_judge, variant) if deep_judge is not None else None
        reader = Reader(variant, spec, vocab_json, v)
        jr = GS.judge_of(game)
        head = GS.head(game)
        snap = str(GS.path_of(game, head))
        res = PS.check([{"start": r.get("start"), "moves": r["moves"]} for r in rows], variant=variant, store=snap, spec=spec, judge=judge, seconds=seconds, wake_kw=wake_kw, verdicts=v)
        say(f"OBS targeted diagnose: snapshot {head} with {len(GS.info(game, head)['lemmas'])} lemma(s), {len(rows)} problem(s); the lemma operator was silent at "
            f"{sum(1 for c in res if 'lemma:NoLine' in c['steps'])} of them")
        out = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, (row, c) in enumerate(zip(rows, res)):
                p = {"start": row.get("start"), "moves": list(row["moves"])}
                rd = reader.read(p, "problem")
                tag = f"problem {i + 1}"
                d = {"index": i, "move": c["move"], "keeps_now": c["keeps"]}
                if rd is None or c["move"] is None:
                    say(f"OBS targeted diagnose {tag}: no decision or the game is over")
                    out.append(d)
                    continue
                fen = CE.state(variant, p["start"], p["moves"])[0]
                cs = CE.build_cards([(variant, p["start"], p["moves"], "p", "g")], spec, vocab_json, ledger_depth, 20_000_000, 0, (jr or {}).get("worth", {}), bool((jr or {}).get("ours_first", False)))
                card = json.loads(cs.card(0))
                val = {m["name"]: m["value"] for m in card["moves"]}
                best = max(val.values())
                played_v = val.get(c["move"])
                keep_v = sorted({val[m] for m in rd.good if m in val})
                tie_keep = [m for m in rd.good if val.get(m) == best]
                d.update({"values": val, "best": best, "played_value": played_v, "good": rd.good, "bad": rd.bad, "cut": card["cut"]})
                say(f"OBS targeted diagnose {tag}: {fen}; decision {c['move']} ({'keeps' if c['keeps'] else 'loses'} the verdict {c['before']} -> {c['after']}), steps {c['steps']}; "
                    f"keeping moves {sorted(rd.good)}, losing {sorted(rd.bad)}")
                say(f"OBS targeted diagnose {tag}: the ledger's own value at depth {ledger_depth}{' (cut)' if card['cut'] else ''}: best {best}, the move played {played_v}, keeping moves {keep_v}; "
                    f"a keeping move ties the best: {bool(tie_keep)}; the move played is {'strictly above every keeping move: no order of proposals can change the choice' if played_v is not None and tie_keep == [] and played_v >= best else 'tied with the best'}")
                d["order_can_fix"] = bool(tie_keep)
                orc = ground_oracle(card, rd.keep)
                if orc is None:
                    say(f"OBS targeted diagnose {tag}: no pattern over the moves' effects separates keeping from losing moves")
                else:
                    lem = orc["lemma"]
                    lem["name"] = AB.name_of(lem)
                    snap2 = trial_snapshot(game, lem, tmp, vocabulary)
                    r2 = PS.check([p], variant=variant, store=snap2, spec=spec, judge=judge, seconds=seconds, wake_kw=wake_kw, verdicts=v)[0]
                    d["oracle_keeps"] = r2["keeps"]
                    d["oracle_move"] = r2["move"]
                    say(f"OBS targeted diagnose {tag}: an ORACLE rule that fires only here and names only {orc['matched']} (head {head_text(orc['head'])}): the decision chooses {r2['move']}, keeps the verdict "
                        f"{r2['keeps']}, steps {r2['steps']}")
                if dv is not None:
                    r_deep = dv.of(p["start"], p["moves"])
                    role, _ = dv.seat(p["start"], p["moves"])
                    after = dv.after(p["start"], p["moves"] + [c["move"]], role)
                    survive = PS.rank_of(after) < PS.rank_of(r_deep)
                    keeps_deep = [m for m in rd.keep if PS.rank_of(dv.after(p["start"], p["moves"] + [m], role)) >= PS.rank_of(r_deep)]
                    d["deep"] = {"before": r_deep, "after": after, "survives": survive, "keeping": keeps_deep}
                    say(f"OBS targeted diagnose {tag}: deeper judge: class {r_deep}, after {c['move']} {after}; the problem {'SURVIVES' if survive else 'does NOT survive (verdict noise)'}; "
                        f"moves that keep it {sorted(keeps_deep)}")
                out.append(d)
        n = len(out)
        say(f"OBS targeted diagnose summary: problems {n}, the ledger ties a keeping move with its best at {sum(1 for d in out if d.get('order_can_fix'))}, the oracle rule makes the decision keep the "
            f"verdict at {sum(1 for d in out if d.get('oracle_keeps'))}" + (f", problems that survive the deeper judge {sum(1 for d in out if d.get('deep', {}).get('survives'))}" if dv is not None else ""))
    return {"rows": out}


def diagnose_sleep(game: str, log: str, problems_path, *, store_root, teacher, variant: "str | None" = None, settings=None, say=print) -> dict:
    """What the existing statistical lemma sleep does with the problems: they are fed to it as `extra_positions` (a group each) with a rank source, nothing is written, and the report's
    regret and rejection lists are read per problem: did it become a seed (the regret confirmed) or tied / unconfirmed / cut, did a candidate climb from it, where was it rejected and why."""
    from catspace.incubator.analyst.discovery import sleep as SL
    variant = variant or game
    rows = PS.load_problems(problems_path)
    extra = [{"id": f"problem:{i}:{len(r['moves'])}", "start": r.get("start"), "moves": list(r["moves"]), "played": r["played"], "group": f"problem:{i}"} for i, r in enumerate(rows)]
    with sandbox(store_root):
        rep = SL.sleep(game, log, variant=variant, settings=settings, teacher=teacher, write=False, say=lambda *_a, **_k: None, extra_positions=extra)
    reg = rep.get("regret", {})
    from catspace.incubator.analyst.discovery import detectors as D
    st = settings or SL.Settings()
    per = {}
    for i, e in enumerate(extra):
        k = e["id"]
        dev = SL.is_dev(e["group"], st)
        why = None
        ranking = D.ask(teacher, variant, e["moves"], e["start"]) if teacher is not None else []
        legal = [m for m, _ in ranking]
        ev = D.Regret(teacher).rule(ranking, e["played"], st.teacher_top_k) if ranking else None
        top = max(ranking, key=lambda x: x[1]) if ranking else None
        sc = dict(ranking)
        stage = ("held-out half of the games: never a seed (seeds come from development cards only)" if not dev else
                 "seed (regret confirmed by the exact reading)" if k in reg.get("confirmed", []) else "tied under the exact reading (the move played is as good as the best at the card's depth)" if k in reg.get("tied", [])
                 else "unconfirmed (played move not on the card)" if k in reg.get("unconfirmed", []) else "cut" if k in reg.get("cut", [])
                 else "no regret surprise from the rank source (it does not rank the played move below its top by its gap)" if ev is None else "regret surprise raised but not sampled or not listed")
        rej = [r for r in rep["rejected"] if r["seed"].startswith(k + " ")]
        adm = [l["name"] for l in rep["lemmas"] if (l["meta"].get("surprise") or {}).get("id") == k]
        per[i] = {"status": stage, "dev": dev, "rejected": [(r["seed"], r["why"]) for r in rej], "admitted": adm, "rank_top": top, "played_score": sc.get(e["played"])}
        say(f"OBS targeted diagnose sleep problem {i + 1}: group {'development' if dev else 'held-out'}; rank source's top {top[0] if top else None} ({top[1] if top else None}), the move played in the playout "
            f"{e['played']} ({sc.get(e['played'])}), gap {getattr(teacher, 'gap', None)}; {stage}; seeds rejected {len(rej)}"
            + "".join(f"; {r['seed'].split(' ', 1)[1]}: {r['why']}" + (f" (dev {r['dev']})" if r.get("dev") else "") + (f" (held {r['held']})" if r.get("held") else "") for r in rej) + (f"; admitted {adm}" if adm else ""))
    f = rep["funnel"]
    stride = max(1, f.get("cards_dev", 0) // max(1, st.teacher_positions))
    say(f"OBS targeted diagnose sleep sampling: {f.get('cards_dev')} development cards, the rank source is asked about at most {st.teacher_positions} of them, one in {stride} (evenly spread); "
        f"regret surprises raised {f.get('regret_surprises')}, confirmed {f.get('regret_confirmed')}, problems among the development cards {sum(1 for e in extra if SL.is_dev(e['group'], st))}")
    say(f"OBS targeted diagnose sleep funnel: seeds {f.get('seeds')}, climbed {f.get('climbed')}, supported on dev {f.get('supported_dev')}, verified held {f.get('verified_held')}, lemmas admitted {len(rep['lemmas'])}")
    return {"per": per, "funnel": f}


def main(argv: list) -> int:
    from catspace.incubator.analyst.discovery import judges as JG
    from catspace.incubator.analyst.discovery import sleep as SL
    from catspace.incubator.analyst.discovery import wdl_sleep as WS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--problems", default=None)
    ap.add_argument("--judge-engine", required=True)
    ap.add_argument("--judge-depth", type=int, default=12)
    ap.add_argument("--judge-cache", default=None)
    ap.add_argument("--judge-option", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--vocabulary", default=None)
    ap.add_argument("--seconds", type=float, default=0.2)
    ap.add_argument("--log", action="append", default=[])
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--diagnose-sleep", action="store_true")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--teacher-ms", type=int, default=100)
    ap.add_argument("--teacher-cache", default=None)
    ap.add_argument("--deep-depth", type=int, default=None)
    ap.add_argument("--deep-cache", default=None)
    ap.add_argument("--deep-option", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a targeted setting: " + ", ".join(SETTINGS))
    ap.add_argument("--sleep-set", action="append", default=[], metavar="NAME=VALUE", help="(--diagnose-sleep) a lemma-sleep Settings field")
    a = ap.parse_args(argv)
    from catspace.incubator.analyst.discovery import initialize as IN
    root = Path(a.store_root)
    problems = a.problems or str(root / "learned" / a.game / "playout_problems.json")
    opts = lambda pairs: dict(p.partition("=")[::2] for p in pairs)  # noqa: E731
    judge = WS.UciWdlJudge(a.judge_engine, a.judge_depth, a.judge_cache, opts(a.judge_option))
    deep = WS.UciWdlJudge(a.judge_engine, a.deep_depth, a.deep_cache, opts(a.deep_option)) if a.deep_depth else None
    teacher = None
    if a.teacher:
        teacher = SL.UciTeacher(a.teacher, a.teacher_ms, {})
        if a.teacher_cache:
            teacher = JG.Cached(teacher, a.teacher_cache)
    st = {k: (int(v)) for k, v in (p.partition("=")[::2] for p in a.set)}
    try:
        if a.diagnose:
            diagnose_problems(a.game, problems, variant=a.variant, store_root=root, judge=judge, deep_judge=deep, vocabulary=a.vocabulary, seconds=a.seconds)
        if a.diagnose_sleep:
            ss = IN.settings_from({"judge_by": "source", **({"vocabulary": a.vocabulary} if a.vocabulary else {}), **IN.settings_pairs(a.sleep_set)})
            diagnose_sleep(a.game, a.log[0], problems, store_root=root, teacher=teacher, variant=a.variant, settings=ss)
        if a.solve:
            solve_problems(a.game, problems, variant=a.variant, store_root=root, judge=deep or judge, teacher=teacher, logs=a.log, seconds=a.seconds, max_rounds=a.max_rounds,
                           vocabulary=a.vocabulary, settings=st)
    finally:
        judge.close()
        if deep is not None:
            deep.close()
        if teacher is not None:
            teacher.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
