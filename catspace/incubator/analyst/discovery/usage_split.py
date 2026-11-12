"""PLAN 78.35 (5) THE USE / GENERATION SPLIT, A STANDING MEASUREMENT beside every exam grade. TRAINING ONLY (a labeller is asked). Game-free: the variant, the specs, the labeller and the
snapshot are passed in; no object of any game is named here.

Use and generation are different questions. GENERATION: is there a rule for the decision (a missing rule is a missing word or a missing lemma). USE: given the rules, does the engine reason
with them. For each position of a graded exam the rules of the snapshot are run by the ENGINE'S OWN matcher (apply_audit.proposals: one card, one-hot value per move, the lemma's case) to
say which moves each rule NAMES; the labeller says which moves keep the position's result class. Then, with `d` the decision the exam graded:

  decision KEPT    the rule(s) named the move played (named_played); rules named other moves only (named_other); no rule named a move (no_rule)
  decision FAILED  a rule named a keeping move and another move was played (usage); of those, some rule also named a losing move (usage_conflict, a subset of usage); rules named only
                   losing moves (wrong_rule); no rule named a move (no_rule)

  split(exam, graded, *, variant, snapshot_path, specs, labeller) -> {kept {named_played, named_other, no_rule}, failed {usage, usage_conflict, wrong_rule, no_rule}, rows [..], rules {name: lemma},
        by_group {group: failed count}}; `graded` is exam.grade's result (its `decisions` are reused: nothing is graded again); a row is {i, group, before, move, keeps, bucket, named {rule: [moves]},
        keeping [moves]}
  say_split(label, result, layers=None, say=print) -> prints
        OBS usage split <label>: kept K (a rule named the move played A, rules named other moves only B, no rule named a move C); failed F (a rule named a keeping move and another was played U,
            of them with a rule also naming a losing move UC; rules named only losing moves W; no rule named a move N)
        and, with `layers` ({relation: "general" | "game"}, word_registry.layers), of the decisions where a rule named the move played
        OBS usage split by rule kind <label>: a rule named the move played in A decisions: named by a general rule G (only by game rules C), by a hand rule H (only by machine rules M)
  PLAN 78.36: a SUGGESTION rule (`suggest`: true) is left out of `rules` and of every count above; the result gains `suggestion_rules` (how many the snapshot holds) and `suggestions_named_played`
  (decisions whose move a suggestion rule named), and where the snapshot holds a suggestion the first OBS line ends `; suggestions named the move played S`.
  rules_by_kind(lemmas, layers) -> {general {hand, machine}, game {hand, machine}};  say_rules_by_kind(lemmas, layers, say) -> prints
        OBS rules by kind: general G (hand a, machine b), game C (hand c, machine d)
"""
from __future__ import annotations

import json
from collections import Counter

import cs_engine as CE

from catspace.incubator.analyst.discovery import apply_audit as AU
from catspace.incubator.analyst.discovery import word_registry as WR


def split(exam: dict, graded: dict, *, variant: str, snapshot_path: str, specs: dict, labeller) -> dict:
    snap = json.loads(open(snapshot_path, "rb").read())
    lemmas = [l for l in snap["lemmas"] if "claim" not in l and not l.get("suggest")]          # (plan 78.36: a claim rule names no move; a SUGGESTION's moves are counted apart, below)
    sugg = [l for l in snap["lemmas"] if "claim" not in l and l.get("suggest")]
    vocab = json.dumps(snap["vocabulary"])
    kept, failed, by_group, rows = Counter(), Counter(), Counter(), []
    sug_played = 0
    for p, d in zip(exam["positions"], graded["decisions"]):
        keep = labeller.moves_keep(variant, p["start"], p["moves"]) or {}
        named = {}
        sug_named = {}
        if lemmas or sugg:
            cs = CE.build_cards([(variant, p["start"], list(p["moves"]), "x", "x")], specs, vocab, 1)
            for l in lemmas:
                pr = AU.proposals(cs, l)
                if pr:
                    named[l["name"]] = pr
            for l in sugg:
                pr = AU.proposals(cs, l)
                if pr:
                    sug_named[l["name"]] = pr
        sug_played += any(d["move"] in ms for ms in sug_named.values())
        any_good = any(keep.get(m) for ms in named.values() for m in ms)
        any_bad = any(not keep.get(m) for ms in named.values() for m in ms)
        if d["keeps"]:
            b = "named_played" if any(d["move"] in ms for ms in named.values()) else ("named_other" if named else "no_rule")
            kept[b] += 1
        else:
            if any_good:
                b = "usage"
                failed["usage"] += 1
                failed["usage_conflict"] += any_bad
            elif any_bad:
                b = "wrong_rule"
                failed["wrong_rule"] += 1
            else:
                b = "no_rule"
                failed["no_rule"] += 1
            by_group[p.get("group")] += 1
        rows.append({"i": d["i"], "group": p.get("group"), "before": p.get("before"), "move": d["move"], "keeps": d["keeps"], "bucket": b + ("_conflict" if b == "usage" and any_bad else ""),
                     "named": named, "keeping": sorted(m for m, k in keep.items() if k)})
    return {"kept": {k: kept[k] for k in ("named_played", "named_other", "no_rule")}, "failed": {k: failed[k] for k in ("usage", "usage_conflict", "wrong_rule", "no_rule")},
            "rows": rows, "rules": {l["name"]: l for l in lemmas}, "by_group": dict(by_group),
            "suggestion_rules": len(sugg), "suggestions_named_played": sug_played}


def by_kind(result: dict, layers: dict) -> dict:
    """Of the kept decisions where a rule named the move played: how many had a general rule among those naming it, and a hand rule among them."""
    n = g = h = 0
    for r in result["rows"]:
        if r["bucket"] != "named_played":
            continue
        who = [result["rules"][name] for name, ms in r["named"].items() if r["move"] in ms]
        n += 1
        g += any(WR.rule_layer(l, layers) == "general" for l in who)
        h += any(WR.origin_of(l) == "hand" for l in who)
    return {"decisions": n, "general": g, "game_only": n - g, "hand": h, "machine_only": n - h}


def say_split(label: str, result: dict, layers: "dict | None" = None, say=print) -> None:
    k, f = result["kept"], result["failed"]
    say(f"OBS usage split {label}: kept {sum(k.values())} (a rule named the move played {k['named_played']}, rules named other moves only {k['named_other']}, no rule named a move {k['no_rule']}); "
        f"failed {f['usage'] + f['wrong_rule'] + f['no_rule']} (a rule named a keeping move and another was played {f['usage']}, of them with a rule also naming a losing move {f['usage_conflict']}; "
        f"rules named only losing moves {f['wrong_rule']}; no rule named a move {f['no_rule']})"
        + (f"; suggestions named the move played {result['suggestions_named_played']}" if result.get("suggestion_rules") else ""))          # (plan 78.36: only where the snapshot holds a suggestion)
    if layers is not None:
        b = by_kind(result, layers)
        say(f"OBS usage split by rule kind {label}: a rule named the move played in {b['decisions']} decisions: named by a general rule {b['general']} (only by game rules {b['game_only']}), "
            f"by a hand rule {b['hand']} (only by machine rules {b['machine_only']})")


def rules_by_kind(lemmas: list, layers: dict) -> dict:
    out = {"general": Counter(), "game": Counter()}
    for l in lemmas:
        out[WR.rule_layer(l, layers)][WR.origin_of(l)] += 1
    return {k: {"hand": v["hand"], "machine": v["machine"]} for k, v in out.items()}


def say_rules_by_kind(lemmas: list, layers: dict, say=print) -> dict:
    c = rules_by_kind(lemmas, layers)
    g, m = c["general"], c["game"]
    say(f"OBS rules by kind: general {g['hand'] + g['machine']} (hand {g['hand']}, machine {g['machine']}), game {m['hand'] + m['machine']} (hand {m['hand']}, machine {m['machine']})")
    return c
