"""PLAN 78.32 INDUCTION FROM EXACT LABELS. A PROBLEM is a position where the wake's own decision plays a move after which the exact class for the mover is lower (playout_sleep). The corpus
(corpus.py) holds tens of thousands of positions with every legal move labelled keeps / loses by an exact labeller. INDUCTION is local and deterministic, no sampled bar:

  HEAD   the response of the rule, from the exact Reading of the problem (targeted.heads_for: a pattern over a move's effect that keeping moves have and no losing move has at the problem), the
         heads that single out the fewest moves first.
  BODY   a beam search over LIFTED literals true at the problem (targeted.body_candidates: every constant of the position one variable everywhere, the head's own variables shared with the
         head), scored on the corpus's DEVELOPMENT part by the full drill (corpus.drill_many: right = every named move keeps the class, wrong = a named move loses it, trivial = every move keeps):
         zero wrong first, then the most non-trivial right firings, then the fewest literals. It stops at the first conjunction size that has a body with zero wrong and at least `min_right`
         non-trivial right firings. At most `max_literals` literals; `beam` conjunctions kept per size (the single literals are first read on every `stride`-th development position, to save time, and the `beam`
         kept are read again on the whole part; every figure printed and every admission is on the whole part). A child's firings are a subset of its parent's, so a child is read only where the parent fired.
         NO LITERAL OVER A MOVE COUNTER: a fact of one numeric argument (relation(N): a half-move clock or move number) differs between positions by the game's age alone, so it is never a candidate
         (targeted.is_counter, generic: no relation is named).
  ADMISSION  (a) zero wrong and at least `min_right` non-trivial right firings on the development part, (b) zero wrong and at least `min_held` non-trivial right firings on the HELD-OUT part (groups the
         search never read), (c) the problem's own check: the wake's decision with a trial snapshot that holds the rule keeps the class, (d) no problem solved before is broken. The EXAM is never
         used here. The rule is written by the store's own writer (game_store.write_lemma + publish) with provenance `origin induced, mined_from problem`; no verdict of the labeller and no count of
         a drill is in the lemma or the snapshot.

Game-free: the variant string, the store's vocabulary, objects passed in (corpus, labeller); no game object is named here.

  induce(problem, corpus, *, reader, ...) -> {status "found" | "open", lemma, head, dev, held, why, stage, heads_tried}      (the search and the held-out part of the admission)
  solve_problems(game, problems, corpus, *, labeller, ...) -> {total, attempted, admitted, solved, still_open, reasons, rows, snapshot}      (the loop, the problem's check, the writer)
"""
from __future__ import annotations

import json
import tempfile
import time
from collections import Counter

from catspace.incubator.analyst.discovery import abstraction as AB
from catspace.incubator.analyst.discovery import apply_audit as AU
from catspace.incubator.analyst.discovery import corpus as CP
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import playout_sleep as PS
from catspace.incubator.analyst.discovery import targeted as TG

SETTINGS = {"max_literals": 4, "beam": 8, "min_right": 10, "min_held": 5, "max_heads": 4, "finalists": 3, "max_bindings": 256, "stride": 3}
STAGES = ("position", "head", "dev", "held", "check", "broken")          # how far a candidate got: the reason of the furthest one is the problem's


def _nconst(body: list) -> int:
    return sum(1 for l in body for a in l["args"] if not a.startswith("?"))


def _key(t: dict, body: list) -> tuple:
    return (t["wrong"], -t["right"], len(body), _nconst(body), json.dumps(body, sort_keys=True))


def search_body(corpus: "CP.Corpus", facts: list, names: dict, head: dict, *, max_literals: int, beam: int, min_right: int, finalists: int, max_bindings: int, part: str = "dev", stride: int = 1) -> dict:
    """The beam search for one head -> {found: [(body, tally)] (zero wrong and >= min_right right, the best first; [] if none), best: (body, tally) with the fewest wrong then the most right
    seen at any size, levels: [{literals, evaluated, kept}]}."""
    cands = TG.body_candidates(facts, names, set())
    levels, best = [], None
    frontier = [([], None)]                                         # (body, tally of the parent: None = read on the whole part)
    seen: set = set()
    for size in range(1, max_literals + 1):
        children = []
        for body, ptally in frontier:
            have = {json.dumps(l, sort_keys=True) for l in body}
            todo = []
            for lit, _desc in cands:
                lk = json.dumps(lit, sort_keys=True)
                if lk in have:
                    continue
                nb = body + [lit]
                ck = frozenset(json.dumps(l, sort_keys=True) for l in nb)
                if ck in seen:
                    continue
                seen.add(ck)
                todo.append(nb)
            if not todo:
                continue
            idx = corpus.idx(part)[::max(1, stride)] if ptally is None else sorted(ptally["right_at"] + ptally["wrong_at"])
            if idx is not None and not idx:
                continue
            ts = CP.drill_many([TG.lemma_of(nb, head) for nb in todo], corpus, part=part, indices=idx, max_bindings=max_bindings)
            for nb, t in zip(todo, ts):
                if t["truncated"] or t["right"] == 0:
                    continue
                children.append((nb, t))
        levels.append({"literals": size, "evaluated": len(children)})
        if not children:
            break
        children.sort(key=lambda c: _key(c[1], c[0]))
        if size == 1 and stride > 1:                                # the first size was read on every `stride`-th position: the conjunctions kept are read again on the whole part
            top = children[:beam]
            full = CP.drill_many([TG.lemma_of(nb, head) for nb, _t in top], corpus, part=part, max_bindings=max_bindings)
            children = [(nb, t) for (nb, _o), t in zip(top, full) if not t["truncated"] and t["right"] > 0]
            if not children:
                break
            children.sort(key=lambda c: _key(c[1], c[0]))
        if best is None or _key(children[0][1], children[0][0])[:2] < _key(best[1], best[0])[:2]:
            best = children[0]
        good = [c for c in children if c[1]["wrong"] == 0 and c[1]["right"] >= min_right]
        levels[-1]["kept"] = min(beam, len(children))
        if good:
            good.sort(key=lambda c: (-c[1]["right"], len(c[0]), _nconst(c[0]), json.dumps(c[0], sort_keys=True)))
            return {"found": good[:finalists], "best": best, "levels": levels}
        frontier = children[:beam]
    return {"found": [], "best": best, "levels": levels}


def _further(cur: dict, new: dict) -> dict:
    """The candidate that got further (STAGES), at the same stage on the development part the one with fewer wrong, then more right firings."""
    a, b = STAGES.index(cur["stage"]), STAGES.index(new["stage"])
    if b != a:
        return new if b > a else cur
    if new["stage"] == "dev" and new["dev"] is not None and (cur["dev"] is None or (new["dev"]["wrong"], -new["dev"]["right"]) < (cur["dev"]["wrong"], -cur["dev"]["right"])):
        return new
    return cur


def rule_text(lemma: "dict | None") -> str:
    return AU.render(lemma).splitlines()[0] if lemma else "none"


def _counts(t: "dict | None") -> str:
    return "n/a" if t is None else f"{t['binds']}/{t['right']}/{t['wrong']}/{t['trivial']}"


def induce(problem: dict, corpus: "CP.Corpus", *, reader: "TG.Reader", max_literals: int = SETTINGS["max_literals"], beam: int = SETTINGS["beam"], min_right: int = SETTINGS["min_right"],
           min_held: int = SETTINGS["min_held"], max_heads: int = SETTINGS["max_heads"], finalists: int = SETTINGS["finalists"], max_bindings: int = SETTINGS["max_bindings"], stride: int = SETTINGS["stride"], accept=None, say=print, tag: str = "") -> dict:
    """A rule for `problem` ({start, moves}) from the corpus's exact labels: the heads of its exact Reading, a body by beam search on the development part, the held-out admission on the held-out
    part, then `accept(candidate) -> (ok, why, stage)` when given (solve_problems': the problem's own check and the no-regression check on a trial snapshot; a rule it refuses
    is not found, and the search goes on with the next candidate and head). -> {status: "found" | "open", lemma, head, dev, held (drill counts), why, stage (how far the furthest candidate
    got: STAGES), heads_tried, candidates ([the accepted rule])}."""
    p = {"start": problem.get("start"), "moves": list(problem["moves"])}
    try:
        rd = reader.read(p, "problem", "problem")
    except LookupError as e:
        return {"status": "open", "why": f"the labeller does not cover the position ({e})", "stage": "position", "lemma": None, "head": None, "dev": None, "held": None, "heads_tried": 0, "candidates": []}
    if rd is None:
        return {"status": "open", "why": "the position is over", "stage": "position", "lemma": None, "head": None, "dev": None, "held": None, "heads_tried": 0, "candidates": []}
    if not rd.good or not rd.bad:
        why = "no move keeps the class" if not rd.good else "every move keeps the class"
        return {"status": "open", "why": why, "stage": "position", "lemma": None, "head": None, "dev": None, "held": None, "heads_tried": 0, "candidates": []}
    card = json.loads(reader.cards([(p["start"], p["moves"], "p", "problem")]).card(0))
    heads = TG.heads_for(card, rd.keep, limit=max_heads)
    if not heads:
        return {"status": "open", "why": "no pattern over the moves' effects separates the keeping moves from the losing ones", "stage": "head", "lemma": None, "head": None, "dev": None, "held": None,
                "heads_tried": 0, "candidates": []}
    say(f"OBS induce {tag}: position {len(p['moves'])} plies, class {rd.before}, {len(rd.keep)} legal moves, keep {sorted(rd.good)}, lose {sorted(rd.bad)}; {len(heads)} head(s): "
        + "; ".join(f"{TG.head_text(h['head'])} names {h['matched']}" for h in heads))
    furthest = {"stage": "head", "why": "", "lemma": None, "head": None, "dev": None, "held": None}
    for hi, h in enumerate(heads, 1):
        r = search_body(corpus, card["facts"], h["names"], h["head"], max_literals=max_literals, beam=beam, min_right=min_right, finalists=finalists, max_bindings=max_bindings, stride=stride)
        if not r["found"]:
            b = r["best"]
            if b is None:
                why = f"head {hi}: no literal true at the problem leaves a non-trivial right firing on the development part"
                lem, dev = None, None
            else:
                lem = TG.lemma_of(b[0], h["head"])
                dev = {k: b[1][k] for k in ("binds", "names_a_move", "right", "wrong", "trivial")}
                why = (f"head {hi}: no conjunction of at most {max_literals} literals has zero wrong and at least {min_right} right on the development part (the best has {b[1]['wrong']} wrong, "
                       f"{b[1]['right']} right with {len(b[0])} literal(s))")
            furthest = _further(furthest, {"stage": "dev", "why": why, "lemma": lem, "head": h["head"], "dev": dev, "held": None})
            say(f"OBS induce {tag} head {hi}: {TG.head_text(h['head'])}: {why}")
            continue
        passed = []
        for body, t in r["found"]:
            lem = TG.lemma_of(body, h["head"])
            lem["name"] = AB.name_of(lem)
            dev = CP.drill(lem, corpus, part="dev", max_bindings=max_bindings)
            held = CP.drill(lem, corpus, part="held", max_bindings=max_bindings)
            ok = held["wrong"] == 0 and held["right"] >= min_held and not held["truncated"] and dev["wrong"] == 0
            say(f"OBS induce {tag} head {hi} candidate: rule {rule_text(lem)}; dev binds/right/wrong/trivial {_counts(dev)}, held {_counts(held)}; "
                + ("passes the held-out part" if ok else f"REJECTED on the held-out part: " + (f"{held['wrong']} wrong firing(s)" if held["wrong"] else f"{held['right']} right firing(s), need {min_held}")))
            if ok:
                cand = {"lemma": lem, "head": h["head"], "dev": dev, "held": held}
                good, why2, stage2 = accept(cand) if accept else (True, "", "held")
                if good:
                    passed.append(cand)
                else:
                    furthest = _further(furthest, {"stage": stage2, "why": f"head {hi}: {why2}", "lemma": lem, "head": h["head"], "dev": dev, "held": held})
            else:
                furthest = _further(furthest, {"stage": "held", "why": (f"head {hi}: the rule passes the development part but has {held['wrong']} wrong firing(s) on the held-out part" if held["wrong"]
                                                                       else f"head {hi}: the rule passes the development part but only {held['right']} right firing(s) on the held-out part (need {min_held})"),
                                               "lemma": lem, "head": h["head"], "dev": dev, "held": held})
        if passed:
            return {"status": "found", "lemma": passed[0]["lemma"], "head": passed[0]["head"], "dev": passed[0]["dev"], "held": passed[0]["held"], "why": "", "stage": "held", "heads_tried": hi,
                    "candidates": passed}
    return {"status": "open", "lemma": furthest["lemma"], "head": furthest["head"], "dev": furthest["dev"], "held": furthest["held"], "why": furthest["why"] or "no head gave a rule", "stage": furthest["stage"],
            "heads_tried": len(heads), "candidates": []}


# ------------------------------------------------------------------------------------------------------------------------------------ the loop over the problems


def drill_admit(lemmas: list, corpus, spec: dict, vocab_json: str, *, max_bindings: int = 256, say=print) -> tuple:
    """The full drill as the admission test of rules that came from ANOTHER miner (the lemma sleep's): each rule is run over both parts of the exact corpus and kept only with zero
    wrong firings on each. -> (kept, dropped) lists of lemma dicts; one OBS line per rule. Nothing of the drill is written into a lemma."""
    corpus.cards(spec, vocab_json)
    kept, dropped = [], []
    for lem in lemmas:
        dev = CP.drill(lem, corpus, part="dev", max_bindings=max_bindings)
        held = CP.drill(lem, corpus, part="held", max_bindings=max_bindings)
        ok = dev["wrong"] == 0 and held["wrong"] == 0
        (kept if ok else dropped).append(lem)
        say(f"OBS drill admit {lem['name']}: dev binds/right/wrong/trivial {dev['binds']}/{dev['right']}/{dev['wrong']}/{dev['trivial']}, held {held['binds']}/{held['right']}/{held['wrong']}/{held['trivial']}; "
            + ("KEPT" if ok else "DROPPED: wrong firings on the exact corpus"))
    say(f"OBS drill admit total: rules {len(lemmas)}, kept {len(kept)}, dropped {len(dropped)}")
    return kept, dropped


def trial_snapshot(game: str, lemma: dict, directory, vocabulary: "str | None" = None, judge: "str | None" = None) -> str:
    """The snapshot the store would publish with `lemma` added (the same content, so the same id as the published one), written under `directory`: `judge` is the NAME of the valuation record the wake
    plays by (game_store.judge_of; None: the store's record in use), so the trial reads the position as the wake will. (targeted.trial_snapshot leaves the record out.)"""
    import cs_engine as CE
    from pathlib import Path
    data = json.loads(GS.build(game, parent=GS.head(game), teacher_used=True, judge=GS.judge_of(game, judge) if judge else "store", vocabulary=vocabulary))
    data["lemmas"] = sorted([l for l in data["lemmas"] if l["name"] != lemma["name"]] + [lemma], key=lambda l: GS.slug(l["name"]))
    data["teacher_used"] = True
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    p = Path(directory) / f"{CE.content_id(raw)}.json"
    p.write_bytes(raw)
    CE.snapshot_info(str(p))
    return str(p)


def provenance(problem: dict, variant: str, labeller, vocabulary: "str | None") -> dict:
    """The lemma's meta: where it came from. NO number and no verdict of the labeller."""
    return {"origin": "induced", "plan": "78.32", "game": variant, "mined_from": "problem", "problem": {"start": problem.get("start"), "moves": len(problem["moves"]), "played": problem.get("played")},
            "teacher": True, "source": getattr(labeller, "source", "exact"), **({"vocabulary": vocabulary} if vocabulary else {})}


def solve_problems(game: str, problems, corpus: "CP.Corpus", *, variant: "str | None" = None, labeller, vocabulary: "str | None" = None, spec: "dict | None" = None, seconds: float = 0.2,
                   wake_kw: "dict | None" = None, judge: "str | None" = None, settings: "dict | None" = None, say=print) -> dict:
    """Induce a rule for each OPEN problem of `problems` (rows {start, moves, played?}: the cycle's rows, or a problem file's path) and admit it when the held-out part and the problem's own check say
    so. `judge`: the name of the valuation record the wake plays by (trial snapshots and the published snapshot carry it). The store paths must already point at the sandbox (initialize.run_size does that, or targeted.sandbox). The rules are written with game_store.write_lemma and a snapshot is published, as the
    lemma sleep does; after each admission every problem is checked again with the head snapshot (a rule may solve several). `settings`: SETTINGS.
    -> {total, attempted, admitted [{name, text, problem, solved_others, snapshot, dev, held}], solved (open at the start, kept at the end), still_open, reasons {why: problems}, rows, snapshot}.
    Prints `OBS induce problem i/N: ...` per problem and `OBS induce total: ...`."""
    t0 = time.perf_counter()
    s = {**SETTINGS, **(settings or {})}
    variant = variant or game
    rows = list(problems) if isinstance(problems, (list, tuple)) else PS.load_problems(problems)
    spec = spec if spec is not None else GS.spec_texts(game, vocabulary)
    vocab_json = json.dumps(GS.vocabulary(game, vocabulary))
    corpus.cards(spec, vocab_json)
    verdicts = PS.Verdicts(labeller, variant)
    reader = TG.Reader(variant, spec, vocab_json, verdicts)
    wk = wake_kw or {}

    def check_rows(snapshot_path: "str | None", which: list) -> list:
        return PS.check([{"start": r.get("start"), "moves": r["moves"]} for r in which], variant=variant, store=snapshot_path, spec=spec, judge=labeller, seconds=seconds, wake_kw=wk, verdicts=verdicts)

    def head_path() -> "str | None":
        h = GS.head(game)
        return str(GS.path_of(game, h)) if h else None

    res = check_rows(head_path(), rows)
    status = ["solved" if c["keeps"] else "open" for c in res]
    start_open = [i for i, st in enumerate(status) if st == "open"]
    info = {i: {"index": i, "status": status[i], "why": ""} for i in range(len(rows))}
    admitted, attempted, reasons, done = [], 0, Counter(), set()
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(len(rows)):
            if status[i] != "open" or i in done:
                continue
            done.add(i)
            attempted += 1
            tag = f"problem {i + 1}/{len(rows)}"
            solved_rows = [rows[j] for j in range(len(rows)) if status[j] == "solved"]
            seen_check: dict = {}

            def accept(cand, i=i, solved_rows=solved_rows, seen_check=seen_check):
                lem = cand["lemma"]
                lem["meta"] = provenance(rows[i], variant, labeller, vocabulary)
                lem["stats"] = {"status": "INDUCED"}
                GS.check_lemma(lem)
                snap = trial_snapshot(game, lem, tmp, vocabulary, judge)
                chk = check_rows(snap, [rows[i]] + solved_rows)
                own, broken = chk[0], [r for r, c in zip(solved_rows, chk[1:]) if not c["keeps"]]
                seen_check["keeps"] = own["keeps"]
                if not own["keeps"]:
                    return False, f"the problem's own check fails with the rule: the decision still chooses {own['move']}", "check"
                if broken:
                    return False, f"the rule breaks {len(broken)} problem(s) solved before", "broken"
                return True, "", "held"
            out = induce(rows[i], corpus, reader=reader, max_literals=s["max_literals"], beam=s["beam"], min_right=s["min_right"], min_held=s["min_held"], max_heads=s["max_heads"],
                         finalists=s["finalists"], max_bindings=s["max_bindings"], stride=s["stride"], accept=accept, say=say, tag=tag)
            why, stage, own_keeps = out["why"], out["stage"], seen_check.get("keeps")
            final = out["lemma"] if out["status"] == "found" else None
            line = (f"OBS induce {tag}: head {TG.head_text(out['head']) if out['head'] else 'none'}, rule {rule_text(out['lemma'])}, dev binds/right/wrong/trivial {_counts(out['dev'])}, "
                    f"held binds/right/wrong/trivial {_counts(out['held'])}, problem check kept {own_keeps if own_keeps is not None else 'not run'}; ")
            if final is None:
                info[i]["why"] = why
                reasons[_reason_key(stage, why)] += 1
                say(line + f"open because {why}")
                continue
            GS.write_lemma(game, final)
            sid = GS.publish(game, teacher_used=True, vocabulary=vocabulary, **({"judge": GS.judge_of(game, judge)} if judge else {}))
            now_res = check_rows(head_path(), rows)
            now = ["solved" if c["keeps"] else "open" for c in now_res]
            newly = [j for j in range(len(rows)) if status[j] == "open" and now[j] == "solved"]
            lost = [j for j in range(len(rows)) if status[j] == "solved" and now[j] == "open"]
            status = now
            for j in newly:
                info[j]["status"] = "solved"
                info[j]["why"] = f"by rule {final['name']}"
            admitted.append({"name": final["name"], "text": rule_text(final), "problem": i, "solved_others": [j for j in newly if j != i], "broke": lost, "snapshot": sid, "dev": out["dev"], "held": out["held"]})
            say(line + f"ADMITTED {final['name']} (snapshot {sid}); solved {len(newly)} problem(s) in all, the others {[j + 1 for j in newly if j != i]}; broke {len(lost)}")
    solved = sum(1 for i in start_open if status[i] == "solved")
    open_n = sum(1 for st in status if st == "open")
    say(f"OBS induce total: problems {len(rows)}, attempted {attempted}, admitted {len(admitted)}, solved {solved}, still open {open_n}, reasons {dict(reasons)}")
    for a in admitted:
        say(f"OBS induce rule {a['name']}: {a['text']}; dev binds/right/wrong/trivial {_counts(a['dev'])}, held {_counts(a['held'])}; mined from problem {a['problem'] + 1}, also solved {[j + 1 for j in a['solved_others']]}")
    say(f"OBS induce time: {time.perf_counter() - t0:.1f} s, labeller calls {verdicts.calls}")
    return {"total": len(rows), "attempted": attempted, "admitted": admitted, "solved": solved, "still_open": open_n, "reasons": dict(reasons), "rows": [info[i] for i in range(len(rows))],
            "snapshot": GS.head(game)}


def _reason_key(stage: str, why: str) -> str:
    """A short stable name for the reason a problem stayed open (the counts of `OBS induce total`)."""
    if stage == "dev":
        return "no conjunction with zero wrong and enough right on the development part"
    if stage == "held":
        return "held-out part: wrong firings" if "wrong firing" in why else "held-out part: too few right firings"
    if stage == "check":
        return "the problem's own check fails with the rule"
    if stage == "broken":
        return "the rule breaks a problem solved before"
    return why.split(" (")[0]
