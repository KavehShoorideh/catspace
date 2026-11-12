"""PLAN 78.34 THE PROBLEMS THE GAMES DO NOT REACH, AND A CORPUS THAT GROWS WITH THE GAMES. Training only (the wake, game_store's snapshot writer and the Rust sources never read this module).

Two gaps were measured: the games surface a handful of problems per cycle although the exam shows about thirty wrong decisions; and a rule with zero wrong firings on the FIXED corpus was
still wrong in the games it changed, because the corpus does not hold the positions the new rule steers play into. Game-free: the variant string, the corpus, the labeller and the store are
objects passed in.

  sample_dev(corpus, n, seed, cycle) -> row indices      the DEVELOPMENT part's non-trivial positions (some move keeps the class, some loses it), spread evenly over the groups, deterministic for
                                                         (seed, cycle) and different from one cycle to the next (random.Random(seed + cycle)); never the held-out part
  corpus_problems(corpus, rows, ...) -> {asked, wrong, new, known, rows, new_rows}
        the wake's own decision (playout_sleep.check, the call the exam's grade makes) on the sample; a decision that loses the exact class (the corpus row's label) is a problem, merged into the
        problem rows with source "corpus" (a row without a source counts as "playout")
  fired_positions(log) -> [{start, moves}]               every position of a wake log where a rule of the store fired (the `fired` facts, apply_audit.audit reads the same)
  grow(corpus, positions, labeller, *, exclude) -> {...} the positions labelled by the exact labeller and ADDED to the corpus (Corpus.extend: only the new cards are built); the group of a position is
        its game's START's group, so the split by group holds; a start with no group yet gets a new group `grown<k>` whose part (dev / held) is the parity of the SHA-1 of the start text (deterministic,
        no draw); a position already there (by the kernel's key, so a transposition too), an exam position or one the labeller does not cover is not added
  grow_and_withdraw(game, corpus, log, ...) -> report     grow, then every rule of the sandbox store drilled on both parts of the grown corpus; a rule with any wrong firing is WITHDRAWN
        (game_store.remove_lemma, a snapshot published with the wake's valuation record), and each position where it was wrong is a problem (source "withdrawn", in the problem file) where the
        wake's decision without the rule is also wrong, else only recorded in the report
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

import cs_engine as CE

from catspace.incubator.analyst.discovery import corpus as CP
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import playout_sleep as PS
from catspace.incubator.analyst.discovery import sleep as SL

SOURCES = ("playout", "corpus", "withdrawn")
MAX_CHECK_PER_RULE = 10          # a withdrawn rule's wrong positions the wake's decision is asked about (the first ones by corpus row), the others are only counted


def source_of(row: dict) -> str:
    """The source of a problem row: "playout" (also a row written before 78.34, which has no field), "corpus" or "withdrawn"."""
    return row.get("source") or "playout"


def position_text(start, moves: list) -> str:
    return f"{start} after {len(moves)} move(s) [{' '.join(moves[-6:])}]" if moves else f"{start} (the start)"


# ------------------------------------------------------------------------------------------------------------------------------------ 1: problems from the corpus

def sample_dev(corpus: "CP.Corpus", n: int, seed: int, cycle: int) -> list:
    """Row indices of at most `n` non-trivial positions of the DEVELOPMENT part, taken round-robin over the groups (each group's positions shuffled by Random(seed + cycle), the order of the groups
    too, so the remainder falls on different groups), sorted. Nothing of the held-out part; the exam is not in a corpus passed in (corpus.excluding)."""
    if n <= 0:
        return []
    by: dict = {}
    for i in corpus.part("dev"):
        if corpus.non_trivial(i):
            by.setdefault(corpus.rows[i]["group"], []).append(i)
    rng = random.Random(seed + cycle)
    order = sorted(by)
    for g in order:
        rng.shuffle(by[g])
    rng.shuffle(order)
    out, queues = [], [by[g] for g in order]
    while queues and len(out) < n:
        nxt = []
        for q in queues:
            if len(out) >= n:
                break
            out.append(q.pop())
            if q:
                nxt.append(q)
        queues = nxt
    return sorted(out)


def corpus_problems(corpus: "CP.Corpus", rows: list, *, cycle: int, seed: int, n: int, variant: str, store: "str | None", spec, judge, seconds: float, wake_kw: "dict | None" = None, say=print) -> dict:
    """The wake's own decision (`PS.check` with the snapshot `store`) on `n` sampled development positions; a decision whose move loses the exact class (the corpus row's label: the move is labelled
    False) is a problem. The new ones are merged into `rows` (the problem rows) with source "corpus" and a `before` record in their history. A decision that failed (no move) is not a problem.
    -> {asked, wrong, new, known (wrong and already in `rows`), rows (the merged list), new_rows}. Prints `OBS corpus problems cycle C: asked N, wrong W, new problems P (already known K)`."""
    idx = sample_dev(corpus, n, seed, cycle)
    pos = [{"start": corpus.rows[i]["start"], "moves": list(corpus.rows[i]["moves"])} for i in idx]
    res = PS.check(pos, variant=variant, store=store, spec=spec, judge=judge, seconds=seconds, wake_kw=wake_kw) if pos else []
    found, by_key = [], {}
    for i, p, c in zip(idx, pos, res):
        mv = c["move"]
        if mv is not None and corpus.rows[i]["keep"].get(mv) is False:
            found.append({"start": p["start"], "moves": p["moves"], "played": mv, "source": "corpus"})
            by_key[PS.key_of(p)] = c
    merged, new = PS.merge_problems(rows, found, cycle)
    PS.record(new, [{**by_key[PS.key_of(r)], "keeps": False} for r in new], cycle, "before")
    known = len(found) - len(new)
    say(f"OBS corpus problems cycle {cycle}: asked {len(pos)}, wrong {len(found)}, new problems {len(new)} (already known {known})")
    return {"cycle": cycle, "asked": len(pos), "wrong": len(found), "new": len(new), "known": known, "rows": merged, "new_rows": new}


def source_counts(rows: list) -> dict:
    """{source: {total, open}} over the problem rows (open = the last recorded status is open)."""
    out = {s: {"total": 0, "open": 0} for s in SOURCES}
    for r in rows:
        e = out.setdefault(source_of(r), {"total": 0, "open": 0})
        e["total"] += 1
        e["open"] += bool(r.get("history")) and r["history"][-1]["status"] == "open"
    return out


def say_sources(rows: list, cycle: int, say=print) -> dict:
    """`OBS problems by source cycle C: playout total T open O, corpus total T open O, withdrawn total T open O` (open: the last check of the cycle)."""
    c = source_counts(rows)
    say(f"OBS problems by source cycle {cycle}: " + ", ".join(f"{s} total {c[s]['total']} open {c[s]['open']}" for s in c))
    return c


# ------------------------------------------------------------------------------------------------------------------------------------ 2: the corpus grows

def fired_positions(log: str) -> list:
    """Every distinct position {start, moves} of a wake log's decisions where a rule of the store fired (a `fired` fact in the decision's row), in log order."""
    decisions, games = SL.read_log(log)
    out, seen = [], set()
    for (run, g, ply), d in sorted(decisions.items()):
        if not any(f.get("relation") == "fired" for f in (d["row"].get("facts") or [])):
            continue
        row = games.get((run, g))
        if row is None:
            continue
        k = (row.get("start"), tuple(row["moves"][:ply]))
        if k not in seen:
            seen.add(k)
            out.append({"start": k[0], "moves": list(k[1])})
    return out


def part_of_new_start(start) -> str:
    """The part a start without a group gets: "dev" for an even SHA-1 of the start's text, else "held" (deterministic; the split by group stays a split by group)."""
    return "dev" if int(hashlib.sha1(str(start).encode()).hexdigest(), 16) % 2 == 0 else "held"


def grow(corpus: "CP.Corpus", positions: list, labeller, *, exclude=frozenset(), say=print) -> dict:
    """Label `positions` ({start, moves}) with `labeller` (moves_keep and result; a position it does not cover is skipped) and add them to `corpus` (Corpus.extend). Not added: a position already in the
    corpus (the kernel's position key), an `exclude`d one (a set of (start, tuple(moves)): the exam's, by position key too), one whose game is over. A position's group is its start's group in the
    corpus; a start with no group gets a new group `grown<k>` in the part `part_of_new_start`. -> {added, dev, held, known, exam, over, uncovered, new_groups, rows (the added rows)}."""
    variant = corpus.variant
    have = corpus.position_keys()
    ex_keys = {CE.state(variant, s, list(m))[4] for s, m in exclude}
    group_of: dict = {}
    for r in corpus.rows:
        group_of.setdefault(r["start"], r["group"])
    new_rows, seen, c = [], set(), Counter()
    new_groups = []
    for p in positions:
        start, moves = p.get("start"), list(p["moves"])
        _f, mover, legal, outcome, key = CE.state(variant, start, moves)
        if mover is None or outcome is not None or not legal:
            c["over"] += 1
            continue
        if (start, tuple(moves)) in exclude or key in ex_keys:
            c["exam"] += 1
            continue
        if key in have or key in seen:
            c["known"] += 1
            continue
        seen.add(key)
        keep = labeller.moves_keep(variant, start, moves)
        before = labeller.result(variant, start, moves) if keep is not None else None
        if keep is None or before is None:
            c["uncovered"] += 1
            continue
        if start not in group_of:
            name = f"grown{sum(1 for g in corpus.groups if g.startswith('grown'))}"
            corpus.groups[name] = part_of_new_start(start)
            group_of[start] = name
            new_groups.append(name)
        new_rows.append({"start": start, "moves": moves, "group": group_of[start], "before": before, "keep": keep})
    corpus.extend(new_rows)
    dev = sum(1 for r in new_rows if corpus.groups[r["group"]] == "dev")
    return {"added": len(new_rows), "dev": dev, "held": len(new_rows) - dev, "known": c["known"], "exam": c["exam"], "over": c["over"], "uncovered": c["uncovered"], "new_groups": new_groups,
            "rows": new_rows}


def drill_rules(corpus: "CP.Corpus", rules: list) -> list:
    """Each rule on both parts: [{name, dev, held, wrong_rows (corpus row indices, sorted)}]; the cards must be built."""
    dev = CP.drill_many(rules, corpus, part="dev")
    held = CP.drill_many(rules, corpus, part="held")
    out = []
    for r, d, h in zip(rules, dev, held):
        rows = sorted({corpus.row_of[ci] for ci in d["wrong_at"] + h["wrong_at"]})
        out.append({"name": r["name"], "dev": d, "held": h, "wrong": d["wrong"] + h["wrong"], "wrong_rows": rows})
    return out


def grow_and_withdraw(game: str, corpus: "CP.Corpus", log: str, *, variant: str, labeller, spec: dict, vocabulary: "str | None" = None, judge: "str | None" = None, seconds: float = 0.2,
                      wake_kw: "dict | None" = None, problems_file=None, cycle: int = 1, exclude=frozenset(), grown_path=None, max_check: int = MAX_CHECK_PER_RULE, say=print) -> dict:
    """After a cycle's wake (`log`, played with the cycle's final snapshot): the positions where a rule fired are labelled and added to `corpus` (`grow`), every rule of the store (`game_store`
    must point at the sandbox) is drilled on both parts of the grown corpus, a rule with any wrong firing is WITHDRAWN (its lemma file removed, a snapshot published with the wake's valuation
    record `judge`), and each position where it was wrong, up to `max_check` per rule, is asked of the wake's decision with the new snapshot: wrong there too -> a problem of source "withdrawn" in
    `problems_file` (when given); else only recorded in the report. `grown_path`: where the grown corpus is saved (never the shared cached file). Prints
      OBS corpus grow cycle C: positions with a firing F, added A (dev D, held H), rules drilled R, withdrawn W [names]
      OBS corpus grow detail cycle C: ...                                  (what was not added and why, the corpus size)
      OBS corpus withdrawn cycle C: rule N, wrong firings W (dev D, held H), e.g. <position> where it names a losing move (losing moves [..]); the decision without the rule is wrong at X of the Y
        positions asked, new problems P
    -> {positions, added, dev, held, rules, drilled, withdrawn [{name, wrong, dev, held, example, asked, wrong_without, new_problems, recorded}], snapshot (the head after), grown_path, size}"""
    vocab_json = json.dumps(GS.vocabulary(game, vocabulary))
    vkw = {} if vocabulary is None else {"vocabulary": vocabulary}
    corpus.cards(spec, vocab_json)                                    # (built before the growth, so that only the new rows' cards are built)
    fired = fired_positions(log)
    g = grow(corpus, fired, labeller, exclude=exclude, say=say)
    rules = GS.read_lemmas(game, vocabulary)
    drilled = drill_rules(corpus, rules) if rules else []
    bad = [d for d in drilled if d["wrong"] > 0]
    withdrawn = []
    for d in bad:
        GS.remove_lemma(game, d["name"])
    sid = None
    if bad:
        sid = GS.publish(game, **({"judge": GS.judge_of(game, judge)} if judge else {}), **vkw)
    head = sid or GS.head(game)
    # the wake's decision without the withdrawn rules, at the positions where they were wrong
    asks: dict = {}
    for d in bad:
        for ri in d["wrong_rows"][:max_check]:
            asks.setdefault(ri, None)
    order = sorted(asks)
    res = PS.check([{"start": corpus.rows[ri]["start"], "moves": list(corpus.rows[ri]["moves"])} for ri in order], variant=variant, store=str(GS.path_of(game, head)) if head else None, spec=spec,
                   judge=labeller, seconds=seconds, wake_kw=wake_kw) if order else []
    decided = dict(zip(order, res))
    rows = PS.load_problems(problems_file) if problems_file else []
    for d in bad:
        ex_ri = d["wrong_rows"][0]
        ex = corpus.rows[ex_ri]
        asked = d["wrong_rows"][:max_check]
        wrong_without = [ri for ri in asked if decided[ri]["move"] is not None and corpus.rows[ri]["keep"].get(decided[ri]["move"]) is False]
        found = [{"start": corpus.rows[ri]["start"], "moves": list(corpus.rows[ri]["moves"]), "played": decided[ri]["move"], "source": "withdrawn"} for ri in wrong_without]
        rows, new = PS.merge_problems(rows, found, cycle)
        at = {(corpus.rows[ri]["start"], tuple(corpus.rows[ri]["moves"])): ri for ri in wrong_without}
        PS.record(new, [{**decided[at[PS.key_of(n)]], "keeps": False} for n in new], cycle, "after")
        withdrawn.append({"name": d["name"], "wrong": d["wrong"], "dev": d["dev"]["wrong"], "held": d["held"]["wrong"], "example": {"start": ex["start"], "moves": list(ex["moves"])},
                          "asked": len(asked), "wrong_without": len(wrong_without), "new_problems": len(new), "recorded": [{"start": corpus.rows[ri]["start"], "moves": list(corpus.rows[ri]["moves"])}
                                                                                                                    for ri in asked if ri not in wrong_without]})
        say(f"OBS corpus withdrawn cycle {cycle}: rule {d['name']}, wrong firings {d['wrong']} (dev {d['dev']['wrong']}, held {d['held']['wrong']}), e.g. {position_text(ex['start'], ex['moves'])} "
            f"(losing moves {sorted(m for m, k in ex['keep'].items() if not k)}); the decision without the rule is wrong at {len(wrong_without)} of the {len(asked)} positions asked, new problems {len(new)}")
    if problems_file and withdrawn:
        PS.save_problems(problems_file, rows)
    names = [w["name"] for w in withdrawn]
    say(f"OBS corpus grow cycle {cycle}: positions with a firing {len(fired)}, added {g['added']} (dev {g['dev']}, held {g['held']}), rules drilled {len(rules)}, withdrawn {len(names)} {names}")
    say(f"OBS corpus grow detail cycle {cycle}: already in the corpus {g['known']}, exam positions left out {g['exam']}, game over {g['over']}, not covered by the labeller {g['uncovered']}, "
        f"new start groups {g['new_groups']}, corpus size now {len(corpus)}")
    saved = None
    if grown_path:
        saved = corpus.save(grown_path)
    return {"cycle": cycle, "positions": len(fired), "added": g["added"], "dev": g["dev"], "held": g["held"], "known": g["known"], "exam": g["exam"], "uncovered": g["uncovered"],
            "new_groups": g["new_groups"], "rules": [r["name"] for r in rules], "drilled": len(rules), "withdrawn": withdrawn, "snapshot": head, "grown_path": saved, "size": len(corpus)}
