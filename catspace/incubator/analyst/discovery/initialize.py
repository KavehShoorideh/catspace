"""PLAN 78.24 C THE LOOP FOR ONE MATERIAL SIZE, ON A SANDBOX STORE: wake (from the size's starts), sleep (a rank source is the judge; lemmas filed, the next snapshot published), wake again
with that snapshot, then the audit of what was mined and how it was applied. Generic: the game is a parameter (its position codec, vocabulary and rank source are the only game knowledge).

The store is a SANDBOX (`store_root`, default paths.derived("init_store")): `game_store.LEARNED_ROOT` / `SNAPSHOT_ROOT` point into it for the run and are restored afterwards; the game's
vocabulary files are copied in once from the real rules/learned/<game>/ (never the other way); lemmas already in the sandbox are kept, so the store accumulates across sizes and runs.

THIS IS A MECHANISM CHECK, NOT A STRENGTH CLAIM: one size, a few dozen games, a rank source's regret rate on a sample of positions. Two wakes of a few games differ by chance alone.

  python -m catspace.incubator.analyst.discovery.initialize chess --size 3 --teacher <uci> [--teacher-ms 100] [--teacher-option NAME=VALUE ...] [--store-root <dir>] [--judge <record>]
         [--games-per-start 4] [--limit N] [--set NAME=VALUE ...] [--vocabulary NAME] [--gate NAME=VALUE ...] [--held-gate NAME=VALUE ...]
         [--teacher-mate-bucket N] [--teacher-cache FILE] [--tag T] [--wdl [--cycles K] [--wdl-judge-depth D] [--wdl-judge-cache FILE] [--wdl-set NAME=VALUE ...]]
         [--playouts [--playout-set NAME=VALUE ...]]       (plan 78.28; the judge is the --teacher engine at --wdl-judge-depth, default 12)
         [--targeted [--targeted-set NAME=VALUE ...]]       (plan 78.29, with --playouts: after the lemma sleep, one targeted rule per open problem; default off)
         [--exact] [--induce [--induce-set NAME=VALUE ...] [--corpus-n N]] [--exam PATH] [--exam-size N]       (plan 78.32: the exact labeller judges the playouts and problems; rules induced
          from the exact corpus for the open problems; a fixed exam graded before cycle 1 and after every cycle; defaults off)
         [--corpus-problems N] [--grow-corpus]       (plan 78.34, both need --exact: N problems per cycle from the development part of the corpus; the corpus grows with the positions where a rule
          fired and a rule with a wrong firing on it is withdrawn; --induce-set max_problems=N head_tiers=relative,named)
         [--suggest-bar X]       (plan 78.36, with --seed-rules: response rules failing zero-wrong but right at least X of the time enter as suggestions; default off)
         [--conversion N]       (plan 78.36, with an exam: after every exam grading, N games played out from exam positions against the exact labeller's defence: `OBS conversion ..`; default 0 = off)
         [--usage-split]       (plan 78.35, with an exam: the use / generation split after every exam grading, and the rules by kind at the end; --induce-set name_bodies=N names an admitted body of N literals or more as a word)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import shutil
import sys
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import abstraction as AB
from catspace.incubator.analyst.discovery import apply_audit as AU
from catspace.incubator.analyst.discovery import conversion as CV
from catspace.incubator.analyst.discovery import corpus as CP
from catspace.incubator.analyst.discovery import corpus_problems as CPB
from catspace.incubator.analyst.discovery import curriculum as CU
from catspace.incubator.analyst.discovery import detectors as D
from catspace.incubator.analyst.discovery import exact as EX
from catspace.incubator.analyst.discovery import exam as EXM
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import induce as IND
from catspace.incubator.analyst.discovery import judges as JG
from catspace.incubator.analyst.discovery import playout_sleep as PS
from catspace.incubator.analyst.discovery import sleep as SL
from catspace.incubator.analyst.discovery import targeted as TG
from catspace.incubator.analyst.discovery import wake as WK
from catspace.incubator.analyst.discovery import usage_split as US
from catspace.incubator.analyst.discovery import wdl_sleep as WS
from catspace.incubator.analyst.discovery import word_registry as WR
from catspace.io import paths


def inside(path, root) -> Path:
    """`path` resolved, asserted to lie inside `root`: every write of the sandbox goes through this."""
    p, r = Path(path).resolve(), Path(root).resolve()
    assert p == r or r in p.parents, f"{p} is outside the sandbox {r}"
    return p


def prepare_sandbox(game: str, root) -> Path:
    """Create the sandbox and copy the game's vocabulary files (vocabulary*.json, graph_vocabulary.json) from the REAL rules/learned/<game>/ when the sandbox has none yet."""
    root = Path(root)
    (root / "snapshots").mkdir(parents=True, exist_ok=True)
    dest = inside(root / "learned" / game, root)
    dest.mkdir(parents=True, exist_ok=True)
    real = GS.RULES / "learned" / game
    for src in sorted(real.glob("vocabulary*.json")) + sorted(real.glob("graph_vocabulary.json")):
        target = inside(dest / src.name, root)
        if not target.exists():
            shutil.copyfile(src, target)
    return root


def typed(field: dataclasses.Field, text: str):
    """`text` as the type of the dataclass field (read from its default; an unset default -- None -- tries int, float, then text)."""
    d = field.default
    if isinstance(d, bool):
        if text.lower() not in ("0", "1", "true", "false", "yes", "no"):
            raise ValueError(f"{field.name}: {text!r} is not a boolean")
        return text.lower() in ("1", "true", "yes")
    if isinstance(d, (int, float, str)):
        return type(d)(text)
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def settings_from(overrides: "dict | None") -> "SL.Settings":
    """`Settings` with `overrides` applied over its fields; an unknown name is an error."""
    fields = {f.name: f for f in dataclasses.fields(SL.Settings)}
    bad = sorted(set(overrides or {}) - set(fields))
    if bad:
        raise ValueError(f"unknown sleep setting(s) {bad}; the fields are {sorted(fields)}")
    return SL.Settings(**(overrides or {}))


def settings_pairs(pairs: list) -> dict:
    """['NAME=VALUE', ...] -> {NAME: value typed from the Settings field}."""
    fields = {f.name: f for f in dataclasses.fields(SL.Settings)}
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        if k not in fields:
            raise ValueError(f"unknown sleep setting {k!r}; the fields are {sorted(fields)}")
        out[k] = typed(fields[k], v)
    return out


def outcomes(games: list) -> dict:
    """{result: games}: the game's outcome as text, or `unfinished`."""
    out: dict = {}
    for g in games:
        k = "unfinished" if g["unfinished"] else str(g["outcome"])
        out[k] = out.get(k, 0) + 1
    return out


def regret_rate(game: str, log: str, source, *, sample: int = 200, seed: int = 1, variant: "str | None" = None) -> dict:
    """The source's regret on a seeded sample of the log's decisions: the share whose played move's grade is below the best grade. A decision the source does not rank completely
    (the played move missing) is not counted. -> {sampled, ranked, regret, rate}."""
    variant = variant or game
    decisions, games = SL.read_log(log)
    keys = sorted(k for k in decisions if (k[0], k[1]) in games)
    rng = random.Random(seed)
    pick = keys if len(keys) <= sample else sorted(rng.sample(keys, sample))
    grade, ranked, regret = JG.grader(source), 0, 0
    for run, g, ply in pick:
        row = games[(run, g)]
        ranking = D.ask(source, variant, list(row["moves"][:ply]), row.get("start"))
        score = {m: int(grade(s)) for m, s in ranking}
        played = decisions[(run, g, ply)]["move"]
        if played not in score:
            continue
        ranked += 1
        regret += score[played] < max(score.values())
    return {"sampled": len(pick), "ranked": ranked, "regret": regret, "rate": round(regret / ranked, 4) if ranked else None}


def corpus_positions(logs: list, playout_games: list) -> list:
    """Plan 78.30: the corpus the sleep may look for evidence in: EVERY position of the wake log(s)' games (group `run:game`, as the sleep's own split reads them) and every position the
    playout games visited (group `playout:<k>`, one group per playout). -> [{start, moves, group}], deduplicated by position within a group."""
    out, seen = [], set()
    for lg in logs:
        _decisions, games = SL.read_log(lg)
        for (run, g), row in sorted(games.items()):
            for ply in range(len(row["moves"])):
                k = (row.get("start"), tuple(row["moves"][:ply]))
                if k not in seen:
                    seen.add(k)
                    out.append({"start": row.get("start"), "moves": list(row["moves"][:ply]), "group": f"{run}:{g}"})
    for k, pg in enumerate(playout_games):
        full = list(pg["prefix"]) + list(pg["moves"])
        for i in range(len(pg["prefix"]), len(full) + 1):
            key = (pg.get("start"), tuple(full[:i]))
            if key not in seen:
                seen.add(key)
                out.append({"start": pg.get("start"), "moves": full[:i], "group": f"playout:{k}"})
    return out


def forced_positions(rows: list, turning_extras: list) -> list:
    """Plan 78.29: every OPEN problem of the problem file (the carried ones too) as a forced seed of the lemma sleep ({id, start, moves, played, group, force}); the move played is the
    decision's last (the move that loses the verdict now), else the playout's. The turning positions equal to one of them are left out of `turning_extras` (no duplicate card).
    -> (forced, the remaining turning extras)."""
    forced, keys = [], set()
    for k, r in enumerate(rows):
        h = r["history"][-1] if r.get("history") else None
        if h is not None and h["status"] != "open":
            continue
        keys.add(PS.key_of(r))
        forced.append({"id": f"problem:{k}:{len(r['moves'])}", "start": r.get("start"), "moves": list(r["moves"]), "played": (h or {}).get("move") or r.get("played"), "group": f"problem:{k}", "force": True})
    return forced, [e for e in turning_extras if (e.get("start"), tuple(e["moves"])) not in keys]


PLAYOUT_DEFAULTS = {"positions": 20, "per_position": 2, "max_plies": 120, "max_turning": 20, "max_scan": 60, "check_seconds": None}


def playout_before(game: str, log: str, snapshot: str, cycle: int, *, variant: str, specs: dict, judge, opts: dict, seconds: float, seed: int, problems_file, wake_kw: dict, say=print) -> dict:
    """Plan 78.28, before the lemma sleep of `cycle`: playouts with the rules of `snapshot` from the log's positions (match_share before), their turning moves, and the problem file checked
    with that snapshot (the wake's own decision, `seconds` and `wake_kw` as the wake plays). New candidate problems that the decision already keeps are dropped. -> {extra_positions,
    match, turnings, problems (the rows), new, path, store, positions}."""
    o = {**PLAYOUT_DEFAULTS, **opts}
    store = str(GS.path_of(game, snapshot))
    r = PS.sleep_playouts(game, [log], variant=variant, store=store, judge=judge, spec=specs, rollout="rules", positions=int(o["positions"]), per_position=int(o["per_position"]), seed=seed,
                          max_plies=int(o["max_plies"]), max_turning=int(o["max_turning"]), max_scan=int(o["max_scan"]), say=say)
    rows, new = PS.merge_problems(PS.load_problems(problems_file), [{"start": t["start"], "moves": t["moves"], "played": t["played"]} for t in r["turnings"]], cycle)
    secs = float(o["check_seconds"] or seconds)
    res = PS.check(rows, variant=variant, store=store, spec=specs, judge=judge, seconds=secs, wake_kw=wake_kw)
    PS.record(rows, res, cycle, "before")
    fresh = {PS.key_of(n) for n in new}
    dropped = [x for x in rows if PS.key_of(x) in fresh and x["history"][-1]["status"] == "solved"]
    gone = {PS.key_of(x) for x in dropped}
    rows = [x for x in rows if PS.key_of(x) not in gone]
    say(f"OBS playout_sleep candidates {game}: turning positions {len(r['turnings'])}, new problems {len(new) - len(dropped)}, candidates the engine's own decision already keeps (dropped) {len(dropped)}, "
        f"problems carried {len(rows) - len(new) + len(dropped)}")
    return {"games": r["games"], "extra_positions": r["extra_positions"], "match": r["match"], "turnings": r["turnings"], "problems": rows, "new": len(new) - len(dropped), "path": problems_file, "store": store,
            "positions": r["positions"], "dropped": len(dropped), "cycle": cycle}


def playout_after(game: str, log: str, snapshot: str, pre: dict, *, variant: str, specs: dict, judge, opts: dict, seconds: float, seed: int, wake_kw: dict, say=print) -> dict:
    """Plan 78.28, after the lemma sleep: the problems checked again with the new snapshot, the counts printed and the file saved, and the same playouts played again with the new
    snapshot's rules (match_share after). -> the cycle's playout report."""
    o = {**PLAYOUT_DEFAULTS, **opts}
    store = str(GS.path_of(game, snapshot))
    cycle = pre["cycle"]
    rows = pre["problems"]
    res = PS.check(rows, variant=variant, store=store, spec=specs, judge=judge, seconds=float(o["check_seconds"] or seconds), wake_kw=wake_kw)
    PS.record(rows, res, cycle, "after")
    PS.save_problems(pre["path"], rows)
    counts = PS.problem_counts(rows, cycle, pre["new"])
    PS.say_counts(counts, say)
    r = PS.sleep_playouts(game, [log], variant=variant, store=store, judge=judge, spec=specs, rollout="rules", positions=int(o["positions"]), per_position=int(o["per_position"]), seed=seed,
                          max_plies=int(o["max_plies"]), max_turning=int(o["max_turning"]), max_scan=int(o["max_scan"]), say=say)
    say(f"OBS playout_sleep cycle {game}: match share before {pre['match']['share']} ({pre['match']['matched']}/{pre['match']['finished']}), after {r['match']['share']} "
        f"({r['match']['matched']}/{r['match']['finished']}); turning moves before {len(pre['turnings'])}, after {len(r['turnings'])}; snapshot {pre['store'].rsplit('/', 1)[-1][:-5]} -> {snapshot}")
    return {"match_before": pre["match"], "match_after": r["match"], "turnings_before": len(pre["turnings"]), "turnings_after": len(r["turnings"]), "problems": counts,
            "problem_file": str(pre["path"]), "extra_positions": len(pre["extra_positions"])}


def run_size(game: str, size: int, *, source, store_root=None, variant: "str | None" = None, judge: "str | None" = None, games_per_start: int = 4, seconds: float = 0.2,
             random_plies: int = 2, max_plies: int = 80, limit: "int | None" = None, seed: int = 1, sleep_settings: "dict | None" = None, vocabulary: "str | None" = None,
             tag: "str | None" = None, sample: int = 200, wdl: bool = False, cycles: int = 1, wdl_judge=None, wdl_settings: "dict | None" = None, playouts: bool = False, playout_judge=None, playout_settings: "dict | None" = None, targeted: bool = False, targeted_settings: "dict | None" = None,
             force_problems: bool = True, standard: str = "score", induce: bool = False, induce_settings: "dict | None" = None, drill_sleep: bool = False, seed_rules: "str | None" = None, exam: "str | None" = None, exam_size: "int | None" = None,
             exact: bool = False, corpus_n: int = 30000, corpus_problems: int = 0, grow_corpus: bool = False, usage_split: bool = False, conversion: int = 0, suggest_bar: "float | None" = None, say=print) -> dict:
    """The loop for one size: starts -> publish -> wake a -> sleep (source as judge) -> wake b -> audit -> compare. Returns everything and writes it as JSON next to the logs.

    Plan 78.27: `wdl=True` plays every wake with the win / draw / loss estimate and the sandbox's table (`<root>/learned/<game>/wdl_table.json`, none until a sleep writes it), prints the
    wake's gaps, and after the lemma sleep runs `wdl_sleep.sleep_wdl` (`wdl_judge`: a judge of readings, training only, or None; `wdl_settings`: its keyword arguments, e.g. deep_seconds,
    max_deep, sample, max_settle) which carries the previous table forward; the next wake uses the new snapshot and table. `cycles=k` repeats sleep then wake k times and adds a `cycles`
    list to the report. With wdl=False and cycles=1 nothing changes.

    Plan 78.28: `playouts=True` adds the playout sleep to every cycle (`playout_judge`: a wdl judge, training only, falling back to `wdl_judge`; `playout_settings`: positions,
    per_position, max_plies, max_turning, max_scan, check_seconds). Per cycle: playouts from sample positions of the wake's log with the current snapshot's rules, match_share (before);
    the turning moves; the problem file (`<root>/learned/<game>/playout_problems.json`, training data the wake never reads) is checked with the current snapshot (the wake's own
    decision); the turning positions go into the lemma sleep as extra seeds; the problems are checked again with the new snapshot; the playouts are played again with it (match_share
    after). The report's `cycles[i]["playouts"]` holds all of it. Off, nothing changes.

    Plan 78.29: `targeted=True` (needs `playouts=True`) runs `targeted.solve_problems` after the lemma sleep and before the problems are checked again: for each open problem the targeted miner
    looks for the simplest rule the problem's counter-positions allow and admits it only when the problem's own check passes (`targeted_settings`: its settings, see targeted.SETTINGS). The
    rules are written and a snapshot published as the lemma sleep does; the final check reads the new snapshot; `cycles[i]["targeted"]` and `cycles[i]["playouts"]["targeted"]` hold the counts.
    Off (the default), nothing changes.

    Plan 78.29 (second route): `force_problems=True` (the default with playouts on) hands EVERY open problem of the problem file, the carried ones too, to each lemma sleep as a forced seed
    (`sleep.sleep(extra_positions=[... "force": True])`: always in the development half, always shown to the rank source, regret raised when the move played is not best, climbed first; the
    admission gates are unchanged). `--no-force-problems` turns it off.

    Plan 78.31: `standard="result"` (CLI `--standard result`, with the rank source built with grade_mode "result") switches EVERYTHING together to the owner's definition of a mistake -- a change of
    RESULT: the rank source grades by win / draw / loss class, the lemma sleep's gates, the forced seeds, the evidence, the audit's claim-held / played-optimal and the regret count of `OBS init compare`
    all read it, and every named move must be in the best class. The first OBS line of each tool prints the standard. Default "score": nothing changes.

    Plan 78.32: `exact=True` makes the EXACT labeller of the game (exact.labeller_of: Syzygy for chess; training only) the judge of the playouts and of the problems instead of `playout_judge`. `induce=True`
    (needs playouts and exact) runs `induce.solve_problems` after the lemma sleep (and the targeted miner) on the open problems, against the exact CORPUS (corpus.ensure_corpus, `corpus_n` positions, cached
    under paths.derived; the exam's positions are taken out of it), before the problems are checked again. `exam` (a path; with `exam_size` positions, 300 when not given, built once from the corpus when the
    file is missing; needs exact) is graded with the wake's own decision before cycle 1 (twice, to print the clock's spread) and with the final snapshot of every cycle; the report gains `exam` (the
    curve). The exam is only graded, never used to admit anything. Off, nothing changes.

    Plan 78.34: `corpus_problems=N` (needs exact and playouts) takes the wake's own decision, before the lemma sleep, on N sampled non-trivial positions of the corpus's DEVELOPMENT part (different
    positions each cycle: corpus_problems.sample_dev); those where it loses the exact class join the problem rows with source "corpus" after the lemma sleep, before induction. `grow_corpus=True`
    (needs exact) reads each cycle's wake log, labels the positions where a rule fired and adds them to the corpus (its group the game's start's group), drills every rule of the sandbox store on
    both parts of it and WITHDRAWS a rule with any wrong firing (a new snapshot; its wrong positions become problems of source "withdrawn"); this happens after the cycle's wake and BEFORE the exam of
    the cycle is graded, so the exam grades the snapshot that survives; the next cycle's playouts and problem checks play that snapshot (`wake_b.survivor`). The grown corpus is kept in memory and
    saved to `<root>/learned/<game>/corpus_grown.json`, never over the cached corpus file. Off, nothing changes.

    Plan 78.36: `conversion=N` (needs an exam) plays N games out from exam positions after every exam grading (conversion.py: the wake's own decision for the side to move at the start, the exact
    labeller's defence for the other side) and prints `OBS conversion <label>: ..`; the report's exam curve gains the counts. Off (0), nothing changes.

    Plan 78.35: `usage_split=True` (needs an exam) runs usage_split.split on every exam grading whose snapshot holds rules, reusing that grading's decisions (nothing is graded a second time), and
    prints `OBS usage split <label>: ...` and, with the rules' layers (word_registry.layers), `OBS usage split by rule kind <label>: ...`; once at the end `OBS rules by kind: general G (hand a,
    machine b), game C (hand c, machine d)` for the final snapshot's rules; the report gains `usage_split`. Off, nothing changes."""
    if cycles < 1 or (cycles > 1 and not (wdl or playouts)):
        raise ValueError("cycles must be at least 1, and more than one needs wdl=True or playouts=True")
    if corpus_problems < 0:
        raise ValueError("corpus_problems is a count of positions, 0 or more")
    if (corpus_problems or grow_corpus) and not exact:
        raise ValueError("corpus_problems and grow_corpus need the exact corpus (exact=True): the labels of the positions are what says a decision is wrong and a rule is withdrawn")
    if corpus_problems and not playouts:
        raise ValueError("corpus_problems needs playouts=True: its problems join the problem file of the playout sleep")
    if induce and not (playouts and exact):
        raise ValueError("induce=True needs playouts=True (the problems it solves) and exact=True (the labels it learns from)")
    if (exam or exam_size) and not exact:
        raise ValueError("an exam needs exact=True: the exact labeller says whether a decision keeps the result")
    if conversion and not (exam or exam_size):
        raise ValueError("conversion=N needs an exam (exam or exam_size): the games start from the exam's positions")
    if usage_split and not (exam or exam_size):
        raise ValueError("usage_split=True needs an exam (exam or exam_size): it reads the decisions of an exam grading")
    if targeted and not playouts:
        raise ValueError("targeted=True needs playouts=True: the problems it solves are the playout sleep's")
    if getattr(source, "grade_mode", "score") != standard:
        raise ValueError(f"standard {standard!r} but the rank source grades by {getattr(source, 'grade_mode', 'score')!r}: build the source with grade_mode={standard!r}")
    labeller = EX.labeller_of(game) if exact else None
    pjudge = labeller if exact else (playout_judge or wdl_judge)
    if playouts and pjudge is None:
        raise ValueError("playouts=True needs a judge of verdicts (playout_judge or wdl_judge): the outside engine, at sleep only")
    variant = variant or game
    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    root = prepare_sandbox(game, store_root if store_root is not None else paths.derived("init_store"))
    vkw = {} if vocabulary is None else {"vocabulary": vocabulary}
    say("OBS init standard: " + SL.standard_text(SL.Settings(standard=standard), source))
    settings = settings_from({"standard": standard, "judge_by": "source", **({} if judge is None else {"judge": judge}), **({} if vocabulary is None else {"vocabulary": vocabulary}), **(sleep_settings or {})})
    settings = SL.apply_standard(settings, source)
    old = GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT
    GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT = inside(root / "learned", root), inside(root / "snapshots", root)
    rep: dict = {"game": game, "size": size, "tag": tag, "store_root": str(root), "settings": {"games_per_start": games_per_start, "seconds": seconds, "random_plies": random_plies,
                 "max_plies": max_plies, "limit": limit, "seed": seed, "sleep": {k: (v.as_dict() if hasattr(v, "as_dict") else v) for k, v in (sleep_settings or {}).items()}, "vocabulary": vocabulary, "judge": judge}}
    try:
        say(f"OBS init sandbox {game}: store {root}, lemmas already there {len(GS.read_lemmas(game, GS.ANY))}")
        starts = CU.starts(game, size, variant=variant, limit=limit, seed=seed)
        rep["starts"] = len(starts)
        say(f"OBS init starts {game}: size {size}, {len(starts)} start positions")
        specs = GS.spec_texts(game, vocabulary)
        corpus = exam_data = None
        if induce or exam or exam_size or corpus_problems or grow_corpus:
            corpus = CP.ensure_corpus(game, size, variant=variant, n=corpus_n, seed=seed, labeller=labeller, say=say)
            labeller.save()
        if exam or exam_size:
            n_exam = int(exam_size or 300)
            exam_data = EXM.ensure_exam(game, size, n=n_exam, seed=seed, corpus=corpus, path=str(exam) if exam else None, say=say)
            corpus = corpus.excluding(EXM.exam_keys(exam_data))
        exam_prev, exam_curve, usage_rows = None, [], []

        def grade_exam(label: str, snapshot: str, twice: bool = False) -> dict:
            nonlocal exam_prev
            kw = dict(variant=variant, store=str(GS.path_of(game, snapshot)), spec=specs, seconds=seconds, wake_kw={}, labeller=labeller)
            if twice:
                g, _g2, _d = EXM.grade_twice(exam_data, say=say, **kw)
            else:
                g = EXM.grade(exam_data, **kw)
            cmp = EXM.compare(exam_prev, g)
            EXM.say_exam(label, g, cmp, say)
            exam_prev = g
            if usage_split and GS.info(game, snapshot)["lemmas"]:
                sp = US.split(exam_data, g, variant=variant, snapshot_path=str(GS.path_of(game, snapshot)), specs=specs, labeller=labeller)
                US.say_split(label, sp, WR.layers(game, vocabulary), say)
                usage_rows.append({"label": label, **{k: sp[k] for k in ("kept", "failed", "by_group")}})
            exam_curve.append({"label": label, "snapshot": snapshot, "kept": g["kept"], "total": g["total"], "by_class": g["by_class"], "gained": len(cmp["gained"]), "lost": len(cmp["lost"])})
            if conversion:
                cv = CV.convert(exam_data, variant=variant, store=str(GS.path_of(game, snapshot)), spec=specs, seconds=seconds, wake_kw={}, labeller=labeller, n=conversion)
                CV.say_conversion(label, cv, say)
                exam_curve[-1]["conversion"] = {k: cv[k] for k in ("games", "win_games", "draw_games", "kept", "kept_win", "kept_draw", "finished", "mean_plies", "median_drop_ply", "max_plies")}
            return g

        table_path = inside(root / "learned" / game / "wdl_table.json", root)

        def wake(label: str, run: str) -> tuple:
            sid = GS.publish(game, **({"judge": GS.judge_of(game, judge)} if judge else {}), **vkw)
            log = paths.experiment(f"wake_{game}_init{size}_{tag}_{label}.jsonl")
            if Path(log).exists():
                Path(log).unlink()
            t0 = time.perf_counter()
            out = WK.play_self(variant, games_per_start * len(starts), seconds=seconds, random_plies=random_plies, max_plies=max_plies, seed=seed,
                               store=str(GS.path_of(game, sid)), specs=specs, log=log, run=run, ties=True, starts=starts,
                               **({"wdl": True, "wdl_table": str(table_path) if table_path.exists() else None} if wdl else {}))
            res = outcomes(out)
            say(f"OBS init wake {label} {game}: snapshot {sid} ({len(GS.info(game, sid)['lemmas'])} lemma(s)), games {len(out)}, plies {sum(len(g['moves']) for g in out)}, "
                f"outcomes {res}, {time.perf_counter() - t0:.0f} s -> {log}")
            return {"snapshot": sid, "log": log, "games": len(out), "plies": sum(len(g['moves']) for g in out), "outcomes": res, **({"wdl_table": str(table_path) if table_path.exists() else None} if wdl else {})}, log

        def gaps_of(label: str, log: str) -> dict:
            g = WS.gaps(log)
            say(f"OBS init gaps {game} wake {label}: decisions {g['decisions']}, " + ", ".join(f"{k} {v}" for k, v in g["counts"].items()) + f", modes {g['modes']}, table {g['table']}, worklist {len(g['worklist'])}")
            return {k: v for k, v in g.items() if k != "worklist"} | {"worklist": len(g["worklist"])}

        if seed_rules:
            # plan 78.32, the hand step: rules proposed from OUTSIDE the loop (a person, or the assistant by hand when the loop goes in circles) enter the sandbox store only through the
            # same admission test as an induced rule: zero wrong firings on both parts of the exact corpus. The exam is graded before them so their effect is seen on its own.
            if corpus is None:
                raise ValueError("seed_rules needs the exact corpus (exact=True with induce or an exam): a proposed rule is admitted only by the full drill")
            if exam_data is not None:
                grade_exam("before the seed rules", GS.publish(game, **({"judge": GS.judge_of(game, judge)} if judge else {}), **vkw))
            proposed = json.loads(Path(seed_rules).read_text())["rules"]
            if suggest_bar is None:
                kept, dropped = IND.drill_admit(proposed, corpus, specs, json.dumps(GS.vocabulary(game, vocabulary)), say=say)
                suggested = []
            else:
                # plan 78.36: a response rule that fails zero-wrong but is right at least `suggest_bar` of the time enters as a SUGGESTION (the engine uses its moves only where the claim search classes them a win)
                kept, dropped, suggested = IND.drill_admit(proposed, corpus, specs, json.dumps(GS.vocabulary(game, vocabulary)), say=say, suggest_bar=suggest_bar)
            for l in kept + suggested:
                GS.check_lemma(l)
                GS.write_lemma(game, l)
            rep["seed_rules"] = {"file": str(seed_rules), "kept": [l["name"] for l in kept], "dropped": [l["name"] for l in dropped], **({"suggested": [l["name"] for l in suggested]} if suggest_bar is not None else {})}
        rep["wake_a"], log_a = wake("a", f"i{size}{tag}a")
        if exam_data is not None:
            grade_exam(f"before cycle 1 ({len(GS.info(game, rep['wake_a']['snapshot'])['lemmas'])} lemmas)", rep["wake_a"]["snapshot"], twice=True)
        log_b = log_a
        rep["cycles"] = []
        for c in range(cycles):
            log_prev, wake_prev = log_b, rep["wake_b"] if c else rep["wake_a"]
            cyc: dict = {"cycle": c + 1}
            if wdl:
                cyc["gaps_before"] = gaps_of("a" if c == 0 else f"w{c}", log_prev)
            pre = None
            snap_prev = wake_prev.get("survivor") or wake_prev["snapshot"]                # (plan 78.34: the snapshot that survived the withdrawals after that wake)
            if playouts:
                pre = playout_before(game, log_prev, snap_prev, c + 1, variant=variant, specs=specs, judge=pjudge, opts=playout_settings or {}, seconds=seconds, seed=seed,
                                     problems_file=inside(root / "learned" / game / "playout_problems.json", root),
                                     wake_kw={"wdl": True, "wdl_table": str(table_path) if table_path.exists() else None} if wdl else {}, say=say)
            cpr = None
            if pre and corpus_problems:
                # plan 78.34: the wake's own decision, with the snapshot the problems of this cycle are checked with, on a sample of the development part; merged into the problem rows after the lemma sleep
                o = {**PLAYOUT_DEFAULTS, **(playout_settings or {})}
                cpr = CPB.corpus_problems(corpus, pre["problems"], cycle=c + 1, seed=seed, n=corpus_problems, variant=variant, store=pre["store"], spec=specs, judge=pjudge,
                                          seconds=float(o["check_seconds"] or seconds), wake_kw={"wdl": True, "wdl_table": str(table_path) if table_path.exists() else None} if wdl else {}, say=say)
                cyc["corpus_problems"] = {k: cpr[k] for k in ("cycle", "asked", "wrong", "new", "known")}
            extras = pre["extra_positions"] if pre else None
            if pre and force_problems:
                forced, rest = forced_positions(pre["problems"], pre["extra_positions"])
                extras = forced + rest
            kw: dict = {}
            if pre and force_problems:
                kw["corpus"] = corpus_positions([log_prev], pre.get("games", []))
                seed_pos = {e["id"]: (e["start"], e["moves"]) for e in extras if e.get("force")}
                kw["prefer"] = TG.keep_at_problem(variant, specs, json.dumps(GS.vocabulary(game, vocabulary)), pjudge, seed_pos)
            gate_by_drill = drill_sleep and corpus is not None
            sl = SL.sleep(game, log_prev, variant=variant, settings=settings, teacher=source, write=not gate_by_drill, say=say, **({"extra_positions": extras} if pre else {}), **kw)
            if gate_by_drill:
                # plan 78.32: the lemma sleep's rules pass the same admission test as the induced ones (zero wrong firings on the exact corpus) before any is filed
                kept, _dropped = IND.drill_admit(sl["lemmas"], corpus, specs, json.dumps(GS.vocabulary(game, vocabulary)), say=say)
                for l in kept:
                    GS.write_lemma(game, l)
                sl["lemmas"] = kept
                sl["snapshot"] = GS.publish(game, teacher_used=True, **({"judge": GS.judge_of(game, judge)} if judge else {}), **vkw)
            if induce or exam_data is not None:
                # the lemma sleep publishes without the valuation record the wake plays by; the snapshot the NEXT WAKE will play (the same content, so the same id) is what induction and the exam read
                GS.publish(game, **({"judge": GS.judge_of(game, judge)} if judge else {}), **vkw)
            if pre and force_problems and sl.get("snapshot"):
                TG.say_forced_rules(game, sl["lemmas"], pre["problems"], str(GS.path_of(game, sl["snapshot"])), variant=variant, spec=specs, judge=pjudge, seconds=seconds,
                                    vocab_json=json.dumps(GS.vocabulary(game, vocabulary)), say=say)
            rep["sleep"] = {"funnel": sl["funnel"], "lemmas": [l["name"] for l in sl["lemmas"]], "rejected": len(sl["rejected"]), "snapshot": sl.get("snapshot"), "judge": sl["judge"]}
            say(f"OBS init sleep {game}: lemmas admitted {len(sl['lemmas'])} {rep['sleep']['lemmas']}, rejected {len(sl['rejected'])}, funnel {sl['funnel']}")
            cyc["sleep"] = rep["sleep"]
            if cpr is not None:
                pre["problems"].extend(cpr["new_rows"])
            if playouts:
                wake_kw = {"wdl": True, "wdl_table": str(table_path) if table_path.exists() else None} if wdl else {}
                if targeted:
                    o = {**PLAYOUT_DEFAULTS, **(playout_settings or {})}
                    cyc["targeted"] = TG.solve_problems(game, pre["problems"], variant=variant, judge=pjudge, teacher=source, logs=[log_prev], seconds=float(o["check_seconds"] or seconds), vocabulary=vocabulary,
                                                        spec=specs, wake_kw=wake_kw, settings={**(targeted_settings or {}), "sleep_settings": settings}, say=say)
                if induce:
                    o = {**PLAYOUT_DEFAULTS, **(playout_settings or {})}
                    cyc["induce"] = IND.solve_problems(game, pre["problems"], corpus, variant=variant, labeller=labeller, vocabulary=vocabulary, spec=specs, seconds=float(o["check_seconds"] or seconds),
                                                       wake_kw=wake_kw, judge=judge, settings=induce_settings, say=say)
                after = GS.head(game) if (targeted or induce) else (sl.get("snapshot") or GS.head(game))
                cyc["playouts"] = playout_after(game, log_prev, after, pre, variant=variant, specs=specs, judge=pjudge, opts=playout_settings or {}, seconds=seconds, seed=seed, wake_kw=wake_kw, say=say)
                if targeted:
                    cyc["playouts"]["targeted"] = {k: cyc["targeted"][k] for k in ("total", "attempted", "solved", "still_open")} | {"rules_admitted": len(cyc["targeted"]["admitted"])}
                if corpus_problems or grow_corpus:
                    cyc["playouts"]["by_source"] = CPB.say_sources(pre["problems"], c + 1, say)
            if wdl:
                kw = WS.judge_kw_of_snapshot(str(GS.path_of(game, wake_prev["snapshot"])))
                opts = {"deep_seconds": 0.2, "max_deep": 20, "sample": 10, "max_settle": 10, **(wdl_settings or {})}
                cyc["wdl_sleep"] = WS.sleep_wdl(game, [log_prev], variant=variant, prior_table=str(table_path) if table_path.exists() else None, judge=wdl_judge, judge_kw=kw,
                                                out=str(table_path), seed=seed, say=say, **opts)

            def exam_of_cycle(c=c, cyc=cyc):
                cyc["exam"] = grade_exam(f"after cycle {c + 1}", GS.head(game))
                cyc["exam"] = {k: v for k, v in cyc["exam"].items() if k != "decisions"} | {"label": exam_curve[-1]["label"], "gained": exam_curve[-1]["gained"], "lost": exam_curve[-1]["lost"]}
            if exam_data is not None and not grow_corpus:
                exam_of_cycle()
            label = "b" if c == 0 else f"w{c + 1}"
            rep["wake_b"], log_b = wake(label, f"i{size}{tag}{label}")
            if grow_corpus:
                # plan 78.34: the corpus grows with the positions of THIS wake where a rule fired; a rule with a wrong firing on the grown corpus is withdrawn; then the exam grades what survives
                cyc["corpus_grow"] = CPB.grow_and_withdraw(game, corpus, log_b, variant=variant, labeller=labeller, spec=specs, vocabulary=vocabulary, judge=judge, seconds=seconds,
                                                           wake_kw={"wdl": True, "wdl_table": str(table_path) if table_path.exists() else None} if wdl else {},
                                                           problems_file=inside(root / "learned" / game / "playout_problems.json", root), cycle=c + 1,
                                                           exclude=EXM.exam_keys(exam_data) if exam_data is not None else frozenset(),
                                                           grown_path=inside(root / "learned" / game / "corpus_grown.json", root), say=say)
                if cyc["corpus_grow"]["withdrawn"]:
                    rep["wake_b"]["survivor"] = GS.head(game)
                if exam_data is not None:
                    exam_of_cycle()
            if wdl:
                cyc["gaps_after"] = gaps_of(label, log_b)
            cyc["wake"] = rep["wake_b"]
            rep["cycles"].append(cyc)
        if exam_data is not None:
            rep["exam"] = exam_curve
            say("OBS exam curve: " + ", ".join(f"{e['label']} {e['kept']}/{e['total']}" for e in exam_curve))
        if usage_split:
            rep["usage_split"] = usage_rows
            final = GS.head(game)
            if final:
                US.say_rules_by_kind(json.loads(GS.path_of(game, final).read_bytes())["lemmas"], WR.layers(game, vocabulary), say)
        if not (wdl or playouts or exam_data is not None):
            del rep["cycles"]
        rep["mined"] = AU.mined(game, vocabulary)
        for line in rep["mined"]:
            say("OBS mined " + line)
        if not rep["mined"]:
            say(f"OBS mined {game}: no lemma in the store")
        rep["audit"] = AU.audit(game, log_b, variant=variant, source=source, vocabulary=vocabulary, say=say, all_named=(standard == "result"))
        rep["compare"] = {"wake_a": {"outcomes": rep["wake_a"]["outcomes"], **regret_rate(game, log_a, source, sample=sample, seed=seed, variant=variant)},
                          "wake_b": {"outcomes": rep["wake_b"]["outcomes"], **regret_rate(game, log_b, source, sample=sample, seed=seed, variant=variant)}}
        a, b = rep["compare"]["wake_a"], rep["compare"]["wake_b"]
        say(f"OBS init compare {game} size {size}: wake a outcomes {a['outcomes']}, regret {a['regret']}/{a['ranked']} ({a['rate']}); wake b outcomes {b['outcomes']}, regret {b['regret']}/{b['ranked']} "
            f"({b['rate']}) under the {standard} standard; a mechanism check on {len(starts)} starts, not a strength claim")
    finally:
        GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT = old
        if labeller is not None:
            labeller.close()
    out = paths.experiment(f"init_{game}_{size}_{tag}.json")
    json.dump(rep, open(out, "w"), indent=1, default=str)
    say(f"OBS init report {game} size {size} -> {out}")
    rep["report"] = out
    return rep


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--teacher", required=True, help="the UCI rank source: the judge of the sleep and of the audit")
    ap.add_argument("--teacher-ms", type=int, default=100)
    ap.add_argument("--teacher-option", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--store-root", default=None)
    ap.add_argument("--judge", default=None, help="the store VALUATION record the engine plays by")
    ap.add_argument("--games-per-start", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a sleep Settings field (repeatable), typed from the field")
    ap.add_argument("--vocabulary", default=None)
    ap.add_argument("--gate", action="append", default=[], metavar="NAME=VALUE", help="an admission gate (repeatable), on the development and the held-out cards")
    ap.add_argument("--held-gate", action="append", default=[], metavar="NAME=VALUE", help="a gate on the held-out cards only (repeatable)")
    ap.add_argument("--teacher-mate-bucket", type=int, default=1, help="plies: mates within one bucket tie under the source's grade (1: exact)")
    ap.add_argument("--teacher-cache", default=None, help="a file that remembers the source's answers (judges.Cached): one verdict per position across the sleep, the audit and re-runs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wdl", action="store_true", help="plan 78.27: play the wakes with the win / draw / loss estimate and its results table, and sleep the table between wakes")
    ap.add_argument("--cycles", type=int, default=1, help="plan 78.27 (with --wdl): how many times to sleep then wake")
    ap.add_argument("--wdl-judge-depth", type=int, default=None, help="plan 78.27: the --teacher engine also judges the estimate's readings at this depth (training only; its numbers never enter the table)")
    ap.add_argument("--wdl-judge-cache", default=None, help="plan 78.27: a file that remembers the judge's readings by position, so a re-run repeats")
    ap.add_argument("--playouts", action="store_true", help="plan 78.28: per cycle, playouts to a real result diagnosed by the --teacher engine as judge (sleep only): turning moves seed the lemma sleep, problems are checked before and after it")
    ap.add_argument("--playout-set", action="append", default=[], metavar="NAME=VALUE", help="plan 78.28: positions, per_position, max_plies, max_turning, max_scan, check_seconds (repeatable)")
    ap.add_argument("--standard", default="score", choices=["score", "result"], help="plan 78.31: what a mistake is: 'score' (the rank source's graded score, any named move best) or 'result' (a change of win / draw / loss, every named move best); switches the sleep, forced seeds, evidence, audit and regret together")
    ap.add_argument("--decisive", type=int, default=None, help="plan 78.31 (--standard result): a score of at least this many centipawns is a win for the mover (default 300, read off Stockfish with SyzygyPath at three units)")
    ap.add_argument("--no-force-problems", action="store_true", help="plan 78.29: do not hand the open problems to the lemma sleep as forced seeds (with --playouts they are by default)")
    ap.add_argument("--targeted", action="store_true", help="plan 78.29 (with --playouts): after the lemma sleep, a targeted rule for each open problem, admitted only when the problem's own check passes")
    ap.add_argument("--targeted-set", action="append", default=[], metavar="NAME=VALUE", help="plan 78.29: a targeted setting (repeatable): " + ", ".join(TG.SETTINGS))
    ap.add_argument("--exact", action="store_true", help="plan 78.32: the exact labeller (Syzygy for chess, training only) judges the playouts and the problems instead of the UCI WDL judge")
    ap.add_argument("--induce", action="store_true", help="plan 78.32 (with --playouts --exact): after the lemma sleep, a rule induced from the exact corpus for each open problem")
    ap.add_argument("--induce-set", action="append", default=[], metavar="NAME=VALUE", help="plan 78.32: an induction setting (repeatable): " + ", ".join(IND.SETTINGS))
    ap.add_argument("--seed-rules", default=None, metavar="FILE", help="plan 78.32: rules proposed from outside the loop ({rules: [lemma, ..]}); each enters the sandbox store only with zero wrong firings on the exact corpus")
    ap.add_argument("--drill-sleep", action="store_true", help="plan 78.32: the lemma sleep's rules are filed only with zero wrong firings on the exact corpus (the induced rules' admission test)")
    ap.add_argument("--corpus-n", type=int, default=30000, help="plan 78.32: positions of the exact corpus (built once, cached under paths.derived)")
    ap.add_argument("--exam", default=None, metavar="PATH", help="plan 78.32 (with --exact): the fixed exam file; built from the corpus when missing; graded before cycle 1 and after every cycle")
    ap.add_argument("--exam-size", type=int, default=None, help="plan 78.32 (with --exact): positions of the exam (300 when only --exam is given); with no --exam the default file for this size")
    ap.add_argument("--corpus-problems", type=int, default=0, metavar="N", help="plan 78.34 (with --playouts --exact): per cycle, the wake's own decision on N sampled non-trivial positions of the corpus's development part; the wrong ones are problems (source corpus)")
    ap.add_argument("--grow-corpus", action="store_true", help="plan 78.34 (with --exact): after each cycle's wake the positions where a rule fired are labelled and added to the corpus, every rule is drilled on the grown corpus and one with a wrong firing is withdrawn")
    ap.add_argument("--suggest-bar", type=float, default=None, metavar="X", help="plan 78.36 (with --seed-rules): a response rule that fails zero-wrong but is right at least this share of the time (and right at least 20 times) enters as a SUGGESTION; default off")
    ap.add_argument("--conversion", type=int, default=0, metavar="N", help="plan 78.36 (with an exam): after every exam grading, N games played out from exam positions (the wake's decision for the side to move, the exact labeller's defence for the other): `OBS conversion ..`; 0 = off")
    ap.add_argument("--usage-split", action="store_true", help="plan 78.35 (with an exam): after every exam grading with rules in the snapshot, the use / generation split (OBS usage split ..), and at the end OBS rules by kind")
    ap.add_argument("--wdl-set", action="append", default=[], metavar="NAME=VALUE", help="plan 78.27: a keyword of the estimate's sleep (deep_seconds, max_deep, sample, max_settle), repeatable")
    a = ap.parse_args(argv)
    overrides = settings_pairs(a.set)
    if a.gate or a.held_gate:
        overrides["gates"] = AB.Gates().override(a.gate)
        overrides["held"] = AB.Gates(min_fires=3, min_groups=2).override(a.gate).override(a.held_gate)
    teacher = SL.UciTeacher(a.teacher, a.teacher_ms, dict(o.partition("=")[::2] for o in a.teacher_option), mate_bucket=a.teacher_mate_bucket, grade_mode=a.standard, decisive=a.decisive)
    if a.teacher_cache:
        teacher = JG.Cached(teacher, a.teacher_cache)
    wj = WS.UciWdlJudge(a.teacher, a.wdl_judge_depth or 12, a.wdl_judge_cache) if ((a.wdl and a.wdl_judge_depth) or a.playouts) and not a.exact else None
    isets = {}
    for pair in a.induce_set:
        k, _, v = pair.partition("=")
        if k not in IND.SETTINGS:
            raise ValueError(f"unknown --induce-set name {k!r}; the names are {sorted(IND.SETTINGS)}")
        isets[k] = v.split(",") if isinstance(IND.SETTINGS[k], list) else int(v)
    psets = {}
    for pair in a.playout_set:
        k, _, v = pair.partition("=")
        if k not in PLAYOUT_DEFAULTS:
            raise ValueError(f"unknown --playout-set name {k!r}; the names are {sorted(PLAYOUT_DEFAULTS)}")
        psets[k] = float(v) if k == "check_seconds" else int(v)
    tsets = {}
    for pair in a.targeted_set:
        k, _, v = pair.partition("=")
        if k not in TG.SETTINGS:
            raise ValueError(f"unknown --targeted-set name {k!r}; the names are {sorted(TG.SETTINGS)}")
        tsets[k] = int(v)
    kinds = {"deep_seconds": float, "max_deep": int, "sample": int, "max_settle": int}
    wdl_settings = {}
    for pair in a.wdl_set:
        k, _, v = pair.partition("=")
        if k not in kinds:
            raise ValueError(f"unknown --wdl-set name {k!r}; the names are {sorted(kinds)}")
        wdl_settings[k] = kinds[k](v)
    try:
        run_size(a.game, a.size, source=teacher, store_root=a.store_root, variant=a.variant, judge=a.judge, games_per_start=a.games_per_start, seconds=a.seconds, limit=a.limit, seed=a.seed,
                 sleep_settings=overrides, vocabulary=a.vocabulary, tag=a.tag, sample=a.sample, wdl=a.wdl, cycles=a.cycles, wdl_judge=wj if a.wdl_judge_depth or not a.wdl else None, wdl_settings=wdl_settings or None,
                 playouts=a.playouts, playout_judge=wj if a.playouts else None, playout_settings=psets or None, targeted=a.targeted, targeted_settings=tsets or None, force_problems=not a.no_force_problems, standard=a.standard,
                 induce=a.induce, induce_settings=isets or None, drill_sleep=a.drill_sleep, seed_rules=a.seed_rules, exam=a.exam, exam_size=a.exam_size, exact=a.exact, corpus_n=a.corpus_n, corpus_problems=a.corpus_problems, grow_corpus=a.grow_corpus, usage_split=a.usage_split, conversion=a.conversion, suggest_bar=a.suggest_bar)
    finally:
        teacher.close()
        if wj is not None:
            wj.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
