"""PLAN 78.24 C THE LOOP FOR ONE MATERIAL SIZE, ON A SANDBOX STORE: wake (from the size's starts), sleep (a rank source is the judge; lemmas filed, the next snapshot published), wake again
with that snapshot, then the audit of what was mined and how it was applied. Generic: the game is a parameter (its position codec, vocabulary and rank source are the only game knowledge).

The store is a SANDBOX (`store_root`, default paths.derived("init_store")): `game_store.LEARNED_ROOT` / `SNAPSHOT_ROOT` point into it for the run and are restored afterwards; the game's
vocabulary files are copied in once from the real rules/learned/<game>/ (never the other way); lemmas already in the sandbox are kept, so the store accumulates across sizes and runs.

THIS IS A MECHANISM CHECK, NOT A STRENGTH CLAIM: one size, a few dozen games, a rank source's regret rate on a sample of positions. Two wakes of a few games differ by chance alone.

  python -m catspace.incubator.analyst.discovery.initialize chess --size 3 --teacher <uci> [--teacher-ms 100] [--teacher-option NAME=VALUE ...] [--store-root <dir>] [--judge <record>]
         [--games-per-start 4] [--limit N] [--set NAME=VALUE ...] [--vocabulary NAME] [--gate NAME=VALUE ...] [--held-gate NAME=VALUE ...]
         [--teacher-mate-bucket N] [--teacher-cache FILE] [--tag T]
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
from catspace.incubator.analyst.discovery import curriculum as CU
from catspace.incubator.analyst.discovery import detectors as D
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import judges as JG
from catspace.incubator.analyst.discovery import sleep as SL
from catspace.incubator.analyst.discovery import wake as WK
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


def run_size(game: str, size: int, *, source, store_root=None, variant: "str | None" = None, judge: "str | None" = None, games_per_start: int = 4, seconds: float = 0.2,
             random_plies: int = 2, max_plies: int = 80, limit: "int | None" = None, seed: int = 1, sleep_settings: "dict | None" = None, vocabulary: "str | None" = None,
             tag: "str | None" = None, sample: int = 200, say=print) -> dict:
    """The loop for one size: starts -> publish -> wake a -> sleep (source as judge) -> wake b -> audit -> compare. Returns everything and writes it as JSON next to the logs."""
    variant = variant or game
    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    root = prepare_sandbox(game, store_root if store_root is not None else paths.derived("init_store"))
    vkw = {} if vocabulary is None else {"vocabulary": vocabulary}
    settings = settings_from({"judge_by": "source", **({} if judge is None else {"judge": judge}), **({} if vocabulary is None else {"vocabulary": vocabulary}), **(sleep_settings or {})})
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

        def wake(label: str, run: str) -> tuple:
            sid = GS.publish(game, **({"judge": GS.judge_of(game, judge)} if judge else {}), **vkw)
            log = paths.experiment(f"wake_{game}_init{size}_{tag}_{label}.jsonl")
            if Path(log).exists():
                Path(log).unlink()
            t0 = time.perf_counter()
            out = WK.play_self(variant, games_per_start * len(starts), seconds=seconds, random_plies=random_plies, max_plies=max_plies, seed=seed,
                               store=str(GS.path_of(game, sid)), specs=specs, log=log, run=run, ties=True, starts=starts)
            res = outcomes(out)
            say(f"OBS init wake {label} {game}: snapshot {sid} ({len(GS.info(game, sid)['lemmas'])} lemma(s)), games {len(out)}, plies {sum(len(g['moves']) for g in out)}, "
                f"outcomes {res}, {time.perf_counter() - t0:.0f} s -> {log}")
            return {"snapshot": sid, "log": log, "games": len(out), "plies": sum(len(g['moves']) for g in out), "outcomes": res}, log

        rep["wake_a"], log_a = wake("a", f"i{size}{tag}a")
        sl = SL.sleep(game, log_a, variant=variant, settings=settings, teacher=source, write=True, say=say)
        rep["sleep"] = {"funnel": sl["funnel"], "lemmas": [l["name"] for l in sl["lemmas"]], "rejected": len(sl["rejected"]), "snapshot": sl.get("snapshot"), "judge": sl["judge"]}
        say(f"OBS init sleep {game}: lemmas admitted {len(sl['lemmas'])} {rep['sleep']['lemmas']}, rejected {len(sl['rejected'])}, funnel {sl['funnel']}")
        rep["wake_b"], log_b = wake("b", f"i{size}{tag}b")
        rep["mined"] = AU.mined(game, vocabulary)
        for line in rep["mined"]:
            say("OBS mined " + line)
        if not rep["mined"]:
            say(f"OBS mined {game}: no lemma in the store")
        rep["audit"] = AU.audit(game, log_b, variant=variant, source=source, vocabulary=vocabulary, say=say)
        rep["compare"] = {"wake_a": {"outcomes": rep["wake_a"]["outcomes"], **regret_rate(game, log_a, source, sample=sample, seed=seed, variant=variant)},
                          "wake_b": {"outcomes": rep["wake_b"]["outcomes"], **regret_rate(game, log_b, source, sample=sample, seed=seed, variant=variant)}}
        a, b = rep["compare"]["wake_a"], rep["compare"]["wake_b"]
        say(f"OBS init compare {game} size {size}: wake a outcomes {a['outcomes']}, regret {a['regret']}/{a['ranked']} ({a['rate']}); wake b outcomes {b['outcomes']}, regret {b['regret']}/{b['ranked']} "
            f"({b['rate']}); a mechanism check on {len(starts)} starts, not a strength claim")
    finally:
        GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT = old
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
    a = ap.parse_args(argv)
    overrides = settings_pairs(a.set)
    if a.gate or a.held_gate:
        overrides["gates"] = AB.Gates().override(a.gate)
        overrides["held"] = AB.Gates(min_fires=3, min_groups=2).override(a.gate).override(a.held_gate)
    teacher = SL.UciTeacher(a.teacher, a.teacher_ms, dict(o.partition("=")[::2] for o in a.teacher_option), mate_bucket=a.teacher_mate_bucket)
    if a.teacher_cache:
        teacher = JG.Cached(teacher, a.teacher_cache)
    try:
        run_size(a.game, a.size, source=teacher, store_root=a.store_root, variant=a.variant, judge=a.judge, games_per_start=a.games_per_start, seconds=a.seconds, limit=a.limit, seed=a.seed,
                 sleep_settings=overrides, vocabulary=a.vocabulary, tag=a.tag, sample=a.sample)
    finally:
        teacher.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
