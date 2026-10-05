"""THE DRIVER: one round, six stages, a ledger, a commit.
    python -m catspace.incubator.analyst.loop.round --round N [--stage NAME] [--config extra.json] [--force]
                                                     [--render-only] [--no-commit]
- refuses round N >= 1 without a gate decision for round N-1, and after a 'swap STAGE' gate until the config names a
  different module for that stage (config["stages"][STAGE]);
- runs the stages in order; a stage is not rerun when its key -- its module, the module's source, its config, the
  previous stage's OUTPUT, the round -- matches what is on disk, so a stage that re-runs with a new result re-runs
  everything after it (--force reruns all; --stage NAME reruns that stage, and what follows it reruns when stale);
- the round's frozen inputs (inputs.json: the basis head it started from) are written once; stages read the basis as
  it stood then;
- appends the round's sound new ideas to the basis with their accepted uses, renders artifacts/loop/round_NNN/LEDGER.md
  with its controls, and commits the round's record and the basis (--no-commit to skip); --render-only re-renders the
  ledger from the stage outputs on disk and runs no stage."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import time
from pathlib import Path

from . import basis, gate
from .ctx import Ctx
from .ledger import controls, render
from .stages.measure import enrich
from .store import code_version, digest, read_json, record_dir, round_dir, write_json

STAGES = ("frame", "mine", "translate", "prove", "use", "measure")
MODULE = "catspace.incubator.analyst.loop.stages.{}"

ROUND0_ARMS = {
    "backrank": "frame_puzzles_lichess_zone_candidates_proved1_off17_backRankMate_lean1004.json",
    "pawnendings": "frame_puzzles_lichess_zone_candidates_passer_off17_pawnEndgame_lean1005c.json",
    "general": "frame_puzzles_lichess_zone_candidates_loopactive000_off300_loop000full.json",   # round 0's wiring replayed (the 10-04 dump had the back-rank closure only)
}


class RoundError(Exception):
    pass


def default_config(n: int) -> dict:
    if n == 0:                                                               # the baseline and a consistency check
        return dict(stages={}, frame=dict(skip=True), mine=dict(skip=True), translate=dict(recheck_registered=True), prove={}, use={},
                    measure=dict(mode="dumps", arms=dict(ROUND0_ARMS), time_baselines=True))
    return dict(stages={}, frame=dict(puzzles=150, games_max=400, games_max_records=150, horizon=4, min_loss=2),
                mine=dict(max_size=2, max_frames=300, cap=120, retest_positions=6000, retest_cap=400),
                translate=dict(recheck_registered=True), prove={}, use={}, measure=dict(mode="player"))


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def check_gate(n: int, config: dict) -> None:
    if n == 0:
        return
    g = gate.gate_of(n - 1)
    if g is None:
        raise RoundError(f"round {n - 1} has no gate decision: read artifacts/loop/round_{n - 1:03d}/LEDGER.md, then "
                         f"python -m catspace.incubator.analyst.loop.gate --round {n - 1} --decision continue|rollback|swap --note ...")
    if g["decision"] == "swap":
        st = g["stage"]
        prev = (read_json(record_dir(n - 1) / "inputs.json") or {}).get("config", {}).get("stages", {}).get(st, MODULE.format(st))
        if config.get("stages", {}).get(st, MODULE.format(st)) == prev:
            raise RoundError(f"round {n - 1} was gated 'swap {st}': name the replacement module in the config (stages.{st})")


def previous_measure(n: int) -> dict | None:
    """The measure record of the latest round before n whose ideas were kept (gated continue or swap): its arm is the
    player before round n's ideas, so pairing against it attributes the round."""
    for k in range(n - 1, -1, -1):
        g = gate.gate_of(k)
        if g is not None and g["decision"] != "rollback":
            return read_json(record_dir(k) / "measure.json")
    return None


def frozen_inputs(n: int, config: dict) -> dict:
    """The round's frozen inputs, written once: the basis it started from and its code. A re-run under another config
    records the change and keeps the basis the round started from (it used to take the basis as it stands now: a
    reconfigured round 0 would have read round 1's ideas as its own; BUGS 2026-10-05)."""
    rec = record_dir(n)
    inputs = read_json(rec / "inputs.json")
    if not inputs:
        inputs = dict(round=n, code=code_version(), config=config,
                      basis_before=basis.head(), basis_before_ids=sorted(basis.active()), time=time.strftime("%Y-%m-%d %H:%M %Z"))
        write_json(rec / "inputs.json", inputs)
    elif inputs.get("config") != config:
        inputs = dict(inputs, config=config, reconfigured=inputs.get("reconfigured", []) + [
            dict(time=time.strftime("%Y-%m-%d %H:%M %Z"), code=code_version(), before=inputs.get("config"))])
        write_json(rec / "inputs.json", inputs)
    return inputs


def module_source(modname: str) -> str | None:
    """A digest of a stage module's own source. A cached stage output is reused only when the code that made it is the
    code that would run now (the key used to omit the code: BUGS 2026-10-05)."""
    import importlib.util
    spec = importlib.util.find_spec(modname)
    return digest(open(spec.origin).read()) if spec and spec.origin else None


def run_round(n: int, only: str | None = None, extra: dict | None = None, force: bool = False, commit: bool = True,
              render_only: bool = False) -> dict:
    config = merge(default_config(n), extra or {})
    check_gate(n, config)
    d, rec = round_dir(n), record_dir(n)
    inputs = frozen_inputs(n, config)
    head_before = inputs["basis_before"]
    ctx = Ctx(n, d, rec, config, inputs["code"], basis_before=inputs["basis_before"])
    timings = read_json(d / "timings.json", {}) or {}
    prev_hash = digest(dict(basis=inputs["basis_before"], config=config.get("stages", {})))
    stale: list = []                                                         # --render-only: stages whose code changed since they ran
    for st in STAGES:
        modname = config.get("stages", {}).get(st, MODULE.format(st))
        src = module_source(modname)
        h = digest(dict(stage=st, module=modname, source=src, config=config.get(st), prev=prev_hash, round=n))
        cached = read_json(d / f"{st}.json")
        if render_only:                                                      # the ledger from the outputs on disk, whatever made them
            if cached is None:
                raise RoundError(f"--render-only: stage {st} has no output")
            if cached.get("source") != src:
                stale.append(st)
            prev_hash = digest(cached["output"])
            continue
        if cached and cached["hash"] == h and not force and only != st:
            ctx.log(f"stage {st}: cached")
            prev_hash = digest(cached["output"])
            continue
        ctx.log(f"stage {st}: running ({modname})")
        t0 = time.perf_counter()
        out = importlib.import_module(modname).run(ctx)
        timings[st] = time.perf_counter() - t0
        write_json(d / f"{st}.json", dict(hash=h, output=out, seconds=timings[st], source=src, code=code_version()))
        write_json(d / "timings.json", timings)
        prev_hash = digest(read_json(d / f"{st}.json")["output"])            # the NEXT stage's key holds this OUTPUT, so a stage
                                                                             # re-run with a new result re-runs everything after it
    outs = {st: (read_json(d / f"{st}.json") or {}).get("output") for st in STAGES}
    if outs.get("measure"):                                                  # expected scores, and the previous kept round's arm,
        outs["measure"], obs = enrich(outs["measure"], previous_measure(n))  # from the dumps (rounds measured before the fields too)
        for line in obs:
            ctx.log(line)
    ctl = controls(outs, inputs)
    # the basis: the round's sound ideas that are new, with their accepted uses
    promoted = (outs.get("prove") or {}).get("promoted", [])
    uses: dict = {}
    for u in (outs.get("use") or {}).get("uses", []):
        if not u["refused"]:
            uses.setdefault(u["idea"], []).append(u)
    already = {e["id"] for e in basis.read() if e["op"] == "promote" and e["round"] == n}
    new = [p for p in promoted if not p["unsound"] and p["candidate"]["id"] not in basis.active() and p["candidate"]["id"] not in already]
    head_after = basis.promote(n, new, uses) if new else basis.head()
    heads = dict(before=head_before, after=head_after, after_count=len(basis.active()), render_code=code_version(),
                 render_only=render_only, stale=stale)
    text = render(n, outs, inputs, ctl, timings, heads)
    (rec / "LEDGER.md").write_text(text)
    write_json(rec / "controls.json", ctl)
    write_json(rec / "promoted.json", [dict(id=p["candidate"]["id"], tier=p["tier"], unsound=p["unsound"], reason=p["reason"],
                                            bridge={k: v for k, v in (p["evidence"].get("bridge") or {}).items() if k != "parts"},
                                            axioms=p["evidence"].get("axioms")) for p in promoted])
    write_json(rec / "uses.json", (outs.get("use") or {}).get("uses", []))
    libs: dict = {}
    lemmas = []                                                              # the promoted mined lemmas' full definitions, in the
    for p in promoted:                                                       # committed record: the basis survives the loss of the
        o = p["candidate"].get("origin") or {}                               # git-ignored library files
        if o.get("library") and Path(o["library"]).exists():
            lib = libs.setdefault(o["library"], read_json(Path(o["library"])))
            if 0 <= o.get("index", -1) < len(lib) and lib[o["index"]].get("id") == p["candidate"].get("guard"):
                lemmas.append(dict(id=p["candidate"]["id"], entry=lib[o["index"]]))
    if lemmas:
        write_json(rec / "lemmas.json", lemmas)
    write_json(rec / "measure.json", dict(measurements=(outs.get("measure") or {}).get("measurements", []), fire=(outs.get("measure") or {}).get("fire", {})))
    n_new = sum(1 for e in basis.read() if e["op"] == "promote" and e["round"] == n)   # promoted by this round (a re-render too)
    write_json(rec / "round.json", dict(round=n, timings=timings, heads=heads, new_ideas=n_new, red=ctl["red"]))
    if commit:
        headline = f"{n_new} new ideas; " + ("UNSOUND red; " if ctl["red"] else "") + ("nothing fires; " if ctl["nothing_fires"] else "") + \
                   ("results fall; " if ctl["results_fall"] else "") + f"{sum(timings.values()) / 60:.1f} min"
        gate._commit([rec, basis.path()], f"loop round {n:03d}: {headline} (artifacts/loop/round_{n:03d}/LEDGER.md)")
    ctx.log(f"ledger: {rec / 'LEDGER.md'}")
    return dict(controls=ctl, heads=heads, timings=timings)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--stage", default=None, choices=list(STAGES))
    ap.add_argument("--config", default=None, help="a JSON file merged over the round's default configuration")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-commit", action="store_true")
    ap.add_argument("--render-only", action="store_true", help="re-render the ledger from the stage outputs on disk; never run a stage")
    a = ap.parse_args(argv)
    extra = json.load(open(a.config)) if a.config else None
    try:
        r = run_round(a.round, a.stage, extra, a.force, commit=not a.no_commit, render_only=a.render_only)
    except (RoundError, gate.GateError, basis.BasisError) as e:
        raise SystemExit(f"REFUSED: {e}")
    print(f"round {a.round}: red={r['controls']['red']} nothing_fires={r['controls']['nothing_fires']} "
          f"results_fall={bool(r['controls']['results_fall'])}; ledger artifacts/loop/round_{a.round:03d}/LEDGER.md")


if __name__ == "__main__":
    main()
