"""MINE: guards over relation literals around the framed batch -- lemma_gen `run` on the framed positions, then its
`prove` re-test on the disjoint held-out part of the general population. Each lemma becomes a Guard with its held-out
status (lemma_gen's PROVED means "survived the re-test", which is evidence, not a proof: the loop's tier for it is
conjecture). Research slot R4 (vocabulary feedback from failed proofs) changes what this stage can say."""
from __future__ import annotations

import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

from catspace.io import paths

from ..contracts import Guard, to_dict


def run(ctx) -> dict:
    cfg = ctx.cfg("mine")
    framed = (ctx.load("frame") or {}).get("framed", [])
    if cfg.get("skip") or not framed:
        return dict(guards=[], skipped=True)
    fens = sorted({f["fen"] for f in framed})
    fpath = ctx.dir / "mine_fens.txt"
    fpath.write_text("\n".join(fens) + "\n")
    tag = f"loop{ctx.round:03d}"
    run_cmd = [sys.executable, "-m", "catspace.incubator.analyst.lemma_gen", "run", "--source", "fens", "--fens-file", str(fpath),
               "--positions", str(len(fens)), "--tag", tag, "--max-size", str(cfg.get("max_size", 2)),
               "--max-frames", str(cfg.get("max_frames", 300)), "--cap", str(cfg.get("cap", 120))]
    prove_cmd = [sys.executable, "-m", "catspace.incubator.analyst.lemma_gen", "prove", "--source", "lichess", "--split", "test",
                 "--positions", str(cfg.get("retest_positions", 6000)), "--cap", str(cfg.get("retest_cap", 400)), "--tag", tag]
    for name, cmd in (("mine_run", run_cmd), ("mine_prove", prove_cmd)):
        ctx.log(f"{name}: " + " ".join(cmd[2:]))
        r = subprocess.run(cmd, cwd=paths.REPO_ROOT, capture_output=True, text=True)
        (ctx.dir / f"{name}.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f"lemma_gen {name} failed: " + r.stderr[-2000:])
    found = sorted(glob.glob(str(paths.REPO_ROOT / "data" / "derived" / "lemmas" / f"library_*_{tag}.json")))
    if len(found) != 1:
        raise RuntimeError(f"expected one library for tag {tag}, found {found}")
    made_proved = found[0].replace("library_", "proved_")
    if not Path(made_proved).exists():
        raise RuntimeError(f"the re-test wrote no {made_proved}")
    lib_path, proved_path = str(ctx.dir / "library.json"), str(ctx.dir / "proved.json")   # the round's own copies: lemma_gen
    shutil.copyfile(found[0], lib_path)                                                    # rewrites its files when a stage
    shutil.copyfile(made_proved, proved_path)                                              # re-runs; the wiring reads these
    lib = json.load(open(lib_path))
    proved = json.load(open(proved_path))
    if len(proved) != len(lib):
        raise RuntimeError(f"library {len(lib)} lemmas, re-test {len(proved)} entries")
    guards = []
    for i, (e, p) in enumerate(zip(lib, proved)):
        held = dict(status=p.get("status"), coverage=p.get("coverage"), purity=p.get("purity"), support=p.get("support"))
        guards.append(Guard(e["id"], f"{e['frame']} :: slot {e['slot']} {e['claim']}", e["types"], e["literals"],
                            (e.get("inherited") or []) + (e.get("guard") or []), int(e["slot"]), e["claim"], e.get("value") or {},
                            int(e["support"]), int(e["holds"]), float(e.get("purity", 1.0)), held, lib_path, i,
                            ctx.prov("mine", [])))
    from collections import Counter
    ctx.log(f"OBS mine: {len(fens)} framed positions -> {len(guards)} lemmas; held-out re-test {dict(Counter(g.held_out['status'] for g in guards))}")
    return dict(guards=[to_dict(g) for g in guards], library=lib_path, proved=proved_path)
