"""POPPER (learning from failures, Cropper & Morel 2021) on a frame's rows, imported as written (cloned beside the
repo; clingo via pip). The frame's rows become a relational ILP task: facts over (row, slot...) for every domain
relation, the acting slot, the exceptions as positives, the rest as negatives. Popper then learns `exc(R)` as a
Prolog program with SLOT VARIABLES -- a rule like exc(R) :- actor(R,A), attacks(R,B,A), between(R,B,A,C) is a schema
over slots, not a guard tied to slot numbers -- and with predicate invention on. This is the comparison for our
greedy DNF guard search on the residual frames (two rooks), and the positive control on the pin.

    python -m catspace.incubator.analyst.popper_residual --frame "[wR wR] same_side(0,1)" --slot 0 --cap 300 --timeout 300
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

from .lemma_gen import Generator, parse_frame

OUT = Path("data/derived/lemmas/popper")
POPPER = Path("/Users/kav/code/remote/github/Popper")


def build_task(frame_text: str, slot: int, cap: int, n_rows: int, relative: bool, seed: int = 0) -> Path:
    from .chess_domain import domain
    dom = domain(); g = Generator(dom, cap=cap, log=lambda *a, **k: None, relative=relative)
    fr = parse_frame(frame_text)
    rows, complete = g.rows_of(fr, slot)
    rng = random.Random(seed)
    exc = [r for r in rows if r[3] < 0]; hold = [r for r in rows if r[3] >= 0]
    k = min(n_rows // 2, len(exc), len(hold))
    sample = [(r, True) for r in rng.sample(exc, k)] + [(r, False) for r in rng.sample(hold, k)]
    task = OUT / (frame_text.replace(" ", "_").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "") + f"_s{slot}")
    task.mkdir(parents=True, exist_ok=True)
    bk = []; pos = []; neg = []
    unary = [r for r in dom.relations if not r.on_action and r.arity == 1]
    binary = [r for r in dom.relations if not r.on_action and r.arity == 2]
    ternary = [r for r in dom.relations if not r.on_action and r.arity == 3]
    act0 = [r for r in dom.relations if r.on_action and r.arity == 0]
    act1 = [r for r in dom.relations if r.on_action and r.arity == 1]
    k_slots = fr.size
    for i, ((state, ids, a, d), is_exc) in enumerate(sample):
        rid = f"r{i}"
        (pos if is_exc else neg).append(f"exc({rid}).")
        bk.append(f"actor({rid},s{slot}).")
        for s_ in range(k_slots):
            bk.append(f"type_{fr.types[s_].lower()}({rid},s{s_}).")
        for r in unary:
            for s_ in range(k_slots):
                try:
                    v = bool(r.fn(state, ids[s_]))
                except Exception:
                    v = False
                bk.append(f"{'' if v else 'not_'}{r.name}({rid},s{s_}).")
        for r in binary:
            for s1 in range(k_slots):
                for s2 in range(k_slots):
                    if s1 == s2:
                        continue
                    try:
                        v = bool(r.fn(state, ids[s1], ids[s2]))
                    except Exception:
                        v = False
                    bk.append(f"{'' if v else 'not_'}{r.name}({rid},s{s1},s{s2}).")
        for r in ternary:
            for s1 in range(k_slots):
                for s2 in range(k_slots):
                    for s3 in range(k_slots):
                        if len({s1, s2, s3}) < 3:
                            continue
                        try:
                            if r.fn(state, ids[s1], ids[s2], ids[s3]):
                                bk.append(f"{r.name}({rid},s{s1},s{s2},s{s3}).")
                        except Exception:
                            pass
        for r in act0:
            try:
                v = bool(r.fn(state, a))
            except Exception:
                v = False
            bk.append(f"{'' if v else 'not_'}{r.name}({rid}).")
        for r in act1:
            for s_ in range(k_slots):
                try:
                    v = bool(r.fn(state, a, ids[s_]))
                except Exception:
                    v = False
                bk.append(f"{'' if v else 'not_'}{r.name}({rid},s{s_}).")
    # Prolog wants each predicate declared (a predicate with no facts is otherwise "unknown") and its facts contiguous
    preds = sorted({(f.split("(")[0], f.count(",") + 1) for f in bk})
    bk_sorted = sorted(bk, key=lambda f: f.split("(")[0])
    (task / "bk.pl").write_text("\n".join(f":- dynamic {n}/{ar}." for n, ar in preds) + "\n" + "\n".join(bk_sorted) + "\n")
    (task / "exs.pl").write_text("\n".join(f"pos({p[:-1]})." for p in pos) + "\n" + "\n".join(f"neg({n[:-1]})." for n in neg) + "\n")
    bias = ["head_pred(exc,1).", "body_pred(actor,2).", "max_vars(5).", "max_body(4).", "max_clauses(3).", "enable_pi.", "type(exc,(row,)).", "type(actor,(row,slot))."]
    for t in sorted({t.lower() for t in fr.types}):
        bias.append(f"body_pred(type_{t},2)."); bias.append(f"type(type_{t},(row,slot)).")
    for r in unary:
        bias += [f"body_pred({r.name},2).", f"body_pred(not_{r.name},2).", f"type({r.name},(row,slot)).", f"type(not_{r.name},(row,slot))."]
    for r in binary:
        bias += [f"body_pred({r.name},3).", f"type({r.name},(row,slot,slot)).", f"body_pred(not_{r.name},3).", f"type(not_{r.name},(row,slot,slot))."]
    for r in ternary:
        bias += [f"body_pred({r.name},4).", f"type({r.name},(row,slot,slot,slot))."]
    for r in act0:
        bias += [f"body_pred({r.name},1).", f"body_pred(not_{r.name},1).", f"type({r.name},(row,)).", f"type(not_{r.name},(row,))."]
    for r in act1:
        bias += [f"body_pred({r.name},2).", f"body_pred(not_{r.name},2).", f"type({r.name},(row,slot)).", f"type(not_{r.name},(row,slot))."]
    (task / "bias.pl").write_text("\n".join(bias) + "\n")
    declared = {n for n, _ in preds}
    extra = []
    for line in bias:
        if line.startswith("body_pred("):
            name, ar = line[len("body_pred("):-2].split(",")
            if name not in declared:
                extra.append(f":- dynamic {name}/{int(ar) }.")
    if extra:
        (task / "bk.pl").write_text("\n".join(extra) + "\n" + (task / "bk.pl").read_text())
    print(f"[popper] task {task}: {len(pos)} pos, {len(neg)} neg, {len(bk)} facts, {len(rows)} rows total ({'complete' if complete else 'sampled'})")
    return task


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True); ap.add_argument("--slot", type=int, required=True); ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--rows", type=int, default=400); ap.add_argument("--timeout", type=int, default=300); ap.add_argument("--relative", action="store_true"); ap.add_argument("--noisy", action="store_true"); ap.add_argument("--max-vars", type=int, default=4); ap.add_argument("--max-body", type=int, default=3)
    a = ap.parse_args(argv)
    task = build_task(a.frame, a.slot, a.cap, a.rows, a.relative)
    cmd = [sys.executable, str(POPPER / "popper.py"), str(task.resolve()), "--timeout", str(a.timeout)] + (["--noisy"] if a.noisy else [])
    t0 = time.perf_counter()
    cmd += ["--max-vars", str(a.max_vars), "--max-body", str(a.max_body)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(POPPER), timeout=a.timeout + 90)
        out, err, code = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired as ex:
        out, err, code = (ex.stdout or b"").decode() if isinstance(ex.stdout, bytes) else (ex.stdout or ""), "hard timeout", -1
    print(f"[popper] {time.perf_counter() - t0:.0f}s exit {code}")
    print("\n".join(l for l in out.splitlines() if not l.startswith("Warning") and l.strip())[-3000:]); print(err[-800:])


if __name__ == "__main__":
    main()
