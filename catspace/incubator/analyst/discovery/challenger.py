"""THE CHALLENGER (Kaveh 2026-10-12: "my idea of bringing a rook into this and interposing is a challenge that I bring up as
a challenger, but you cannot bring it up against yourself ... you need your reasoner to reason both sides"; the design
after two literature rounds -- JOURNAL 2026-10-12 19:55 -- oracle-free, so it carries to domains where the ending
playbook is partial).

  A FRAME is a ranking certificate (its measure falls); a COUNTER-FRAME is its dual, a TRAP: a region R where whatever we
  do, the defender has a reply keeping us in R with our measure no lower (Lean Ideas.Forced.hold_blocks).

  1 MINE    from the frame's REFUTED claims in a round's games (any falsifier made them: the tablebase, our own search, a
            simulator): R = {term >= theta} for a term of the vocabulary, chosen as a PREDICTOR of the refutations --
            precision and lift on held-out games (split by game)
  2 CHECK   the local trap check on boards of R (vocab.rs trap_check: one move against every reply, no replay of any
            oracle); passing it, R is a COUNTER-FRAME; nothing expressible passing it -> a VOCABULARY GAP (logged with its
            boards: where new predicates come from)
  3 ANSWER  the attacker's lemma-incorporation -- "leave R" (the trap's term, minimised) ranked FIRST in the frame --
            verified on R's refuted boards by the verifier; promoted by its rules
  4 ARCHIVE every counter-frame and gap kept with its status (survived the local check on N boards / predictor only /
            refuted by the attacker's answer) -- a never-forget archive

  python -m catspace.incubator.analyst.discovery.challenger --run level2b --round 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

from catspace.io import paths

ARCHIVE = "counter_frames.json"
FAILED = ("no shrink", "other outcome", "lost ground")


def data(res: list, material: str) -> tuple:
    """(fen, refuted?, game) for every frame move of ours in the material while ahead."""
    out = []
    for g in res:
        for r in g["rows"]:
            if r["material"] == material and r["lead"] > 0 and r["claim"]["kind"] == "frame" and r.get("verdict"):
                out.append((r["fen"], r["verdict"] in FAILED, g["seed"]))
    return out


def best_threshold(v: np.ndarray, y: np.ndarray) -> tuple:
    """theta maximising (precision - base) * recall for R = {v >= theta} predicting y -> (theta, precision, recall, score)."""
    base = y.mean() if len(y) else 0.0
    best = (None, 0.0, 0.0, -1.0)
    for th in np.unique(v):
        r = v >= th
        if r.sum() < 5:
            continue
        prec, rec = y[r].mean(), (y & r).sum() / max(y.sum(), 1)
        sc = (prec - base) * rec
        if sc > best[3]:
            best = (int(th), float(prec), float(rec), float(sc))
    return best


def mine(res: list, frames: dict, material: str, top: int = 5, check_n: int = 60, seed: int = 0, margin: float = 0.10) -> dict:
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    from . import base_vocabulary as V
    t0 = time.time()
    rows = data(res, material)
    if len(rows) < 50:
        return dict(material=material, status="too little data", n=len(rows))
    fens = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    train = np.array([r[2] % 2 == 0 for r in rows])                    # split by game: the even seeds learn, the odd test
    terms = [V.sexpr(t) for t in V.enumerate_numbers()]
    seen, cands = set(), []
    for t in terms:
        v = np.array(A.vocab_values(fens, t))
        key = tuple(v[train][:200])
        if v.std() == 0 or key in seen:
            continue
        seen.add(key)
        th, prec, rec, sc = best_threshold(v[train], y[train])
        if th is not None:
            cands.append((sc, t, th, v))
    cands.sort(key=lambda c: -c[0])
    base_rate = float(y[~train].mean())
    print(f"OBS challenger, {material}: {len(rows)} frame claims ({100 * y.mean():.0f}% refuted), {len(seen)} distinct terms tried")
    out = []
    frame = frames.get(material, [])
    for sc, t, th, v in cands[:top]:
        r = (v >= th) & ~train
        prec = float(y[r].mean()) if r.sum() else 0.0
        rec = float((y & r & ~train).sum() / max((y & ~train).sum(), 1))
        inside = [f for f, ok in zip(fens, r & y) if ok]                   # refuted test boards inside R
        sample = random.Random(seed).sample(inside, min(check_n, len(inside)))
        chk = [c for c in A.trap_check(sample, frame, t, th) if c is not None]
        trap = sum(chk) / max(len(chk), 1)
        out.append(dict(term=t, theta=th, gloss=V.gloss(V.parse(t)), precision=prec, recall=rec, lift=prec / max(base_rate, 1e-9),
                        trap=trap, checked=len(chk), boards=inside))
        print(f"OBS   R = {{ {V.gloss(V.parse(t))} >= {th} }}: held-out precision {prec:.2f} (base {base_rate:.2f}, lift {prec / max(base_rate, 1e-9):.2f}), "
              f"recall {rec:.2f}; the local trap check holds on {sum(chk)}/{len(chk)} of its refuted boards")
    accepted = [c for c in out if c["precision"] >= max(0.6, base_rate + margin) and c["trap"] >= 0.8 and c["checked"] >= 10]
    print(f"OBS   ({time.time() - t0:.0f} s)")
    if not accepted:
        return dict(material=material, status="vocabulary gap", candidates=[{k: v for k, v in c.items() if k != "boards"} for c in out],
                    gap_boards=[f for f, ok in zip(fens, y & ~train) if ok][:200], frame=frame)
    best = max(accepted, key=lambda c: (c["trap"], c["lift"]))
    return dict(material=material, status="counter-frame", frame=frame, **{k: v for k, v in best.items() if k != "boards"},
                boards=best["boards"], candidates=[{k: v for k, v in c.items() if k != "boards"} for c in out])


def answer(cf: dict, run: str, rnd: int, frames: dict, defender: str = "tablebase", n_boards: int = 50, budget: int = 60) -> dict:
    """The attacker's lemma-incorporation as a STAGE (the red team: "a new frame whose goal is 'leave R'"): a situational
    frame "<material>:if:<term>>=<theta>", active while R holds, its measure MINED by the pipeline like any other on R's
    refuted boards; its verifier games succeed on leaving R (or the material goal) without losing ground."""
    from .pipeline import run as pipeline_run
    from . import defenders as D
    import chess
    def fresh(f):
        b = chess.Board(f); b.halfmove_clock = 0
        return b.fen()
    boards = [fresh(f) for f in cf["boards"]][:100]
    fkey = f"{cf['material']}:if:{cf['term']}>={cf['theta']}"
    hole = dict(name=f"stage_{run}_{cf['material']}_r{rnd}", material=cf["material"], frame_key=fkey, frames=frames, piece="*",
                base=[], k=2, stuck=boards, sampled=len(boards), leave=[cf["term"], cf["theta"]])
    json.dump(hole, open(paths.derived(f"hole_{hole['name']}.json"), "w"))
    out = pipeline_run(hole["name"], budget, hole=hole, defender=defender, n_boards=n_boards)
    base, best = out[0], max(out[1:], key=lambda r: r["score"]) if len(out) > 1 else None
    as_o = lambda r: dict(n=r["n"], lost=r["lost"], won=r["mated"], kept=r["certified"], claims=r["played"], states=r.get("states", 0))
    wins = bool(best) and (D.better(as_o(best), as_o(base)) if "ranks" in base else best["score"] > base["score"] + 0.05)
    show = lambda r: f"left R or won {r['mated']}/{r['n']}, lost ground {r['lost']}"
    print(f"OBS   the attacker's answer -- a STAGE for R ({fkey}): the best mined measure {best and best['term']} -- {best and show(best)}; "
          f"without the stage {show(base)} -> {'PROMOTED' if wins else 'not better'}")
    return dict(frame_key=fkey, term=best and best["term"], schema=best and best["schema"], base=show(base), answer=best and show(best), promoted=bool(wins))


def archive(entry: dict) -> None:
    p = paths.derived(ARCHIVE)
    arc = json.load(open(p)) if os.path.exists(p) else []
    arc.append({k: v for k, v in entry.items() if k not in ("boards",)})
    json.dump(arc, open(p, "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--material", default="KRRvKR")
    ap.add_argument("--no-answer", action="store_true")
    ap.add_argument("--margin", type=float, default=0.10, help="a trap must predict the refutations this much above their base rate")
    a = ap.parse_args(argv)
    stem = f"surprise_loop_KRRvKR_{a.run}"
    history = json.load(open(paths.derived(f"{stem}.json")))
    frames = next(h["frames"] for h in history if h["round"] == a.round)
    res = json.load(open(paths.experiment(f"{stem}_r{a.round}.json")))
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    for k, v in frames.items():
        A.set_material_frame(k, v)
    cf = mine(res, frames, a.material, margin=a.margin)
    entry = dict(run=a.run, round=a.round, t=time.strftime("%Y-%m-%d %H:%M"), **cf)
    if cf["status"] == "counter-frame":
        print(f"OBS COUNTER-FRAME for {a.material}: when {cf['gloss']} >= {cf['theta']}, the defender holds the frame "
              f"{cf['frame']} (local trap check {cf['trap']:.2f} on {cf['checked']} boards; held-out precision {cf['precision']:.2f})")
        if not a.no_answer:
            entry["answer"] = answer(cf, a.run, a.round, frames)
            entry["status"] = "refuted by the attacker's answer" if entry["answer"]["promoted"] else "standing"
    else:
        print(f"OBS {cf['status'].upper()} for {a.material}: no trap of the vocabulary both predicts the refutations and passes the local check"
              + (f" -- {len(cf.get('gap_boards', []))} refuted boards kept for new predicates" if cf["status"] == "vocabulary gap" else ""))
    archive(entry)


if __name__ == "__main__":
    main()


def round_(res: list, frames: dict, run: str, rnd: int, register: bool = True) -> dict:
    """The challenger inside the loop, after a round's mining: per material with a frame of ours, a trap mined and checked;
    a counter-frame answered by a stage; a promoted stage joins the frames (the next round's games play it). -> frames."""
    new = dict(frames)
    for material in sorted({k for k in frames if ":" not in k}):
        cf = mine(res, frames, material)
        entry = dict(run=run, round=rnd, t=time.strftime("%Y-%m-%d %H:%M"), **{k: v for k, v in cf.items() if k not in ("gap_boards",)})
        if cf["status"] == "counter-frame":
            print(f"OBS COUNTER-FRAME for {material}: when {cf['gloss']} >= {cf['theta']}, the defender holds {cf['frame']} "
                  f"(local trap check {cf['trap']:.2f} on {cf['checked']} boards; held-out precision {cf['precision']:.2f})")
            ans = answer(cf, run, rnd, new)
            entry["answer"] = ans
            if ans["promoted"]:
                new[ans["frame_key"]] = [ans["term"]]
                entry["status"] = "answered by a stage"
                if register:
                    from . import registry as R
                    R.add_generated(f"stage_{material.lower()}_{run}_r{rnd}", ans["term"], "measure", f"stage_{run}_{material}_r{rnd}",
                                    {"verifier": f"{ans['answer']} against {ans['base']} without the stage", "trap": f"{cf['gloss']} >= {cf['theta']}",
                                     "status": "a conjecture: the stage's claim is not proved"}, used_by=f"the stage {ans['frame_key']}")
                    R.export()
            else:
                entry["status"] = "standing (no stage beat it)"
        else:
            print(f"OBS {cf['status'].upper()} for {material}")
        archive(entry)
    return new
