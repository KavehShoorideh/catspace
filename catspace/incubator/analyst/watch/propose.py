"""PROPOSALS (Kaveh 2026-10-05): "on each big surprise, the watcher retrieves its nearest cases, mines the split, and
proposes a rule with its exception for me [Claude] to review and you [Kaveh] to gate."

For every large ATTACK surprise of a batch (a sacrifice whose outcome the forecast gave little chance), the nearest
indexed cases of the same event are retrieved (cases.similar), a shallow tree is fitted to them (cases.mine's tree),
and the branch that holds the surprising position becomes a PROPOSED exception under the rule that forecast it:
condition = the same capturing and captured pieces and the branch's conditions. It is scored on the WHOLE index (not
only the neighbours it was mined from): its cases n, its success rate r, its parent's rate q on the same cases, and
the bits it would save, n x KL(r || q), and whether it moves toward what its cornerstone did (a proposal that
improves the forecast without explaining its own surprise points at a word the vocabulary lacks). Proposals go to artifacts/watch/proposals.jsonl (append-only: each line a
proposal or a decision); Claude accepts (the rule is written) or rejects (with the reason); Kaveh retires any rule.

    python -m catspace.incubator.analyst.watch.propose run [--batch N] [--top 5] [--min-bits 1.5]
    python -m catspace.incubator.analyst.watch.propose list
    python -m catspace.incubator.analyst.watch.propose accept --id P1 [--text "..."]
    python -m catspace.incubator.analyst.watch.propose reject --id P1 --note "..."
"""
from __future__ import annotations

import argparse
import json
import math
import time

from .cases import FEATURES, counts_for, similar
from .rules import RuleBase, add_rule, record_root

PHRASE = {"king_dist": "squares to the enemy king", "elo": "the player's rating", "king_zone_attackers": "pieces bearing on the king's squares",
          "king_shield": "pawns shielding the king", "investment": "pawns invested", "gives_check": "check", "phase": "pieces' worth left",
          "material": "material balance", "elo_diff": "rating edge"}


def path() -> "Path":
    return record_root() / "proposals.jsonl"


def read() -> list[dict]:
    p = path()
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def state() -> dict:
    """id -> the proposal with its latest decision."""
    out: dict = {}
    for e in read():
        if "decision" in e:
            out[e["id"]].update(status=e["decision"], decided=e)
        else:
            out[e["id"]] = dict(e, status="proposed")
    return out


def _append(e: dict) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(e, sort_keys=True) + "\n")


def kl(r: float, q: float) -> float:
    eps = 1e-9
    r, q = min(max(r, eps), 1 - eps), min(max(q, eps), 1 - eps)
    return r * math.log2(r / q) + (1 - r) * math.log2((1 - r) / (1 - q))


def branch(cases: list[dict], query: dict, depth: int = 2, min_leaf: int = 12) -> list[dict]:
    """The conditions on the path of a shallow tree, fitted to `cases`, that holds `query`, as rule-language literals."""
    from sklearn.tree import DecisionTreeClassifier
    X = [[c[f] for f in FEATURES] for c in cases]
    y = [c["success"] for c in cases]
    if len(set(y)) < 2:
        return []
    t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, random_state=0).fit(X, y).tree_
    node, lits = 0, []
    while t.children_left[node] != -1:
        f, thr = FEATURES[t.feature[node]], round(float(t.threshold[node]), 1)
        if query[f] <= thr:
            lits.append({"cmp": f, "op": "<=", "value": thr})
            node = t.children_left[node]
        else:
            lits.append({"cmp": f, "op": ">", "value": thr})
            node = t.children_right[node]
    return lits


def text_of(capturer: str, victim: str, lits: list[dict]) -> str:
    parts = [f"{PHRASE.get(l['cmp'], l['cmp'])} {'at most' if l['op'] == '<=' else 'above'} {l['value']:g}" for l in lits]
    return f"{capturer} takes {victim} as a sacrifice" + (", " + ", ".join(parts) if parts else "")


def chain(rb: RuleBase, rid: str) -> list[dict]:
    conds, r = [], rb.rules[rid]
    while r is not None and r.parent is not None:
        conds.append(r.cond)
        r = rb.rules[r.parent]
    return conds


def run(batch: int | None = None, top: int = 5, min_bits: float = 1.5, k: int = 200) -> list[dict]:
    """Proposals from a batch's largest attack surprises."""
    from .watcher import data_root, load_state
    st = load_state()
    b = len(st["batches"]) - 1 if batch is None else batch
    cases = json.loads((data_root() / f"batch_{b:03d}.json").read_text())["cases"]
    rb = RuleBase.load()
    seen = {(p["cornerstone"]) for p in state().values()}
    made = []
    for c in sorted((c for c in cases if "attack" in c), key=lambda c: -c["attack"]["bits"])[:top]:
        if c["attack"]["bits"] < min_bits or f"{c['game']}:{c['ply']}" in seen:
            continue
        q = dict(c["attack"]["attrs"], kind="sacrifice", game=c["game"], ply=c["ply"])
        near = similar(q, k)
        if len(near) < 30:
            continue
        lits = branch(near, q)
        cond = {"all": [{"is": "capturer", "eq": q["capturer"]}, {"is": "victim", "eq": q["victim"]}] + lits}
        parent = c["attack"]["rule"]
        f_c, s_c = counts_for(chain(rb, parent) + [cond])
        f_p, s_p = counts_for(chain(rb, parent))
        n = f_c + s_c
        if n == 0:
            continue
        r, qr = s_c / n, s_p / max(f_p + s_p, 1)
        pid = f"P{max([int(x[1:]) for x in state()] + [0]) + 1}"              # state() already holds this run's proposals
        e = dict(id=pid, batch=b, kind="attack", parent=parent, cond=cond, text=text_of(q["capturer"], q["victim"], lits),
                 cornerstone=f"{c['game']}:{c['ply']}", surprise_bits=round(c["attack"]["bits"], 2), outcome=c["attack"]["outcome"],
                 neighbours=len(near), cases=n, rate=round(r, 4), parent_rate=round(qr, 4),
                 bits_per_case=round(kl(r, qr), 4), bits_total=round(n * kl(r, qr), 1), time=time.strftime("%Y-%m-%d %H:%M %Z"),
                 explains=(r > qr) == (c["attack"]["outcome"] == "pays"))     # does it move toward what its cornerstone did?
        _append(e)
        made.append(e)
        print(f"OBS proposal {pid} (from {e['cornerstone']}, {e['surprise_bits']} bits, it {e['outcome']}): under {parent}, "
              f"\"{e['text']}\" -- {n} indexed cases pay {100 * r:.0f}% against the parent's {100 * qr:.0f}%; "
              f"would save {e['bits_total']:.0f} bits ({e['bits_per_case']:.3f} a case); "
              f"{'moves toward' if e['explains'] else 'moves AWAY from'} its cornerstone's outcome")
    if not made:
        print("OBS no proposal: no attack surprise above the threshold, or none with enough neighbours")
    return made


def accept(pid: str, text: str | None = None) -> str:
    from .watcher import ATTACK_ATTRS
    p = state()[pid]
    if p["status"] != "proposed":
        raise SystemExit(f"{pid} is already {p['status']}")
    r = add_rule(p["kind"], p["parent"], p["cond"], (text or p["text"]) + f" (proposed by the watcher, {pid})", [p["cornerstone"]],
                 attrs=set(ATTACK_ATTRS))
    _append(dict(id=pid, decision="accepted", rule=r.id, by="claude", time=time.strftime("%Y-%m-%d %H:%M %Z")))
    print(f"accepted {pid} as rule {r.id}")
    return r.id


def reject(pid: str, note: str) -> None:
    p = state()[pid]
    if p["status"] != "proposed":
        raise SystemExit(f"{pid} is already {p['status']}")
    _append(dict(id=pid, decision="rejected", note=note, by="claude", time=time.strftime("%Y-%m-%d %H:%M %Z")))
    print(f"rejected {pid}: {note}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "list", "accept", "reject"])
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--min-bits", type=float, default=1.5)
    ap.add_argument("--id", default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.batch, a.top, a.min_bits)
    elif a.cmd == "list":
        for p in state().values():
            print(f"{p['id']} [{p['status']}] under {p['parent']}: {p['text']} -- {p['cases']} cases, {100 * p['rate']:.0f}% vs "
                  f"{100 * p['parent_rate']:.0f}%, {p['bits_total']:.0f} bits (from {p['cornerstone']})")
    elif a.cmd == "accept":
        accept(a.id, a.text)
    else:
        reject(a.id, a.note)


if __name__ == "__main__":
    main()
