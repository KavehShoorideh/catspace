"""PLAN 78.62 D: THE SMALLEST FRAME. For each input that carries weight in the heads of stage C (here: the CONCEPTS of 78.54, the only inputs that are a conjunction of relations the engine exposes, kept by the
fit on the thinned rows), is there a PART of it of one relation less, among the 36152 concepts of 78.54, that carries the same weight; the input is kept only for what it adds. The statements, the kinds, the edge
distances and the ratings have no parts of that kind in what the engine exposes: NOT READ (stated, not hidden).

A concept's words are clauses joined by `;`. A part of one relation less: the words with one clause removed (variables that no clause mentions any more drop out; the attribute of a variable, written at its first
mention, stays with it). It is looked up among all concepts by the form of its words under every renaming of the variables of each type (a concept is the same pattern under a renaming). Its counts (ours less theirs)
are read on the thinned rows by the engine (`pattern_count`) exactly as for the concept itself. THE TEST: on the thinned rows, the head with the concept against the head with the concept replaced by the part (every
other input as in stage C, refitted on the same fit rows), the log loss on the test games with a 90 pct band over games; the part carries the same weight if replacing the concept does not make the loss worse
(the band of the loss with the part less the loss with the concept includes zero or is below it); then the concept adds nothing and the part is the smaller frame. The inputs that survive are printed in words.

  .venv/bin/python -m catspace.incubator.analyst.discovery.tactical_smallest --state-dir D --chances-dir C --out-dir O
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2 as LAYER2
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import tactical_chances as TC
from catspace.incubator.analyst.discovery import tactical_state as TS
from catspace.incubator.analyst.discovery import tactical_tables as TT
from catspace.io import paths

VAR = re.compile(r"\b([upl])(\d+)\b")
ATTR = re.compile(r"\b([upl]\d+) \(([^)]*)\)")
MAX_PERMS = 5000
DRAWS = 300


def parse(words: str) -> "tuple[list, dict]":
    """(clauses with their variables, {variable: attribute words}) of a concept's words."""
    clauses = [c.strip() for c in words.split(";") if c.strip()]
    attrs = {}
    out = []
    for c in clauses:
        for v, a in ATTR.findall(c):
            attrs.setdefault(v, a)
        out.append(re.sub(r" \([^)]*\)", "", c))
    return out, attrs


def canon(clauses: list, attrs: dict) -> "tuple | None":
    """The form of (clauses, attributes) least under every renaming of the variables of each type; None when there are too many renamings."""
    used = sorted({m.group(0) for c in clauses for m in VAR.finditer(c)})
    by_type = {}
    for v in used:
        by_type.setdefault(v[0], []).append(v)
    n = 1
    for vs in by_type.values():
        for k in range(2, len(vs) + 1):
            n *= k
    if n > MAX_PERMS:
        return None
    best = None
    types = sorted(by_type)
    for perms in itertools.product(*[itertools.permutations(range(len(by_type[t]))) for t in types]):
        ren = {}
        for t, pm in zip(types, perms):
            for v, k in zip(by_type[t], pm):
                ren[v] = f"{t}{k}"
        cl = tuple(sorted(VAR.sub(lambda m: ren[m.group(0)], c) for c in clauses))
        at = tuple(sorted((ren[v], a) for v, a in attrs.items() if v in ren))
        if best is None or (cl, at) < best:
            best = (cl, at)
    return best


def shape(clauses: list, attrs: dict) -> tuple:
    """A cheap form that equal patterns share: the clauses with each variable reduced to its type, and the attribute words, sorted."""
    cl = tuple(sorted(VAR.sub(lambda m: m.group(1), c) for c in clauses))
    used = {m.group(0) for c in clauses for m in VAR.finditer(c)}
    return cl, tuple(sorted(a for v, a in attrs.items() if v in used))


def parts_of(words: str) -> list:
    """The words with one clause removed, as (clauses, attrs), where something is left."""
    cl, at = parse(words)
    out = []
    if len(cl) < 2:
        return out
    for i in range(len(cl)):
        rest = cl[:i] + cl[i + 1:]
        used = {m.group(0) for c in rest for m in VAR.finditer(c)}
        out.append((rest, {v: a for v, a in at.items() if v in used}))
    return out


def load_concepts(cdir: Path) -> "tuple[list, dict]":
    keys = json.loads((cdir / "concept_keys.json").read_text())
    rows = []
    with open(cdir / "concepts.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append((r["dc"], r["definition"]))
    return rows, keys


def find_parts(words: str, by_shape: dict) -> list:
    """The concepts (dc ids) that equal the concept's words with one clause removed."""
    found = []
    for cl, at in parts_of(words):
        cand = by_shape.get(shape(cl, at), [])
        if not cand:
            continue
        c0 = canon(cl, at)
        if c0 is None:
            continue
        for dc, c1 in cand:
            if c1 == c0 and dc not in found:
                found.append(dc)
    return found


def counts_on(model, goal: str, games: list, rows: "tuple[np.ndarray, np.ndarray]", keys: list, threads: int) -> np.ndarray:
    """The counts (ours less theirs) of the concepts with `keys` at the rows (game numbers, plies)."""
    g, p = rows
    order, wires, rls = [], [], []
    for gx in np.unique(g):
        s = np.flatnonzero(g == gx)
        order.append(s)
        wires.append((None, games[int(gx)]["moves"]))
        rls.append([int(v) for v in p[s]])
    order = np.concatenate(order)
    raw = model.pattern_count(wires, rls, keys, 3, list(LAYER2.LEVELS), goal, threads)
    us, them = LAYER2.counts_matrix(raw, len(order), len(keys))
    diff = (us - them).toarray().astype(float)
    out = np.zeros_like(diff)
    out[order] = diff
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state-dir", default=paths.experiment("tactical_78_62", "state"))
    ap.add_argument("--chances-dir", default=paths.experiment("tactical_78_62", "chances"))
    ap.add_argument("--heads-dir", default=paths.experiment(*TS.HEADS_DIR))
    ap.add_argument("--concepts-dir", default=paths.experiment("unit_fates_78_54", "games_s5016_n500"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--draws", type=int, default=DRAWS)
    a = ap.parse_args(argv)
    out = Path(a.out_dir or paths.experiment("tactical_78_62", "smallest"))
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS tactical smallest: {msg}")
        print(lines[-1], flush=True)

    d = TT.load_all(Path(a.state_dir))
    t = TC.target_rows(d, Path(a.heads_dir))
    games_all = [json.loads(x) for x in (Path(a.heads_dir) / "games.jsonl").read_text().splitlines() if x.strip()]
    ids_sample = [g["id"] for g in d["games"]]
    sample_games = [next(g for g in games_all if g["id"] == i) for i in ids_sample]
    mdl = json.loads((Path(a.chances_dir) / "models.json").read_text())
    nfull = len(mdl["names"]["K+S+E+R"])
    rows, keys = load_concepts(Path(a.concepts_dir))
    say(f"concepts of 78.54 read: {len(rows)}; the kept 64 carry the rows' concept columns")
    by_shape: dict = {}
    nskip = 0
    for dc, w in rows:
        cl, at = parse(w)
        c = canon(cl, at)
        if c is None:
            nskip += 1
            continue
        by_shape.setdefault(shape(cl, at), []).append((dc, c))
    say(f"concepts indexed by their form: {sum(len(v) for v in by_shape.values())}; too many renamings to index (not matched): {nskip}")
    concept_names = t["meta"]["concepts"]                              # "dcN: words"
    cdc = [n.split(":", 1)[0] for n in concept_names]
    words_of = dict(rows)
    thin = np.flatnonzero(t["cmask"])
    fit, test = np.flatnonzero(t["fit"]), np.flatnonzero(~t["fit"])
    tf, tt = np.intersect1d(thin, fit), np.intersect1d(thin, test)
    base_x = TC.sets_of(t)["K+S+E+R"]
    model = GL.model(HD.GAME, HD.VARIANT)
    model.learn_profile([(None, g["moves"]) for g in sample_games[:50]])
    goal = G2M.goal_kind_of(HD.VARIANT)
    result = []
    for tag, key in (("us", "y_us"), ("them", "y_them")):
        m = mdl["models"][f"{tag}|K+S+E+R+C thinned"]
        coef = np.array(m["coef"])
        cw = coef[nfull:]
        carry = [i for i in np.flatnonzero(cw != 0)]
        say(f"[{tag}] concepts that carry weight in the head with concepts (thinned rows): {len(carry)} of {len(cw)}")
        y = t[key].astype(float)
        for i in sorted(carry, key=lambda k: -abs(cw[k])):
            dc = cdc[i]
            parts = find_parts(words_of[dc], by_shape)
            parts = [p for p in parts if p in keys]
            if not parts:
                result.append((tag, dc, None))
                say(f"[{tag}] {dc} (weight {cw[i]:+.3f} per standard deviation): {words_of[dc]} -- no part of one relation less among the concepts of 78.54; it stays")
                continue
            pk = [int(keys[p]["key"]) for p in parts]
            px = counts_on(model, goal, sample_games, (t["game"][thin], t["ply"][thin]), pk, a.threads)
            for pdc, col in zip(parts, px.T):
                full = np.zeros((len(t["game"]), 1)); full[thin, 0] = col
                x_with = np.hstack([base_x, t["C"][:, [i]]])
                x_part = np.hstack([base_x, full])
                m1 = TC.fit_logit(x_with[tf], y[tf], t["game"][tf])
                m2 = TC.fit_logit(x_part[tf], y[tf], t["game"][tf])
                l1 = TC.row_loss(TC.predict_logit(m1, x_with[tt]), y[tt])
                l2 = TC.row_loss(TC.predict_logit(m2, x_part[tt]), y[tt])
                g, lo, hi = TC.paired_gain(t["game"][tt], l1, l2, a.draws, 6263)       # loss with the concept less loss with the part: positive = the part is better
                same = hi >= 0
                result.append((tag, dc, (pdc, g, lo, hi, same)))
                say(f"[{tag}] {dc} (weight {cw[i]:+.3f}): {words_of[dc]} -- part {pdc}: {words_of[pdc]}; test log loss with the concept less with the part {g:+.5f} [{lo:+.5f}, {hi:+.5f}] over {len(tt)} thinned test rows; "
                    f"{'the part carries the same weight (the concept adds nothing the band can show)' if same else 'the concept carries more than the part: it stays'}")
    # the inputs that survive
    for tag in ("us", "them"):
        every = {dc for tg, dc, r in result if tg == tag}
        gone = sorted({dc for tg, dc, r in result if tg == tag and r is not None and r[4]})        # some part carries the same weight
        surv = sorted(every - set(gone))
        say(f"[{tag}] inputs that survive as they are (no smaller part carries the same weight): {len(surv)}: " + "; ".join(f"{dc}: {words_of[dc]}" for dc in surv))
        say(f"[{tag}] concepts replaced by a part of one relation less: {len(gone)}")
    say("statements, kinds, edge distances and ratings: no parts of one relation less are exposed by the engine; not read")
    (out / "smallest_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
