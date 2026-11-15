"""PLAN 78.53: concepts mined bottom-up as patterns over the layer-1 graph, and the evidence for them on the arrival target of 78.49. A measurement; nothing is wired into any player, no concept enters
any value (AGENTS rule 7), nothing is written to the real store. Game-free: no object of any game is named here; a kind is a word read from the graph.

  B. MINE on the fit rows only (the quiet positions of the games with an even id): sizes 1 to 3 (graph_layer2.enumerate_patterns, the matching in Rust), the named fuse per size and level, then the
     reduction test (graph_layer2.determined) on the fit games.
  C. NAME: dcN by order of first mining, concepts.tsv and reduced.tsv under the experiment directory (our name empty); a mapping file beside is read if present.
  D. EVIDENCE: the 78.49 fit (graph, sparse linear L1 by cross-validation grouped by game on the even games; the rank correlation with the arrival target on the odd games with 90 pct bands by game) over
     the 26 base statements (P), the concepts alone (C), both (P+C), at 4, 8 and 16 plies; per kept concept its weight and its own correlation. An OBSERVATION beside, no fit: for the top
     concepts, the units each side had lost 8 plies later, by kind, in the instances where the concept held for a side, against all quiet positions.

  .venv/bin/python -m catspace.incubator.analyst.discovery.graph_layer2_measure --game NAME --games-dir D --reads R --opp-table T [--games N] [--fuse K] [--out-dir O]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import forcing_opportunity_measure as FO
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2 as G2
from catspace.incubator.analyst.discovery import subgoal_fit_measure as SF
from catspace.io import paths

HORIZON_OBS = 8
TOP = 8


def layout(t: dict, idx: np.ndarray) -> "tuple[list, list]":
    """(game indexes, the plies of each) of the table rows `idx` (ascending), in the order the engine numbers rows: across these games, by position in the list."""
    games, rows = [], []
    for i in idx:
        g = int(t["game"][i])
        if not games or games[-1] != g:
            games.append(g)
            rows.append([])
        rows[-1].append(int(t["ply"][i]))
    return games, rows


def goal_kind_of(variant: str) -> "str | None":
    import cs_engine as CE
    from catspace.incubator.analyst.discovery import question_sheet as QS
    env = QS.Env(variant)
    return QS.Reading(env, CE.state(variant, None, [])[0], {}, light=True).goal_kind()


def reduce_patterns(kept: dict, counts: dict, key_col: dict, fit_games: np.ndarray, fit_sel: np.ndarray) -> dict:
    """THE REDUCTION TEST on the fit rows for every kept pattern of size 2 and above: -> {hash: (reduced, number of parts used)}. `counts` (us, them) sparse columns by key_col."""
    us, them = counts
    half_a = (fit_games % 4 == 0)
    half = np.concatenate([half_a, half_a])
    out = {}
    cols: dict = {}

    def col(h: int) -> np.ndarray:
        if h not in cols:
            c = key_col[h]
            cols[h] = np.concatenate([us[:, c].toarray().ravel()[fit_sel], them[:, c].toarray().ravel()[fit_sel]])
        return cols[h]
    for size in sorted(kept):
        if size == 1:
            continue
        for h, r in kept[size].items():
            parts = [p for p in r["parts"] if p in key_col]
            if r["level"] == 3 and r["free_parent"] in key_col:
                parts.append(r["free_parent"])
            vec = np.stack([col(p) for p in parts], axis=1) if parts else np.zeros((len(half), 0))
            out[h] = (G2.determined(vec, col(h), half), len(parts))
    return out


def observe_losses(tab: np.ndarray, kinds: list, concept_counts, concept_cols: list, names: list, label: str) -> list:
    """The observation: per top concept, over the (position, side) instances where it held for a side, the units each side had lost HORIZON_OBS plies later, by kind, per instance; beside all instances."""
    nk = len(kinds)
    valid = tab[:, 0] == 1
    now_us, now_them = tab[:, 1:1 + nk], tab[:, 1 + nk:1 + 2 * nk]
    then_us, then_them = tab[:, 1 + 2 * nk:1 + 3 * nk], tab[:, 1 + 3 * nk:1 + 4 * nk]
    lost_us = np.maximum(0, now_us - then_us).astype(float)
    lost_them = np.maximum(0, now_them - then_them).astype(float)

    def row_text(who: str, holder: np.ndarray, other: np.ndarray, n: int) -> str:
        return f"{who}: " + ", ".join(f"{k} {holder[i]:.3f} / {other[i]:.3f}" for i, k in enumerate(kinds)) + f" (instances {n})"
    lines = []
    # base: every valid quiet position, each side as the holder in turn
    nv = int(valid.sum())
    base_h = (lost_us[valid].sum(0) + lost_them[valid].sum(0)) / (2 * max(1, nv))
    base_o = base_h.copy()
    lines.append(f"OBS graph layer2 {label}: lost {HORIZON_OBS} plies later, per instance, units of kind (the side holding / the other side), BASE all {2 * nv} (quiet position, side) instances: " + ", ".join(f"{k} {base_h[i]:.3f}" for i, k in enumerate(kinds)))
    us_m, them_m = concept_counts
    for c, name in zip(concept_cols, names):
        cu = us_m[:, c].toarray().ravel() > 0
        ct = them_m[:, c].toarray().ravel() > 0
        hu, ht = cu & valid, ct & valid
        n = int(hu.sum() + ht.sum())
        if n == 0:
            lines.append(f"OBS graph layer2 {label}: {name}: held in no instance with the horizon inside the game")
            continue
        holder = (lost_us[hu].sum(0) + lost_them[ht].sum(0)) / n
        other = (lost_them[hu].sum(0) + lost_us[ht].sum(0)) / n
        lines.append(f"OBS graph layer2 {label}: {name} {row_text('lost per instance by kind, holding side / other side', holder, other, n)}")
    return lines


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", required=True)
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--opp-table", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--games", type=int, default=0, help="a subset: the first N games by id")
    ap.add_argument("--fuse", type=int, default=G2.FUSE)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--no-fit", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    records = FO.read_games(a.games_dir)
    variant = records[0]["variant"]
    label = Path(a.games_dir).name + (f"_first{a.games}" if a.games else "")
    out = Path(a.out_dir or paths.experiment("graph_layer2_78_53", label))
    out.mkdir(parents=True, exist_ok=True)
    reads_by_id = {}
    with open(a.reads) as f:
        for line in f:
            g = json.loads(line)
            reads_by_id[g["game"]] = g["plies"]
    records = sorted(records, key=lambda r: r["game"])
    z = np.load(a.opp_table)
    a_opp = {k: z[k] for k in z.files if k != "n_games"}
    if a.games:
        records = records[:a.games]
    names, stats = SF.statements_of(variant, records, a.threads, str(out / f"statements_n{len(records)}.npz"))
    t = SF.table_of(records, reads_by_id, stats, a_opp)
    fit, test = SF.split_rows(t)
    lines = [f"OBS graph layer2 {label}: games {len(records)}, quiet rows {len(t['ply'])} (fit, even ids: {len(fit)}; test, odd: {len(test)})"]
    model = GL.model(a.game, variant)
    kinds = [n[len("kind:"):] for ty, n in model.static_part()["nodes"] if ty == "kind"]
    wires = [(r.get("start"), list(r["moves"])) for r in records]
    goal = goal_kind_of(variant)
    all_idx = np.arange(len(t["ply"]))

    # ---- B: mine on the fit rows
    fg, fr = layout(t, fit)
    t1 = time.perf_counter()
    mined = G2.enumerate_patterns(model, [wires[g] for g in fg], fr, goal, a.threads, a.fuse, log=lambda m: lines.append(f"OBS graph layer2 {label}: {m}"))
    lines.append(f"OBS graph layer2 {label}: mining on the fit rows {time.perf_counter() - t1:.1f} s wall over {a.threads} threads (fuse {a.fuse} patterns per size and level)")
    kept = mined["kept"]
    # ---- counts of every kept pattern and the kind-free parents of the kind-bound ones, on every quiet row
    keys: list = []
    key_col: dict = {}
    for size in sorted(kept):
        for h in kept[size]:
            key_col[h] = len(keys)
            keys.append(h)
    for size in sorted(kept):
        for r in kept[size].values():
            if r["level"] == 3 and r["free_parent"] not in key_col:
                key_col[r["free_parent"]] = len(keys)
                keys.append(r["free_parent"])
    ag, ar = layout(t, all_idx)
    t2 = time.perf_counter()
    raw = model.pattern_count([wires[g] for g in ag], ar, keys, G2.MAX_SIZE, list(G2.LEVELS), goal, a.threads)
    us, them = G2.counts_matrix(raw, len(all_idx), len(keys))
    lines.append(f"OBS graph layer2 {label}: counting {len(keys)} patterns on {len(all_idx)} quiet rows {time.perf_counter() - t2:.1f} s wall over {a.threads} threads")
    # ---- the reduction test on the fit rows
    t3 = time.perf_counter()
    red = reduce_patterns(kept, (us, them), key_col, t["id"][fit].astype(int), fit)
    lines.append(f"OBS graph layer2 {label}: reduction test {time.perf_counter() - t3:.1f} s")
    concepts, reduced_rows = [], []
    for size in sorted(kept):
        for lv in G2.LEVELS:
            rs = sorted(((h, r) for h, r in kept[size].items() if r["level"] == lv), key=lambda x: (-x[1]["support"], x[0]))
            nred = sum(1 for h, _ in rs if red.get(h, (False, 0))[0])
            f = mined["found"][size][lv]
            lines.append(f"OBS graph layer2 {label}: size {size} level {lv}: found {f}, dropped by the fuse {f - len(rs)}, tested {len(rs)}, reduced {nred}, kept as concepts {len(rs) - nred}")
            for h, r in rs:
                pat = G2.decode(r["bytes"])
                text = G2.canonical_text(pat, kinds)
                ident = G2.hash_of(text)
                if red.get(h, (False, 0))[0]:
                    reduced_rows.append((ident, size, lv, G2.plain_words(pat, kinds), "; ".join(sorted(G2.hash_of(G2.canonical_text(G2.decode(kept[size - 1][p]["bytes"]), kinds))[:12] for p in r["parts"] if p in kept.get(size - 1, {})))))
                else:
                    concepts.append(dict(h=h, size=size, level=lv, ident=ident, text=text, words=G2.plain_words(pat, kinds), support=r["support"]))
    ids = [c["ident"] for c in concepts]
    assert len(set(ids)) == len(ids), "two concepts with one hash"
    for n, c in enumerate(concepts, 1):
        c["dc"] = f"dc{n}"
    G2.write_concepts(str(out / "concepts.tsv"), [(c["dc"], c["ident"], c["size"], c["level"], c["words"]) for c in concepts])
    (out / "concept_keys.json").write_text(json.dumps({c["dc"]: {"hash": c["ident"], "key": str(c["h"]), "bytes": kept[c["size"]][c["h"]]["bytes"].hex(), "size": c["size"], "level": c["level"]} for c in concepts}))
    (out / "reduced.tsv").write_text("hash\tsize\tlevel\tdefinition\tparts\n" + "\n".join("\t".join(str(x) for x in r) for r in reduced_rows) + "\n")
    mapping = G2.read_mapping(str(out / "mapping.tsv"))
    lines.append(f"OBS graph layer2 {label}: concepts {len(concepts)} (dc1..dc{len(concepts)}), reduced {len(reduced_rows)}, mapping file names {len(mapping)}")
    cols = [key_col[c["h"]] for c in concepts]
    cdiff = (us[:, cols] - them[:, cols]).toarray()
    np.savez_compressed(str(out / "concept_values.npz"), values=cdiff.astype(np.int16), game=t["game"], ply=t["ply"])
    if a.no_fit:
        for line in lines:
            print(line, flush=True)
        return 0

    # ---- D: the fit
    t4 = time.perf_counter()
    stat_names = [n for n, _x in names]
    cnames = [c["dc"] for c in concepts]
    feats = {"P": t["P"], "C": cdiff, "PC": np.hstack([t["P"], cdiff])}
    fnames = {"P": stat_names, "C": cnames, "PC": stat_names + cnames}
    models = {}
    for which in ("P", "C", "PC"):
        for h in SF.HORIZONS:
            y = t[f"arr{h}"][fit]
            ok = ~np.isnan(y)
            rows = fit[ok]
            m = SF.fit_sparse(feats[which][rows], t[f"arr{h}"][rows], t["game"][rows])
            models[(which, h)] = m
            lines.append(f"OBS graph layer2 {label}: fit {which} arrival h={h}: penalty {m['alpha']:.5f} by {SF.FOLDS}-fold cross-validation grouped by game on the even games, features kept {int((m['coef'] != 0).sum())} of {len(m['coef'])}")
    ta = {k: t[k][test] for k in t if k not in ("P", "F")}
    for (which, h), m in models.items():
        p = SF.predict(m, feats[which][test])
        ta[f"pred_{which}_arr{h}"] = p
    for i in range(len(cnames)):
        ta[f"c{i}"] = cdiff[test][:, i]
    for h in SF.HORIZONS:
        y = f"arr{h}"
        readings = [(f"h={h} arrival target: fit on the 26 base statements", f"pred_P_{y}", y), (f"h={h} arrival target: fit on the concepts alone", f"pred_C_{y}", y),
                    (f"h={h} arrival target: fit on both", f"pred_PC_{y}", y), (f"h={h} arrival target: count difference of stage A (78.49)", "count", y)]
        more, _vals = SF.band_lines(f"graph layer2 {label}", ta, a.draws, readings)
        lines += more
    for (which, h), m in sorted(models.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if which == "P":
            continue
        y = f"arr{h}"
        kept_i = [i for i, c in enumerate(m["coef"]) if c != 0]
        for i in sorted(kept_i, key=lambda j: -abs(m["coef"][j]))[:25]:
            nm = fnames[which][i]
            if nm.startswith("dc"):
                k = int(nm[2:]) - 1
                own = SF.spec_for(f"c{k}", y)(ta)
                lines.append(f"OBS graph layer2 {label}: kept {which} h={h}: {nm} [size {concepts[k]['size']}, level {concepts[k]['level']}] weight {m['coef'][i]:+.5f} per standard deviation, own rank correlation on the test half {own:+.4f}: {concepts[k]['words']}")
            else:
                lines.append(f"OBS graph layer2 {label}: kept {which} h={h}: {nm} (base statement) weight {m['coef'][i]:+.5f} per standard deviation")
        if not kept_i:
            lines.append(f"OBS graph layer2 {label}: kept {which} h={h}: none")
    lines.append(f"OBS graph layer2 {label}: fits and bands {time.perf_counter() - t4:.1f} s")

    # ---- the observation: what each side had lost 8 plies later where the top concepts held (no fit)
    top: list = []
    m8 = models[("C", HORIZON_OBS)]
    for i in sorted((i for i, c in enumerate(m8["coef"]) if c != 0), key=lambda j: -abs(m8["coef"][j]))[:TOP]:
        top.append(i)
    kn, raw_t = model.unit_tables([wires[g] for g in ag], ar, HORIZON_OBS, a.threads)
    tab = np.frombuffer(raw_t, dtype=np.int16).reshape(len(all_idx), 1 + 4 * len(kn)).astype(int)
    lines += observe_losses(tab, kn, (us, them), [cols[i] for i in top], [concepts[i]["dc"] for i in top], label)
    lines.append(f"OBS graph layer2 {label}: total wall {time.perf_counter() - t0:.1f} s")
    for line in lines:
        print(line, flush=True)
    (out / "obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
