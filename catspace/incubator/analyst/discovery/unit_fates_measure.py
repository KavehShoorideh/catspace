"""PLAN 78.54: WHAT BECOMES OF EACH UNIT A CONCEPT BINDS. A measurement and a concepts file in a sandbox; nothing is wired into any player, nothing is written to the real store. Game-free: no unit, place or kind
of any game is named here; a kind is a word read from the graph at run time. No worth table.

THE RULES OF SELECTION, stated here BEFORE the 500-game run (and in docs/ANALYST_PLAN.md):
  Rows: the quiet positions of 78.49 (forcing_arrival_measure.quiet_indices). Halves by game id: A = id mod 4 = 0, B = id mod 4 = 2 (together the even games, the fit half of 78.49), T = the odd games (the test
  half, SF.split_rows). A selects, B confirms, T reports; the base rates of a half are taken over that half's own quiet positions.
  MINIMUM SUPPORT: a pattern is a candidate when it occurs on at least MIN_SUPPORT = 30 quiet positions of half A (the fuse by support of 78.53 is gone: sizes 1 to 3, all four levels, every pattern with that support;
  growth is exact because a pattern's parts occur wherever it does). What is still cut: a pattern with fewer than 30 positions on A, and a pattern none of whose cells is READABLE.
  CELL: (pattern, unit variable, horizon in 4, 8, 16). READABLE on A: at least MIN_CELL_N = 30 distinct units bound over the positions, in at least MIN_CELL_GAMES = 10 games. Its DEPARTURE d = taken rate less the
  base rate of being taken for the same kinds and side-to-move states of the units actually bound (the mean of the units' own residuals); its STANDARD ERROR is from the games' summed residuals (units of one game are
  not independent draws): se = sqrt(sum of squares of the games' residual sums) / n; z = d / se.
  SCORE of a pattern on A: the largest |z| over its readable cells; that cell (variable, horizon) is the pattern's SELECTED CELL.
  CONFIRMED on B: the selected cell has at least MIN_B_N = 15 units on B, the same sign of d, and |d| at least HALF the size of d on A. Nothing else is tuned on B; B is not searched for a better cell.
  REDUCED: a confirmed pattern is reduced to a part (one relation node less, connected) when, for the variable the part also holds, d on A is within REDUCE_Z = 2 standard errors of the pattern's own of the part's d
  at the same horizon (|d_pattern - d_part| <= 2 se_pattern). Reduced patterns are recorded with their part, they are not concepts.
  KEPT = confirmed and not reduced. Reported on T: sorted by |z| of the selected cell on T. A reference line, not a filter: the largest |z| that N independent noise cells would reach, sqrt(2 ln N), N the readable cells on A.

  .venv/bin/python -m catspace.incubator.analyst.discovery.unit_fates_measure --game NAME --games-dir D --reads R [--games N] [--out-dir O] [--threads K]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import forcing_arrival_measure as FA
from catspace.incubator.analyst.discovery import forcing_opportunity_measure as FO
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2 as G2
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import subgoal_fit_measure as SF
from catspace.incubator.analyst.discovery import unit_fates as UF
from catspace.io import paths

HORIZONS = (4, 8, 16)
MAX_SIZE = 3
MIN_SUPPORT = 30
MIN_CELL_N = 30
MIN_CELL_GAMES = 10
MIN_B_N = 15
HALF_SIZE = 0.5
REDUCE_Z = 2.0
PROFILE_GAMES = 50
AGREE_FIRST, AGREE_GAMES, AGREE_EVERY = 100, 60, 3
TOP = 15
MAXV = 9


def rates_of(counts: np.ndarray, group: int, nk: int, ncls: int) -> np.ndarray:
    """Base rates [horizon][kind][state][class] of one group from the counts the engine returns ([group][horizon][kind][state][classes then the total])."""
    c = counts.reshape(-1, len(HORIZONS), nk, 2, ncls + 1)[group].astype(float)
    tot = c[..., ncls:]
    return np.where(tot > 0, c[..., :ncls] / np.maximum(tot, 1), 0.0)


def var_words(p: G2.Pattern, v: int, kinds: list) -> str:
    bits = ["mine" if p.sides[v] else "theirs"]
    if p.level >= 1 and p.goal[v]:
        bits.append("the goal unit of its side")
    if p.level == 2:
        bits.append("on an edge" if p.edge[v] else "not on an edge")
    if p.level == 3:
        bits.append(kinds[p.kind[v]])
    return f"u{v} ({', '.join(bits)})"


def cell_table(flat: np.ndarray, ncls: int) -> dict:
    """The cells the engine returns as arrays: cand, var, hi, n, games, d, se, z and the class counts and expected."""
    a = flat.reshape(-1, 7 + 2 * ncls)
    n, r1, r2 = a[:, 3], a[:, 5], a[:, 6]
    ok = (n > 0) & (r2 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(n > 0, r1 / np.maximum(n, 1), 0.0)
        se = np.where(ok, np.sqrt(r2) / np.maximum(n, 1), np.inf)
        z = np.where(ok, r1 / np.sqrt(np.where(ok, r2, 1.0)), 0.0)
    t = dict(cand=a[:, 0].astype(int), var=a[:, 1].astype(int), hi=a[:, 2].astype(int), n=n, games=a[:, 4], d=d, se=se, z=z, cnt=a[:, 7:7 + ncls], exp=a[:, 7 + ncls:])
    t["index"] = {(int(c), int(v), int(h)): i for i, (c, v, h) in enumerate(zip(t["cand"], t["var"], t["hi"]))}
    return t


def lines_for_base(label: str, kinds: list, counts: np.ndarray, groups: dict, ncls: int) -> list:
    """OBS lines: per kind and side-to-move state and horizon, the fates over all quiet positions (the three halves summed) and the share taken, by whom."""
    nk = len(kinds)
    c = counts.reshape(-1, len(HORIZONS), nk, 2, ncls + 1).astype(np.int64).sum(0)
    names = UF.class_names(kinds)
    out = []
    for hi, h in enumerate(HORIZONS):
        for k, kind in enumerate(kinds):
            for tm, tmw in ((1, "side to move"), (0, "side not to move")):
                row = c[hi, k, tm]
                n = int(row[ncls])
                if n == 0:
                    continue
                taken = int(row[3:ncls].sum())
                by = ", ".join(f"{names[3 + j]} {int(row[3 + j])}" for j in range(nk) if row[3 + j])
                out.append(f"OBS unit fates {label}: base h={h} kind {kind} ({tmw}): units {n}; stayed {int(row[0])} ({row[0] / n:.3f}), moved {int(row[1])} ({row[1] / n:.3f}), other {int(row[2])} ({row[2] / n:.3f}), "
                           f"taken {taken} ({taken / n:.3f}){': ' + by if by else ''}")
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", required=True)
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--games", type=int, default=0, help="a subset: the first N games by id")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    records = FO.read_games(a.games_dir)
    variant = records[0]["variant"]
    label = Path(a.games_dir).name + (f"_first{a.games}" if a.games else "")
    out = Path(a.out_dir or paths.experiment("unit_fates_78_54", label))
    out.mkdir(parents=True, exist_ok=True)
    reads_by_id = {}
    with open(a.reads) as f:
        for line in f:
            g = json.loads(line)
            reads_by_id[g["game"]] = g["plies"]
    records = sorted(records, key=lambda r: r["game"])
    if a.games:
        records = records[:a.games]
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS unit fates {label}: {msg}")
        print(lines[-1], flush=True)

    # ---- the quiet rows, the halves
    ids, games_ix, plies = [], [], []
    for g, r in enumerate(records):
        pl = reads_by_id[r["game"]]
        own, oth = FA.counts_of_game(pl)
        for t in FA.quiet_indices(pl, own, oth):
            ids.append(r["game"])
            games_ix.append(g)
            plies.append(t)
    t = {"id": np.array(ids), "game": np.array(games_ix), "ply": np.array(plies)}
    fit, test = SF.split_rows(t)
    half_a = fit[t["id"][fit] % 4 == 0]
    half_b = fit[t["id"][fit] % 4 == 2]
    halves = {"A": half_a, "B": half_b, "T": test}
    say(f"games {len(records)}, quiet rows {len(ids)}: half A (id mod 4 = 0) {len(half_a)}, half B (id mod 4 = 2) {len(half_b)}, test half T (odd ids) {len(test)}")
    say(f"rules: minimum support {MIN_SUPPORT} positions on A; cell readable at {MIN_CELL_N} units in {MIN_CELL_GAMES} games; confirmed on B at {MIN_B_N} units, same sign, {HALF_SIZE:.0%} of the size; reduced within {REDUCE_Z} standard errors of a part; sizes 1 to {MAX_SIZE}")
    model = GL.model(a.game, variant)
    kinds = [n[len("kind:"):] for ty, n in model.static_part()["nodes"] if ty == "kind"]
    nk = len(kinds)
    wires = [(r.get("start"), list(r["moves"])) for r in records]
    goal = G2M.goal_kind_of(variant)

    # ---- A: the profile and the agreement with the rules
    t1 = time.perf_counter()
    prof = model.learn_profile(wires[:PROFILE_GAMES])
    say(f"A profile learned from the bear nodes of every position of the first {PROFILE_GAMES} games: {len(prof)} (kind, line family, heading) continue: " + "; ".join(f"{k} {d} {'forward' if f else 'backward'}" for k, d, f in prof))
    import cs_engine as CE
    tot = [0] * 5
    positions = 0
    first = AGREE_FIRST if len(wires) >= AGREE_FIRST + AGREE_GAMES else 0
    for g in range(first, min(len(wires), first + AGREE_GAMES)):
        s, mv = wires[g]
        for ply in range(0, len(mv), AGREE_EVERY):
            at = CE.state(variant, s, mv[:ply])[0]
            tot = [x + y for x, y in zip(tot, model.through_check(at))]
            positions += 1
    say(f"A agreement with the rules' own answer for the side to move over {positions} positions of {min(AGREE_GAMES, len(wires))} games (units of the side to move, the end place held by the other side): through relations {tot[0]}, shut relations {tot[1]}, in both {tot[2]}, through only {tot[3]}, shut only {tot[4]}")
    both = {"mover": 0, "other": 0}
    for g in range(first, min(len(wires), first + 20)):
        s, mv = wires[g]
        for ply in range(0, len(mv), 5):
            gr = GL.Graph(model.static_part(), model.position(CE.state(variant, s, mv[:ply])[0]))
            if gr.mover is None:
                continue
            for nid in gr.of_type("through"):
                owner = nid.split(":")[1]
                both["mover" if owner == gr.mover else "other"] += 1
    say(f"A through relations held by the side to move {both['mover']} and by the other side {both['other']} over 20 games, every 5th position")
    say(f"A done in {time.perf_counter() - t1:.1f} s")

    # ---- B: the tracker and the base rates
    t2 = time.perf_counter()
    tc = model.track_counts(wires, a.threads)
    say(f"B tracker over {len(wires)} games: plies {tc[0]}, unit steps {tc[1]}: alive {tc[2]}, taken {tc[3]}, other {tc[4]}; units paired among several of one owner and kind {tc[5]}")
    all_idx = np.arange(len(ids))
    ag, ar = G2M.layout(t, all_idx)
    group_of = {}
    for gi in set(ag):
        gid = records[gi]["game"]
        group_of[gi] = 0 if gid % 4 == 0 else 1 if gid % 4 == 2 else 2 if gid % 2 == 1 else 1
    names_k, craw = model.fate_base([wires[g] for g in ag], ar, [group_of[g] for g in ag], 3, list(HORIZONS), a.threads)
    assert names_k == kinds
    ncls = 3 + nk
    counts = np.array(craw, dtype=np.int64)
    for ln in lines_for_base(label, kinds, counts, group_of, ncls):
        lines.append(ln)
        print(ln, flush=True)
    say(f"B tracker and base rates {time.perf_counter() - t2:.1f} s")
    rates = {"A": rates_of(counts, 0, nk, ncls), "B": rates_of(counts, 1, nk, ncls), "T": rates_of(counts, 2, nk, ncls)}

    # ---- D (candidates): mine on A by the minimum support
    t3 = time.perf_counter()
    fg, fr = G2M.layout(t, half_a)
    allowed = None
    cands: list = []
    found_by_size: dict = {}
    for size in range(1, MAX_SIZE + 1):
        found, rec = model.pattern_mine([wires[g] for g in fg], fr, size, list(G2.LEVELS), allowed, goal, a.threads, MIN_SUPPORT)
        found_by_size[size] = (found, [sum(1 for r in rec if r[1] == lv) for lv in G2.LEVELS])
        say(f"D mined size {size} on half A: patterns met per level {found}, with at least {MIN_SUPPORT} positions per level {found_by_size[size][1]}")
        allowed = [r[0] for r in rec]
        for h, lv, sup, tot_, b, parts, maps in rec:
            p = G2.decode(bytes(b))
            cands.append(dict(h=h, size=size, level=lv, support=sup, bytes=bytes(b), pat=p, parts=parts, maps=maps))
    say(f"D candidates {len(cands)} mined in {time.perf_counter() - t3:.1f} s wall over {a.threads} threads")
    cand_of_hash = {c["h"]: i for i, c in enumerate(cands)}
    keys = [c["h"] for c in cands]

    # ---- C: the cells on A, the selection
    t4 = time.perf_counter()

    def read_cells(which: str, ckeys: list) -> dict:
        gi, rows_ = G2M.layout(t, halves[which])
        nc, flat = model.fate_cells([wires[g] for g in gi], rows_, ckeys, rates[which].ravel().tolist(), list(HORIZONS), MAX_SIZE, goal, a.threads)
        assert nc == ncls
        return cell_table(np.array(flat), ncls)
    cell_a = read_cells("A", keys)
    say(f"C cells on half A: {len(cell_a['n'])} (pattern, variable, horizon) cells with a unit, read in {time.perf_counter() - t4:.1f} s wall")
    readable = (cell_a["n"] >= MIN_CELL_N) & (cell_a["games"] >= MIN_CELL_GAMES) & np.isfinite(cell_a["se"])
    n_read = int(readable.sum())
    star = math.sqrt(2 * math.log(max(2, n_read)))
    say(f"D readable cells on A {n_read}; reference line sqrt(2 ln N) = {star:.2f} standard errors; patterns with a readable cell {len(set(cell_a['cand'][readable].tolist()))} of {len(cands)}")
    best: dict = {}
    for i in np.flatnonzero(readable):
        c = int(cell_a["cand"][i])
        if c not in best or abs(cell_a["z"][i]) > abs(cell_a["z"][best[c]]):
            best[c] = int(i)
    over_star = sum(1 for i in best.values() if abs(cell_a["z"][i]) >= star)
    say(f"D patterns scored on A {len(best)}; cut for no readable cell {len(cands) - len(best)}; of those scored, with |z| on A at or above the reference line {over_star}")
    # ---- B: confirm
    t5 = time.perf_counter()
    scored = sorted(best)
    cell_b = read_cells("B", [cands[c]["h"] for c in scored])
    pos_in_scored = {c: j for j, c in enumerate(scored)}
    confirmed: list = []
    for c in scored:
        ia = best[c]
        key = (pos_in_scored[c], int(cell_a["var"][ia]), int(cell_a["hi"][ia]))
        ib = cell_b["index"].get(key)
        if ib is None or cell_b["n"][ib] < MIN_B_N or not np.isfinite(cell_b["se"][ib]):
            continue
        da, db = cell_a["d"][ia], cell_b["d"][ib]
        if np.sign(da) == np.sign(db) and abs(db) >= HALF_SIZE * abs(da):
            confirmed.append((c, ia, ib))
    say(f"D confirmed on half B (same sign, at least {HALF_SIZE:.0%} of the size, at least {MIN_B_N} units): {len(confirmed)} of {len(scored)} scored; cut on B {len(scored) - len(confirmed)}; read in {time.perf_counter() - t5:.1f} s wall")
    # ---- reduction on A
    kept, reduced = [], []
    for c, ia, ib in confirmed:
        v, hi = int(cell_a["var"][ia]), int(cell_a["hi"][ia])
        why = None
        for ph, mp in cands[c]["maps"]:
            pc = cand_of_hash.get(ph)
            if pc is None or v >= len(mp) or mp[v] == 255:
                continue
            ip = cell_a["index"].get((pc, int(mp[v]), hi))
            if ip is None or cell_a["n"][ip] == 0:
                continue
            gap = abs(cell_a["d"][ia] - cell_a["d"][ip])
            if gap <= REDUCE_Z * cell_a["se"][ia] and (why is None or gap < why[1]):
                why = (pc, gap)
        (reduced if why else kept).append((c, ia, ib, why))
    say(f"D reduced to a part {len(reduced)}; kept as concepts {len(kept)}")
    # ---- T: report
    t6 = time.perf_counter()
    kept_c = [c for c, *_ in kept]
    cell_t = read_cells("T", [cands[c]["h"] for c in kept_c])
    pos_kept = {c: j for j, c in enumerate(kept_c)}
    rows = []
    for c, ia, ib, _ in kept:
        v, hi = int(cell_a["var"][ia]), int(cell_a["hi"][ia])
        it = cell_t["index"].get((pos_kept[c], v, hi))
        ib_cells = None
        rows.append(dict(c=c, v=v, hi=hi, ia=ia, ib=ib, it=it))
    say(f"T read in {time.perf_counter() - t6:.1f} s wall")
    # dcN by order of the sort by T
    def zt(r: dict) -> float:
        return abs(cell_t["z"][r["it"]]) if r["it"] is not None else -1.0
    rows.sort(key=lambda r: -zt(r))
    names_cls = UF.class_names(kinds)
    concepts = []
    for n, r in enumerate(rows, 1):
        c = cands[r["c"]]
        text = G2.canonical_text(c["pat"], kinds)
        r.update(dc=f"dc{n}", ident=G2.hash_of(text), text=text, words=G2.plain_words(c["pat"], kinds))
        concepts.append(r)
    # concepts whose selected cell on T holds the very same fates (same count, same by class) as a higher ranked one bind the same units: noted, not removed
    first_of: dict = {}
    for r in concepts:
        i = r["it"]
        if i is None:
            r["same_as"] = ""
            continue
        sig = (r["hi"], round(float(cell_t["n"][i])), tuple(int(x) for x in cell_t["cnt"][i]))
        first = first_of.setdefault(sig, r["dc"])
        r["same_as"] = "" if first == r["dc"] else first
    distinct = [r for r in concepts if not r["same_as"]]
    mapping = G2.read_mapping(str(out / "mapping.tsv"))
    HEAD = ["dc", "hash", "size", "level", "definition", "our name", "variable", "horizon", "support_A", "n_A", "taken_A", "base_A", "z_A", "n_B", "taken_B", "base_B", "z_B", "n_T", "taken_T", "base_T", "z_T", "same_units_as"]

    def cell_fields(tab: dict, i: "int | None") -> list:
        if i is None:
            return ["", "", "", ""]
        n = tab["n"][i]
        taken = tab["cnt"][i][3:].sum()
        base = tab["exp"][i][3:].sum()
        return [f"{n:.0f}", f"{taken / n:.4f}", f"{base / n:.4f}", f"{tab['z'][i]:+.2f}"]
    tsv = ["\t".join(HEAD)]
    for r in concepts:
        c = cands[r["c"]]
        tsv.append("\t".join([r["dc"], r["ident"], str(c["size"]), str(c["level"]), r["words"], mapping.get(r["ident"], ""), var_words(c["pat"], r["v"], kinds), str(HORIZONS[r["hi"]]), str(c["support"])]
                             + cell_fields(cell_a, r["ia"]) + cell_fields(cell_b, r["ib"]) + cell_fields(cell_t, r["it"]) + [r["same_as"]]))
    (out / "concepts.tsv").write_text("\n".join(tsv) + "\n")
    (out / "concept_keys.json").write_text(json.dumps({r["dc"]: {"hash": r["ident"], "key": str(cands[r["c"]]["h"]), "bytes": cands[r["c"]]["bytes"].hex(), "size": cands[r["c"]]["size"], "level": cands[r["c"]]["level"],
                                                                 "variable": r["v"], "horizon": HORIZONS[r["hi"]], "words": r["words"]} for r in concepts}))
    red_lines = ["hash\tsize\tlevel\tdefinition\tvariable\thorizon\td_A\tpart_hash\tgap_to_part"]
    for c, ia, ib, why in reduced:
        pc = cands[why[0]]
        red_lines.append("\t".join([G2.hash_of(G2.canonical_text(cands[c]["pat"], kinds)), str(cands[c]["size"]), str(cands[c]["level"]), G2.plain_words(cands[c]["pat"], kinds), str(int(cell_a["var"][ia])),
                                    str(HORIZONS[int(cell_a["hi"][ia])]), f"{cell_a['d'][ia]:+.4f}", G2.hash_of(G2.canonical_text(pc["pat"], kinds)), f"{why[1]:.4f}"]))
    (out / "reduced.tsv").write_text("\n".join(red_lines) + "\n")
    # the per-variable record on T of every kept concept: every variable, every horizon, every class
    rec = ["dc\tvariable\thorizon\tn\tgames\t" + "\t".join(f"{nm}\tbase {nm}" for nm in names_cls)]
    for r in concepts:
        pat = cands[r["c"]]["pat"]
        for v in range(pat.n_units()):
            for hi in range(len(HORIZONS)):
                i = cell_t["index"].get((pos_kept[r["c"]], v, hi))
                if i is None:
                    continue
                n = cell_t["n"][i]
                rec.append("\t".join([r["dc"], var_words(pat, v, kinds), str(HORIZONS[hi]), f"{n:.0f}", f"{cell_t['games'][i]:.0f}"] + [f"{cell_t['cnt'][i][k] / n:.4f}\t{cell_t['exp'][i][k] / n:.4f}" for k in range(ncls)]))
    (out / "fate_records.tsv").write_text("\n".join(rec) + "\n")
    say(f"D concepts {len(concepts)} (dc1..dc{len(concepts)}, by order of |z| on T), reduced {len(reduced)}, mapping file names {len(mapping)}")
    kept_in_a_sign = sum(1 for r in concepts if r["it"] is not None and np.sign(cell_t["d"][r["it"]]) == np.sign(cell_a["d"][r["ia"]]))
    say(f"D of the {len(concepts)} kept, the selected cell keeps its sign on T in {kept_in_a_sign}; |z| on T at or above 2: {sum(1 for r in concepts if zt(r) >= 2)}")
    say(f"D concepts that bind the same units as a higher ranked one (same selected cell on T, same fates): {len(concepts) - len(distinct)}; distinct {len(distinct)}")
    for rank, r in enumerate(distinct[:TOP], 1):
        c = cands[r["c"]]
        pat = c["pat"]
        i, ia = r["it"], r["ia"]
        n = cell_t["n"][i]
        taken, base = cell_t["cnt"][i][3:].sum(), cell_t["exp"][i][3:].sum()
        by = ", ".join(f"{names_cls[3 + k]} {cell_t['cnt'][i][3 + k]:.0f} (expected {cell_t['exp'][i][3 + k]:.1f})" for k in range(nk) if cell_t["cnt"][i][3 + k] or cell_t["exp"][i][3 + k] >= 0.5)
        say(f"TOP {rank}: {r['dc']} [size {c['size']}, level {c['level']}] {r['words']} | variable {var_words(pat, r['v'], kinds)} at h={HORIZONS[r['hi']]} | positions on A {c['support']}; "
            f"T: units {n:.0f} in {cell_t['games'][i]:.0f} games, taken rate {taken / n:.4f} against base {base / n:.4f}, departure {cell_t['d'][i]:+.4f}, z {cell_t['z'][i]:+.2f}; "
            f"A: z {cell_a['z'][ia]:+.2f}; B: z {cell_b['z'][r['ib']]:+.2f}; also held by {sum(1 for q in concepts if q['same_as'] == r['dc'])} other kept concepts with the same units; by taker: {by}")

    # ---- E: baseline and correction, on T, for the top concepts
    t7 = time.perf_counter()
    top = distinct[:TOP]
    tg, tr_ = G2M.layout(t, test)
    raw = model.pattern_count([wires[g] for g in tg], tr_, [cands[r["c"]]["h"] for r in top], MAX_SIZE, list(G2.LEVELS), goal, a.threads)
    us, them = G2.counts_matrix(raw, len(test), len(top))
    tables = {}
    for h in sorted({HORIZONS[r["hi"]] for r in top}):
        kn, rawt = model.unit_tables([wires[g] for g in tg], tr_, h, a.threads)
        tables[h] = np.frombuffer(rawt, dtype=np.int16).reshape(len(test), 1 + 4 * nk).astype(int)
    for j, r in enumerate(top):
        h = HORIZONS[r["hi"]]
        tab = tables[h]
        valid = tab[:, 0] == 1
        now_us, now_them, then_us, then_them = (tab[:, 1 + q * nk:1 + (q + 1) * nk] for q in range(4))
        lost_us = np.maximum(0, now_us - then_us).astype(float)
        lost_them = np.maximum(0, now_them - then_them).astype(float)
        net_mover = lost_them - lost_us                                   # theirs less ours, the side to move as the holder
        base_mover = net_mover[valid].sum(0) / max(1, int(valid.sum()))    # per quiet position
        hu = (us.toarray()[:, j] > 0) & valid
        ht = (them.toarray()[:, j] > 0) & valid
        n = int(hu.sum() + ht.sum())
        if n == 0:
            say(f"E {r['dc']} h={h}: held in no position of T with the horizon inside the game")
            continue
        concept_net = net_mover[hu].sum(0) - net_mover[ht].sum(0)           # the holder is the other side where only they hold it: its net is the mirror
        base_net = base_mover * (int(hu.sum()) - int(ht.sum()))
        corr = (concept_net - base_net) / n
        fmt = lambda v: ", ".join(f"{k} {x:+.3f}" for k, x in zip(kinds, v))
        say(f"E {r['dc']} h={h}: held for the side to move in {int(hu.sum())} positions and for the other side in {int(ht.sum())} (T); kinds lost, theirs less ours, per instance where it holds: {fmt(concept_net / n)}; "
            f"the same over all quiet positions with the same side to move: {fmt(base_net / n)}; CORRECTION per instance: {fmt(corr)}")
    say(f"E done in {time.perf_counter() - t7:.1f} s")
    say(f"total wall {time.perf_counter() - t0:.1f} s")
    (out / "obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
