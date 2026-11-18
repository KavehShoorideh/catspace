"""PLAN 78.81 (SYSTEM_SPEC 7 "subgoals as symbols that go with winning" and "reachable has two layers", 9 the score, 10 the endings risk): THE STAGES that make the symbols of plan 78.80 into fewer, cleaner reasons, fill the thin
endings for test one, and read layer one a first time. A sandbox: no player is changed; nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/symbols_78_81/<tag>/. Game-free: the game
is named only through the pieces that exist (ladder_measure, human_heads_data, graph_layer1/2, subgoal_goals). The 78.80 outputs (sample, events, keys, head rows, the events picked on FIND) are READ, never rebuilt: the sample, the split
and the event matrix are the ones of 78.80. The ladder's accepted moves are read in ONE place, `score_ladder_reasons` (the same wall as 78.68 and 78.80).

INTERFACE (what each stage takes and writes; who calls it). Each stage is a command and prints OBS lines (also to obs.txt) and saves its rows. Inputs: symbols_78_80/full (sample.jsonl, events.npz, keys.json, picked_find.pkl,
passing.json, head.npz, head_us.npz, head_them.npz); the games of the 78.59 file; events_78_64/fill/fill_games.json (the games events_fill.py `generate` played: generated, never a human game continued).
  controls   (B)  per tied move of the sample: whether the move takes a unit (the board's own difference across the move: the replay of subgoal_goals.Facts, which names no move part) and the kind word of the unit that moved
                  (a category). Writes controls.npz. Prints the shares.
  choice     (A, B) the choice rows of the FIND part (choice_rows: human FIND positions only), the events picked on FIND (78.80) as columns, symbols.drop_duplicates, then symbols.fit_choice three times: controls alone, events alone,
                  both; the reasons are the events with a weight in the fit with both. Writes fits.pkl, reasons.json. Prints the counts, the penalties (chosen by games inside FIND) and the held-out log likelihoods.
  fill       (C)  prints the coverage of the sample and of the head rows by units left; builds head rows from the generated games at eight units or fewer (marked generated; ratings and ply at the centre of the human FIND rows),
                  and reruns test one (symbols.term_table) for the patterns of the reasons with and without them; prints which terms change sign or lose their band. They enter test one only: no choice fit reads them.
  ahead      (D)  layer one on CONFIRM for the strongest mover-made-true reasons: symbols.reach_rows over the mover's earlier own positions (k = 1, 2, 3 own moves back), symbols.one_short and symbols.nearer over the tie set (the
                  search at the wake's clock) at those positions. Writes ahead.json. Prints the tables per k with intervals and how often the limit is hit.
  score      (E)  agreement with the move played on CONFIRM, deep_heads.score on the ladder's scored part and hit_rate, each for the new pick against 78.80's pick and the search's first move, paired, with bands by units left
                  and by taking. CONFIRM is read here a SECOND time (78.80 read it once): the penalty and the reasons were chosen on FIND only. Prints the largest share one reason has of a pick.
  report          the reasons in numbers (side, direction, weight, taking share of the carriers, the mover kinds that carry it).
  all             the stages in order.
Called by: the command line only; ladder_print.py reads reasons.json, ahead.json and the sample for the cases.

  .venv/bin/python -m catspace.incubator.analyst.discovery.symbols_reasons_measure choice|... --tag short --limit 300 | --tag full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import os
import pickle
import time
from collections import Counter as Tally
from pathlib import Path

import numpy as np
from scipy import sparse

from catspace.incubator.analyst.discovery import deep_heads as DH
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2 as L2
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import symbols as SY
from catspace.incubator.analyst.discovery import symbols_measure as SM
from catspace.io import paths

OUT = ("symbols_78_81",)
SRC = ("symbols_78_80", "full")                                   # the 78.80 run this one reads
FILL = ("events_78_64", "fill", "fill_games.json")                # the games events_fill.py `generate` played
SEED = 8181
TOP = 20                                                          # the strongest reasons read by the cases and by layer one
AHEAD_LIMIT = 3                                                   # own moves back (a cost guard: the table says how often it is hit)
NEAR_CAP = 3000                                                   # earlier positions read for `nearer` at most (a cost guard: the table says how many were read)
FILL_UNITS = 8                                                    # the bands filled: eight units or fewer
FILL_EVERY = 2                                                    # a head row from every this-many-th ply of a generated game
DRAWS = 100
LEVELS = SM.LEVELS


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def src_root() -> Path:
    return Path(paths.experiment(*SRC))


class Say(SM.Say):
    def __call__(self, msg: str) -> None:
        line = f"OBS reasons {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


# ------------------------------------------------------------------------------------------------------------------------------ the inputs

def load_inputs(limit: int = 0) -> dict:
    """The 78.80 files: the sample (file order), the event matrix (a row per tied move), the keys, the events picked on FIND and those that passed. `limit` keeps the first positions only (a short run)."""
    r = src_root()
    sample = SM.jsonl(r / "sample.jsonl")
    M = sparse.load_npz(r / "events.npz").tocsr()
    keys = json.loads((r / "keys.json").read_text())
    picked = pickle.load(open(r / "picked_find.pkl", "rb"))
    passing = json.loads((r / "passing.json").read_text())
    if limit:
        sample = sample[:limit]
        M = M[:sum(len(x["tied"]) for x in sample)]
    return {"sample": sample, "M": M, "keys": keys, "picked": picked, "passing": passing, "n": len(keys)}


def load_controls(root: Path, limit: int = 0) -> dict:
    z = np.load(root / "controls.npz", allow_pickle=False)
    kinds = json.loads((root / "controls_kinds.json").read_text())
    return {"takes": z["takes"].astype(float), "kind": z["kind"].astype(int), "kinds": kinds}


# ------------------------------------------------------------------------------------------------------------------------------ (B) the controls

_FACTS = None


def _controls_job(chunk: list) -> list:
    global _FACTS
    import cs_engine as CE
    from catspace.incubator.analyst.discovery import subgoal_goals as SG
    if _FACTS is None:
        _FACTS = SG.Facts(HD.VARIANT)
    out = []
    for moves, tied in chunk:
        text = CE.state(HD.VARIANT, None, moves)[0]
        for m in tied:
            pl = _FACTS.replay(text, [m])
            out.append((pl[0]["took"] is not None, pl[0]["kind"] or "") if pl else (False, ""))
    return out


def stage_controls(root: Path, workers: int, limit: int, say) -> None:
    inp = load_inputs(limit)
    sample = inp["sample"]
    chunks = [[(r["moves"], r["tied"]) for r in sample[i:i + 100]] for i in range(0, len(sample), 100)]
    t0 = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_controls_job, chunks, chunksize=1)
    flat = [x for p in parts for x in p]
    kinds = sorted({k for _t, k in flat})
    takes = np.array([t for t, _k in flat], np.uint8)
    kind = np.array([kinds.index(k) for _t, k in flat], np.int16)
    np.savez_compressed(root / "controls.npz", takes=takes, kind=kind)
    (root / "controls_kinds.json").write_text(json.dumps(kinds))
    played = np.array([m == r["played"] for r in sample for m in r["tied"]])
    say(f"CONTROLS for {len(sample)} positions, {len(flat)} tied moves ({time.perf_counter() - t0:.0f} s): the move takes a unit in {100 * takes[played].mean():.1f} pct of the played moves, {100 * takes[~played].mean():.1f} pct of the other tied moves; "
        f"the kind of unit moved (a category of {len(kinds)}): " + ", ".join(f"{k or 'none'} {int((kind == i).sum())}" for i, k in enumerate(kinds)))


# ------------------------------------------------------------------------------------------------------------------------------ (A) the choice rows and the fits

def choice_rows(sample: list, M, cols: list, ctl: dict, part: str) -> SY.Choice:
    """The choice rows of the positions of one part of the sample, every one of them a human game's position (the sample holds human games only; a generated game never comes through here): the events in `cols` (columns of
    the event matrix M), the controls (takes a unit; one column for each kind of unit moved), the game of each row. Row order = the sample's order."""
    sizes = np.array([len(r["tied"]) for r in sample])
    start = np.concatenate([[0], np.cumsum(sizes)])
    keep = [i for i, r in enumerate(sample) if r["part"] == part]
    rows = np.concatenate([np.arange(start[i], start[i + 1]) for i in keep]) if keep else np.zeros(0, int)
    E = M.tocsr()[rows][:, cols] if len(rows) else sparse.csr_matrix((0, len(cols)))
    kinds = ctl["kinds"]
    C = np.zeros((len(rows), 1 + len(kinds)))
    C[:, 0] = ctl["takes"][rows]
    C[np.arange(len(rows)), 1 + ctl["kind"][rows]] = 1.0
    pos = np.repeat(np.arange(len(keep)), sizes[keep]) if keep else np.zeros(0, int)
    played = np.array([m == sample[i]["played"] for i in keep for m in sample[i]["tied"]], bool)
    game = np.array([sample[i]["game"] for i in keep for _m in sample[i]["tied"]])
    return SY.Choice(E, C, ["takes"] + [f"moves:{k or 'none'}" for k in kinds], pos, played, game, np.array([part] * len(rows)), np.array(["human"] * len(rows)))


def meta_of(keys: list, n: int, col_event: int) -> dict:
    col, side, d = SY.event_of(col_event, n)
    k = keys[col]
    return {"column": col, "side": side, "dir": d, "event": col_event, "key": k["key"], "id": k["id"], "size": k["size"], "level": k["level"], "words": k["words"], "text": k["text"], "bytes": k["bytes"]}


def uniform_loglik(rows: SY.Choice) -> float:
    sizes = np.bincount(rows.pos)
    return float(-np.mean(np.log(sizes)))


def stage_choice(root: Path, workers: int, limit: int, say) -> None:
    inp = load_inputs(limit)
    ctl = load_controls(root, limit)
    n = inp["n"]
    events = [SY.event_index(p["column"], p["side"], p["dir"], n) for p in inp["picked"]]
    rows = choice_rows(inp["sample"], inp["M"], events, ctl, "find")
    say(f"THE CHOICE ROWS (FIND, human games only): {rows.n_positions()} positions, {len(rows.pos)} tied moves, {len(events)} events picked on FIND by 78.80's rule as the columns; controls: {', '.join(rows.control_names)}")
    order = [(inp["keys"][SY.event_of(e, n)[0]]["size"], inp["keys"][SY.event_of(e, n)[0]]["level"], inp["keys"][SY.event_of(e, n)[0]]["key"]) for e in events]
    kept, info = SY.drop_duplicates(rows, order, SY.DUP_THRESHOLD)
    say(f"THE DUPLICATE DROP (carrier sets on the FIND rows, Jaccard at or above {SY.DUP_THRESHOLD}, the smaller pattern kept): events in {info['in']}, with no carrier {info['empty']}, dropped as exact copies {info['exact']}, as near copies {info['near']}, after the drop {info['kept']}")
    t0 = time.perf_counter()
    names = tuple(rows.control_names)
    f_ctl = SY.fit_choice(rows, controls=names, events=[], seed=SEED)
    f_ev = SY.fit_choice(rows, controls=(), events=kept, seed=SEED)
    f_both = SY.fit_choice(rows, controls=names, events=kept, seed=SEED)
    uni = uniform_loglik(rows)
    for label, f in (("controls alone", f_ctl), ("events alone", f_ev), ("both", f_both)):
        cv = f["cv"][f["pick"]] if f.get("cv") else None
        say(f"FIT {label}: events offered {f['n_events']}, with a nonzero weight {len(f['events'])}; penalty {f['lambda']:.5f}" + (f" (the path's best {f['cv'][f['best']][0]:.5f})" if cv else "") + f"; log likelihood a position on all FIND rows {f.get('loglik', float('nan')):.4f}"
            + (f", held out inside FIND at the chosen penalty {cv[1]:.4f} (standard error {cv[2]:.4f})" if cv else "") + f"; the uniform pick over the tie {uni:.4f}")
        if f["controls"]:
            say(f"FIT {label}: controls " + ", ".join(f"{k} {v:+.3f}" for k, v in f["controls"].items()))
    wb = dict(zip(f_both["events"], f_both["weights"]))
    we = dict(zip(f_ev["events"], f_ev["weights"]))
    both_only = sum(1 for e in wb if e not in we)
    say(f"REASONS: events in {len(events)}, after the duplicate drop {len(kept)}, with a nonzero weight in the fit with events alone {len(we)}, in the fit with controls and events {len(wb)} (the reasons kept: a weight in the fit with both); "
        f"of those with a weight in events alone {len(wb) - both_only}; sign differs between the two fits for {sum(1 for e in wb if e in we and np.sign(we[e]) != np.sign(wb[e]))}; {time.perf_counter() - t0:.0f} s")
    out = []
    for j, w in wb.items():
        d = meta_of(inp["keys"], n, events[j])
        out.append(dict(d, weight=float(w), weight_events_alone=float(we[j]) if j in we else None))
    out.sort(key=lambda d: -abs(d["weight"]))
    (root / "reasons.json").write_text(json.dumps(out))
    pickle.dump({"ctl": f_ctl, "events": f_ev, "both": f_both, "kept": kept, "event_cols": events, "info": info, "control_names": list(names)}, open(root / "fits.pkl", "wb"))
    for k, d in enumerate(out[:TOP], 1):
        say(f"REASON {k}: {d['id'][:8]} ({d['size']},{d['level']}) {d['side']} {d['dir']}: weight {d['weight']:+.3f} (events alone " + (f"{d['weight_events_alone']:+.3f}" if d["weight_events_alone"] is not None else "none") + f")")


# ------------------------------------------------------------------------------------------------------------------------------ (C) the endings

def units_by_row(model, games_moves: list, rows_of: list, starts: "list | None" = None) -> list:
    """Units on the board after each wanted ply of each game: [[units]] (the table's counts, goal units included)."""
    wires = [((starts[i] if starts else None), m) for i, m in enumerate(games_moves)]
    names, raw = model.position_table(wires, max(1, (os.cpu_count() or 2) - 1))
    w = 2 + 2 * len(names)
    flat = np.frombuffer(raw, dtype=np.int16)
    out, off = [], 0
    for i, m in enumerate(games_moves):
        t = flat[off:off + (len(m) + 1) * w].reshape(len(m) + 1, w).astype(np.int32)
        off += (len(m) + 1) * w
        out.append([int(t[p, 2:].sum()) for p in rows_of[i]])
    return out


def fill_head_rows(model, head: dict, tf: dict, say) -> dict:
    """Head rows from the generated games (events_fill's file): every FILL_EVERY-th ply from the third on (after the two random plies) where the position is open and at most FILL_UNITS units are left. The ratings and the
    ply are set at the centre of the human FIND rows (a generated game has none), so only the kinds speak; the result is the first mover's, turned to the side to move as 78.59's rows are."""
    games = json.loads(Path(paths.experiment(*FILL)).read_text())
    games = [g for g in games if not g.get("unfinished")]
    names, raw = model.position_table([(g["start"], g["moves"]) for g in games], max(1, (os.cpu_count() or 2) - 1))
    nk = len(names)
    goal = G2M.goal_kind_of(HD.VARIANT)
    use = [k for k, nm in enumerate(names) if nm != goal]
    flat = np.frombuffer(raw, dtype=np.int16)
    w = 2 + 2 * nk
    off = 0
    gi, ply, y, kd, wires, rows = [], [], [], [], [], []
    for i, g in enumerate(games):
        n = len(g["moves"]) + 1
        t = flat[off:off + n * w].reshape(n, w).astype(np.int32)
        off += n * w
        take = []
        for p in range(2, n, FILL_EVERY):
            if t[p, 0] != 1 or t[p, 1] != 1 or int(t[p, 2:].sum()) > FILL_UNITS:
                continue
            take.append(p)
            gi.append(i)
            ply.append(p)
            y.append(g["y"] if p % 2 == 0 else 2 - g["y"])
            kd.append((t[p, 2:2 + nk] - t[p, 2 + nk:2 + 2 * nk])[use])
        if take:
            wires.append((g["start"], g["moves"]))
            rows.append(take)
    say(f"FILL ROWS from {len(games)} generated games (events_fill's file; the player of 78.61's generation 2 from positions of thin small sets, finished games only): {len(gi)} head rows at {FILL_UNITS} units or fewer, "
        f"from {len(set(gi))} games; marked generated")
    x = np.hstack([np.zeros((len(gi), 3)), np.array(kd, float)])                     # R at the centre (standardised zero), the kinds as the human rows have them
    return {"game": np.array(gi), "ply": np.array(ply), "y": np.array(y, int), "x": x, "wires": wires, "rows": rows}


def stage_fill(root: Path, workers: int, limit: int, say) -> None:
    inp = load_inputs(limit)
    sample = inp["sample"]
    model = GL.model(HD.GAME, HD.VARIANT)
    goal = G2M.goal_kind_of(HD.VARIANT)
    # coverage: the kept positions of the sample and the head rows of the held-out games, by units left
    bins = [SM.bin_of(r["units"]) for r in sample]
    head, us, them = SM.load_head(src_root())
    games = {g["ord"]: g for g in SM.held_out(SM.all_games(), SM.ladder_ordinals())}
    ords = sorted({int(o) for o in head["game"]})
    rows_of = [[int(p) for p in head["ply"][head["game"] == o]] for o in ords]
    unit_rows = units_by_row(model, [games[o]["moves"] for o in ords], rows_of)
    hb = Tally(SM.bin_of(u) for us_ in unit_rows for u in us_)
    sb = Tally(bins)
    order = [SM.bin_of(a) for a, _h in SM.BINS]
    say("ENDGAME COVERAGE by units left (the sample's kept tie positions / the head rows of the held-out games): " + ", ".join(f"{b}: {sb[b]} / {hb[b]}" for b in order)
        + f"; at {FILL_UNITS} units or fewer: {sum(1 for r in sample if r['units'] <= FILL_UNITS)} of {len(sample)} kept positions, {sum(hb[b] for b in order[:2])} of {sum(hb.values())} head rows")
    reasons = json.loads((root / "reasons.json").read_text())
    if not reasons:
        say("no reasons to rerun test one for; stage ends")
        return
    cols = sorted({r["column"] for r in reasons})
    find = head["find"]
    m, tf = SM.head_fit(head, find, say)
    xh, _t = SM.head_inputs(head, find)
    off_h = SM.head_scores(m, xh)
    fidx = np.flatnonzero(find)
    fill = fill_head_rows(model, head, tf, say)
    counter = SY.Counter(model, [int(inp["keys"][c]["key"]) for c in cols], goal, 2, LEVELS, workers)
    fus, fth = counter.count(fill["wires"], fill["rows"])
    off_f = SM.head_scores(m, fill["x"])
    usf, themf = us[fidx][:, cols], them[fidx][:, cols]
    gh = head["game"][fidx]
    base = SY.term_table(off_h[fidx], head["y"][fidx].astype(int), gh, usf, themf, list(range(len(cols))), DRAWS, SEED)
    allus = sparse.vstack([usf, fus.tocsr()], format="csr")
    allth = sparse.vstack([themf, fth.tocsr()], format="csr")
    game_all = np.concatenate([gh, 10 ** 7 + fill["game"]])                           # a generated game is its own cluster, apart from every human game's ordinal
    is_gen = np.concatenate([np.zeros(len(fidx)), np.ones(len(fill["game"]))])[:, None]                # a shift of its own for the generated rows: the head's scores are not calibrated for them
    with_fill = SY.term_table(np.vstack([off_h[fidx], off_f]), np.concatenate([head["y"][fidx].astype(int), fill["y"]]), game_all, allus, allth, list(range(len(cols))), DRAWS, SEED, free=is_gen)
    flip = lost = 0
    detail = []
    cidx = {c: j for j, c in enumerate(cols)}
    for k, r in enumerate(reasons):
        j = cidx[r["column"]]
        s = SY.SIDES.index(r["side"])
        a, b = base[j], with_fill[j]
        sa, sb_ = np.sign(a["term"][s]), np.sign(b["term"][s])
        ba = a["lo"][s] > 0 or a["hi"][s] < 0
        bb = b["lo"][s] > 0 or b["hi"][s] < 0
        flip += int(sa != sb_)
        lost += int(ba and not bb)
        detail.append((k, r, a["term"][s], b["term"][s], ba, bb, a["rows"], b["rows"]))
    say(f"TEST ONE with the generated rows (head rows: {len(fidx)} human FIND + {len(fill['game'])} generated, a class shift of their own; {DRAWS} resamples, a generated game a cluster of its own) for the {len(reasons)} reasons' patterns: "
        f"terms that change sign {flip}, bands that held zero out and now hold it {lost}, bands that did not exclude zero and now do {sum(1 for d in detail if not d[4] and d[5])}")
    for k, r, ta, tb, ba, bb, na, nb in detail:
        if np.sign(ta) != np.sign(tb) or (ba and not bb) or k < 10:
            say(f"TEST ONE reason {k + 1} ({r['id'][:8]} {r['side']} {r['dir']}): term {ta:+.3f} (band excludes zero: {ba}, rows {na}) -> {tb:+.3f} (band excludes zero: {bb}, rows {nb})" + ("  SIGN CHANGES" if np.sign(ta) != np.sign(tb) else "") + ("  BAND LOST" if ba and not bb else ""))
    pickle.dump({"base": base, "with_fill": with_fill, "cols": cols}, open(root / "fill_test_one.pkl", "wb"))


# ------------------------------------------------------------------------------------------------------------------------------ (D) layer one

_ENV: dict = {}


def _worker_env() -> dict:
    if not _ENV:
        from catspace.incubator.analyst.discovery import subgoal_goals as SG
        model = GL.model(HD.GAME, HD.VARIANT)
        kinds = [n[len("kind:"):] for ty, n in model.static_part()["nodes"] if ty == "kind"]
        _ENV["env"] = SY.Reach(model, HD.VARIANT, G2M.goal_kind_of(HD.VARIANT), kinds, SG.Facts(HD.VARIANT))
        _ENV["tools"] = LM.Tools()
    return _ENV


def _near_job(job: tuple) -> dict:
    """One earlier position: the tie set there (the search at the wake's clock) and, for each wanted pattern, whether the move played there and each other tied move makes some instance one fact short."""
    moves, played, wanted = job
    e = _worker_env()
    env = e["env"]
    s = e["tools"].search({"moves": moves}, LM.LIMITS["wake_seconds"])
    tied = list(s["equals"])
    moves_read = [played] + [m for m in tied if m != played]
    text = env.text(moves)
    out = {"tied": len(tied), "played_in_tie": played in tied, "near": {}}
    for ridx, raw in wanted:
        pat = L2.decode(bytes.fromhex(raw))
        items, units = env.items(text)
        before = SY.one_short(items, units, pat, env.kinds, 0)
        out["near"][ridx] = [SY.nearer(env, text, m, pat, before) for m in moves_read]
    return out


def wilson_pair(k: int, n: int) -> str:
    lo, hi = DH.wilson(k, n)
    return f"{100 * k / max(1, n):.1f} pct [{100 * lo:.1f}, {100 * hi:.1f}] ({k} of {n})"


def stage_ahead(root: Path, workers: int, limit: int, say) -> None:
    inp = load_inputs(limit)
    sample = inp["sample"]
    reasons = json.loads((root / "reasons.json").read_text())
    n_all = len(reasons)
    top_all = reasons[:TOP]
    up = [r for r in reasons if r["side"] == "mover" and r["dir"] == "up"]
    use = up[:TOP]
    say(f"LAYER ONE reads reasons made true for the mover only (a reason made false, or for the other side, has no walk back defined here): of the {len(top_all)} strongest reasons {sum(1 for r in top_all if r['side'] == 'mover' and r['dir'] == 'up')} are of that kind; "
        f"of the {n_all} reasons {len(up)}; the {len(use)} strongest of those are read; by size {dict(Tally(r['size'] for r in use))}")
    if not use:
        say("no reason to read; stage ends")
        return
    model = GL.model(HD.GAME, HD.VARIANT)
    goal = G2M.goal_kind_of(HD.VARIANT)
    kinds = [n[len("kind:"):] for ty, n in model.static_part()["nodes"] if ty == "kind"]
    from catspace.incubator.analyst.discovery import subgoal_goals as SG
    env = SY.Reach(model, HD.VARIANT, goal, kinds, SG.Facts(HD.VARIANT))
    counter = SY.Counter(model, [int(r["key"]) for r in use], goal, 2, LEVELS, workers)
    n = inp["n"]
    sizes = np.array([len(r["tied"]) for r in sample])
    start = np.concatenate([[0], np.cumsum(sizes)])
    M = inp["M"].tocsc()
    conf = [i for i, r in enumerate(sample) if r["part"] == "confirm"]
    rng = np.random.default_rng(SEED)
    cases, ctrl = {}, {}
    for j, r in enumerate(use):
        col = M[:, SY.event_index(r["column"], r["side"], r["dir"], n)]
        hit = set(col.indices.tolist())
        yes = [i for i in conf if start[i] + sample[i]["tied"].index(sample[i]["played"]) in hit]
        no = [i for i in conf if i not in set(yes)]
        no = sorted(rng.choice(no, min(len(no), len(yes)), replace=False).tolist()) if no else []
        cases[j], ctrl[j] = yes, no
    say(f"CASES on CONFIRM (the move played makes the reason true): per reason {', '.join(str(len(cases[j])) for j in cases)}; the same number of positions where it does not, drawn at random, as the control")
    # the earlier positions: one table for all of them
    need: dict = {}

    def add(i: int, k: int) -> "tuple | None":
        r = sample[i]
        moves = r["moves"]
        at = len(moves) - 2 * k
        if at < 0:
            return None
        key = (r["game"], at)
        if key not in need:
            need[key] = {"moves": moves[:at], "played": moves[at], "i": i}
        return key
    for j in cases:
        for i in cases[j] + ctrl[j]:
            for k in range(1, AHEAD_LIMIT + 1):
                add(i, k)
    keys_list = sorted(need)
    t0 = time.perf_counter()
    reach = {}
    for a in range(0, len(keys_list), 300):
        chunk = keys_list[a:a + 300]
        res = SY.reach_rows(env, [{"moves": need[k]["moves"]} for k in chunk], counter)
        for k, rr in zip(chunk, res):
            reach[k] = rr
    say(f"REACH read at {len(keys_list)} earlier own positions (every legal move; {time.perf_counter() - t0:.0f} s, load average {os.getloadavg()[0]:.2f})")
    # the table per k: reachable at p_k (cases vs control), the run length ('ahead')
    tab = {}
    runs = {j: [] for j in cases}
    runs_ctrl = {j: [] for j in cases}
    per = {j: {"case": [0, 0, 0, 0, 0, 0], "ctrl": [0, 0, 0, 0, 0, 0]} for j in cases}                # per reason, per tag: [reachable, total] at k = 1, 2, 3
    for j in cases:
        r = use[j]
        for tag, lst in (("case", cases[j]), ("ctrl", ctrl[j])):
            for i in lst:
                run = 0
                alive = True
                for k in range(1, AHEAD_LIMIT + 1):
                    key = add(i, k)
                    if key is None:
                        break
                    ok = bool(reach[key]["makes"].get(int(r["key"])))
                    c = tab.setdefault((tag, k), [0, 0])
                    c[1] += 1
                    c[0] += ok
                    per[j][tag][2 * (k - 1)] += ok
                    per[j][tag][2 * (k - 1) + 1] += 1
                    if alive and ok:
                        run += 1
                    else:
                        alive = False
                (runs if tag == "case" else runs_ctrl)[j].append(run)
    for k in range(1, AHEAD_LIMIT + 1):
        a, b = tab.get(("case", k), [0, 0]), tab.get(("ctrl", k), [0, 0])
        say(f"REACHABLE in exactly one own move, {k} own move(s) earlier (pooled over the {len(use)} reasons): where the move played makes the reason true {wilson_pair(a[0], a[1])}; where it does not (control) {wilson_pair(b[0], b[1])}")
    for tag, rr in (("cases (the move played makes the reason true)", runs), ("the control positions", runs_ctrl)):
        allruns = [x for j in rr for x in rr[j]]
        hit_limit = allruns.count(AHEAD_LIMIT)
        say(f"AHEAD for {tag} (the unbroken run of earlier own positions where the reason was one move away, limit {AHEAD_LIMIT}): over {len(allruns)} positions run 0: {allruns.count(0)}, 1: {allruns.count(1)}, 2: {allruns.count(2)}, "
            f"the limit hit: {hit_limit} ({100 * hit_limit / max(1, len(allruns)):.1f} pct)")
    for j in cases:
        pc, pt = per[j]["case"], per[j]["ctrl"]
        say(f"REACHABLE reason {j + 1} ({use[j]['id'][:8]}, size {use[j]['size']}, weight {use[j]['weight']:+.3f}): cases {len(cases[j])}; reachable at k=1,2,3 where the move played makes it true "
            + ", ".join(f"{100 * pc[2 * q] / max(1, pc[2 * q + 1]):.0f}" for q in range(3)) + " pct; control " + ", ".join(f"{100 * pt[2 * q] / max(1, pt[2 * q + 1]):.0f}" for q in range(3)) + " pct")
    # nearer: size-2 reasons, reachable cases, the tie set at the earlier position
    jobs, keyrows = {}, {}
    for j in cases:
        if use[j]["size"] != 2:
            continue
        for i in cases[j]:
            for k in range(1, AHEAD_LIMIT + 1):
                key = add(i, k)
                if key is None:
                    break
                if reach[key]["makes"].get(int(use[j]["key"])):
                    jobs.setdefault(key, set()).add(j)
                    keyrows.setdefault((j, k), []).append((i, key))
    keys_near = sorted(jobs)
    capped = len(keys_near) > NEAR_CAP
    if capped:
        keys_near = [keys_near[x] for x in sorted(rng.choice(len(keys_near), NEAR_CAP, replace=False))]
    say(f"NEARER: earlier positions to read {len(jobs)}" + (f", cost guard: {len(keys_near)} of them drawn at random" if capped else "") + "; the other moves are the tie set of the search at the wake's clock there (the played move counted apart)")
    t1 = time.perf_counter()
    near: dict = {}
    if keys_near:
        todo = [(need[k]["moves"], need[k]["played"], [(j, use[j]["bytes"]) for j in sorted(jobs[k])]) for k in keys_near]
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for k, res in zip(keys_near, pool.imap(_near_job, todo, chunksize=4)):
                near[k] = res
    say(f"NEARER read in {time.perf_counter() - t1:.0f} s, load average {os.getloadavg()[0]:.2f}")
    out = {"reasons": [{"column": r["column"], "id": r["id"], "key": r["key"], "size": r["size"]} for r in use], "tab": {f"{a}:{k}": v for (a, k), v in tab.items()}, "runs": runs}
    for k in range(1, AHEAD_LIMIT + 1):
        played_n, played_k, other_share, games = 0, 0, [], []
        for j in cases:
            for i, key in keyrows.get((j, k), []):
                res = near.get(key)
                if res is None or not res["near"].get(j):
                    continue
                v = res["near"][j]
                if v[0] is None:
                    continue
                oth = [x for x in v[1:] if x is not None]
                if not oth:
                    continue
                played_n += 1
                played_k += bool(v[0])
                other_share.append(float(np.mean(oth)))
                games.append((sample[i]["game"], float(bool(v[0]))))
        if played_n:
            gm = np.array([g for g, _x in games])
            m, lo, hi = SM.paired(np.array([x for _g, x in games]), np.array(other_share), gm, SEED + k)
            say(f"NEARER {k} own move(s) earlier, where the reason was one move away ({played_n} cases of the size-2 reasons): the move played made some instance one fact short {wilson_pair(played_k, played_n)}; the other tied moves, mean share {100 * np.mean(other_share):.1f} pct; "
                f"paired difference {100 * m:+.1f} points [{100 * lo:+.1f}, {100 * hi:+.1f}] (90 pct bootstrap over games)")
            out[f"near_{k}"] = {"n": played_n, "played": played_k, "other": float(np.mean(other_share)), "diff": [m, lo, hi]}
        else:
            say(f"NEARER {k} own move(s) earlier: no case read")
    (root / "ahead.json").write_text(json.dumps(out))


# ------------------------------------------------------------------------------------------------------------------------------ (E) the scores

def choice_of(rows: SY.Choice, weights: dict, controls: "dict | None" = None) -> dict:
    """A fit dict over the columns of rows.E from {column: weight} (and, optionally, the controls' weights)."""
    cols = sorted(weights)
    return {"events": cols, "weights": [weights[c] for c in cols], "controls": controls or {}}


def agree(rows: SY.Choice, fit: dict, controls: bool = False) -> np.ndarray:
    """1 where the pick of the fit is the move played (per position)."""
    pick = SY.score_choice(rows, fit, controls)
    sizes = np.bincount(rows.pos)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    played_at = np.zeros(rows.n_positions(), int)
    pr = np.flatnonzero(rows.played)
    played_at[rows.pos[pr]] = pr - start[rows.pos[pr]]
    return (pick == played_at).astype(float)


def loglik_per_position(rows: SY.Choice, fit: dict, controls: bool) -> float:
    s = SY.choice_scores(rows, fit, controls)
    P = rows.n_positions()
    sizes = np.bincount(rows.pos)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    tot = 0.0
    for p in range(P):
        seg = s[start[p]:start[p] + sizes[p]]
        m = seg.max()
        lse = m + np.log(np.exp(seg - m).sum())
        tot += float(s[start[p] + int(np.flatnonzero(rows.played[start[p]:start[p] + sizes[p]])[0])] - lse)
    return tot / max(1, P)


def share_stats(rows: SY.Choice, fit: dict) -> dict:
    """The largest share one reason has of a pick's score: for each position where the pick is carried by at least one weighted event (not silent), the largest |weight| of one event the picked move carries over the sum of
    the |weights| of the events it carries; the median, the maximum and the count of positions above one half."""
    pick = SY.score_choice(rows, fit)
    sizes = np.bincount(rows.pos)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    E = rows.E.tocsr()[:, fit["events"]] if fit["events"] else None
    w = np.abs(np.asarray(fit["weights"], float))
    best = []
    for p in range(rows.n_positions()):
        if E is None:
            break
        r = start[p] + pick[p]
        cols = E.indices[E.indptr[r]:E.indptr[r + 1]]
        if len(cols) and w[cols].sum() > 0:
            best.append(float(w[cols].max() / w[cols].sum()))
    if not best:
        return {"n": 0}
    b = np.array(best)
    return {"n": len(b), "median": float(np.median(b)), "max": float(b.max()), "over_half": int((b > 0.5).sum()), "mean": float(b.mean())}


def band_of_units(u: int) -> str:
    return LM.band_of(u)


def score_confirm_reasons(root: Path, inp: dict, ctl: dict, say) -> dict:
    fits = pickle.load(open(root / "fits.pkl", "rb"))
    reasons = json.loads((root / "reasons.json").read_text())
    n = inp["n"]
    events = fits["event_cols"]
    # the columns of the event matrix: the reasons', the events-alone fit's and 78.80's passing events
    p80 = {SY.event_index(d["column"], d["side"], d["dir"], n): d["weight"] for d in inp["passing"]}
    union = sorted(set(events[j] for j in fits["events"]["events"]) | set(events[j] for j in fits["both"]["events"]) | set(p80))
    ci = {e: i for i, e in enumerate(union)}
    rows = choice_rows(inp["sample"], inp["M"], union, ctl, "confirm")
    sample_c = [r for r in inp["sample"] if r["part"] == "confirm"]
    P = rows.n_positions()

    def fit_of(f: dict) -> dict:
        return {"events": [ci[events[j]] for j in f["events"]], "weights": list(f["weights"]), "controls": f["controls"]}
    f_new, f_ev, f_ctl = fit_of(fits["both"]), fit_of(fits["events"]), fits["ctl"]
    f_80 = {"events": [ci[e] for e in sorted(p80)], "weights": [p80[e] for e in sorted(p80)], "controls": {}}
    played_first = np.array([r["tied"][0] == r["played"] for r in sample_c], float)
    rnd = np.array([1.0 / len(r["tied"]) for r in sample_c])
    hits = {"new": agree(rows, f_new), "new with the controls in the pick": agree(rows, f_new, True), "events alone": agree(rows, f_ev), "controls alone": agree(rows, f_ctl, True), "78.80 pick": agree(rows, f_80),
            "search first": played_first, "random": rnd}
    game = np.array([r["game"] for r in sample_c])
    say(f"SCORE 1, agreement with the move played on the CONFIRM part ({P} positions; this is its SECOND reading: 78.80 read it once, and the penalty and the reasons here were chosen on FIND only)")
    for k, v in hits.items():
        if k != "random":
            say(f"SCORE 1 {k}: {wilson_line(v.sum(), P)}")
    say(f"SCORE 1 random pick (expected) {100 * rnd.mean():.1f} pct")
    for a, b in (("new", "search first"), ("new", "78.80 pick"), ("78.80 pick", "search first"), ("new with the controls in the pick", "search first"), ("events alone", "search first"), ("controls alone", "search first")):
        m, lo, hi = SM.paired(hits[a], hits[b], game, SEED + 11)
        say(f"SCORE 1 paired difference, {a} less {b}: {100 * m:+.2f} points [{100 * lo:+.2f}, {100 * hi:+.2f}] (90 pct bootstrap over games)")
    # held-out log likelihood of the choice on CONFIRM for the three fits (not used to choose anything)
    uni = float(-np.mean(np.log(np.bincount(rows.pos))))
    say(f"CHOICE log likelihood a position on CONFIRM: controls alone {loglik_per_position(rows, f_ctl, True):.4f}, events alone {loglik_per_position(rows, f_ev, False):.4f}, both {loglik_per_position(rows, f_new, True):.4f}; uniform over the tie {uni:.4f}")
    # by units left and by taking
    units = np.array([r["units"] for r in sample_c])
    bands = np.array([band_of_units(u) for u in units])
    sizes = np.bincount(rows.pos)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    takes_row = rows.C[:, 0]
    some_take = np.array([takes_row[start[p]:start[p] + sizes[p]].any() for p in range(P)])
    groups = [(f"band {b}", bands == b) for b, _l, _h in LM.BANDS] + [("no tied move takes a unit", ~some_take), ("some tied move takes a unit", some_take)]
    for label, sel in groups:
        m = int(sel.sum())
        if m < 5:
            say(f"SCORE 1 by {label}: {m} positions (too few)")
            continue
        parts = [f"new {wilson_line(hits['new'][sel].sum(), m)}", f"78.80 pick {wilson_line(hits['78.80 pick'][sel].sum(), m)}", f"search first {wilson_line(hits['search first'][sel].sum(), m)}"]
        dfirst = SM.paired(hits["new"][sel], hits["search first"][sel], game[sel], SEED + 12)
        dprev = SM.paired(hits["new"][sel], hits["78.80 pick"][sel], game[sel], SEED + 13)
        say(f"SCORE 1 by {label}: {m} positions; " + "; ".join(parts) + f"; paired new less first {100 * dfirst[0]:+.2f} [{100 * dfirst[1]:+.2f}, {100 * dfirst[2]:+.2f}], new less 78.80 {100 * dprev[0]:+.2f} [{100 * dprev[1]:+.2f}, {100 * dprev[2]:+.2f}]")
    s_new, s_80 = share_stats(rows, f_new), share_stats(rows, f_80)
    say(f"LARGEST SHARE one reason has of a pick's score on CONFIRM (the picked move's carried weights, positions where the pick is carried): new {s_new}; 78.80's passing events {s_80}")
    say(f"SILENCE of the new pick on CONFIRM (no reason separates the tied moves, the search's first move is kept): {int(sum(1 for p in range(P) if _silent(rows, f_new, p, start, sizes)))} of {P}")
    return {"rows": rows, "hits": hits, "game": game}


def _silent(rows: SY.Choice, fit: dict, p: int, start, sizes) -> bool:
    s = SY.choice_scores(rows, fit, False)[start[p]:start[p] + sizes[p]]
    return bool(np.all(s == s[0]))


def wilson_line(k: float, n: int) -> str:
    return SM.wilson_line(k, n)


def score_ladder_reasons(root: Path, workers: int, say) -> None:
    """SCORES 2 AND 3 for the new pick, 78.80's pick and the search's first move: deep_heads.score on the ladder's scored part and hit_rate against the accepted moves there. The one function of this module that reads the
    ladder's answers; no pick has read them."""
    from catspace.incubator.analyst.discovery import subgoal_pick_measure as SPM
    lr = LM.out_root("full")
    positions = {p["id"]: p for p in SM.jsonl(lr / "positions.jsonl")}
    results = {d["id"]: d for d in SM.jsonl(lr / "results.jsonl")}
    open_ids = [r["id"] for r in SM.jsonl(lr / "surprise_list.jsonl")]
    resolved_ids = [i for i, d in results.items() if d["result"] and d["result"]["rung"] in (1, 2)]
    preps = pickle.loads((Path(paths.experiment("subgoal_pick_78_79", "full")) / "preps.pkl").read_bytes())
    reasons = json.loads((root / "reasons.json").read_text())
    keys80 = json.loads((src_root() / "keys.json").read_text())
    passing = json.loads((src_root() / "passing.json").read_text())
    model = GL.model(HD.GAME, HD.VARIANT)
    goal = G2M.goal_kind_of(HD.VARIANT)
    counter80 = SY.Counter(model, [int(k["key"]) for k in keys80], goal, 2, LEVELS, workers)
    pats80 = SM.patterns_of(passing)
    tables80 = SM.ladder_tables(counter80, positions, preps)
    new_cols = sorted({r["column"] for r in reasons})
    counter_new = SY.Counter(model, [int(keys80[c]["key"]) for c in new_cols], goal, 2, LEVELS, workers)
    ci = {c: j for j, c in enumerate(new_cols)}
    pats_new = [{"column": ci[r["column"]], "side": r["side"], "dir": r["dir"], "weight": r["weight"]} for r in reasons]
    tables_new = SM.ladder_tables(counter_new, positions, preps)

    def make(pats, tables, name):
        def pick(pid: str, prep: dict) -> str:
            if len(prep["eq"]) < 2 or prep.get("tied") is None or pid not in tables:
                return prep["move"]
            return SY.symbol_pick({"moves": positions[pid]["moves"]}, prep["tied"], pats, None, tables[pid])
        return {"name": name, "pick": pick, "cost": lambda p: 0.0, "random": False}
    cands = [SPM.candidates()[0], make(pats80, tables80, "78.80's pick: the sum of the weights of the passing events"), make(pats_new, tables_new, "78.81's pick: the sum of the weights of the reasons (the joint fit)")]
    ans = {d["id"]: d["accepted"] for d in SM.jsonl(lr / "answers.jsonl")}
    scored = [i for i in positions if DH.split_part(i) == "scored" and i in preps]
    table = [DH.score(c, preps, open_ids, resolved_ids, ans, "scored") for c in cands]
    hit_ids = [i for i in scored if DH.tie_eligible(preps[i], set(ans[i]))]
    hitv = [np.array([DH.hit_rate(c, preps, [i], ans, "scored")["hits"] for i in hit_ids]) for c in cands]
    for s, hv in zip(table, hitv):
        lo_r, hi_r = DH.wilson(s["resolved"], s["n_open"])
        lo_c, hi_c = DH.wilson(s["created"], s["n_resolved_sample"])
        lo_h, hi_h = DH.wilson(hv.sum(), len(hv))
        say(f"SCORE 2 (STAGE H) {s['name']}: resolved {s['resolved']:.1f} of {s['n_open']} open surprises (share {s['resolved'] / max(1, s['n_open']):.3f} [{lo_r:.3f}, {hi_r:.3f}]); created {s['created']:.1f} of {s['n_resolved_sample']} resolved positions "
            f"(share {s['created'] / max(1, s['n_resolved_sample']):.3f} [{lo_c:.3f}, {hi_c:.3f}]); net {s['resolved'] - s['created']:+.1f}")
        say(f"SCORE 3 (hit against the accepted moves) {s['name']}: {100 * hv.mean():.1f} pct [{100 * lo_h:.1f}, {100 * hi_h:.1f}] ({hv.sum():.1f} of {len(hv)} ties, 90 pct Wilson)")
    rng = np.random.default_rng(SEED)
    ctrl = table[0]
    ia = rng.integers(0, len(ctrl["res_vec"]), (SM.DRAWS_BOOT, len(ctrl["res_vec"])))
    ib = rng.integers(0, max(1, len(ctrl["keep_vec"])), (SM.DRAWS_BOOT, len(ctrl["keep_vec"])))
    ih = rng.integers(0, len(hit_ids), (SM.DRAWS_BOOT, len(hit_ids)))
    for a, b, label in ((1, 0, "78.80's pick less the search's first move"), (2, 0, "the new pick less the search's first move"), (2, 1, "the new pick less 78.80's pick")):
        d_open = np.array(table[a]["res_vec"]) - np.array(table[b]["res_vec"])
        d_keep = np.array(table[a]["keep_vec"]) - np.array(table[b]["keep_vec"])
        diff = d_open[ia].sum(axis=1) + d_keep[ib].sum(axis=1)
        lo, hi = np.percentile(diff, [5, 95])
        dh = hitv[a] - hitv[b]
        hd = dh[ih].mean(axis=1)
        hlo, hhi = np.percentile(hd, [5, 95])
        say(f"SCORE 2 net, {label}: {d_open.sum() + d_keep.sum():+.1f} positions [{lo:+.1f}, {hi:+.1f}] (90 pct bootstrap over positions, paired); SCORE 3 hit, {label}: {100 * dh.mean():+.1f} points [{100 * hlo:+.1f}, {100 * hhi:+.1f}] (paired, 90 pct bootstrap over positions)")
    tied_ids = [i for i in scored if len(preps[i]["eq"]) >= 2 and preps[i].get("tied") and i in tables_new]
    w = {(q["column"], q["side"], q["dir"]): q["weight"] for q in pats_new}
    sil = sum(len({sum(w.get(e, 0.0) for e in tables_new[i][m]) for m in preps[i]["tied"]}) == 1 for i in tied_ids)
    say(f"SILENCE of the new pick on the ladder's scored part: of {len(tied_ids)} tie positions no reason separates the tied moves in {sil} ({100 * sil / max(1, len(tied_ids)):.1f} pct)")


def stage_score(root: Path, workers: int, limit: int, say) -> None:
    inp = load_inputs(limit)
    ctl = load_controls(root, limit)
    score_confirm_reasons(root, inp, ctl, say)
    if not limit:
        score_ladder_reasons(root, workers, say)
    else:
        say("short run: the ladder's scoring is left to the full run")


# ------------------------------------------------------------------------------------------------------------------------------ the report

def stage_report(root: Path, workers: int, limit: int, say) -> None:
    inp = load_inputs(limit)
    ctl = load_controls(root, limit)
    reasons = json.loads((root / "reasons.json").read_text())
    n = inp["n"]
    M = inp["M"].tocsc()
    sample = inp["sample"]
    kinds = ctl["kinds"]
    for k, r in enumerate(reasons[:TOP], 1):
        col = M[:, r["event"]]
        rows = col.indices
        tk = ctl["takes"][rows].mean() if len(rows) else float("nan")
        kc = Tally(kinds[int(x)] or "none" for x in ctl["kind"][rows])
        top = kc.most_common(2)
        say(f"REASON {k}: {r['id'][:8]} ({r['size']},{r['level']}) {r['side']} {r['dir']} weight {r['weight']:+.3f}; carried by {len(rows)} tied moves of the sample; {100 * tk:.1f} pct of them take a unit (all tied moves: {100 * ctl['takes'].mean():.1f} pct); "
            f"most common kind moved: " + ", ".join(f"{a} {100 * b / max(1, len(rows)):.0f} pct" for a, b in top))
    say(f"REASONS by side and direction: " + ", ".join(f"{s} {d}: {sum(1 for r in reasons if r['side'] == s and r['dir'] == d)}" for s in SY.SIDES for d in SY.DIRS) + f"; by size and level: " + ", ".join(f"({a},{b}): {sum(1 for r in reasons if r['size'] == a and r['level'] == b)}" for a in SM.SIZES for b in LEVELS))


# ------------------------------------------------------------------------------------------------------------------------------ main

def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("controls", "choice", "fill", "ahead", "score", "report", "all"))
    ap.add_argument("--tag", default="short")
    ap.add_argument("--limit", type=int, default=0, help="a short run: only this many positions of 78.80's sample are used")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    stages = ["controls", "choice", "fill", "ahead", "score", "report"] if a.stage == "all" else [a.stage]
    for st in stages:
        say = Say(root, st)
        t0 = time.perf_counter()
        say(f"stage start; tag {a.tag}; limit {a.limit}; load average {os.getloadavg()[0]:.2f}")
        {"controls": lambda: stage_controls(root, a.workers, a.limit, say), "choice": lambda: stage_choice(root, a.workers, a.limit, say), "fill": lambda: stage_fill(root, a.workers, a.limit, say),
         "ahead": lambda: stage_ahead(root, a.workers, a.limit, say), "score": lambda: stage_score(root, a.workers, a.limit, say), "report": lambda: stage_report(root, a.workers, a.limit, say)}[st]()
        say(f"stage done in {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
