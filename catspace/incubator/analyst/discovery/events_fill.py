"""PLAN 78.64 E: THE ENDGAME COVERAGE AND THE FILL, owed since 78.62. A measurement in a sandbox; nothing is wired into any player, nothing is written to rules/learned or data/derived. Game-free: the
kinds, sides and units are the engine's words read at run time; none is named here. No engine evaluation: the fill is GAMES PLAYED BY THE 78.61 GENERATION PLAYER (its judge is the odds record of
generation 2, a record of results of games), from positions sampled from the human games; what is learned from them is what happened on the board and how the game ended.

THE STANDING ENDGAME RULE. `coverage` prints, for a sample of human games: the share of positions and of games by units left on the board; the small sets of units (the kinds each side holds, the goal
units not counted, at most SMALL units in all, the two sides in either order) by the positions in them and the games they are in, how many have fewer than THIN games; and how the games ended (the
record's result against whether the last position is one the rules call over: a game stopped before the rules' own end is a resignation or an agreement).

THE FILL. The sets of at most SMALL units that are in fewer than THIN FIT games are thin. SETS of them (a hash order) are sampled; for each, up to PER_SET positions from different fit games (the first
position of the game in the set); from each position the 78.61 player plays GAMES_PER_START games against itself, each after OPENING random plies (the position is the start; seconds a move SECONDS, ply cap MAX_PLIES), whose
moves are kept. A fill game is a FIT game only: it never enters the test rows, so the human test games stay what they were. A game that reaches the ply cap is unfinished; the fill is compared on the classes of
taking only (not on the class of the end of the game, which a fill game that was cut by the cap would misstate). THE COMPARISON: heads of the classes of stage B on the input set K+S+E (the fill games have no ratings and
no meaningful ply) fitted on the human fit rows (as stage B) and on those with the fill rows added, the penalty chosen once on all the fit rows with the whole fill and kept for the smaller shares, the same test rows, read on all human test rows and on the test rows with at most ENDGAME_MEN units on
the board; a curve over the share of the fill games used (none, half, all).

  .venv/bin/python -m catspace.incubator.analyst.discovery.events_fill coverage --sample-dir D
  .venv/bin/python -m catspace.incubator.analyst.discovery.events_fill generate --sample-dir D --out-dir O
  .venv/bin/python -m catspace.incubator.analyst.discovery.events_fill compare  --sample-dir D --fill-dir O
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import events_hazard as EH
from catspace.incubator.analyst.discovery import events_rules as ER
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import sleep_loop as SL
from catspace.io import paths

SMALL = 6
THIN = 30
SETS = 600
PER_SET = 2
GAMES_PER_START = 2
OPENING = 2
SECONDS = 0.05
MAX_PLIES = 120
ENDGAME_MEN = 8
VARIANT = ER.VARIANT
TAKING = "taken"                                                    # the class types of events_rules read in the comparison: every taking class, and the derived reading of the taking kept
KEPT = "keeps"


def h(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


# ------------------------------------------------------------------------------------------------------------------------------ coverage

def set_keys(census: np.ndarray, goal_index: int) -> "tuple[np.ndarray, np.ndarray]":
    """(units on the board without the goal units, a key of the set of units per position: the two sides' counts by kind, in either order) for the positions of one game."""
    nk = census.shape[2]
    keep = [k for k in range(nk) if k != goal_index]
    c = census[:, :, keep].astype(np.int64)
    base = 17 ** np.arange(len(keep))
    code = (c * base).sum(axis=2)                                           # (positions, 2)
    lo, hi = code.min(axis=1), code.max(axis=1)
    return c.sum(axis=(1, 2)), lo * (17 ** len(keep)) + hi


def decode(key: int, kinds: list, goal: str) -> str:
    keep = [k for k in kinds if k != goal]
    n = len(keep)
    lo, hi = divmod(int(key), 17 ** n)
    out = []
    for code in (lo, hi):
        words = []
        for k in keep:
            code, c = divmod(code, 17)
            words += [k] * c
        out.append("+".join(words) or "-")
    return " | ".join(out)


def coverage(res: list, games: list, say, label: str) -> dict:
    """The coverage lines for one set of games; returns {key: games set} of the small sets."""
    goal = G2M.goal_kind_of(VARIANT)
    kinds = res[0]["kinds"]
    gi = kinds.index(goal)
    pos = games_n = 0
    cuts = (24, 16, 12, 8, 6, 4)
    at = {c: [0, set()] for c in cuts}
    cnt: dict = {}
    gms: dict = {}
    small_pos = 0
    for i, r in enumerate(res):
        men_all = r["census"].sum(axis=(1, 2))
        non, key = set_keys(r["census"], gi)
        pos += len(men_all)
        games_n += 1
        for c in cuts:
            sel = men_all <= c
            at[c][0] += int(sel.sum())
            if sel.any():
                at[c][1].add(i)
        for k, n in zip(key[non <= SMALL], non[non <= SMALL]):
            cnt[int(k)] = cnt.get(int(k), 0) + 1
            gms.setdefault(int(k), set()).add(i)
            small_pos += 1
    say(f"coverage [{label}]: games {games_n}, positions {pos}")
    for c in cuts:
        say(f"coverage [{label}]: units left <= {c}: positions {at[c][0]} ({at[c][0] / pos * 100:.2f} pct), games reaching it {len(at[c][1])} ({len(at[c][1]) / games_n * 100:.1f} pct)")
    thin = [k for k in cnt if len(gms[k]) < THIN]
    say(f"coverage [{label}]: small sets (at most {SMALL} units, the goal units not counted, the two sides in either order): {len(cnt)} distinct, {sum(1 for k in cnt if len(gms[k]) >= THIN)} in {THIN} games or more, {len(thin)} thin (under {THIN} games), {sum(1 for k in cnt if len(gms[k]) < 5)} in under 5 games; positions in them {small_pos} ({small_pos / pos * 100:.2f} pct of all), in the thin ones {sum(cnt[k] for k in thin)}")
    for k in sorted(cnt, key=lambda k: -cnt[k])[:12]:
        say(f"coverage [{label}]: set {decode(k, kinds, goal)}: positions {cnt[k]}, games {len(gms[k])}")
    end = {"decisive, the rules' end on the board": 0, "decisive, stopped before it": 0, "drawn, the rules' end on the board": 0, "drawn, stopped before it": 0}
    for r, g in zip(res, games):
        d = g["y"] != 1
        end[("decisive" if d else "drawn") + (", the rules' end on the board" if r["over"] else ", stopped before it")] += 1
    say(f"coverage [{label}]: how the games ended: " + ", ".join(f"{k} {v} ({v / games_n * 100:.1f} pct)" for k, v in end.items()))
    return gms


def cmd_coverage(a) -> int:
    z = EH.load_sample(Path(a.sample_dir))
    allg = {json.loads(x)["id"]: json.loads(x) for x in (Path(paths.experiment(*ER.HEADS_DIR)) / "games.jsonl").read_text().splitlines() if x.strip()}
    games = [allg[i] for i in z["ids"]]
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS endgame: {m}")
        print(lines[-1], flush=True)
    for label, pick in (("fit games", [g["fit"] for g in games]), ("test games", [not g["fit"] for g in games])):
        idx = [i for i, p in enumerate(pick) if p]
        coverage([z["res"][i] for i in idx], [games[i] for i in idx], say, label)
    out = Path(a.out_dir or a.sample_dir)
    (out / "coverage_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ the fill

def pick_starts(z: dict, games: list, say) -> list:
    goal = G2M.goal_kind_of(VARIANT)
    gi = z["res"][0]["kinds"].index(goal)
    occ: dict = {}
    for i, (r, g) in enumerate(zip(z["res"], games)):
        if not g["fit"]:
            continue
        non, key = set_keys(r["census"], gi)
        seen = {}
        for p in np.flatnonzero(non <= SMALL):
            seen.setdefault(int(key[p]), int(p))
        for k, p in seen.items():
            occ.setdefault(k, []).append((i, p))
    thin = sorted((k for k, v in occ.items() if len(v) < THIN), key=lambda k: h(f"set:{k}"))
    say(f"fill: thin sets in the fit games (under {THIN} games, at most {SMALL} units): {len(thin)} of {len(occ)}; sampling {min(SETS, len(thin))} of them in a hash order, up to {PER_SET} positions each from different fit games")
    starts = []
    for k in thin:
        if len([s for s in starts]) >= SETS * PER_SET:
            break
        cand = sorted(occ[k], key=lambda x: h(f"pick:{k}:{x[0]}"))[:PER_SET]
        for gi_, p in cand:
            text = CE.state(VARIANT, None, games[gi_]["moves"][:p])[0]
            if OM.playable(text):
                starts.append({"set": k, "game": games[gi_]["id"], "ply": p, "text": text})
    return starts


def cmd_generate(a) -> int:
    t0 = time.perf_counter()
    z = EH.load_sample(Path(a.sample_dir))
    allg = {json.loads(x)["id"]: json.loads(x) for x in (Path(paths.experiment(*ER.HEADS_DIR)) / "games.jsonl").read_text().splitlines() if x.strip()}
    games = [allg[i] for i in z["ids"]]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS endgame: {m}")
        print(lines[-1], flush=True)
    starts = pick_starts(z, games, say)
    rec = json.loads((Path(paths.experiment("odds_78_61", "full")) / "gen2" / "record.json").read_text())
    player = SL.player_from_values(rec["heads"]["odds_effect"]["kind_values"])
    say(f"fill: the player of generation 2 of 78.61 (judge fed from its odds record, results of games): {player['record_kinds']}; {len(starts)} start positions, {GAMES_PER_START} games each after {OPENING} random plies (different seeds), {SECONDS} s a move, ply cap {MAX_PLIES}, {a.workers} workers; machine load {os.getloadavg()[0]:.2f}")
    specs = []
    for j, s in enumerate(starts):
        for k in range(GAMES_PER_START):
            op = OM.random_opening(s["text"], 6400 + 10 * j + k, OPENING)
            specs.append({"gid": f"fill:{j}:{k}", "start": s["set"], "text": s["text"], "opening": op, "players": [player, player], "seconds": SECONDS, "max_plies": MAX_PLIES, "scratch": str(out / "scratch")})
    specs = specs[:a.limit] if a.limit else specs
    out_file = out / "played.jsonl"
    played = OM.run_games(specs, a.workers, out_file, say=say)
    by = {s["gid"]: s for s in specs}
    fill = []
    for g in played:
        s = by[g["gid"]]
        first = CE.state(VARIANT, s["text"], [])[1]
        roles = ER.role_words(ER.rule_clauses(ER.spec_and_vocab()[0]))
        if g["outcome"] is None:
            y = 1
        else:
            win_first = (g["outcome"] == 100) if first == roles[0] else (g["outcome"] == 0)
            y = 0 if win_first else (1 if g["outcome"] == 50 else 2)
        fill.append({"id": g["gid"], "moves": g["moves"], "start": s["text"], "y": y, "unfinished": bool(g["unfinished"]), "welo": 0, "belo": 0, "band": 0, "fit": True, "arrival": False})
    (out / "fill_games.json").write_text(json.dumps(fill))
    ended = Counter_of([("unfinished" if g["unfinished"] else ("first mover wins" if g["y"] == 0 else ("draw" if g["y"] == 1 else "first mover loses"))) for g in fill])
    say(f"fill: {len(fill)} games, plies {sum(len(g['moves']) for g in fill)}; ended: {ended}; wall {time.perf_counter() - t0:.1f} s")
    (out / "generate_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


def Counter_of(items: list) -> dict:
    out: dict = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def fill_sample(fill_dir: Path, threads: int, say) -> "tuple[dict, dict]":
    """(sample, rows): the fill games read as the human ones are (events, edge distances) and their settled rows (the kinds and statements as 78.59 reads them), the rows from the third ply on."""
    games = json.loads((fill_dir / "fill_games.json").read_text())
    HD.OPEN_PLIES = OPENING
    rows = HD.build_rows(games, threads, say, concepts=False)
    quiet: dict = {}
    for g, p in zip(rows["game"].tolist(), rows["ply"].tolist()):
        quiet.setdefault(g, []).append(p)
    for i, g in enumerate(games):
        g["quiet"] = quiet.get(i, [])
    res = ER.read_games(games, {}, {}, threads)
    return {"ids": [g["id"] for g in games], "res": res, "games": games}, rows


def concat(a: dict, b: dict, na: int) -> dict:
    out = {}
    for k in a:
        if k == "meta":
            out[k] = a[k]
        elif k in ("state", "C"):
            out[k] = None
        elif k == "g":
            out[k] = np.concatenate([a[k], b[k] + na])
        else:
            out[k] = np.concatenate([a[k], b[k]])
    return out


def cmd_compare(a) -> int:
    t0 = time.perf_counter()
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS endgame: {m}")
        print(lines[-1], flush=True)
    heads_dir = Path(paths.experiment(*ER.HEADS_DIR))
    z = EH.load_sample(Path(a.sample_dir))
    allg = {json.loads(x)["id"]: json.loads(x) for x in (heads_dir / "games.jsonl").read_text().splitlines() if x.strip()}
    spec, _v = ER.spec_and_vocab()
    roles = ER.role_words(ER.rule_clauses(spec))
    EH.end_events(z["res"], z["ids"], allg, roles)
    human = EH.build_rows(z, heads_dir, None)
    fz, frows = fill_sample(Path(a.fill_dir), a.threads, say)
    fidx = {g["id"]: i for i, g in enumerate(fz["games"])}
    fill_rows = EH.build_rows({"ids": fz["ids"], "res": fz["res"]}, None, None, rows_src=frows, index=fidx)
    nh = len(z["res"])
    res = z["res"] + fz["res"]
    rows = concat(human, fill_rows, nh)
    men = np.array([r["census"][p].sum() for r, p in zip((res[int(g)] for g in rows["g"]), rows["ply"])])
    is_fill = rows["g"] >= nh
    say(f"rows: human {int((~is_fill).sum())} (fit {int((human['fit']).sum())}, test {int((~human['fit']).sum())}), fill {int(is_fill.sum())} from {len(fz['res'])} games; test rows with at most {ENDGAME_MEN} units on the board: {int(((~is_fill) & (~rows['fit']) & (men <= ENDGAME_MEN)).sum())} of {int(((~is_fill) & (~rows['fit'])).sum())}; fill rows with at most {ENDGAME_MEN} units: {int((is_fill & (men <= ENDGAME_MEN)).sum())}")
    X = np.hstack([rows["K"], rows["S"], rows["E"]])
    fit_h = np.flatnonzero(rows["fit"] & ~is_fill)
    test_all = np.flatnonzero(~rows["fit"] & ~is_fill)
    test_end = test_all[men[test_all] <= ENDGAME_MEN]
    fill_games = np.unique(rows["g"][is_fill])
    perm = np.random.default_rng(EH.SEED + 5).permutation(fill_games)
    fit_flags = np.array([allg[i]["fit"] for i in z["ids"]])
    cores = [c for c in sorted(EH.cores_of(z["res"], roles, fit_flags)) if c[0] == TAKING or c == (KEPT,)]
    say(f"comparison heads (every class of {TAKING} and the reading {KEPT} of any): " + ", ".join(" | ".join(c) for c in cores))
    for core in cores:
        for rel in ("us", "them"):
            hd = EH.Head(core, rel, *EH.targets(core, rel, roles, rows, res))
            hd.activate(rows["fit"])
            name = f"{' | '.join(core)} [{rel}]"
            if int((hd.has & rows["fit"] & ~is_fill).sum()) < ER.MIN_EVENTS:
                continue
            lam, out_tot = None, {}
            for share in (1.0, 0.0, 0.5):
                use = perm[:int(round(len(perm) * share))]
                fit_sel = np.concatenate([fit_h, np.flatnonzero(np.isin(rows["g"], use) & is_fill)]) if share else fit_h
                m = EH.fit_best(hd, X, rows, fit_sel, lam)
                lam = m["lam"] if lam is None else lam
                bb = hd.base_logit(fit_sel)
                for lab, sel_test in (("all human test rows", test_all), (f"test rows with at most {ENDGAME_MEN} units", test_end)):
                    tot_h, _p = EH.eval_head(hd, X, m, sel_test)
                    tot_b, _p2 = EH.eval_head(hd, None, None, sel_test, bb)
                    den = hd.steps()[sel_test].astype(np.float64)
                    out_tot[(share, lab)] = (tot_h, tot_b)
                    g, lo, hi = EH.boot(rows["g"][sel_test], tot_b - tot_h, den, EH.DRAWS, EH.SEED)
                    v = tot_h.sum() / den.sum()
                    say(f"HEAD {name} on {lab} ({len(sel_test)} rows, {int(hd.has[sel_test].sum())} events): fill share {share} ({len(use)} fill games, {len(fit_sel)} fit rows): log loss per step {v:.5f}; gain over its base hazard {g:+.5f} [{lo:+.5f}, {hi:+.5f}]")
            for lab, sel_test in (("all human test rows", test_all), (f"test rows with at most {ENDGAME_MEN} units", test_end)):
                den = hd.steps()[sel_test].astype(np.float64)
                g, lo, hi = EH.boot(rows["g"][sel_test], out_tot[(0.0, lab)][0] - out_tot[(1.0, lab)][0], den, EH.DRAWS, EH.SEED)
                say(f"HEAD {name} on {lab}: log loss per step WITHOUT the fill less WITH all of it {g:+.5f} [{lo:+.5f}, {hi:+.5f}] (positive: the fill helps)")
    say(f"wall {time.perf_counter() - t0:.1f} s")
    (Path(a.fill_dir) / "compare_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--sample-dir", required=True)
    c.add_argument("--out-dir", default=None)
    g = sub.add_parser("generate")
    g.add_argument("--sample-dir", required=True)
    g.add_argument("--out-dir", required=True)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--limit", type=int, default=0)
    k = sub.add_parser("compare")
    k.add_argument("--sample-dir", required=True)
    k.add_argument("--fill-dir", required=True)
    k.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    return {"coverage": cmd_coverage, "generate": cmd_generate, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
