"""PLAN 78.46 SOUNDNESS OF THE FORCING READ on ordinary positions: every threat move the Rust forcing read (rust/engine/tactics/src/sheet/forcing.rs) CONFIRMS is checked against the
engine's own ledger search (judge by worth, the sheet's own worth table, no pruning heuristics) from the position after the move.

For every position of every game of a games directory (games.jsonl: `variant`, `start`, `moves`) where the side to move has a legal move:
  the forcing read is made (the fuse in positions as given, `--fuse`);
  every CONFIRMED threat move is checked: the ledger search is deepened from the position after the move to the depth that covers the read's longest line
      depth = (the longest line of any reply, the reply first, counted in plies) + 2 x (the deepening level of the verdict) + 1,
  and the captures after the horizon are resolved by the search itself (a quiet search, each side may stop). The search's value for the mover (what is won less what is lost from there on)
  must not be below what the read claims as certain from there on: the read's value less what the move itself took. A claim of the goal is checked against a won game (the ledger's number).
  A search the node fuse (`--search-nodes`) or the depth cap (`--depth-cap`) cut before the depth is NOT a check: it is counted apart, never as holding.
PLAN 78.47 E: the read itself now has the search as the last word (a verdict CONFIRMED stands only after the ledger search to the covering depth does not contradict it; the others are NOT BORNE OUT, with both
lines). So the measure here checks DEEPER than the verdict's own check: `--extra-plies` (default 2) plies more than the depth the read used, and counts apart the moves the read's own check turned into NOT BORNE OUT.
Printed: positions, confirmed claims, checked, contradicted, unchecked (by reason), how often the read confirms anything, seconds per position; every contradiction with its position, move, the
read's line (the reply that held the least, and its line) and the search's line.

  .venv/bin/python -m catspace.incubator.analyst.discovery.forcing_soundness_measure --games-dir DIR [--games N] [--jobs N] [--fuse N] [--tag NAME]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from collections import Counter
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import question_sheet as QS
from catspace.incubator.analyst.discovery import sheet_rust as SR

GOAL = 1e9
EXTRA_PLIES = 2                    # plan 78.47 E: the measure checks this many plies deeper than the verdict's own check, so it is not the verdict checking itself
_ST: dict = {}


def _setup(variant: str) -> dict:
    if not _ST:
        env = QS.Env(variant)
        worth = QS.store_worth(variant)
        _ST.update(env=env, model=SR.model(env, worth))
    return _ST


def needed_depth(threat: dict) -> int:
    """The depth in plies the check by search must cover: the longest line of the answer (the reply first) + two plies per deepening level + one."""
    return threat["plies"] + 2 * threat["level"] + 1


def judge_claim(threat: dict, check: dict, win: int) -> dict:
    """Compare one confirmed threat with the search from the position after the move. -> {status: holds | contradicted | unchecked, why}."""
    if check["cut"]:
        return {"status": "unchecked", "why": "search cut (node fuse) before the depth"}
    value = threat["value"]
    if value >= GOAL / 2:
        ok = check["value"] >= win - 1000
        return {"status": "holds" if ok else "contradicted", "why": f"goal claimed; search value {check['value']}"}
    claimed = value - threat["took_worth"]                       # what is certain from the position after the move on
    ok = check["value"] >= claimed - 1e-9
    return {"status": "holds" if ok else "contradicted", "why": f"read claims {claimed:g} from there; search value {check['value']}"}


def weakest_reply(threat: dict) -> dict:
    return min(threat["replies"], key=lambda r: r["value"]) if threat["replies"] else {}


def run_game(spec: dict) -> dict:
    st = _setup(spec["variant"])
    model = st["model"]
    v, start, moves = spec["variant"], spec["start"], spec["moves"]
    out = {"positions": 0, "read": 0, "with_confirmed": 0, "confirmed": 0, "checked": 0, "holds": 0, "contradicted": 0, "unchecked": Counter(), "ends_game": 0, "seconds": 0.0, "nodes": 0, "cut": 0,
           "found": 0, "unsettled": 0, "contradictions": [], "levels": Counter(), "check_seconds": 0.0, "not_borne_out": 0, "not_borne_out_list": [], "in_read_unchecked": 0}
    for ply in range(len(moves) + 1):
        pos = CE.state(v, start, moves[:ply])[0]
        out["positions"] += 1
        t0 = time.perf_counter()
        try:
            res = model.forcing(pos, spec["fuse"])
        except ValueError:
            continue
        out["seconds"] += time.perf_counter() - t0
        out["read"] += 1
        out["nodes"] += res["nodes"]
        out["fuse"] = res["fuse"]
        out["cut"] += bool(res["cut"])
        out["found"] += res["found"]
        out["unsettled"] += res["unsettled"]
        for t in res["threats"]:
            if t["verdict"] == "NOT BORNE OUT":
                ck = t["search"]
                w = weakest_reply(t)
                out["not_borne_out"] += 1
                out["not_borne_out_list"].append({"game": spec["game"], "ply": ply, "position": pos, "move": t["move"], "value": t["value"], "claimed": ck["claimed"], "search_value": ck["value"], "depth": ck["depth"],
                                                  "read_line": [t["move"]] + list(w.get("line", [])), "search_line": [t["move"]] + list(ck["line"])})
        conf = [t for t in res["threats"] if t["verdict"] == "CONFIRMED"]
        out["in_read_unchecked"] += sum(1 for t in conf if t["search"] and t["search"]["status"] == "unchecked")
        out["confirmed"] += len(conf)
        out["with_confirmed"] += bool(conf)
        for t in conf:
            out["levels"][t["level"]] += 1
            if not t["replies"]:
                out["ends_game"] += 1                              # the move itself ends the game in our favour: the rules say so, no search needed
                continue
            if spec.get("skip_check"):
                continue
            depth = needed_depth(t) + spec.get("extra", EXTRA_PLIES)
            if depth > spec["depth_cap"] + spec.get("extra", EXTRA_PLIES):
                out["unchecked"]["depth over the cap"] += 1
                continue
            t1 = time.perf_counter()
            chk = model.forcing_check(pos, t["move"], depth, spec["search_nodes"])
            out["check_seconds"] += time.perf_counter() - t1
            j = judge_claim(t, chk, chk["win"])
            if j["status"] == "unchecked":
                out["unchecked"][j["why"]] += 1
                continue
            out["checked"] += 1
            if j["status"] == "holds":
                out["holds"] += 1
            else:
                out["contradicted"] += 1
                w = weakest_reply(t)
                out["contradictions"].append({"game": spec["game"], "ply": ply, "position": pos, "move": t["move"], "value": t["value"], "took_worth": t["took_worth"], "level": t["level"],
                                              "depth": depth, "why": j["why"], "claims": [(c["target"], c["kind"], c["value"]) for c in t["claims"]],
                                              "read_line": [t["move"]] + list(w.get("line", [])), "weakest_reply": w.get("reply"), "weakest_value": w.get("value"),
                                              "search_line": [t["move"]] + list(chk["line"]), "search_depth": chk["depth"]})
    return out


def merge(parts: list) -> dict:
    tot = {"positions": 0, "read": 0, "with_confirmed": 0, "confirmed": 0, "checked": 0, "holds": 0, "contradicted": 0, "unchecked": Counter(), "ends_game": 0, "seconds": 0.0, "nodes": 0, "cut": 0,
           "found": 0, "unsettled": 0, "contradictions": [], "levels": Counter(), "check_seconds": 0.0, "not_borne_out": 0, "not_borne_out_list": [], "in_read_unchecked": 0}
    for p in parts:
        for k, v in p.items():
            if k == "fuse":
                continue
            if isinstance(v, Counter):
                tot[k].update(v)
            elif isinstance(v, list):
                tot[k].extend(v)
            else:
                tot[k] += v
    return tot


def format_obs(tot: dict, label: str, wall: float, jobs: int, fuse: int) -> list:
    n, rd = max(1, tot["positions"]), max(1, tot["read"])
    L = [f"OBS forcing soundness {label}: positions {tot['positions']} (read {tot['read']}), fuse {fuse} positions, jobs {jobs}, wall {wall:.1f} s",
         f"OBS forcing soundness {label}: confirmed claims {tot['confirmed']} (the move ends the game itself {tot['ends_game']}); checked {tot['checked']}, holds {tot['holds']}, CONTRADICTED {tot['contradicted']}; "
         f"unchecked {dict(tot['unchecked'])}; checked {EXTRA_PLIES} plies deeper than the verdict's own check",
         f"OBS forcing soundness {label}: the read's own check (the search as the last word): NOT BORNE OUT {tot['not_borne_out']}; confirmed moves it could not check (search cut or depth over the cap) {tot['in_read_unchecked']}",
         f"OBS forcing soundness {label}: positions with a confirmed claim {tot['with_confirmed']} ({100 * tot['with_confirmed'] / rd:.2f} pct of the read); confirmed claims per position {tot['confirmed'] / rd:.4f}; "
         f"by the level of the verdict {dict(sorted(tot['levels'].items()))}",
         f"OBS forcing soundness {label}: threat moves found per position {tot['found'] / rd:.2f}, not settled per position {tot['unsettled'] / rd:.2f}, positions where the fuse stopped the read {tot['cut']} ({100 * tot['cut'] / rd:.1f} pct), "
         f"positions read per position {tot['nodes'] / rd:.0f}",
         f"OBS forcing soundness {label}: seconds per position: the read {tot['seconds'] / rd:.4f}, the checks by search {tot['check_seconds'] / rd:.4f} (summed over workers)"]
    for c in tot["not_borne_out_list"]:
        L.append(f"OBS forcing soundness {label}: NOT BORNE OUT game {c['game']} ply {c['ply']} position {c['position']} move {c['move']} (value {c['value']:g}, claimed from there {c['claimed']:g}, search value {c['search_value']}, "
                 f"depth {c['depth']}): the read's line {c['read_line']}; the search's line {c['search_line']}")
    for c in tot["contradictions"]:
        L.append(f"OBS forcing soundness {label}: CONTRADICTION game {c['game']} ply {c['ply']} position {c['position']} move {c['move']} (value {c['value']:g}, the move took {c['took_worth']:g}, level {c['level']}, "
                 f"search depth {c['depth']}): {c['why']}; claims {c['claims']}; the read's line {c['read_line']} (weakest reply {c['weakest_reply']} = {c['weakest_value']}); the search's line {c['search_line']} (depth read {c['search_depth']})")
    return L


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--fuse", type=int, default=None, help="positions the forcing read may read (default: the engine's FUSE_POSITIONS)")
    ap.add_argument("--depth-cap", type=int, default=14)
    ap.add_argument("--extra-plies", type=int, default=EXTRA_PLIES)
    ap.add_argument("--search-nodes", type=int, default=20_000_000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip-check", action="store_true")
    a = ap.parse_args(argv)
    recs = [json.loads(line) for line in (Path(a.games_dir) / "games.jsonl").read_text().splitlines() if line.strip()]
    if a.games:
        recs = recs[:a.games]
    fuse = a.fuse
    specs = [dict(variant=r["variant"], start=r.get("start"), moves=r["moves"], game=r.get("game", i), fuse=fuse, depth_cap=a.depth_cap, search_nodes=a.search_nodes, skip_check=a.skip_check, extra=a.extra_plies) for i, r in enumerate(recs)]
    label = a.tag or Path(a.games_dir).name
    t0 = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(a.jobs) as pool:
        parts = pool.map(run_game, specs, chunksize=1)
    wall = time.perf_counter() - t0
    tot = merge(parts)
    fuse = max((p.get("fuse", 0) for p in parts), default=fuse)
    for line in format_obs(tot, label, wall, a.jobs, fuse):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
