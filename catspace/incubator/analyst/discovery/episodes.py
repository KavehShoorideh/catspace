"""EPISODES, THE RATCHET, AND THE DIVERGENCE (Kaveh 2026-10-12: "how do we measure progress? Are we measuring progress
properly? ... a frame where we have a desired outcome and ... a non-desired outcome ... episodic, so basically a threat
that is created, and later a different threat that is created ... once we figured out that we're not progressing, we
should go back and identify the point at which we started to diverge").

  EPISODES    a material phase of a game cut into runs of one KIND, read from the planner's own decisions: in check (their
              threat on our king), frame (our measure steering), proof (a forced gain), arming (a threat prepared), defence
              (their threat answered), known (a forced entry into a material we know)
  OUTCOMES    each episode, desired or not: material won (desired) or lost (not); a threat cashed in or parried; otherwise
              THE RATCHET -- did the frame's measure reach a NEW BEST of the whole phase during it (a value below every one
              before)? A loop cannot make a new best: coming back down to an old level is not progress
  DIVERGENCE  in a game that did not progress: the start of the first failed episode after the last desired one; our own
              prover (the forced entry, a bigger budget) asked there whether a way into the known was missed

  python -m catspace.incubator.analyst.discovery.episodes --run approach
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from catspace.io import paths


def kind_of(row: dict, in_check: bool) -> str:
    if in_check:
        return "check"
    if "forces entry into the known" in row["why"] or "prepares a forced entry" in row["why"]:
        return "known"
    return row["claim"]["kind"]


def episodes(rows: list, idx: list, terms: list) -> list:
    """The episodes of one material phase (the indices `idx` of the game's rows) under the frame `terms`."""
    import chess
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    fens = [rows[i]["fen"] for i in idx]
    meas = A.component_values(fens, terms) if terms else [[] for _ in fens]
    kinds = [kind_of(rows[i], chess.Board(rows[i]["fen"]).is_check()) for i in idx]
    out, start = [], 0
    best = meas[0]
    for j in range(1, len(idx) + 1):
        if j == len(idx) or kinds[j] != kinds[start]:
            seg = range(start, j)
            # the first row after the episode -- for the phase's last episode the row after the phase (its capture or
            # trade happens on the way out of the phase)
            end = idx[j] if j < len(idx) else (idx[-1] + 1 if idx[-1] + 1 < len(rows) else None)
            lead0, theirs0 = rows[idx[start]]["lead"], rows[idx[start]]["theirs"]
            nxt = rows[end] if end is not None else rows[idx[-1]]
            lead1 = min(rows[idx[k]]["lead_after"] for k in seg)
            new_best = False
            for k in seg:
                if meas[k] < best:
                    best, new_best = meas[k], True
            if lead1 < lead0:
                outcome = "material lost"
            elif nxt["theirs"] < theirs0 and (end is not None):
                outcome = "material won"
            elif kinds[start] in ("proof", "arming", "known"):
                outcome = "threat parried"
            else:
                outcome = "new best" if new_best else "no new best"
            out.append(dict(kind=kinds[start], start=idx[start], length=j - start, outcome=outcome))
            start = j
    return out


DESIRED = ("material won", "new best")


def measure(run: str, material: str = "KRRvKR", budget: int = 200_000, levels: tuple = (1,), level_budget: int = 20_000) -> None:
    import chess
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    stem = f"surprise_loop_KRRvKR_{run}"
    history = json.load(open(paths.derived(f"{stem}.json")))
    tails, wons = Counter(), Counter()
    tails_k, wons_k = Counter(), Counter()
    since, missed, asked = [], 0, 0
    import time
    by_level, t_level = Counter(), Counter()
    for h in history:
        rnd, frames = h["round"], h["frames"]
        for f in frames:
            A.set_material_frame(f, frames[f])
        res = json.load(open(paths.experiment(f"{stem}_r{rnd}.json")))
        terms = frames.get(material, [])
        for g in res:
            rows = g["rows"]
            idx = [i for i, r in enumerate(rows) if r["material"] == material]
            if len(idx) < 2:
                continue
            eps = episodes(rows, idx, terms)
            drawn = g["out"] == "draw" and rows[-1]["material"] == material
            won_phase = idx[-1] + 1 < len(rows) and rows[idx[-1] + 1]["theirs"] < rows[idx[-1]]["theirs"] or (idx[-1] == len(rows) - 1 and g["out"] == "win")
            if drawn:
                tail = [e for e in eps if e["start"] >= idx[max(0, len(idx) - 30)]]
                tails.update(e["outcome"] for e in tail); tails_k.update(e["kind"] for e in tail)
                # THE DIVERGENCE: the start of the first failed episode after the last desired one
                last_good = max((k for k, e in enumerate(eps) if e["outcome"] in DESIRED), default=-1)
                if last_good + 1 < len(eps):
                    d = eps[last_good + 1]["start"]
                    since.append(sum(1 for i in idx if i >= d))
                    for i in [i for i in idx if i >= d][:6]:                       # around it: was a way in missed?
                        asked += 1
                        if 1 in levels and A.known_entry_of(rows[i]["fen"], budget):
                            missed += 1
                        for lv in levels:
                            if lv >= 2:
                                t0 = time.time()
                                if A.known_level_of(rows[i]["fen"], lv, level_budget):
                                    by_level[lv] += 1
                                t_level[lv] += time.time() - t0
            elif won_phase:
                wons.update(e["outcome"] for e in eps); wons_k.update(e["kind"] for e in eps)
    pct = lambda c: f"{100 * sum(c[o] for o in DESIRED) / max(sum(c.values()), 1):.0f}% desired ({dict(c.most_common())})"
    print(f"OBS run {run}, {material}: EPISODES in the drawn games' last 30 moves: {sum(tails.values())}, {pct(tails)}")
    print(f"OBS   their kinds: {dict(tails_k.most_common())}")
    print(f"OBS   in the won phases: {sum(wons.values())}, {pct(wons)}")
    print(f"OBS   their kinds: {dict(wons_k.most_common())}")
    if since:
        s = sorted(since)
        print(f"OBS   THE DIVERGENCE in {len(s)} drawn games: moves of ours from it to the end -- median {s[len(s) // 2]}, quartiles {s[len(s) // 4]}-{s[3 * len(s) // 4]}")
        print(f"OBS   our prover at the divergence and the five positions after it (the forced entry, budget {budget:,}): a way into the known in {missed} of {asked}")
        for lv in sorted(t_level):
            print(f"OBS   REGRESSION LEVEL {lv} there (each forced-entry question {level_budget:,}): a way in CREATED in {by_level[lv]} of {asked} "
                  f"({100 * by_level[lv] / max(asked, 1):.1f}%) -- {1000 * t_level[lv] / max(asked, 1):.0f} ms a position")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="approach")
    ap.add_argument("--material", default="KRRvKR")
    ap.add_argument("--budget", type=int, default=200_000)
    ap.add_argument("--levels", default="1", help="the regression levels to ask at the divergence, e.g. 1,2")
    ap.add_argument("--level-budget", type=int, default=20_000)
    a = ap.parse_args(argv)
    measure(a.run, a.material, a.budget, tuple(int(x) for x in a.levels.split(",")), a.level_budget)


if __name__ == "__main__":
    main()
