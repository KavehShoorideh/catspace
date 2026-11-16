"""PLAN 78.62 E: THE SACRIFICE, AS A READING. In the TEST games of stage C: the positions where the side to move gave up a unit for nothing by the frame read and did not lose the game; were its chances by the head of
stage C above those of the positions where the unit was simply lost. A count and a comparison, no verdict.

THE EVENT: a row of stage A at which a colour c is to move, a unit of c stands lost by the estimate and was in fact taken for nothing within HORIZON plies, and that unit still stands lost at the next row (c's own
move did not save it): c gave it up. THE TWO GROUPS: S, the events of games that c did not lose (won or drew); L, the events of games that c lost ("simply lost"). THE CHANCES: of the first settled (quiet) row of the 78.59
file in the same game within HORIZON plies after the event, read by the heads of stage C (inputs K+S+E+R): c's chance of a unit of the other side standing lost with c to move within HORIZON plies, and the other side's
chance against c (the heads' own, swapped when it is the other side's turn at that row); beside them what happened (the targets of stage C observed). The comparison is the mean of S less the mean of L with a 90 pct band
over games, over all events, within each kind of unit given up (matched: the kind-weighted difference), and by band of rating. Everything is read on test games; the heads were fitted on fit games.

  .venv/bin/python -m catspace.incubator.analyst.discovery.tactical_sacrifice --state-dir D --chances-dir C --out-dir O
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import tactical_tables as TT
from catspace.io import paths

HORIZON = 8
DRAWS = 300
SEED = 6264


def events(d: dict) -> list:
    """[(game slot number, row, colour, kind of a unit given up)] over all games of the state file."""
    st, tb = d["st"], d["tb"]
    x = tb["lost_est"] > 0
    out = []
    ids, first, last = TT.game_slices(st)
    for g, a, b in zip(ids, first, last):
        for i in range(a, b - 1):
            c = int(st["mover"][i])
            if not x[i, c] or not x[i + 1, c] or tb["lost_for_nothing"][i, c] == 0:
                continue
            u = st["units"][st["uoff"][i]:st["uoff"][i + 1]]
            kinds = [int(r[1]) for r in u if int(r[2]) in (0, 2) and int(r[3]) == 2 and int(r[0]) == 1 and int(r[5]) == 1]
            out.append((int(g), i, c, kinds[0] if kinds else -1))
    return out


def band_diff(games: np.ndarray, s_mask: np.ndarray, v: np.ndarray, draws: int, seed: int, weights: "np.ndarray | None" = None) -> "tuple[float, float, float]":
    """Mean of v over the S events less the mean over the L events, a 90 pct band over resamples of the games (an unmatched difference; with weights the kind-matched one)."""
    ids, inv = np.unique(games, return_inverse=True)
    rng = np.random.default_rng(seed)

    def diff(sel_games: np.ndarray) -> float:
        # sel_games: count of each game in the resample
        w = sel_games[inv]
        s, l = s_mask, ~s_mask
        if (w[s]).sum() == 0 or (w[l]).sum() == 0:
            return float("nan")
        return float((w[s] * v[s]).sum() / w[s].sum() - (w[l] * v[l]).sum() / w[l].sum())
    full = diff(np.ones(len(ids)))
    vals = [diff(np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)).astype(float)) for _ in range(draws)]
    vals = [x for x in vals if np.isfinite(x)]
    return full, float(np.percentile(vals, 5)) if vals else float("nan"), float(np.percentile(vals, 95)) if vals else float("nan")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state-dir", default=paths.experiment("tactical_78_62", "state"))
    ap.add_argument("--chances-dir", default=paths.experiment("tactical_78_62", "chances"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--draws", type=int, default=DRAWS)
    a = ap.parse_args(argv)
    out = Path(a.out_dir or paths.experiment("tactical_78_62", "sacrifice"))
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS tactical sacrifice: {msg}")
        print(lines[-1], flush=True)

    d = TT.load_all(Path(a.state_dir))
    st, games = d["st"], d["games"]
    z = np.load(Path(a.chances_dir) / "chances.npz")
    kinds = st["meta"]["kinds"]
    # predictions of the two heads on the test rows (rows of the 78.59 file in the games of stage A), the targets observed
    n = len(z["game"])
    p_us = np.full(n, np.nan)
    p_them = np.full(n, np.nan)
    p_us[z["test_us"]] = z["pred_us"]
    p_them[z["test_them"]] = z["pred_them"]
    row_of = {}
    for j in range(n):
        if not np.isnan(p_us[j]):
            row_of.setdefault(int(z["game"][j]), []).append((int(z["ply"][j]), j))
    ev = events(d)
    test_games = {int(g) for g in z["game"][~z["fit"]]}
    ev = [e for e in ev if e[0] in test_games]
    say(f"events (a colour to move, a unit of it lost by the estimate and taken for nothing within {HORIZON} plies, still standing lost after its own move) in the test games: {len(ev)} in {len({e[0] for e in ev})} games")
    recs = []
    dropped = 0
    ply_of = st["ply"]
    for g, i, c, kd in ev:
        res = games[g]["y"] if c == 0 else 2 - games[g]["y"]
        nxt = [j for (pl, j) in row_of.get(g, []) if int(ply_of[i]) < pl <= int(ply_of[i]) + HORIZON]
        if not nxt:
            dropped += 1
            continue
        j = nxt[0]
        i2 = int(z["state_row"][j])
        mover_there = int(st["mover"][i2])
        gain, lose = (p_us[j], p_them[j]) if mover_there == c else (p_them[j], p_us[j])
        og, ol = (z["y_us"][j], z["y_them"][j]) if mover_there == c else (z["y_them"][j], z["y_us"][j])
        recs.append((g, int(res != 2), kd, gain, lose, og, ol, games[g]["band"]))
    say(f"events with a settled row of the 78.59 file within {HORIZON} plies after them: {len(recs)} (dropped for none: {dropped})")
    if len(recs) < 20:
        say("too few events to compare")
        (out / "sacrifice_obs.txt").write_text("\n".join(lines) + "\n")
        return 0
    r = np.array(recs, dtype=float)
    gm, notlost, kd = r[:, 0].astype(int), r[:, 1] > 0, r[:, 2].astype(int)
    say(f"group S (the colour did not lose the game): {int(notlost.sum())} events in {len(np.unique(gm[notlost]))} games; group L (it lost): {int((~notlost).sum())} events in {len(np.unique(gm[~notlost]))} games")
    for lab, col in (("the head's chance that the colour gains a unit (a unit of the other side stands lost with it to move within 8 plies)", 3), ("the head's chance that the other side gains one against it", 4),
                     ("observed: the colour gained one", 5), ("observed: the other side gained one against it", 6)):
        v, lo, hi = band_diff(gm, notlost, r[:, col], a.draws, SEED)
        say(f"{lab}: S mean {r[notlost, col].mean():.4f}, L mean {r[~notlost, col].mean():.4f}, S less L {v:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    net = r[:, 3] - r[:, 4]
    v, lo, hi = band_diff(gm, notlost, net, a.draws, SEED)
    say(f"the head's net chance (gain less loss): S mean {net[notlost].mean():+.4f}, L mean {net[~notlost].mean():+.4f}, S less L {v:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    # by kind of the unit given up
    for k in np.unique(kd):
        m = kd == k
        if (m & notlost).sum() >= 5 and (m & ~notlost).sum() >= 5:
            nm = kinds[k] if k >= 0 else "unknown"
            v, lo, hi = band_diff(gm[m], notlost[m], net[m], max(100, a.draws // 2), SEED)
            say(f"unit given up {nm}: S {int((m & notlost).sum())} events, L {int((m & ~notlost).sum())}; net chance S less L {v:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    # by band of rating
    bnd = r[:, 7].astype(int)
    for b in range(4):
        m = bnd == b
        if (m & notlost).sum() >= 5 and (m & ~notlost).sum() >= 5:
            v, lo, hi = band_diff(gm[m], notlost[m], net[m], max(100, a.draws // 2), SEED)
            say(f"band {HD.BAND_NAMES[b]}: S {int((m & notlost).sum())} events, L {int((m & ~notlost).sum())}; net chance S less L {v:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    # the reference: the mean net chance of every test row
    ok = ~np.isnan(p_us) & ~np.isnan(p_them)
    say(f"reference: mean of the head's net chance over all {int(ok.sum())} test rows (mover's gain less loss) {float((p_us[ok] - p_them[ok]).mean()):+.4f}")
    (out / "sacrifice_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
