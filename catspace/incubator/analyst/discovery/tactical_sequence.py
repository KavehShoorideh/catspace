"""PLAN 78.62 B: THE SEQUENCE, DESCRIBED WITH NO MODEL. Along a sampled human game the tactical state of stage A is a sequence over the plies. Printed, by band of the mean rating of the game and for all:
 (1) how the plies divide: no standing loss for either colour, one colour's units standing lost by the estimate, both;
 (2) how long a QUIET STRETCH lasts (consecutive plies with no unit of either colour standing lost by the estimate; and, beside it, with no confirmed threat move for either side by the rules' forcing read of 78.59
     where it was kept): runs that end in a standing loss, and runs cut by the end of the game, counted apart;
 (3) the ONSETS: a colour's first ply of a standing loss (the ply before it had none): within the next HORIZON plies, how often the loss HAPPENS (a unit of the loss was taken, and the share taken for nothing),
     how often it is PARRIED (no unit of it was taken and the colour's standing loss is gone at some ply), how often a COUNTER-CHANCE of the other colour appears (a unit of the other colour stands lost); beside it
     the same counter-chance rate over the plies with no standing loss for either colour; split by whether the colour with the loss is to move at the onset (it can parry at once) or not;
 (4) the plain transition table from the state at a ply to the state at the next.
Direction only. A standing loss is lost-by-estimate in the sense of tactical_tables.py. No model, no fit, no number from any engine evaluation.

  .venv/bin/python -m catspace.incubator.analyst.discovery.tactical_sequence --state-dir D --out-dir O
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import tactical_tables as TT
from catspace.io import paths

HORIZON = 8
STATE_WORDS = ["none", "colour 0 only", "colour 1 only", "both"]


def stretches(flag: np.ndarray, first: np.ndarray, last: np.ndarray) -> "tuple[list, list]":
    """(lengths of the runs of True that end because a row with False follows, lengths of the runs cut by the end of the game's rows), over the games."""
    ended, cut = [], []
    for a, b in zip(first, last):
        run = 0
        for i in range(a, b):
            if flag[i]:
                run += 1
            else:
                if run:
                    ended.append(run)
                run = 0
        if run:
            cut.append(run)
    return ended, cut


def qline(v: list) -> str:
    if not v:
        return "none"
    a = np.array(v)
    return f"{len(a)} runs, mean {a.mean():.2f}, median {np.median(a):.1f}, 90th percentile {np.percentile(a, 90):.1f}, longest {int(a.max())}"


def onsets(tb: dict, st: dict, first: np.ndarray, last: np.ndarray) -> dict:
    """Per onset: (game slot, colour, owner to move, happened, for nothing, parried, counter, band) for the onsets with HORIZON rows after them in their game."""
    x = tb["lost_est"] > 0
    out = []
    for gi, (a, b) in enumerate(zip(first, last)):
        for i in range(a + 1, b - HORIZON):
            for c in (0, 1):
                if not x[i, c] or x[i - 1, c]:
                    continue
                happened = tb["lost_taken"][i, c] > 0
                nothing = tb["lost_for_nothing"][i, c] > 0
                nxt = slice(i + 1, i + 1 + HORIZON)
                parried = (not happened) and bool((~x[nxt, c]).any())
                counter = bool(x[nxt, 1 - c].any())
                out.append((gi, c, int(st["mover"][i] == c), bool(happened), bool(nothing), parried, counter))
    return np.array(out, dtype=np.int32).reshape(-1, 7)


def baseline_counter(tb: dict, first: np.ndarray, last: np.ndarray) -> "tuple[int, int]":
    """Over the rows with no standing loss for either colour (and HORIZON rows after them): (rows, (row, colour) pairs where that colour has a standing loss within the next HORIZON rows)."""
    x = tb["lost_est"] > 0
    n = hit = 0
    for a, b in zip(first, last):
        for i in range(a, b - HORIZON):
            if x[i].any():
                continue
            for c in (0, 1):                                    # one colour at a time, as an onset's counter-chance is one colour's
                n += 1
                hit += bool(x[i + 1:i + 1 + HORIZON, c].any())
    return n, hit


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state-dir", default=paths.experiment("tactical_78_62", "state"))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out_dir or paths.experiment("tactical_78_62", "sequence"))
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS tactical sequence: {msg}")
        print(lines[-1], flush=True)

    d = TT.load_all(Path(a.state_dir))
    st, tb, games, fc = d["st"], d["tb"], d["games"], d["forcing"]
    ids, first, last = TT.game_slices(st)
    band = np.array([games[int(g)]["band"] for g in ids])
    say(f"games {len(ids)}, rows {len(st['ply'])}, horizon {HORIZON} plies; a standing loss is a unit of the colour lost by the estimate")
    x = tb["lost_est"] > 0
    state = x[:, 0].astype(int) + 2 * x[:, 1].astype(int)
    say("(1) plies by state: " + ", ".join(f"{w} {int((state == k).sum())} ({(state == k).mean():.4f})" for k, w in enumerate(STATE_WORDS)))
    mv = st["mover"].astype(int)
    own_lost = np.array([x[i, mv[i]] for i in range(len(mv))])
    oth_lost = np.array([x[i, 1 - mv[i]] for i in range(len(mv))])
    say(f"(1) plies where a unit of the side to move stands lost {int(own_lost.sum())} ({own_lost.mean():.4f}); where a unit of the other side stands lost (the side to move may win it) {int(oth_lost.sum())} ({oth_lost.mean():.4f})")
    say(f"(1) units lost by the estimate {int(tb['lost_est'].sum())}, lost resting on a bound {int(tb['lost_bound'].sum())}, lost forced or pruned {int(tb['lost_firm'].sum())}; units forced safe {int(tb['safe_firm'].sum())}, safe by the estimate {int(tb['safe_est'].sum())}; trapped {int(tb['trapped'].sum())}")
    # (2) quiet stretches
    quiet = ~(x[:, 0] | x[:, 1])
    for lab, sel in [("all", np.ones(len(ids), bool))] + [(HD.BAND_NAMES[b], band == b) for b in range(4)]:
        e, c = stretches(quiet, first[sel], last[sel])
        say(f"(2) quiet stretch (no unit of either colour lost by the estimate) [{lab}]: ended by a standing loss: {qline(e)}; cut by the end of the game: {qline(c)}")
    if fc is not None:
        have = (fc[:, 0] >= 0)
        qf = have & (fc[:, 0] == 0) & (fc[:, 1] == 0)
        e, c = stretches(qf, first, last)
        say(f"(2) quiet stretch by the rules' forcing read (no confirmed threat move for either side), all games: ended: {qline(e)}; cut: {qline(c)}; rows with a read {int(have.sum())} of {len(have)}")
        say(f"(2) beside it, over the rows with a forcing read: a standing loss by the estimate holds on {float(x[qf].any(axis=1).mean()):.4f} of the {int(qf.sum())} rows with no confirmed threat move for either side, and on "
            f"{float(x[have & ~qf].any(axis=1).mean()):.4f} of the {int((have & ~qf).sum())} rows where some side holds one")
    # (3) onsets
    ons = onsets(tb, st, first, last)
    n0, h0 = baseline_counter(tb, first, last)
    say(f"(3) onsets of a standing loss with {HORIZON} rows after them: {len(ons)} (colour 0: {int((ons[:, 1] == 0).sum())}, colour 1: {int((ons[:, 1] == 1).sum())}); the counter-chance base: of {n0} (row, colour) pairs over rows with no standing loss for either colour, the colour has one within {HORIZON} plies in {h0 / max(1, n0):.4f}")
    for lab, sel in [("all", np.ones(len(ons), bool)), ("owner to move", ons[:, 2] == 1), ("owner not to move", ons[:, 2] == 0)] + [(HD.BAND_NAMES[b], band[ons[:, 0]] == b) for b in range(4)]:
        o = ons[sel]
        if not len(o):
            continue
        say(f"(3) onsets [{lab}]: n {len(o)}; the loss happens (a unit of it taken) {o[:, 3].mean():.4f}, taken for nothing {o[:, 4].mean():.4f}; parried (nothing taken, the colour's standing loss gone at some ply) {o[:, 5].mean():.4f}; "
            f"a counter-chance of the other colour appears {o[:, 6].mean():.4f}")
    # (4) transitions
    t = np.zeros((4, 4), np.int64)
    for a_, b_ in zip(first, last):
        for i in range(a_, b_ - 1):
            t[state[i], state[i + 1]] += 1
    for k in range(4):
        s = t[k].sum()
        say(f"(4) from [{STATE_WORDS[k]}] (n {int(s)}) to the next ply: " + ", ".join(f"{STATE_WORDS[j]} {t[k, j] / max(1, s):.4f}" for j in range(4)))
    # the duration of a standing loss
    for lab, sel in [("all", np.ones(len(ids), bool))]:
        durs = []
        for c in (0, 1):
            e, cu = stretches(x[:, c], first[sel], last[sel])
            durs += e
        say(f"(3) duration of one colour's standing loss in plies (runs that ended): {qline(durs)}")
    (out / "sequence_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
