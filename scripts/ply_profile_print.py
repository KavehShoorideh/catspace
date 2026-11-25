"""A MEASURE ASKED BY KAVEH 2026-11-25 (SYSTEM_SPEC 11, last block, (ix); ledger row 55; piece: rules; LAYER 2): the time of the question "the allowed moves" by ply along human games.
  python scripts/ply_profile_print.py games <records folder name> <out file> [<games>]   the moves of the first <games> (default 300) games of at least 60 plies, a game a line; no name of a player is read or written
  python scripts/ply_profile_print.py plot <rows file> <tag>                              the figures (through catspace.io.paths) and OBS lines, from the rows of rust/engine/game/tests/translated_ply_profile.rs
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

from catspace.io import paths


def games(folder: str, out: str, n: int) -> int:
    import pyarrow.parquet as pq
    root = paths.records_dir() / folder
    lines = []
    for f in sorted(root.glob("records_*.parquet")):
        t = pq.ParquetFile(f).read(columns=["n_plies", "moves"])
        for plies, moves in zip(t.column("n_plies").to_pylist(), t.column("moves").to_pylist()):
            if plies >= 60:
                lines.append(moves.strip())
                if len(lines) >= n:
                    break
        if len(lines) >= n:
            break
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"OBS ply profile games: {len(lines)} games of at least 60 plies from {folder} written to {out}")
    return 0


def read_rows(path: str):
    T = collections.defaultdict(dict)       # label -> ply -> ns an ask
    P = collections.defaultdict(lambda: collections.defaultdict(dict))  # label -> piece -> ply -> (ns, rows, calls)
    S = {}
    for line in Path(path).read_text().splitlines():
        c = line.split(",")
        if c[0] == "T":
            T[c[1]][int(c[2])] = int(c[4]) / int(c[3])
        elif c[0] == "P":
            P[c[1]][c[3]][int(c[2])] = (int(c[4]), int(c[5]), int(c[6]))
        elif c[0] == "S":
            S[int(c[1])] = (int(c[2]), int(c[3]) / int(c[2]), int(c[4]) / int(c[2]))
    return T, P, S


def plot(rows: str, tag: str) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T, P, S = read_rows(rows)
    plies = sorted(p for p in S if S[p][0] >= 30)          # a ply held by fewer than 30 games is left out
    out = []
    # figure 1: the time of the question by ply, each way
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for label in ("fast", "carry", "lemma"):
        ax[0].plot(plies, [T[label][p] / 1000 for p in plies], label=f"translated, {label}")
    ax[0].set_xlabel("ply"); ax[0].set_ylabel("microseconds a question"); ax[0].set_title("Allowed moves: time by ply, translated module"); ax[0].legend(); ax[0].set_ylim(bottom=0); ax[0].grid(alpha=0.3)
    ax[1].plot(plies, [T["library"][p] for p in plies], color="black", label="outside library")
    ax[1].set_xlabel("ply"); ax[1].set_ylabel("nanoseconds a question"); ax[1].set_title("Allowed moves: time by ply, outside library"); ax[1].set_ylim(bottom=0); ax[1].grid(alpha=0.3); ax[1].legend()
    f1 = paths.figure(f"ply_profile_{tag}_time.png"); fig.tight_layout(); fig.savefig(f1, dpi=130); plt.close(fig); out.append(f1)
    # figure 2: what stands at a ply, and the time against it
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(plies, [S[p][1] for p in plies], label="allowed moves"); ax[0].plot(plies, [S[p][2] for p in plies], label="units on the board")
    ax[0].set_xlabel("ply"); ax[0].set_title("What stands at a ply (mean)"); ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_ylim(bottom=0)
    ax[1].scatter([S[p][1] for p in plies], [T["fast"][p] / 1000 for p in plies], s=10, label="fast"); ax[1].scatter([S[p][1] for p in plies], [T["lemma"][p] / 1000 for p in plies], s=10, label="lemma")
    ax[1].set_xlabel("allowed moves (mean at the ply)"); ax[1].set_ylabel("microseconds a question"); ax[1].set_title("Time against the number of allowed moves"); ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_ylim(bottom=0); ax[1].set_xlim(left=0)
    ax[2].plot(plies, [T["fast"][p] / T["library"][p] for p in plies], label="fast"); ax[2].plot(plies, [T["lemma"][p] / T["library"][p] for p in plies], label="lemma")
    ax[2].set_xlabel("ply"); ax[2].set_ylabel("times slower than the outside library"); ax[2].set_title("The gap by ply"); ax[2].legend(); ax[2].grid(alpha=0.3); ax[2].set_ylim(bottom=0)
    f2 = paths.figure(f"ply_profile_{tag}_what.png"); fig.tight_layout(); fig.savefig(f2, dpi=130); plt.close(fig); out.append(f2)
    # figure 3: where the time goes, by piece, by ply (the profile is on: the sum is above the clock of figure 1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for a, label in zip(ax, ("fast", "carry", "lemma")):
        tot = {pc: sum(v[0] for p, v in d.items() if p in set(plies)) for pc, d in P[label].items()}
        top = sorted(tot, key=tot.get, reverse=True)[:8]
        n = {p: S[p][0] * 2 for p in plies}
        ys = [[P[label][pc].get(p, (0, 0, 0))[0] / n[p] / 1000 for p in plies] for pc in top]
        rest = [sum(d.get(p, (0, 0, 0))[0] for pc, d in P[label].items() if pc not in top) / n[p] / 1000 for p in plies]
        a.stackplot(plies, *ys, rest, labels=[pc[:28] for pc in top] + ["all other pieces"])
        a.set_xlabel("ply"); a.set_ylabel("microseconds a question (profile on)"); a.set_title(f"Where the time goes: {label}"); a.legend(fontsize=7, loc="upper right"); a.grid(alpha=0.3)
        whole = sum(tot.values())
        for pc in top:
            rows_ = sum(v[1] for p, v in P[label][pc].items() if p in set(plies)); calls = sum(v[2] for p, v in P[label][pc].items() if p in set(plies))
            print(f"OBS ply profile piece {label}: {pc} share {100 * tot[pc] / whole:.1f}% of the pieces' time, {tot[pc] / max(calls, 1):.0f} ns a call, {rows_ / max(calls, 1):.1f} rows a call")
    f3 = paths.figure(f"ply_profile_{tag}_pieces.png"); fig.tight_layout(); fig.savefig(f3, dpi=130); plt.close(fig); out.append(f3)
    for label in ("library", "fast", "carry", "lemma"):
        first, early, mid, late = T[label][0], T[label][plies[1]], T[label][plies[len(plies) // 2]], T[label][plies[-1]]
        print(f"OBS ply profile time {label}: ply 0 {first:.0f} ns, ply {plies[1]} {early:.0f} ns, ply {plies[len(plies) // 2]} {mid:.0f} ns, ply {plies[-1]} {late:.0f} ns")
    print(f"OBS ply profile: plies 0 to {plies[-1]} held by at least 30 games ({S[0][0]} games at ply 0, {S[plies[-1]][0]} at the last); figures: " + ", ".join(str(p) for p in out))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "games":
        sys.exit(games(a[1], a[2], int(a[3]) if len(a) > 3 else 300))
    if len(a) == 3 and a[0] == "plot":
        sys.exit(plot(a[1], a[2]))
    print(__doc__)
    sys.exit(2)
