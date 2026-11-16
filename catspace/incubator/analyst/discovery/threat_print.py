"""PLAN 78.70 E: THE PRINT of the threat list for the owner to read. A printing script, NOT a layer, sheet, binding or measure file: it is the one place that renders a position with the game's own renderer (the
python-chess board, the library the rule kernel's own spelling comes from: names.py) so that a reader can set the position up, and prints the rows the list holds (threat_list.py) with the engine's own names for kinds and
places (no mapping file names any of them: the mapping file of spec 2 holds names of mined concepts only, and no concept is printed here).

THE CHOICE OF THE TEN POSITIONS (a stated rule, not by hand): the held-out games of the pickle of `threat_measure read`, in its order (the md5 order of their ids); the plies of the game's settled rows are cut into
EARLY (ply under 20), MIDDLE (20 to 49) and LATE (50 on); for each phase in turn the games are scanned in order and the first settled row of the game inside the phase that has at least one row in band 1 (high, no
response) by the spread of the readings, on either side, is taken, one position per game and no game twice; 3 early, 4 middle, 3 late. A phase that runs out of such rows is left short and the print says so.

  .venv/bin/python -m catspace.incubator.analyst.discovery.threat_print [--out-dir D]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import threat_list as TL
from catspace.incubator.analyst.discovery import threat_measure as TM
from catspace.incubator.analyst.discovery import two_readings as TR
from catspace.io import paths

PHASES = (("early", 0, 19, 3), ("middle", 20, 49, 4), ("late", 50, 10_000, 3))


def choose(z: dict, lists: list) -> list:
    """The ten positions by the stated rule: [(phase, index of the row in the pickle)]."""
    order = {g: i for i, g in enumerate(z["games"])}
    by_game: dict = {}
    for j in range(len(lists)):
        by_game.setdefault(int(z["g"][j]), []).append(j)
    for g in by_game:
        by_game[g].sort(key=lambda j: int(z["ply"][j]))
    chosen, used = [], set()
    for phase, lo, hi, want in PHASES:
        got = 0
        for g in sorted(by_game, key=lambda g: order[g]):
            if got == want:
                break
            if g in used:
                continue
            for j in by_game[g]:
                if not lo <= int(z["ply"][j]) <= hi:
                    continue
                rows = lists[j]["mover"] + lists[j]["other"]
                if any(r["b_spread"] == 1 for r in rows):
                    chosen.append((phase, j))
                    used.add(g)
                    got += 1
                    break
    return chosen


def render(z: dict, j: int, game_id: str, moves: list) -> str:
    raw = z["rows"][j]
    ply = int(z["ply"][j])
    board = chess.Board()
    for mv in moves[:ply]:
        board.push_uci(mv)
    roles = z["names"]["roles"]
    kinds, places = z["names"]["kinds"], z["names"]["places"]
    mine = {(roles[o], kinds[k], places[p]) for o, k, p in raw["board"]}
    theirs = {("white" if pc.color else "black", chess.piece_name(pc.piece_type), chess.square_name(sq)) for sq, pc in board.piece_map().items()}
    agree = "the engine's board and the renderer's agree" if mine == theirs else f"THE ENGINE'S BOARD AND THE RENDERER'S DIFFER in {len(mine ^ theirs)} units"
    side = roles[raw["mover"]]
    lines = [f"game {game_id}, ply {ply} (the position after the first {ply} moves; {side} to move); moves so far: " + " ".join(moves[:ply]),
             f"set up: {board.fen()}   [{agree}]", str(board)]
    for col in ("white", "black"):
        lines.append(f"{col}: " + ", ".join(f"{k} {p}" for (c, k, p) in sorted(mine, key=lambda x: (x[1], x[2])) if c == col))
    return "\n".join(lines)


def row_text(r: dict, kinds_owner_word: str) -> str:
    st = "no stake (goal kind)" if r["stake"] is None else f"stake {r['stake']:+.3f}"
    by = ", ".join(f"{k} {p}" for k, p in r["by"]) or "no unit bears on it (a standing claim)"
    if r["response"]:
        first = r["response"][0]
        more = f" and {len(r['response']) - 1} more" if len(r["response"]) > 1 else ""
        resp = f"response: {first[0][0]}-{first[0][1]} ({first[1]}){more}" + (", found in a wider frame (" + "; ".join(f"{f['n_members']} units {f['class']}" for f in r["frames"]) + ")" if r["widened"] else "")
    else:
        resp = "response: none found"
    frames = "the frame that wins has " + str(r["win_frame"]["n_members"]) + " units" if len(r["frames"]) == 1 else "frames " + ", ".join(f"{f['n_members']} units ({f['class']}, {f['answers']} answers)" for f in r["frames"])
    return f"{r['kind']} {r['place']} [{r['class']}], {st}; by: {by}; {resp}; {frames}; the line ended: {r['ended']} (the read rests on: {r['basis']})"


def render_list(lst: dict, spread: float, roles: list, mover: int) -> str:
    out = [f"spread of the two readings at the position: {spread:.4f}"]
    for side, title in (("mover", f"THREATS AGAINST {roles[mover].upper()} (the side to move: their threats against us)"), ("other", f"THREATS AGAINST {roles[1 - mover].upper()} (the other side: ours against them)")):
        out.append(title)
        rows = lst[side]
        for b in range(1, 5):
            inb = [r for r in rows if r["b_spread"] == b]
            out.append(f"  band {b} by the spread, {TL.BAND_WORDS[b - 1]}: " + ("none" if not inb else ""))
            for r in sorted(inb, key=lambda r: -r["stake"]):
                rk = r["b_rank"]
                out.append(f"    - {row_text(r, side)} [band by rank: {rk if rk is not None else '-'}]")
        nos = [r for r in rows if r["b_spread"] is None]
        if nos:
            out.append(f"  no band (no stake: the unit is not a kind of the head's inputs): {len(nos)}")
            for r in nos:
                out.append(f"    - {row_text(r, side)}")
    return "\n".join(out)


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    out = TM.out_dir(a.out_dir)
    z = pickle.loads((out / "raw.pkl").read_bytes())
    lists = TM.build_lists(z)
    d = TR.load(lambda m: None)
    allg = d["allg"]
    picked = choose(z, lists)
    text = [f"THE TEN POSITIONS (rule in threat_print.py: held-out games of the 78.59 split in the md5 order of their ids; per phase, the first settled row of a game in the phase with a row in band 1 by the spread, one per game); chosen {len(picked)}; the lists come from the frame read with answers (threat_list.py); the stake is the change of the owner's chance of the result by the kinds reading of 78.69; the spread is that of the two readings; `us` is the side to move",
            "How to read a row: kind and place of the unit threatened [class], its stake; by: the units that bear on it (kind place); the response (the move, from-to, and its reason: a taking, the unit moves, a guard is added, a counter threat, a unit is put between), `none found` when no candidate move of the read answers; the frame(s) the claim was read in (the number of units in each); how the line ended (spec 4) and what the claim rests on. For the other side's units the response is what its owner would answer after we pass (the move is not available to us now)."]
    for k, (phase, j) in enumerate(picked, 1):
        gid = z["sample_ids"][z["games"].index(int(z["g"][j]))]
        text.append("")
        text.append(f"=== POSITION {k} of {len(picked)} ({phase} play) ===")
        text.append(render(z, j, gid, allg[gid]["moves"]))
        text.append(render_list(lists[j], float(z["spread"][j]), z["names"]["roles"], z["rows"][j]["mover"]))
    for phase, _lo, _hi, want in PHASES:
        got = sum(1 for p, _j in picked if p == phase)
        if got < want:
            text.append(f"NOTE: the {phase} phase has {got} of the {want} positions asked: the sample ran out of such rows")
    (out / "print.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
