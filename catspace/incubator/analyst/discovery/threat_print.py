"""PLAN 78.70 E: THE PRINT of the threat list for the owner to read. A printing script, NOT a layer, sheet, binding or measure file: it is the one place that renders a position with the game's own renderer (the
python-chess board, the library the rule kernel's own spelling comes from: names.py) so that a reader can set the position up, and prints the rows the list holds (threat_list.py) with the engine's own names for kinds and
places (no mapping file names any of them: the mapping file of spec 2 holds names of mined concepts only, and no concept is printed here).

THE CHOICE OF THE TEN POSITIONS (a stated rule, not by hand): the held-out games of the pickle of `threat_measure read`, in its order (the md5 order of their ids); the plies of the game's settled rows are cut into
EARLY (ply under 20), MIDDLE (20 to 49) and LATE (50 on); for each phase in turn the games are scanned in order and the first settled row of the game inside the phase that has at least one row in band 1 (high, no
response) by the spread of the readings, on either side, is taken, one position per game and no game twice; 3 early, 4 middle, 3 late. A phase that runs out of such rows is left short and the print says so.

PLAN 78.72 F: with `--concrete` the print gives, for the SAME ten positions (and ten new ones by the same rule, the games after those the ten used), each position's 78.70 list and its concrete list
(threat_concrete.py: every row read as a line to a settled end with a verdict; exchanges apart; the goal rows; each kept response a line read the same way), and the time the lists took. Units and squares are named
here only, in the words of the game's own renderer; the concrete list itself names none.

  .venv/bin/python -m catspace.incubator.analyst.discovery.threat_print [--out-dir D] [--concrete]
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


def choose(z: dict, lists: list, skip: "set | None" = None) -> list:
    """The ten positions by the stated rule: [(phase, index of the row in the pickle)]. `skip`: game numbers not to use (the ten new positions of 78.72 are chosen by the same rule in the games the first ten did not use)."""
    order = {g: i for i, g in enumerate(z["games"])}
    by_game: dict = {}
    for j in range(len(lists)):
        by_game.setdefault(int(z["g"][j]), []).append(j)
    for g in by_game:
        by_game[g].sort(key=lambda j: int(z["ply"][j]))
    chosen, used = [], set(skip or ())
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


# ------------------------------------------------------------------------------------------------------------------------------ plan 78.72: the concrete list

def letter_names() -> dict:
    """The game's word for a kind as the line search words it (one letter) -> the kind's full name (the inverse of the record's mapping)."""
    from catspace.incubator.analyst.discovery import odds_measure as OM
    return {v: k for k, v in OM.kernel_names().items()}


def ledger_text(row: dict, nm: dict) -> str:
    o = row.get("outcome")
    if not o:
        return "no ledger"
    lost = ", ".join(nm.get(k, k) for k in o["taker_lost"]) or "nothing"
    won = ", ".join(nm.get(k, k) for k in o["taker_won"]) or "nothing"
    return f"the taker takes {won} and loses {lost}"


def line_text(names: list) -> str:
    return " ".join(names) if names else "(none)"


def concrete_row_text(r: dict, nm: dict) -> str:
    if r.get("goal"):
        rs = r["response"]
        resp = ("responses (after each the forcing read finds no forced end): " + ", ".join(x["names"][0] for x in rs[:6]) + (f" and {len(rs) - 6} more" if len(rs) > 6 else "")) if rs else "responses: none found"
        return f"THE GAME ITSELF: a forced end of the game for the taker, {r['verdict']}: {line_text(r['line'])}; stake: the highest; {resp} ({r['resp_read']} read, {r['resp_unread']} unread); the line ended: {r['ended']} ({r['reason']})"
    st = "no stake (goal kind)" if r["stake"] is None else f"stake {r['stake']:+.3f}"
    two = f" THREAT IN TWO MOVES, made by the move {r['line'][0]} (the other side's best replies are read, the taker still ends ahead);" if r.get("two_moves") else ""
    head = f"{r['verdict']}: {r['kind']} {r['place']} [{r['class']}], {st};{two} line: {line_text(r['line'])}; {ledger_text(r, nm)} (net for the taker {r['net']} in the record's worths, resolution {r.get('resolution', 0):.0f}); the line ended: {r['ended']} ({r['reason']})"
    if r.get("deferred"):
        return head + "; response: none needed yet (the answer comes after the taker's first move)"
    if r["verdict"] == "UNREAD":
        return head + "; response: not read (suspected only); the frame read named " + str(len(r.get("response_ix") or [])) + " candidates"
    if r["verdict"] == "CONFIRMED":
        rs = r["response"]
        if rs:
            parts = []
            for x in rs[:3]:
                lost = ", ".join(nm.get(k, k) for k in x["taker_lost"]) or "nothing"
                won = ", ".join(nm.get(k, k) for k in x["taker_won"]) or "nothing"
                parts.append(f"{x['names'][0]} ({x['how']}; {x['why']}; the line {line_text(x['names'])}: the taker takes {won}, loses {lost})")
            more = f" and {len(rs) - 3} more" if len(rs) > 3 else ""
            return head + f"; responses ({r['resp_read']} read, {r['resp_read'] - len(rs)} dropped, {r['resp_unread']} unread): " + "; ".join(parts) + more
        return head + f"; response: none found ({r['resp_read']} read, all dropped, {r['resp_unread']} unread)"
    return head


def render_concrete(c: dict, spread: float, roles: list, mover: int, nm: dict) -> str:
    out = []
    for side, title in (("mover", f"THREATS AGAINST {roles[mover].upper()} (the side to move: their threats against us)"), ("other", f"THREATS AGAINST {roles[1 - mover].upper()} (the other side: ours against them)")):
        out.append(title)
        rows = c[side]
        for b in range(1, 5):
            inb = [r for r in rows if r.get("b_spread") == b]
            out.append(f"  band {b} by the spread, {TL.BAND_WORDS[b - 1]}: " + ("none" if not inb else ""))
            for r in inb:
                out.append(f"    - {concrete_row_text(r, nm)} [band by rank: {r.get('b_rank') if r.get('b_rank') is not None else '-'}]")
        nos = [r for r in rows if r.get("b_spread") is None]
        if nos:
            out.append(f"  no band (no stake): {len(nos)}")
            for r in nos:
                out.append(f"    - {concrete_row_text(r, nm)}")
    if c["exchange"]:
        out.append("EXCHANGES (units are traded and neither side ends ahead; printed apart, not threats)")
        for r in c["exchange"]:
            owner = roles[r["owner"]]
            out.append(f"    - {owner}'s {r['kind']} {r['place']} [{r['class']}]: line {line_text(r['line'])}; {ledger_text(r, nm)}; net for the taker {r['net']} within the resolution {r.get('resolution', 0):.0f}; ended: {r['ended']} ({r['reason']})")
    out.append("DROPPED (counted): " + ("; ".join(f"{k}: {v}" for k, v in sorted(c["dropped"].items())) or "none") + f"; responses read {c['resp_read']}, dropped {c['resp_dropped']}, sweeps of every legal move {c['resp_sweeps']}; rows UNREAD {c['unread_rows']}")
    return "\n".join(out)


def paths_dir_78_70() -> Path:
    return Path(paths.experiment("threats_78_70"))


def main72(a) -> int:
    import time

    from catspace.incubator.analyst.discovery import threat_concrete as TC
    out = TM.out_dir(a.out_dir or paths.experiment("threats_78_72"))
    z = pickle.loads((paths_dir_78_70() / "raw.pkl").read_bytes())
    lists = TM.build_lists(z)
    d = TR.load(lambda m: None)
    allg = d["allg"]
    first = choose(z, lists)
    used = {int(z["g"][j]) for _p, j in first}
    second = choose(z, lists, used)
    picked = first + second
    rec = TC.Record.load()
    lines1 = TC.Lines(rec, threads=1)
    nm = letter_names()
    reader = TL.ThreatReader(1)
    text = [f"THE POSITIONS OF PLAN 78.72 F: positions 1 to {len(first)} are the SAME ten as in artifacts/experiments/threats_78_70/print.txt (the games and plies of the rule in threat_print.py); positions {len(first) + 1} to {len(picked)} are ten new ones by the same rule in the games the first ten did not use. Each position: the board, the OLD list (plan 78.70, the frame read's classes) and the NEW list (plan 78.72, every row a line read to a settled end).",
            "How to read a NEW row: VERDICT: unit [the frame read's class], stake (the change in the owner's chance of the result by the kinds reading); line: the moves read (the taker's first); the ledger of the line (what the taker takes and loses, in the measured record's worths: net, and the resolution: the gap the record can tell apart); how the line ended in the words of spec 4; the responses (each a line read the same way, kept only if after it the taker no longer ends ahead; the count read and dropped). CONFIRMED: the taker ends ahead; EXCHANGE: units are traded, neither ends ahead (printed apart); UNREAD: a fuse cut the line (suspected only); a REFUTED claim is dropped and counted.",
            f"the judge: the measured record {rec.source}; line depth {lines1.depth} plies after the first move, node fuse {lines1.fuse} per line; the forcing read for the goal: {lines1.end_plies} moves of the side, node fuse {lines1.end_fuse}"]
    times = []
    for k, (phase, j) in enumerate(picked, 1):
        gid = z["sample_ids"][z["games"].index(int(z["g"][j]))]
        moves = allg[gid]["moves"]
        ply = int(z["ply"][j])
        t0 = time.perf_counter()
        reader.read([(None, moves)], [[ply]])
        t_frame = time.perf_counter() - t0
        item = {"start": None, "moves": moves[:ply], "raw": z["rows"][j], "names": z["names"], "list": lists[j]}
        t1 = time.perf_counter()
        c = TC.concrete_lists([item], lines1)[0]
        t_conc = time.perf_counter() - t1
        times.append((t_frame, t_conc))
        both = c["mover"] + c["other"]
        spread = float(z["spread"][j])
        rk = TL.rank_high_of(both)
        for r in both:
            r["b_spread"] = TL.band_of(r, spread, None, "spread")
            r["b_rank"] = TL.band_of(r, None, rk.get(id(r), False), "rank")
        text.append("")
        text.append(f"=== POSITION {k} of {len(picked)} ({phase} play){' [same as position ' + str(k) + ' of the 78.70 print]' if k <= len(first) else ' [new]'} ===")
        text.append(render(z, j, gid, moves))
        text.append("--- OLD LIST (plan 78.70) ---")
        text.append(render_list(lists[j], spread, z["names"]["roles"], z["rows"][j]["mover"]))
        text.append(f"--- NEW LIST (plan 78.72); the spread of the two readings at the position: {spread:.4f}; time: the frame read {t_frame:.2f} s, the concrete read {t_conc:.2f} s, one thread ---")
        text.append(render_concrete(c, spread, z["names"]["roles"], z["rows"][j]["mover"], nm))
    ft = [x[0] for x in times]
    ct = [x[1] for x in times]
    text.append("")
    text.append(f"OBS print72: time per position over the {len(times)} printed (one thread each): the frame read median {sorted(ft)[len(ft) // 2]:.2f} s, the concrete read median {sorted(ct)[len(ct) // 2]:.2f} s, slowest {max(ct):.2f} s, mean {sum(ct) / len(ct):.2f} s")
    (out / "print.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    return 0


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--concrete", action="store_true")
    a = ap.parse_args(argv)
    if a.concrete:
        return main72(a)
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
