"""PLAN 78.97 (SYSTEM_SPEC 5 the tactical reasoner's equal reading and the PROPOSED verified choice, 9 the oracle as a measure and labeller in sleep, 11 the judge against the board): THE TABLES OF "WHY A LOSING MOVE IS HELD EQUAL, AND WHAT THE VERIFIED CHOICE TAKES OF THE ROOM". A MEASURE: no player
is changed, no switch is turned on in play, no match is played, nothing is written to rules/learned or data/derived. This file never calls the oracle and reads none of its numbers; it reads what room_reads.py (our searches), room_picks.py (the picks) and room_lines.py (the sleep stage that asked the oracle)
wrote under artifacts/experiments/room_78_97/<tag>/, and the labels of 78.96 (read only). Every number is an OBS line (also in <tag>/obs.txt). One run each: directions, not verdicts. 90 pct bands by resampling games.

THE ROOM is the 78.96 positions where the equal moves differ in keeping the class (room_reads.room_ids); a move KEEPS the class when its class after it is not lower than before (prior_equal.keeps); a class-losing move is one that does not keep it.

STAGES (each a command).
  one     part one: by band and by class of the position, FIND and CONFIRM apart and together: the share of class-losing equal moves that LEAVE the equal set under r1 (a second run of the search that made the set: the noise control), five times the clock, twenty-five times it, and the play clock with the
          prunings off; the share of class-keeping moves that leave (the price); the positions where every class-losing move leaves and a class-keeping one stays (the room emptied); the share of class-losing moves that stay at all three (the stuck set); how many runs were cut or failed.
  two     part two: for the search's own move, the pattern pick and the prior of 78.96, with the verification off (arm r1) and on (arm on): the share of room positions where the move played keeps the class; how often the two arms play a different move; of those, losing to keeping, keeping to losing and the same; the same
          with the positions split by the verification's own state (replaced, or not: the difference without a replacement is the noise of two runs); the line that answers the debt of 78.93.
  three   part three: the stuck moves by band, tactical or slow (room_lines.py), the count by kind of the missing statement.
  cases   the twenty judged positions (cases97.json for ladder_print --room97): ten tactical and ten slow stuck moves, CONFIRM positions, one move a position, in the order of the hash.

INTERFACE (written before the bodies).
  group_masks(reads, labels) -> [(name, mask)]    the groups of the tables: all, the three bands, the classes of the position, band by class, each for find, confirm and both (pure).
  ratio(vals, game) -> (mean, lo, hi)             the mean of per-move or per-position values with a 90 pct band by games (prior_measure.share).
  Called by: the command line only; ladder_print.py reads cases97.json.

  .venv/bin/python -m catspace.incubator.analyst.discovery.room_measure one|two|three|cases|all --tag short|full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import prior_measure as PMS
from catspace.incubator.analyst.discovery import room_lines as RL
from catspace.incubator.analyst.discovery import room_picks as RP
from catspace.incubator.analyst.discovery import room_reads as RR
from catspace.incubator.analyst.discovery import symbols_measure as SM

BAND_NAMES = ("few", "mid", "many")
CLASS_WORDS = {1: "winning", 0: "neither", -1: "losing"}
CLOCKS = ("r1", "x5", "x25", "off")
CLOCK_WORDS = {"r1": "the play clock again (noise)", "x5": "5 times the clock", "x25": "25 times the clock", "off": "prunings off"}
SEED = 9797


def ratio(vals, game) -> tuple:
    return PMS.share(np.asarray(vals, float), np.asarray(game), SEED)


def fmt(t: tuple, plus: bool = False) -> str:
    return PMS.fmt(t, plus=plus)


def group_masks(reads: list, labels: dict) -> list:
    part = np.array([r["part"] for r in reads])
    band = np.array([r["band"] for r in reads])
    cls = np.array([r["cls"] for r in reads])
    out = []
    for pname, pm in (("find", part == "find"), ("confirm", part == "confirm"), ("both", np.ones(len(reads), bool))):
        out.append((f"{pname} all", pm))
        out += [(f"{pname} {b}", pm & (band == b)) for b in BAND_NAMES]
        out += [(f"{pname} class {CLASS_WORDS[c]}", pm & (cls == c)) for c in (1, 0)]
        out += [(f"{pname} {b} class {CLASS_WORDS[c]}", pm & (band == b) & (cls == c)) for b in BAND_NAMES for c in (1, 0)]
    return out


def load(tag: str) -> tuple:
    root = RR.out_root(tag)
    reads = RR.load(root)
    labels = {r["id"]: r for r in PE.load(RR.labels_root())}
    return root, reads, labels


# ------------------------------------------------------------------------------------------------------------------------------ part one

def cmd_one(tag: str, say) -> None:
    root, reads, labels = load(tag)
    game = np.array([r["ord"] for r in reads])
    say(f"the room: {len(reads)} positions of the 78.96 labels where the equal moves differ in keeping the class (FIND {sum(r['part'] == 'find' for r in reads)}, CONFIRM {sum(r['part'] == 'confirm' for r in reads)}); every move below is one of the labelled equal moves; "
        f"a move leaves a set when it is not among the moves that run holds equal; a run cut by the clock or failed has no answer and is left out of that setting's shares")
    for n in CLOCKS + ("on",):
        rows = [r["rows"][n] for r in reads]
        say(f"setting {n} ({CLOCK_WORDS.get(n, 'the verification of the choice on')}): runs cut {sum(x['cut'] for x in rows)} of {len(rows)}, failed {sum(bool(x['error']) for x in rows)}; mean ledger depth {np.mean([x['depth'] or 0 for x in rows]):.2f}; "
            f"mean equal moves held {np.mean([len(x['equals']) for x in rows]):.2f}; mean seconds on one thread {np.mean([x['seconds'] for x in rows]):.2f}")
    per = {}
    for i, r in enumerate(reads):
        lab = labels[r["id"]]
        per[i] = {n: RR.leaves(lab, r["rows"][n]) for n in CLOCKS}
    for name, m in group_masks(reads, labels):
        ix = np.flatnonzero(m)
        if len(ix) < 30:
            say(f"ONE {name}: {len(ix)} room positions: too few")
            continue
        parts = []
        for n in CLOCKS:
            lose_v, keep_v, g_l, g_k, emptied, ans = [], [], [], [], [], 0
            for i in ix:
                lv = per[i][n]
                if lv is None:
                    continue
                ans += 1
                lose_v += [float(x) for x in lv["lose"]]
                g_l += [game[i]] * len(lv["lose"])
                keep_v += [float(x) for x in lv["keep"]]
                g_k += [game[i]] * len(lv["keep"])
                emptied.append(float(all(lv["lose"]) and not all(lv["keep"])))
            if not lose_v:
                parts.append(f"{n}: no answer")
                continue
            parts.append(f"{n}: losing moves leave {fmt(ratio(lose_v, g_l))} pct of {len(lose_v)}, keeping moves leave {fmt(ratio(keep_v, g_k))} pct of {len(keep_v)}, room emptied {fmt(ratio(emptied, game[[i for i in ix if per[i][n] is not None]]))} pct of {ans} positions")
        st_v, st_g, st_pos = [], [], []
        for i in ix:
            r, lab = reads[i], labels[reads[i]["id"]]
            if not all(RR.answered(r["rows"].get(n)) for n in ("x5", "x25", "off")):
                continue
            stuck = set(RR.stuck_of(lab, r["rows"]))
            losing = [m for m, k in zip(lab["tied"], RR.keepvec(lab)) if not k]
            st_v += [float(m in stuck) for m in losing]
            st_g += [game[i]] * len(losing)
            st_pos.append((game[i], float(len(stuck) > 0)))
        stuck_txt = (f"STUCK (class-losing moves that stay at 5 times, 25 times and prunings off): {fmt(ratio(st_v, st_g))} pct of {len(st_v)} class-losing moves; positions with at least one stuck move "
                     f"{fmt(ratio([v for _g, v in st_pos], [g for g, _v in st_pos]))} pct of {len(st_pos)}") if st_v else "STUCK: no position read at all three"
        say(f"ONE {name} ({len(ix)} room positions): " + "; ".join(parts) + f"; {stuck_txt} (pct, 90 pct bands, games resampled)")


# ------------------------------------------------------------------------------------------------------------------------------ part two

def class_of_move(pid: str, move: str, labels: dict, classes: dict) -> "int | None":
    lab = labels[pid]
    if move in lab["tied"]:
        return lab["cm"][lab["tied"].index(move)]
    return classes.get(RL.key_of(pid, move))


def move_kept(pid: str, move: str, labels: dict, classes: dict) -> "bool | None":
    c = class_of_move(pid, move, labels, classes)
    return None if c is None else PE.keeps(labels[pid]["cls"], c)


def cmd_two(tag: str, say) -> None:
    root, reads, labels = load(tag)
    picks = RP.load(root)
    classes = RL.load_classes(root)
    game = np.array([r["ord"] for r in reads])
    kinds = ("first", "pattern", "prior")
    ok = [i for i, r in enumerate(reads) if picks.get(r["id"], {}).get("r1") and picks[r["id"]].get("on")]
    ok_set = set(ok)
    unk = 0
    kept = {}
    for k in kinds:
        for arm in RP.ARMS:
            v = np.full(len(reads), np.nan)
            for i in ok:
                r = reads[i]
                mv = picks[r["id"]][arm][k]
                kp = move_kept(r["id"], mv, labels, classes)
                if kp is None:
                    unk += 1
                else:
                    v[i] = float(kp)
            kept[(k, arm)] = v
    orig = np.array([float(PE.keeps(labels[r["id"]]["cls"], labels[r["id"]]["cm"][0])) for r in reads])
    say(f"the room: {len(reads)} positions, {len(ok)} read in both arms (r1: the verification off; on: the verification of the choice on, mode slot); the oracle's class of a move the labels do not hold: {len(classes)} moves asked in sleep, {unk} picked moves left without a class")
    for name, m in group_masks(reads, labels):
        ix = np.array([i for i in np.flatnonzero(m) if i in ok_set])
        if len(ix) < 30:
            say(f"TWO {name}: {len(ix)} room positions: too few")
            continue
        txt = [f"the 78.96 run's own first move keeps the class {fmt(ratio(orig[ix], game[ix]))}"]
        for k in kinds:
            a, b = kept[(k, "r1")], kept[(k, "on")]
            good = np.array([i for i in ix if not np.isnan(a[i]) and not np.isnan(b[i])])
            if len(good) < 30:
                txt.append(f"{k}: too few")
                continue
            d = SM.paired(b[good], a[good], game[good], SEED)
            txt.append(f"{k}: off {fmt(ratio(a[good], game[good]))}, on {fmt(ratio(b[good], game[good]))}, on less off {fmt(d, plus=True)} pct points")
        say(f"TWO {name} ({len(ix)} room positions): share where the move played keeps the class, " + "; ".join(txt) + " (pct, 90 pct bands, games resampled)")
    # replacements
    for k in kinds:
        diff = {"all": [], "replaced": [], "not replaced": []}
        for i in ok:
            r = reads[i]
            a, b = picks[r["id"]]["r1"][k], picks[r["id"]]["on"][k]
            st = r["rows"]["on"]["state"]
            diff["all"].append((i, a, b, st))
            diff["replaced" if st == 2 else "not replaced"].append((i, a, b, st))
        for grp, rows in diff.items():
            n = len(rows)
            ch = [(i, a, b) for i, a, b, _s in rows if a != b]
            lk = kl = same = unknown = 0
            for i, a, b in ch:
                ka, kb = move_kept(reads[i]["id"], a, labels, classes), move_kept(reads[i]["id"], b, labels, classes)
                if ka is None or kb is None:
                    unknown += 1
                elif (not ka) and kb:
                    lk += 1
                elif ka and not kb:
                    kl += 1
                else:
                    same += 1
            say(f"TWO REPLACEMENTS {k}, positions where the verification {grp}: {n} positions; the move played differs between the two arms in {len(ch)} ({100 * len(ch) / max(1, n):.1f} pct); of those, losing the class to keeping it {lk}, keeping it to losing it {kl}, "
                f"the same either way {same}, without a class {unknown}")
    states = {s: sum(1 for i in ok if reads[i]["rows"]["on"]["state"] == s) for s in range(5)}
    say(f"the verification's state over the {len(ok)} read positions: " + ", ".join(f"{RR.STATE_WORDS[s]} {n} ({100 * n / max(1, len(ok)):.1f} pct)" for s, n in states.items()))
    a, b = kept[("first", "r1")], kept[("first", "on")]
    rep = [i for i in ok if reads[i]["rows"]["on"]["state"] == 2]
    lk = sum(1 for i in rep if a[i] == 0 and b[i] == 1)
    kl = sum(1 for i in rep if a[i] == 1 and b[i] == 0)
    good = np.array([i for i in ok if not np.isnan(a[i]) and not np.isnan(b[i])])
    d = SM.paired(b[good], a[good], game[good], SEED)
    say(f"THE DEBT OF 78.93 (why a verified choice that replaces few choices moves the score), room positions only: the verification replaced the search's choice in {len(rep)} of {len(ok)} ({100 * len(rep) / max(1, len(ok)):.1f} pct); of those, with the class of the move off and the move on, "
        f"{lk} went from losing the class to keeping it and {kl} the other way; the share of room positions where the search's move keeps the class, on less off: {fmt(d, plus=True)} pct points; in the room the verified choice therefore works through the share of positions it replaces times the class the replaced move gains or loses, "
        f"and a replaced choice is one in {len(ok) / max(1, len(rep)):.0f} room positions")


# ------------------------------------------------------------------------------------------------------------------------------ part three

def cmd_three(tag: str, say) -> None:
    root, reads, labels = load(tag)
    lines = [r for r in RL.load_lines(root) if "error" not in r]
    bad = [r for r in RL.load_lines(root) if "error" in r]
    byid = {r["id"]: r for r in reads}
    games = {r["id"]: r["ord"] for r in reads}
    say(f"the stuck moves: {len(lines) + len(bad)} class-losing equal moves that stay at 5 times the clock, 25 times it and with the prunings off; the oracle's line asked for each (depth {RL.LO.FUTURE_DEPTH}); {len(bad)} the reading could not take")
    agree = [float(PE.keeps(labels[r["id"]]["cls"], r["cls_again"]) is False) for r in lines]
    say(f"the oracle asked again for the class after each stuck move: it again does not keep the class in {100 * np.mean(agree):.1f} pct of {len(agree)} (the class edge is noisy: the stored label and a second call can differ)")
    groups = [("all", lambda r: True)] + [(b, (lambda r, b=b: byid[r["id"]]["band"] == b)) for b in BAND_NAMES] + [(f"class {CLASS_WORDS[c]}", (lambda r, c=c: byid[r["id"]]["cls"] == c)) for c in (1, 0)]
    for name, f in groups:
        sub = [r for r in lines if f(r)]
        if len(sub) < 30:
            say(f"THREE {name}: {len(sub)} stuck moves: too few")
            continue
        g = [games[r["id"]] for r in sub]
        tac = [float(r["tactical"]) for r in sub]
        forced = sum(1 for r in sub if r["forced_in"] is not None and r["forced_in"] < 0 and r["tactical"])
        kinds = {k: sum(1 for r in sub if r["kind"] == k) for k in RL.KINDS}
        npos = len({r["id"] for r in sub})
        say(f"THREE {name}: {len(sub)} stuck moves in {npos} positions; TACTICAL (units lost within six plies in the measured worths, or a forced end against the mover) {fmt(ratio(tac, g))} pct ({int(sum(tac))}), SLOW {fmt(ratio([1 - t for t in tac], g))} pct ({len(sub) - int(sum(tac))}); "
            f"of the tactical, a forced end {forced}; the statement missing, by kind: " + ", ".join(f"{k} {n}" for k, n in kinds.items()) + " (pct, 90 pct bands, games resampled)")
    tac = [r for r in lines if r["tactical"]]
    if tac:
        say(f"THREE the tactical stuck moves, our reading of the other side's reply at the position after the move: threat list names it {sum(bool(r['seen']) for r in tac)} of {len(tac)}; line reading confirms it {sum(r['verdict'] == 'CONFIRMED' for r in tac)}; "
            f"the search at the play clock plays or holds it {sum(bool(r['held']) for r in tac)}")
        words = {}
        for r in tac:
            words[r["verdict"] or "none"] = words.get(r["verdict"] or "none", 0) + 1
        say("THREE the line reading's verdict on the other side's first reply, tactical stuck moves: " + ", ".join(f"{v} {n}" for v, n in sorted(words.items())))


# ------------------------------------------------------------------------------------------------------------------------------ the twenty positions

def cmd_cases(tag: str, say) -> None:
    root, reads, labels = load(tag)
    lines = [r for r in RL.load_lines(root) if "error" not in r]
    byid = {r["id"]: r for r in reads}
    games = {g["ord"]: g for g in FL.universe()}
    conf = [r for r in lines if byid[r["id"]]["part"] == "confirm"]
    conf.sort(key=lambda r: PE.order_key(byid[r["id"]]["ord"], byid[r["id"]]["ply"]))
    out, used = [], set()
    for want in (True, False):
        n = 0
        for r in conf:
            if r["tactical"] != want or r["id"] in used:
                continue
            used.add(r["id"])
            rd, lab = byid[r["id"]], labels[r["id"]]
            out.append({"group": "tactical" if want else "slow", "id": r["id"], "moves": games[lab["ord"]]["moves"][:lab["ply"]], "move": r["move"], "cls": lab["cls"], "cls_after": lab["cm"][lab["tied"].index(r["move"])], "cls_again": r["cls_again"],
                        "units": lab["units"], "band": lab["band"], "best": lab["best"], "best_cls": lab["best_cls"], "tied": lab["tied"], "cm": lab["cm"], "pv": r["pv"], "net": r["net"], "forced_in": r["forced_in"],
                        "reply": r["reply"], "seen": r["seen"], "verdict": r["verdict"], "held": r["held"], "kind": r["kind"], "first": lab["first"],
                        "sets": {n_: sorted(set(rd["rows"][n_]["equals"]) | {rd["rows"][n_]["first"]}) for n_ in ("x5", "x25", "off")}, "depths": {n_: rd["rows"][n_]["depth"] for n_ in RR.SETTINGS}})
            n += 1
            if n == 10:
                break
    (root / "cases97.json").write_text(json.dumps(out))
    say(f"CASES: {sum(c['group'] == 'tactical' for c in out)} tactical and {sum(c['group'] == 'slow' for c in out)} slow stuck moves (CONFIRM positions, one move a position, the order of the hash); written to cases97.json for ladder_print --room97")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["one", "two", "three", "cases", "all"])
    ap.add_argument("--tag", default="short")
    a = ap.parse_args(argv)
    say = RR.Say(RR.out_root(a.tag), a.stage)
    for st, f in (("one", cmd_one), ("two", cmd_two), ("three", cmd_three), ("cases", cmd_cases)):
        if a.stage in (st, "all"):
            f(a.tag, RR.Say(RR.out_root(a.tag), st))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
