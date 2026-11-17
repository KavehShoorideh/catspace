"""PLAN 78.72 A to D: THE THREAT LIST MADE CONCRETE. SYSTEM_SPEC section 8 "The threat list" under section 4 (where a concrete answer can be reached it is reached; how a line ends) and section 3 (the exchange; a sacrifice needs a reason). The piece is the
FRAME READ'S view for the wake, one step on from threat_list.py (plan 78.70): every row of that list is read as a LINE to a settled end, with a verdict, and so is every response. A sandbox: no player is changed, nothing
is written to rules/learned or data/derived. Game-free: a kind, a place or a role is a word or a number read from the engine at run time; no unit, place or kind is named here. No engine evaluation, no labeller, no external
model, no assumed worth: the only worths are the MEASURED record of the kinds (the odds games of 78.61, generation 2), and the record is also the judge's own measure of how far apart two ledgers must be to be told apart.

INTERFACE (what this module takes, what it returns, who calls it).
  Record.load(path)             the measured record of the kinds: its integer worths by the game's word for the kind (the judge of the line search, as 78.61 plays it), and the half width of the record's own confidence
                                interval of each kind (its RESOLUTION: the least gap between two ledgers that the record can tell apart).
  Lines(record, ...)            one per process: `ask(games, queries)` is the Rust `cs_engine.read_lines` (rust/engine/tactics/src/lines.rs: a line is read to a settled end by the ledger search with the record as judge;
                                the forced end of the game by the forcing read). It reads nothing else of the engine and keeps the wall time and node counts it spent (`cost`).
  verdict_of(read, record)      the verdict of one line read FOR THE SIDE THAT TAKES: CONFIRMED (it ends ahead), EXCHANGE (units are traded and neither ends ahead), REFUTED (it ends behind), UNREAD (the fuse cut the line:
                                printed as suspected, no forced claim is made from a cut line), with the units each side is left with and the reason the line ended in the words of spec 4. "Ahead" is the record's:
                                the sum of the worths of what the taker took less what it lost is above the resolution of the units in that ledger (the half widths added in quadrature), or the game itself ends in its favour.
  response_of(after, passed, record, how, source)
                                one candidate response read two ways (the taking of the unit after it, the whole-board line of the move) and judged: KEPT, STILL WON, WORSE THAN PASSING or UNREAD.
  concrete_lists(items, lines)  takes, per position, the list of plan 78.70 (threat_list.threat_rows, with its stake) and the position (start text or None, the moves played), and returns per position
                                {"mover": rows, "other": rows, "exchange": rows, "goal": rows, "dropped": counts, "cost": seconds}: the rows against each side that are CONFIRMED or UNREAD (a threat), the EXCHANGE rows apart,
                                and the rows of the goal (a forced end of the game within the forcing read's reach, for either side, with its line). Each row keeps the fields of threat_list and gains
                                line, outcome, verdict, net, ended, lead (the move that makes a threat in two moves) and `response`: the kept responses, each a line read the same way (kept only if after it the taker is no
                                longer ahead), ordered by what each side is left with, the count of those read and dropped in `resp_read`. `band_of` of threat_list gives the band as before (a row is high or low by its stake; with
                                a response or not by the KEPT responses); a row of the goal has the highest stake.
  same_read_set(a, b)           THE LIST CARRIED FROM MOVE TO MOVE (E): the read set of a claim (threat_list.read_set_of) is the same at two positions. A claim whose read set is unchanged at the position two plies on is
                                kept without being asked again; one whose read set changed is asked again. What carrying saves and how often the carried verdict is the one a fresh read gives are measured in
                                threat_concrete_measure.py (`carry`); the wake does not call it yet.
  Called by: threat_concrete_measure.py (the checks and the carry) and threat_print.py (the print) now; the wake's choice of move and the sleep's ladder later. No player calls it.

THE LINE OF A CLAIM. The taker is the side that does not own the unit. For a unit of the side to move the taker moves after the side to move passes once (what is threatened if nothing is done: the pass of the game; where the
game gives none the row is UNREAD with that reason); for a unit of the other side the taker is the side to move. The taking moves are the legal moves of the taker that take a unit on the unit's place. Each is read to a settled end
(the ledger search with the record as judge, best replies for both sides, the captures resolved at the horizon, a node fuse per line) and the best of them for the taker is the line of the row. A claim with no taking move: if no
unit bears on it, it is a row only if one quiet move of the taker makes a taking stand (a THREAT IN TWO MOVES, printed with that move); otherwise it is dropped and counted.
THE RESPONSES. The candidates are the answers the frame read named (threat_list's `response_ix`); each is read as a line the same way, from the side that owns the unit, and kept only if the taker is no longer ahead after it. Every
legal move of the owner is also read the same way (the sweep, at a shorter horizon: the frame read stops at its first sound answers, so its list is not complete) and the row says where each response came from. Responses are ordered by what the taker is left with, the least first. The count read and dropped is printed.
THE GOAL (D). The forced end of the game within the forcing read's reach (a named depth in moves of the side and a node fuse), for either side, the goal as the rules give it (the game's own outcome): a row of the highest stake with its
line, and its responses (the moves after which the forcing read finds none), for the side that is forced against.
HOW A LINE ENDED (spec 4), as this build maps it (a choice of this build, not stated by the spec): CONCRETE: the game itself ends in the line; QUIET: the settled end, the captures resolved at the horizon of the line search (its depth
in plies is printed); OUT OF BUDGET: the fuse cut the line (UNREAD). LOST BEYOND DOUBT is a verdict of the readings of section 6 and is not produced here.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import threat_list as TL
from catspace.io import paths

RECORD = ("odds_78_61", "full", "gen2", "record.json")           # the measured record of the kinds the lines are judged by (generation 2 of the odds games of 78.61)
DEPTH = 6                                                       # plies after the first move: the horizon of a line (a named fuse of the read; the captures are resolved beyond it)
FUSE = 40000                                                    # the node fuse of ONE line: a line it cuts is UNREAD
SWEEP_DEPTH = 4                                                 # the sweep of every legal move reads each at a shorter horizon (and a smaller fuse) than the lines of the rows: its cost is the cost of the list
SWEEP_FUSE = 12000
END_PLIES = 7                                                   # plies within which the end of the game is looked for (the search that counts nothing; the exact forcing read confirms the first move)
END_FUSE = 1500000
VERDICTS = ("CONFIRMED", "EXCHANGE", "REFUTED", "UNREAD")
WHY = ("a taking", "the unit moves", "a guard is added", "a counter threat", "a unit is put between")


class Record:
    """The measured record of the kinds and what it can tell apart."""

    def __init__(self, worth: dict, half: dict, source: str):
        self.worth, self.half, self.source = worth, half, source

    @classmethod
    def load(cls, path: "str | Path | None" = None) -> "Record":
        p = Path(path) if path else Path(paths.experiment(*RECORD))
        rec = json.loads(p.read_text())
        head = rec["heads"]["odds_effect"]
        words = OM.kernel_names()                                # the record's name of a kind -> the game's word for it
        worth = {words[k]: int(v) for k, v in head["kind_values"].items()}
        half = {words[k]: 500.0 * (head["per_kind"][k]["one"]["hi"] - head["per_kind"][k]["one"]["lo"]) for k in head["kind_values"] if k in head["per_kind"]}
        return cls(worth, half, str(p))

    def resolution(self, kinds: list) -> float:
        """The least gap the record can tell apart in a ledger of these units: the half widths of their kinds added in quadrature (in the worths' own integers)."""
        return math.sqrt(sum(self.half.get(k, 0.0) ** 2 for k in kinds))


class Lines:
    """One per process: the Rust line read with the record as judge."""

    def __init__(self, record: Record, threads: int = 1, depth: int = DEPTH, fuse: int = FUSE, end_plies: int = END_PLIES, end_fuse: int = END_FUSE, variant: "str | None" = None,
                 sweep_depth: int = SWEEP_DEPTH, sweep_fuse: int = SWEEP_FUSE):
        from catspace.incubator.analyst.discovery import human_heads_data as HD
        self.record, self.threads, self.depth, self.fuse, self.end_plies, self.end_fuse = record, threads, depth, fuse, end_plies, end_fuse
        self.sweep_depth, self.sweep_fuse = sweep_depth, sweep_fuse                          # the horizon and the fuse of the sweep of every legal move (PLAN 78.73 C reads it at the rows' depth)
        self.variant = variant or HD.VARIANT
        self.cost = {"wall": 0.0, "queries": 0, "calls": 0}

    def ask(self, games: list, queries: list, depth: "int | None" = None, fuse: "int | None" = None) -> list:
        if not queries:
            return []
        t0 = time.perf_counter()
        res = CE.read_lines(self.variant, games, queries, self.record.worth, depth or self.depth, fuse or self.fuse, self.end_plies, self.end_fuse, self.threads)
        self.cost["wall"] += time.perf_counter() - t0
        self.cost["queries"] += len(queries)
        self.cost["calls"] += 1
        return res


# ------------------------------------------------------------------------------------------------------------------------------ the verdict of a line

def ledger_of(read: dict) -> "tuple[list, list]":
    """(the units the taker lost, the units the taker took) along the line, as the game's words for the kinds."""
    lost = [k for mine, k in read["events"] if mine]
    won = [k for mine, k in read["events"] if not mine]
    return lost, won


def cancel_like(lost: list, won: list) -> "tuple[list, list, int]":
    """PLAN 78.73 A: units of the same kind taken and lost cancel EXACTLY (a unit lost against a unit of the same kind taken is no gain to either side, whatever the record says of the kind): returns (the kinds the taker
    is left having lost, the kinds it is left having taken, the number of pairs cancelled)."""
    left_won = list(won)
    left_lost = []
    pairs = 0
    for k in lost:
        if k in left_won:
            left_won.remove(k)
            pairs += 1
        else:
            left_lost.append(k)
    return left_lost, left_won, pairs


def verdict_of(read: dict, record: Record) -> dict:
    """The verdict of one line read for the side that takes (the line was read with that side as `me`): see the module doc."""
    lost, won = ledger_of(read)
    net_search = int(read["value"])
    left_lost, left_won, pairs = cancel_like(lost, won)
    net = sum(record.worth.get(k, 0) for k in left_won) - sum(record.worth.get(k, 0) for k in left_lost)      # PLAN 78.73 A: the net of the kinds LEFT after like for like cancels
    res = record.resolution(left_lost + left_won)                                                           # and the resolution is built only from them (none left: nothing to tell apart)
    base = {"names": read["names"][:read["settled"]], "full_names": read["names"], "taker_lost": lost, "taker_won": won, "left_lost": left_lost, "left_won": left_won, "cancelled": pairs, "net": net,
            "net_search": net_search, "resolution": res, "depth": read["depth"], "end": read["end"], "nodes": read["nodes"], "pv_disagrees": net != net_search}
    if read["cut"]:
        return dict(base, verdict="UNREAD", ended=TL.ENDED[3], reason="out of budget: the fuse cut the line; no forced claim is made from it")
    if read["end"] == 1:
        return dict(base, verdict="CONFIRMED", ended=TL.ENDED[0], reason="the end of the game, won by the taker, decides it")
    if read["end"] == 2:
        return dict(base, verdict="REFUTED", ended=TL.ENDED[0], reason="the end of the game, lost by the taker, decides it")
    why = f"the settled end: no capture is left to either side after {read['depth']} plies" + (f"; {pairs} like for like cancelled" if pairs else "")
    if net > res:
        v = "CONFIRMED"
    elif net < -res:
        v = "REFUTED"
    else:
        v = "EXCHANGE" if ((lost and won) or (left_lost and left_won)) else ("CONFIRMED" if left_won else "REFUTED")
    return dict(base, verdict=v, ended=TL.ENDED[1], reason=why)


def best_taking(reads: list, record: Record) -> "dict | None":
    """The line of a row from its taking lines: the best for the taker; UNREAD when a line was cut and no line that was read stands CONFIRMED."""
    if not reads:
        return None
    vs = [verdict_of(r, record) for r in reads]
    for v, r in zip(vs, reads):
        v["first"] = r["first"]
        v["lead"] = r.get("lead")
        v["prefix"] = r.get("prefix", [])
    for v in vs:
        v["nodes_all"] = sum(r["nodes"] for r in reads)
    done = [v for v in vs if v["verdict"] != "UNREAD"]
    conf = [v for v in done if v["verdict"] == "CONFIRMED"]
    cut = [v for v in vs if v["verdict"] == "UNREAD"]
    if conf:
        return max(conf, key=lambda v: v["net"])
    if cut:
        return max(cut, key=lambda v: v["net"])
    return max(done, key=lambda v: v["net"])


def outcome_words(v: dict, kinds_word) -> dict:
    """What each side is left with after a line: the units the taker lost and the units it took, by the caller's word for each kind."""
    return {"taker_lost": [kinds_word(k) for k in v["taker_lost"]], "taker_won": [kinds_word(k) for k in v["taker_won"]]}


# ------------------------------------------------------------------------------------------------------------------------------ the concrete list

def _roles_of(row: dict, mover: int) -> dict:
    """For a row: the side that takes and the side that answers, and where each moves first: `taker_p` / `owner_p` = 0 when the side moves at the position itself, 1 after the side to move passes once."""
    owner_is_mover = row["owner"] == mover
    return {"taker_p": 1 if owner_is_mover else 0, "owner_p": 0 if owner_is_mover else 1}


def _kept(v: dict) -> bool:
    return v["state"] == "KEPT"


def response_of(after: dict, passed: "dict | None", record: Record, how: str, source: str, row_net: "int | None" = None) -> dict:
    """One candidate response read two ways (rust `read_after`): the taking lines of the unit AFTER it (is the unit still won?) and the whole-board line of the move (what each side is left with). KEPT only if
    after it the taker's best taking of the unit is no longer CONFIRMED (or none is legal) AND nothing larger is lost by it: the taker's whole-board line is not better for it than what it could count on if the owner
    passed (`passed`, the best line of the taker at the position after a pass) NOR than what it takes by the unit (PLAN 78.73 B, `row_net`: the response may not lose more than the threat would take). STILL WON: the
    unit is still taken ahead. WORSE THAN PASSING / COSTS MORE THAN THE THREAT: it saves the unit and loses more elsewhere. UNREAD: a fuse cut a line.
    PLAN 78.73 B: the response carries ITS OWN END, what the responder takes and loses in its line after like for like cancels (`resp_takes`, `resp_loses`; `loses_nothing`): the nets are the ledger's, in the record's worths."""
    whole = after["whole"]
    wv = verdict_of(whole, record)
    tk = best_taking(after["takes"], record) if after["takes"] else None
    pnet = verdict_of(passed, record)["net"] if passed is not None else None
    if whole["cut"] or (tk is not None and tk["verdict"] == "UNREAD") or passed is None or passed["cut"]:
        state, why = "UNREAD", "a fuse cut a line: the response is not judged"
    elif whole["end"] == 1 or (tk is not None and tk["verdict"] == "CONFIRMED"):
        state, why = "STILL WON", "after it the taker still ends ahead by taking the unit" if tk is not None else "after it the game is lost"
    elif whole["end"] != 2 and wv["net"] > pnet:
        state, why = "WORSE THAN PASSING", "it saves the unit and the taker ends further ahead elsewhere than if the owner had passed"
    elif whole["end"] != 2 and row_net is not None and wv["net"] > row_net:
        state, why = "COSTS MORE THAN THE THREAT", "it saves the unit and loses more than the taking of the unit would have cost"
    else:
        state, why = "KEPT", ("no taking of the unit is legal after it" if tk is None else "the taking of the unit is " + tk["verdict"])
    return {"first": whole["first"], "names": whole["names"], "net": wv["net"], "taker_lost": wv["taker_lost"], "taker_won": wv["taker_won"], "resp_takes": wv["left_lost"], "resp_loses": wv["left_won"], "cancelled": wv["cancelled"],
            "loses_nothing": not wv["left_won"], "ended": wv["ended"], "end_reason": wv["reason"], "depth": whole["depth"], "settled": whole["settled"], "end": whole["end"],
            "nodes": whole["nodes"] + sum(x["nodes"] for x in after["takes"]), "state": state, "why": why, "how": how, "source": source, "after": (None if tk is None else {"verdict": tk["verdict"], "names": tk["names"], "net": tk["net"]})}


def _fill(row: dict, best: dict, two: bool) -> None:
    row.update(verdict=best["verdict"], ended=best["ended"], reason=best["reason"], net=best["net"], line=(best["prefix"] + best["names"]) if two else best["names"], lead=best["lead"] if two else None,
               outcome={"taker_lost": best["taker_lost"], "taker_won": best["taker_won"], "left_lost": best["left_lost"], "left_won": best["left_won"], "cancelled": best["cancelled"]}, resolution=best["resolution"], depth=best["depth"], first=best["first"], pv_disagrees=best["pv_disagrees"], two_moves=two,
               nodes=best["nodes_all"])


def concrete_lists(items: list, lines: Lines, record: "Record | None" = None) -> list:
    """The concrete list of every item: {"start", "moves", "raw", "names", "list" (threat_list.threat_rows)} -> see the module doc. Three rounds of Rust calls over all the positions at once."""
    rec = record or lines.record
    games = [(it["start"], list(it["moves"])) for it in items]
    out = [{"mover": [], "other": [], "exchange": [], "goal": [], "dropped": {}, "dropped_rows": [], "resp_read": 0, "resp_dropped": 0, "resp_sweeps": 0, "unread_rows": 0} for _ in items]
    flat = []                                                    # every row of every list, a copy the reading fills in
    for i, it in enumerate(items):
        for side in ("mover", "other"):
            for r in it["list"][side]:
                row = dict(r)
                row.update(_i=i, _resp=[], _pending=[], verdict=None)
                flat.append(row)

    def drop(i: int, why: str, row: "dict | None" = None, code: "str | None" = None) -> None:
        out[i]["dropped"][why] = out[i]["dropped"].get(why, 0) + 1
        if row is not None:                                      # PLAN 78.71: a dropped claim is kept as a row (with the code of why and the line read, if any): the source of the goals that do not work yet
            out[i]["dropped_rows"].append(dict({k: v for k, v in row.items() if not k.startswith("_")}, why=code, _i=i))

    # ---- round 1: the taking lines of every row and the forced end of the game for each side
    q1, plan1 = [], []
    for row in flat:
        i = row["_i"]
        row["_ro"] = _roles_of(row, items[i]["raw"]["mover"])
        plan1.append(("take", row, len(q1)))
        q1.append((i, 0, row["_ro"]["taker_p"], 1, row["place_ix"], 0, 0))
    for i in range(len(items)):
        for p in (0, 1):
            plan1.append(("goal", (i, p), len(q1)))
            q1.append((i, 4, p, 1, 0, 0, 0))
    r1 = lines.ask(games, q1)
    goal_hits = []
    for kind, a, qi in plan1:
        res = r1[qi]
        if kind == "goal":
            i, p = a
            if res["ok"] and res.get("forced") in ("proved", "suspected"):
                goal_hits.append((i, p, res))
            elif res["ok"] and res.get("forced") == "gap":
                drop(i, "a forced end of the game read out of budget (a gap, not a refutation)")
            continue
        row = a
        if not res["ok"]:
            row.update(verdict="UNREAD", ended=TL.ENDED[3], reason="the side to move must answer: the game gives no pass, so what the other side would take cannot be read", net=None, line=[], lead=None, outcome=None, two_moves=False)
            continue
        best = best_taking(res["reads"], rec)
        if best is None:
            row["_notake"] = True
        else:
            _fill(row, best, False)
    # ---- round 2: threats in two moves (no taking move), the responses of CONFIRMED rows, the responses to a forced end of the game
    q2, plan2, best_of = [], [], {}
    for row in flat:
        i, ro = row["_i"], row["_ro"]
        if row.get("_notake"):
            plan2.append(("two", row, len(q2)))
            q2.append((i, 3, ro["taker_p"], 1, row["place_ix"], 0, 0))
        elif row["verdict"] == "CONFIRMED" and not row.get("two_moves"):
            if (i, ro["taker_p"]) not in best_of:                  # what the taker can count on if the owner does nothing: asked once per position and side
                best_of[(i, ro["taker_p"])] = len(q2)
                plan2.append(("best", (i, ro["taker_p"]), len(q2)))
                q2.append((i, 8, ro["taker_p"], 1, 0, 0, 0))
            for (fr, to, how, _um) in row["response_ix"]:
                plan2.append(("resp", (row, how), len(q2)))
                q2.append((i, 6, ro["owner_p"], 0, fr, to, row["place_ix"]))
    for i, p, res in goal_hits:
        plan2.append(("goalresp", (i, p, res), len(q2)))
        q2.append((i, 5, 1 - p, 1, 0, 0, 0))
    r2 = lines.ask(games, q2)
    goal_resp: dict = {}
    passed_value: dict = {}
    for kind, a, qi in plan2:
        res = r2[qi]
        if kind == "two":
            row = a
            best = best_taking(res["reads"], rec) if res["ok"] else None
            if best is not None and best["verdict"] in ("CONFIRMED", "UNREAD"):
                _fill(row, best, True)
            else:
                if best is not None:                             # PLAN 78.71: a quiet move makes a taking stand and the line does not end ahead: the line is kept with the row
                    _fill(row, best, True)
                    row["dropped_verdict"] = row["verdict"]
                code = ("two_move_not_ahead" if best is not None else ("no_bearer_no_move" if not row["by"] else "no_taking_legal"))
                row["verdict"] = "DROPPED"
                drop(row["_i"], "no unit bears on it and no move of the taker makes a taking stand" if not row["by"] else "a unit bears on it but no taking is legal", row, code)
        elif kind == "best":
            passed_value[a] = res["reads"][0] if res["ok"] and res["reads"] else None
        elif kind == "resp":
            row, how = a
            row["_pending"].append((how, res))
        else:
            i, p, g = a
            goal_resp[(i, p)] = (g, res)
    for row in flat:
        for how, res in row.pop("_pending", []):
            for af in (res["afters"] if res["ok"] else []):
                row["_resp"].append(response_of(af, passed_value.get((row["_i"], row["_ro"]["taker_p"])), rec, WHY[how], "the frame read's answer", row["net"]))
    # ---- round 3: the sweep of every legal move for every CONFIRMED row that owes a response (the frame read stops at its first sound answers, so its list of answers is not complete)
    q3, plan3, sweep_at, base3 = [], [], {}, {}
    for row in flat:
        if row["verdict"] != "CONFIRMED" or row.get("two_moves"):
            continue
        key = (row["_i"], row["_ro"]["owner_p"], row["place_ix"])
        if key not in sweep_at:
            sweep_at[key] = len(q3)
            q3.append((row["_i"], 7, row["_ro"]["owner_p"], 0, 0, 0, row["place_ix"]))
        bk = (row["_i"], row["_ro"]["taker_p"])
        if bk not in base3:
            base3[bk] = len(q3)
            q3.append((row["_i"], 8, row["_ro"]["taker_p"], 1, 0, 0, 0))
        plan3.append((row, sweep_at[key], base3[bk]))
    r3 = lines.ask(games, q3, lines.sweep_depth, lines.sweep_fuse)
    for row, qi, bi in plan3:
        res = r3[qi]
        base = r3[bi]["reads"][0] if r3[bi]["ok"] and r3[bi]["reads"] else None
        out[row["_i"]]["resp_sweeps"] += 1
        for af in (res["afters"] if res["ok"] else []):
            row["_resp"].append(response_of(af, base, rec, f"found by the sweep of every legal move (read to {lines.sweep_depth} plies)", "the sweep", row["net"]))
    # ---- round 4 (PLAN 78.73 B): a move the sweep's small fuse cut is read again with the fuse of a line (a quiet move costs the most nodes: the plain retreat of the attacked unit was the move cut and dropped in 78.72); one
    # still cut is NOT dropped silently: it stays UNREAD and is listed by name
    q4, plan4, at4 = [], [], {}
    for row, qi, bi in plan3:
        for k, x in enumerate(row["_resp"]):
            if x["source"] == "the sweep" and x["state"] == "UNREAD":
                key = (row["_i"], row["_ro"]["owner_p"], x["first"], row["place_ix"])
                if key not in at4:
                    at4[key] = len(q4)
                    q4.append((row["_i"], 6, row["_ro"]["owner_p"], 0, x["first"][0], x["first"][1], row["place_ix"]))
                plan4.append((row, k, at4[key], bi))
    r4 = lines.ask(games, q4, lines.sweep_depth, lines.fuse)
    for row, k, qi, bi in plan4:
        res = r4[qi]
        base = r3[bi]["reads"][0] if r3[bi]["ok"] and r3[bi]["reads"] else None
        old = row["_resp"][k]
        again = [response_of(af, base, rec, old["how"] + f"; read again with the fuse of a line ({lines.fuse})", "the sweep", row["net"]) for af in (res["afters"] if res["ok"] else [])]
        again = [a for a in again if a["names"][0] == old["names"][0]]
        if again:
            row["_resp"][k] = again[0]
        out[row["_i"]]["resp_reread"] = out[row["_i"]].get("resp_reread", 0) + 1
    # ---- the nodes each row cost: its taking lines, the responses read for it, and its share of a sweep (a sweep is shared by the rows that asked for the same one)
    shared = {}
    for row, qi, bi in plan3:
        shared[qi] = shared.get(qi, 0) + 1
    for row, qi, bi in plan3:
        res = r3[qi]
        row["nodes"] = row.get("nodes", 0) + sum(a["whole"]["nodes"] + sum(y["nodes"] for y in a["takes"]) for a in (res["afters"] if res["ok"] else [])) // shared[qi]
    for row in flat:
        row["nodes"] = row.get("nodes", 0) + sum(x["nodes"] for x in row["_resp"] if x["source"] == "the frame read's answer")
    # ---- assemble
    for row in flat:
        i = row["_i"]
        v = row["verdict"]
        if v == "DROPPED":
            continue
        if v is None:
            drop(i, "no reading")
            continue
        read = row["_resp"]
        firsts = {x["first"] for x in read}
        if v == "CONFIRMED" and row.get("two_moves"):
            row["response"], row["resp_read"], row["resp_unread"], row["deferred"] = [], 0, 0, True
        elif v == "CONFIRMED":
            seen, ks = set(), []
            for x in read:                                       # a move once: the frame read's answer (read to the full depth) comes before the sweep's reading of the same move
                if x["first"] not in seen:
                    seen.add(x["first"])
                    if _kept(x):
                        ks.append(x)
            ks.sort(key=lambda x: (not x["loses_nothing"], x["net"], x["names"]))     # PLAN 78.73 B: the responses that lose nothing first, then by what the taker is left with, the least first
            row["response"], row["resp_read"] = ks, len(firsts)
            row["resp_unread"] = len({x["first"] for x in read if x["state"] == "UNREAD"})
            row["resp_unread_moves"] = sorted({x["names"][0] for x in read if x["state"] == "UNREAD"})
            row["resp_states"] = {st: len({x["first"] for x in read if x["state"] == st}) for st in {x["state"] for x in read}}
            out[i]["resp_read"] += len(firsts)
            out[i]["resp_dropped"] += len(firsts) - len(ks)
        else:
            row["response"], row["resp_read"], row["resp_unread"] = [], 0, 0
        if v == "EXCHANGE":
            out[i]["exchange"].append(row)
        elif v == "REFUTED":
            drop(i, "the taker ends behind: the claim is refuted", row, "refuted")
        else:
            if v == "UNREAD":
                out[i]["unread_rows"] += 1
            out[i][row["side"]].append(row)
    # ---- the goal rows
    for i, p, res in goal_hits:
        g, ra = goal_resp.get((i, p), (res, {"ok": False}))
        mover = items[i]["raw"]["mover"]
        attacker = mover if p == 0 else 1 - mover
        owner = 1 - attacker
        side = "mover" if owner == mover else "other"
        allans = ra.get("answers") or []
        answers = [a for a in allans if a[2] == 0]
        proved = res["forced"] == "proved"
        row = {"side": side, "owner": owner, "goal": True, "kind": None, "place": None, "place_ix": None, "class": "goal", "by": [], "stake": float("inf"), "verdict": "CONFIRMED" if proved else "UNREAD", "ended": TL.ENDED[0] if proved else TL.ENDED[3],
               "reason": (f"the end of the game within {res['depth']} moves of the taker: found by the search and confirmed by the exact forcing read" if proved else f"the search finds a forced end within {res['depth']} moves, the exact forcing read could not confirm it within its fuse"), "line": res["line"], "net": None, "outcome": None, "lead": None, "two_moves": False,
               "response": [{"first": a[0], "names": [a[1]], "how": "after it the forcing read finds no forced end", "source": "the forcing read", "verdict": "ANSWERS", "net": 0, "taker_lost": [], "taker_won": [], "state": "KEPT", "why": "after it the forcing read finds no forced end"} for a in answers],
               "resp_read": len(allans), "resp_unread": len([a for a in allans if a[2] == 2]), "response_ix": [], "frames": [], "win_frame": None, "read_set": ("goal", attacker), "basis": "forced", "_i": i, "_resp": [],
               "depth": res["depth"], "first": res["first"]}
        out[i]["goal"].append(row)
        out[i][side].append(row)
        out[i]["resp_read"] += row["resp_read"]
    for o in out:
        for side in ("mover", "other"):
            o[side].sort(key=lambda r: (0 if r.get("goal") else 1, -(r["stake"] if r["stake"] is not None else -9.0), r["place_ix"] if r["place_ix"] is not None else -1))
    return out


def same_read_set(a: dict, b: dict) -> bool:
    """THE CARRY TEST of E: the read set of a claim (threat_list.read_set_of: the side to move, the members and the places of its frame) is the same at the two positions."""
    return a.get("read_set") == b.get("read_set")
