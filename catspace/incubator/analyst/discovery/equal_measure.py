"""PLAN 78.98 PART ONE, THE COUNT OF CAUSES (SYSTEM_SPEC 4 the tactical reasoner's equal reading, 5 the value, 12 the debt of plan 78.84; piece: tactical reasoning). Reads the traces of equal_trace.py (trace.jsonl under artifacts/experiments/equal_78_98/<tag>/) and prints
OBS lines; writes nothing else. No oracle, no engine call, no number put in by hand: the worths used to read an exchange are the judge's own record (odds_measure.players_of), the table the search itself sums.

STAGE `count`: the causes (b), (c), (a), unread and not held (equal_trace.cause_of) over the stuck moves of kind "all present", by band (few, mid, many units) with a 90 pct band by games resampled whole; the same counts of the traced moves the rule holds; the time the traces took.
For the moves of cause (a): which units were exchanged (the units the mover gave and took along the line the value comes from, the engine's own words) and what the judge's record makes of that exchange (the sum of the record's worths of what the mover took less what it gave: more than nothing,
nothing, less than nothing), by band. For the moves of cause (b): which reading removes it (the exact value, the answer with the prunings off, the same-depth reading's own best).
STAGE `records` (a diagnostic of cause (a), not a mend): what the rule holds equal at D under each of the three measured records of the kinds (equal_trace.py `records`), for the stuck moves and for the class-keeping moves of the same positions.
INTERFACE. cmd_count(tag, say), cmd_records(tag, say); Called by the command line only.

  .venv/bin/python -m catspace.incubator.analyst.discovery.equal_measure count|records --tag short | --tag full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
from collections import Counter

import numpy as np

from catspace.incubator.analyst.discovery import equal_trace as ET
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import prior_measure as PMS

BAND_NAMES = ("few", "mid", "many")
SEED = 9898


def share(vals, game) -> tuple:
    return PMS.share(np.asarray(vals, float), np.asarray(game), SEED)


def fmt(t: tuple) -> str:
    return PMS.fmt(t)


def game_of(rec: dict) -> int:
    return int(rec["id"].split("_")[0])


def net_by_record(ex: dict, worth: dict) -> int:
    """The record's worths of what the mover took less what it gave along a line (the judge's own sum, without the kind changes)."""
    return sum(worth.get(k, 0) for k in ex["took"]) - sum(worth.get(k, 0) for k in ex["gave"])


def sign_word(x: int) -> str:
    return "more than nothing" if x > 0 else ("nothing" if x == 0 else "less than nothing")


def cmd_count(tag: str, say) -> None:
    root = ET.out_root(tag)
    rows = [r for r in ET.load(root) if "error" not in r and r.get("tag") == "all present"]
    errs = [r for r in ET.load(root) if "error" in r]
    worth = dict(OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")["worth"])
    say(f"RULE: a root move is held equal iff its null-window answer at the depth D the deepening finished is at least the value v of the position, asked on the deepening's warm table with the late-move reduction and the null move on; margin zero, the judge's integer sum; the judge is the measured record {worth}")
    say(f"the traced stuck moves of kind all present: {len(rows)} ({len(errs)} errors); D is the play clock's depth; one thread's seconds a move traced {np.mean([r['seconds'] for r in rows]):.1f}" if rows else "no rows")
    if not rows:
        return
    game = np.array([game_of(r) for r in rows])
    band = np.array([r["band"] for r in rows])
    cause = np.array([r["cause"] for r in rows])
    for c in ET.CAUSES:
        say(f"cause {c}: all {fmt(share(cause == c, game))} pct ({int((cause == c).sum())} of {len(rows)}); " + "; ".join(f"{b} {fmt(share((cause == c)[band == b], game[band == b]))} ({int(((cause == c) & (band == b)).sum())} of {int((band == b).sum())})" for b in BAND_NAMES))
    held = cause != "not held"
    for c in ET.CAUSES[1:]:
        say(f"cause {c} among the moves the rule held at D in the trace ({int(held.sum())}): {fmt(share((cause == c)[held], game[held]))} pct")
    b_rows = [r for r in rows if r["cause"] == "b"]
    if b_rows:
        def exact_under(r):
            s = r["stuck_row"]
            return s["exact"] is not None and s["exact"] < r["value"]

        def plain_under(r):
            s = r["stuck_row"]
            return s["plain"] is not None and s["plain"] < r["value"]

        def same_depth(r):
            d = r.get("deep") or {}
            return d.get("other_d") is not None and d.get("best_d") is not None and d["other_d"] < d["best_d"]
        say(f"cause b, which reading shows it ({len(b_rows)} moves): exact value under v {sum(map(exact_under, b_rows))}; answer with the prunings off under v {sum(map(plain_under, b_rows))}; the prunings-off reading at D puts the move under its own best {sum(map(same_depth, b_rows))}; "
            f"of the last, the best of that reading is above v in {sum(1 for r in b_rows if same_depth(r) and r['deep']['best_d'] > r['value'])}")
    a_rows = [r for r in rows if r["cause"] == "a"]
    if a_rows:
        sig = Counter((" ".join(r["stuck_line"]["exchange"]["gave"]) or "-", " ".join(r["stuck_line"]["exchange"]["took"]) or "-") for r in a_rows if r.get("stuck_line", {}).get("ok"))
        say(f"cause a ({len(a_rows)} moves), the units the mover gave and took along the line the value comes from (gave | took: moves): " + "; ".join(f"{g} | {t}: {n}" for (g, t), n in sig.most_common(12)))
        nets = Counter()
        by_band = {b: Counter() for b in BAND_NAMES}
        for r in a_rows:
            if r.get("stuck_line", {}).get("ok"):
                w = sign_word(net_by_record(r["stuck_line"]["exchange"], worth))
                nets[w] += 1
                by_band[r["band"]][w] += 1
        say(f"cause a, what the judge's record makes of the exchange along the line: " + "; ".join(f"{w}: {n}" for w, n in nets.most_common()) + " (by band: " + "; ".join(f"{b} " + ", ".join(f"{w} {n}" for w, n in by_band[b].most_common()) for b in BAND_NAMES) + ")")
        order = " < ".join(k for k, _w in sorted(worth.items(), key=lambda kv: kv[1]))
        unlike = [r for r in a_rows if r.get("stuck_line", {}).get("ok") and len(r["stuck_line"]["exchange"]["gave"]) == len(r["stuck_line"]["exchange"]["took"]) > 0 and r["stuck_line"]["exchange"]["gave"] != r["stuck_line"]["exchange"]["took"]]
        gain = [r for r in unlike if net_by_record(r["stuck_line"]["exchange"], worth) > 0]
        say(f"cause a, the record's order of the kinds by worth: {order}; exchanges of the same number of units of different kinds along the line ({len(unlike)} moves; by band "
            + ", ".join(f"{b} {sum(1 for r in unlike if r['band'] == b)}" for b in BAND_NAMES) + f"): the record makes the exchange a gain for the mover in {len(gain)}, a loss in {sum(1 for r in unlike if net_by_record(r['stuck_line']['exchange'], worth) < 0)}, nothing in {sum(1 for r in unlike if net_by_record(r['stuck_line']['exchange'], worth) == 0)}")
        say(f"cause a, the value of the position at D: under nothing in {sum(1 for r in a_rows if r['value'] < 0)}, nothing in {sum(1 for r in a_rows if r['value'] == 0)}, over nothing in {sum(1 for r in a_rows if r['value'] > 0)}; the value of the line the stuck move's value comes from is the value of the position in "
            f"{sum(1 for r in a_rows if r.get('stuck_line', {}).get('value') == r['value'])} of {len(a_rows)}")
        pure = sum(1 for r in a_rows if r.get("stuck_line", {}).get("ok") and not r["stuck_line"]["exchange"]["gave"] and not r["stuck_line"]["exchange"]["took"])
        say(f"cause a, moves whose line gives and takes no unit: {pure} of {len(a_rows)}; with a kind change on the line: {sum(1 for r in a_rows if r.get('stuck_line', {}).get('ok') and r['stuck_line']['exchange']['changed'])}")
        later = sum(1 for r in a_rows if (r.get("keeper_line") or {}).get("ok") and r["keeper_line"]["exchange"] == r["stuck_line"]["exchange"] and (r["keeper_line"]["exchange"]["gave"] or r["keeper_line"]["exchange"]["took"]))
        say(f"cause a, moves whose class-keeping equal move reaches the same exchange (the same units given and taken along its line): {later} of {len(a_rows)}")
    unread = [r for r in rows if r["cause"] == "unread"]
    if unread:
        pe = sum(1 for r in unread if (r.get("deep") or {}).get("pruned_other") is not None and (r["deep"].get("pruned_best") is not None) and r["deep"]["pruned_other"] < r["deep"]["pruned_best"])
        say(f"unread ({len(unread)}): the node budget stopped the prunings-off reading at D+{ET.DEEP}; of these the reading with the prunings on, at D+{ET.DEEP}, puts the move under its best in {pe}")
    both = Counter((r["cause"]) for r in rows if r.get("deep") and r["deep"].get("pruned_other") is not None and r["deep"].get("pruned_best") is not None and r["deep"]["pruned_other"] < r["deep"]["pruned_best"])
    say(f"the reading with the prunings on at D+{ET.DEEP} puts the move under its best, by cause: " + ", ".join(f"{c} {n}" for c, n in both.most_common()))
    named = [r for r in ET.load(root) if str(r.get("tag", "")).startswith("named")]
    for r in named:
        s, k = r.get("stuck_row") or {}, r.get("keeper_row") or {}
        say(f"named {r['tag']} {r['id']}: D {r.get('depth')} v {r.get('value')}; stuck {r['stuck']} gain {s.get('gain')} warm {s.get('warm')} plain {s.get('plain')} exact {s.get('exact')}; keeper {r.get('keeper')} gain {k.get('gain')} warm {k.get('warm')} plain {k.get('plain')} exact {k.get('exact')}; cause {r.get('cause')}")


def cmd_records(tag: str, say) -> None:
    """A diagnostic of cause (a), not a mend: the same positions' equal reading at D under the three measured records of the kinds. Printed: each record's worths and the half widths it carries (threat_concrete.Record), then, for the stuck moves of kind all present and for the
    class-keeping moves of the same positions' label sets, the share each record's rule no longer holds equal (the move leaves the set), with a 90 pct band by games."""
    from catspace.incubator.analyst.discovery import threat_concrete as TC
    root = ET.out_root(tag)
    recs = ET.load_records(root)
    stuck = {}
    for r in ET.load(root):
        if r.get("tag") == "all present" and "error" not in r:
            stuck.setdefault(r["id"], []).append(r["stuck"])
    for name in ET.RECORDS:
        rec = TC.Record.load(name)
        order = " < ".join(k for k, _w in sorted(rec.worth.items(), key=lambda kv: kv[1]))
        say(f"record {name}: worths {rec.worth}; half widths {{" + ", ".join(f"{k}: {v:.0f}" for k, v in rec.half.items()) + f"}}; order by worth {order}")
    for name in ET.RECORDS:
        lose_s, game_s, keep_s, game_k = [], [], [], []
        cut = err = 0
        for r in recs:
            x = r["records"].get(name, {})
            if "error" in x:
                err += 1
                continue
            if x.get("tie_cut"):
                cut += 1
                continue
            held = set(x["held"])
            for m in stuck.get(r["id"], []):
                lose_s.append(m not in held)
                game_s.append(game_of(r))
            for m, k in zip(r["tied"], r["keep"]):
                if k:
                    keep_s.append(m not in held)
                    game_k.append(game_of(r))
        say(f"under {name}: of the stuck moves, leave the equal set {fmt(share(lose_s, game_s))} pct ({int(sum(lose_s))} of {len(lose_s)}); of the class-keeping moves of the same positions, leave {fmt(share(keep_s, game_k))} pct ({int(sum(keep_s))} of {len(keep_s)}); positions cut {cut}, failed {err} of {len(recs)}")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["count", "records"])
    ap.add_argument("--tag", default="short")
    a = ap.parse_args(argv)
    {"count": cmd_count, "records": cmd_records}[a.stage](a.tag, ET.Say(ET.out_root(a.tag), a.stage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
