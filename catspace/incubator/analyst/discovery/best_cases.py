"""PLAN 79.03 THE JUDGE'S POSITIONS AS DATA (SYSTEM_SPEC 1 and 11: a piece that reasons is judged against the board; piece: tactical reasoning, a MEASURE). No engine change, no oracle call; outputs under artifacts/experiments/best_79_03/<tag>/cases_best.json, rendered by ladder_print --best03 (the only place the game's renderer may appear).

THE RULE FOR THE FIFTEEN (fixed here before any was read): position 2 of 79.01's fifteen (leaf_79_01/full/cases_leaf.json) and the first FOURTEEN positions, in the sample's hash order (prior_equal.order_key), where best_count's kind is `unit` (the best root move's own value at D is one unit of the weakest kind or more above the value v the deepening reports, and neither is the game's end), not counting position 2.

PER POSITION: the moves; the depth D the stored run finished; at that depth with the switch off and on (the engine's gap audit, node budgets): the value v, the move chosen, the best root move's own value and its name(s); the line that starts with the move chosen and the line that starts with the best move (equal_trace's line read, mode 9, under the stack's judge, at D);
the unpruned deepening's best at D (best_which's reading, where it was read); 79.00's stored class after each of the moves in its equal set; and, where the clocked reads of best_mend exist, the move played and the pick under A (switch off) and M (switch on).

INTERFACE (written before the bodies).
  pick_ids(rows, order_key, named) -> [id]            the fifteen by the rule (pure; tested).
  Cases()                                             per process: the stack's tracer (lines) and the engine's gap audit. `.case(rec, moves, row) -> dict`.
  run(tag, tag_gap, say)                              writes cases_best.json.
  Called by: the command line, ladder_print --best03.

  .venv/bin/python -m catspace.incubator.analyst.discovery.best_cases --tag full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
from pathlib import Path

from catspace.incubator.analyst.discovery import best_count as BC
from catspace.incubator.analyst.discovery import best_mend as BM
from catspace.incubator.analyst.discovery import best_which as BW
from catspace.incubator.analyst.discovery import cut_count as CC
from catspace.incubator.analyst.discovery import equal_trace as ET
from catspace.incubator.analyst.discovery import leaf_ask as LA
from catspace.incubator.analyst.discovery import leaf_firm as LF
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import stack_causes as SC
from catspace.incubator.analyst.discovery.best_out import Say, out_root

FROM_79_01 = 2
MORE = 14


def pick_ids(rows: list, named: str, more: int = MORE) -> list:
    """The id of 79.01's position 2, then the first `more` rows of kind `unit` in the sample's hash order, not the named one."""
    unit = sorted((r for r in rows if r["kind"] == "unit" and r["id"] != named), key=lambda r: PE.order_key(r["game"], int(r["id"].rsplit("_", 1)[1])))
    return [named] + [r["id"] for r in unit[:more]]


class Cases(SC.StackTracer):
    def gap(self, moves: list, depth: int, flag: bool) -> dict:
        import cs_engine as CE
        a = CE.gap_audit(OM.VARIANT, list(moves), self.worth, ET.RUN_NODES, int(depth), ET.BIG, True, flag)
        ex = sorted(((r["exact"], r["name"]) for r in a["rows"] if r["exact"] is not None), reverse=True)
        return {"depth": int(a["depth"]), "value": int(a["value"]), "first": a["first"], "run_nodes": int(a["run_nodes"]), "exact": ex}

    def case(self, rec: dict, moves: list, row: dict, which: "dict | None", mend: "dict | None") -> dict:
        d = rec["depth"]
        off, on = self.gap(moves, d, False), self.gap(moves, d, True)
        best = off["exact"][0][1] if off["exact"] else None
        lines = {}
        for tag, m in (("played", off["first"]), ("best", best)):
            lines[tag] = {k: v for k, v in self.line(moves, m, d).items() if k in ("ok", "names", "settled", "value", "cut", "exchange")} if m else None
        return {"id": rec["id"], "moves": list(moves), "band": rec["band"], "units": rec["units"], "cls": rec["cls"], "depth": d, "kind": row["kind"], "gap": row["gap"], "off": off, "on": on, "best": best, "lines": lines,
                "tied": list(rec["tied"]), "cm": list(rec["cm"]), "first": rec["first"], "pick": rec["pick"],
                "which": ({"plain_best": which["fl"]["best_plain"], "plain_other": which["fl"]["other_plain"], "no_late_best": which["fl"]["best_no_late"], "no_null_best": which["fl"]["best_no_null"], "cause": which["cause"]} if which else None),
                "clock": ({a: {k: mend[a].get(k) for k in ("first", "pick", "depth", "cut", "seconds")} for a in BM.ARMS} if mend else None)}


def run(tag: str, tag_gap: str, say) -> None:
    rows = BC.load_rows(tag_gap)
    old = json.loads((Path(LF.out_root("full")) / "cases_leaf.json").read_text())
    named = old[FROM_79_01 - 1]["id"]
    ids = pick_ids(rows, named)
    byid = {r["id"]: r for r in rows}
    recs = {r["id"]: r for r in LA.src_records()}
    items = {rec["id"]: moves for rec, moves in CC.items_of(LA.SRC, 0) if rec["id"] in ids}
    which = {}
    wp = out_root(tag_gap) / "which_rows.jsonl"
    if wp.exists():
        which = {r["id"]: r for r in (json.loads(x) for x in wp.read_text().splitlines() if x.strip())}
    mend = {}
    mp = out_root(tag_gap) / "mend.jsonl"
    if mp.exists():
        mend = {r["id"]: r for r in BM.load(out_root(tag_gap)) if "error" not in r and all(r.get(a) for a in BM.ARMS)}
    cs = Cases()
    out = [cs.case(recs[i], items[i], byid[i], which.get(i), mend.get(i)) for i in ids]
    (out_root(tag) / "cases_best.json").write_text(json.dumps(out))
    say(f"{len(out)} judged positions written to {out_root(tag) / 'cases_best.json'}: position {FROM_79_01} of 79.01 ({named}) and the first {MORE} of kind `unit` in the sample's hash order; no oracle call")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="full")
    ap.add_argument("--tag-gap", default=None)
    a = ap.parse_args(argv)
    run(a.tag, a.tag_gap or a.tag, Say(out_root(a.tag), "cases"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
