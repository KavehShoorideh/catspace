"""Plan 59 comparison harness: the agenda with agenda.TIERS off against the saved result of a run made BEFORE plan 59's first commit (decisions, every claim of every root agenda,
the root traces without the STATUS lines, the result fields). Each target runs in a fresh process.

  .venv/bin/python scripts/compare_tiers.py --save base.pkl     # run on the pre-change tree (TIERS is simply set when the attribute exists)
  .venv/bin/python scripts/compare_tiers.py --check base.pkl    # TIERS off now vs the saved run
  .venv/bin/python scripts/compare_tiers.py --check base.pkl --targets rand1_10
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_regress import PUZZLES, puzzles                      # noqa: E402,F401

TARGETS = ["07Z0P", "0LRTF", "rand1_10", "rand2_10", "rand3_10", "rand22_5"]
PUZZLES.setdefault("0LRTF", dict(id="0LRTF", fen="2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 6 62"))


def worker(target: str, out: str) -> None:
    from catspace.incubator.analyst.discovery import agenda as AG
    from catspace.incubator.analyst.discovery import analyse as AN
    AG.TIERS = False
    roots: list = []
    fin = AG.Agenda._ask_finish

    def spy(self):
        fin(self)
        if self.chase_depth == 0 and self.ctx is not None:
            roots.append(dict(claims=[(c.kind, c.move, c.what, c.value, c.lemma, c.line, c.goal, c.fixes, c.qid) for c in self.state.claims],
                              gaps=list(self.state.gaps), kills=list(self.kills),
                              trace=[x for x in self.lines if not x.startswith("budget:") and not x.lstrip().startswith("STATUS")]))
    AG.Agenda._ask_finish = spy
    t0 = time.time()
    res = [AN.analyse(p, quiet=True, memory=False, log=lambda *a, **k: None) for p in puzzles(target)]
    secs = time.time() - t0
    for r in res:
        r.pop("secs", None)
    pickle.dump(dict(results=res, roots=roots, seconds=secs), open(out, "wb"))


def run(target: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "r.pkl")
        subprocess.run([sys.executable, __file__, "--worker", target, out], check=True)
        return pickle.load(open(out, "rb"))


def run_all(targets: list) -> dict:
    with ThreadPoolExecutor(max_workers=len(targets)) as ex:
        return dict(zip(targets, ex.map(run, targets)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--save")
    ap.add_argument("--check")
    ap.add_argument("--worker", nargs=2)
    a = ap.parse_args()
    if a.worker:
        worker(*a.worker)
        return
    if a.save:
        res = run_all(a.targets)
        pickle.dump(res, open(a.save, "wb"))
        for t, r in res.items():
            print(f"OBS tiers baseline {t}: {len(r['results'])} puzzles, " + " ".join(f"{x['id']}={x['decision']}" for x in r["results"][:12]) + f", seconds {r['seconds']:.1f}")
        return
    base = pickle.load(open(a.check, "rb"))
    new = run_all(a.targets)
    total = 0
    for t in a.targets:
        old, nw = base[t], new[t]
        dec = [(o["id"], o["decision"], n["decision"]) for o, n in zip(old["results"], nw["results"]) if (o["decision"], o["move"], o["match"], o["decided"]) != (n["decision"], n["move"], n["match"], n["decided"])]
        if len(old["results"]) != len(nw["results"]):
            dec.append(("length", len(old["results"]), len(nw["results"])))
        cl, tr = [], []
        for i, (o, n) in enumerate(zip(old["roots"], nw["roots"])):
            for k in ("claims", "gaps", "kills"):
                if o[k] != n[k]:
                    cl.append((i, k))
            if o["trace"] != n["trace"]:
                tr.append(i)
        if len(old["roots"]) != len(nw["roots"]):
            cl.append(("root agendas", len(old["roots"]), len(nw["roots"])))
        total += len(dec) + len(cl) + len(tr)
        print(f"OBS tiers compare {t}: decisions {'identical' if not dec else 'different %s' % dec}, claims {'identical' if not cl else 'different %s' % cl}, "
              f"trace {'identical' if not tr else 'different (roots %s)' % tr} ({sum(len(r['claims']) for r in old['roots'])} claims of {len(old['roots'])} root agendas), seconds old {old['seconds']:.1f} new {nw['seconds']:.1f}")
    print(f"OBS tiers compare total differences: {total}")
    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()
