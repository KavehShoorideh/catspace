"""Plan 56 (1) comparison harness: the agenda with the recursive chase (agenda.SCHEDULER = False) and with the task graph run by the worker loop (True), each in a fresh process (the memo
caches must not leak between the two), diffing every decision, EVERY CLAIM of every root agenda (kind, move, what, value, lemma, line, question id), the counters that must match (the
analyser's `chase` tuple: inner agendas, inner questions, replies closed by rule, fuse hits, ...; questions asked, kills, gaps, the fact store's counts) and, as the product, the TRACE
(the lines of the root agenda, in order).

  .venv/bin/python scripts/compare_scheduler.py                       # 07Z0P, 0LRTF, rand1_10, rand2_10, rand3_10
  .venv/bin/python scripts/compare_scheduler.py --targets rand1_10 --prior      # (the prior needs data/derived/question_prior.json)
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_regress import PUZZLES, puzzles                      # noqa: E402,F401  (the same five targets)


def worker(target: str, mode: str, prior: bool, out: str) -> None:
    """mode: rec (the recursive chase), py (the task graph, the Python scheduler), rust (the task graph, the Rust core: scheduler.CORE_RUST)."""
    sched = mode != "rec"
    from catspace.incubator.analyst.discovery import agenda as AG
    import os
    if os.environ.get("F61"):                                       # plan 61 A: F61=1 PARRIES on both sides; F61=2 also the delayed threat, no safe move and the tiers (TIERS, so tier 1+ runs under the count budget)
        AG.PARRIES = True
        if os.environ["F61"] == "2":
            AG.DELAYED_THREAT = AG.NO_SAFE_MOVE = AG.TIERS = True
    if os.environ.get("EST61"):                                     # plan 61 B: EST61=1 ESTIMATE on both sides (the estimates log of the harness goes to its own file, removed by main)
        AG.ESTIMATE = True
        from catspace.incubator.analyst.discovery import question_prior as QP0
        QP0.EST_LOG = "estimates_log_harness.jsonl"
    from catspace.incubator.analyst.discovery import analyse as AN
    AG.SCHEDULER = sched
    AG.PRIOR = prior
    if mode in ("rust", "route"):
        from catspace.incubator.analyst.discovery import scheduler as SC0
        SC0.CORE_RUST = True
        SC0.ROUTE_RUST = mode == "route"                            # plan 56 (2b): the Rust route reader
    roots: list = []
    fin = AG.Agenda._ask_finish

    def spy(self):
        fin(self)
        if self.chase_depth == 0 and self.ctx is not None:
            cx = self.ctx
            roots.append(dict(
                claims=[(c.kind, c.move, c.what, c.value, c.lemma, c.line, c.goal, c.fixes, c.qid) for c in self.state.claims],
                estimated=[(c.kind, c.move, c.what, c.value, c.line, c.proved, c.confidence, c.estimator) for c in self.state.estimated],
                gaps=list(self.state.gaps), kills=list(self.kills), trace=[x for x in self.lines if not x.startswith("budget:")],
                counters=dict(asked=self.asked, tempo=self.killed_tempo, agendas=cx.agendas, inner_questions=cx.asked, closed=cx.closed, fuse_hits=cx.budget_hits, depth_cuts=cx.depth_cuts,
                              standing=cx.standing, inherited=cx.inherited, reopened=cx.reopened, fresh=cx.fresh, gap_reused=cx.gap_reused, refuter_fixes=cx.refuter_fixes, reasoned=cx.reasoned,
                              unmet=cx.unmet, unaddressed=cx.unaddressed, static_read=cx.static_read, theirs=self.their_runs, left=cx.left),
                tasks=cx.stats))
    AG.Agenda._ask_finish = spy
    if sched:
        from catspace.incubator.analyst.discovery import scheduler as SC
        run_root = SC.run_root

        def wrapped(ag):
            run_root(ag)
            roots[-1]["tasks"] = ag.ctx.stats                       # (set when the worker loop has ended, after the root's `_ask_finish`)
        SC.run_root = wrapped
    t0 = time.time()
    res = [AN.analyse(p, quiet=True, memory=False, log=lambda *a, **k: None) for p in puzzles(target)]
    secs = time.time() - t0
    for r in res:
        r.pop("secs", None)
        for t in r.get("tiers", ()):
            t.pop("secs", None)                                      # (plan 61: tiers run -> their wall seconds are not a decision)
    pickle.dump(dict(results=res, roots=roots, seconds=secs), open(out, "wb"))


def run(target: str, mode: str, prior: bool) -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "r.pkl")
        subprocess.run([sys.executable, __file__, "--worker", target, mode, "1" if prior else "0", out], check=True)
        return pickle.load(open(out, "rb"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["07Z0P", "0LRTF", "rand1_10", "rand2_10", "rand3_10"])
    ap.add_argument("--prior", action="store_true", help="both sides with agenda.PRIOR on")
    ap.add_argument("--core", action="store_true", help="plan 56 (2a): the Python scheduler (old) vs the Rust core (new)")
    ap.add_argument("--core-vs-recursive", action="store_true", help="the recursive chase (old) vs the Rust core (new)")
    ap.add_argument("--route", action="store_true", help="plan 56 (2b): the Rust core (old) vs the Rust core with the Rust route reader (new)")
    ap.add_argument("--route-vs-recursive", action="store_true", help="the recursive chase (old) vs the Rust core with the Rust route reader (new)")
    ap.add_argument("--forcing61", choices=["1", "2"], help="plan 61 A: 1 = PARRIES on both sides; 2 = + DELAYED_THREAT, NO_SAFE_MOVE and TIERS")
    ap.add_argument("--estimate", action="store_true", help="plan 61 B: ESTIMATE on both sides (the estimated claims and the ESTIMATED decisions are compared like the proved ones)")
    ap.add_argument("--worker", nargs=4, metavar=("TARGET", "SCHED", "PRIOR", "OUT"))
    a = ap.parse_args()
    if a.worker:
        worker(a.worker[0], a.worker[1], a.worker[2] == "1", a.worker[3])
        return
    total = 0
    import os
    if a.forcing61:
        os.environ["F61"] = a.forcing61
    if a.estimate:
        os.environ["EST61"] = "1"
    m_old, m_new = ("rust", "route") if a.route else ("rec", "route") if a.route_vs_recursive else ("py", "rust") if a.core else ("rec", "rust") if a.core_vs_recursive else ("rec", "py")
    print(f"OBS scheduler compare: old = {m_old}, new = {m_new}")
    for t in a.targets:
        old, new = run(t, m_old, a.prior), run(t, m_new, a.prior)
        dec = [(o["id"], o["decision"], o["move"], n["decision"], n["move"]) for o, n in zip(old["results"], new["results"]) if (o["decision"], o["move"], o["match"], o["decided"]) != (n["decision"], n["move"], n["match"], n["decided"])]
        if len(old["results"]) != len(new["results"]):
            dec.append(("length", len(old["results"]), len(new["results"])))
        cl, cnt, tr = [], [], []
        for i, (o, n) in enumerate(zip(old["roots"], new["roots"])):
            for k in ("claims", "estimated", "gaps", "kills"):
                if o[k] != n[k]:
                    cl.append((i, k, [x for x in o[k] if x not in n[k]][:2], [x for x in n[k] if x not in o[k]][:2]))
            cnt += [(i, k, o["counters"][k], n["counters"][k]) for k in o["counters"] if o["counters"][k] != n["counters"][k]]
            if o["trace"] != n["trace"]:
                j = next((j for j, (x, y) in enumerate(zip(o["trace"], n["trace"])) if x != y), min(len(o["trace"]), len(n["trace"])))
                tr.append((i, j, o["trace"][j:j + 1], n["trace"][j:j + 1]))
        if len(old["roots"]) != len(new["roots"]):
            cl.append(("root agendas", len(old["roots"]), len(new["roots"])))
        cnt += [(r["id"], "result fields", {k: v for k, v in r.items() if v != n.get(k)}, {k: n.get(k) for k in r if r[k] != n.get(k)})
                for r, n in zip(old["results"], new["results"]) if {k: v for k, v in r.items() if k != "fx"} != {k: v for k, v in n.items() if k != "fx"}]
        total += len(dec) + len(cl) + len(cnt)
        nq = sum(len(r["claims"]) for r in old["roots"])
        print(f"OBS scheduler compare {t}: decisions {'identical' if not dec else 'different (%d)' % len(dec)}, claims {'identical' if not cl else 'different (%d differences)' % len(cl)} "
              f"({nq} claims of {len(old['roots'])} root agendas), counters {'identical' if not cnt else 'different (%d differences)' % len(cnt)}, "
              f"trace {'identical' if not tr else 'different (%d of %d roots)' % (len(tr), len(old['roots']))}, seconds old {old['seconds']:.1f} new {new['seconds']:.1f}")
        st = [r["tasks"] for r in new["roots"] if r["tasks"]]
        if st:
            print(f"OBS scheduler tasks {t}: " + ", ".join(f"{k} {sum(s.get(k, 0) for s in st)}" for k in ("node", "ask", "chase", "route", "reply")) + f", their_pass leaves {sum(s['leaves'].get('their_pass', [0])[0] for s in st)}"
                  f", worker steps {sum(s['steps'] for s in st)}, most ready at once {max(s['ready_max'] for s in st)}")
        for lst, name in ((dec, "decision"), (cl, "claim"), (cnt, "counter"), (tr, "trace")):
            for d in lst[:3]:
                print(f"  {name} diff:", d)
    if a.estimate:
        from catspace.io import paths
        h = paths.experiment("estimates_log_harness.jsonl")
        if os.path.exists(h):
            os.remove(h)
    print(f"OBS scheduler compare total differences (decisions + claims + counters): {total}")
    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()
