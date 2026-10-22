"""THE ANALYSER (ANALYST_PLAN 52.2): the state, the agenda trace, the claims, the gaps, and one OBS line per puzzle.

  .venv/bin/python -m catspace.incubator.analyst.discovery.analyse --set dev_20 --id 01IFU
  .venv/bin/python -m catspace.incubator.analyst.discovery.analyse --set dev_20            # the batch: one OBS line per puzzle + a tally (--quiet: OBS only)
"""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter

import chess

from catspace.io import paths
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery import remember as RM
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import state_of


AGENDA_STORE = "agenda_states.jsonl"       # 52.9: the agenda's own store (plan 50's reasoned_states.jsonl keeps its record of the exhaustive ladder)


def analyse(p: dict, quiet: bool = False, log=print, memory: bool = True, store: bool = False, set_name: str = "", k: int = 3, leaf_all: bool = False, seconds: "float | None" = None, carry=None, ponder: "float | None" = None, qlog: bool = False, sf: bool = False) -> dict:
    BV.install_terms()
    t0 = time.time()
    fen = p["fen"]
    b = chess.Board(fen)
    us = "w" if b.turn else "b"
    st = state_of(fen, us)
    if not quiet:
        log(f"position {p['id']}: {fen}  {'white' if b.turn else 'black'} to move; the puzzle's window: gain {p['value']}")
        for line in st.lines():
            log(line)
    recalled = RM.recall(st.tokens(), k, exclude_id=p["id"], path=paths.derived(AGENDA_STORE)) if memory else []
    if not quiet:
        for sim, r in recalled:
            log(f"recall: seen {r['id']} (sim {sim}): {r.get('goal')} by {r.get('lemma')}, fixes {r.get('fixes')}")
    pc0 = R.PASS_CLAIMS[0]
    if carry is not None:                                        # 52.19 (4): --memory keeps the fact store between the puzzles of a set (unrelated positions: expect few hits)
        carry.facts.advance(b)
    ag = Agenda(st, memory=recalled, leaf_all=leaf_all, chase_log=p["id"], seconds=seconds, carry=carry)
    ag.ctx.est_set = set_name                                    # plan 61 B2: the estimates log rows carry the set
    ag.ask()
    fx = ag.facts_delta()
    ans = b.san(chess.Move.from_uci(p["answer"]))
    d = ag.decision()
    secs = round(time.time() - t0, 1)
    if not quiet:
        log("AGENDA (top to bottom, by stake)")
        for n in ag.notes:
            log("  note: " + n)
        log(ag.trace())
        log("CLAIMS")
        for c in st.claims:
            log(f"  {c.kind} {c.what} value {c.value} lemma {c.lemma}: {c.line}")
        if not st.claims:
            log("  none")
        for c in st.estimated:
            log(f"  ESTIMATED {c.kind} {c.what} value {c.value} (not proved; confidence {c.confidence:.2f} by {c.estimator}): {c.line}")
        log("GAPS")
        for g in st.gaps or ["none"]:
            log("  " + g)
    if store and d is not None:
        rec = RM.make_record(p, set_name, fen, us, d.move, d.lemma, d.value, d.move[:4] == p["answer"][:4], RM.threat_token(fen, us, check=b.is_check()),
                             goal=d.goal, fixes=d.fixes, lemma=d.lemma)
        RM.store(rec, paths.derived(AGENDA_STORE))
    for line in ag.leaf_obs:
        log(f"OBS leaf {p['id']}: {line}")
    decided = d.what if d else "none"
    estimated = ag.decision_estimated()
    if ag.ctx.est_routes or ag.ctx.est_why:
        log(f"OBS estimate gaps {p['id']}: not estimated {dict(ag.ctx.est_why)}")
    if ag.ctx.est_routes:
        log(f"OBS estimate {p['id']}: gap routes closed as estimated {ag.ctx.est_routes}, estimated claims {len(st.estimated)}, decision {decided}" + (" ESTIMATED" if estimated else " (proved)" if d else ""))
    sfv = (None, None, None)
    if sf:                                                       # plan 53 (1): Stockfish grades the decision (the grader, never a premise)
        from catspace.incubator.analyst.discovery import verdict as VD
        sfv = VD.verdict(fen, d.move if d else None)
        log(f"OBS verdict {p['id']}: {decided} {sfv[0]}" + (f" (loss {sfv[1]} cp vs {chess.Board(fen).san(chess.Move.from_uci(sfv[2]))})" if sfv[2] else ""))
    if qlog:
        _write_qlog(p, set_name, fen, ag, d, sfv, estimated)
    cx = ag.ctx
    # a depth-two decision reads "Nxf6+ then Bxh7": it MATCHES the puzzle when its FIRST move is the answer (the second move is the plan's, not the puzzle's)
    verdict = "MATCH" if d and d.move[:4] == p["answer"][:4] and (len(d.move) < 5 or len(p["answer"]) < 5 or d.move[4] == p["answer"][4]) else "MISS"
    log(f"OBS agenda {p['id']}: questions asked {ag.asked}, killed by tempo {ag.killed_tempo}, claims {len(st.claims)}, decision {decided} by "
        f"{d.lemma if d else '-'}, answer {ans} {verdict}, their agendas {ag.their_runs}, chase: inner agendas {cx.agendas}, inner questions {cx.asked}, inherited {cx.inherited} (claims {cx.inherited_claims}), reopened {cx.reopened}, new {cx.fresh}, "
        f"fuse hits {cx.budget_hits} (depth {cx.depth_cuts}), gap routes reused {cx.gap_reused}, replies closed by rule {cx.closed}, replies left open (goal unmet) {cx.unmet}, refuter fixes {cx.refuter_fixes}, standing kills {cx.standing}, chase log rows {cx.logged}, static leaves read {cx.static_read}, parried {cx.static_parried}, unparried {cx.static_unparried}, killed by reasoning {cx.reasoned}, routes not addressing a defect {cx.unaddressed}, time outs {cx.time_outs}, landings served from the table {cx.landing_served}, forcing by a claim after a pass {R.PASS_CLAIMS[0] - pc0}, "
        f"facts derived {fx['derived']}, served from the store {fx['served']}, foreground recomputations {fx['recomputed']}, background queue {fx['queue']} (top stake {fx['top']}), seconds {secs}, "
        f"root lists served {cx.roots_served}, chase claims served {cx.chase_served}, routes finished after the deadline {cx.finished}")
    if cx.parry_seen or cx.delayed_routes or cx.nosafe_routes:
        log(f"OBS forcing61 {p['id']}: closed replies checked against the route's threat {cx.parry_seen}, of which closed by the threat (no parry) {cx.parry_closed}, routes admitted by the delayed threat {cx.delayed_routes}, by no safe move {cx.nosafe_routes}")
    if cx.judge_routes or cx.judge_cuts or cx.cycles:
        spn = f"{cx.judge_t / cx.judge_n:.3f}" if cx.judge_n else "-"
        log(f"OBS cost61 {p['id']}: root routes judged {cx.judge_routes}, nodes read on {cx.judge_reads}, cuts by the judge {cx.judge_cuts}, seconds per node {spn} ({cx.judge_n} timed), cycle claims {cx.cycles}")
    tiers = getattr(ag, "tier_info", None)
    if tiers:                                                    # plan 59: the status, the line, the tiers run, the tier that decided, seconds per tier
        sti = ag.status_info
        head = (f"status {sti['word']} (balance {sti['balance']:+d}, their threat {sti['threat']}), line {sti['line']}" if sti else "status mate (a mate of ours keeps every status)")
        log(f"OBS tiers {p['id']}: {head}, tiers run {[t['tier'] for t in tiers]}, kept {tiers[-1]['kept']}, decided by tier {ag.tier_decided if decided != "none" else "-"}, seconds per tier {[t['secs'] for t in tiers]}, decision {decided}")
    if ponder:                                                   # 52.19 (3): the background tier after the puzzle: the queue drained for `ponder` seconds
        from catspace.incubator.analyst.discovery import ponder as PD
        pr = PD.drain(ag.ctx, ag.us, ponder, log=lambda t: log(f"OBS ponder {p['id']}: {t}"))
        log(f"OBS ponder {p['id']}: drained {pr['done']} of {pr['queue']} queued items in {pr['secs']} s; queue after {pr['left']}")
    return dict(id=p["id"], asked=ag.asked, tempo=ag.killed_tempo, claims=len(st.claims), decision=decided, match=verdict == "MATCH",
                decided=d is not None, estimated=estimated, est_routes=ag.ctx.est_routes, kills=[r.split(":")[0] for _, r in ag.kills], gaps=len(st.gaps), secs=secs, theirs=ag.their_runs, fx=fx, chase=(cx.agendas, cx.inherited, cx.reopened, cx.budget_hits, cx.asked, cx.inherited_claims, cx.fresh, cx.closed, cx.standing, cx.logged, cx.gap_reused, cx.depth_cuts, cx.static_read, cx.static_parried, cx.static_unparried, cx.reasoned, cx.unaddressed, cx.time_outs, cx.landing_served, R.PASS_CLAIMS[0] - pc0, cx.roots_served, cx.chase_served, cx.finished, cx.unmet, cx.refuter_fixes), f61=(cx.parry_seen, cx.parry_closed, cx.delayed_routes, cx.nosafe_routes), until=ag.asked_at_claim, move=d.move if d else None, leaf=ag.leaf_stats,
                **(dict(tiers=tiers, status=ag.status_info, tier_decided=ag.tier_decided) if tiers else {}))


def _append_once(path, text: str) -> None:
    """Append `text` with ONE write() on an O_APPEND descriptor: parallel workers never interleave inside it."""
    import os
    if not text:
        return
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        data = text.encode()
        while data:
            n = os.write(fd, data)
            data = data[n:]
    finally:
        os.close(fd)


_FLAGS = ("PRIOR", "NEED_AWARE_CLOSE", "EVASION_CHASE", "NEED_PROPAGATION", "GOAL_JUDGE", "DEEP_WIN_CHASE", "CHASE_CLOCK", "TIERS", "TIERS_UNDER_TIME", "CLOSE_AFTER_CAPTURE", "RAISE_TO_WIN", "PARRIES", "DELAYED_THREAT", "NO_SAFE_MOVE", "ESTIMATE", "COST_JUDGE", "CYCLE")


def _init_worker(flags: dict) -> None:
    """A pool worker (spawned: module globals are not inherited): the agenda's switches as the parent set them."""
    from catspace.incubator.analyst.discovery import agenda as AGM
    for k, v in flags.items():
        setattr(AGM, k, v)
    BV.install_terms()


def _one(job: tuple) -> dict:
    p, kw = job
    return analyse(p, quiet=True, **kw)


def _write_qlog(p: dict, set_name: str, fen: str, ag, d, sfv, estimated: bool = False) -> None:
    """plan 53 (1): one row per question of the ROOT agenda in the order asked: what was known before it was asked (features), its outcome (status, the word that answered
    it: the claim's lemma or the kill's reason), whether it decided, the decision, the puzzle's answer and the Stockfish verdict. The prior trains on this."""
    import json
    ans = p["answer"]
    match = bool(d and d.move[:4] == ans[:4])
    import os
    from catspace.incubator.analyst.discovery import question_prior as QP0
    res61 = QP0.forcing_resources(ag.board, ag.us)               # plan 61 C3: the forcing resources of the root board (logged for plan 60, no decision use)
    buf = []                                                     # (the rows of one agenda are appended in ONE write: puzzles run in parallel processes, 2026-10-20)
    if True:
        from catspace.incubator.analyst.discovery import question_prior as QP
        for i, q in enumerate(ag.order):
            f = QP.features(q, ag.board, ag.us, 0)
            word = next((c.lemma for c in ag.state.claims if c.qid == q.id), None) if q.status == "verified" else None
            word = word or (q.claimed[1].lemma if q.claimed is not None else (q.reason.split(":")[0] if q.reason else q.status))
            t = q.pattern.get("target", q.pattern.get("at", q.pattern.get("pawn")))
            from catspace.incubator.analyst.discovery import agenda as AGM
            buf.append(json.dumps(dict(id=p["id"], set=set_name, prior=bool(AGM.PRIOR), fuse_hits=ag.ctx.budget_hits, fen=fen, order=i, qid=q.id, goal=q.goal, stake=q.stake, text=q.text, status=q.status, word=word,
                                     decided=bool(d and d.qid == q.id), decision=d.move if d else None, estimated=estimated, answer=ans, match=match, sf=sfv[0], sf_loss=sfv[1], sf_best=sfv[2],
                                     target_sq=t, move=q.pattern.get("move"), features=f, resources=res61)) + "\n")
    _append_once(os.environ.get("CATSPACE_QLOG") or paths.experiment("question_log.jsonl"), "".join(buf))      # (CATSPACE_QLOG: a run beside another that writes the default log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev_20")
    ap.add_argument("--id")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--store", action="store_true", help="write a record per decided puzzle (per-item checkpoint: stored ids are skipped)")
    ap.add_argument("--memory", action="store_true", help="recall stored states to order equal stakes (OFF by default: with 11 records it asked MORE questions on 5 of 60 fresh puzzles, 2026-10-17)")
    ap.add_argument("--leaf-all", action="store_true", help="52.10: read the leaf of every reply of theirs, not only the minimising ones (slower)")
    ap.add_argument("--seconds", type=float, default=None, help="52.17 (5): the budget is TIME: seconds per puzzle (default: none, the question-count fuse as before)")
    ap.add_argument("--ponder", type=float, default=None, help="52.19 (3): after each puzzle drain the background queue (gap routes by chain stake) for this many seconds")
    ap.add_argument("--no-memory", action="store_true", help="(kept for the batch scripts) the stake-only order, the default")
    ap.add_argument("--qlog", action="store_true", help="plan 53 (1): append the QUESTION LOG (the root agenda's questions in the order asked, which one decided, which word answered it) to experiments/question_log.jsonl")
    ap.add_argument("--sf", action="store_true", help="plan 53 (1): grade each decision with Stockfish (RIGHT within 50 cp of the best move, else WRONG); the log's label")
    ap.add_argument("--prior", action="store_true", help="plan 53 (2): the learned prior (data/derived/question_prior.json) orders the root agenda's questions")
    ap.add_argument("--workers", type=int, default=None, help="puzzles in parallel processes (default: all cores but one; 1 = sequential)")
    ap.add_argument("--no-chain", action="store_true", help="plan 64.2: switch the 52.23 chain items OFF (they are ON by default at every tier)")
    ap.add_argument("--chain", action="store_true", help="plan 52.23 (2)-(8) (the default since plan 64.2; kept for the harness): the chain items ON (the need-aware close, the in-check chase, win chases below depth two, the goal judge, the need propagated, the chase clock 4) -- the regime for hard positions under a time budget (x00hu, 2026-10-20)")
    ap.add_argument("--tiers", action="store_true", help="plan 59: STATUS-DRIVEN WIDENING -- the root runs tier 0, then wider tiers (quiet threats below the root; mate-in-two threats; the 52.23 chain items) only while no claim keeps our status; under --seconds each tier keeps its count fuse and the tiers share the time")
    ap.add_argument("--parries", action="store_true", help="plan 61 A2: a closed reply that does not parry the route's threat says so (\"closed by the threat\"); the same values")
    ap.add_argument("--delayed", action="store_true", help="plan 61 A3: a quiet move after which every reply allows a forcing move of ours is a route (tier 1+, the root's chase)")
    ap.add_argument("--nosafe", action="store_true", help="plan 61 A4: a quiet move after which every reply loses at least the stake by rule (zugzwang) is a route (tier 1+, the root's chase)")
    ap.add_argument("--forcing61", action="store_true", help="plan 61 stage A: --parries --delayed --nosafe")
    ap.add_argument("--estimate", action="store_true", help="plan 61 B2: a cut route whose open replies are quiet leaves closes as an ESTIMATED claim (quiet-status); the decision consults it only when no proved claim keeps the status; every estimated leaf is logged to estimates_log.jsonl")
    ap.add_argument("--cost-judge", action="store_true", help="plan 61 C2: under --seconds the cost judge (cost x seconds per node <= the route's time left) replaces the cut by the clock; a cut route is a GAP; the trace says 'judge: ...'")
    ap.add_argument("--no-cost-judge", action="store_true", help="plan 64.3: switch the cost judge OFF (it is ON by default under a time budget)")
    ap.add_argument("--cycle", action="store_true", help="plan 61 C4 (the default; kept for the harness): a chase position that repeats one of the line (or the game history) closes as a proved 'draw' claim worth minus the balance when we are behind")
    ap.add_argument("--no-cycle", action="store_true", help="plan 61 C4: CYCLE off (it is ON by default)")
    ap.add_argument("--no-delayed", action="store_true", help="plan 63.3: switch the delayed threat (61 A3, ON by default) OFF")
    ap.add_argument("--duty", action="store_true", help="plan 63.2: the duty word ('pinned to a threat') in the trap/run test (the default: ON)")
    ap.add_argument("--no-duty", action="store_true", help="plan 63.2: switch the duty word OFF (CATSPACE_DUTY=0, inherited by the workers)")
    a = ap.parse_args()
    if a.no_delayed and not a.delayed and not a.forcing61:
        from catspace.incubator.analyst.discovery import agenda as _AG63
        _AG63.DELAYED_THREAT = False
    if a.no_duty and not a.duty:
        from catspace.incubator.analyst.discovery import regress as _R63
        os.environ["CATSPACE_DUTY"] = "0"
        _R63.DUTY = False
    if a.cost_judge or a.no_cost_judge or a.cycle or a.no_cycle:
        from catspace.incubator.analyst.discovery import agenda as AGM
        AGM.COST_JUDGE = (AGM.COST_JUDGE or a.cost_judge) and not a.no_cost_judge
        AGM.CYCLE = (AGM.CYCLE or a.cycle) and not a.no_cycle
    if a.parries or a.delayed or a.nosafe or a.forcing61:
        from catspace.incubator.analyst.discovery import agenda as AGM
        AGM.PARRIES = AGM.PARRIES or a.parries or a.forcing61
        AGM.DELAYED_THREAT = AGM.DELAYED_THREAT or a.delayed or a.forcing61
        AGM.NO_SAFE_MOVE = AGM.NO_SAFE_MOVE or a.nosafe or a.forcing61
    if a.estimate:
        from catspace.incubator.analyst.discovery import agenda as AGM
        AGM.ESTIMATE = True
    if a.tiers:
        from catspace.incubator.analyst.discovery import agenda as AGM
        AGM.TIERS = True
    if a.prior:
        from catspace.incubator.analyst.discovery import agenda as AGM
        AGM.PRIOR = True
    if a.chain:
        from catspace.incubator.analyst.discovery import agenda as AGM
        for k in ("NEED_AWARE_CLOSE", "EVASION_CHASE", "NEED_PROPAGATION", "GOAL_JUDGE", "DEEP_WIN_CHASE"):
            setattr(AGM, k, True)
        AGM.CHASE_CLOCK = 4
    if a.no_chain:                                               # plan 64.2: the chain items are ON by default; this is the comparison condition
        from catspace.incubator.analyst.discovery import agenda as AGM
        for k in ("NEED_AWARE_CLOSE", "EVASION_CHASE", "NEED_PROPAGATION", "GOAL_JUDGE", "DEEP_WIN_CHASE"):
            setattr(AGM, k, False)
        AGM.CHASE_CLOCK = 99
    if a.sf:                                                     # plan 53 (1): the engine beside an agenda stalled the set runs (its thread; 2026-10-20): grade the log in its own pass instead
        raise SystemExit("--sf is retired: run `question_prior grade` on the log after `analyse --qlog` (the grader never runs beside an agenda)")
    from catspace.incubator.analyst.discovery.agenda import ChaseCtx
    carry = ChaseCtx() if a.memory else None                     # 52.19 (4): --memory also keeps the fact store (and the exact memos) between the puzzles
    ps = PA.load_set(a.set)
    if a.store:
        done = RM.stored_ids(paths.derived(AGENDA_STORE))
        ps = [p for p in ps if p["id"] not in done] if not a.id else ps
    if a.id:
        analyse(next(x for x in ps if x["id"] == a.id), memory=a.memory and not a.no_memory, store=a.store, set_name=a.set, leaf_all=a.leaf_all, seconds=a.seconds, carry=carry, ponder=a.ponder, qlog=a.qlog, sf=a.sf)
        return
    kw = dict(memory=a.memory and not a.no_memory, store=a.store, set_name=a.set, leaf_all=a.leaf_all, seconds=a.seconds, ponder=a.ponder, qlog=a.qlog, sf=a.sf)
    workers = a.workers if a.workers is not None else max(1, (os.cpu_count() or 2) - 1)      # Kaveh's standing rule: every scan on all cores (exchange_law._workers)
    t0 = time.monotonic()
    if workers > 1 and len(ps) > 1 and carry is None and not a.store:                 # (--memory carries the fact store between puzzles and --store writes one file: those stay sequential)
        import multiprocessing as mp
        from catspace.incubator.analyst.discovery import agenda as AGM
        flags = {k: getattr(AGM, k) for k in _FLAGS}
        with mp.get_context("spawn").Pool(min(workers, len(ps)), initializer=_init_worker, initargs=(flags,)) as pool:
            res = pool.map(_one, [(p, kw) for p in ps], chunksize=1)
    else:
        workers = 1
        res = [analyse(p, quiet=True, carry=carry, **kw) for p in ps]
    print(f"OBS set wall time {a.set}: {time.monotonic() - t0:.1f} s on {workers} worker(s) for {len(ps)} puzzles (the per-puzzle seconds below are each puzzle's own)")
    if a.store:
        print(f"OBS agenda store: {len(RM.load(paths.derived(AGENDA_STORE)))} records")
    for r in res:
        print(f"OBS questions asked until the deciding claim {r['id']}: {r['until']} (of {r['asked']} asked), decision {r['decision']} {r['move']}")
    print(f"OBS questions asked until the deciding claim, total: {sum(r['until'] for r in res)} (of {sum(r['asked'] for r in res)} asked)")
    print(f"OBS forcing61 {a.set}: closed replies checked against the route's threat {sum(r['f61'][0] for r in res)}, closed by the threat (no parry) {sum(r['f61'][1] for r in res)}, "
          f"routes admitted by the delayed threat {sum(r['f61'][2] for r in res)}, by no safe move {sum(r['f61'][3] for r in res)}")
    if any(r["est_routes"] for r in res):
        print(f"OBS estimate {a.set}: gap routes closed as estimated {sum(r['est_routes'] for r in res)} on {sum(1 for r in res if r['est_routes'])} puzzles, decisions ESTIMATED {sum(r['estimated'] for r in res)} ({' '.join(r['id'] for r in res if r['estimated'])})")
    kills = Counter(k for r in res for k in r["kills"])
    dec = [r["id"] for r in res if r["decided"]]
    gap = [r["id"] for r in res if not r["decided"]]
    print(f"OBS leaf {a.set}: claims read at the leaf {sum(r['leaf']['read'] for r in res)}, downgraded {sum(r['leaf']['down'] for r in res)}, "
          f"not trusted {sum(r['leaf']['untrusted'] for r in res)}")
    print(f"OBS agenda {a.set}: {len(res)} puzzles, decided {len(dec)} ({' '.join(dec)}), MATCH {sum(r['match'] for r in res)}, "
          f"not decided {len(gap)} ({' '.join(gap)}), questions asked {sum(r['asked'] for r in res)}, killed by tempo {sum(r['tempo'] for r in res)}, "
          f"killing reasons {dict(kills.most_common())}, their agendas {sum(r['theirs'] for r in res)}, chase: inner agendas {sum(r['chase'][0] for r in res)}, "
          f"inner questions {sum(r['chase'][4] for r in res)} (mean {sum(r['chase'][4] for r in res) / max(sum(r['chase'][0] for r in res), 1):.1f} per inner agenda), "
          f"inherited {sum(r['chase'][1] for r in res)} (claims {sum(r['chase'][5] for r in res)}), reopened {sum(r['chase'][2] for r in res)}, new {sum(r['chase'][6] for r in res)}, fuse hits {sum(r['chase'][3] for r in res)} (depth {sum(r['chase'][11] for r in res)}), gap routes reused {sum(r['chase'][10] for r in res)}, replies closed by rule {sum(r['chase'][7] for r in res)}, replies left open (goal unmet) {sum(r['chase'][23] for r in res)}, refuter fixes {sum(r['chase'][24] for r in res)}, standing kills {sum(r['chase'][8] for r in res)}, chase log rows {sum(r['chase'][9] for r in res)}, static leaves read {sum(r['chase'][12] for r in res)}, parried {sum(r['chase'][13] for r in res)}, unparried {sum(r['chase'][14] for r in res)}, killed by reasoning {sum(r['chase'][15] for r in res)}, routes not addressing a defect {sum(r['chase'][16] for r in res)}, time outs {sum(r['chase'][17] for r in res)}, landings served from the table {sum(r['chase'][18] for r in res)}, forcing by a claim after a pass {sum(r['chase'][19] for r in res)}, "
          f"facts derived {sum(r['fx']['derived'] for r in res)}, served from the store {sum(r['fx']['served'] for r in res)}, foreground recomputations {sum(r['fx']['recomputed'] for r in res)}, background queue at decision {sum(r['fx']['queue'] for r in res)} (top stake {max(r['fx']['top'] for r in res)}), seconds {round(sum(r['secs'] for r in res), 1)}, "
          f"root lists served {sum(r['chase'][20] for r in res)}, chase claims served {sum(r['chase'][21] for r in res)}, routes finished after the deadline {sum(r['chase'][22] for r in res)}")


if __name__ == "__main__":
    main()
