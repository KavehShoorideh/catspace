"""PLAN 56 (1) THE QUESTION SCHEDULER: the chase as an explicit TASK GRAPH on a priority queue, run by ONE worker loop in ONE process -- not a recursion.

The recursive chase of agenda.py (kept, reachable with `agenda.SCHEDULER = False`, the default) asks a root to its end: its routes, every reply, the inner agenda on each unclosed reply,
recursively. Here every step of that recursion is a TASK, and a task that needs another does not call it -- it YIELDS it and is suspended (it WAITS ON it); the worker loop pops the
highest-priority READY task, runs it to its next wait, and when a task ends its result is handed to the task that waited on it (a CONTINUATION). The kinds:

  node     an agenda's pool of questions (the root's, or the inner agenda of a reply): pops its questions in the order of today's heap (stake, the recall score, depth, creation sequence; the prior
           at the root when PRIOR is on) and waits on one `ask` at a time
  ask      one question at a node: read its defects, fixes, routes; verify; else open a `chase`
  chase    the forcing routes of a question that would be killed: opens a `route` per forcing move (the cheap pass, then the full pass), then the refuter fixes
  route    one forcing move: every reply of theirs read at the static leaf; what the rules close is closed; each other reply spawns a `reply`
  reply    the inner agenda on the position after the route and the reply (memoised by position): opens a `node`
  their_pass  their agenda with the move passed to them (a leaf: it chases nothing; recorded, run in line by the task that needs it)

The values travel by continuation with EXACTLY the rules of agenda.py -- a route's value is the least over its replies (`route_value`), a question's claim is its best route
(`best_route`, the tie-break of 53 (3) at the node's `_better`), a question is killed when no route claims (`question_killed`), a node's answer is its best claim against its need -- and
the shared state (the fuse `ctx.left` charged in the order the routes are read, the memos, the deadline, the fact store's listener) is read and written in today's order, because in stage 1 the
priority IS today's order: only one task is ever ready, since a parent needs its child's value before it goes on. The ported bodies below are agenda.py's, line for line, with each call that
used to recurse written as a yield of a task. Stage 2 (not here) adds urgency, re-prioritisation and the prior at every node; stage 3 workers as processes.

THE PORT IS A COPY: the generator bodies (`*_g`) mirror agenda.py's `_ask_loop`, `_ask`, `_ask_*`, `_verify`, `_chase*`, `_inner`, `_refuter_fix*`; a change to one must be made to the other
until the recursive path is retired. `scripts/compare_scheduler.py` is the guard: every decision, every claim and the counters, flag off and on, in separate processes.
The background queue of 52.19 (ponder.py) is NOT integrated in stage 1: the Ponderer drains the fact store's gap queue between moves on the recursive path (its agendas are not roots).
"""
from __future__ import annotations

import dataclasses
import heapq
import itertools
import time

import analyst_core as AC
import chess

from catspace.incubator.analyst import lean_passer_v2_proto as PV
from catspace.incubator.analyst.discovery import agenda as AG
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import binder as BD
from catspace.incubator.analyst.discovery import facts as FS
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import landing as LD
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import question_prior as QP
from catspace.incubator.analyst.discovery import reasoned as RS
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.state import Claim, State


# ---------------------------------------------------------------------------------------------------------------------------- the rules (the values travel by these)

CORE_RUST = False        # plan 56 (2a): the task graph, the heap and the rules live in Rust (analyst_core.Scheduler, sched_*); the generator bodies below stay Python. The same answers (scripts/compare_scheduler.py --core)

ROUTE_RUST = False       # plan 56 (2b): the route's replies are read by the Rust handler analyst_core.read_route (one call per route); the effects stay in the Python body. The same answers (scripts/compare_scheduler.py --route)


def route_value(gains: list) -> int:
    """A route's value: the least over its replies (each closed by a rule or answered by an inner agenda); a route with no reply read is worth 0."""
    if CORE_RUST:
        return AC.sched_route_value(list(gains))
    return min(gains) if gains else 0


def best_route(done: list) -> tuple:
    """A question's claim: its best route -- the greatest value, the move's uci for the tie. `done` is [(value, route, result)]; it is sorted in place. -> the best triple."""
    if CORE_RUST:
        return done[AC.sched_best_route([t[0] for t in done], [t[1].move.uci() for t in done])]
    done.sort(key=lambda t: (-t[0], t[1].move.uci()))
    return done[0]


def question_killed(done: list) -> bool:
    """A chased question is killed when no route claims (every route was killed, cut, or worth less than 1)."""
    if CORE_RUST:
        return AC.sched_question_killed(len(done))
    return not done


# ---------------------------------------------------------------------------------------------------------------------------- tasks and the worker loop

class Task:
    """One node of the task graph. `gen` is the suspended computation (a generator); `waits` the tasks it is suspended on; `parent` the task waiting on this one."""
    __slots__ = ("kind", "gen", "node", "prio", "seq", "parent", "waits", "value", "exc", "started", "done", "result", "error", "label", "depth", "children", "id")

    def __init__(self, kind: str, gen, node=None, prio: tuple = (), label: str = ""):
        self.kind, self.gen, self.node, self.prio, self.label = kind, gen, node, prio, label
        self.seq = -1
        self.id = -1                                        # the id in the Rust core (CORE_RUST)
        self.parent = None
        self.waits: list = []
        self.value = None                                  # what the child it waited on answered (sent in when it is resumed)
        self.exc = None                                    # ... or what the child raised (thrown in)
        self.started = self.done = False
        self.result = None
        self.error = None
        self.depth = 0
        self.children: list = []                           # the graph, when the scheduler records it

    def __repr__(self) -> str:
        return f"<{self.kind} {self.label} {'done' if self.done else 'waiting on %d' % len(self.waits) if self.waits else 'ready'}>"


def T(kind: str, gen, node=None, prio: tuple = (), label: str = "") -> Task:
    return Task(kind, gen, node, prio, label)


class Scheduler:
    """The worker loop. Two queues (URGENT, BACKGROUND) of ready tasks; stage 1 uses the urgent queue only (everything the current decision waits on). A task's priority key is
    (queue, its priority tuple, the order it became ready): for an `ask` task the question's heap tuple (-stake, -recall score, depth, creation sequence), so the lowest key is today's order."""
    URGENT, BACKGROUND = 0, 1

    def __init__(self, record: bool = False):
        self.heap: list = []
        self.count = itertools.count()
        self.record = record                               # keep the graph (every task's children) -- the tests and the harness read it
        self.counts: dict = {}                             # tasks run, by kind
        self.leaves: dict = {}                             # leaf tasks (their_pass), by kind: [asked, served from the memo]
        self.ready_max = 0
        self.steps = 0
        self.root = None
        self.core = AC.Scheduler() if CORE_RUST else None   # the graph and the order in Rust; `tasks` maps its ids to the Python tasks (generators, nodes)
        self.tasks: dict = {}

    def push(self, t: Task, queue: int = 0) -> None:
        heapq.heappush(self.heap, ((queue, t.prio, next(self.count)), t))
        self.ready_max = max(self.ready_max, len(self.heap))

    def leaf(self, kind: str, node, key, served: bool) -> None:
        """A task that needs no suspension (it spawns nothing on the chase), run in line by the task that needs it: recorded as a leaf."""
        row = self.leaves.setdefault(kind, [0, 0])
        row[0] += 1
        row[1] += 1 if served else 0

    def submit(self, t: Task, parent: "Task | None") -> None:
        t.parent = parent
        t.depth = parent.depth + 1 if parent is not None else 0
        t.seq = next(self.count)
        if self.record and parent is not None:
            parent.children.append(t)
        self.counts[t.kind] = self.counts.get(t.kind, 0) + 1
        if self.core is not None:
            t.id = self.core.push(t.kind, parent.id if parent is not None else None, 0, [float(x) for x in t.prio])
            self.tasks[t.id] = t
            return
        self.push(t)

    def _finish(self, t: Task) -> None:
        t.done = True
        t.gen = None
        if self.core is not None:
            self.core.complete(t.id, None if t.error is not None else t.result)       # wakes the parent when this was the last child it waited on
            if not self.record:
                self.tasks.pop(t.id, None)
        p = t.parent
        if p is None:
            return
        p.waits.remove(t)
        if t.error is not None:
            p.exc = t.error                                 # an exception travels up the graph as it did up the stack: thrown into the task that waited, whose `finally` clauses run
        else:
            p.value = self.core.result(t.id) if self.core is not None else t.result       # (the Rust core's result slot)
        if self.core is None and not p.waits:
            self.push(p)

    def run(self, root: Task):
        """Run the graph from `root` to its end: -> the root's result (or raise what it raised)."""
        self.root = root
        self.submit(root, None)
        while True:
            if self.core is not None:
                i = self.core.pop()
                if i is None:
                    break
                t = self.tasks[i]
            else:
                if not self.heap:
                    break
                _, t = heapq.heappop(self.heap)
            self.steps += 1
            try:
                if not t.started:
                    t.started = True
                    req = next(t.gen)
                elif t.exc is not None:
                    exc, t.exc = t.exc, None
                    req = t.gen.throw(exc)
                else:
                    val, t.value = t.value, None
                    req = t.gen.send(val)
            except StopIteration as e:
                t.result = e.value
                self._finish(t)
            except BaseException as e:                      # noqa: BLE001 -- carried up to the task that waited
                t.error = e
                self._finish(t)
            else:
                kids = list(req) if isinstance(req, (list, tuple)) else [req]
                t.waits = list(kids)
                for k in kids:
                    self.submit(k, t)
                    if self.core is not None:
                        self.core.wait(t.id, k.id)
        if self.core is not None:
            self.ready_max = self.core.stats()["ready_max"]
        if root.error is not None:
            raise root.error
        return root.result


def run_root(agenda) -> None:
    """The dispatch point of `Agenda.ask` (agenda.SCHEDULER on): the root agenda's loop is the root node task of a graph run by one worker loop."""
    sched = Scheduler()
    agenda.ctx.sched = sched
    try:
        sched.run(T("node", agenda_g(agenda), agenda))
    finally:
        agenda.ctx.sched = None
    agenda.ctx.stats = dict(sched.counts, leaves=dict(sched.leaves), steps=sched.steps, ready_max=sched.ready_max)


def agenda_g(node):
    """The node task: the agenda's pool of questions, then what `ask` adds to the trace."""
    yield from ask_loop_g(node)
    node._ask_finish()


# ---------------------------------------------------------------------------------------------------------------------------- the ported bodies (agenda.py's, with each recursion a yield)

def ask_loop_g(node) -> None:
    while True:
        if not node._threats_done and not (node.best and node.best[0] >= R.STAKE_MATE) and (not node.heap or node.heap[0][4].stake < R.STAKE_MATE):
            node._threats_done = True                        # a mate of ours ends the game, so their threats are asked only once no mate root is left to ask
            node._threat_roots(node._hang)
        if not node.heap:
            break
        nstake = node.heap[0][4].stake
        if node.best and (node.best[0] > nstake or node.best[0] >= BD.MATE_VALUE):      # 53 (3): STRICTLY above the next stake (a question of equal stake may still claim as much with a better shape; the order asked must not decide), a mate ends everything
            node.lines.append(f"STOP (a): the best claim ({node.best[2].what}, {node.best[0]}) is above the stake of every open question (next: {nstake})")
            return
        if node.asked >= node.budget:
            left = [t[4] for t in node.heap]
            node.lines.append(f"STOP (c): budget {node.budget} questions asked; {len(left)} open questions become gaps")
            node.state.gaps += [f"budget: {q.text} (stake {q.stake}) not asked" for q in left]
            return
        if node.chase_depth and node.ctx.out_of_time() and not (node.late and node.ctx.finishing(node.frame)):       # 52.17 (5): an inner agenda that meets the deadline stops where it is (its answer is cut, never memoised, never read as "no claim"); 52.21 (3): the ones opened by the route in hand read on
            node.ctx.budget_hits += 1
            node.lines.append("STOP (t): out of time")
            return
        plain = min(node.heap, key=lambda t: (t[0], t[2], t[3]))                # the stake-only order
        if AG.PRIOR and not node.chase_depth and QP.load() is not None:          # plan 53 (2): THE PRIOR ORDERS the root agenda's pool: the open question with the highest score is asked next
            scored = [(QP.score(QP.features(t[4], node.board, node.us, node.chase_depth)), -t[0], -t[3], t) for t in node.heap]        # (it orders, it never decides: the stops stay by stakes and claims)
            best = max(scored)[3]
            node.heap.remove(best)
            heapq.heapify(node.heap)
            item = best
            q = best[4]
            if plain[4] is not q:
                node.lines.append(f"prior: Q{q.id} asked first (score {max(scored)[0]:+.2f}; the stake order would ask Q{plain[4].id})")
                node.prior_moves += 1
        else:
            item = heapq.heappop(node.heap)
            q = item[4]
            if plain[4] is not q:
                sc, rec = node._recall_score(q)
                node.lines.append(f"memory: Q{q.id} moved up by {sc} (seen {rec['id']}: {rec['goal']} by {rec['lemma']})")
        node.order.append(q)                                   # plan 53 (1): the order asked (the features are read at log time: the root board does not change while its agenda runs)
        if node._inherit(q):
            continue
        if node.chase_depth:                                 # an inner agenda pays the shared budget as it asks (a chain of checks meets it on the way down)
            node.ctx.left -= 1
            before = node.asked
        node._cur, node._qdeps, n0 = q, [], len(node.state.claims)
        (yield T('ask', ask_g(node, q), node, prio=item[:4]))
        if node.ctx is not None:
            node._file(q, node.state.claims[n0:])
        if node.chase_depth:
            node.ctx.left -= max(node.asked - before - 1, 0)
    node.lines.append("STOP (b): every question is closed")


def ask_g(node, q: R.Question) -> None:
    b, us = node.board, node.us
    node.asked += 1
    head = f"Q{q.id} {q.text} (stake {q.stake})"
    if q.goal == "save" and "checkers" in q.pattern:
        (yield from ask_save_king_g(node, q, head))
        return
    if q.goal == "promote":
        (yield from ask_promote_g(node, q, head))
        return
    if q.goal == "fork":
        node._ask_fork(q, head)
        return
    if q.goal == "mate" and node._reasoned_kill(q, head):
        return
    if q.goal == "mate" and q.pattern.get("reach"):
        (yield from ask_reach_g(node, q, head))
        return
    if q.goal == "save" and "promoter" in q.pattern:
        (yield from ask_stop_pawn_g(node, q, head))
        return
    if q.goal == "save" and "threat" in q.pattern:
        (yield from ask_save_threat_g(node, q, head))
        return
    if q.goal == "save":
        (yield from ask_save_piece_g(node, q, head))
        return
    if q.pattern.get("checker"):                             # 52.15 (1): the checker taken by any legal capture, read by the schemas as a win root's routes are
        node.lines.append(head + ": the checker is taken by a legal capture (no defect to fix)")
        (yield from verify_g(node, q, node._direct_routes(q), mates_only=False))
        return
    if q.goal == "mate":
        q.defects = node._landing_defects(q)
    else:
        q.defects = R.material_defects(b, us, q.pattern["target"])
    node.lines.append(head + (": " + "; ".join(f"defect {d.kind} ({d.why})" for d in q.defects) if q.defects else ": no defect -- the pattern already stands"))
    if not q.defects:
        routes = node._direct_routes(q)
    else:
        for d in q.defects:
            R.fixes_of(d, q, b, us)
        node.asked += sum(node._count(f) for f in q.fixes)
        for f in q.fixes:
            node._trace_fix(f, 1)
        if not R.tempo_ok(q):
            if (yield T('chase', chase_g(node, q), node)):
                return
            node._kill(q, q.reason, tempo=True)
            return
        node.lines.append(f"  tempo: {len(q.defects)} defect(s), each has a forcing fix")
        routes = node._merge_routes(q)
    (yield from verify_g(node, q, routes, mates_only=q.goal == "mate"))


def ask_reach_g(node, q: R.Question, head: str) -> None:
    """52.12 (4): the root whose only defect is the reach. The fix routes are the first steps; a step is FORCING when it is a check or after it a pass leaves us a
    mate in one (a mate threat: their every reply must meet it). The binder reads each over every reply; when none binds the chase regresses through their replies."""
    b, us = node.board, node.us
    d = q.defects[0]
    node.lines.append(head + f": defect reach ({d.why})")
    steps = [chess.Move.from_uci(u) for u in q.pattern["reach"]]
    f = R.new_question("cover", q.stake, dict(q.pattern), q, defect_key=d.key(), text=f"step toward {G.name(q.pattern['at'])} with tempo")
    forcing, quiet = [], []
    for m in steps:
        why = R.forcing_why(b, m, q.stake, us, mate_threats=True)
        (forcing if why else quiet).append(R.Route(m, b.san(m), why) if why else b.san(m))
    f.routes, f.unforced = forcing, quiet
    q.fixes.append(f)
    node.asked += 1
    node.lines.append(f"  -> Q{f.id} {f.text}: " + (", ".join(f"{r.san} ({r.why})" for r in forcing) if forcing else "no forcing step" + (f" (quiet steps: {', '.join(quiet[:4])})" if quiet else "")))
    q.reads = R.reads_of(q, b)
    if not forcing:
        if (yield T('chase', chase_g(node, q), node)):
            return
        node._kill(q, "tempo: reach: only quiet steps, the opponent gets a free reply" if quiet else "tempo: reach: no step", tempo=True)
        return
    (yield from verify_g(node, q, forcing, mates_only=True))


def ask_save_king_g(node, q: R.Question, head: str) -> None:
    b = node.board
    ck = q.pattern["checkers"]
    q.defects = [R.Defect("in check", c, [], f"{b.piece_at(c).symbol().upper()} on {G.name(c)} gives check") for c in ck]
    node.lines.append(head + ": " + "; ".join(f"defect {d.kind} ({d.why})" for d in q.defects))
    takes = [R.Route(m, b.san(m), f"capture {G.worth(b.piece_at(m.to_square))}") for m in b.legal_moves if m.to_square in ck]
    others = [R.Route(m, b.san(m), "evade") for m in b.legal_moves if m.to_square not in ck]
    f1 = R.new_question("remove", q.stake, dict(q.pattern), q, defect_key=q.defects[0].key(), text="take the checker")
    f1.routes = takes
    node.asked += 1
    node.lines.append(f"  -> Q{f1.id} {f1.text}: " + (", ".join(f"{r.san} ({r.why})" for r in takes) or "no capture of the checker"))
    q.fixes.append(f1)
    before = len(node.state.claims)
    if takes:
        (yield from verify_g(node, q, takes, mates_only=False))
    if len(node.state.claims) == before and others:
        f2 = R.new_question("cover", q.stake, dict(q.pattern), q, defect_key=q.defects[0].key(), text="step away or interpose")
        f2.routes = others
        node.asked += 1
        node.lines.append(f"  -> Q{f2.id} {f2.text}: " + ", ".join(r.san for r in others))
        q.fixes.append(f2)
        (yield from verify_g(node, q, others, mates_only=False))
    if len(node.state.claims) == before and not node._decide_by_smallest_loss(q) and node._t(AG.EVASION_CHASE, 3):
        (yield T('chase', chase_g(node, q), node))                                       # 52.23 (3): an INNER agenda in check chases its evasions (every legal move answers the check, so each is a route): zBouR's Rhxf2 cxb7+ Kb8 Rxf2 Qh1+


def ask_save_piece_g(node, q: R.Question, head: str) -> None:
    b = node.board
    sq = q.pattern["target"]
    parries = RS._threat_moves(b, node.us, [sq])
    q.defects = [R.Defect("hangs", sq, sorted(b.attackers(not node.me, sq)), f"they win {G.see(b, sq)} by the exchange on {G.name(sq)}")]
    node.lines.append(head + f": defect hangs ({q.defects[0].why})")
    f = R.new_question("remove", q.stake, dict(q.pattern), q, defect_key=q.defects[0].key(), text="parry it, or check, or take as much")
    f.routes = [R.Route(m, b.san(m), "parry") for m in parries]
    node.asked += 1
    node.lines.append(f"  -> Q{f.id} {f.text}: {len(f.routes)} moves")
    q.fixes.append(f)
    before = len(node.state.claims)
    (yield from verify_g(node, q, f.routes, mates_only=False))
    if len(node.state.claims) == before and not node._decide_by_smallest_loss(q):
        node.state.gaps.append(f"Q{q.id} {q.text}: no parry binds a guaranteed gain and none could be read from their view")


def ask_save_threat_g(node, q: R.Question, head: str) -> None:
    """A root from THEIR agenda on the pass board: parries are our moves of the threatened pieces, checks, captures of the threatening piece, and moves that add
    an attacker of a threatened piece (all legal moves for a mate threat); each is verified by the binder and viewed by their agenda on the board after it."""
    b = node.board
    p = q.pattern
    m = chess.Move.from_uci(p["threat"])
    q.defects = [R.Defect("threatened", m.to_square, [m.from_square], f"their {p['what']} wins {p['value']} if we pass")]
    node.lines.append(head + f": defect threatened (their view: {p['what']} wins {p['value']}, lemma {p['lemma']}, {p['line']})")
    tg = p["targets"]
    parries = []
    for mv in b.legal_moves:
        if p["value"] >= BD.MATE_VALUE or mv.from_square in tg or b.gives_check(mv) or (b.is_capture(mv) and mv.to_square == m.from_square):
            parries.append(mv)
            continue
        c = b.copy(stack=False)
        c.push(mv)
        if any(len(c.attackers(node.me, t)) > len(b.attackers(node.me, t)) for t in tg):
            parries.append(mv)
    f = R.new_question("remove", q.stake, dict(q.pattern), q, defect_key=q.defects[0].key(), text="parry the threat (move the target, check, take the threatener, add a guard)")
    f.routes = [R.Route(mv, b.san(mv), "parry") for mv in parries]
    node.asked += 1
    node.lines.append(f"  -> Q{f.id} {f.text}: {len(f.routes)} moves")
    q.fixes.append(f)
    if node.their is not None:                               # 52.14: a threat on a static leaf is parried by KEEPING the gain (the binder reads a gain of at least 1 from a balance <= 0, and we are ahead here)
        node._parry_keeps(q, f.routes)
        return
    before = len(node.state.claims)
    (yield from verify_g(node, q, f.routes, mates_only=False))
    if len(node.state.claims) == before and not node._decide_by_smallest_loss(q):
        node.state.gaps.append(f"Q{q.id} {q.text}: no parry binds a guaranteed gain and none could be read from their view")


def ask_stop_pawn_g(node, q: R.Question, head: str) -> None:
    """THEIR passer, a claim of their agenda, as OUR save root: the routes are the stoppers' moves (our king runs toward the promotion square, a capture of
    the pawn); the binder verifies them as any save root; the tempo arithmetic says `forced` when our king's distance equals the pawn's pushes."""
    b = node.board
    p = q.pattern
    pw = p["promoter"]
    them = not node.me
    n = G.word(b, ("pushes", G.name(pw)), them)
    promo = R.pawn_path(b, pw, n)[-1]
    kd = G.word(b, ("dist", "L", ("promo", G.name(pw))), them)               # their pawn's words, read from its owner's side: L = our king
    q.defects = [R.Defect("promotes", promo, [pw], f"their {p['what']} promotes on {G.name(promo)} in {n} ({p['value']}, lemma {p['lemma']})")]
    node.lines.append(head + f": defect promotes (their view: {p['what']}, {p['line']})")
    arith = f"our K reaches {G.name(promo)} in {kd}, the pawn needs {n}"
    ksq = b.king(node.me)
    runs = []
    for m in b.legal_moves:
        if m.to_square == pw:
            runs.append(R.Route(m, b.san(m), f"capture {G.worth(b.piece_at(pw))}"))
        elif m.from_square == ksq:
            c = b.copy(stack=False)
            c.push(m)
            if G.word(c, ("dist", "L", ("promo", G.name(pw))), them) < kd:
                runs.append(R.Route(m, b.san(m), "K runs back"))
    f = R.new_question("remove", q.stake, dict(q.pattern), q, defect_key=q.defects[0].key(), text="stop it: our king runs toward the promotion square, or takes the pawn")
    f.routes = runs
    node.asked += 1
    node.lines.append(f"  -> Q{f.id} {f.text}: {arith}; " + (", ".join(f"{r.san} ({r.why})" for r in runs) or "no such move"))
    q.fixes.append(f)
    # WE move first on the real board and the pawn is promoted by its nth push, so we make n moves before it; the king stops it when it is next to the
    # promotion square then (it takes the new queen), i.e. distance <= n + 1 (the same inequality as the lemma's kingOutside, n < dist, for the defender
    # with n - 1 moves). Distance = n + 1 is exact: every move of ours is spent.
    if kd == n + 1:
        node.lines.append(f"     forced: every move of ours is spent stopping the pawn (distance {kd} = pushes {n} + 1: next to {G.name(promo)} after our {n} moves)")
    elif kd > n + 1:
        node.lines.append(f"     no stopper in time: distance {kd} > pushes {n} + 1")
    else:
        node.lines.append(f"     slack: distance {kd} <= pushes {n}: {n - kd + 1} spare move(s)")
    before = len(node.state.claims)
    if node.their is not None:                               # 52.14: on a static leaf the stopper keeps the gain or not
        node._parry_keeps(q, runs)
        return
    if runs:
        (yield from verify_g(node, q, runs, mates_only=False))
    if len(node.state.claims) == before:
        node.lines.append(f"  no route binds a guaranteed gain against the passer (the binder has no promotion tail); {arith}")
        if kd > n + 1:
            node.state.gaps.append(f"Q{q.id} {q.text}: no stopper in time ({arith}) and no capture binds")


def ask_promote_g(node, q: R.Question, head: str) -> None:
    """The defects of the pawn's path are read with the words; a stopper in time kills the question WITH THE DISTANCE REASON; no defect = unstoppable by the
    tempo arithmetic, and then the claim is made ONLY when the proved passer (Ideas.Passer2.unstoppablePasser2_sound, via proved_passers) fires; if it does
    not, the question ends as a GAP naming the hypothesis that fails."""
    b = node.board
    sq = q.pattern["pawn"]
    q.defects, facts = R.promote_defects(b, sq)
    n, promo = facts["pushes"], facts["promo"]
    q.reads = R.reads_of(q, b) | frozenset(facts["path"]) | {sq}            # 52.16 (4): the pawn, its path (the promotion square included), the defects' squares and their pieces: what the question read
    kd = facts["king_dist"]
    node.lines.append(head + (": " + "; ".join(f"defect {d.kind} ({d.why})" for d in q.defects) if q.defects else
                              ": no defect -- unstoppable by the tempo arithmetic" + (f" (their K needs {kd} to {G.name(promo)}, the pawn {n})" if kd is not None else "")))
    if q.defects:
        for d in q.defects:
            f = R.promote_fixes(b, sq, d, q)
            rs = ", ".join(f"{r.san} ({r.why})" for r in f.routes)
            node.lines.append(f"  -> Q{f.id} {f.text}: " + (f"forcing: {rs}" if rs else "no forcing move"))
        node.asked += len(q.fixes)
        stop = [d for d in q.defects if d.kind == "stopper"]
        if stop:
            reason = "stopped: " + "; ".join(d.why for d in stop)
        else:
            reason = "tempo: " + "; ".join(f"{d.kind} {G.name(d.sq)}: {d.why}" for d in q.defects)
        if (yield T('chase', chase_g(node, q), node)):                                   # 52.15 (2): a promotion that the defects stop is chased: make them move, ask the promote root again on each reply
            return
        node._kill(q, reason, tempo=True)
        return
    ps = dict(PV.proved_passers(b, bishops=True))
    if sq in ps:
        push = chess.Move(sq, facts["path"][0], promotion=chess.QUEEN if n == 1 else None)
        san = b.san(push)
        rest = [G.name(t) for t in facts["path"][1:-1]]
        line = f"{san}, then {', '.join(rest)}{', ' if rest else ''}{G.name(promo)}=Q within {ps[sq]} plies whatever they do ({n} pushes)" if n > 1 else f"{san} within {ps[sq]} ply"
        claim = Claim("promote", san, R.STAKE_PROMOTE, "unstoppablePasser2_sound", line, move=push.uci())
        node._stamp(claim, q)
        node.state.claims.append(claim)
        q.status = "verified"
        node.lines.append(f"  -> CLAIM promote {san} (value {claim.value}), lemma {claim.lemma}, line {claim.line}")
        if node._better(claim.value, 0, claim):
            node.best = (claim.value, 0, claim)
            node.asked_at_claim = node.asked
        return
    ex = node._promotion_exchange(q, sq, facts)
    if ex is not None:
        return
    h = PV.passer_v2(b, sq)
    bad = [k for k in PV.HYPS if not h[k]]
    why = []
    for k in bad:
        if k == "enemyPieces":
            why.append("enemyPieces: " + ", ".join(f"their {b.piece_at(s).symbol().upper()} on {G.name(s)}" for s in chess.SquareSet((b.rooks | b.queens | b.knights) & b.occupied_co[not node.me])))
        elif k == "bishopsFar":
            why.append("bishopsFar: " + ", ".join(f"their B on {G.name(s)}" for s in chess.SquareSet(b.bishops & b.occupied_co[not node.me])))
        elif k == "noSliders":
            why.append("noSliders: " + ", ".join(f"our {b.piece_at(s).symbol().upper()} on {G.name(s)}" for s in chess.SquareSet((b.rooks | b.bishops | b.queens) & b.occupied_co[node.me])))
        else:
            why.append(k)
    msg = (f"Q{q.id} {q.text}: the defects say unstoppable but the lemma unstoppablePasser2_sound does not fire -- failing hypothesis "
           + ("; ".join(why) if why else "none in the Python mirror (the Lean detector declined)") + " (the next lemma to prove)")
    q.status = "gap"
    q.reason = msg
    node.state.gaps.append(msg)
    node.lines.append(f"  -> GAP: {msg}")


def verify_g(node, q: R.Question, routes: list, mates_only: bool) -> None:
    b, us = node.board, node.us
    if not routes:
        if (yield T('chase', chase_g(node, q), node)):
            return
        node._kill(q, "no route: the fixes supply no move", tempo=False)
        return
    hits, why = [], []
    for r in routes:
        bind, tried = BD.verify_tried(r.move, b, us, mates_only, trap=r.trap, fork=r.fork)
        asked = node._asked(tried)
        if bind is None:
            node.lines.append(f"  verify {r.san}: nothing binds ({asked})")
            why.append(node._escape_note(r, tried))
            continue
        head = f"  verify {r.san}: binds {bind.schema} ({len(bind.replies)} replies{': ' + ' / '.join(bind.replies[:4]) if len(bind.replies) <= 4 else ''}), value {bind.value}"
        opp, theirs, net = [], None, bind.value
        if bind.value >= BD.MATE_VALUE or not node.side_depth:
            node.lines.append(head + ("" if bind.value >= BD.MATE_VALUE else "; their view: not asked (this is their agenda)"))
        else:
            theirs, _ = node.their_view(r.move)
            if theirs is not None:
                net = bind.value - theirs.value
                node.lines.append(head + f"; their view: their {theirs.what} wins {theirs.value} (their claim, lemma {theirs.lemma}; {theirs.line}) -> net {net}")
            else:
                opp = R.opponent_defects(b, r.move, us)
                node.lines.append(head + "; their view: no claim of theirs binds" + (f" (fallback label: {'; '.join(opp)})" if opp else ""))
        if node.side_depth and bind.schema.startswith("delayed_balance") and bind.schema.endswith("_d1"):
            ref = node._leaf_read(bind, r)
            if ref is not None:
                gap = f"{r.san}: refuted at the leaf: their {ref['claim'].what} wins {ref['claim'].value} (lemma {ref['claim'].lemma}) after {ref['line']}"
                node.state.gaps.append(f"Q{q.id} {q.text}: {gap}")
                node.lines.append(f"  {r.san} DOWNGRADED to a gap: {gap}; our claimed gain {bind.value} is not decided by")
                why.append(f"{r.san}: {gap}")
                continue
        if theirs is not None and theirs.value >= q.stake and net <= 0:
            why.append(f"{r.san}: their {theirs.what} wins {theirs.value} (lemma {theirs.lemma}) -- the future loses it")
            node.lines.append(f"  {r.san} KILLED as a route: their {theirs.what} wins {theirs.value} (lemma {theirs.lemma}) -- the future gives them a claim of at least our stake {q.stake} and nets {net}")
            continue
        hits.append((bind, opp, r, net, theirs))
    if not hits:
        if (yield T('chase', chase_g(node, q), node)):
            return
        node._kill(q, f"verify: no route binds ({', '.join(r.san for r in routes[:6])}{' ...' if len(routes) > 6 else ''})"
                   + ("; " + "; ".join(w for w in why if w and "no mate in one" in w)[:300] if mates_only else "")
                   + ("; " + "; ".join(w for w in why if "the future loses it" in w)[:300]), tempo=False)
        return
    hits.sort(key=lambda h: (-h[3], len(h[1])))               # net value first; the heuristic label only breaks a tie where no claim of theirs bound
    if node._t(AG.NEED_PROPAGATION, 3) and node.chase_depth and hits[0][3] < node.min_stake and (yield T('chase', chase_g(node, q), node)):      # 52.23 (7): a bound below what this agenda must reach rescues nothing: the question chases first (zBouR: Rxe3+ bound 1, 6 needed, Qh1+ drives the king onto the skewer line)
        return
    bind, opp, r, net, theirs = hits[0]
    for other, oopp, orr, onet, otheirs in hits[1:]:
        if onet == net:
            if oopp or opp:
                if len(oopp) > len(opp):
                    node.lines.append(f"  {orr.san} ties at net {onet} but no claim of theirs binds and its future has a defect of theirs (fallback label: {'; '.join(oopp)}): {r.san} is preferred")
        elif otheirs is not None and net > onet:
            node.lines.append(f"  {orr.san} demoted: bound value {other.value} less their {otheirs.what} ({otheirs.value}, lemma {otheirs.lemma}) = net {onet}, against {r.san} net {net}")
    kind = "mate" if bind.value >= BD.MATE_VALUE else ("save" if q.goal == "save" else "win")
    claim = Claim(kind, bind.san, bind.value, bind.theorem, node._line_of(bind), move=bind.assignment["x"] + bind.assignment["y"])
    node._stamp(claim, q)
    node.state.claims.append(claim)
    q.status = "verified"
    node._record_claim(q, net, claim, bind)
    window = " [a gain WITHIN THE WINDOW: the leaf is not read to quiescence -- 0UqOv's queen was trapped two plies beyond it]" if "balance" in str(claim.lemma) else ""
    node.lines.append(f"  -> CLAIM {kind} {claim.what} (value {claim.value}{'' if net == claim.value else f', net {net} of their best claim'}), lemma {claim.lemma}, line {claim.line}{window}")
    key = (net, -len(opp))
    if q.pattern.get("inner_threat"):                        # 52.14: the parry of a threat on a static leaf keeps the gain; it is not a gain of its own (the reply's value stays dbal)
        return
    if node._better(key[0], key[1], claim):
        node.best = (net, -len(opp), claim)
        node.asked_at_claim = node.asked


def chase_g(node, q: R.Question) -> bool:
    """52.12: the chase of a question that would be killed. -> True when a route CLAIMS (the question is verified)."""
    evading = node._t(AG.EVASION_CHASE, 3) and q.goal == "save" and "checkers" in q.pattern          # 52.23 (3): the save-king question of an agenda in check: its routes are the evasions, all forced (`_verify` calls the chase too: the switch is read HERE)
    if node.ctx is None or (q.goal not in ("mate", "win", "promote") and not evading) or node.min_stake >= BD.MATE_VALUE or (node.chase_depth >= 2 and (q.goal == "promote" or (q.goal == "win" and not node._t(AG.DEEP_WIN_CHASE, 3)))):      # a mate-only read (the reply already gained, or lost too much for anything but a mate) does not chase on
        return False                                               # 52.23 (5): a win question chases at every depth; its checks must address a defect (52.17 (2)) -- the depth-two gate was a numbered limit
    if node.chase_depth and node.since_capture >= node._clock() and not evading:      # 52.23 (8): the chase clock -- a capture that is not bad for us is one-way progress (material never comes back); without one for
        node.ctx.clock_stops += 1                                   # CHASE_CLOCK of our moves the chain goes nowhere and ends here (a hyperparameter to tune, Kaveh 2026-10-20)
        q._chain_cut = True
        node.lines.append(f"  chase: not started -- the chase clock: {node.since_capture} moves of ours without a capture that is not bad for us")
        q._chased = True
        return False
    if evading and node.evaded:                                    # 52.23 (3): ONE intermezzo check is followed; a second check of theirs against our evasion is their chain, not our line (read as no claim)
        node.lines.append("  chase: not started -- a second check of theirs in a row (their chain, not our line)")
        q._chain_cut = True
        return False
    if node.best is not None and node.best[0] >= q.stake:      # 52.13 (2c): a question is chased only when its stake exceeds the best claim so far (stop (a) would end the agenda anyway)
        node.lines.append(f"  chase: not asked, the best claim ({node.best[2].what}, {node.best[0]}) is at least the stake {q.stake}")
        return False
    if node.ctx.out_of_time():                                 # 52.17 (5): past the deadline a question is asked only for what closes at the static leaf: no chase
        node.lines.append("  chase: out of time -- not started (a GAP, never a kill)")
        node.ctx.budget_hits += 1                                # the agenda this question belongs to is cut: its route is a GAP, never refuted by what was not chased
        node._note_resume(q, None, dict(gap="chase not started"))
        node.state.gaps.append(f"Q{q.id} {q.text}: out of time: chase not started")
        return False
    allow = None
    if q.goal in ("mate", "win") and node._t(AG.GOAL_JUDGE, 3):             # 52.23 (6): TEMPO ARITHMETIC ENDS THE CHAIN (the goal-progress judge, Kaveh 2026-10-20: "mating the king is like mating the bishop").
        fresh = q.goal == "win" and node.chase_depth > 0 and node.since_capture == 0      # A pattern with d defects needs at most d prep moves of ours, each fixing one (regress.py's tempo arithmetic); a fix that
        top = node.ctx.mate_levels[-1] if node.ctx.mate_levels else None                  # itself needs two moves (drive the king onto the skewer line, then skewer) says so (`steps`). A chain that has spent
        parent = None if fresh or top is None or q.stake > top[1] else top[0]             # (a BIGGER goal found on the chain is a new goal with its own distance: zBouR's queen under the f2 pawn's chase)
        dist = max([1, len(q.defects)] + [f.pattern.get("steps", 1) for f in q.fixes])      # them with the goal unmet is the walk. For a WIN goal a capture that is not bad for us is one-way progress (Kaveh
        allow = dist if parent is None else min(parent - (0 if node.free_level else 1), dist)      # 2026-10-20): the allowance starts again (zBouR: Qh1+ Rf1 Qxf1+ Kd2 Rxf2+); a mate's distance is its defects alone.
        if allow <= 0:
            node.ctx.mate_walks += 1
            q._chain_cut = True                                  # 52.23 (9): a kill by the chain's state is not a fact of the position: never filed, never served
            node.lines.append(f"  chase: not started -- tempo arithmetic: the chain has spent the prep moves the goal's defects allow ({len(q.defects)} defect(s), {parent} level(s) left under the {top[1]}-stake goal)")
            q._chased = True
            return False
    b = node.board
    routes, notes = [], []
    for m in R._moves(b, node.us):
        if evading:                                          # 52.23 (3): in check every legal move is a route (it is forced: the check must be answered), none is quiet
            pc = b.piece_at(m.to_square)
            routes.append(R.Route(m, b.san(m), f"capture {G.worth(pc)}" if pc is not None else "evade"))
            continue
        why = R.forcing_why(b, m, q.stake, node.us, mate_threats=True, notes=notes, mate2=node.ctx.tier >= 2,      # plan 59 tier 2: a mate-in-two threat is forcing too
                            delayed=AG.DELAYED_THREAT and node.ctx.tier >= 1 and not node.chase_depth, nosafe=AG.NO_SAFE_MOVE and node.ctx.tier >= 1 and not node.chase_depth)      # plan 61 A3 / A4 (mirror of agenda._chase)
        if why:
            if why.startswith("delayed threat"):
                node.ctx.delayed_routes += 1
            elif why.startswith("no safe move"):
                node.ctx.nosafe_routes += 1
            ok, note = node._addresses(q, m, why)
            if ok:
                routes.append(R.Route(m, b.san(m), why))
                if note:
                    notes.append(note)
            else:
                node.ctx.unaddressed += 1
                notes.append(note)
    if q.goal == "promote" or (q.goal == "win" and node._t(AG.DEEP_WIN_CHASE, 3)):      # 52.15 (2): the fix routes of the defects (take the stopper or the blocker, push with check) are routes too, at every depth; 52.23 (4): a win's named fixes too (the skewer, the drive onto its line)
        seen = {r.move for r in routes}
        for f in q.fixes:
            for r in R.all_routes(f):
                if r.move not in seen:
                    seen.add(r.move)
                    routes.append(R.Route(r.move, r.san, f"fix: {r.why}"))
    if q.goal == "promote" and not node.chase_depth:        # 52.15 (2): plus a move that attacks a named STOPPER of the question ("threat on the stopper"), or offers the pawn to it
        routes += node._stopper_routes(q, [r.move for r in routes], notes)
    if node.chase_depth and not evading:                    # inner levels follow the most forcing moves only: a check, a capture, a mate threat (the quiet "threat n" moves are the root's)
        follow = ("check", "threat of mate", "capture") + (("threat",) if node.ctx.tier >= 1 else ())      # plan 59 tier 1 lifts the cut for the quiet threats worth the stake (and the tier-2 mate-in-two threat)
        routes = [r for r in routes if r.why.removeprefix("fix: ").startswith(follow)]
    if node.chase_depth >= 2 and q.goal == "win":           # 52.23 (5): below the first inner level a win question follows only the NAMED FIXES of its defects that CHECK (the skewer, the fork-check:
        fix_moves = {r.move for f in q.fixes for r in R.all_routes(f)}     # the reply is forced); every other forcing move is the root's and the first level's (the named fixes that do not check still cost 0LRTF 4x)
        routes = [r for r in routes if r.move in fix_moves and b.gives_check(r.move)]
    routes.sort(key=lambda r: (0 if r.why.removeprefix("fix: ") == "check" else 1 if r.why.removeprefix("fix: ").startswith("capture") else 2))      # the most forcing first: a check, a capture, a threat (stable inside)
    node.promote_stake = q.stake if q.goal == "promote" else 0
    if not routes:
        node.lines.append("  chase: no forcing move on this board")
        q._chased = True                                     # nothing to follow: the chase ran to its end
        return False
    for n in notes[:6]:
        node.lines.append("  chase: " + n)
    node.lines.append(f"  chase: make them move -- {len(routes)} forcing route(s): " + ", ".join(f"{r.san} ({r.why})" for r in routes[:12]) + (" ..." if len(routes) > 12 else ""))
    done, cut, route_facts = [], False, []
    results: dict = {}
    pool = node.ctx.left
    node._chasing = q if q.goal == "promote" else None
    node._chase_q = q
    if allow is not None:
        node.ctx.mate_levels.append((allow, q.stake))
    try:
        return (yield from chase_routes_g(node, q, routes, cut, done, route_facts, results, pool))
    finally:
        if allow is not None:
            node.ctx.mate_levels.pop()


def chase_routes_g(node, q: R.Question, routes: list, cut: bool, done: list, route_facts: list, results: dict, pool: int) -> bool:
    """52.23 (6): the body of `_chase` from the routes on (split so the mate allowance is pushed and popped around it)."""
    # 52.17 (4): CHEAPEST REPLY FIRST. Before any route opens an inner agenda, every route's replies are read at the static leaf and what the rules close is closed (a standing
    # kill, a mate or stalemate against us, a mate in one, the exchange with nothing of theirs): a route killed there never draws a share of the fuse, and the routes still open
    # share it in order.
    cheap_res = {}
    root_final = None                                        # 52.23 (11): the root's share of the time (set at the first open route)
    for i, r in enumerate(routes):
        got = (yield T('route', chase_route_g(node, r.move, promote=q.goal == "promote", cheap=True), node))
        if not got.get("pending"):
            cheap_res[i] = got
    for i, r in enumerate(routes):
        saved = node.ctx.left
        open_left = sum(1 for j in range(i, len(routes)) if j not in cheap_res)
        share = max(saved // max(open_left, 1), 1) if saved > 0 else 0
        if node.chase_depth == 0 and saved > 0 and i not in cheap_res and node.ctx.final is None:             # the root's fuse is shared FAIRLY (52.23 (1): by TIME when seconds are given) by the routes of this chase: each gets an equal share of what is left (a chain of checks must not spend what Qd5 needs: 0DoFB)
            node.ctx.left = share
        before = node.ctx.left
        if i in cheap_res:
            res = cheap_res[i]
        elif node.ctx.out_of_time():                           # 52.17 (5): the open routes end as GAPs
            res = dict(value=0, killed=None, gap=f"out of time: {r.san} not started", parts=[], lemmas=[])
            node.ctx.budget_hits += 1
        else:
            if node.chase_depth == 0 and node.ctx.final is not None:       # ... and the time is shared FAIRLY like the fuse: each open route gets an equal slice of what is left
                now = time.monotonic()
                if root_final is None:                                     # 52.23 (11): and, like the fuse, the ROOT gets at most half of what is left while another root can beat the best claim (BEKFf: the first mate root ate the whole minute)
                    cap = any(t[4].goal in ("mate", "win", "promote") and (node.best is None or node.best[0] < t[4].stake) for t in node.heap)
                    root_final = now + (node.ctx.final - now) / 2 if cap else node.ctx.final
                node.ctx.deadline = now + max(min(root_final, node.ctx.final) - now, 0) / max(open_left, 1)
                node.ctx.hand = None                           # (a new slice: the route in hand is the one this slice sees)
            res = (yield T('route', chase_route_g(node, r.move, promote=q.goal == "promote"), node))
            if node.chase_depth == 0 and node.ctx.final is not None:
                node.ctx.deadline = node.ctx.final
                node.ctx.hand = None
        if node.chase_depth == 0 and i not in cheap_res:
            # 52.16 (2): a route is CHARGED at most its share. The frames open when the fuse runs out finish their questions (that is how 08pSj's Rf1+ and 0VSUT's Qxe5 are decided),
            # but that finishing is free: charged in full it took the share of every later route (0LRTF: 239 questions on a share of 120, then Rd7 got 1).
            node.ctx.left = saved - min(max(before - node.ctx.left, 0), share) if saved > 0 else node.ctx.left
        route_facts += res.get("facts", [])
        results[i] = res
        if res["gap"] and res["gap"].startswith("out of time"):
            node._note_resume(q, r, res)
        if res["gap"]:
            cut = True
            node.lines.append(f"    {r.san}: GAP, {res['gap']}" + (" (seen before with no more fuse: not re-run)" if res.get("reused") else ""))
            node.state.gaps.append(f"Q{q.id} {q.text}: {res['gap']}")
        elif res["killed"]:
            node.lines.append(f"    {r.san}: killed -- {res['killed']}")
        elif res["value"] >= 1:
            node.lines.append(f"    {r.san}: " + " / ".join(res["parts"][:6]) + (f" / ... {len(res['parts'])} replies" if len(res["parts"]) > 6 else "") + f" => value {res['value']}")
            done.append((res["value"], r, res))
        else:
            node.lines.append(f"    {r.san}: value {res['value']} (no gain on the worst reply)")
    if q.goal == "win" and not node.ctx.out_of_time():
        (yield from refuter_fixes_g(node, q, routes, results, done, pool))
    node._chasing = None
    node._chase_q = None
    q._chased = not cut                                      # 52.19: a kill is "chased" only when no route was cut (a GAP); a cut chase is as weak as no chase
    node._qdeps = list(dict.fromkeys(route_facts))           # a kill stands on every route's facts ...
    if node.chase_depth == 0 and pool > 1 and any(t[4].goal in ("mate", "win", "promote") and (node.best is None or node.best[0] < t[4].stake) for t in node.heap):
        # ... and a ROOT question is charged at most half of the pool it started with, while another root that can still beat the best claim is open: by stake order the
        # question that matters most gets the larger share, never all of it (0GJec: the promotion's chase took 618 of 570 and every later root was read with nothing)
        charged = pool - max(node.ctx.left, pool - pool // 2)
        node.ctx.left = max(node.ctx.left, pool - pool // 2)
        node.lines.append(f"    fuse: this root is charged {charged} of the pool {pool} (routes at most their shares, the root at most half while another root can beat the best claim)")
    node.lines.append(f"    (chase: {node.ctx.agendas} inner agendas, {node.ctx.asked} inner questions, {node.ctx.budget_hits} fuse hit(s), {node.ctx.gap_reused} gap route(s) reused, fuse left {node.ctx.left})")
    node.lines.append("    chase ended: " + ("the FUSE was hit on at least one route (a GAP, never a kill)" if cut else "every route closed (claim on every reply or a refuting reply)"))
    if question_killed(done):
        return False
    value, r, res = best_route(done)
    node._qdeps = list(res.get("facts", []))                 # ... a claim only on the facts of the route that decided it
    node._extra_reads = frozenset(res.get("extra_reads", ()))     # 52.22 (3): and on the squares of the refuter it exchanged
    net = value
    kind = "mate" if value >= BD.MATE_VALUE else ("promote" if q.goal == "promote" and not res.get("material") else "save" if q.goal == "save" else "win")
    parts = " / ".join(res["parts"][:4]) + (" / ..." if len(res["parts"]) > 4 else "")
    lemma = "chase: step_after+step_least over " + ", ".join(sorted(set(res["lemmas"]))[:3]) if res["lemmas"] else "chase: step_after+step_least"
    claim = Claim(kind, r.san, value, lemma, f"{r.san}; {parts}", move=r.move.uci())
    node._stamp(claim, q)
    claim.fixes = claim.fixes + ("chase",)
    node.state.claims.append(claim)
    q.status = "verified"
    node.lines.append(f"  -> CLAIM {kind} {r.san} (value {value}), lemma {lemma}, line {claim.line}")
    if node._better(net, 0, claim):
        node.best = (net, 0, claim)
        node.asked_at_claim = node.asked
    return True


def chase_route_g(node, m: chess.Move, promote: bool = False, cheap: bool = False, override: "dict | None" = None) -> dict:
    """52.19: the route of `_chase_route_body`, with the facts it read (inner answers, their agendas) kept on the result (`facts`) for the claim or kill that stands on it, and the listener's
    frame for the reply being read restored on the way out."""
    saved_rf, saved_frame = node._rf, node._frame
    node._rf, node._frame = [], []
    node._route_move = m
    if node.chase_depth == 0 and node.ctx.deadline is not None:
        node.ctx.slice_len = max(node.ctx.deadline - time.monotonic(), 0)      # 52.23 (11): the slice this root route was given
    ctx = node.ctx
    ctx.route_depth += 1
    ctx.route_names.append(node.board.san(m))
    try:
        res = (yield from (chase_route_body_rs_g if ROUTE_RUST else chase_route_body_g)(node, m, promote, cheap, override))
        res["facts"] = list(dict.fromkeys(res.get("facts", []) + node._rf))
        if not cheap:
            node._gap_fact(m, promote, res)
        return res
    finally:
        if ctx.hand is not None and not ctx.abort and ctx.route_depth == ctx.hand and not cheap:
            ctx.finished += 1                                    # 52.21 (3): the route in hand ran to its end (or to a GAP of its own)
        ctx.route_depth -= 1
        ctx.route_names.pop()
        node.ctx.facts.restore(node._frame)
        node._rf, node._frame = saved_rf, saved_frame


def chase_route_body_g(node, m: chess.Move, promote: bool = False, cheap: bool = False, override: "dict | None" = None) -> dict:
    """One forcing route of the chase: every reply of theirs, the whole agenda on each resulting position. -> {value, killed, gap, parts, lemmas}.
    52.17 (4) `cheap`: CHEAPEST REPLY FIRST -- only what the rules close is read (standing kills, a reply that mates or stalemates us, a mate in one, the exchange with nothing of
    theirs, a promotion that stands); a reply that needs an inner agenda leaves the route PENDING (not memoised) and the closed replies are kept for the second pass.
    52.15 (2) `promote`: the question is a promotion; a reply is answered when the inner agenda claims the promotion (or a mate), or the reply itself gave up at least the
    stake in material; the roots asked there are those of stake >= the promote stake, and the route's value is the promote stake (a mate: 100)."""
    b, ctx = node.board, node.ctx
    key = (AG._pos_key(b.fen()), m.uci() + (":promote" if promote else "") + (":refuter" if override else ""))      # 52.22 (3) `override`: reply uci -> (value, part, facts) read by the refuter fix, not by an inner agenda
    pre = node._pre.pop(key, None) if not cheap else None      # the replies the cheap pass closed by rule (value, why)
    cq = node._chase_q
    gstake = cq.stake if cq is not None and cq.goal == "win" and cq.pattern.get("target") is not None else 0        # 52.22 (1): what the exchange must reach to close a reply by rule
    if key in ctx.routes:
        hit = ctx.routes[key]
        stale = hit["gap"] is None and hit["killed"] is None and gstake and hit.get("closed_min", gstake) < gstake and hit.get("stake_checked", 0) < gstake       # a value closed by a short exchange under a lesser goal is not this goal's
        if hit["gap"] is None and not stale:
            return hit
        slice_now = max(ctx.deadline - time.monotonic(), 0) if ctx.deadline is not None else None
        if not stale and (slice_now <= hit.get("cut_slice", 0) if slice_now is not None else ctx.left <= hit["cut_left"]):      # 52.13 (2a): a GAP stays a GAP while the fuse left is no more than when it was cut; 52.23 (11): under a time budget, while its slice is no longer than the one it was cut on
            ctx.gap_reused += 1
            return dict(hit, reused=True)
    san = b.san(m)
    c = b.copy(stack=False)
    c.push(m)
    res = dict(value=0, killed=None, gap=None, parts=[], lemmas=[])
    replies = list(c.legal_moves)
    if not replies:
        if c.is_checkmate():
            res.update(value=BD.MATE_VALUE, parts=[f"{san} mates"])
        else:
            res["killed"] = "stalemate"
        ctx.routes[key] = res
        return res
    # 52.13 (5): the replies are read through the Rust `Pos` (no FEN, no board copy per reply); a python-chess board is built only for a reply that is read further
    P0 = AC.Pos(b.fen())
    P1 = P0.push(m.uci())
    shift = P1.balance(node.me) - P0.balance(node.me)
    rows = [(shift + dbal, uci, chk, cap, rec, mate1) for uci, dbal, chk, cap, rec, mate1 in P1.replies()]
    pos_of: dict = {}

    def P2_of(uci: str) -> "AC.Pos":
        if uci not in pos_of:
            pos_of[uci] = P1.push(uci)
        return pos_of[uci]

    def san_of(uci: str) -> str:
        return c.san(chess.Move.from_uci(uci))
    # 52.13 (4): the SHORTLIST. A standing kill (an earlier route's reply, the same reply move, nothing its reads touch has changed in the position it leaves) ends the
    # route at once and is counted; otherwise the replies are read recaptures on the route's square, then checks, then captures, then the rest, the worst balance first inside a class.
    kill_uci = None                                          # 52.22 (3): the reply that killed the route: its piece is the REFUTER (Ideas.Refuter.Refutes)
    for dbal, uci, chk, cap, rec, mate1 in rows:
        sk = None if promote or (override and uci in override) else node._standing_kill(uci, P2_of(uci), dbal)
        if sk is not None:
            kill_uci = uci
            res["killed"] = f"after {san_of(uci)}: {sk[0]} (standing kill: nothing the inner agenda read after {sk[1]} {san_of(uci)} has changed)"
            ctx.standing += 1
            node._log(key, uci, "standing", dbal, None, sk[2], res["killed"])
            break
    if res["killed"] is not None:
        res["refuter"] = kill_uci
        ctx.routes[key] = res
        return res
    rows.sort(key=lambda t: (0 if t[4] else 1 if t[2] else 2 if t[3] else 3, t[0], t[1]))
    gains = []
    closed_now = {}
    closed_min = BD.MATE_VALUE                               # the least value an exchange closed a reply at (the memo entry keeps it: another goal's stake may ask for more)
    for dbal, uci, chk, cap, rec, mate1 in rows:
        P2 = P2_of(uci)
        san_r = san_of(uci)
        kill_uci = uci
        if override and uci in override:                     # 52.22 (3): the reply the refuter fix answered
            val, part, ofacts = override[uci]
            gains.append(val)
            res["parts"].append(part)
            node._rf.extend(f for f in ofacts if f not in node._rf)
            continue
        if pre is not None and uci in pre:                   # closed by rule in the cheap pass: counted and logged there
            val, why = pre[uci]
            if val < BD.MATE_VALUE:
                closed_min = min(closed_min, val)
            gains.append(val)
            res["parts"].append(f"after {san_r}: closed {node._closed_tag(why)}: {why} -> {val}")
            continue
        if P2.legal_count() == 0:
            res["killed"] = f"after {san_r} " + ("they mate us" if P2.is_check() else "stalemate")
            node._log(key, uci, "killed", dbal, None, None, res["killed"])
            break
        closed = ((BD.MATE_VALUE, "mate in one") if (P2.mate_in_one() if mate1 is None else mate1) else None) if promote else node._closed_by_rule(P2, dbal, mate1)
        if closed is not None and not promote:
            closed = node._parry_label(m, uci, closed)         # plan 61 A2 (mirror of agenda._chase_route_body)
        if closed is not None:
            val, why = closed
            if val < BD.MATE_VALUE:
                closed_min = min(closed_min, val)
            gains.append(val)
            res["parts"].append(f"after {san_r}: closed {node._closed_tag(why)}: {why} -> {val}")
            ctx.closed += 1
            closed_now[uci] = closed
            node._log(key, uci, "closed", dbal, val, None, why)
            continue
        if promote and dbal < node.promote_stake:
            why = node._promote_stands(m, P2)
            if why is not None:
                res["killed"] = f"after {san_r}: {why}"
                ctx.promote_stands += 1
                node._log(key, uci, "killed", dbal, None, None, res["killed"])
                break
        if cheap:                                            # the reply needs an inner agenda: the route stays open for the second pass
            res["pending"] = True
            break                                            # (the replies after it stay for the second pass: reading them all by rule cost 20x their agendas)
        target = max(1, node.min_stake) if node._t(AG.NEED_PROPAGATION, 3) else 1      # 52.23 (7): THE NEED PROPAGATES -- the route must reach what this agenda serves (its min_stake), not 1 (zBouR: Qh1+ Rf1 read to 5 when 6 was needed)
        need = min(R.STAKE_PROMOTE, 1 - dbal) if promote and dbal <= 0 else R.STAKE_PROMOTE if promote else target - dbal if dbal < target else BD.MATE_VALUE          # a reply that already gains the need is read for a mate only; else its static gain must survive their threats (below)
        node._enter_reply(frozenset(P0.diff(P2)))
        inn = (yield T('reply', inner_g(node, P2.fen(), need, frozenset(P0.diff(P2)), since=0 if node.board.is_capture(node._route_move) and dbal >= 0 else node.since_capture + 1,
                          exchange=node.board.is_capture(node._route_move) and chess.Move.from_uci(uci).to_square == node._route_move.to_square), node))      # 52.23 (6): a recapture on the route's square: the exchange is ONE event of the chain
        if inn is None:
            res["gap"] = node._cut(san, san_r)
            res["reply"] = uci
            ctx.budget_hits += 1
            node._log(key, uci, "gap", dbal, None, None, res["gap"])
            break
        if promote:                                          # 52.15 (2): the inner agenda asked the promote root (and the mates) again on this reply
            pc = inn.get("promote")
            if inn["value"] >= BD.MATE_VALUE:
                g, part = BD.MATE_VALUE, f"{inn['claim'].what} ({inn['claim'].line}; value {inn['claim'].value})"
            elif pc is not None:
                g, part = R.STAKE_PROMOTE, f"{pc.what} ({pc.line}; value {pc.value}, lemma {pc.lemma})"
            elif inn["claim"] is not None and inn["value"] >= node.promote_stake:     # a claim of at least the stake (the agenda stops at it: the promote root itself is then not asked)
                g, part = R.STAKE_PROMOTE, f"{inn['claim'].what} ({inn['claim'].line}; value {inn['claim'].value}, at least the stake {node.promote_stake})"
            elif dbal >= node.promote_stake:
                g, part = dbal, f"material {dbal:+d} (the reply gave up at least the stake {node.promote_stake})"
            elif dbal <= 0 and inn["claim"] is not None and inn["value"] >= 1 - dbal:       # no promotion, but the material the reply cost is won back and more (the ordinary chase's reading)
                g, part = dbal + inn["value"], f"{inn['claim'].what} ({inn['claim'].line}; value {inn['claim'].value}) wins the material back: {dbal:+d} + {inn['value']}"
                res["material"] = True
            elif inn.get("cut"):
                res["gap"] = node._cut(san, san_r, True)
                res["reply"] = uci
                ctx.budget_hits += 1
                node._log(key, uci, "gap", dbal, inn["value"], None, res["gap"])
                break
            else:
                res["killed"] = f"after {san_r}: the promotion is not claimed (no promote claim of ours; the reply took {-dbal})" if dbal <= 0 else f"after {san_r}: the promotion is not claimed ({dbal:+d}, below the stake)"
                node._log(key, uci, "killed", dbal, inn["value"], inn.get("reads"), res["killed"])
                break
            gains.append(g)
            res["parts"].append(f"after {san_r}: {part} -> {g}")
            node._log(key, uci, "claim", dbal, inn["value"], inn.get("reads"), res["parts"][-1])
            if pc is not None:
                res["lemmas"].append(pc.lemma)
            continue
        static = dbal >= 1 and inn["value"] < BD.MATE_VALUE
        t_their, claim_their, unparried, parry_note = 0, None, 0, None
        if static and not P2.is_check():                    # 52.13 (3), 52.14: a gain that stands only as a static balance (us to move) is a result only once THEIR agenda has read it, with the move passed to them, at every chase depth
            t_their, claim_their = node._their_pass(P2.fen())
            if claim_their is not None:
                ctx.static_read += 1
            if claim_their is not None and dbal - t_their < 1:
                # 52.14: their claim is a THREAT and a SAVE root of the inner agenda on this position (we are to move: a parry, a counter, taking the attacker); the reply is
                # killed only by the threats no parry of ours is verified for
                inn2 = (yield T('reply', inner_g(node, P2.fen(), need, frozenset(P0.diff(P2)), their=ctx.their_ags.get(AG._pos_key(G.with_turn(chess.Board(P2.fen()), not node.me).fen())), keep=dbal), node))
                if inn2 is None:
                    res["gap"] = node._cut(san, san_r)
                    res["reply"] = uci
                    ctx.budget_hits += 1
                    node._log(key, uci, "gap", dbal, None, None, res["gap"])
                    break
                inn = dict(inn, cut=inn.get("cut") or inn2.get("cut"))
                held = [(cl, pc) for cl, pc in inn2["threats"] if pc is not None]
                lost = [cl for cl, pc in inn2["threats"] if pc is None]
                unparried = sum(cl.value for cl in lost)
                parry_note = "; ".join(f"their {cl.what} wins {cl.value} if we pass; parry: {pc.what} (lemma {pc.lemma}) keeps {pc.value:+d}" for cl, pc in held)
                if lost:
                    ctx.static_unparried += 1
                else:
                    ctx.static_parried += 1
                node._static_line(san, san_r, dbal, claim_their, parry_note, lost)
                if dbal - unparried < 1:
                    worst = max(lost, key=lambda cl: cl.value) if lost else claim_their
                    res["killed"] = f"after {san_r}: {dbal:+d} stands only as a balance, their {worst.what} wins {worst.value} (lemma {worst.lemma}), no parry"
                    node._log(key, uci, "killed", dbal, inn["value"], None, res["killed"])
                    break
                claim_their = None
            else:
                unparried = t_their
        if dbal + inn["value"] < 1 and inn["value"] < BD.MATE_VALUE and inn.get("cut"):
            res["gap"] = node._cut(san, san_r, True)
            res["reply"] = uci
            ctx.budget_hits += 1
            node._log(key, uci, "gap", dbal, inn["value"], None, res["gap"])
            break
        if dbal + inn["value"] < 1 and inn["value"] < BD.MATE_VALUE:
            res["killed"] = f"after {san_r}: " + (f"our best claim {inn['claim'].what} wins {inn['value']}, the reply took {-dbal}" if inn["claim"] else "no claim of ours and no material gain") + f" ({dbal:+d})"
            if inn["claim"] is None and not inn.get("cut"):          # a standing kill: the later routes' same reply is skipped while the reads stay untouched
                ctx.reply_kills.setdefault(uci, []).append((P2, dbal, inn.get("reads"), san))
            node._log(key, uci, "killed", dbal, inn["value"], inn.get("reads"), res["killed"])
            break
        cl = inn["claim"] if inn["value"] >= need else None
        gains.append(BD.MATE_VALUE if inn["value"] >= BD.MATE_VALUE else max(dbal - unparried, dbal + inn["value"] if cl and node._t(AG.NEED_PROPAGATION, 3) else dbal - unparried) if static else dbal + (inn["value"] if cl else 0))      # 52.23 (7): a reply that already gains carries the inner claim too (it answers every reply of theirs): zBouR's Rxf1+ Kd2 Rf2+, +5 then the queen
        res["parts"].append(f"after {san_r}: " + (f"{cl.what} ({cl.line}; value {cl.value})" if cl else f"material {dbal:+d}" + (f", their {claim_their.what} wins {t_their} if we pass" if claim_their else "") + (f", {parry_note}" if parry_note else "")) + (f" -> {gains[-1]}" if cl or claim_their or parry_note else ""))
        node._log(key, uci, "claim" if cl else "material", dbal, inn["value"], inn.get("reads"), res["parts"][-1])
        if cl:
            res["lemmas"].append(cl.lemma)
    if cheap and res.get("pending") and res["killed"] is None:
        node._pre[key] = closed_now                          # not memoised: the second pass finishes it
        return res
    res["closed_min"], res["stake_checked"] = closed_min, gstake
    if res["killed"] is not None and kill_uci is not None:
        res["refuter"] = kill_uci
    if res["killed"] is None and res["gap"] is None:
        res["value"] = route_value(gains)
    if res["gap"] is not None:
        res["cut_left"] = ctx.left
        res["cut_slice"] = ctx.slice_len                     # 52.23 (11): the slice this route had when it was cut (a later, longer slice re-runs it)
        if res["gap"].startswith("out of time"):                # 52.17 (5): the deadline mark on the GAP memo entry beside cut_left: the background tier resumes exactly here
            res["timed"], res["mark"] = True, round(time.monotonic() - ctx.t0, 3) if ctx.t0 is not None else None
    ctx.routes[key] = res
    return res


def chase_route_body_rs_g(node, m: chess.Move, promote: bool = False, cheap: bool = False, override: "dict | None" = None) -> dict:
    """One forcing route of the chase: every reply of theirs, the whole agenda on each resulting position. -> {value, killed, gap, parts, lemmas}.
    52.17 (4) `cheap`: CHEAPEST REPLY FIRST -- only what the rules close is read (standing kills, a reply that mates or stalemates us, a mate in one, the exchange with nothing of
    theirs, a promotion that stands); a reply that needs an inner agenda leaves the route PENDING (not memoised) and the closed replies are kept for the second pass.
    52.15 (2) `promote`: the question is a promotion; a reply is answered when the inner agenda claims the promotion (or a mate), or the reply itself gave up at least the
    stake in material; the roots asked there are those of stake >= the promote stake, and the route's value is the promote stake (a mate: 100).
    PLAN 56 (2b): the replies are read by the Rust handler `analyst_core.read_route` (one call per route: the standing kill, the reading order, the dead / mate-in-one / exchange / open kind of every reply, the need and the chain flags); the effects on shared state stay here, in the order of the reads."""
    b, ctx = node.board, node.ctx
    key = (AG._pos_key(b.fen()), m.uci() + (":promote" if promote else "") + (":refuter" if override else ""))      # 52.22 (3) `override`: reply uci -> (value, part, facts) read by the refuter fix, not by an inner agenda
    pre = node._pre.pop(key, None) if not cheap else None      # the replies the cheap pass closed by rule (value, why)
    cq = node._chase_q
    gstake = cq.stake if cq is not None and cq.goal == "win" and cq.pattern.get("target") is not None else 0        # 52.22 (1): what the exchange must reach to close a reply by rule
    if key in ctx.routes:
        hit = ctx.routes[key]
        stale = hit["gap"] is None and hit["killed"] is None and gstake and hit.get("closed_min", gstake) < gstake and hit.get("stake_checked", 0) < gstake       # a value closed by a short exchange under a lesser goal is not this goal's
        if hit["gap"] is None and not stale:
            return hit
        slice_now = max(ctx.deadline - time.monotonic(), 0) if ctx.deadline is not None else None
        if not stale and (slice_now <= hit.get("cut_slice", 0) if slice_now is not None else ctx.left <= hit["cut_left"]):      # 52.13 (2a): a GAP stays a GAP while the fuse left is no more than when it was cut; 52.23 (11): under a time budget, while its slice is no longer than the one it was cut on
            ctx.gap_reused += 1
            return dict(hit, reused=True)
    san = b.san(m)
    c = b.copy(stack=False)
    c.push(m)
    res = dict(value=0, killed=None, gap=None, parts=[], lemmas=[])
    rr = AC.read_route(b.fen(), m.uci(), bool(node.me), node.min_stake, node._t(AG.NEED_AWARE_CLOSE, 3), bool(promote), max(1, node.min_stake) if node._t(AG.NEED_PROPAGATION, 3) else 1,
                       cq.stake if cq is not None and cq.goal == "win" and cq.pattern.get("target") is not None else None, BD.MATE_VALUE, R.STAKE_PROMOTE, list(override) if override else [], list(pre) if pre else [], ctx.reply_kills if not promote else None)
    P0, P1 = rr["p0"], rr["p1"]
    if P1.legal_count() == 0:
        if c.is_checkmate():
            res.update(value=BD.MATE_VALUE, parts=[f"{san} mates"])
        else:
            res["killed"] = "stalemate"
        ctx.routes[key] = res
        return res
    pos_of: dict = {}

    def P2_of(uci: str) -> "AC.Pos":
        if uci not in pos_of:
            pos_of[uci] = P1.push(uci)
        return pos_of[uci]

    def san_of(uci: str) -> str:
        return c.san(chess.Move.from_uci(uci))
    # the standing kill (52.13 (4)) was read by the handler over every reply before anything else; a hit ends the route at once and is counted
    kill_uci = None                                          # 52.22 (3): the reply that killed the route: its piece is the REFUTER (Ideas.Refuter.Refutes)
    if rr["standing"] is not None:
        kill_uci, sk_dbal, sk_k = rr["standing"]
        _, _, sk_reads, sk_san = ctx.reply_kills[kill_uci][sk_k]
        res["killed"] = f"after {san_of(kill_uci)}: no claim of ours and no material gain (standing kill: nothing the inner agenda read after {sk_san} {san_of(kill_uci)} has changed)"
        ctx.standing += 1
        node._log(key, kill_uci, "standing", sk_dbal, None, sorted(sk_reads), res["killed"])
        res["refuter"] = kill_uci
        ctx.routes[key] = res
        return res
    gains = []
    closed_now = {}
    closed_min = BD.MATE_VALUE                               # the least value an exchange closed a reply at (the memo entry keeps it: another goal's stake may ask for more)
    for uci, dbal, kind, best, chk, fen2, need, since0, exch, unmet in rr["rows"]:
        san_r = san_of(uci)
        kill_uci = uci
        if override and uci in override:                     # 52.22 (3): the reply the refuter fix answered
            val, part, ofacts = override[uci]
            gains.append(val)
            res["parts"].append(part)
            node._rf.extend(f for f in ofacts if f not in node._rf)
            continue
        if pre is not None and uci in pre:                   # closed by rule in the cheap pass: counted and logged there
            val, why = pre[uci]
            if val < BD.MATE_VALUE:
                closed_min = min(closed_min, val)
            gains.append(val)
            res["parts"].append(f"after {san_r}: closed {node._closed_tag(why)}: {why} -> {val}")
            continue
        if kind == "dead":
            res["killed"] = f"after {san_r} " + ("they mate us" if chk else "stalemate")
            node._log(key, uci, "killed", dbal, None, None, res["killed"])
            break
        closed = (BD.MATE_VALUE, "mate in one") if kind == "mate1" else None
        if kind == "exch":                                   # the exchange meets the route's need: the goal's unmet fix and THEIR pass (whole agendas) decide, in the order of the reads
            blocked = False
            if unmet:
                fix = node._unmet_fix(P2_of(uci), cq)
                if fix:
                    ctx.unmet += 1
                    line = f"not closed by rule: the goal ({cq.text}, stake {cq.stake}) is not met by the exchange ({dbal + best:+d}); a forcing fix remains ({fix})"
                    if line not in ctx.open_lines:
                        ctx.open_lines.append(line)
                    blocked = True
            if not blocked:
                t_, _cl = node._their_pass(fen2)
                if not t_ and AG.CLOSE_AFTER_CAPTURE:            # (mirror of agenda._closed_by_rule, 2026-10-20: their claims AFTER our capture too)
                    t2, cl2 = node._their_after_capture(P2_of(uci))
                    if t2:
                        ctx.overload_opens += 1
                        line = f"not closed by rule: after {cl2[0]} their {cl2[1].what} wins {t2} (the capturing piece left a duty)"
                        if line not in ctx.open_lines:
                            ctx.open_lines.append(line)
                        t_ = t2
                if not t_:
                    closed = (dbal + best, f"exchange +{best} ({R.exchange_read(chess.Board(fen2))}), nothing of theirs")
        if closed is not None and not promote:
            closed = node._parry_label(m, uci, closed)         # plan 61 A2 (mirror of agenda._chase_route_body)
        if closed is not None:
            val, why = closed
            if val < BD.MATE_VALUE:
                closed_min = min(closed_min, val)
            gains.append(val)
            res["parts"].append(f"after {san_r}: closed {node._closed_tag(why)}: {why} -> {val}")
            ctx.closed += 1
            closed_now[uci] = closed
            node._log(key, uci, "closed", dbal, val, None, why)
            continue
        if promote and dbal < node.promote_stake:
            why = node._promote_stands(m, P2_of(uci))
            if why is not None:
                res["killed"] = f"after {san_r}: {why}"
                ctx.promote_stands += 1
                node._log(key, uci, "killed", dbal, None, None, res["killed"])
                break
        if cheap:                                            # the reply needs an inner agenda: the route stays open for the second pass
            res["pending"] = True
            break                                            # (the replies after it stay for the second pass: reading them all by rule cost 20x their agendas)
        P2 = P2_of(uci)
        dfs = frozenset(P0.diff(P2))
        node._enter_reply(dfs)
        inn = (yield T('reply', inner_g(node, fen2, need, dfs, since=0 if since0 else node.since_capture + 1, exchange=exch), node))      # 52.23 (6): a recapture on the route's square: the exchange is ONE event of the chain
        if inn is None:
            res["gap"] = node._cut(san, san_r)
            res["reply"] = uci
            ctx.budget_hits += 1
            node._log(key, uci, "gap", dbal, None, None, res["gap"])
            break
        if promote:                                          # 52.15 (2): the inner agenda asked the promote root (and the mates) again on this reply
            pc = inn.get("promote")
            if inn["value"] >= BD.MATE_VALUE:
                g, part = BD.MATE_VALUE, f"{inn['claim'].what} ({inn['claim'].line}; value {inn['claim'].value})"
            elif pc is not None:
                g, part = R.STAKE_PROMOTE, f"{pc.what} ({pc.line}; value {pc.value}, lemma {pc.lemma})"
            elif inn["claim"] is not None and inn["value"] >= node.promote_stake:     # a claim of at least the stake (the agenda stops at it: the promote root itself is then not asked)
                g, part = R.STAKE_PROMOTE, f"{inn['claim'].what} ({inn['claim'].line}; value {inn['claim'].value}, at least the stake {node.promote_stake})"
            elif dbal >= node.promote_stake:
                g, part = dbal, f"material {dbal:+d} (the reply gave up at least the stake {node.promote_stake})"
            elif dbal <= 0 and inn["claim"] is not None and inn["value"] >= 1 - dbal:       # no promotion, but the material the reply cost is won back and more (the ordinary chase's reading)
                g, part = dbal + inn["value"], f"{inn['claim'].what} ({inn['claim'].line}; value {inn['claim'].value}) wins the material back: {dbal:+d} + {inn['value']}"
                res["material"] = True
            elif inn.get("cut"):
                res["gap"] = node._cut(san, san_r, True)
                res["reply"] = uci
                ctx.budget_hits += 1
                node._log(key, uci, "gap", dbal, inn["value"], None, res["gap"])
                break
            else:
                res["killed"] = f"after {san_r}: the promotion is not claimed (no promote claim of ours; the reply took {-dbal})" if dbal <= 0 else f"after {san_r}: the promotion is not claimed ({dbal:+d}, below the stake)"
                node._log(key, uci, "killed", dbal, inn["value"], inn.get("reads"), res["killed"])
                break
            gains.append(g)
            res["parts"].append(f"after {san_r}: {part} -> {g}")
            node._log(key, uci, "claim", dbal, inn["value"], inn.get("reads"), res["parts"][-1])
            if pc is not None:
                res["lemmas"].append(pc.lemma)
            continue
        static = dbal >= 1 and inn["value"] < BD.MATE_VALUE
        t_their, claim_their, unparried, parry_note = 0, None, 0, None
        if static and not chk:                    # 52.13 (3), 52.14: a gain that stands only as a static balance (us to move) is a result only once THEIR agenda has read it, with the move passed to them, at every chase depth
            t_their, claim_their = node._their_pass(fen2)
            if claim_their is not None:
                ctx.static_read += 1
            if claim_their is not None and dbal - t_their < 1:
                # 52.14: their claim is a THREAT and a SAVE root of the inner agenda on this position (we are to move: a parry, a counter, taking the attacker); the reply is
                # killed only by the threats no parry of ours is verified for
                inn2 = (yield T('reply', inner_g(node, fen2, need, dfs, their=ctx.their_ags.get(AG._pos_key(G.with_turn(chess.Board(fen2), not node.me).fen())), keep=dbal), node))
                if inn2 is None:
                    res["gap"] = node._cut(san, san_r)
                    res["reply"] = uci
                    ctx.budget_hits += 1
                    node._log(key, uci, "gap", dbal, None, None, res["gap"])
                    break
                inn = dict(inn, cut=inn.get("cut") or inn2.get("cut"))
                held = [(cl, pc) for cl, pc in inn2["threats"] if pc is not None]
                lost = [cl for cl, pc in inn2["threats"] if pc is None]
                unparried = sum(cl.value for cl in lost)
                parry_note = "; ".join(f"their {cl.what} wins {cl.value} if we pass; parry: {pc.what} (lemma {pc.lemma}) keeps {pc.value:+d}" for cl, pc in held)
                if lost:
                    ctx.static_unparried += 1
                else:
                    ctx.static_parried += 1
                node._static_line(san, san_r, dbal, claim_their, parry_note, lost)
                if dbal - unparried < 1:
                    worst = max(lost, key=lambda cl: cl.value) if lost else claim_their
                    res["killed"] = f"after {san_r}: {dbal:+d} stands only as a balance, their {worst.what} wins {worst.value} (lemma {worst.lemma}), no parry"
                    node._log(key, uci, "killed", dbal, inn["value"], None, res["killed"])
                    break
                claim_their = None
            else:
                unparried = t_their
        if dbal + inn["value"] < 1 and inn["value"] < BD.MATE_VALUE and inn.get("cut"):
            res["gap"] = node._cut(san, san_r, True)
            res["reply"] = uci
            ctx.budget_hits += 1
            node._log(key, uci, "gap", dbal, inn["value"], None, res["gap"])
            break
        if dbal + inn["value"] < 1 and inn["value"] < BD.MATE_VALUE:
            res["killed"] = f"after {san_r}: " + (f"our best claim {inn['claim'].what} wins {inn['value']}, the reply took {-dbal}" if inn["claim"] else "no claim of ours and no material gain") + f" ({dbal:+d})"
            if inn["claim"] is None and not inn.get("cut"):          # a standing kill: the later routes' same reply is skipped while the reads stay untouched
                ctx.reply_kills.setdefault(uci, []).append((P2, dbal, inn.get("reads"), san))
            node._log(key, uci, "killed", dbal, inn["value"], inn.get("reads"), res["killed"])
            break
        cl = inn["claim"] if inn["value"] >= need else None
        gains.append(BD.MATE_VALUE if inn["value"] >= BD.MATE_VALUE else max(dbal - unparried, dbal + inn["value"] if cl and node._t(AG.NEED_PROPAGATION, 3) else dbal - unparried) if static else dbal + (inn["value"] if cl else 0))      # 52.23 (7): a reply that already gains carries the inner claim too (it answers every reply of theirs): zBouR's Rxf1+ Kd2 Rf2+, +5 then the queen
        res["parts"].append(f"after {san_r}: " + (f"{cl.what} ({cl.line}; value {cl.value})" if cl else f"material {dbal:+d}" + (f", their {claim_their.what} wins {t_their} if we pass" if claim_their else "") + (f", {parry_note}" if parry_note else "")) + (f" -> {gains[-1]}" if cl or claim_their or parry_note else ""))
        node._log(key, uci, "claim" if cl else "material", dbal, inn["value"], inn.get("reads"), res["parts"][-1])
        if cl:
            res["lemmas"].append(cl.lemma)
    if cheap and res.get("pending") and res["killed"] is None:
        node._pre[key] = closed_now                          # not memoised: the second pass finishes it
        return res
    res["closed_min"], res["stake_checked"] = closed_min, gstake
    if res["killed"] is not None and kill_uci is not None:
        res["refuter"] = kill_uci
    if res["killed"] is None and res["gap"] is None:
        res["value"] = route_value(gains)
    if res["gap"] is not None:
        res["cut_left"] = ctx.left
        res["cut_slice"] = ctx.slice_len                     # 52.23 (11): the slice this route had when it was cut (a later, longer slice re-runs it)
        if res["gap"].startswith("out of time"):                # 52.17 (5): the deadline mark on the GAP memo entry beside cut_left: the background tier resumes exactly here
            res["timed"], res["mark"] = True, round(time.monotonic() - ctx.t0, 3) if ctx.t0 is not None else None
    ctx.routes[key] = res
    return res



def inner_g(node, fen: str, need: int, diff: "frozenset | None" = None, their: "Agenda | None" = None, keep: int = 0, since: int | None = None, exchange: bool = False) -> "dict | None":
    """The WHOLE agenda asked again on `fen` (our move), only the roots of stake >= `need` (a smaller claim cannot rescue the route): -> {value, claim, asked, need};
    None when the shared budget is spent. Memoised by FEN (an answer asked with a smaller `need` serves a larger one); a FEN being asked (a cycle) answers no claim.
    52.14: with `their` (their agenda there, the move passed to them) their claims are also asked as SAVE roots; the answer lists them (`threats`: (their claim, our parry claim or None)).
    The memo is keyed by (position, with threats): an answer asked WITH threats serves a later ask without them, not the reverse."""
    ctx = node.ctx
    pk = AG._pos_key(fen)
    wt = their is not None
    evading = node._chase_q is not None and node._chase_q.goal == "save" and "checkers" in node._chase_q.pattern
    chain = (ctx.mate_levels[-1] if ctx.mate_levels else None, node.since_capture + 1 if since is None else since, evading, exchange)     # 52.23 (9): the chain's state (the allowance, the clock, an intermezzo, an exchange)
    hit = ctx.memo.get((pk, wt, keep if wt else 0, chain))          # ... an answer cut by the chain serves the same chain state only; one not cut serves every chain
    if hit is None:
        hit = ctx.memo.get((pk, True, keep)) if wt else (ctx.memo.get((pk, False, 0)) or next((h for k_, h in ctx.memo.items() if len(k_) == 3 and k_[0] == pk and k_[1]), None))     # (position, with threats, the gain kept): an answer WITH threats serves an ask without, not the reverse
    if hit is not None and hit["need"] <= need:
        node._touch(hit.get("fact"))
        return hit
    if pk in ctx.active:
        return dict(value=0, claim=None, asked=0, need=need, threats=[(cl, None) for cl in their.state.claims] if wt else [])     # a cycle parries nothing
    if ctx.left <= 0 or node.chase_depth >= AG.DEPTH_FUSE or (ctx.out_of_time() and not ctx.finishing(ctx.route_depth)):      # 52.21 (3): only the route in hand opens an agenda past the deadline
        if ctx.left > 0:
            ctx.depth_cuts += 1
        return None
    ctx.active.add(pk)
    hits0 = ctx.budget_hits
    saved = R._ids[0]
    evading = node._chase_q is not None and node._chase_q.goal == "save" and "checkers" in node._chase_q.pattern
    ag = AG.Agenda(State(fen=fen, us=node.us, pieces=[]), side_depth=0, chase_depth=node.chase_depth + 1, ctx=ctx, min_stake=need, their=their, keep=keep, evaded=evading,
                since=node.since_capture + 1 if since is None else since, free_level=exchange)      # (52.19: what it inherits comes from the fact store, not from this agenda alone)
    yield T('node', agenda_g(ag), ag)                   # the inner agenda is a node task: its pool of questions is asked by the worker loop
    R._ids[0] = saved
    ctx.active.discard(pk)
    ctx.agendas += 1
    ctx.asked += ag.asked
    best = ag.best[2] if ag.best and ag.best[0] > 0 else None
    chain_cut = best is None and any(getattr(x, "_chain_cut", False) for x in ag.all)      # 52.23 (9): "no claim" after a chain cut (the allowance, the clock, a second check) is this chain's answer, not the position's
    reads = None if any(q.reads is None for q in ag.all) else frozenset().union(*(q.reads for q in ag.all))     # 52.13 (4): everything this agenda read (None: some question did not record it)
    threats = [(cl, q.claimed[1] if q.status == "verified" and q.claimed is not None else None) for q, cl in ag.threat_qs]
    out = dict(value=ag.best[0] if best else 0, claim=best, asked=ag.asked, need=need, cut=ctx.budget_hits > hits0, reads=reads, threats=threats,
               promote=next((c for c in ag.state.claims if c.kind == "promote"), None))      # 52.15 (2): the promote claim of this position, whether or not it is the best claim
    cfid = ag.claim_fid.get(id(best)) if best is not None else None
    if chain_cut:                                           # 52.23 (9): this chain's "no claim" kills the route here; memoised for this chain state only, never filed (another chain may reach further)
        if not out["cut"]:
            ctx.memo[(pk, wt, keep if wt else 0, chain)] = out
        return out
    f = ctx.facts.add("gap" if out["cut"] else ("claim" if best is not None else "kill"), ("inner", pk, wt, keep), node.me, f"the agenda on {pk.split()[0]}" + (f": {best.what}" if best is not None else ": no claim"),
                      value=out["value"], reads=reads, just=[cfid] if cfid is not None else ag.premises, stake=node._cur.stake if node._cur is not None else need, depth=node.chase_depth + 1,
                      under=f"Q{node._cur.id} at depth {node.chase_depth}" if node._cur is not None else f"depth {node.chase_depth}", pk=pk, portable=False)
    out["fact"] = f.id
    node._touch(f.id)
    if not out["cut"]:                                      # an answer the budget cut short is not memoised and is never read as "no claim"
        ctx.memo[(pk, wt, keep if wt else 0)] = out
    return out


def refuter_fixes_g(node, q: R.Question, routes: list, results: dict, done: list, pool: int) -> None:
    """52.22 (3): THE REFUTER FIX (Lean `Ideas.Refuter.Refutes`). A route killed by a reply names the piece that made it, the REFUTER ("Bb3 fails to Qxb3"). The fix is the one the agenda applies to a
    defender or a blocker: take the refuter with tempo -- a route of this chase that captures it where it stands and is itself killed only by the RECAPTURE, by a piece that cannot make the
    refuting move -- and ask the killed route again on the board after the exchange (a new agenda, one level in; the goal there is the piece now on the target square). The capturing route is
    then read again with that one reply answered by the re-asked route (every other reply is read as before); a route the exchange saves is a claim of its own, filed with the refuter's squares in
    its read set."""
    b, ctx = node.board, node.ctx
    P0 = AC.Pos(b.fen())
    share = max(pool // max(len(routes), 1), 1)                # the fix is the continuation of a route: charged at most an equal share of the fuse the chase started with
    for i, r_k in enumerate(routes):
        res_k = results.get(i)
        if res_k is None or res_k["killed"] is None or not res_k.get("refuter") or res_k.get("gap"):
            continue
        ref = chess.Move.from_uci(res_k["refuter"])
        for j, r2 in enumerate(routes):
            res2 = results.get(j)
            if j == i or res2 is None or res2["killed"] is None or res2.get("gap") or res2.get("fixed_by") or not res2.get("refuter") or not b.is_capture(r2.move) \
                    or r2.move.to_square != ref.from_square or r_k.move.to_square == ref.from_square:
                continue
            saved = ctx.left
            if saved < pool:
                ctx.left = pool                              # (read on the whole pool the chase started with, charged at most its share: the finishing of a route is free, as in 52.16 (2))
            before = ctx.left
            try:
                got = (yield from refuter_fix_g(node, q, r_k, ref, r2, res2["refuter"], P0))
                new = (yield T('route', chase_route_g(node, r2.move, override={res2["refuter"]: (got["value"], got["part"], got["facts"])}), node)) if got is not None else None
            finally:
                ctx.left = saved - min(max(before - ctx.left, 0), share) if saved > 0 else saved
            if new is None:
                continue
            if new["killed"] is not None or new["gap"] is not None or new["value"] < 1:
                continue
            new.update(extra_reads=got["reads"], fixed_by=r_k.san)
            results[j] = new
            node.lines.append(f"    {r2.san}: refuter fix -- {got['text']} => value {new['value']}")
            ctx.refuter_fixes += 1
            done.append((new["value"], r2, new))


def refuter_fix_g(node, q: R.Question, r_k: R.Route, ref: chess.Move, r2: R.Route, rep_uci: str, P0: "AC.Pos") -> "dict | None":
    """52.22 (3): the exchange `r2`, `rep_uci` (the recapture that killed r2), then `r_k` asked again: -> {value (the exchange's balance + the route's), part, facts, reads, text} or None
    (the reply is not a recapture on the square, the recapturing piece can make the refuting move, the route is no longer legal, or it is not worth a gain there)."""
    b, me, ctx = node.board, node.me, node.ctx
    c = b.copy(stack=False)
    pc_ref = c.piece_at(ref.from_square)
    c.push(r2.move)
    x = chess.Move.from_uci(rep_uci)
    if x not in c.legal_moves or not c.is_capture(x) or x.to_square != r2.move.to_square:
        return None
    san_x = c.san(x)
    san_ref_c = b.copy(stack=False)
    san_ref_c.push(r_k.move)
    san_ref = san_ref_c.san(ref)
    c2 = c.copy(stack=False)
    c2.push(x)
    if r_k.move not in c2.legal_moves:
        return None
    c3 = c2.copy(stack=False)
    c3.push(r_k.move)
    if any(mv.from_square == x.to_square and mv.to_square == ref.to_square for mv in c3.legal_moves):
        return None                                          # the piece that recaptures can make the refuting move
    hits = [s for s in chess.SquareSet(c3.attacks_mask(r_k.move.to_square)) if c3.piece_at(s) is not None and c3.piece_at(s).color != me and c3.piece_at(s).piece_type != chess.KING]
    tgt = max(hits, key=lambda s: (G.worth(c3.piece_at(s)), -s)) if hits else None         # the goal asked again: the best piece the route's piece attacks once the refuter is gone
    pc_t = c2.piece_at(tgt) if tgt is not None else None
    dbal = AC.Pos(b.fen()).push(r2.move.uci()).push(x.uci()).balance(me) - P0.balance(me)
    saved = R._ids[0]
    q2 = R.new_question("win", G.worth(pc_t), dict(target=tgt), text=f"win their {pc_t.symbol().upper()} on {G.name(tgt)}") if pc_t is not None else None
    R._ids[0] = saved
    ag2 = AG.Agenda(State(fen=c2.fen(), us=node.us, pieces=[]), side_depth=0, chase_depth=node.chase_depth + 1, ctx=ctx, min_stake=1)
    ag2._chase_q = q2
    res3 = (yield T('route', chase_route_g(ag2, r_k.move), ag2))
    if res3["gap"] is not None or res3["killed"] is not None or res3["value"] < 1 - dbal:
        return None
    v = dbal + res3["value"]
    reads = frozenset({ref.from_square, ref.to_square, r2.move.to_square, r_k.move.from_square, r_k.move.to_square})
    part = f"after {san_x}: {r_k.san} asked again on the board after the exchange ({'; '.join(res3['parts'][:2])}) -> {v}"
    text = (f"{r_k.san} fails to {san_ref} (the refuter: the {pc_ref.symbol().upper()} on {G.name(ref.from_square)}); exchange it first -- {r2.san} {san_x}: the "
            f"{c2.piece_at(x.to_square).symbol().upper()} on {G.name(x.to_square)} cannot make that move; {r_k.san} is asked again after it")
    return dict(value=v, part=part, facts=list(res3.get("facts", [])), reads=reads, text=text)


