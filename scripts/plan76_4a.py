"""plan 76.4a THE HARNESS: the questions of the guard positions, the hand tables' answers to them, and the regression's (docs/ANALYST_PLAN.md 76.4a).

  .venv/bin/python scripts/plan76_4a.py harvest <dir> --sets ten|wide --tier 0|2 --jobs N
        every guard case (scripts/trace_identity.py's, the count budget, PYTHONHASHSEED=0) run with RECORDERS on the hand tables: every mate question (the board, the
        side, the placement) and every counterfactual the landing table reads, every material question (the board, the side, the target), every candidate list the
        fix filter reads (`still_routes`: the board, the candidates, the question's spec, the verdicts), every promotion question, every save question the agenda
        builds, every defect's fixes (the sub-questions and their routes) -- distinct per case, each with the hand table's answer and its milliseconds; one case per
        process, at most N at once -> <dir>/<sets>_t<tier>/<id>.jsonl
  .venv/bin/python scripts/plan76_4a.py regress <dir> --jobs N
        (a) THE REGRESSION ON THEM: per goal kind, the questions regressed, the defects derived, the fixes derived (the candidates the fix test reads), ms per question
        (the regression's and the hand table's)
  .venv/bin/python scripts/plan76_4a.py compare <dir> --jobs N
        (b) THE COMPARISON: per question, the derived defects against the hand table's AS SETS (the names map each hand kind to its place of the spec, printed), every
        disagreement printed; the same for the fixes (the fix filter's verdict per candidate; the fix families' routes against the derived fixes)
Outputs under artifacts/experiments/ (untracked)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))          # noqa: the guard's case list (scripts/trace_identity.py) is a script, not a module of the package

import chess  # noqa: E402

from catspace.incubator.analyst.discovery import gdl_chess as GC  # noqa: E402
from catspace.incubator.analyst.discovery import lemma_store as LS  # noqa: E402
from catspace.incubator.analyst.discovery import regression as RG  # noqa: E402


def _j(x):
    """A JSON-safe copy (tuples to lists, sets sorted)."""
    if isinstance(x, (list, tuple)):
        return [_j(y) for y in x]
    if isinstance(x, (set, frozenset)):
        return sorted((_j(y) for y in x), key=repr)
    if isinstance(x, dict):
        return {str(k): _j(v) for k, v in x.items()}
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


# ------------------------------------------------------------------------------------------------------------------------------------------ harvest

def _install(rec: list) -> None:
    """The recorders: each wraps a hand table, records the question and the answer (with its milliseconds), and returns the answer unchanged."""
    from catspace.incubator.analyst.discovery import agenda as AG
    from catspace.incubator.analyst.discovery import graph as G
    from catspace.incubator.analyst.discovery import regress as R

    def defs(ds):
        return [[d.kind, d.sq, list(d.by), d.why] for d in ds]

    om, omc, omat, opro, ofix = R.mate_defects, R.mate_defects_on, R.material_defects, R.promote_defects, R.fixes_of
    ostill = R._A.still_routes_pos

    def mate_defects(board, us, placement):
        t = time.perf_counter()
        out = om(board, us, placement)
        rec.append(["mate", G.exact_fen(board), G.colour_of(us), [[p.symbol(), at, frm] for p, at, frm in placement], defs(out), (time.perf_counter() - t) * 1000])
        return out

    def mate_defects_on(cf, me):
        t = time.perf_counter()
        out = omc(cf, me)
        rec.append(["mate_cf", cf.fen(), me, defs(out), (time.perf_counter() - t) * 1000])
        return out

    def material_defects(board, us, target):
        t = time.perf_counter()
        out = omat(board, us, target)
        rec.append(["win", G.exact_fen(board), G.colour_of(us), target, defs(out), (time.perf_counter() - t) * 1000])
        return out

    def promote_defects(board, sq, lift=False, priced=None):
        t = time.perf_counter()
        out = opro(board, sq, lift=lift, priced=priced)
        rec.append(["promote", G.exact_fen(board), sq, bool(lift), priced, defs(out[0]), _j(out[1]), (time.perf_counter() - t) * 1000])
        return out

    def still_routes_pos(pos, me, ucis, spec):
        t = time.perf_counter()
        out = ostill(pos, me, ucis, spec)
        rec.append(["still", pos.fen(), me, list(ucis), _j(spec), [s for _u, s in out], (time.perf_counter() - t) * 1000])
        return out

    def fixes_of(defect, q, board, us):
        out = ofix(defect, q, board, us)
        rec.append(["fixes", G.exact_fen(board), G.colour_of(us), q.goal, _j(q.pattern), [defect.kind, defect.sq, list(defect.by)],
                    [[f.goal, f.text, f.status, [r.move.uci() for r in R.all_routes(f)], list(f.unforced)] for f in out]])
        return out

    R.mate_defects, R.mate_defects_on, R.material_defects, R.promote_defects, R.fixes_of = mate_defects, mate_defects_on, material_defects, promote_defects, fixes_of
    R._A.still_routes_pos = still_routes_pos
    from catspace.incubator.analyst.discovery import words as WD
    from catspace.incubator.analyst.discovery import landing as LD
    LD.R.mate_defects_on = mate_defects_on                       # (landing.py reads regress as R: the same module object, patched once)
    _ = WD
    for meth in ("_ask_save_king", "_ask_save_piece", "_ask_save_threat", "_ask_stop_pawn"):
        orig = getattr(AG.Agenda, meth)

        def wrapped(self, q, head, _orig=orig, _m=meth):
            got = _orig(self, q, head)
            rec.append(["save", _m, G.exact_fen(self.board), self.me, _j(q.pattern), defs(q.defects)])
            return got
        setattr(AG.Agenda, meth, wrapped)


def harvest_case(out: str, sets: str, tier: int, pid: str) -> None:
    import trace_identity as TI
    rec: list = []
    _install(rec)
    for p, fen, us in TI._cases(sets):
        if p != pid:
            continue
        t = time.perf_counter()
        TI._run(fen, us, tier=tier)
        wall = time.perf_counter() - t
        seen, rows = set(), []
        for r in rec:
            k = json.dumps(r[:-1] if r[0] in ("mate", "mate_cf", "win", "promote", "still") else r)
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(out, pid + ".jsonl"), "w") as f:
            f.write(json.dumps(["case", pid, fen, us, tier, wall, len(rec)]) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"harvest case done: {pid} {len(rec)} {len(rows)}", flush=True)
        return
    raise SystemExit(f"no case {pid} in {sets}")


def harvest(d: str, sets: str, tier: int, jobs: int) -> None:
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    import trace_identity as TI
    out = os.path.join(d, f"{sets}_t{tier}")
    cases = [pid for pid, _f, _u in TI._cases(sets)]
    env = dict(os.environ, PYTHONHASHSEED="0")
    t0 = time.perf_counter()

    def child(pid):
        r = subprocess.run([sys.executable, __file__, "case", out, sets, str(tier), pid], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"case {pid}: {r.stderr[-3000:]}")
        return r.stdout.strip().splitlines()[-1]

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        done = list(ex.map(child, cases))
    calls = sum(int(x.split()[-2]) for x in done)
    rows = sum(int(x.split()[-1]) for x in done)
    print(f"OBS 76.4a harvest {sets} t{tier}: {len(cases)} cases, hand-table calls {calls}, distinct questions {rows}, wall {time.perf_counter() - t0:.1f} s -> {out}")


# ------------------------------------------------------------------------------------------------------------------------------------------ the regression on the questions

class Ctx:
    """One process's regressor (the spec, the binding, the store's names), shared by the questions of a case."""

    def __init__(self):
        self.B = GC.Binding()
        self.spec = RG.Spec(self.B.prog, certified=GC.CERTIFIED)
        self.R = RG.Regressor(self.spec, self.B)
        self.names = LS.defect_names()
        self.goals = LS.gdl_goals()
        self.unit = LS.goal_kinds().valuation.unit

    def mate(self, cf: chess.Board, me: bool, placement=()):
        g = RG.goal_of("mate", self.goals["mate"], self.B, self.B.role(me), extra=tuple(tuple(p) for p in placement))
        reg = self.R.regress(cf, g)
        return reg, RG.name(reg, self.names, self.B, cf)

    def win(self, b: chess.Board, me: bool, t: int):
        g = RG.goal_of("win", self.goals["win"], self.B, self.B.role(me), target=t, unit=self.unit)
        reg = self.R.regress(b, g)
        return reg, RG.name(reg, self.names, self.B, b)


def _future(board: chess.Board, placement: list, me: bool) -> chess.Board:
    from catspace.incubator.analyst.discovery import regress as R
    return R.future_board(board, [(chess.Piece.from_symbol(s), at, frm) for s, at, frm in placement], not me)


def _still_derived(C: Ctx, rec) -> list:
    """The derived fix test on one candidate list of the fix filter: for each candidate, is it NOT a fix (the defect stands, or a defect is new)? (the hand verdict's sense)."""
    _k, fen, me, ucis, spec, _stills, _ms = rec
    root_goal, q_goal, dkey, pl, before, tgt, frm, blk, win_tgt = spec
    b = chess.Board(fen)
    out = []
    if root_goal == "mate":
        key = (dkey[0], dkey[1])
        bef = {(k, s) for k, s in before}
        for u in ucis:
            x1 = b.copy(stack=False)
            x1.push(chess.Move.from_uci(u))
            cf = _future(x1, pl, me)
            _reg, names = C.mate(cf, me, pl)
            keys = [(n.kind, n.sq) for n in names]
            out.append(not (key not in keys and all(k in bef for k in keys)))
    elif root_goal == "win":
        for u in ucis:
            x1 = b.copy(stack=False)
            x1.push(chess.Move.from_uci(u))
            _reg, names = C.win(x1, me, win_tgt)
            out.append(bool(names))
    return out


def _regress_file(path: str) -> dict:
    """(a) on one case's questions: per goal kind, questions, defects and fixes derived and the milliseconds (the regression's, the hand table's)."""
    C = Ctx()
    st = {}

    def add(kind, q=0, d=0, f=0, ms=0.0, hms=0.0, cands=0):
        s = st.setdefault(kind, dict(questions=0, defects=0, fixes=0, ms=0.0, hand_ms=0.0, candidates=0))
        s["questions"] += q
        s["defects"] += d
        s["fixes"] += f
        s["ms"] += ms
        s["hand_ms"] += hms
        s["candidates"] += cands

    for ln in open(path):
        r = json.loads(ln)
        k = r[0]
        if k == "mate":
            cf = _future(chess.Board(r[1]), r[3], r[2])
            t = time.perf_counter()
            _reg, names = C.mate(cf, r[2], r[3])
            add("mate", 1, len(names), 0, (time.perf_counter() - t) * 1000, r[-1])
        elif k == "mate_cf":
            cf = chess.Board(r[1])
            t = time.perf_counter()
            _reg, names = C.mate(cf, r[2])
            add("mate", 1, len(names), 0, (time.perf_counter() - t) * 1000, r[-1])
        elif k == "win":
            b = chess.Board(r[1])
            t = time.perf_counter()
            _reg, names = C.win(b, r[2], r[3])
            add("win", 1, len(names), 0, (time.perf_counter() - t) * 1000, r[-1])
        elif k == "still":
            t = time.perf_counter()
            got = _still_derived(C, r)
            add(f"fix ({r[4][0]})", 1, 0, sum(1 for s in got if not s), (time.perf_counter() - t) * 1000, r[-1], len(got))
    return st


def _pool(fn, files: list, jobs: int) -> list:
    from multiprocessing import Pool
    with Pool(jobs) as p:
        return p.map(fn, files, chunksize=1)


ONLY: list = []                                         # `--only <set>_t<tier>`: the case files of that set alone


def _files(d: str) -> list:
    return sorted(str(p) for p in Path(d).glob("*_t*/*.jsonl") if not ONLY or p.parent.name in ONLY)


def regress(d: str, jobs: int) -> None:
    files = _files(d)
    t0 = time.perf_counter()
    tot: dict = {}
    for st in _pool(_regress_file, files, jobs):
        for k, s in st.items():
            T = tot.setdefault(k, dict(questions=0, defects=0, fixes=0, ms=0.0, hand_ms=0.0, candidates=0))
            for f in T:
                T[f] += s[f]
    for k in sorted(tot):
        T = tot[k]
        n = max(T["questions"], 1)
        if k.startswith("fix"):
            c = max(T["candidates"], 1)
            print(f"OBS 76.4a regress {k}: candidate lists {T['questions']}, candidates {T['candidates']}, fixes derived {T['fixes']}, "
                  f"ms per candidate {T['ms'] / c:.3f} (the hand filter's {T['hand_ms'] / c:.4f})")
        else:
            print(f"OBS 76.4a regress {k}: questions regressed {T['questions']}, defects derived {T['defects']}, ms per question {T['ms'] / n:.3f} "
                  f"(the hand table's {T['hand_ms'] / n:.4f})")
    print(f"OBS 76.4a regress: {len(files)} case files, wall {time.perf_counter() - t0:.1f} s")


# ------------------------------------------------------------------------------------------------------------------------------------------ (c) the cost in the engine

def cost_case(sets: str, tier: int, pid: str) -> None:
    """One guard case with clocks on the engine's defect and fix-filter calls (after the switch-over: the regression's) -> one line: calls and ms per kind, the wall."""
    import trace_identity as TI
    from catspace.incubator.analyst.discovery import regress as R
    tally = {}

    def wrap(name, fn):
        def w(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                s = tally.setdefault(name, [0, 0.0])
                s[0] += 1
                s[1] += (time.perf_counter() - t) * 1000
        return w
    R.mate_defects = wrap("mate", R.mate_defects)
    R.mate_defects_on = wrap("mate", R.mate_defects_on)
    R.material_defects = wrap("win", R.material_defects)
    R._still = wrap("fix filter (per candidate)", R._still)
    R.threat_defects = wrap("keep", R.threat_defects)
    for p, fen, us in TI._cases(sets):
        if p == pid:
            t = time.perf_counter()
            TI._run(fen, us, tier=tier)
            print("cost case done: " + json.dumps(dict(pid=pid, wall=time.perf_counter() - t, tally=tally, stats=R.regressor().stats)), flush=True)
            return


def cost(sets: str, tier: int, jobs: int) -> None:
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    import trace_identity as TI
    env = dict(os.environ, PYTHONHASHSEED="0")
    cases = [pid for pid, _f, _u in TI._cases(sets)]

    def child(pid):
        r = subprocess.run([sys.executable, __file__, "costcase", sets, str(tier), pid], capture_output=True, text=True, env=env)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("cost case done: ")]
        if r.returncode != 0 or not line:
            raise RuntimeError(f"case {pid}: {r.stderr[-3000:]}")
        return json.loads(line[-1][len("cost case done: "):])
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        got = list(ex.map(child, cases))
    tot: dict = {}
    for g in got:
        for k, (n, ms) in g["tally"].items():
            s = tot.setdefault(k, [0, 0.0])
            s[0] += n
            s[1] += ms
    st = {k: sum(g["stats"][k] for g in got) for k in ("regressions", "cached", "validated")}
    print(f"OBS 76.4a cost {sets} t{tier}: cases {len(cases)}, the runs' wall summed {sum(g['wall'] for g in got):.1f} s (wall {time.perf_counter() - t0:.1f} s, --jobs {jobs}); "
          f"regressions read {st['regressions']}, served from the cache {st['cached']}, validated {st['validated']}")
    for k in sorted(tot):
        n, ms = tot[k]
        print(f"OBS 76.4a cost {sets} t{tier} {k}: calls {n}, ms {ms:.0f}, ms per call {ms / max(n, 1):.4f}")


# ------------------------------------------------------------------------------------------------------------------------------------------ (d) the new goals

def _san(b: chess.Board, action) -> str:
    return b.san(chess.Move.from_uci(GC.uci_of(action[1])))


def exchange_trace(C: Ctx, fen: str, frm: int, to: int, log=print) -> dict:
    """THE NEW GOAL 'win this exchange' (the store's GOAL `exchange`: the capture of the target by the unit on `frm`, its value the exchange begun by that capture)
    regressed on one position, printed: the derivation, the named defects, the fixes (our legal moves after which the same capture wins and nothing is new)."""
    b = chess.Board(fen)
    me = b.turn
    us = C.B.role(me)
    g = RG.goal_of("exchange", C.goals["exchange"], C.B, us, target=to, unit=C.unit, frm=frm)
    reg = C.R.regress(b, g)
    names = RG.name(reg, C.names, C.B, b)
    cap = b.san(chess.Move(frm, to)) if chess.Move(frm, to) in b.legal_moves else f"{chess.square_name(frm)}x{chess.square_name(to)}"
    log(f"EXCHANGE {fen}: win this exchange -- {cap} (capture({', '.join(GDL_text(a) for a in g.head[1])}), value: the exchange begun by it, need {C.unit})")
    if reg.clauses:
        for idx, lits in reg.clauses:
            log(f"  derivation: capture's clause {idx}: " + "; ".join(f"{kind} {rel}{tuple(a) if a else ''} {'TRUE' if t else 'FALSE'}" for _j, kind, (rel, a), t in lits if rel))
    else:
        log(f"  derivation: the head holds ({len(reg.rows)} row(s)); the value read {reg.value if reg.value is not None else [d.value for d in reg.defects]}")
    for n in names:
        log(f"  defect {n.kind} ({n.why}); by {[chess.square_name(s) for s in n.by]}")
    fixes = []
    if names:
        key = (names[0].kind, names[0].sq)
        acts = [a for a in RG.actions(C.R, b, us) if not (C.B.ends(a)[0] == frm)]
        for a, mended, new in RG.fixes(C.R, b, g, key, acts, (lambda x: x), C.names):
            if mended and not new:
                fixes.append(_san(b, a))
        log(f"  fixes (our moves after which {cap} wins by the exchange, nothing new): " + (", ".join(fixes) if fixes else "none"))
    else:
        log("  no defect: the exchange already wins")
    return dict(defects=[(n.kind, n.why) for n in names], fixes=fixes)


def GDL_text(a) -> str:
    return a if isinstance(a, str) else "(" + " ".join(GDL_text(y) for y in a) + ")"


def floor_trace(C: Ctx, fen: str, log=print) -> dict:
    """THE MINED K+R TECHNIQUE'S FLOOR (its measure's floor: a mate within two moves): the mate goal (the store's `mate`, the same names) regressed within two of
    our moves, every reply read (regression.within), printed: per first move its escapes (each with our best answer's named defects) and the fixes."""
    b = chess.Board(fen)
    us = C.B.role(b.turn)
    g = RG.goal_of("mate", C.goals["mate"], C.B, us)
    res = RG.within(C.R, b, g, us, 2, C.names)
    fixes = [_san(b, w.action) for w in res if w.holds]
    log(f"FLOOR {fen}: mate within 2 moves of ours (the mate goal regressed at the leaves, every reply read)")
    for w in res:
        x1 = C.B.apply(b, w.action)
        if w.holds:
            met = [f"{_san(x1, r)} {next(_san(C.B.apply(x1, r), v.action) for v in RG.within(C.R, C.B.apply(x1, r), g, us, 1, C.names) if v.holds)}#"
                   for r in RG.actions(C.R, x1, C.B.other(us))]
            log(f"  {_san(b, w.action)}: a FIX -- every reply met: {', '.join(met)}")
        else:
            esc = "; ".join(f"{_san(x1, r)} escapes ({', '.join(f'{n.kind} {chess.square_name(n.sq) if n.sq is not None else chess.square_name(0)[:0]}' for n in ns) or 'no landing reads a mate'})"
                            for r, ns in w.escapes[:3])
            log(f"  {_san(b, w.action)}: {len(w.escapes)} escape(s) -- {esc}{' ...' if len(w.escapes) > 3 else ''}")
    log(f"  fixes (mate in two): {', '.join(fixes) if fixes else 'none'}")
    return dict(fixes=fixes)


def newgoals(d: str) -> None:
    """(d): the exchange goal on the guard positions (every capture of ours that loses now and has a fix: the first few printed, the counts as OBS) and the K+R floor."""
    import trace_identity as TI
    C = Ctx()
    n_caps = n_def = n_fix = 0
    shown = 0
    for sets in ("ten", "wide"):
        for pid, fen, us in TI._cases(sets):
            b = chess.Board(fen)
            me = b.turn
            for m in b.legal_moves:
                if not b.is_capture(m) or b.is_en_passant(m) or b.piece_type_at(m.to_square) == chess.KING:
                    continue
                n_caps += 1
                g = RG.goal_of("exchange", C.goals["exchange"], C.B, C.B.role(me), target=m.to_square, unit=C.unit, frm=m.from_square)
                names = RG.name(C.R.regress(b, g), C.names, C.B, b)
                if not names:
                    continue
                n_def += 1
                key = (names[0].kind, names[0].sq)
                acts = [a for a in RG.actions(C.R, b, C.B.role(me)) if C.B.ends(a)[0] != m.from_square]
                fx = [a for a, mended, new in RG.fixes(C.R, b, g, key, acts, (lambda x: x), C.names) if mended and not new]
                if fx:
                    n_fix += 1
                    if shown < 3:
                        shown += 1
                        print(f"-- {sets}/{pid}")
                        exchange_trace(C, fen, m.from_square, m.to_square)
    print(f"OBS 76.4a (d) exchange goal on the ten + wide root positions: captures of ours {n_caps}, losing now (a defect) {n_def}, with a fix {n_fix}")
    print("-- the K+R floor")
    got = floor_trace(C, "k7/8/1K6/8/8/8/8/7R w - - 0 1")
    print(f"OBS 76.4a (d) the K+R floor k7/8/1K6/8/8/8/8/7R w: the mate goal within two moves, fixes {got['fixes']}")


# ------------------------------------------------------------------------------------------------------------------------------------------ (b) the comparison

MAPPING = [
    ("no check", "mate's literal 0 `check(?c)` FALSE (the deepest false atom: attack_by onto their king, or the king's cell absent)"),
    ("checker takeable", "a witness of the true negated `has_move` (a legal move of theirs) flipping literal 0 of an `attack_by` clause: the attacker's cell (THE CAPTURE ATOM)"),
    ("flight", "a witness flipping check's literal 1, the checked unit's cell (THE KING-MOVE ATOM), not when it flips an attacker's cell too"),
    ("interposition", "a witness flipping empty's literal 1, `not occupied` on the checking ray, reached through slide's recursive clause (THE BLOCKING ATOM)"),
    ("not attacked", "win = capture(?k, us, ?f1, ?r1, target) with us on move: its clause 0, literal 0 (move_by) FALSE (THE CAPTURE ATOM)"),
    ("defended", "the capture holds, its value (the exchange law, the Rust see) short of the valuation's unit; by = attack(them, ?, ?, target) (THE ATTACK ATOMS)"),
    ("blocked", "the push chain: move_by(pawn, us, f, r, f, r+1) on the pawn placed at each square of its path, clause 1's literal 4 `empty` FALSE"),
    ("path attacked / promo square attacked", "the exchange law's value for them on the landing square with the pawn (the queen on the last rank) placed there > 0 (the hand compares COUNTS)"),
    ("stopper", "NOT A LITERAL OF ONE STATE: a unit of theirs reaching the path within the pawn's pushes (a multi-move reach: the tempo race) -- not regressed"),
    ("hangs", "the opponent's win goal holding on our unit (capture by them, the exchange >= the unit)"),
    ("in check", "check(us) true: the attackers of our king in its derivation (attack_by onto the king's cell)"),
    ("threatened / promotes / reach", "NOT LITERALS OF ONE STATE: their move after a pass, their passer's lemma claim, a mate root two moves away -- not regressed"),
]


FILTER_FAMILIES = ("cover the flight", "guard", "remove a blocker", "attack with tempo", "add an attacker", "remove the defender")      # routes_of's: the fix filter decides


def _family(text: str) -> str:
    """The hand fix table's family of a sub-question, by the text `_fixes_of` and its helpers give it."""
    for pre, fam in (("cover the flight", "cover the flight"), ("guard ", "guard"), ("take or pin", "take or pin the taker"),
                     ("take the interposer", "take the interposer"), ("remove the defender", "remove the defender"), ("remove ", "remove a blocker"),
                     ("attack ", "attack with tempo"), ("add an attacker", "add an attacker"), ("take its squares", "trap"), ("fork", "fork"),
                     ("discover", "discovery"), ("skewer", "skewer"), ("drive their king onto", "drive onto a skewer"), ("bring", "bring a unit (the join)")):
        if text.startswith(pre):
            return fam
    return " ".join(text.split()[:3])


def _key3(d) -> tuple:
    return (d[0], d[1], tuple(d[2]))


def _cmp(hand: list, der: list) -> str:
    if [tuple(_key3(x)) + (x[3],) for x in hand] == [tuple(_key3(x)) + (x[3],) for x in der]:
        return "exact"
    hs, ds = {_key3(x) for x in hand}, {_key3(x) for x in der}
    if hs == ds:
        return "set"
    if hs < ds:
        return "derived extra"
    if ds < hs:
        return "hand extra"
    return "both differ"


def _named(names) -> list:
    return [[n.kind, n.sq, list(n.by), n.why] for n in names]


def _xray_cover(cf: chess.Board, me: bool, target: int) -> bool:
    """The square attacked by `me` once their king is lifted off the board (a line through the king) but not with it standing (the hand's cover test)."""
    k = cf.king(not me)
    if k is None or cf.is_attacked_by(me, target):
        return False
    c = cf.copy(stack=False)
    c.remove_piece_at(k)
    return c.is_attacked_by(me, target)


def _promote_derived(C: Ctx, fen: str, sq: int, facts: dict) -> tuple:
    """The promotion regressed push by push: for each square of the path, the pawn placed on the square before it (its own on the first push), its owner on move:
    move_by(pawn, us, f, r, f, r+1)'s clause 1 read -- `empty` false is 'blocked'; the exchange law for them on the landing square with the pawn (the queen on the
    last rank) placed there > 0 is 'attacked'. -> (blocked squares, attacked squares)."""
    b = chess.Board(fen)
    pc = b.piece_at(sq)
    own = pc.color
    path = facts["path"]
    blocked, attacked = [], []
    prev = sq
    for t in path:
        x = b.copy(stack=False)
        x.remove_piece_at(sq)
        x.set_piece_at(prev, chess.Piece(chess.PAWN, own))
        x.turn = own
        x.ep_square = None
        f0, r0 = GC.sq(prev)
        f1, r1 = GC.sq(t)
        g = RG.Goal("push", ("move_by", ("pawn", C.B.role(own), f0, r0, f1, r1)))
        reg = C.R.regress(x, g)
        if any(d.kind == "false" and d.clause == 1 and d.atom[0] == "empty" for d in reg.defects):
            blocked.append(t)
        land = x.copy(stack=False)
        land.remove_piece_at(prev)
        land.set_piece_at(t, chess.Piece(chess.QUEEN if t == facts["promo"] else chess.PAWN, own))
        v = C.B.value(land, "exchange", GC.sq(t) + (C.B.role(not own),))
        if v is not None and v > 0:
            attacked.append(t)
        prev = t
    return blocked, attacked


def _compare_file(path: str) -> dict:
    """(b) on one case's questions -> {class counters, samples of every disagreement class}."""
    from collections import Counter
    from catspace.incubator.analyst.discovery import regress as R
    C = Ctx()
    cnt: Counter = Counter()
    samples: dict = {}
    case = Path(path).parent.name + "/" + Path(path).stem

    def sample(cls, what):
        s = samples.setdefault(cls, [])
        if len(s) < 4:
            s.append(f"{case}: {what}")

    roots: dict = {}
    for ln in open(path):
        r = json.loads(ln)
        k = r[0]
        if k in ("mate", "mate_cf"):
            if k == "mate":
                b = chess.Board(r[1])
                cf = _future(b, r[3], r[2])
                _reg, names = C.mate(cf, r[2], r[3])
                hand = r[4]
            else:
                cf = chess.Board(r[1])
                _reg, names = C.mate(cf, r[2])
                hand = r[3]
            der = _named(names)
            c = _cmp(hand, der)
            cnt[f"mate {c}"] += 1
            for x in hand:
                cnt[f"mate hand defect {x[0]}"] += 1
            for x in der:
                cnt[f"mate derived defect {x[0]}"] += 1
            if c != "exact":
                sample(f"mate {c}", f"{cf.fen()} hand {hand} derived {der}")
        elif k == "win":
            b = chess.Board(r[1])
            _reg, names = C.win(b, r[2], r[3])
            der = _named(names)
            c = _cmp(r[4], der)
            cnt[f"win {c}"] += 1
            for x in r[4]:
                cnt[f"win hand defect {x[0]}"] += 1
            if c != "exact":
                sample(f"win {c}", f"{r[1]} target {chess.square_name(r[3])} hand {r[4]} derived {der}")
        elif k == "still":
            _k, fen, me, ucis, spec, stills, _ms = r
            got = _still_derived(C, r)
            goal = spec[0]
            for u, h, d in zip(ucis, stills, got):
                if h == d:
                    cnt[f"fix ({goal}) agree"] += 1
                    continue
                b = chess.Board(fen)
                if goal == "mate" and h and not d:
                    x1 = b.copy(stack=False)
                    x1.push(chess.Move.from_uci(u))
                    cf = _future(x1, spec[3], me)
                    why = "a line through their king (the hand's cover test reads the square with the king on it)" if spec[5] is not None and _xray_cover(cf, me, spec[5]) else "other"
                    cls = f"fix (mate) derived fix, hand still: {why}"
                elif goal == "win" and not h and d:
                    why = "the candidate captures the target (the hand reads the exchange on our own unit there)" if chess.parse_square(u[2:4]) == spec[8] else "other"
                    cls = f"fix (win) hand fix, derived still: {why}"
                else:
                    cls = f"fix ({goal}) {'derived fix, hand still' if h else 'hand fix, derived still'}: other"
                cnt[cls] += 1
                sample(cls, f"{fen} {spec[1]} {spec[2]} candidate {u}")
        elif k == "fixes":
            _k, fen, me, qgoal, pattern, defect, subs = r
            b = chess.Board(fen)
            if qgoal == "mate":
                pl = [[pattern["sym"], pattern["at"], pattern.get("frm")]] + [list(x) for x in pattern.get("more", [])]
                g = RG.goal_of("mate", C.goals["mate"], C.B, C.B.role(me), extra=tuple(tuple(p) for p in pl))
                fut = (lambda x, pl=pl, me=me: _future(x, pl, me))
            elif qgoal == "win":
                g = RG.goal_of("win", C.goals["win"], C.B, C.B.role(me), target=pattern["target"], unit=C.unit)
                fut = (lambda x: x)
            else:
                continue
            key = (defect[0], defect[1])
            for goal_, text, status, routes, _unf in subs:
                fam = _family(text)
                if fam in FILTER_FAMILIES:
                    continue                                   # (the fix filter's families: their routes are the filter's verdicts, compared per candidate above)
                cnt[f"family {qgoal} {defect[0]} / {fam}: sub-questions"] += 1
                acts = [(C.B.role(me), GC.move_term(u)) for u in routes if chess.Move.from_uci(u) in b.legal_moves]
                if not acts:
                    continue
                for (a, mended, new) in RG.fixes(C.R, b, g, key, acts, fut, C.names):
                    ok = mended and not new
                    cnt[f"family {qgoal} {defect[0]} / {fam}: routes {'a derived fix' if ok else 'NOT a derived fix'}"] += 1
                    if not ok:
                        sample(f"family {qgoal} {defect[0]} / {fam}: a route that is not a derived fix",
                               f"{fen} {qgoal} {defect} route {GC.uci_of(a[1])} mended {mended} new {new}")
        elif k == "promote":
            _k, fen, sq, lift, priced, hand, facts, _ms = r
            if "path" not in facts:
                continue
            blocked, attacked = _promote_derived(C, fen, sq, facts)
            hb = sorted(x[1] for x in hand if x[0] == "blocked")
            ha = sorted(x[1] for x in hand if x[0] in ("path attacked", "promo square attacked"))
            cnt["promote blocked " + ("agree" if hb == sorted(blocked) else "differ")] += 1
            cnt["promote attacked " + ("agree" if ha == sorted(attacked) else "differ")] += 1
            cnt["promote hand stoppers (not regressed)"] += sum(1 for x in hand if x[0] == "stopper")
            if hb != sorted(blocked):
                sample("promote blocked differ", f"{fen} pawn {chess.square_name(sq)} hand {hb} derived {sorted(blocked)}")
            if ha != sorted(attacked):
                sample("promote attacked differ", f"{fen} pawn {chess.square_name(sq)} hand {[chess.square_name(s) for s in ha]} derived {[chess.square_name(s) for s in sorted(attacked)]}")
        elif k == "save":
            _k, meth, fen, me, pattern, hand = r
            b = chess.Board(fen)
            if meth == "_ask_save_king":
                ksq = b.king(me)
                P = C.R.prover(b)
                checkers = sorted(C.B.loc(lf.atom) for lf in P.leaves("check", (C.B.role(me),)) if lf.at[0] == "attack_by" and lf.at[2] == 0) if ksq is not None and b.turn == me and P.holds("check", (C.B.role(me),)) else []
                hs = sorted(x[1] for x in hand if x[0] == "in check")
                cnt["save in check " + ("agree" if hs == checkers else "differ")] += 1
                if hs != checkers:
                    sample("save in check differ", f"{fen} hand {hs} derived {checkers}")
            elif meth == "_ask_save_piece":
                t = pattern.get("target")
                reg, _n = C.win(b, not me, t)
                hh = any(x[0] == "hangs" for x in hand)
                cnt["save hangs " + ("agree" if hh == reg.holds else "differ")] += 1
                if hh != reg.holds:
                    sample("save hangs differ", f"{fen} target {chess.square_name(t)} hand {hand} derived holds {reg.holds}")
            else:
                cnt[f"save {meth} (not regressed)"] += 1
    _ = roots, R
    return dict(counts=dict(cnt), samples=samples)


def compare(d: str, jobs: int) -> None:
    from collections import Counter
    files = _files(d)
    t0 = time.perf_counter()
    tot: Counter = Counter()
    samples: dict = {}
    for got in _pool(_compare_file, files, jobs):
        tot.update(got["counts"])
        for cls, xs in got["samples"].items():
            s = samples.setdefault(cls, [])
            s += xs[: max(0, 6 - len(s))]
    print("OBS 76.4a the mapping (each hand kind -> its place of the spec):")
    for k, v in MAPPING:
        print(f"   {k}: {v}")
    for k in sorted(tot):
        print(f"OBS 76.4a compare {k}: {tot[k]}")
    for g in ("mate", "win"):
        n = sum(v for k, v in tot.items() if k.startswith(g + " ") and k.split(" ", 1)[1] in ("exact", "set", "derived extra", "hand extra", "both differ"))
        if n:
            print(f"OBS 76.4a agreement {g}: questions {n}, exact (kinds, squares, pieces, texts, order) {tot[g + ' exact']}, as sets {tot[g + ' exact'] + tot[g + ' set']} "
                  f"({100.0 * (tot[g + ' exact'] + tot[g + ' set']) / n:.3f}%)")
    for g in ("mate", "win"):
        n = sum(v for k, v in tot.items() if k.startswith(f"fix ({g})"))
        if n:
            print(f"OBS 76.4a agreement fix filter ({g}): candidates {n}, the same verdict {tot[f'fix ({g}) agree']} ({100.0 * tot[f'fix ({g}) agree'] / n:.4f}%)")
    print("OBS 76.4a disagreements, samples (every class):")
    for cls in sorted(samples):
        for s in samples[cls]:
            print(f"   [{cls}] {s[:400]}")
    print(f"OBS 76.4a compare: {len(files)} case files, wall {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    a = sys.argv[1:]

    def opt(name, default):
        return a[a.index(name) + 1] if name in a else default
    if "--only" in a:
        ONLY.append(opt("--only", ""))
    if a[0] == "case":
        harvest_case(a[1], a[2], int(a[3]), a[4])
    elif a[0] == "harvest":
        harvest(a[1], opt("--sets", "ten"), int(opt("--tier", "0")), int(opt("--jobs", "1")))
    elif a[0] == "regress":
        regress(a[1], int(opt("--jobs", "1")))
    elif a[0] == "compare":
        compare(a[1], int(opt("--jobs", "1")))
    elif a[0] == "costcase":
        cost_case(a[1], int(a[2]), a[3])
    elif a[0] == "cost":
        cost(opt("--sets", "ten"), int(opt("--tier", "0")), int(opt("--jobs", "1")))
    elif a[0] == "newgoals":
        newgoals(a[1] if len(a) > 1 else "")
    else:
        sys.exit(__doc__)
