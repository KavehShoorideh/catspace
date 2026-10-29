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


def _files(d: str) -> list:
    return sorted(str(p) for p in Path(d).glob("*_t*/*.jsonl"))


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


if __name__ == "__main__":
    a = sys.argv[1:]

    def opt(name, default):
        return a[a.index(name) + 1] if name in a else default
    if a[0] == "case":
        harvest_case(a[1], a[2], int(a[3]), a[4])
    elif a[0] == "harvest":
        harvest(a[1], opt("--sets", "ten"), int(opt("--tier", "0")), int(opt("--jobs", "1")))
    elif a[0] == "regress":
        regress(a[1], int(opt("--jobs", "1")))
    else:
        sys.exit(__doc__)
