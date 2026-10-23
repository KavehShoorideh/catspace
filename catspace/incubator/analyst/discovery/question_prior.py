"""PLAN 53: THE PRIOR THAT ORDERS THE QUESTIONS (Kaveh 2026-10-20: "wire in the prior that orders the questions to be asked, the next question out of the
pool ... training data: did the deciding line agree with Stockfish and did we go down it; if we neglected to ask the right question, a negative; never
penalise the right question wrongly answered -- that is a bug or a missing lemma").

The prior ORDERS, it never decides (the all-weapons rule): the agenda still stops by stakes and claims; the prior only says which open question to ask next.

  features(q, board, us, depth)   -- what is known of a question BEFORE it is asked (its goal, stake, the target's piece and the counts on its square,
                                     the check state, the depth); generic, no square names
  score(features)                 -- the model's log-odds that this question leads to the (right) decision; 0 with no model
  build                           -- train on the question log (analyse --qlog [--sf]): per agenda the questions in the order asked; label 1 for the
                                     question that decided a RIGHT decision (Stockfish agrees, or the puzzle's answer when no Stockfish verdict), label 0 for
                                     every question asked before it; when the decision was wrong or none, the question whose goal names the answer's
                                     target is the one that should have been asked (1) and those asked before it are 0. Logistic regression over the
                                     feature dict (sklearn at build time only); the model is a JSON of weights the agenda reads without sklearn.

Run:  .venv/bin/python -m catspace.incubator.analyst.discovery.question_prior build [--log question_log.jsonl] [--out question_prior.json]
      .venv/bin/python -m catspace.incubator.analyst.discovery.question_prior loop --seeds 9-12 --rounds 2 [--per-round N]   (plan 53 (3): sample, run, grade, retrain, eval)
      .venv/bin/python -m catspace.incubator.analyst.discovery.question_prior calibrate [--log estimates_log.jsonl]   (plan 61 B3: Stockfish scores the logged estimated leaves; the HELD fraction per class)
      .venv/bin/python -m catspace.incubator.analyst.discovery.question_prior eval  (held-out by puzzle id: how often the deciding question ranks first)
"""
from __future__ import annotations

import argparse
import json
import math
import os

import chess

from catspace.io import paths
from catspace.incubator.analyst.discovery import graph as G

LOG = "question_log.jsonl"
MODEL = "question_prior.json"
_model = [None]            # the loaded model: dict(weights: {feature: w}, bias: float) or None
_loaded = [False]


def features(q, board, us, depth: int, status: dict | None = None, game=None) -> dict:
    """What is known of a question before it is asked. Keys are generic words; values 1 (one-hot) or small counts. plan 69.1: with the STATUS (plan 59: balance, their threat, the line) the
    need and the stake's distance to it are features too (plan 68's deferral, learned instead of written)."""
    from catspace.incubator.analyst.discovery import game as GM
    game = game or GM.CHESS
    me = G.colour_of(us)
    f = {"goal=" + q.goal: 1, "stake": q.stake, "depth": min(depth, 3), "in_check": int(game.in_check(board))}
    if status is not None:
        need = status["line"] - status["balance"]
        f["need"] = need
        f["stake_minus_need"] = q.stake - need
        f["stake_reaches_need"] = int(q.stake >= need or q.goal == "save")
        f["their_threat"] = status.get("threat", 0)
    t = q.pattern.get("target", q.pattern.get("at", q.pattern.get("pawn")))
    if t is not None:
        pc = game.unit_at(board, t)
        if pc is not None:
            f["target=" + game.kind_name(pc.kind)] = 1
            f["target_ours"] = int(pc.owner == me)
        f["target_attacked_by_us"] = len(game.attackers(board, me, t))
        f["target_guarded"] = len(game.attackers(board, not me, t))
    k = game.royal(board, not me)
    if k is not None:
        f["their_king_attacked_squares"] = sum(1 for s in game.attacks_from(board, k) if game.attackers(board, me, s))
    arr = q.pattern.get("arrangement")                                               # plan 72.8 (11d): the matched skeleton arrangement's stake - material - counter, or 0 -- a saved model has no
    f["arrangement"] = (q.stake - arr["material"] - arr["counter"]) if arr else 0     # weight for it (`score` reads w.get(k, 0)), so its scores do not change until a retrain is adopted
    f["our_pieces"] = sum(1 for u in game.units(board, me) if u.worth != 1)         # (every unit of ours that is not a pawn, the king included, as before)
    f["captures_available"] = sum(1 for m in game.moves(board) if game.is_capture(board, m)) if game.turn(board) == me else 0
    return f


def load(path=None) -> dict | None:
    if not _loaded[0]:
        _loaded[0] = True
        p = path or paths.derived(MODEL)
        try:
            _model[0] = json.load(open(p))
        except FileNotFoundError:
            _model[0] = None
    return _model[0]


def score(f: dict, model: dict | None = None) -> float:
    m = model or _model[0]
    if m is None:
        return 0.0
    w = m["weights"]
    return m["bias"] + sum(w.get(k, 0.0) * v for k, v in f.items())


def _rows(path):
    return [json.loads(l) for l in open(path)]


def _akey(r: dict) -> tuple:
    """One agenda = one (puzzle, set, prior on/off): the loop logs the same set twice (with the prior and the stake order) and they must not merge."""
    return (r["id"], r.get("set"), bool(r.get("prior", False)), r.get("run"))      # plan 69.3: the grade RUN too (the same puzzle is logged by every grade)


def _label(rows_of_agenda: list) -> list:
    """(row, label) for one agenda's questions in the order asked, by the rule in the module doc. Rows after the positive are not used (not asked before it)."""
    rows = sorted(rows_of_agenda, key=lambda r: r["order"])
    right = rows[0].get("sf") == "RIGHT" if rows[0].get("sf") else rows[0].get("match")
    pos = None
    if rows[0].get("decision") and right:
        pos = next((i for i, r in enumerate(rows) if r.get("decided")), None)
    else:
        ans = rows[0].get("answer")
        if ans:
            to = chess.parse_square(ans[2:4])
            pos = next((i for i, r in enumerate(rows) if r.get("target_sq") == to or r.get("move") == ans), None)
    if pos is None:
        return []
    return [(r, 1 if i == pos else 0) for i, r in enumerate(rows[:pos + 1])]


def build(log_path, out_path, verbose=True) -> dict:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    rows = _rows(log_path)
    by = {}
    for r in rows:
        by.setdefault(_akey(r), []).append(r)
    X, y, groups = [], [], []
    for key, rs in by.items():
        for r, lab in _label(rs):
            X.append(r["features"]); y.append(lab); groups.append(key[0])
    if not X or len(set(y)) < 2:
        raise SystemExit(f"OBS prior: not enough labelled rows ({len(X)} rows, labels {sorted(set(y))})")
    v = DictVectorizer(sparse=False)
    M = v.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(M, y)
    model = {"weights": {n: float(w) for n, w in zip(v.get_feature_names_out(), clf.coef_[0])}, "bias": float(clf.intercept_[0]),
             "rows": len(X), "agendas": len(set(groups)), "positives": int(sum(y))}
    json.dump(model, open(out_path, "w"), indent=1)
    if verbose:
        top = sorted(model["weights"].items(), key=lambda t: -abs(t[1]))[:12]
        print(f"OBS prior built: {len(X)} rows from {len(set(groups))} agendas, {sum(y)} positives -> {out_path}")
        print("OBS prior weights (largest): " + ", ".join(f"{k} {w:+.2f}" for k, w in top))
    return model


def evaluate(log_path, folds=5) -> None:
    """Held out by puzzle: with the prior's order, how often does the question that should decide rank FIRST among the ones asked before it (vs the stake order)?"""
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    rows = _rows(log_path)
    by = {}
    for r in rows:
        by.setdefault(_akey(r), []).append(r)
    items = [(key, _label(rs)) for key, rs in by.items()]
    items = [(k, l) for k, l in items if len(l) > 1]
    ids = sorted({k[0] for k, _ in items})
    hit_prior = hit_stake = n = 0
    for fold in range(folds):
        test = {i for j, i in enumerate(ids) if j % folds == fold}
        tr = [(r, lab) for k, l in items if k[0] not in test for r, lab in l]
        te = [(k, l) for k, l in items if k[0] in test]
        if not tr or not te:
            continue
        v = DictVectorizer(sparse=False)
        M = v.fit_transform([r["features"] for r, _ in tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(M, [lab for _, lab in tr])
        w = dict(zip(v.get_feature_names_out(), clf.coef_[0]))
        model = {"weights": {k: float(x) for k, x in w.items()}, "bias": float(clf.intercept_[0])}
        for k, l in te:
            n += 1
            best = max(l, key=lambda t: score(t[0]["features"], model))
            hit_prior += best[1]
            hit_stake += l[0][1] if len(l) == 1 else int(max(l, key=lambda t: (t[0]["stake"], -t[0]["order"]))[1])
    print(f"OBS prior eval ({folds}-fold by puzzle, {n} agendas with a question asked before the deciding one): the deciding question ranks first "
          f"under the prior {hit_prior}/{n}, under the stake order {hit_stake}/{n}")


def grade(log_path, depth: int = 14) -> None:
    """plan 53 (1): the STOCKFISH VERDICT written into the log's rows (a separate pass in its own process: the engine never runs beside an agenda -- its thread
    stalled the set runs; and the grader is never a premise). One engine call per (position, decision), cached."""
    from catspace.incubator.analyst.discovery import verdict as VD
    rows = _rows(log_path)
    cache, n = {}, 0
    for r in rows:
        if r.get("sf") is not None:                      # already graded (the loop regrades nothing)
            continue
        key = (r["fen"], r.get("decision"))
        if key not in cache:
            cache[key] = VD.verdict(r["fen"], r.get("decision"), depth)
            n += 1
        r["sf"], r["sf_loss"], r["sf_best"] = cache[key]
    VD.close()
    with open(log_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    agendas = {(r["id"], r.get("set")): r["sf"] for r in rows}
    from collections import Counter
    print(f"OBS prior grade: {n} decisions graded by Stockfish (depth {depth}) over {len(agendas)} agendas: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(agendas.values()).items(), key=lambda t: str(t[0]))))
    # plan 65.3: a decision is a PROVED claim or an UNPROVED pick (the best unfinished line, Claim.word); the grade keeps them apart, so an unproved guess never
    # reads as a proof.  `proved` is None in logs written before 65.3 (every decision there was proved).
    by = {}
    for r in rows:
        k = (r["id"], r.get("set"))
        p = r.get("proved")
        by[k] = ("none" if r.get("decision") is None else "table" if r.get("proved_by") == "table" else "proved" if p is None or p else "unproved", r["sf"])      # plan 62.3: a table-closed decision is its own label
    if any(v[0] in ("unproved", "table") for v in by.values()):
        c = Counter(f"{kind} {sf}" for kind, sf in by.values() if kind != "none")
        print("OBS prior grade by proof: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) + f", none {sum(1 for v in by.values() if v[0] == 'none')}")


EST_LOG = "estimates_log.jsonl"
CALIBRATION = "estimate_calibration.json"
DEFAULT_CONFIDENCE = 0.5                          # plan 61 B2: the confidence of an estimated leaf's class until `calibrate` has measured it
_cal = [None]


def estimate_confidence(cls: str) -> float:
    """plan 61 B2/B3: the confidence of an estimated leaf of class `cls` -- the HELD fraction `calibrate` wrote to data/derived/estimate_calibration.json, nothing else; DEFAULT_CONFIDENCE
    when the file or the class is absent."""
    if _cal[0] is None:
        try:
            with open(paths.derived(CALIBRATION)) as fh:
                _cal[0] = json.load(fh)
        except (OSError, ValueError):
            _cal[0] = {}
    row = _cal[0].get(cls)
    return float(row["confidence"]) if isinstance(row, dict) and row.get("n") else DEFAULT_CONFIDENCE


def held(status: int, sf_cp: "int | None", tol: int = 50) -> "bool | None":
    """plan 61 B3: did an estimate HOLD? The estimate's status is the leaf's material balance (ours) minus their pass threat; its SIGN against the level line is 'level or better' (>= 0) or
    'worse' (< 0). Stockfish's centipawns from our side give the same sign with `tol` cp of slack around level (verdict.TOL): HELD iff the two signs agree. None when ungraded."""
    if sf_cp is None:
        return None
    return (status >= 0) == (sf_cp >= -tol)


def calibrate(log_path=None, depth: int = 14, out=None) -> dict:
    """plan 61 B3: the CALIBRATION PASS (a separate process, never beside an agenda: the engine stalls the set runs). Every logged estimated leaf without a grade is scored by Stockfish at
    `depth` (one call per FEN, cached) from the side to move at the leaf (ours); the rows are rewritten with `sf_cp` and `held`; the HELD fraction per class is written to
    estimate_calibration.json -- the only source of an estimated claim's confidence. -> {class: {n, held, confidence}}."""
    import chess.engine
    from catspace.incubator.analyst.discovery import verdict as VD
    log_path = log_path or paths.experiment(EST_LOG)
    if not os.path.exists(log_path):
        print("OBS estimate calibrate: no estimates log (no estimated leaf was logged); nothing written")
        return {}
    rows = _rows(log_path)
    cache, n = {}, 0
    for r in rows:
        if r.get("sf_cp") is not None:
            continue
        e = VD._get()
        if e is None:
            break
        if r["fen"] not in cache:
            b = chess.Board(r["fen"])
            cache[r["fen"]] = e.analyse(b, chess.engine.Limit(depth=depth))["score"].pov(chess.WHITE if r["us"] == "w" else chess.BLACK).score(mate_score=10000)
            n += 1
        r["sf_cp"] = cache[r["fen"]]
        r["held"] = held(r["status"], r["sf_cp"])
    VD.close()
    tmp = str(log_path) + ".tmp"
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, log_path)
    table: dict = {}
    for r in rows:
        if r.get("held") is None:
            continue
        t = table.setdefault(r["cls"], dict(n=0, held=0))
        t["n"] += 1
        t["held"] += int(bool(r["held"]))
    for t in table.values():
        t["confidence"] = round(t["held"] / t["n"], 4)
    if not table:                                                   # nothing graded: the file keeps what it had (an empty table would erase the measured numbers)
        print(f"OBS estimate calibrate: {len(rows)} logged leaves, none graded; nothing written")
        return table
    out = out or paths.derived(CALIBRATION)
    with open(out, "w") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
    _cal[0] = None
    print(f"OBS estimate calibrate: {len(rows)} logged leaves, {n} scored by Stockfish (depth {depth}), "
          + (", ".join(f"{k}: held {v['held']} of {v['n']} (confidence {v['confidence']})" for k, v in sorted(table.items())) or "none graded"))
    return table


def reward(rows: list) -> dict:
    """plan 53 (3): the loop's reward over a set of log rows (pure). Per agenda (id, set): right = Stockfish RIGHT (or the puzzle's answer when ungraded), wrong = a decision
    that is not right, none = no decision. The reward is the mean number of questions asked BEFORE the deciding one (its `order`), over right decisions only; lower is better."""
    by = {}
    for r in rows:
        by.setdefault(_akey(r), []).append(r)
    right = wrong = none = 0
    before = []
    for rs in by.values():
        rs = sorted(rs, key=lambda r: r["order"])
        if not rs[0].get("decision"):
            none += 1
            continue
        ok = rs[0]["sf"] == "RIGHT" if rs[0].get("sf") else bool(rs[0].get("match"))
        if not ok:
            wrong += 1
            continue
        right += 1
        d = next((r for r in rs if r.get("decided")), None)
        if d is not None:
            before.append(d["order"])
    return dict(agendas=len(by), right=right, wrong=wrong, none=none, mean_before=(sum(before) / len(before)) if before else None)


def loop(seeds: str, rounds: int, per_round: int = 1, n: int = 10) -> None:
    """plan 53 (3): draw fresh samples, run them (a subprocess per set: the agenda's process-level caches, and the grader never beside an agenda) with the current prior,
    grade the new rows, retrain on all rows so far, evaluate. One OBS line per round."""
    import subprocess
    import sys
    from catspace.incubator.analyst.discovery import sample_puzzles as SP
    a, _, b = seeds.partition("-")
    pool = list(range(int(a), int(b or a) + 1))
    log, model_path = paths.experiment(LOG), paths.derived(MODEL)
    for rd in range(1, rounds + 1):
        used, pool = pool[:per_round], pool[per_round:]
        if not used:
            print(f"OBS prior loop: seed range {seeds} exhausted after {rd - 1} rounds")
            return
        names = []
        for seed in used:
            name = f"rand{seed}_{n}"
            json.dump([SP.record(r) for r in SP.rows(n, seed)], open(paths.derived(f"ledger_puzzles_{name}.json"), "w"), indent=1)
            names.append(name)
        prev = json.load(open(model_path)) if os.path.exists(model_path) else None
        for name in names:                                       # each set TWICE: the stake order (the baseline: the invariant's witness and the negatives' source) and the prior
            base = [sys.executable, "-m", "catspace.incubator.analyst.discovery.analyse", "--set", name, "--quiet", "--qlog"]
            subprocess.run(base, check=True)
            if prev is not None:
                subprocess.run(base + ["--prior"], check=True)
        grade(log)
        model = build(log, model_path, verbose=False)
        rows = _rows(log)
        mine = [r for r in rows if r.get("set") in names]
        m = loop_metrics(mine, prev, model)
        m.update(round=rd, seeds=used, model_rows=model["rows"], positives=model["positives"], negatives=model["rows"] - model["positives"], model_age_rows=(prev or {}).get("rows"))
        with open(paths.experiment(METRICS), "a") as fh:
            fh.write(json.dumps(m) + "\n")
        mb = lambda v: "n/a" if v is None else f"{v:.2f}"
        print(f"OBS prior loop round {rd} (seed {','.join(map(str, used))}): agendas {m['agendas']}, decisions (prior run) right {m['right']} wrong {m['wrong']} none {m['none']}; "
              f"INVARIANT decisions identical with and without the prior {m['identical']}/{m['compared']} (differ by the budget's cut {m['differ_budget']}, differ otherwise = BUG {m['differ_bug']}); questions before the deciding one, same puzzles: prior {mb(m['mean_before_prior'])} vs stake order {mb(m['mean_before_base'])}; "
              f"model rows {model['rows']} positives {model['positives']} negatives {model['rows'] - model['positives']} (built from {m['model_age_rows']} rows before this round), weight drift {mb(m['weight_drift'])}")
        evaluate(log)


METRICS = "prior_loop_metrics.jsonl"


def loop_metrics(rows: list, prev_model: dict | None, model: dict) -> dict:
    """plan 53 (3): what says the loop behaves as planned, on one round's rows (both runs of each set). (a) THE INVARIANT: the prior orders, never decides -- the decision of
    every puzzle run with the prior equals the stake order's (`identical`/`compared`); (b) the PAIRED reward: on the same puzzles, the mean questions asked before the deciding
    one with the prior vs with the stake order (right decisions only, by the grader); (c) the prior run's right/wrong/none; (d) weight drift: the L2 distance between the model
    before and after the round (0 = nothing learnt; a jump = the round moved it)."""
    by = {}
    for r in rows:
        by.setdefault(_akey(r), []).append(r)
    dec = {}
    for k, rs in by.items():
        rs.sort(key=lambda r: r["order"])
        dec[k] = rs[0]
    compared = identical = differ_budget = differ_bug = 0
    for (pid, st, pr, run), r in dec.items():                  # (plan 69.3: the key carries the grade run)
        if pr and (pid, st, False, run) in dec:
            compared += 1
            if dec[(pid, st, False, run)].get("decision") == r.get("decision"):
                identical += 1
            elif any(x.get("status") == "gap" or x.get("fuse_hits") for x in by[(pid, st, False, run)] + by[(pid, st, True, run)]):      # (`fuse_hits`: the agenda's budget cuts, logged on every row; a question cut on a route keeps the status 'killed')
                differ_budget += 1                       # the budget cut one run's chase: the ORDER decided what got asked in time (GYWyJ: the stake order's fuse went to the mates, the prior asked the knight first and decided Nd5, SF RIGHT)
            else:
                differ_bug += 1                          # the same questions, the same budget, a different decision: a BUG (the prior must never decide)
    def before(flag):
        out = []
        for (pid, st, pr, run), rs in by.items():      # (plan 69.3: the key carries the grade run)
            if pr != flag or not rs[0].get("decision"):
                continue
            ok = rs[0]["sf"] == "RIGHT" if rs[0].get("sf") else bool(rs[0].get("match"))
            d = next((x for x in rs if x.get("decided")), None)
            if ok and d is not None and (flag or (pid, st, True, run) in dec):      # the stake order's figure over the puzzles that also ran with the prior
                out.append(d["order"])
        return (sum(out) / len(out)) if out else None
    pr_rows = [r for r in rows if r.get("prior")]
    rw = reward(pr_rows if pr_rows else rows)
    drift = None
    if prev_model is not None:
        keys = set(prev_model["weights"]) | set(model["weights"])
        drift = math.sqrt(sum((prev_model["weights"].get(k, 0.0) - model["weights"].get(k, 0.0)) ** 2 for k in keys) + (prev_model["bias"] - model["bias"]) ** 2)
    return dict(agendas=rw["agendas"], right=rw["right"], wrong=rw["wrong"], none=rw["none"], compared=compared, identical=identical, differ_budget=differ_budget, differ_bug=differ_bug,
                mean_before_prior=before(True), mean_before_base=before(False), weight_drift=drift)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("build", "eval", "grade", "calibrate", "loop"))
    ap.add_argument("--seeds", default="9-10")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--per-round", type=int, default=1)
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    log = a.log or paths.experiment(LOG)
    if a.mode == "loop":
        loop(a.seeds, a.rounds, a.per_round)
    elif a.mode == "build":
        build(log, a.out or paths.derived(MODEL))
    elif a.mode == "grade":
        grade(log)
    elif a.mode == "calibrate":
        calibrate(a.log, out=a.out)
    else:
        evaluate(log)


if __name__ == "__main__":
    main()


def forcing_resources(board, us) -> dict:
    """plan 61 C3 (logged for plan 60, no decision use): the forcing resources on the root board: OUR forcing moves now (`forcing_why`, stake 1), OUR pieces bearing on their king's region (they attack the
    king's square or one of the squares next to it), the king's FLIGHTS (adjacent squares not held by their own pieces and not attacked by us) and BLOCKERS (adjacent squares held by their own pieces)."""
    import chess
    from catspace.incubator.analyst.discovery import graph as G
    from catspace.incubator.analyst.discovery import regress as R
    me = G.colour_of(us)
    b = G.with_turn(board, me) if board.turn != me else board
    ksq = b.king(not me)
    n_forcing = sum(1 for m in b.legal_moves if R.forcing_why(b, m, 1, me, mate_threats=True))
    if ksq is None:
        return dict(forcing_moves=n_forcing, bearing=0, flights=0, blockers=0)
    region = [ksq] + list(chess.SquareSet(chess.BB_KING_ATTACKS[ksq]))
    bearing = {a for s in region for a in b.attackers(me, s) if b.piece_type_at(a) != chess.KING}
    adj = list(chess.SquareSet(chess.BB_KING_ATTACKS[ksq]))
    blockers = sum(1 for s in adj if b.piece_at(s) is not None and b.piece_at(s).color != me)
    flights = sum(1 for s in adj if b.piece_at(s) is None or b.piece_at(s).color == me if not b.is_attacked_by(me, s))
    return dict(forcing_moves=n_forcing, bearing=len(bearing), flights=flights, blockers=blockers)
