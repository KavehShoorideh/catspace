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


def features(q, board: chess.Board, us, depth: int) -> dict:
    """What is known of a question before it is asked. Keys are generic words; values 1 (one-hot) or small counts."""
    me = G.colour_of(us)
    f = {"goal=" + q.goal: 1, "stake": q.stake, "depth": min(depth, 3), "in_check": int(board.is_check())}
    t = q.pattern.get("target", q.pattern.get("at", q.pattern.get("pawn")))
    if t is not None:
        pc = board.piece_at(t)
        if pc is not None:
            f["target=" + chess.piece_name(pc.piece_type)] = 1
            f["target_ours"] = int(pc.color == me)
        f["target_attacked_by_us"] = len(board.attackers(me, t))
        f["target_guarded"] = len(board.attackers(not me, t))
    k = board.king(not me)
    if k is not None:
        f["their_king_attacked_squares"] = sum(1 for s in chess.SquareSet(board.attacks(k)) if board.is_attacked_by(me, s))
    f["our_pieces"] = sum(1 for p in board.piece_map().values() if p.color == me and p.piece_type != chess.PAWN)
    f["captures_available"] = sum(1 for m in board.legal_moves if board.is_capture(m)) if board.turn == me else 0
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
        by.setdefault((r["id"], r.get("set")), []).append(r)
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
        by.setdefault((r["id"], r.get("set")), []).append(r)
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


def reward(rows: list) -> dict:
    """plan 53 (3): the loop's reward over a set of log rows (pure). Per agenda (id, set): right = Stockfish RIGHT (or the puzzle's answer when ungraded), wrong = a decision
    that is not right, none = no decision. The reward is the mean number of questions asked BEFORE the deciding one (its `order`), over right decisions only; lower is better."""
    by = {}
    for r in rows:
        by.setdefault((r["id"], r.get("set")), []).append(r)
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
        for name in names:
            cmd = [sys.executable, "-m", "catspace.incubator.analyst.discovery.analyse", "--set", name, "--quiet", "--qlog"]
            if os.path.exists(model_path):
                cmd.append("--prior")
            subprocess.run(cmd, check=True)
        grade(log)
        model = build(log, model_path, verbose=False)
        mine = [r for r in _rows(log) if r.get("set") in names]
        rw = reward(mine)
        mb = "n/a" if rw["mean_before"] is None else f"{rw['mean_before']:.2f}"
        print(f"OBS prior loop round {rd} (seed {','.join(map(str, used))}): agendas {rw['agendas']}, decisions right {rw['right']} wrong {rw['wrong']} none {rw['none']}, "
              f"mean questions asked before the deciding one {mb} (right decisions only), model rows {model['rows']} positives {model['positives']}")
        evaluate(log)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("build", "eval", "grade", "loop"))
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
    else:
        evaluate(log)


if __name__ == "__main__":
    main()
