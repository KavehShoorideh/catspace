"""plan 60 / 69.1 THE GRAPH QUESTION POLICY (Kaveh 2026-10-22, standing: "any ordering agent must see the board and whatever we've thought about so far, all the
claims; a graph neural network"). A heterogeneous message-passing network (PyTorch Geometric) over

  the BOARD graph   -- `square` nodes (64; file, rank, attacked-by-us / by-them counts) and `piece` nodes (kind, colour, worth, attacked / guarded counts), edges
                       piece -on-> square, piece -attacks-> square, square -adj-> square;
  the FACT graph    -- `question` nodes (goal, stake, depth, status so far: open / verified / killed / gap, the kill word, the claim value), edges question
                       -targets-> square (the piece or square the question is about), question -about-> piece;
  the STATUS        -- a `root` node (balance, their threat, the line, the need, the fraction of the time left) joined to every question.

Output: a score per OPEN question node. It ORDERS only: the stops, the claims and the decision stay the rules' (plan 59 / 65; Lean `claim_mono`: the order
of asking does not touch soundness). Training (69.3): the grade logs -- for every agenda whose deciding question was RIGHT, and for every point k of the
order asked before it, one example: the questions asked before k carry their outcome (status, word, value), the rest are OPEN, and the target is the
deciding question among the open ones (a listwise softmax). Held out by puzzle id. The baseline it must beat is the linear prior (plan 53).

Run:  .venv/bin/python -m catspace.incubator.analyst.discovery.graph_policy train --log <assembled grade log> [--out data/derived/graph_policy.pt] [--epochs N]
      .venv/bin/python -m catspace.incubator.analyst.discovery.graph_policy eval  --log <assembled grade log>          (5-fold by puzzle: rank-first rate vs the prior)
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import chess

from catspace.incubator.analyst.discovery import game as GM
from catspace.incubator.analyst.discovery import graph as G
from catspace.io import paths

MODEL = "graph_policy.pt"
GOALS = ("mate", "win", "save", "promote", "fork", "guard", "cover", "remove", "gap")
STATUSES = ("open", "verified", "killed", "gap", "routed")
WORDS = ("tempo", "verify", "stopped", "reasoned", "chase", "budget", "cut", "none")
RWORDS = ("check", "capture", "threat", "delayed", "denies", "traps", "evade", "support:", "push:", "fix:", "no", "other")      # the route's word (its `why`'s first word)
HIDDEN = 32
# plan 72.5 THE SURVEY AS FEATURES. Feature-set VERSION: 0 = the board/question/route features above (every model trained before 72.5), 1 = 'survey1': per piece +4 (tempi to bear on the enemy
# king's zone, tempi to bear on a unit of higher worth, free landings, holds a passer's path) and per question +1 (fewest tempi of our units to the target; `regress.survey_features`).
# The layer widths are fixed by the first graph a model sees, so the saved model carries its version (`features` in the blob; absent = 0) and `load` builds the probe graph, and
# `score_*` the live graphs, at THAT version: a version-0 model sees exactly the old vectors (its scores do not change). `train` saves version 1 only when TRAIN_SURVEY is on.
# plan 72.8 (11d): VERSION 2 = 'survey1' + 'arrangement': per question one more column, log1p(the matched skeleton arrangement's stake - material - counter) / 5, 0 without one (the
# question dicts carry `arrangement`; the grade log rows carry it in `features`). `train` saves version 2 only when TRAIN_ARRANGEMENT is on; a version-0 or -1 model never sees the column.
TRAIN_SURVEY = False
TRAIN_ARRANGEMENT = False
_feat = [0]                                                    # the loaded model's feature-set version
_model = [None]
_loaded = [False]


def _word_class(word: str | None) -> int:
    w = (word or "none").split(":")[0].split(" ")[0]
    return WORDS.index(w) if w in WORDS else WORDS.index("none")


# ---------------------------------------------------------------------------------------------------------------- the graph

def build(board, us: bool, questions: list, status: dict | None, left_frac: float = 1.0, routes: list | None = None, game=None, survey: tuple | None = None, arrangement: bool = False):
    """-> HeteroData. `questions`: dicts with goal, stake, depth, target (square or None), status (open/verified/killed/gap), word, value. `routes` (plan 69.1, the
    route level): dicts with q (index into questions), uci, why, cost (the judge's first-level cost or None), order. The square and piece features are counts a
    player would name (attacked by us / by them). `survey` (plan 72.5; `regress.survey_features`): (unit features by location, per-question features) appended to the piece and question
    vectors (feature-set version 1); None = the old vectors."""
    import torch
    from torch_geometric.data import HeteroData
    game = game or GM.CHESS
    d = HeteroData()
    them = not us
    locs = list(game.locations())
    # locations (the squares): coordinates and who bears on them
    sq_x = []
    for s in locs:
        a_us, a_them = len(game.attackers(board, us, s)), len(game.attackers(board, them, s))
        sq_x.append(list(game.coords(s)) + [a_us / 4, a_them / 4, float(a_us > 0), float(a_them > 0)])
    d["square"].x = torch.tensor(sq_x, dtype=torch.float)
    # units (the pieces): kind, owner, worth, attacked / guarded, the royal unit in check
    units = game.units(board)
    in_check = game.in_check(board)
    p_x, on_src, on_dst, att_src, att_dst = [], [], [], [], []
    for i, u in enumerate(units):
        kind = [0.0] * game.kinds()
        kind[u.kind - 1] = 1.0
        p_x.append(kind + [float(u.owner == us), u.worth / 9, len(game.attackers(board, them if u.owner == us else us, u.loc)) / 4, len(game.attackers(board, u.owner, u.loc)) / 4,
                           float(u.loc == game.royal(board, u.owner) and in_check and game.turn(board) == u.owner)] + (survey[0].get(u.loc, [1.0, 1.0, 0.0, 0.0]) if survey else []))
        on_src.append(i); on_dst.append(u.loc)
        for t in game.attacks_from(board, u.loc):
            att_src.append(i); att_dst.append(t)
    d["piece"].x = torch.tensor(p_x, dtype=torch.float) if p_x else torch.zeros((0, game.kinds() + 5 + (4 if survey else 0)))
    d["piece", "on", "square"].edge_index = torch.tensor([on_src, on_dst], dtype=torch.long) if on_src else torch.zeros((2, 0), dtype=torch.long)
    d["piece", "attacks", "square"].edge_index = torch.tensor([att_src, att_dst], dtype=torch.long) if att_src else torch.zeros((2, 0), dtype=torch.long)
    adj_src, adj_dst = [], []
    for s in locs:
        for t in game.adjacent(s):
            adj_src.append(s); adj_dst.append(t)
    d["square", "adj", "square"].edge_index = torch.tensor([adj_src, adj_dst], dtype=torch.long)
    # questions
    q_x, qt_src, qt_dst, qp_src, qp_dst = [], [], [], [], []
    sq_index = {u.loc: i for i, u in enumerate(units)}
    for j, q in enumerate(questions):
        goal = [0.0] * len(GOALS)
        if q["goal"] in GOALS:
            goal[GOALS.index(q["goal"])] = 1.0
        st = [0.0] * len(STATUSES)
        st[STATUSES.index(q.get("status", "open")) if q.get("status", "open") in STATUSES else 0] = 1.0
        wd = [0.0] * len(WORDS)
        if q.get("status", "open") != "open":
            wd[_word_class(q.get("word"))] = 1.0
        q_x.append(goal + st + wd + [math.log1p(q["stake"]) / 5, min(q.get("depth", 0), 3) / 3, (q.get("value") or 0) / 10, float(q.get("status", "open") == "open")] + (survey[1][j] if survey else [])
                   + ([math.log1p(max(q.get("arrangement") or 0, 0)) / 5] if arrangement else []))      # plan 72.8: version 2's column
        t = q.get("target")
        if t is not None and t in sq_index or (t is not None and 0 <= t < len(locs)):
            qt_src.append(j); qt_dst.append(t)
            if t in sq_index:
                qp_src.append(j); qp_dst.append(sq_index[t])
    d["question"].x = torch.tensor(q_x, dtype=torch.float)
    d["question", "targets", "square"].edge_index = torch.tensor([qt_src, qt_dst], dtype=torch.long) if qt_src else torch.zeros((2, 0), dtype=torch.long)
    d["question", "about", "piece"].edge_index = torch.tensor([qp_src, qp_dst], dtype=torch.long) if qp_src else torch.zeros((2, 0), dtype=torch.long)
    # the root (status)
    st = status or {}
    bal, thr, line = st.get("balance", 0), st.get("threat", 0), st.get("line", 0)
    d["root"].x = torch.tensor([[bal / 10, min(thr, 20) / 20, float(line), (line - bal) / 10, left_frac, float(in_check)]], dtype=torch.float)
    n = len(questions)
    d["root", "sees", "question"].edge_index = torch.tensor([[0] * n, list(range(n))], dtype=torch.long) if n else torch.zeros((2, 0), dtype=torch.long)
    # routes (plan 69.1): a candidate move of a question, with its words, its price and its squares
    r_x, rq_src, rq_dst, rf_src, rf_dst, rt_src, rt_dst = [], [], [], [], [], [], []
    legal_keys = {game.move_key(m) for m in game.moves(board)}
    for i, r in enumerate(routes or []):
        m = game.move_from_key(board, r["uci"])
        frm, to = game.move_ends(m)
        w = (r.get("why") or "").removeprefix("fix: ").split(" ")[0]
        wd = [0.0] * len(RWORDS)
        wd[RWORDS.index(w) if w in RWORDS else RWORDS.index("other")] = 1.0
        pc = game.unit_at(board, frm)
        cap = game.unit_at(board, to)
        legal = r["uci"] in legal_keys                            # (the dummy route of a training graph need not be a move of the board)
        r_x.append(wd + [float(legal and game.gives_check(board, m)), float(legal and game.is_capture(board, m)), (cap.worth / 9) if cap else 0.0, (pc.worth / 9) if pc else 0.0,
                         math.log1p(r.get("cost") or 0) / 7, min(r.get("order", 0), 12) / 12, float(game.promotes(m)), len(game.attackers(board, not us, to)) / 4])
        rq_src.append(i); rq_dst.append(r["q"])
        rf_src.append(i); rf_dst.append(frm)
        rt_src.append(i); rt_dst.append(to)
    d["route"].x = torch.tensor(r_x, dtype=torch.float) if r_x else torch.zeros((0, len(RWORDS) + 8))
    d["route", "of", "question"].edge_index = torch.tensor([rq_src, rq_dst], dtype=torch.long) if rq_src else torch.zeros((2, 0), dtype=torch.long)
    d["route", "from", "square"].edge_index = torch.tensor([rf_src, rf_dst], dtype=torch.long) if rf_src else torch.zeros((2, 0), dtype=torch.long)
    d["route", "to", "square"].edge_index = torch.tensor([rt_src, rt_dst], dtype=torch.long) if rt_src else torch.zeros((2, 0), dtype=torch.long)
    # reverse edges so that messages flow both ways
    for (a, r, b) in list(d.edge_types):
        if a != b:
            d[b, "rev_" + r, a].edge_index = d[a, r, b].edge_index.flip(0)
    return d


# ---------------------------------------------------------------------------------------------------------------- the model

def survey_for(board, us: bool, questions: list):
    """The survey features (version 1) of a board and its questions' targets."""
    from catspace.incubator.analyst.discovery import regress as RG
    return RG.survey_features(board, us, [q.get("target") for q in questions])


def _sv(board, us, questions):
    return survey_for(board, us, questions) if (TRAIN_SURVEY or TRAIN_ARRANGEMENT) else None


def _train_version() -> int:
    """The feature-set version `train` builds and saves (72.5 / 72.8): 2 with TRAIN_ARRANGEMENT, 1 with TRAIN_SURVEY, else 0."""
    return 2 if TRAIN_ARRANGEMENT else int(TRAIN_SURVEY)


def make_model(meta):
    import torch
    from torch import nn
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.ModuleDict({t: Linear(-1, HIDDEN) for t in meta[0]})
            self.convs = nn.ModuleList([HeteroConv({et: SAGEConv((-1, -1), HIDDEN) for et in meta[1]}, aggr="sum") for _ in range(2)])
            self.out = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
            self.out_route = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

        def forward(self, data, routes: bool = False):
            x = {t: torch.relu(self.inp[t](data[t].x)) for t in data.node_types if data[t].x.numel() or t in ("question", "route")}
            for conv in self.convs:
                y = conv(x, data.edge_index_dict)
                x = {t: torch.relu(y.get(t, x[t])) + x[t] for t in x}
            if routes:
                return self.out_route(x["route"]).squeeze(-1)
            return self.out(x["question"]).squeeze(-1)
    return Net()


# ---------------------------------------------------------------------------------------------------------------- the data from the grade logs

def _rows(path):
    return [json.loads(l) for l in open(path)]


def _agendas(rows):
    by = {}
    for r in rows:
        by.setdefault((r["id"], r.get("set"), bool(r.get("prior", False)), r.get("run")), []).append(r)
    return {k: sorted(v, key=lambda r: r["order"]) for k, v in by.items()}


def _examples(rs):
    """One agenda's examples: for each point k before the deciding question (RIGHT), the questions asked before k revealed, the rest open; target = the decider."""
    right = rs[0].get("sf") == "RIGHT"
    pos = next((i for i, r in enumerate(rs) if r.get("decided")), None) if right and rs[0].get("decision") else None
    if pos is None and rs[0].get("answer"):                      # (as the prior's labelling, plan 53: the question that SHOULD have decided -- the one about the answer's square or move)
        ans = rs[0]["answer"]
        to = chess.parse_square(ans[2:4])
        pos = next((i for i, r in enumerate(rs) if r.get("target_sq") == to or r.get("move") == ans), None)
    if pos is None or pos == 0:
        return []
    board = chess.Board(rs[0]["fen"])
    us = board.turn
    out = []
    for k in range(0, pos):
        qs = []
        for i, r in enumerate(rs):
            revealed = i < k
            qs.append(dict(goal=r["goal"], stake=r["stake"], depth=0, target=r.get("target_sq"),
                           status=(r["status"] if revealed else "open"), word=(r.get("word") if revealed else None), value=(r["stake"] if revealed and r["status"] == "verified" else 0),
                           arrangement=(r.get("features") or {}).get("arrangement", 0)))      # plan 72.8: the logged arrangement feature (0 in logs written before it)
        open_idx = [i for i in range(len(rs)) if i >= k]
        out.append((board, us, qs, open_idx, pos, rs[0]["stake"]))
    return out


_DUMMY_ROUTE = [dict(q=0, uci="a1a2", why="other", cost=0, order=0)]      # (a graph always carries the route node type, so one trained module set serves both heads)


def route_examples(item_log_path):
    """plan 69.1 THE ROUTE LEVEL from the item log: per (puzzle, run, question) the routes read, the positive = the route that CLAIMED, else the route equal to the puzzle's answer;
    the rest asked are negatives. -> [(board, us, questions, routes, pos_index)]."""
    by = {}
    for l in open(item_log_path):
        r = json.loads(l)
        by.setdefault((r["id"], r.get("set"), r.get("fen"), r["qid"], r.get("pool"), r.get("tier")), []).append(r)
    out = []
    for key, rs in by.items():
        rs = sorted(rs, key=lambda r: r["order"])
        if len(rs) < 2:
            continue
        pos = next((i for i, r in enumerate(rs) if r["outcome"] == "claim"), None)
        if pos is None:
            ans = rs[0].get("answer")
            pos = next((i for i, r in enumerate(rs) if ans and r["uci"][:4] == ans[:4]), None)
        if pos is None:
            continue
        board = chess.Board(rs[0]["fen"])
        qs = [dict(goal=rs[0]["goal"], stake=rs[0]["stake"], depth=0, target=None, status="open")]
        routes = [dict(q=0, uci=r["uci"], why=r.get("why", ""), cost=r.get("cost"), order=r["order"]) for r in rs]
        out.append((board, board.turn, qs, routes, pos, rs[0]["id"]))
    return out


def train_routes(item_log_path, model=None, epochs=6, folds=None, seed=1, verbose=True):
    """Train (or evaluate, `folds`) the ROUTE head on the item log; the question head's weights are shared through the trunk. -> model."""
    import torch
    torch.manual_seed(seed); random.seed(seed)
    ex = route_examples(item_log_path)
    ids = sorted({e[5] for e in ex})
    if verbose:
        print(f"OBS route policy data: {len(ex)} (question, routes) groups from {len(ids)} puzzles")

    def fit(train_ex, test_ex, model):
        graphs = [(build(b, us, qs, None, routes=rt, survey=_sv(b, us, qs), arrangement=TRAIN_ARRANGEMENT), pos) for b, us, qs, rt, pos, _id in train_ex]
        if model is None:
            model = make_model((graphs[0][0].node_types, graphs[0][0].edge_types))
            with torch.no_grad():
                model(graphs[0][0]); model(graphs[0][0], routes=True)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        model.train()
        for ep in range(epochs):
            random.shuffle(graphs); tot = 0.0
            for g, pos in graphs:
                logits = model(g, routes=True)
                loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([pos]))
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
            if verbose:
                print(f"OBS route policy epoch {ep + 1}: mean loss {tot / max(len(graphs), 1):.3f} over {len(graphs)} groups")
        model.eval(); hit = hit_order = n = 0
        with torch.no_grad():
            for b, us, qs, rt, pos, _id in test_ex:
                sc = model(build(b, us, qs, None, routes=rt, survey=_sv(b, us, qs), arrangement=TRAIN_ARRANGEMENT), routes=True)
                hit += int(int(torch.argmax(sc)) == pos); hit_order += int(pos == 0); n += 1
        return model, hit, hit_order, n

    if folds:
        H = HO = N = 0
        for f in range(folds):
            test = {i for j, i in enumerate(ids) if j % folds == f}
            tr = [e for e in ex if e[5] not in test]; te = [e for e in ex if e[5] in test]
            if not tr or not te:
                continue
            _m, h, ho, n = fit(tr, te, None)
            H += h; HO += ho; N += n
        print(f"OBS route policy eval ({folds}-fold by puzzle, {N} groups): the deciding route ranks first under the policy {H}/{N}, under the engine's own order {HO}/{N}")
        return None
    model, h, ho, n = fit(ex, ex[: max(1, len(ex) // 10)], model)
    print(f"OBS route policy built: {len(ex)} groups (train-set rank-first {h}/{n}; the engine's order {ho}/{n})")
    return model


def _stake_order_hit(open_idx, qs, pos):
    best = max(open_idx, key=lambda i: (qs[i]["stake"], -i))
    return int(best == pos)


def train(log_path, out_path=None, epochs=6, folds=None, seed=1, verbose=True, item_log=None):
    import torch
    from catspace.incubator.analyst.discovery import question_prior as QP
    torch.manual_seed(seed); random.seed(seed)
    rows = _rows(log_path)
    ags = _agendas(rows)
    items = [(k, _examples(rs)) for k, rs in ags.items()]
    items = [(k, ex) for k, ex in items if ex]
    ids = sorted({k[0] for k, _ in items})
    prior = QP.load()

    def run_fold(train_items, test_items):
        graphs = []
        for k, ex in train_items:
            for board, us, qs, open_idx, pos, stake in ex:
                graphs.append((build(board, us, qs, None, routes=_DUMMY_ROUTE, survey=_sv(board, us, qs), arrangement=TRAIN_ARRANGEMENT), open_idx, pos))
        model = make_model((graphs[0][0].node_types, graphs[0][0].edge_types))
        with torch.no_grad():
            model(graphs[0][0])
            model(graphs[0][0], routes=True)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for ep in range(epochs):
            random.shuffle(graphs)
            tot = 0.0
            for g, open_idx, pos in graphs:
                scores = model(g)
                idx = torch.tensor(open_idx)
                logits = scores[idx]
                target = torch.tensor(open_idx.index(pos))
                loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss)
            if verbose:
                print(f"OBS graph policy epoch {ep + 1}: mean loss {tot / max(len(graphs), 1):.3f} over {len(graphs)} examples")
        hit = hit_prior = hit_stake = n = 0
        model.eval()
        with torch.no_grad():
            for k, ex in test_items:
                for board, us, qs, open_idx, pos, stake in ex:
                    g = build(board, us, qs, None, routes=_DUMMY_ROUTE, survey=_sv(board, us, qs), arrangement=TRAIN_ARRANGEMENT)
                    scores = model(g)
                    best = max(open_idx, key=lambda i: float(scores[i]))
                    hit += int(best == pos); n += 1
                    hit_stake += _stake_order_hit(open_idx, qs, pos)
        return model, hit, hit_stake, n

    if folds:
        H = HS = N = 0
        for f in range(folds):
            test = {i for j, i in enumerate(ids) if j % folds == f}
            tr = [(k, ex) for k, ex in items if k[0] not in test]
            te = [(k, ex) for k, ex in items if k[0] in test]
            if not tr or not te:
                continue
            _m, h, hs, n = run_fold(tr, te)
            H += h; HS += hs; N += n
        print(f"OBS graph policy eval ({folds}-fold by puzzle, {N} decision points): the deciding question ranks first under the graph policy {H}/{N}, under the stake order {HS}/{N}")
        return None
    model, h, hs, n = run_fold(items, items[: max(1, len(items) // 10)])
    if item_log and os.path.exists(item_log):
        model = train_routes(item_log, model=model, epochs=epochs, verbose=verbose)      # plan 69.1: the route head on the item log, the trunk shared
    out = out_path or paths.derived(MODEL)
    torch.save(dict(state=model.state_dict(), meta=(["square", "piece", "question", "root"], None), hidden=HIDDEN, rows=len(rows), agendas=len(items), features=_train_version()), out)
    print(f"OBS graph policy built: {len(items)} agendas, {sum(len(ex) for _, ex in items)} decision points -> {out} (train-set rank-first {h}/{n})")
    return model


# ---------------------------------------------------------------------------------------------------------------- at run time

def load(path=None):
    """The trained graph policy, or None (no file, or torch missing)."""
    if not _loaded[0]:
        _loaded[0] = True
        p = path or paths.derived(MODEL)
        try:
            import torch
            blob = torch.load(p, weights_only=False)
            _feat[0] = int(blob.get("features", 0))
            board = chess.Board()
            pq = [dict(goal="win", stake=1, depth=0, target=0, status="open")]
            g = build(board, True, pq, None, routes=[dict(q=0, uci="e2e4", why="threat 1", cost=3, order=0)], survey=survey_for(board, True, pq) if _feat[0] else None, arrangement=_feat[0] >= 2)
            model = make_model((g.node_types, g.edge_types))
            with torch.no_grad():
                model(g)
                model(g, routes=True)
            model.load_state_dict(blob["state"])
            model.eval()
            _model[0] = model
        except FileNotFoundError:
            _model[0] = None
        except Exception as e:  # pragma: no cover
            print(f"graph policy not loaded: {e!r}", file=sys.stderr)
            _model[0] = None
    return _model[0]


def score_questions(board: chess.Board, us: bool, questions: list, status: dict | None, left_frac: float = 1.0) -> list:
    """Scores for `questions` (dicts as `build` takes), in order; [] when no model."""
    model = load()
    if model is None or not questions:
        return []
    import torch
    with torch.no_grad():
        g = build(board, us, questions, status, left_frac, survey=survey_for(board, us, questions) if _feat[0] else None, arrangement=_feat[0] >= 2)
        return [float(x) for x in model(g)]


def score_routes(board: chess.Board, us: bool, question: dict, routes: list, status: dict | None, left_frac: float = 1.0) -> list:
    """Scores for the `routes` (dicts uci, why, cost, order) of one question; [] when no model."""
    model = load()
    if model is None or not routes:
        return []
    import torch
    with torch.no_grad():
        qs = [dict(question, status="open")]
        g = build(board, us, qs, status, left_frac, routes=[dict(r, q=0) for r in routes], survey=survey_for(board, us, qs) if _feat[0] else None, arrangement=_feat[0] >= 2)
        return [float(x) for x in model(g, routes=True)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "eval", "eval-routes"))
    ap.add_argument("--log", required=True)
    ap.add_argument("--items", default=None, help="the item log (plan 69.3) for the route head")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    if a.mode == "eval":
        train(a.log, folds=a.folds, epochs=a.epochs, verbose=False)
    elif a.mode == "eval-routes":
        train_routes(a.items or a.log, folds=a.folds, epochs=a.epochs, verbose=False)
    else:
        train(a.log, a.out, epochs=a.epochs, item_log=a.items)


if __name__ == "__main__":
    main()
