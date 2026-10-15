"""THE PROPOSER AGAINST PUZZLE SURPRISES (Kaveh 2026-10-15; docs/ANALYST_PLAN.md 40.5). The objective is surprise against puzzles, never Stockfish parity: a Lichess puzzle gives who wins what,
how (the line) and when (its length); the event loop is run on the puzzle position; a SURPRISE is
  (a) the puzzle's move is not among our candidates,
  (b) its value is not among our findings (our interval for it stops below the line's gain),
  (c) our chain for it is longer than the puzzle's line by more than one link.
Each surprise is classified by the loop's own records (`classify`): which EVENT was missing (no condition made the move a candidate), which QUESTION was never opened (the candidate cap /
the budget cut it), which ANSWER was wrong (the move was evaluated and its interval is wrong -- a lemma to demote when a lemma answered, else a word to check), and whether its shapes had no
LEMMA (no question of the decision was answered by one).
The PROPOSER makes three kinds of proposal from a dev batch and keeps one only if it survives a fresh sample (it fixes at least one fresh surprise and introduces no new fresh miss):
  (i)   new CONDITIONS: a rule "the moves after which a vocabulary word (watch.py's lift terms and the base terms) rose (or fell)" become candidates -- ranked by the surprises it
        fixes against the candidates it newly opens on the non-surprises (the cost);
  (ii)  new LEMMA shapes: the chains that recur across the SOLVED puzzles only (the store learned from them, at the dev threshold), each checked on the fresh sample (a lemma is kept when it
        answers questions there without a contradiction and without a new miss);
  (iii) STAKE / PRIORITY adjustments: the base priority of each source of candidates (opportunity, poke, touch, check, capture ...) learned by the STRICT PAIRWISE RANKING of watch.py
        (greedy, lexicographic: the tag that orders most (answer, other) pairs correctly among the pairs still tied) over the (opened, fixed) pairs -- never by hand.
The solver is the checker of any value: every value in the chain is the ledger's, confirmed by the exact solver; a proposal claims no value of its own.
  python -m catspace.incubator.analyst.discovery.ledger_puzzles --propose     (one round on dev 60, then checked on fresh 60)
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import time

import chess

from catspace.io import paths

CLASSES = ("event missing", "question never opened", "answer wrong", "no surprise")
MAX_EXTRA = 4                      # the candidates a proposed condition may add to one decision


# ---------------------------------------------------------------------------------------------------------------------------------------------
# SURPRISES

def classify(r: dict) -> dict:
    """-> {surprise, a, b, c, klass, no_lemma} for one decided puzzle (the dict of ledger_puzzles._one with the event loop). Pure: no engine."""
    l = r["ledger"]
    ans, themes = r["answer"], r["themes"]
    mate = any(t.startswith("mateIn") for t in themes)
    gen = set(l.get("generated", []))
    rows = {x["uci"]: x for x in l.get("rows", [])}
    a = ans not in rows and not (mate and l.get("correct"))
    V = r.get("value", 0)
    b = (ans in rows and not mate and rows[ans]["hi"] < V) or (ans not in rows and not mate)
    plies = l.get("plies", len(themes))
    c = ans in rows and rows[ans].get("depth", 0) > math.ceil(plies / 2) + 1
    hits = (l.get("lemma") or {}).get("hits", 0)
    if ans not in gen and ans not in rows:
        klass = "event missing"
    elif ans not in rows:
        klass = "question never opened"
    elif b or (not l.get("correct") and not mate):
        klass = "answer wrong"
    else:
        klass = "no surprise"
    surprise = (a or b or c) and not (mate and l.get("correct"))
    if not surprise and not l.get("correct") and klass != "no surprise":
        surprise = True
    return dict(surprise=bool(surprise), a=bool(a), b=bool(b), c=bool(c), klass=klass if surprise else "no surprise", no_lemma=(hits == 0),
                lemma_involved=(hits > 0 and klass == "answer wrong"))


def surprise_report(res: list, label: str, log=print) -> list:
    recs = []
    for r in res:
        cl = classify(r)
        recs.append(dict(id=r["id"], fen=r["fen"], answer=r["answer"], chosen=r["ledger"].get("move"), correct=bool(r["ledger"].get("correct")), themes=r["themes"][:4], **cl))
    n = len(recs)
    by = {k: sum(1 for x in recs if x["klass"] == k) for k in CLASSES}
    log(f"OBS surprises {label}: {sum(x['surprise'] for x in recs)} of {n} puzzles; (a) the move not a candidate {sum(x['a'] for x in recs)}, (b) its value not among our findings "
        f"{sum(x['b'] for x in recs)}, (c) our chain more than one link longer than the line {sum(x['c'] for x in recs)}")
    log(f"OBS surprise classes {label}: " + "; ".join(f"{k}: {v}" for k, v in by.items()) +
        f"; of the surprises, no question answered by a lemma: {sum(1 for x in recs if x['surprise'] and x['no_lemma'])}, a lemma involved in a wrong answer: {sum(1 for x in recs if x['lemma_involved'])}")
    return recs


# ---------------------------------------------------------------------------------------------------------------------------------------------
# RUNNING A SET (the harness's worker, with a proposal applied)

def run_set(set_name: str, n: int, lemmas: str = "off", lemma_file: str | None = None, proposal: dict | None = None, only_ids: list | None = None, promote: int = 5) -> list:
    from . import ledger_puzzles as LP
    from . import chain_lemmas as CL
    os.environ.update(LEDGER_LOOP="event", LEDGER_NO_BASELINES="1", LEDGER_LEMMAS=lemmas, LEDGER_LEMMA_FILE=lemma_file or CL.default_path(), LEDGER_STAKE_KNOB="3",
                      LEDGER_PROPOSAL=json.dumps(proposal) if proposal else "null", LEDGER_ONLY_IDS=",".join(only_ids) if only_ids else "")
    rows = LP.dev_rows(n) if set_name == "dev" else LP.dev_rows(n, 11)
    with multiprocessing.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = [x for x in pool.map(LP.one, rows, chunksize=1) if x is not None]
    cut = sum(1 for r in res if r["ledger"].get("cut"))
    print(f"OBS batch {set_name} {n} (lemmas {lemmas}{', proposal ' + json.dumps(proposal)[:90] if proposal else ''}{', ' + str(len(only_ids)) + ' puzzles' if only_ids else ''}): "
          f"{sum(1 for r in res if r['ledger'].get('correct'))} of {len(res)} correct, {cut} decision(s) hit the {os.environ.get('LEDGER_TIME_LIMIT', '300')} s time cap (read off a cut chain: ESTIMATED, never PROVEN)")
    return res


def correct_map(res: list) -> dict:
    return {r["id"]: bool(r["ledger"].get("correct")) for r in res}


def compare(base: dict, new: dict) -> dict:
    fixed = sorted(i for i in base if not base[i] and new.get(i))
    broke = sorted(i for i in base if base[i] and not new.get(i, False))
    return dict(base=sum(base.values()), new=sum(new.values()), fixed=fixed, new_misses=broke, kept=bool(fixed) and not broke)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# (i) CONDITIONS: a vocabulary word whose rise makes the move a candidate

def _terms() -> list:
    from . import watch as W
    return W.threat_terms()[0]


def rule_moves(fen: str, rules: list) -> list:
    """The moves of the side to move after which the rule's word rose (sign +1) or fell (-1) by at least one, in the mover's own reading: the candidates a proposed condition adds."""
    import analyst_core as A
    import catspace.incubator.analyst.discovery  # noqa: F401
    b = chess.Board(fen)
    mover = b.turn
    moves = list(b.legal_moves)
    fens = [fen] + [_after(b, m) for m in moves]
    out = []
    for rule in rules:
        vals = [row[0] for row in A.component_values_us(fens, [rule["term"]], mover == chess.WHITE)]
        base = vals[0]
        sel = [m.uci() for m, v in zip(moves, vals[1:]) if (v - base) * rule["sign"] >= 1]
        out += sel[:MAX_EXTRA]
    return list(dict.fromkeys(out))


def _after(b: chess.Board, m: chess.Move) -> str:
    x = b.copy(stack=False)
    x.push(m)
    return x.fen()


def condition_matrix(res: list, terms: list):
    """Per puzzle: the legal moves and, for every term, the change in the mover's reading after each move -> {id: (moves, delta array [moves x terms])}"""
    import numpy as np
    import analyst_core as A
    import catspace.incubator.analyst.discovery  # noqa: F401
    out = {}
    for r in res:
        b = chess.Board(r["fen"])
        moves = list(b.legal_moves)
        fens = [r["fen"]] + [_after(b, m) for m in moves]
        M = np.array(A.component_values_us(fens, terms, b.turn == chess.WHITE), dtype=np.int64)
        out[r["id"]] = ([m.uci() for m in moves], M[1:] - M[0])
    return out


def propose_conditions(res: list, recs: list, top: int = 3, log=print) -> list:
    """Rank rules (term, sign) by the surprises of class event-missing / never-opened they would fix (the answer is among the moves the word rose for) against the candidates they would
    newly open on the non-surprises; keep the best `top` that fix at least one."""
    import numpy as np
    terms = _terms()
    from . import base_vocabulary as V
    mats = condition_matrix(res, terms)
    byid = {r["id"]: r for r in res}
    sur = [x for x in recs if x["surprise"] and x["klass"] in ("event missing", "question never opened")]
    non = [x for x in recs if not x["surprise"]]
    scored = []
    T = len(terms)
    for sign in (1, -1):
        fixed = np.zeros(T, np.int64)
        for x in sur:
            moves, D = mats[x["id"]]
            if x["answer"] in moves:
                fixed += ((D[moves.index(x["answer"])] * sign) >= 1)
        cost = np.zeros(T, np.float64)
        for x in non:
            moves, D = mats[x["id"]]
            evaluated = {c["uci"] for c in byid[x["id"]]["ledger"].get("rows", [])}
            new = np.array([m not in evaluated for m in moves])
            cost += (((D * sign) >= 1) & new[:, None]).sum(axis=0)
        cost = cost / max(1, len(non))
        for t in range(T):
            if fixed[t] >= 1 and cost[t] <= MAX_EXTRA:
                scored.append((-int(fixed[t]), float(cost[t]), V.size(V.parse(terms[t])), sign, terms[t]))
    scored.sort()
    out = []
    seen = set()
    for nf, cost, size, sign, term in scored:
        key = (term, sign)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(kind="condition", term=term, sign=sign, fixes_dev=-nf, cost=round(cost, 2), gloss=_gloss(term)))
        if len(out) >= top:
            break
    log(f"OBS proposer (i) conditions: {len(scored)} (word, direction) rules fix at least one dev surprise at a cost <= {MAX_EXTRA} new candidates per puzzle; the best {len(out)} proposed")
    for p in out:
        log(f"OBS   propose: candidates are the moves after which '{p['gloss']}' {'rose' if p['sign'] > 0 else 'fell'}  [{p['term']}] -- fixes {p['fixes_dev']} dev surprise(s), opens {p['cost']} new candidates per non-surprise puzzle")
    return out


def _gloss(term: str) -> str:
    from . import watch as W
    try:
        return W.term_gloss(term)
    except Exception:           # noqa: BLE001
        return term


# ---------------------------------------------------------------------------------------------------------------------------------------------
# (iii) STAKE / PRIORITY: the strict pairwise ranking over the sources of candidates

def propose_stakes(res: list, recs: list, log=print) -> dict | None:
    """Pairs (the puzzle's move, another evaluated move) of the puzzles whose move was generated but never evaluated or evaluated and cut: the tags of a move are the features; the greedy
    lexicographic strict pairwise ranking of watch.py picks the tag that orders most pairs correctly among those still tied. -> {"tag_bonus": {...}} or None"""
    byid = {r["id"]: r for r in res}
    pairs = []
    for x in recs:
        if x["klass"] not in ("question never opened", "event missing", "answer wrong"):
            continue
        l = byid[x["id"]]["ledger"]
        tags = l.get("tags", {})
        if x["answer"] not in tags:
            continue
        for c in l.get("rows", []):
            if c["uci"] != x["answer"]:
                pairs.append((set(tags[x["answer"]]), set(tags.get(c["uci"], []))))
    all_tags = sorted({t for a, b in pairs for t in a | b})
    chosen, undecided = [], list(range(len(pairs)))
    for _ in range(3):
        best = None
        for t in all_tags:
            if t in [c[0] for c in chosen]:
                continue
            s = sum((t in pairs[i][0]) - (t in pairs[i][1]) for i in undecided)
            if best is None or abs(s) > abs(best[1]):
                best = (t, s)
        if not best or best[1] <= 0:
            break
        chosen.append(best)
        undecided = [i for i in undecided if ((best[0] in pairs[i][0]) - (best[0] in pairs[i][1])) == 0]
    log(f"OBS proposer (iii) priorities: {len(pairs)} (move, other) pairs from {sum(1 for x in recs if x['klass'] in ('question never opened', 'event missing', 'answer wrong'))} surprise puzzles; "
        f"strict pairwise ranking picked: {chosen or 'nothing that orders more pairs right than wrong'}")
    if not chosen:
        return None
    bonus = {t: 10 * (2 ** (len(chosen) - 1 - i)) for i, (t, _) in enumerate(chosen)}
    return dict(kind="priority", tag_bonus=bonus, ranking=chosen)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# THE ROUND

def moves_map(res: list) -> dict:
    return {r["id"]: r["ledger"].get("move") for r in res}


def wrong_lines(res: list, recs: list, top: int = 3) -> list:
    """The top wrong-answer cases, by the size of the gap: the puzzle's move, what the chain chose, both intervals, and the line of the chain that chose it (the decision's own lines)."""
    out = []
    byid = {r["id"]: r for r in res}
    for x in recs:
        if x["klass"] != "answer wrong" or x["correct"]:
            continue
        r = byid[x["id"]]
        rows = {c["uci"]: c for c in r["ledger"].get("rows", [])}
        ch = rows.get(r["ledger"].get("move"), {})
        an = rows.get(x["answer"], {})
        gap = ch.get("lo", 0) - an.get("lo", 0)
        out.append((gap, x, r, ch, an))
    out.sort(key=lambda t: -t[0])
    cases = []
    for gap, x, r, ch, an in out[:top]:
        lines = [l for l in r["ledger"].get("lines", []) if l.startswith("DECISION:") or l.strip().startswith((an.get("san") or "~") + ":") or l.strip().startswith((ch.get("san") or "~") + ":")]
        cases.append(dict(id=x["id"], fen=r["fen"], themes=x["themes"], answer=x["answer"], answer_interval=[an.get("lo"), an.get("hi")], chosen=r["ledger"].get("move"),
                          chosen_interval=[ch.get("lo"), ch.get("hi")], puzzle_value=r.get("value"), lines=lines[:4]))
    return cases


def main(n: int = 60, promote: int = 5, log=print):
    from . import chain_lemmas as CL
    t0 = time.time()
    store = CL.default_path()
    log(f"OBS proposer: one round on dev {n}, checked on fresh {n}; lemma store {store} (threshold {promote}); a proposal is KEPT when it fixes at least one fresh surprise and introduces no new fresh miss; "
        f"the batches: dev and fresh once without, once with ALL proposals on, then leave-one-out only for proposals that changed a decision, on the changed puzzles only")
    dev = run_set("dev", n, "on", store, None, promote=promote)
    recs = surprise_report(dev, "dev", log)
    json.dump(recs, open(paths.derived("surprises_dev.json"), "w"), indent=1)
    log(f"OBS surprises saved {paths.derived('surprises_dev.json')}")
    cases = wrong_lines(dev, recs)
    for c in cases:
        log(f"OBS WRONG ANSWER {c['id']} {c['themes']} {c['fen']}: the puzzle's move {c['answer']} was valued {c['answer_interval']} (the line gains {c['puzzle_value']}); the chain chose {c['chosen']} valued {c['chosen_interval']}")
        for l in c["lines"]:
            log(f"OBS    {l.strip()[:300]}")
    fresh = run_set("fresh", n, "on", store, None, promote=promote)
    frecs = surprise_report(fresh, "fresh", log)
    dev_c, fresh_c = correct_map(dev), correct_map(fresh)
    log(f"OBS baseline: dev {sum(dev_c.values())} of {len(dev_c)}, fresh {sum(fresh_c.values())} of {len(fresh_c)}")
    proposals = propose_conditions(dev, recs, 3, log)
    pri = propose_stakes(dev, recs, log)
    if pri:
        proposals.append(pri)

    def prop_of(ps):
        return dict(rules=[dict(term=p["term"], sign=p["sign"]) for p in ps if p["kind"] == "condition"], tag_bonus=(next((p["tag_bonus"] for p in ps if p["kind"] == "priority"), {})))

    kept = []
    if proposals:
        d_all = run_set("dev", n, "on", store, prop_of(proposals), promote=promote)
        f_all = run_set("fresh", n, "on", store, prop_of(proposals), promote=promote)
        for tag, base_res, all_res, base_c in (("dev", dev, d_all, dev_c), ("fresh", fresh, f_all, fresh_c)):
            bm, am = moves_map(base_res), moves_map(all_res)
            changed = [i for i in bm if bm[i] != am.get(i)]
            c = compare(base_c, correct_map(all_res))
            log(f"OBS proposals all on, {tag}: {c['new']} of {len(base_c)} correct (without {c['base']}); {len(changed)} decision(s) changed; fixed {c['fixed']}, new misses {c['new_misses']}")
            for p in proposals:
                p.setdefault("loo", {})[tag] = dict(changed=changed, effect=None)
            if changed:
                for p in proposals:
                    rest = [q for q in proposals if q is not p]
                    r2 = run_set(tag, n, "on", store, prop_of(rest), only_ids=changed, promote=promote)
                    cm = correct_map(r2)
                    allc = correct_map(all_res)
                    fixed = sorted(i for i in changed if allc.get(i) and not cm.get(i))          # right with it, wrong without it
                    broke = sorted(i for i in changed if cm.get(i) and not allc.get(i))
                    p["loo"][tag]["effect"] = dict(fixed=fixed, new_misses=broke)
        for p in proposals:
            ef = {t: (p["loo"][t]["effect"] or dict(fixed=[], new_misses=[])) for t in ("dev", "fresh")}
            p["kept"] = bool(ef["fresh"]["fixed"]) and not ef["fresh"]["new_misses"]
            name = p.get("gloss") or json.dumps(p.get("tag_bonus"))
            log(f"OBS ablation [{p['kind']}] {name}: dev: fixes {ef['dev']['fixed']}, new misses {ef['dev']['new_misses']} (of {len(p['loo']['dev']['changed'])} changed decisions); "
                f"fresh: fixes {ef['fresh']['fixed']}, new misses {ef['fresh']['new_misses']} (of {len(p['loo']['fresh']['changed'])} changed) -> {'KEPT' if p['kept'] else 'DROPPED'}")
            if p["kept"]:
                kept.append(p)
    # (ii) lemma shapes from the SOLVED puzzles
    solved = [i for i, ok in dev_c.items() if ok]
    solved_store = paths.derived("chain_lemmas_solved.json")
    if os.path.exists(solved_store):
        os.remove(solved_store)
    learn = run_set("dev", n, "learn", solved_store, None, only_ids=solved, promote=promote)
    summ = CL.merge([x["ledger"]["lemma"] for x in learn if x["ledger"].get("lemma")], solved_store, promote)
    log(f"OBS proposer (ii) lemmas: learned from the {len(solved)} solved dev puzzles only: {json.dumps(summ)}")
    f3 = run_set("fresh", n, "on", solved_store, None, promote=promote)
    rep = {x["id"]: x["ledger"]["lemma"] for x in f3 if x["ledger"].get("lemma")}
    hits = sum(x["hits"] for x in rep.values()); con = sum(len(x["contradictions"]) for x in rep.values())
    by_id: dict = {}
    for x in rep.values():
        for k, v in x.get("hit_by", {}).items():
            by_id[k] = by_id.get(k, 0) + v
    cf = compare(fresh_c, correct_map(f3))
    log(f"OBS proposer (ii) with the solved-puzzle lemmas on fresh: {cf['new']} of {len(fresh_c)} correct (without lemmas from the dev store {cf['base']}), {hits} questions answered by a lemma, {con} contradictions, "
        f"new misses {cf['new_misses']}")
    store_j = CL.load(solved_store)
    lemma_items = []
    for lid in [k for k, _ in sorted(by_id.items(), key=lambda kv: -kv[1])[:3]]:
        e = next(x for x in store_j["lemmas"].values() if x["id"] == lid)
        pk = next(k for k, x in store_j["lemmas"].items() if x["id"] == lid)
        where = [i for i, x in rep.items() if x.get("hit_by", {}).get(lid)]
        tmp = paths.derived("chain_lemmas_ablate.json")
        j = json.loads(json.dumps(store_j))
        j["lemmas"][pk]["promoted"] = False
        json.dump(j, open(tmp, "w"))
        r4 = run_set("fresh", n, "on", tmp, None, only_ids=where, promote=promote)
        c4, c3 = correct_map(r4), correct_map(f3)
        con_i = sum(1 for x in rep.values() for c in x["contradictions"] if c["premise"] == pk)
        fixed = sorted(i for i in where if c3.get(i) and not c4.get(i)); broke = sorted(i for i in where if c4.get(i) and not c3.get(i))
        item = dict(kind="lemma", id=lid, describe=CL.describe(e), premise=pk, fresh_hits=by_id[lid], puzzles=len(where), contradictions=con_i, fixed=fixed, new_misses=broke,
                    kept=(by_id[lid] >= 1 and con_i == 0 and not broke))
        lemma_items.append(item)
        log(f"OBS ablation [lemma] {item['describe']}: {by_id[lid]} fresh hits in {len(where)} puzzles, {con_i} contradictions; with it right where without it wrong: {fixed}; the reverse: {broke} -> {'KEPT' if item['kept'] else 'DROPPED'}")
    out = dict(dev={"correct": sum(dev_c.values()), "n": len(dev_c)}, fresh={"correct": sum(fresh_c.values()), "n": len(fresh_c)}, proposals=proposals, lemmas=lemma_items, wrong_answers=cases,
               kept=[p.get("gloss") or p.get("tag_bonus") for p in kept] + [x["describe"] for x in lemma_items if x["kept"]])
    json.dump(out, open(paths.derived("proposals.json"), "w"), indent=1, default=str)
    log(f"OBS proposer: {len(proposals) + len(lemma_items)} proposed, {len(out['kept'])} kept after the fresh check; {round(time.time() - t0)} s; saved {paths.derived('proposals.json')}")
    return out


if __name__ == "__main__":
    main()
