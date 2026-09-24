"""CHAINS (Kaveh 2026-09-23: "why are we averaging? ... follow up on the threat one, expand the gain one, expand the
drawing one ... a deterministic chain of computations ... stop when no more pieces come in ... each one starting
with a lemma ... find the frames where material is definitely lost -- those are the interesting ones").

An IDEA is a lemma embedded in the position whose verdict on a move carries a stake (the value its verdict implies,
in material units). Each idea seeds a CHAIN over the move: the bodies start as the lemma's own pieces; each step
brings in the most interacting piece (the guide or the relation heuristic), re-applies the library over the enlarged
set, and lets the MOST SPECIFIC lemma that speaks decide. The chain HOLDS while the claim's sign survives, is REFUTED
when a larger frame flips it (the refuting piece and lemma are the finding), and STOPS when a piece joins and no
new lemma speaks (it did not matter), or at the budget. No averaging anywhere.

    python -m catspace.incubator.analyst.chains --fen ...        # the position's chains
    python -m catspace.incubator.analyst.chains gauge --n 150    # conclusions vs the truth, by stake
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import chess

from .lemma_apply import Applier, load_library

OUT = Path("data/derived/lemmas")


def implied_value(c, fired) -> float | None:
    if not c.value:
        return None
    return c.value["exception"] if fired is not None else c.value["holds"]


def ideas(app: Applier, board: chess.Board, min_stake: float = 1.0) -> list:
    """(move, lemma, fired, ids, stake, sign): every embedded lemma whose verdict on a move implies |value| >= min_stake."""
    res = app.apply(board)
    out = []
    for (e, key), vs in res["verdicts"].items():
        mv = chess.Move.from_uci(key)
        for c, pred, fired, ids in vs:
            v = implied_value(c, fired)
            if v is None or abs(v) < min_stake:
                continue
            out.append(dict(move=mv, lemma=c, fired=fired, ids=list(ids), stake=abs(v), value=v))
    out.sort(key=lambda i: -i["stake"])
    return out


def interface(app: Applier, board: chess.Board, bodies: list, mv: chess.Move, lemma, ids: list, cand: list) -> dict:
    """THE INTERFACE (Kaveh 2026-09-23: "the guards should give you a hint: unless something interferes, unless
    something attacks the attacker -- those are the directions to expand"): for every literal of the deciding lemma's
    full guard, and its frame literals, the outside pieces whose removal flips the literal's truth on this board are
    the pieces the literal is about. Returns candidate -> number of literals it flips (weighted by nothing else;
    the policy breaks ties). Generic: literals are programs; removal is the rules."""
    dom = app.dom
    lits = list(lemma.frame.literals) + [l for t in lemma.guard for l in t]
    st = dom.state_of(board, list(ids))
    a = (st.board, mv)
    base = {}
    for r, sl, pol in lits:
        rel = app.rel[r]
        try:
            base[(r, sl)] = bool(rel.fn(st, a, *[ids[x] for x in sl])) if rel.on_action else bool(rel.fn(st, *[ids[x] for x in sl]))
        except Exception:
            base[(r, sl)] = None
    score = {}
    for c in cand:
        b2 = board.copy(stack=False); b2.remove_piece_at(c)
        if not b2.is_valid():
            continue
        st2 = dom.state_of(b2, list(ids)); a2 = (st2.board, mv)
        flips = 0
        for r, sl, pol in lits:
            if base[(r, sl)] is None:
                continue
            rel = app.rel[r]
            try:
                v = bool(rel.fn(st2, a2, *[ids[x] for x in sl])) if rel.on_action else bool(rel.fn(st2, *[ids[x] for x in sl]))
            except Exception:
                continue
            if v != base[(r, sl)]:
                flips += 1
        if flips:
            score[c] = flips
    return score


def expand(app: Applier, board: chess.Board, idea: dict, max_bodies: int = 8, ranker=None, mode: str = "interface", floor: float = 0.15) -> dict:
    """Follow one idea's chain as a DIALOGUE (Kaveh 2026-09-23): after a piece of one side joins, the other side
    answers next -- unless an interface piece of the deciding lemma is waiting (threat analysis, two pieces of one
    side allowed). Silence is not a stop: the chain continues while any candidate is still RELEVANT (an interface
    piece for the current lemma, or a piece the policy scores above `floor` x its top score at the seed), and ends
    on a refutation, when nothing relevant is left, or at the budget. Conclusion: held | refuted | unexpanded."""
    from .body_search import related
    mv = idea["move"]; actor = mv.from_square; our = board.piece_at(actor).color
    bodies = list(dict.fromkeys([actor] + [i for i in idea["ids"] if i != actor]))
    claim_value = idea["value"]; deciding = idea["lemma"]; deciding_ids = list(idea["ids"])
    trace = [dict(bodies=[chess.square_name(b) for b in bodies], value=claim_value, lemma=deciding.text[:110], event="seed", side="ours")]
    ents = dict(app.dom.entities(board))
    last_side = our; top_score = None
    while len(bodies) < max_bodies:
        cand = [e for e in ents if e not in bodies]
        if not cand:
            break
        if ranker is not None:
            try:
                ranked = ranker.rank(board, mv, cand, in_frame=set(bodies))
            except TypeError:
                ranked = ranker.rank(board, mv, cand)
        else:
            ranked = related(app, board, bodies, cand)
        if not ranked:
            break
        if top_score is None:
            top_score = max(abs(sc) for sc, _ in ranked) or 1.0
        score = {c: sc for sc, c in ranked}
        iface = interface(app, board, bodies, mv, deciding, deciding_ids, cand) if mode == "interface" else {}
        # relevance: an interface piece, or a policy score above the floor
        relevant = [c for c in cand if c in iface or abs(score.get(c, 0.0)) >= floor * top_score]
        if not relevant:
            trace.append(dict(bodies=[chess.square_name(b) for b in bodies], value=claim_value, lemma=None, event="exhausted", side=None))
            break
        other = [c for c in relevant if board.piece_at(c).color != last_side]
        pool = [c for c in relevant if c in iface] or other or relevant       # threat analysis first, then the other side's answer, then whatever is left
        nxt = max(pool, key=lambda c: (iface.get(c, 0), score.get(c, 0.0)))
        role = "threat" if nxt in iface else "answer"
        bodies.append(nxt); last_side = board.piece_at(nxt).color
        res = app.apply(board, bodies=set(bodies), actor=actor, only_action=mv)
        vs = [v for vl in res["verdicts"].values() for v in vl]
        spoke = [(c, fired, ids) for c, pred, fired, ids in vs if implied_value(c, fired) is not None and c.frame.size > deciding.frame.size]
        side = "ours" if last_side == our else "theirs"
        if not spoke:
            trace.append(dict(bodies=[chess.square_name(b) for b in bodies], value=claim_value, lemma=None, event="silent", side=side, role=role, by=chess.square_name(nxt)))
            continue
        c, fired, ids = max(spoke, key=lambda x: x[0].frame.size)
        v = implied_value(c, fired)
        event = "held" if (v > 0) == (claim_value > 0) and abs(v) >= 0.5 else "refuted"
        trace.append(dict(bodies=[chess.square_name(b) for b in bodies], value=v, lemma=c.text[:110], event=event, side=side, role=role, by=chess.square_name(nxt)))
        deciding = c; deciding_ids = list(ids)
        if event == "refuted":
            return dict(idea=idea, trace=trace, conclusion="refuted", final=v, bodies=len(bodies))
        claim_value = v
    return dict(idea=idea, trace=trace, conclusion="held" if any(t["event"] == "held" for t in trace) else "unexpanded", final=claim_value, bodies=len(bodies))


def chains(app: Applier, board: chess.Board, max_ideas: int = 12, max_bodies: int = 6, ranker=None, mode: str = "interface") -> list:
    out = []
    seen = set()
    for idea in ideas(app, board):
        key = (idea["move"].uci(), idea["value"] > 0)
        if key in seen:
            continue                                                          # one chain per (move, sign): the highest-stake idea seeds it
        seen.add(key)
        out.append(expand(app, board, idea, max_bodies=max_bodies, ranker=ranker, mode=mode))
        if len(out) >= max_ideas:
            break
    return out


def truth(dom, b: chess.Board, mv: chess.Move) -> float | None:
    from .chess_domain import State, resolved_reply_scalar, resolved_null_reply_scalar
    st = State(b.copy(stack=False), [mv.from_square])
    base = resolved_null_reply_scalar(st, mv.from_square)
    if base is None:
        return None
    after = dom.transition(st, (st.board, mv))
    return max(-10.0, min(10.0, resolved_reply_scalar(after, after.ids[0]) - base))


def gauge(n: int, library: Path, proved: Path, lo: int = 10, hi: int = 80, seed: int = 0, ranker=None, log=print, tag: str = "", mode: str = "interface"):
    """For every chain: does its conclusion agree with the truth's sign? Held chains should; refuted chains' seeds
    should not (the expansion caught a wrong small-world claim). By stake band and by number of bodies."""
    from .chess_domain import domain
    from .dataset import load as load_ds
    dom = domain(); app = Applier(dom, load_library(library, proved))
    meta = load_ds("data/derived/knowledge", meta_only=True)["meta"]
    rows = [m for m in meta if lo <= m["ply"] <= hi and m.get("best")]
    rng = random.Random(seed); rng.shuffle(rows)
    t0 = time.perf_counter(); n_pos = 0
    agree = defaultdict(Counter); by_bodies = defaultdict(Counter); seed_wrong = Counter(); best_seeded = Counter()
    for m_ in rows:
        if n_pos >= n:
            break
        b = chess.Board(m_["fen"])
        if b.is_game_over():
            continue
        n_pos += 1
        best = chess.Move.from_uci(m_["best"]) if m_["best"] in {x.uci() for x in b.legal_moves} else None
        for ch in chains(app, b, ranker=ranker, mode=mode):
            mv = ch["idea"]["move"]; t = truth(dom, b, mv)
            if t is None:
                continue
            band = "big" if ch["idea"]["stake"] >= 3 else "small"
            ok = (ch["final"] > 0) == (t > 0) if abs(t) >= 0.5 else (abs(ch["final"]) < 1.0)
            agree[(ch["conclusion"], band)][ok] += 1
            by_bodies[ch["bodies"]][ok] += 1
            if ch["conclusion"] == "refuted":
                seed_ok = (ch["idea"]["value"] > 0) == (t > 0) if abs(t) >= 0.5 else False
                seed_wrong[not seed_ok] += 1
            if best is not None and mv == best:
                best_seeded[ch["conclusion"]] += 1
        if n_pos % 25 == 0:
            log(f"[chains] {n_pos}/{n} ({time.perf_counter() - t0:.0f}s)")
    log(f"[chains] {n_pos} positions, {time.perf_counter() - t0:.0f}s, library {len(app.lemmas)}")
    for (concl, band), c in sorted(agree.items()):
        tot = sum(c.values())
        log(f"  {concl:10s} {band:5s} stakes: conclusion agrees with the truth {c[True]}/{tot} = {100 * c[True] / max(1, tot):.0f}%")
    log("  by number of bodies at the end: " + ", ".join(f"{k}: {c[True]}/{sum(c.values())}" for k, c in sorted(by_bodies.items())))
    log(f"  refuted chains: the seed was wrong in {seed_wrong[True]}/{sum(seed_wrong.values())} (expansion catching a wrong small-world claim)")
    log(f"  the engine's best move seeded a chain that: {dict(best_seeded)}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(positions=n_pos, agree={f"{k[0]}|{k[1]}": dict(v) for k, v in agree.items()}, by_bodies={k: dict(v) for k, v in by_bodies.items()}, seed_wrong=dict(seed_wrong), best_seeded=dict(best_seeded)), open(OUT / f"gauge_chains{tag}.json", "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="show", choices=["show", "gauge"]); ap.add_argument("--fen"); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--library", default=str(OUT / "library_games_frame.json")); ap.add_argument("--proved", default=str(OUT / "proved_games_frame.json")); ap.add_argument("--guide", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--mode", default="interface", choices=["interface", "policy"])
    a = ap.parse_args(argv)
    from .chess_domain import domain
    ranker = None
    if a.guide == "tf":
        from .body_guide_tf import GraphRanker
        ranker = GraphRanker(domain())
    if a.cmd == "gauge":
        gauge(a.n, Path(a.library), Path(a.proved), ranker=ranker, tag=a.tag, mode=a.mode); return
    app = Applier(domain(), load_library(Path(a.library), Path(a.proved)))
    b = chess.Board(a.fen or chess.STARTING_FEN)
    for ch in chains(app, b, ranker=ranker, mode=a.mode):
        i = ch["idea"]
        print(f"{b.san(i['move']):7s} stake {i['stake']:.1f} ({'gain' if i['value'] > 0 else 'loss'}) -> {ch['conclusion']} at {ch['bodies']} bodies, final {ch['final']:+.1f}")
        for t in ch["trace"]:
            who = f"{t.get('side') or '':6s} {t.get('role') or '':6s}"
            print(f"    {t['event']:9s} {who} bodies {','.join(t['bodies'])}  value {t['value']:+.1f}  {t.get('lemma') or ''}" + (f"  by {t['by']}" if t.get('by') else ""))


if __name__ == "__main__":
    main()
