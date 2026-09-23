"""THE SEARCH OVER BODIES (Kaveh 2026-09-23: "let's do a search over bodies in each position, keep track of which
lemmas we use, see what keeps repeating").

For one move: start from the moving piece alone; at each level bring in one more piece, the one most related to the
current bodies by the domain's binary relations; apply every proved lemma whose frame embeds in the current bodies
with this actor and this move; the level's verdict is the vote. Record the trace: the verdict per level, the FLIPS
(a level whose verdict differs from the previous one -- the new body resolved or overturned the smaller frame's
reading), and the lemmas used (those whose guard fired, or whose claim carried the vote). Generic: the domain's
entities, relations and the library; no chess.

    python -m catspace.incubator.analyst.body_search --n 300          # over the corpus, best move + a random move
    python -m catspace.incubator.analyst.body_search --fen ... --move e2e4
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from .lemma_apply import Applier, load_library

OUT = Path("data/derived/lemmas")


def related(app: Applier, raw, bodies: list, candidates: list, after=None) -> list:
    """Candidates ranked by the binary relations (either direction) tying them to the current bodies, BEFORE the
    action and AFTER it (the piece that attacks the landing square is related only after), each relation weighted by
    its rarity in this position (a relation true of most pairs, e.g. being on the same side, says little): generic."""
    dom = app.dom
    rels = [r for r in dom.relations if not r.on_action and r.arity == 2]
    ents = [e for e, _ in dom.entities(raw)]
    states = [raw] + ([after] if after is not None else [])
    # rarity per relation over all ordered pairs in the position (before)
    freq = Counter(); pairs = 0
    for i in ents:
        for j in ents:
            if i == j:
                continue
            pairs += 1
            st = dom.state_of(raw, [i, j])
            for r in rels:
                try:
                    if r.fn(st, i, j):
                        freq[r.name] += 1
                except Exception:
                    pass
    weight = {r.name: 1.0 - freq[r.name] / max(1, pairs) for r in rels}
    scored = []
    for c in candidates:
        w = 0.0
        for b in bodies:
            for s_ in states:
                if s_ is not raw and (s_.piece_at(b) is None or s_.piece_at(c) is None):
                    continue
                st = dom.state_of(s_, [b, c])
                for r in rels:
                    try:
                        w += weight[r.name] * (bool(r.fn(st, b, c)) or bool(r.fn(st, c, b)))      # a symmetric relation counts once
                    except Exception:
                        pass
        scored.append((w, c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored


def body_search(app: Applier, raw, actor, action, max_bodies: int = 5, ranker=None, margin: float | None = None) -> dict:
    """ranker: optional object with rank(board, move, candidates) -> [(score, id)] (the neural guide); default = the
    rarity-weighted relation heuristic. margin: when given, growth stops once the composed prediction moved by less
    than the margin on the last level (Kaveh 2026-09-23: prune like alpha-beta, by what matters right now)."""
    from .lemma_apply import compose
    ents = dict(app.dom.entities(raw))
    after = None
    try:
        st0 = app.dom.state_of(raw, [actor])
        after = app.dom.transition(st0, (st0.board, action)).board
        if after.piece_at(action.to_square) is not None:
            # the mover's identity moved: relate candidates to its landing square in the after-state via a remap
            pass
    except Exception:
        after = None
    bodies = [actor]; trace = []; prev = None
    while True:
        res = app.apply(raw, bodies=set(bodies), actor=actor, only_action=action)
        vs = [v for (e, k), vl in res["verdicts"].items() for v in vl]
        keeps = sum(1 for _, p, _, _ in vs if p == "keeps"); loses = len(vs) - keeps
        verdict = None if not vs else ("keeps" if keeps > loses else "loses" if loses > keeps else "split")
        used = [(c.idx, c.text, p, fired) for c, p, fired, ids in vs]
        composed = compose(vs)
        level = dict(bodies=list(bodies), n_lemmas=len(vs), keeps=keeps, loses=loses, verdict=verdict, flip=(prev is not None and verdict is not None and verdict != prev), used=used, composed=composed)
        trace.append(level)
        if verdict is not None:
            prev = verdict
        if len(bodies) >= max_bodies:
            break
        if margin is not None and len(trace) >= 2 and trace[-1]["composed"] is not None and trace[-2]["composed"] is not None and abs(trace[-1]["composed"] - trace[-2]["composed"]) < margin:
            break                                                             # the last body changed the composed world by less than what matters now
        cand = [e for e in ents if e not in bodies]
        # in the after-state the actor sits on the landing square: relate through that square
        after_bodies = [action.to_square if b == actor else b for b in bodies]
        if ranker is not None:
            try:
                ranked = ranker.rank(raw, action, cand, in_frame=set(bodies)) if cand else []     # the graph guide conditions on the frame so far
            except TypeError:
                ranked = ranker.rank(raw, action, cand) if cand else []
        else:
            ranked = related(app, raw, bodies, cand)
            if after is not None:
                r2 = dict((c, w) for w, c in related(app, after, after_bodies, [c for c in cand if after.piece_at(c) is not None and c != action.to_square]))
                ranked = sorted(((w + r2.get(c, 0.0), c) for w, c in ranked), key=lambda x: (-x[0], x[1]))
        if not ranked or (ranker is None and ranked[0][0] == 0):
            break                                                             # nothing else touches the frame
        bodies.append(ranked[0][1])
    return dict(actor=actor, action=str(action), trace=trace, settled=max((i for i, l in enumerate(trace) if l["verdict"] is not None and (i == 0 or l["flip"])), default=0) + 1,
                final=next((l["verdict"] for l in reversed(trace) if l["verdict"] is not None), None))


def run_corpus(n: int, lo: int, hi: int, library: Path, proved: Path, max_bodies: int, seed: int = 0, log=print, ranker=None, tag: str = ""):
    import chess
    from .chess_domain import domain
    from .dataset import load as load_ds
    dom = domain(); app = Applier(dom, load_library(library, proved))
    meta = load_ds("data/derived/knowledge", meta_only=True)["meta"]
    rows = [m for m in meta if lo <= m["ply"] <= hi and m.get("best")]
    rng = random.Random(seed); rng.shuffle(rows)
    t0 = time.perf_counter()
    stats = {k: dict(settled=Counter(), flips=0, n=0, final=Counter(), surprise_by_level=Counter(), spoken_by_level=Counter(), no_lemma=0) for k in ("best", "random")}
    usage = Counter(); usage_fired = Counter(); flip_lemmas = Counter(); traces = []
    n_pos = 0
    for m_ in rows:
        if n_pos >= n:
            break
        b = chess.Board(m_["fen"])
        if b.is_game_over():
            continue
        try:
            best = chess.Move.from_uci(m_["best"])
        except Exception:
            continue
        if best not in b.legal_moves:
            continue
        n_pos += 1
        legal = list(b.legal_moves)
        for kind, mv in (("best", best), ("random", rng.choice(legal))):
            r = body_search(app, b, mv.from_square, mv, max_bodies=max_bodies, ranker=ranker)
            st = stats[kind]; st["n"] += 1
            if r["final"] is None:
                st["no_lemma"] += 1; continue
            st["settled"][r["settled"]] += 1; st["final"][r["final"]] += 1
            st["flips"] += sum(1 for l in r["trace"] if l["flip"])
            for i, l in enumerate(r["trace"]):
                if l["verdict"] is not None:
                    st["spoken_by_level"][i + 1] += 1
                    if l["verdict"] == "loses":
                        st["surprise_by_level"][i + 1] += 1
                for idx, text, p, fired in l["used"]:
                    usage[text] += 1
                    if fired:
                        usage_fired[text] += 1
                if l["flip"]:
                    for idx, text, p, fired in l["used"]:
                        if p == l["verdict"]:
                            flip_lemmas[text] += 1
            if kind == "best" and len(traces) < 200:
                traces.append(dict(fen=b.fen(), move=mv.uci(), settled=r["settled"], final=r["final"], levels=[dict(bodies=[chess.square_name(x) for x in l["bodies"]], verdict=l["verdict"], keeps=l["keeps"], loses=l["loses"], flip=l["flip"], used=[(t[:120], p, f) for _, t, p, f in l["used"][:6]]) for l in r["trace"]]))
        if n_pos % 50 == 0:
            log(f"[body_search] {n_pos}/{n} ({time.perf_counter() - t0:.0f}s)")
    log(f"[body_search] {n_pos} positions, max {max_bodies} bodies, {len(app.lemmas)} proved lemmas, {time.perf_counter() - t0:.0f}s")
    for kind in ("best", "random"):
        st = stats[kind]; N = max(1, st["n"] - st["no_lemma"])
        log(f"  {kind:6s} move: no lemma at any level {st['no_lemma']}/{st['n']}; final verdict keeps {st['final']['keeps']} / loses {st['final']['loses']} / split {st['final']['split']} (loses = {100 * st['final']['loses'] / N:.1f}%)")
        log(f"         bodies needed to settle: " + ", ".join(f"{k}: {v} ({100 * v / N:.0f}%)" for k, v in sorted(st["settled"].items())) + f"; flips per move {st['flips'] / N:.2f}")
        log(f"         called LOSING by level: " + ", ".join(f"L{k}: {st['surprise_by_level'][k]}/{st['spoken_by_level'][k]} ({100 * st['surprise_by_level'][k] / max(1, st['spoken_by_level'][k]):.0f}%)" for k in sorted(st["spoken_by_level"])))
    log("  lemmas used most (any level, best + random):")
    for t, k in usage.most_common(10):
        log(f"    {k:5d} ({usage_fired[t]:4d} fired)  {t[:150]}")
    log("  lemmas that carried a FLIP (the new body changed the verdict):")
    for t, k in flip_lemmas.most_common(10):
        log(f"    {k:5d}  {t[:150]}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(stats={k: dict(n=v["n"], no_lemma=v["no_lemma"], settled=dict(v["settled"]), final=dict(v["final"]), flips=v["flips"], surprise_by_level=dict(v["surprise_by_level"]), spoken_by_level=dict(v["spoken_by_level"])) for k, v in stats.items()},
                   usage=usage.most_common(60), usage_fired=usage_fired.most_common(60), flip_lemmas=flip_lemmas.most_common(60), traces=traces), open(OUT / f"body_search{tag}.json", "w"), indent=1)
    log(f"  -> {OUT / 'body_search.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--lo", type=int, default=10); ap.add_argument("--hi", type=int, default=80); ap.add_argument("--max-bodies", type=int, default=5)
    ap.add_argument("--fen"); ap.add_argument("--move"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--guide", default="", help="'' | 'mlp' | 'tf'"); ap.add_argument("--tag", default="")
    ap.add_argument("--library", default=str(OUT / "library_v2_size3.json")); ap.add_argument("--proved", default=str(OUT / "proved_v2_size3.json"))
    a = ap.parse_args(argv)
    if a.fen:
        import chess
        from .chess_domain import domain
        app = Applier(domain(), load_library(Path(a.library), Path(a.proved)))
        b = chess.Board(a.fen); mv = chess.Move.from_uci(a.move)
        r = body_search(app, b, mv.from_square, mv, max_bodies=a.max_bodies)
        print(f"{mv.uci()}: settled at {r['settled']} bodies, final {r['final']}")
        for i, l in enumerate(r["trace"]):
            print(f"  L{i + 1} bodies {[chess.square_name(x) for x in l['bodies']]}: {l['keeps']} keeps / {l['loses']} loses -> {l['verdict']}{'  FLIP' if l['flip'] else ''}")
            for idx, text, p, fired in l["used"][:5]:
                print(f"      {p:5s} {text[:140]}" + (f"  <- {fired}" if fired else ""))
        return
    ranker = None
    if a.guide == "tf":
        from .body_guide_tf import GraphRanker
        from .chess_domain import domain
        ranker = GraphRanker(domain())
    elif a.guide:
        from .body_guide import GuideRanker
        from .chess_domain import domain
        ranker = GuideRanker(domain())
    run_corpus(a.n, a.lo, a.hi, Path(a.library), Path(a.proved), a.max_bodies, seed=a.seed, ranker=ranker, tag=a.tag)


if __name__ == "__main__":
    main()
