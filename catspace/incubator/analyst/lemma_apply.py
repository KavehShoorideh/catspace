"""THE APPLIER: every embedding of every proved frame in a REAL state, and the verdict of each lemma on each action
of its actor. Domain-agnostic: the domain supplies entities(raw) and state_of(raw, ids); the frames, literals and
guards are the library's. This is the function that did not exist when Kaveh asked "how do we make sure in every
position we see all the lemmas that apply?" (2026-09-23).

A lemma's frame was sampled with ONLY its entities (plus what the domain requires) present; applying it to a full
state is an extrapolation. The applier does not hide that: it reports the verdicts, and gauge_lemmas measures how
often they hold and how often they contradict the oracle (the SURPRISE).

    python -m catspace.incubator.analyst.lemma_apply --fen "..."
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .lemma_gen import Domain, Frame, guard_from_json, guard_text, lemma_id, lit_text

OUT = Path("data/derived/lemmas")


@dataclass
class Compiled:
    idx: int
    frame: Frame
    slot: int
    claim: str
    guard: list                    # full guard terms (inherited + own)
    text: str
    value: dict = None             # {"holds": mean outcome where the claim held, "exception": mean outcome of the exceptions}, inside the frame's world

    @property
    def id(self) -> str:
        return self.text.split(" ", 1)[0]


def load_library(library_path: Path, proved_path: Path | None = None, proved_only: bool = True) -> list[Compiled]:
    lib = json.load(open(library_path))
    status = None
    if proved_path and Path(proved_path).exists():
        status = [p.get("status") for p in json.load(open(proved_path))]
    out = []
    for i, e in enumerate(lib):
        if proved_only and status is not None and status[i] not in ("PROVED", "THEOREM"):
            continue
        fr = Frame(tuple(e["types"]), tuple((r, tuple(sl), pol) for r, sl, pol in e["literals"]))
        guard = guard_from_json(e.get("inherited")) + guard_from_json(e.get("guard"))
        lid = e.get("id") or lemma_id(fr.types, fr.literals, e["slot"], e["claim"], guard)
        out.append(Compiled(i, fr, e["slot"], e["claim"], guard, f"{lid} {fr.text()} :: slot {e['slot']} {e['claim']} unless {guard_text(guard) or '(nothing)'}", e.get("value")))
    return out


class Applier:
    def __init__(self, dom: Domain, lemmas: list[Compiled]):
        names = {r for c in lemmas for t in c.guard for r, _, _ in t} | {r for c in lemmas for r, _, _ in c.frame.literals}
        if any(n.startswith("after_") for n in names) and not any(r.name.startswith("after_") for r in dom.relations):
            from .lemma_gen import lift_after
            dom = Domain(**{**dom.__dict__, "relations": list(dom.relations) + lift_after(dom)})   # the library was made with the lifted vocabulary
        self.dom = dom; self.lemmas = lemmas
        self.rel = {r.name: r for r in dom.relations}
        self.by_types: dict = defaultdict(list)
        self.lemmas_by_idx: dict = {c.idx: c for c in lemmas}
        for c in lemmas:
            self.by_types[c.frame.types].append(c)

    def _lit(self, state, ids, r, sl, pol, action=None, cache=None) -> bool:
        key = (r, tuple(ids[s] for s in sl), None if action is None else action[1])
        if cache is not None and key in cache:
            return cache[key] == pol
        rel = self.rel[r]
        try:
            v = bool(rel.fn(state, action, *[ids[s] for s in sl])) if rel.on_action else bool(rel.fn(state, *[ids[s] for s in sl]))
        except Exception:
            v = False
        if cache is not None:
            cache[key] = v
        return v == pol

    def apply(self, raw, bodies=None, actor=None, only_action=None) -> dict:
        """-> {(actor id, action key): [(lemma, predicted 'keeps'|'loses', fired term text or None, ids)]} plus the
        list of embeddings; an action key is str(action[1]) (the move). bodies: restrict the frames' slots to these
        entity ids (the body search); actor / only_action: only that actor's / that action's verdicts."""
        ents = self.dom.entities(raw)
        if bodies is not None:
            ents = [(e, t) for e, t in ents if e in bodies]
        by_type: dict = defaultdict(list)
        for e, t in ents:
            by_type[t].append(e)
        verdicts: dict = defaultdict(list); embeddings = []
        cache: dict = {}
        for types, lemmas in self.by_types.items():
            pools = [by_type.get(t, []) for t in types]
            if any(not p for p in pools):
                continue
            for ids in itertools.product(*pools):
                if len(set(ids)) < len(ids):
                    continue
                state = self.dom.state_of(raw, list(ids))
                for c in lemmas:
                    e = ids[c.slot]
                    if actor is not None and e != actor:
                        continue
                    if self.dom.actor_ok is not None and not self.dom.actor_ok(raw, e):
                        continue
                    if not all(self._lit(state, ids, r, sl, pol, cache=cache) for r, sl, pol in c.frame.literals):
                        continue
                    embeddings.append((c, ids))
                    for a in self.dom.actions(state, e):
                        if only_action is not None and str(a[1]) != str(only_action):
                            continue
                        fired = None
                        for term in c.guard:
                            if all(self._lit(state, ids, r, sl, pol, action=a, cache=cache) for r, sl, pol in term):
                                fired = term; break
                        base = "keeps" if c.claim == "acting keeps" else "loses"
                        pred = base if fired is None else ("loses" if base == "keeps" else "keeps")
                        verdicts[(e, str(a[1]))].append((c, pred, None if fired is None else " & ".join(lit_text(r, sl, pol) for r, sl, pol in fired), ids))
        return dict(verdicts=verdicts, embeddings=embeddings)


def main(argv=None):
    import chess
    from .chess_domain import domain
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True); ap.add_argument("--duties", action="store_true"); ap.add_argument("--library", default=str(OUT / "library_v2_size3.json")); ap.add_argument("--proved", default=str(OUT / "proved_v2_size3.json"))
    a = ap.parse_args(argv)
    dom = domain(); app = Applier(dom, load_library(Path(a.library), Path(a.proved)))
    b = chess.Board(a.fen)
    res = app.apply(b)
    print(f"{len(app.lemmas)} proved lemmas; {len(res['embeddings'])} embeddings in this position; verdicts on {len(res['verdicts'])} moves")
    if a.duties:
        print_duties(b, duties(app, b, res)); return
    for (e, mv), vs in sorted(res["verdicts"].items(), key=lambda kv: (-len(kv[1]), kv[0][1])):
        keeps = sum(1 for _, p, _, _ in vs if p == "keeps"); loses = len(vs) - keeps
        print(f"\n{chess.square_name(e)} {mv}: {keeps} keeps / {loses} loses")
        for c, p, fired, ids in vs[:6]:
            print(f"   {p:5s} {c.text[:150]}" + (f"  <- {fired}" if fired else "") + f"  [{' '.join(chess.square_name(i) for i in ids)}]")



COMPOSE_RULE = "mean"                                                        # "mean" | "specific" (the largest frame's lemma decides) | "sum" (one term per distinct frame)


def compose(verdicts: list) -> float | None:
    """The COMPOSED preference of a move from the lemmas that speak about it (Kaveh 2026-09-23: "lemmas from one
    frame usable in other frames ... prefer this by this much"): each lemma, a self-contained world, predicts the
    outcome its verdict implies (the exception mean when its guard fired, the holds mean otherwise); the move's
    predicted outcome is the mean of those predictions. None when no lemma carries a value."""
    preds = []
    for c, pred, fired, ids in verdicts:
        if not c.value:
            continue
        preds.append((c.frame.size, c.frame.text(), c.value["exception"] if fired is not None else c.value["holds"]))
    if not preds:
        return None
    if COMPOSE_RULE == "specific":
        top = max(p[0] for p in preds)
        vals = [v for k, _, v in preds if k == top]
        return sum(vals) / len(vals)
    if COMPOSE_RULE == "sum":
        by_frame = {}
        for k, t, v in preds:
            by_frame.setdefault(t, []).append(v)
        return sum(sum(v) / len(v) for v in by_frame.values())
    return sum(v for _, _, v in preds) / len(preds)


# ---------------------------------------------------------------- duties and opportunities (Kaveh 2026-09-23)
def duties(app: Applier, raw, res: dict | None = None) -> dict:
    """Per piece: ALL its duties, ranked by stake, and its opportunities.
    A DUTY of piece P = a guard term that fires on P's own moves in some lemma whose claim is 'keeps' (moving P away
    loses): the term names what P is holding (the other slots in the term) and the stake is the lemma's example loss
    magnitude on this board (the 1-ply outcome of the guarded move). An OPPORTUNITY of P = one of its moves called
    losing ONLY through duty terms (every lemma that calls it losing does so by a term naming another piece), with
    the outcome it would have if those terms were false: the tactic waiting on the duty. Hand-off: a move by another
    piece that makes the fired term false lifts the duty; only the lemmas that fired on that term need re-evaluation.
    Generic: terms, slots and the domain's outcome; nothing here knows what a queen or a mate is."""
    dom = app.dom
    res = res or app.apply(raw)
    by_piece: dict = defaultdict(lambda: dict(duties=defaultdict(lambda: dict(stake=0.0, lemmas=set(), moves=set())), opportunities=[]))
    def outcome(e, mv):
        st = dom.state_of(raw, [e])
        after = dom.transition(st, (st.board, mv))
        return dom.reply_scalar(after, after.ids[0]) - dom.scalar(st, e)
    import chess
    for (e, key), vs in res["verdicts"].items():
        mv = chess.Move.from_uci(key)
        d = outcome(e, mv)
        losing = [(c, fired, ids) for c, p, fired, ids in vs if p == "loses"]
        if not losing:
            continue
        only_duty = True
        for c, fired, ids in losing:
            if c.claim != "acting keeps" or fired is None:
                only_duty = False; continue
            # the other pieces named by the fired term (slot indices -> entity ids)
            named = set()
            for term in c.guard:
                if " & ".join(lit_text(r, sl, pol) for r, sl, pol in term) == fired:
                    for r, sl, pol in term:
                        for s_ in sl:
                            if ids[s_] != e:
                                named.add(ids[s_])
            if not named:
                only_duty = False; continue                                   # a term about the move itself (lands attacked): not a duty
            key_ = tuple(sorted(named))
            du = by_piece[e]["duties"][key_]
            du["stake"] = max(du["stake"], -d); du["lemmas"].add(c.text); du["moves"].add(key)
        if only_duty:
            by_piece[e]["opportunities"].append(dict(move=key, outcome_now=d, held_by=[sorted(set(x for c, f, ids in losing for term in c.guard for r, sl, pol in term for x in (ids[s_] for s_ in sl) if x != e))]))
    out = {}
    for e, rec in by_piece.items():
        ds = sorted(((v["stake"], k, sorted(v["lemmas"])[:3], sorted(v["moves"])) for k, v in rec["duties"].items()), key=lambda x: -x[0])
        out[e] = dict(duties=[dict(stake=s, holds=[chess.square_name(x) for x in k], lemmas=l, moves=m) for s, k, l, m in ds],
                      opportunities=[dict(move=o["move"], outcome_now=o["outcome_now"], held_by=[chess.square_name(x) for x in o["held_by"][0]]) for o in rec["opportunities"]])
    return out


def print_duties(raw, table: dict):
    import chess
    for e, rec in sorted(table.items(), key=lambda kv: -(kv[1]["duties"][0]["stake"] if kv[1]["duties"] else 0)):
        p = raw.piece_at(e)
        if not rec["duties"]:
            continue
        print(f"{p.symbol()}{chess.square_name(e)}: duties " + "; ".join(f"holds {','.join(d['holds'])} (stake {d['stake']:+.0f}, {len(d['moves'])} moves guarded)" for d in rec["duties"][:4]))
        for o in rec["opportunities"][:3]:
            print(f"     opportunity {o['move']}: now {o['outcome_now']:+.0f}, held by {','.join(o['held_by'])}")


if __name__ == "__main__":
    main()
