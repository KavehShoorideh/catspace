"""IMPORTANT SQUARES (Kaveh 2026-09-24): as the lemmas fire on a position, note each fired guard term and the squares
it is about. ACTIVE squares are where terms fire now (the tactics of the moment); NEAR squares are where a term is
one literal short of firing (the tactics that almost flow, too many to calculate, so they are counted). A square's
heat is the stake of the lemmas whose terms name it, and every square keeps the lemmas that made it hot. CONTROL
of a move = how it shifts the attack balance on the near squares: a move that wins nothing now but keeps the hot
squares is what the heat is for. Generic: squares are whatever the domain's literals name (slot entities and the
action's landing point); nothing here knows a file from a rank."""
from __future__ import annotations

from collections import defaultdict

import chess

from .lemma_apply import Applier, lit_text


def term_squares(term, ids, action, actor) -> set:
    """The squares a guard term is about: the OTHER entities its literals name (the actor is the agent, not the
    place of the tactic), and the move's destination for action words."""
    out = set()
    for r, sl, pol in term:
        for s in sl:
            if s < len(ids) and ids[s] is not None and ids[s] != actor:
                out.add(ids[s])
        if r.startswith(("lands_", "after_", "then_", "captures", "gives_check", "stays_")) or r.endswith(("lands_attacked", "captures")):
            out.add(action[1].to_square)
    return out


def ledger(app: Applier, board: chess.Board, min_stake: float = 1.0, max_embeddings: int = 4000) -> dict:
    """-> {'active': {square: heat}, 'near': {square: heat}, 'why': {square: [(lemma id, stake, term text, kind)]}}."""
    dom = app.dom
    ents = dom.entities(board)
    by_type = defaultdict(list)
    for e, t in ents:
        by_type[t].append(e)
    active = defaultdict(float); near = defaultdict(float); why = defaultdict(list)
    n_emb = 0
    import itertools
    cache: dict = {}
    for types, lemmas in app.by_types.items():
        pools = [[e for lt in dom.leaves(t) for e in by_type.get(lt, [])] for t in types]
        if any(not p for p in pools):
            continue
        for ids in itertools.product(*pools):
            if len(set(ids)) < len(ids):
                continue
            state = dom.state_of(board, list(ids))
            for c in lemmas:
                stake = abs((c.value or {}).get("exception", 0.0))
                if stake < min_stake or not c.guard:
                    continue
                e = ids[c.slot]
                if not all(app._lit(state, ids, r, sl, pol, cache=cache) for r, sl, pol in c.frame.literals):
                    continue
                n_emb += 1
                if n_emb > max_embeddings:
                    break
                best_here = {}                                                # one embedding contributes at most its stake per square, whatever the number of actions
                for a in dom.actions(state, e):
                    for term in c.guard:
                        vals = [app._lit(state, ids, r, sl, pol, action=a, cache=cache) for r, sl, pol in term]
                        missing = vals.count(False)
                        if missing > 1 or (missing == 1 and len(term) < 2):
                            continue                                          # a one-literal term that is false is not "almost": it is simply off
                        kind = "active" if missing == 0 else "near"
                        others = {ids[s] for r, sl, pol in term for s in sl if s < len(ids) and ids[s] is not None and ids[s] != e}
                        if not others:
                            continue                                          # a term about the actor alone (its own safety) is not a tactic between pieces
                        for sq in term_squares(term, ids, a, e):
                            key = (kind, sq)
                            if stake > best_here.get(key, (0, None))[0]:
                                best_here[key] = (stake, (c.id, round(stake, 1), " & ".join(lit_text(r, sl, pol) for r, sl, pol in term), kind, a[1].uci()))
                for (kind, sq), (w, why_row) in best_here.items():
                    (active if kind == "active" else near)[sq] += w
                    if len(why[sq]) < 12:
                        why[sq].append(why_row)
    return dict(active=dict(active), near=dict(near), why=dict(why), embeddings=n_emb)


def control(board: chess.Board, move: chess.Move, near: dict) -> float:
    """How the move shifts the attack balance on the near squares, weighted by heat: our attackers minus theirs after
    the move, minus the same before. Positive: the move keeps or takes the hot squares."""
    us = board.turn
    def balance(b):
        tot = 0.0
        for sq, w in near.items():
            tot += w * (len(b.attackers(us, sq)) - len(b.attackers(not us, sq)))
        return tot
    before = balance(board)
    b2 = board.copy(stack=False); b2.push(move)
    return balance(b2) - before


def report(app: Applier, board: chess.Board, top: int = 8, log=print) -> dict:
    L = ledger(app, board)
    log(f"[squares] {L['embeddings']} embeddings; active squares {len(L['active'])}, near squares {len(L['near'])}")
    for kind in ("active", "near"):
        hot = sorted(L[kind].items(), key=lambda x: -x[1])[:top]
        log(f"  {kind}: " + ", ".join(f"{chess.square_name(s)} {h:.0f}" for s, h in hot))
        for s, h in hot[:3]:
            for lid, st, term, k, uci in [w for w in L["why"].get(s, []) if w[3] == kind][:2]:
                log(f"      {chess.square_name(s)}: {lid} stake {st} via {uci}: {term}")
    moves = sorted(((control(board, m, L["near"]), m) for m in board.legal_moves), key=lambda x: -x[0])[:top]
    log("  control of the near squares by move: " + ", ".join(f"{board.san(m)} {c:+.0f}" for c, m in moves))
    L["control"] = moves
    return L


def main(argv=None):
    import argparse
    from pathlib import Path
    from .chains import domain_of
    from .lemma_apply import load_library
    from .lemma_gen import OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--fen", required=True); ap.add_argument("--library", default=str(OUT / "library_games_frame6k.json")); ap.add_argument("--proved", default=str(OUT / "proved_games_frame6k.json"))
    a = ap.parse_args(argv)
    app = Applier(domain_of(Path(a.library)), load_library(Path(a.library), Path(a.proved)))
    report(app, chess.Board(a.fen))


if __name__ == "__main__":
    main()
