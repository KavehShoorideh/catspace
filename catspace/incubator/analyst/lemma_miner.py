"""THE LEMMA MINER (Kaveh 2026-10-10: stalemate lemmas -- "if the opponent has a piece that can move they cannot be
stalemated", "give check with every move and you never stalemate" -- "how can I find lemmas like these automatically?").

For a QUESTION (a target in lemma_vocab.TARGETS, e.g. stalemate) the miner finds every MINIMAL conjunction of
vocabulary concepts (lemma_vocab.VOCAB) that implies the target -- or its negation -- with ZERO exceptions:
  data       contexts = (a board, a legal move, the board after it): census positions of at most MAX_PIECES pieces
             and generated endgames (queen / rook / pawn against a bare or nearly bare king -- where stalemates live),
             every legal move of the side to move; facts computed once per context (all cores)
  mining     premises of up to MAX_PREMISE concepts, support >= MIN_SUPPORT, no exception, no subset already enough
  holdout    the same on an independently generated set; a lemma with a holdout exception is a sampling accident
  proof      lemmas whose premise forces the target by its DEFINITION are tied to their Lean theorem
             (Ideas/StalemateLemmas.lean); the others are "exhaustive on the data"
  open cases contexts no lemma decides -> the proposer (lemma_review.py: the review between Kaveh and Claude) writes
             them out; the concepts we agree on come back as vocabulary and the next run mines with them
  registry   accepted lemmas -> data/derived/lemma_registry.json (premise, conclusion, support, holdout, proof, the
             vocabulary they use)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from collections import Counter

import chess

from catspace.io import paths

MAX_PIECES, MAX_PREMISE, MIN_SUPPORT = 12, 4, 20
MATERIALS = ["KQ", "KR", "KP", "KQ/P", "KR/P", "KQ/R", "KP/P", "KQ/N", "KQ/B", "KR/N", "KPP/P", "KQ/PP"]
LEAN = {
    ("defender in check", "not stalemate"): "Ideas.StalemateLemmas.check_not_stalemate",
    ("the move gave check", "not stalemate"): "Ideas.StalemateLemmas.checking_move_never_stalemates",
    ("defender has a free pawn", "not stalemate"): "Ideas.StalemateLemmas.legal_move_not_stalemate",
    ("defender has a free piece", "not stalemate"): "Ideas.StalemateLemmas.legal_move_not_stalemate",
    ("defender's king has a free square", "not stalemate"): "Ideas.StalemateLemmas.legal_move_not_stalemate",
    ("not: defender in check", "not checkmate"): "Ideas.StalemateLemmas.not_check_not_checkmate",
    ("defender has a free pawn", "not checkmate"): "Ideas.StalemateLemmas.legal_move_not_checkmate",
    ("defender has a free piece", "not checkmate"): "Ideas.StalemateLemmas.legal_move_not_checkmate",
    ("defender's king has a free square", "not checkmate"): "Ideas.StalemateLemmas.legal_move_not_checkmate",
}


def random_endgame(rng: random.Random, spec: str) -> chess.Board | None:
    """A legal position with the stronger side to move: spec 'KQ/P' = white K+Q against black K+P (colours randomized)."""
    strong, _, weak = spec.partition("/")
    for _ in range(200):
        b = chess.Board(None)
        sq = rng.sample(range(64), 2 + len(strong) - 1 + len(weak))
        white = rng.random() < 0.5
        sc, wc = (chess.WHITE, chess.BLACK) if white else (chess.BLACK, chess.WHITE)
        b.set_piece_at(sq[0], chess.Piece(chess.KING, sc)); b.set_piece_at(sq[1], chess.Piece(chess.KING, wc))
        rest = list(strong[1:]) + list(weak)
        ok = True
        for i, ch in enumerate(rest):
            color = sc if i < len(strong) - 1 else wc
            s = sq[2 + i]
            if ch == "P" and chess.square_rank(s) in (0, 7):
                ok = False; break
            b.set_piece_at(s, chess.Piece(chess.Piece.from_symbol(ch).piece_type, color))
        if not ok:
            continue
        b.turn = sc
        if b.is_valid() and not b.is_game_over():
            return b
    return None


def _facts(args):
    fen, move = args
    from . import lemma_vocab as V
    before = chess.Board(fen)
    m = chess.Move.from_uci(move)
    after = before.copy(stack=False); after.push(m)
    c = V.Context(before=before, move=m, board=after)
    # every POSITION concept also as its negation ("not: ..."): a stalemate is an ABSENCE (no check, no free move)
    row = frozenset(n if k.fn(c) else f"not: {n}" for n, k in V.VOCAB.items()
                    if k.fn(c) or k.level == "position")
    tgt = frozenset(t for t, f in V.TARGETS.items() if f(c))
    return row, tgt, after.fen(), move


def contexts(seed: int, census: bool, n_endgames: int) -> list:
    out = []
    if census:
        from .tactic_census_measure import load
        for r in load("c300"):
            b = chess.Board(r["fen"])
            if len(b.piece_map()) <= MAX_PIECES:
                out += [(r["fen"], m.uci()) for m in b.legal_moves]
    rng = random.Random(seed)
    for i in range(n_endgames):
        b = random_endgame(rng, MATERIALS[i % len(MATERIALS)])
        if b is not None:
            out += [(b.fen(), m.uci()) for m in b.legal_moves]
    return out


def compute(ctxs, workers):
    from multiprocessing import Pool
    with Pool(workers) as p:
        return list(p.imap_unordered(_facts, ctxs, chunksize=256))


def mine(rows, target: str, names: list[str]):
    """Minimal premises (<= MAX_PREMISE concepts) implying `target` or its negation with no exception -- over the
    DISTINCT (facts, target) rows with their counts (implications depend only on the distinct rows)."""
    pos = [r for r in rows if target in r[1]]
    distinct = Counter((r[0], target in r[1]) for r in rows)
    out = []
    for concl, wanted in ((target, True), (f"not {target}", False)):
        found = []
        for k in range(0, MAX_PREMISE + 1):
            for prem in itertools.combinations(names, k):
                if any(set(f) <= set(prem) for f, _ in found):
                    continue                                          # a subset already implies it: not minimal
                if any(f"not: {x}" in prem for x in prem):
                    continue                                          # a concept and its negation together: empty
                ps = set(prem)
                match = [(f, t, n) for (f, t), n in distinct.items() if ps <= f]
                support = sum(n for _, _, n in match)
                if support < MIN_SUPPORT:
                    continue
                if all(t == wanted for _, t, _ in match):
                    found.append((prem, support))
        out += [dict(premise=list(p), conclusion=concl, support=s) for p, s in found]
    return out, len(pos)


def run(target: str, workers: int, n_endgames: int, proposer_name: str) -> None:
    from . import lemma_vocab as V
    from .lemma_review import NoProposer, ReviewProposer, ApiProposer
    t0 = time.time()
    rows = compute(contexts(1, True, n_endgames), workers)
    hold = compute(contexts(2, False, n_endgames), workers)
    print(f"TIME facts: {time.time() - t0:.0f} s -- {len(rows)} contexts (census + endgames), holdout {len(hold)}", flush=True)
    names = sorted(set(n for r in rows for n in r[0]))
    # deduplicate (implications depend on distinct rows; support counts the duplicates)
    t1 = time.time()
    lemmas, n_pos = mine(rows, target, names)
    hold_rows = [(r[0], r[1]) for r in hold]
    for l in lemmas:
        ps = set(l["premise"]); want = not l["conclusion"].startswith("not ")
        l["holdout_support"] = sum(1 for r in hold_rows if ps <= r[0])
        l["holdout_exceptions"] = sum(1 for r in hold_rows if ps <= r[0] and (target in r[1]) != want)
        key = (l["premise"][0], l["conclusion"]) if len(l["premise"]) == 1 else None
        l["proof"] = LEAN.get(key, "exhaustive on the data") if key else "exhaustive on the data"
        l["vocabulary"] = [dict(name=p, source=V.VOCAB[p.removeprefix("not: ")].source) for p in l["premise"]]
    # a lemma counts as holdout-CHECKED only if the holdout has enough cases of its premise (2026-10-10: rules with no
    # holdout case at all "survived" it); the rest stay UNTESTED and are not accepted
    accepted = [l for l in lemmas if l["holdout_exceptions"] == 0 and l["holdout_support"] >= MIN_SUPPORT // 2]
    untested = [l for l in lemmas if l["holdout_exceptions"] == 0 and l["holdout_support"] < MIN_SUPPORT // 2]
    for l in untested:
        print(f"OBS   UNTESTED (the holdout has {l['holdout_support']} cases of the premise): {' AND '.join(l['premise'])} => {l['conclusion']}")
    print(f"TIME mining: {time.time() - t1:.0f} s; {len(V.VOCAB)} concepts; '{target}' true in {n_pos} contexts")
    print(f"OBS {len(lemmas)} minimal lemmas with zero exceptions on the data; {len(accepted)} survive the holdout")
    for label, group in (("PROVED in Lean", [l for l in accepted if l["proof"].startswith("Ideas.")]),
                         ("EMPIRICAL (zero exceptions on the data and the holdout; not proved)", [l for l in accepted if not l["proof"].startswith("Ideas.")])):
        print(f"OBS {label}: {len(group)}")
        for l in sorted(group, key=lambda l: (l["conclusion"], len(l["premise"]), -l["support"]))[:25]:
            print(f"OBS   {' AND '.join(l['premise']) or '(always)'}  =>  {l['conclusion']}   "
                  f"(support {l['support']}, holdout {l['holdout_support']}; {l['proof']})")
    for l in lemmas:
        if l["holdout_exceptions"]:
            print(f"OBS   REJECTED by the holdout ({l['holdout_exceptions']} exceptions): {' AND '.join(l['premise'])} => {l['conclusion']}")
    # open cases: contexts no accepted lemma decides
    def decided(r):
        return any(set(l["premise"]) <= r[0] for l in accepted)
    open_pos = [r for r in rows if target in r[1] and not decided(r)]
    open_neg = [r for r in rows if target not in r[1] and not decided(r)]
    rng = random.Random(0)
    sample = rng.sample(open_pos, min(8, len(open_pos))) + rng.sample(open_neg, min(4, len(open_neg)))
    report = dict(target=target, lemmas=accepted, positive_count=n_pos, open_count=len(open_pos),
                  open_negative_count=len(open_neg),
                  open_sample=[dict(fen=r[2], move=r[3], facts=sorted(r[0]), target=target in r[1]) for r in sample])
    print(f"OBS open cases: '{target}' true and undecided {len(open_pos)}, false and undecided {len(open_neg)}")
    reg_path = paths.derived("lemma_registry.json")
    reg = json.load(open(reg_path)) if os.path.exists(reg_path) else {}
    reg[target] = dict(date=time.strftime("%Y-%m-%d %H:%M"), lemmas=accepted, open=report["open_count"], open_negative=len(open_neg))
    json.dump(reg, open(reg_path, "w"), indent=1)
    print(f"registry: {reg_path}")
    proposer = {"review": ReviewProposer, "api": ApiProposer, "none": NoProposer}[proposer_name]()
    new = proposer.propose(target, report)
    if new:
        print(f"concepts from reviews in the vocabulary: {new}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="stalemate")
    ap.add_argument("--endgames", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--proposer", default="review", choices=["review", "api", "none"])
    a = ap.parse_args(argv)
    run(a.target, a.workers, a.endgames, a.proposer)


if __name__ == "__main__":
    main()
