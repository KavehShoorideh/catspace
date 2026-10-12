"""HOLES and their LABELS (Kaveh 2026-10-12: the learned prior, modular). A HOLE is a choice an operator must make -- here the
MONOTONE hole: a frame's measure with a component missing, and the boards (random legal positions of the material, the
attacker to move) that the remaining measure cannot certify against every reply. Its LABELS are the exhaustive answer:
every distinct term over the base, in every schema, scored by the verifier on the hole's boards -- the teacher the
learned proposers learn to rank like, and the yardstick they are judged by (how many verifier calls until a near-best
term).

  python -m catspace.incubator.analyst.discovery.holes --make --label
"""
from __future__ import annotations

import argparse
import json
import random
import time

from catspace.io import paths

from . import base_vocabulary as V

HOLES = [
    dict(name="kr_wall", material="KR", base=["fn:wall"]),
    dict(name="kr_derived", material="KR", base=["fn:wall", "fn:opposition"]),
    dict(name="kr_opposition", material="KR", base=["fn:opposition"]),
    dict(name="kq_offedge", material="KQ", base=["fn:kq_offedge"]),
    dict(name="kq_offedge_approach", material="KQ", base=["fn:kq_offedge", "fn:kq_approach"]),
    dict(name="kq_region", material="KQ", base=["fn:kq_region"]),
    dict(name="kq_offedge_region", material="KQ", base=["fn:kq_offedge", "fn:kq_region"]),
    dict(name="kq_approach", material="KQ", base=["fn:kq_approach"]),
    # the EMPTY base: a whole measure from nothing (every board is stuck)
    dict(name="kr_none", material="KR", base=[]),
    dict(name="kq_none", material="KQ", base=[]),
]
MIN_STUCK = 50                                   # a hole with fewer stuck boards teaches nothing
SCHEMAS = ("append", "sum", "replace")
K = 2


def random_positions(material: str, n: int, seed: int = 0) -> list:
    import chess
    rng = random.Random(seed)
    kind = {"Q": chess.QUEEN, "R": chess.ROOK}[material[1]]
    out = []
    while len(out) < n:
        a, b, c = rng.sample(range(64), 3)
        if chess.square_distance(a, c) < 2:
            continue
        brd = chess.Board(None)
        brd.set_piece_at(a, chess.Piece(chess.KING, chess.WHITE))
        brd.set_piece_at(b, chess.Piece(kind, chess.WHITE))
        brd.set_piece_at(c, chess.Piece(chess.KING, chess.BLACK))
        brd.turn = chess.WHITE
        if brd.is_valid() and not brd.is_game_over():
            out.append(brd.fen())
    return out


def make(n: int = 2000, cap: int = 300) -> None:
    import analyst_core as A
    for h in HOLES:
        fens = random_positions(h["material"], n, seed=hash(h["name"]) % 1000)
        mask = A.verify_mask(fens, h["base"], None, "none", K, h["material"][1])
        stuck = [f for f, ok in zip(fens, mask) if not ok][:cap]
        if len(stuck) < MIN_STUCK:
            print(f"OBS hole {h['name']}: only {len(stuck)} stuck of {n} -- dropped")
            continue
        json.dump(dict(h, k=K, stuck=stuck, sampled=n), open(paths.derived(f"hole_{h['name']}.json"), "w"))
        print(f"OBS hole {h['name']}: {len(stuck)} stuck of {n} random {h['material']} positions (base {h['base']})")


def available() -> list:
    import os
    return [h["name"] for h in HOLES if os.path.exists(paths.derived(f"hole_{h['name']}.json"))]


def load(name: str) -> dict:
    return json.load(open(paths.derived(f"hole_{name}.json")))


def label(name: str, probe: int = 100) -> dict:
    """Every distinct term (observationally, on the probe boards) in every schema, verified on the probe."""
    import analyst_core as A
    h = load(name)
    rng = random.Random(1)
    boards = rng.sample(h["stuck"], min(probe, len(h["stuck"])))
    t0 = time.time()
    seen, terms = set(), []
    for t in V.enumerate_numbers():
        v = tuple(A.vocab_values(boards, V.sexpr(t)))
        if len(set(v)) > 1 and v not in seen:
            seen.add(v); terms.append(t)
    # both verifier stages on every (term, schema) -- the exhaustive teacher; each verdict also enters the experience store
    from .verifier import Verifier
    ver = Verifier()
    schemas = SCHEMAS if h["base"] else ("append",)          # with no base the three schemas coincide
    rows = []
    for t in terms:
        for sch in schemas:
            r = ver.verify(h, V.sexpr(t), sch, boards)
            rows.append(dict(term=r["term"], schema=sch, stage1=r["stage1"], mated=r["mated"], certified=r["certified"],
                             played=r["played"], score=r["score"]))
    out = dict(hole=name, boards=boards, n=len(boards), rows=rows)
    json.dump(out, open(paths.derived(f"hole_{name}_labels.json"), "w"))
    best = max(rows, key=lambda r: r["score"])
    top1 = max(rows, key=lambda r: r["stage1"])
    print(f"OBS labels {name}: {len(terms)} distinct terms x {len(schemas)} schemas on {len(boards)} boards ({time.time() - t0:.0f} s); "
          f"best by both stages {best['score']:.3f} (mated {best['mated']}, certified {best['certified']}/{best['played']}) [{best['schema']}] "
          f"{best['term']} = {V.gloss(V.parse(best['term']))}; best by stage 1 alone: {top1['stage1']} [{top1['schema']}] {top1['term']} (score {top1['score']:.3f})")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    if a.make:
        make()
    if a.label:
        import os
        for h in HOLES:
            if (not a.only or h["name"] in a.only.split(",")) and os.path.exists(paths.derived(f"hole_{h['name']}.json")):
                label(h["name"])


if __name__ == "__main__":
    main()
