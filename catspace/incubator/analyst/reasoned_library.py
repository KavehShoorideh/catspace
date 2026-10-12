"""THE LIBRARY OF REASONED SOLUTIONS, for inspiration (Kaveh 2026-10-12: "I want similarly reasoned solutions to exist
somewhere where we can draw inspiration from, so some part of our loop should be able to search for similarly reasoned
proofs").

Precedent: proof by analogy and proof plans (whole methods carried over, not single steps -- Bundy; Kerber), retrieval of
premises before proving (LeanDojo's ReProver), and for these endgames Huberman's per-endgame "better"/"worse" predicates
(1968) and Bratko and Michie's advice tables (holding goals and better goals). The caveat (Melis): storing, retrieving and
replaying an analogy can cost more than proving afresh -- so retrieval only PROPOSES; the discoverer and the Lean proofs
decide.

Each entry is a reasoned solution in the discoverer's terms -- the goal's relaxation, the allocation (each piece's role),
the invariants and pass moves, the measure -- with its source and status. Retrieval is by the SHAPE OF SUPPLY of the
pieces (how each covers the king's box: a king one side; a rook lines; a queen lines and diagonals; a bishop diagonals of
one colour; a knight hops onto the other colour), so a new material finds the entries whose pieces cover alike.

  python -m catspace.incubator.analyst.reasoned_library --pieces KBN
"""
from __future__ import annotations

import argparse

SUPPLY = {"K": {"king", "one side"}, "R": {"lines"}, "Q": {"lines", "diagonals"}, "B": {"diagonals", "one colour"},
          "N": {"hops", "colour flip"}}

LIBRARY = [
    dict(id="krk_push", material="KR", goal="mate", source="Kaveh 2026-10-12; rediscovered by mate_discovery.py",
         status="discovered (window); Lean push lemma to prove",
         relaxation="retreat by check, the back row open",
         allocation="our king in opposition (0, -2) covers the front row; the rook on the king's row, two or more away, "
                    "gives the check and covers the middle row; the reverse is impossible (our king never covers the middle)",
         invariants="the rook's line is a wall (the king stays beyond it); sliding the rook along its line keeps the wall "
                    "and its far side is safe -- the PASS move",
         measure="rows left beyond the wall; then our king's distance to the opposition square; a waiting slide when "
                 "one file off",
         base="on the back rank the retreat is impossible: the push is mate"),
    dict(id="kqk_squeeze", material="KQ", goal="mate", source="Kaveh 2026-10-12; rediscovered by mate_discovery.py",
         status="discovered (window); the box measure to prove",
         relaxation="retreat by zugzwang, the back row (and a side) open",
         allocation="the queen alone a knight's move from the king covers all but the open side; our king covers nothing "
                    "-- its moves are passes",
         invariants="the queen a knight's move away is safe and the box (the rectangle her lines cut) never grows",
         measure="the box's size; then our king's distance to the critical square (c2 for a king on a1/a2)",
         base="two squares on the edge (never one: stalemate); our king on c1-c3, Qb2 mates"),
    dict(id="kqk_region", material="KQ", goal="mate", source="ours 2026-10-12 (mating.rs)",
         status="measured: 1,000 / 1,000, every move certified; tuned, not structural",
         relaxation="(none named)", allocation="(the proved shapes: our king two squares in, the queen on the edge row or "
                                              "directly in front -- Lean edge_shape)",
         invariants="no stalemate, the queen not left to the king, no claimable draw",
         measure="region squares off the edge; distance to the farthest front-zone target; region size; tempo moves",
         base="the net: mate within three, searched"),
    dict(id="krk_bratko", material="KR", goal="mate",
         source="Bratko's KRK strategy; formally verified in Isabelle/HOL (Maric, Janicic, Malikovic, CADE 2015; LMCS 2019) "
                "https://arxiv.org/abs/1801.07528",
         status="literature: proved for n x n boards (~7,800 lines of Isabelle, linear arithmetic by Z3)",
         relaxation="squeeze (the room shrinks)", allocation="the rook cuts the king off; our king approaches the critical "
                                                            "square; ready-to-mate patterns",
         invariants="the rook safe (RookSafe, RookHome), the room never grows (KeepRoom)",
         measure="the room (half-perimeter of the rectangle the rook confines the king to), then the distance to the "
                 "critical square",
         base="ImmediateMate, ReadyToMate"),
    dict(id="huberman_kbnk", material="KBN", goal="mate",
         source="Huberman, A program to play chess end games (Stanford AI Memo 65, 1968) https://apps.dtic.mil/sti/pdfs/AD0673971.pdf",
         status="literature: per-endgame 'better' and 'worse' predicates; content to read",
         relaxation="(predicates of progress, as in chess books)",
         allocation="(to read)", invariants="(to read)",
         measure="drive the lone king away from the centre, then into the corner of the bishop's colour",
         base="mate in the corner of the bishop's colour"),
    dict(id="bratko_advice", material="any", goal="any",
         source="Bratko & Michie, advice tables (AISB 1978); AL1 (IJCAI 1979) https://mlanthology.org/ijcai/1979/bratko1979ijcai-implementing",
         status="literature: the representation",
         relaxation="pieces of advice: a better goal to reach while a holding goal is kept",
         allocation="(per ending)", invariants="the holding goal (= our invariants)", measure="the better goal (= our measure)",
         base="(per ending)"),
]


def signature(material: str) -> set:
    return set().union(*(SUPPLY[p] for p in material if p in SUPPLY)) if material != "any" else set()


def retrieve(pieces: str, k: int = 4):
    """The entries whose pieces cover alike, nearest first (Jaccard on the supply shapes; the general ones last)."""
    want = signature(pieces)
    scored = []
    for e in LIBRARY:
        have = signature(e["material"])
        score = len(want & have) / len(want | have) if have else 0.05
        scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="KBN")
    a = ap.parse_args(argv)
    print(f"OBS inspiration for {'+'.join(a.pieces)} (supply shapes {sorted(signature(a.pieces))}):")
    for score, e in retrieve(a.pieces):
        print(f"OBS   {score:.2f} {e['id']} ({e['material']}; {e['status']}) -- {e['source'].split(' http')[0]}")
        print(f"OBS        relaxation: {e['relaxation']}; measure: {e['measure']}")


if __name__ == "__main__":
    main()
