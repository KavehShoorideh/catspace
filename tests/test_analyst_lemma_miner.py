"""The lemma miner (catspace/incubator/analyst/lemma_miner.py) must rediscover Kaveh's stalemate rules unprompted."""
from catspace.incubator.analyst import lemma_miner as LM


def test_miner_rediscovers_the_stalemate_rules():
    rows = [LM._facts(c) for c in LM.contexts(seed=3, census=False, n_endgames=300)]
    lemmas, n_pos = LM.mine(rows, "stalemate", sorted(set(n for r in rows for n in r[0])))
    found = {(tuple(l["premise"]), l["conclusion"]) for l in lemmas}
    assert (("the move gave check",), "not stalemate") in found          # "give check with every move"
    assert (("defender has a free pawn",), "not stalemate") in found     # "a piece that can move"
    assert all(l["conclusion"] in ("stalemate", "not stalemate") for l in lemmas)
