"""The lemmas we already have, tried first (pipeline.Known / known_terms)."""
from catspace.incubator.analyst.discovery.pipeline import Known, known_terms


def test_known_ranks_only_the_known_in_their_order():
    lab = {"rows": [dict(term=t, schema=s) for t in ("(count edge)", "(count corner)", "(count attacked)") for s in ("append", "replace")]}
    k = Known(["(count attacked)", "(count edge)"])
    assert k.rank(None, lab, {}) == [("(count attacked)", "append"), ("(count attacked)", "replace"),
                                     ("(count edge)", "append"), ("(count edge)", "replace")]


def test_known_terms_reads_the_registry_and_the_frames():
    t = known_terms(dict(frames={"KXvK": ["(count corner)"]}))
    assert "(count corner)" in t and len(t) == len(set(t)) and len(t) > 1    # the registry's measures, then the frames'


def test_a_term_naming_an_absent_piece_is_constant():
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    kbn = ["8/8/8/3k4/8/3K4/3BN3/8 w - - 0 1", "8/8/3k4/8/8/2K5/3BN3/8 w - - 0 1"]
    assert len(set(A.vocab_values(kbn, "(count (segment W O))"))) == 1          # they have no other piece: always 0
    assert len(set(A.vocab_values(kbn, "(count (nearer L P))"))) > 1
