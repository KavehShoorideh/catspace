"""The Lean index (catspace/incubator/analyst/lean_index.py): parses the project's theorems and finds them by concept."""
from catspace.incubator.analyst import lean_index


def test_index_finds_known_theorems():
    decls = lean_index.build()
    names = {d["full"] for d in decls}
    for n in ("Ideas.StalemateLemmas.check_not_stalemate", "Ideas.backRankMate_sound"):
        assert n in names
    # the self-mining loop's files keep a failed premise / conjecture as `sorry` with the error above it (the statement is the finding); every
    # other proof is closed
    assert not [d for d in decls if d["kind"] in ("theorem", "lemma") and d["open"] and not d["name"].startswith(("premise_", "conjecture_"))], \
        "a proof still holds sorry"


def test_search_by_concept():
    hits = [d["full"] for d in lean_index.search(["stalemate"])]
    assert "Ideas.StalemateLemmas.checking_move_never_stalemates" in hits
    assert any("seeXB" in d["full"] for d in lean_index.search(["exchange", "x-ray"]) + lean_index.search(["seeXB"]))
