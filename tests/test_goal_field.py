"""The goal field's stages 1-2 (rust goals.rs): the natural opponent and our goals imagined against it."""
import catspace.incubator.analyst  # noqa: F401
import analyst_core as A


def test_the_natural_opponent():
    assert A.default_reply_of("3r2k1/8/8/8/8/8/5PPP/6K1 b - - 0 1") == "d8d1"        # a mate in one is taken
    assert A.default_reply_of("4k3/8/4p3/3N4/8/8/8/4K3 b - - 0 1") == "e6d5"          # a recapture
    r = A.default_reply_of("4k3/8/8/8/3r4/8/1B6/4K3 b - - 0 1")                       # an attacked rook is saved
    assert r is not None and r.startswith("d4") and r != "d4d2"
    assert A.default_reply_of("4k3/8/8/8/8/8/8/R3K3 b - - 0 1") == "pass"             # nothing to gain: a pass


def test_goals_imagined():
    g, _ = A.goals_of("3B4/8/8/8/8/7K/8/5N1k w - - 0 1", 3, "us")
    assert g[0][:4] == ("mate", "their king", 500, 2) and g[0][4].startswith("Ng3")
    g, _ = A.goals_of("4k3/8/8/3r4/8/2N5/8/4K3 w - - 0 1", 3, "us")
    assert ("material", "their rook", 5, 1) in [x[:4] for x in g]
    g, _ = A.goals_of("8/8/8/8/1P6/8/6k1/K7 w - - 0 1", 5, "us")
    assert ("known", "KQvK") in [x[:2] for x in g]
    g, _ = A.goals_of("8/8/8/8/1P6/8/6k1/K7 w - - 0 1", 3, "us")                     # four pawn moves: beyond three
    assert ("known", "KQvK") not in [x[:2] for x in g]


def test_their_goals_as_if_we_passed():
    g, _ = A.goals_of("3r2k1/8/8/8/8/8/5PPP/6K1 w - - 0 1", 2, "them")                 # their back-rank mate
    assert g and g[0][:2] == ("mate", "their king")
