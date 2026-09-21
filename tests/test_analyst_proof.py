import copy
import chess

from catspace.incubator.analyst import rank
from catspace.incubator.analyst.memo import Memo
from catspace.incubator.analyst.proof import check, Just


def _term():
    b = chess.Board("r3kb1r/ppp1qpp1/3pbn1p/4P3/2BQ1B2/2N5/PPP2PPP/R3K2R w KQkq - 0 1")
    rank.FAMILY_LEMMAS = True; rank.sure_gain(b, b.parse_san("exf6"), Memo())
    return rank.LAST_TERMS["e5f6"]


def test_kernel_accepts_the_ledgers_term_and_counts_what_is_proven():
    t = _term(); c = check(t)
    assert c["ok"] and c["proven"] == 6 and c["holes"] == len(t.replies) - 6      # the six non-capturing discovered checks, closed by the lemma
    assert t.claim <= min(j.value for j in t.replies.values()) + 1e-9


def test_kernel_rejects_forgeries():
    t = _term()
    bad = copy.deepcopy(t); bad.replies["e7f6"] = Just("lemma", 5.0, "forged", "f6e7", (chess.E6, chess.E7))
    assert not check(bad)["ok"]                                                    # Qxf6 is no member of the bishop family
    bad = copy.deepcopy(t); del bad.replies["e7f6"]
    assert not check(bad)["ok"]                                                    # a reply left uncovered
    bad = copy.deepcopy(t); bad.claim = 99.0
    assert not check(bad)["ok"]                                                    # a claim above the minimum
    bad = copy.deepcopy(t); bad.replies["e7f6"] = Just("mate", 1000.0)
    assert not check(bad)["ok"]                                                    # a mate that is not
