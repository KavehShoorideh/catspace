import chess

from catspace.incubator.analyst.lemmas import discovered_check_families, refute, lemmas_after
from catspace.incubator.analyst import rank
from catspace.incubator.analyst.memo import Memo


def test_discovered_check_family_and_its_refutation():
    # after exf6 in the Be6 position: every bishop move discovers check from the queen on e7; fxe7 answers all
    b = chess.Board("r3kb1r/ppp1qpp1/3pbn1p/4P3/2BQ1B2/2N5/PPP2PPP/R3K2R w KQkq - 0 1"); b.push_san("exf6")
    fams = discovered_check_families(b)
    assert len(fams) == 1 and fams[0].mover == chess.E6 and fams[0].opener == chess.E7 and len(fams[0].members) == 7
    lem = refute(b, fams[0])
    assert lem is not None and lem.answer == chess.Move.from_uci("f6e7") and len(lem.covers) == 6      # Bxc4+ captures: searched on its own
    assert "fxe7 answers every one of them" in lem.text


def test_family_lemma_keeps_the_ledger_honest_and_cheaper():
    b = chess.Board("r3kb1r/ppp1qpp1/3pbn1p/4P3/2BQ1B2/2N5/PPP2PPP/R3K2R w KQkq - 0 1"); m = b.parse_san("exf6")
    rank.FAMILY_LEMMAS = False; g0, why0 = rank.sure_gain(b, m, Memo())
    rank.FAMILY_LEMMAS = True; g1, why1 = rank.sure_gain(b, m, Memo())
    assert g1 >= g0 - 1e-9 and "lemma:" in why1 and "lemma:" not in why0


def test_no_family_where_no_line_opens():
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"); b.push_san("Ng5")
    assert lemmas_after(b) == []
