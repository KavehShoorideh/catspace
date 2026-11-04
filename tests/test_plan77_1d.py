"""PLAN 77.1 (d) (2) THE CERTIFY STAGE (docs/ANALYST_PLAN.md 77.1 (d); discovery/certify.py): every fast binding the hot path reads carries a certificate against its oracle, kept
in the store and keyed by the files its answers depend on; the search keeps checking the cheap readers online, and a disagreement demotes the reader to its oracle. Count budgets."""
import copy

import chess

from catspace.incubator.analyst.discovery import certify as CF
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import State

KNIGHT = "4k3/8/2n5/8/8/8/8/3QK3 w - - 0 1"                 # Qa4 attacks the loose knight on c6: 'threat 3'
HIT = "4k3/8/2n5/8/Q7/8/8/4K3 w - - 0 1"                    # the queen on a4 already attacks the loose knight: an exchange claim stands at the root
MATE1 = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"                 # Ra8#: a mate in one
PASSER = "8/5k2/8/1P6/8/8/6K1/8 w - - 0 1"                  # the b-pawn runs: the passer lemma fires
W0VFO = "2r2rk1/R5pp/5p2/2q1p3/3pQ3/8/2P2PPP/R5K1 w - - 4 30"


def _fresh_process_state():
    CF.reset()
    G.clear_caches()
    R._DEF.clear()
    R._BEFORE.clear()


def _planted(monkeypatch, bump: int = 0, drop: str = ""):
    """A WRONG native reader planted for the test: every exchange claim worth `bump` more, or the claims of reader `drop` dropped."""
    real = R._A.definition_rows_pos

    def wrong(P, stake, me, readers, masks):
        before, rows = real(P, stake, me, readers, masks)
        fix = lambda cl: [(r, sq, v + (bump if r == "exchange" else 0)) for r, sq, v in cl if r != drop]
        return fix(before), [(u, pre, cv, fix(cl), lost, [(cu, e, fix(dc), f) for cu, e, dc, f in caps], q) for u, pre, cv, cl, lost, caps, q in rows]
    monkeypatch.setattr(R._A, "definition_rows_pos", wrong)


def test_77_1d_every_binding_has_a_fresh_certificate():
    """The certificates in the store (lemmas/certificates/) are fresh: each exists, found no disagreement, and its key is the current files' (a changed GDL text or a changed
    binding source makes it stale: `python -m catspace.incubator.analyst.discovery.certify run --jobs 11`)."""
    st = CF.status()
    for n in CF.BINDINGS:
        c = CF.load(n)
        print(f"OBS 77.1d certificate {n}: {st[n]}" + (f", positions {c['positions']}, disagreements {c['disagreements']}, {c['date']}" if c else ""))
    assert all(v == "fresh" for v in st.values()), st


def test_77_1d_a_planted_wrong_exchange_reader_is_found_online_demoted_and_the_definition_reads_the_oracle(monkeypatch):
    b = chess.Board(HIT)
    ms = list(b.legal_moves)
    _fresh_process_state()
    want = [R.forcing_by_definition(b, m, 3, True) for m in ms]                   # the readers as built
    want_er = [(R.every_reply(b, m, 3, True) or (None,))[0] for m in ms if b.gives_check(m)]
    try:
        _fresh_process_state()
        _planted(monkeypatch, bump=1)
        monkeypatch.setattr(CF, "ONLINE_SHARE", 1.0)                              # (every node checked: the planted error must be met)
        got = [R.forcing_by_definition(b, m, 3, True) for m in ms]
        assert CF.FOUND and CF.FOUND[0][0] == "reader exchange" and "exchange" in CF.DEMOTED
        assert got == want                                                        # read through the oracle (the regression): the planted reader's values never reach an answer
        assert [(R.every_reply(b, m, 3, True) or (None,))[0] for m in ms if b.gives_check(m)] == want_er
        print(f"OBS 77.1d planted reader: found {CF.FOUND[0][2]}; demoted {sorted(CF.DEMOTED)}; {len(ms)} answers equal the built reader's")
    finally:
        _fresh_process_state()


def test_77_1d_a_too_narrow_domain_is_found_online_and_the_lemma_asked_everywhere(monkeypatch):
    b = chess.Board(PASSER)
    try:
        _fresh_process_state()
        patched = copy.deepcopy(LS.load())
        patched["unstoppablePasser2_sound"]["needs"]["theirs"] = "Q"                # a domain that leaves out the board where the lemma fires
        monkeypatch.setattr(LS, "_STORE", [patched])
        key = LS.goal_kinds().promotion.question                                  # (the passer lemma's claims; the race word's own domain still holds here)
        assert not any(k[0] == key for k in R._promotion_claims(b, 1))            # outside the (wrong) domain: not asked
        CF.online_check(b, [], 1)
        assert any(f[0] == "domain unstoppablePasser2_sound" for f in CF.FOUND) and "domain unstoppablePasser2_sound" in CF.DEMOTED
        assert any(k[0] == key for k in R._promotion_claims(b, 1))                # demoted: asked everywhere, and it fires
    finally:
        _fresh_process_state()


def test_77_1d_the_offline_checks_find_planted_errors(monkeypatch):
    assert CF.exchange_reader_against_regression(HIT) == [] and CF.terminal_reader_against_spec(MATE1) == []
    _planted(monkeypatch, bump=1)
    assert CF.exchange_reader_against_regression(HIT)
    _planted(monkeypatch, drop="terminal")
    assert CF.terminal_reader_against_spec(MATE1)                               # the spec reads Ra8# where the reader (planted) says nothing
    assert CF.domains_against_lemmas(PASSER) == []


def test_77_1d_the_online_share_is_kept_and_printed():
    _fresh_process_state()
    ag = Agenda(State(W0VFO, "w", []))
    ag.ask()
    d = ag.decision()
    sp = dict(CF.SPENT)
    print(f"OBS 77.1d online checks on W0VFO (count budget): positions checked {sp['positions']}, checks {sp['checks']:.3f} s against reading {sp['reading']:.3f} s "
          f"({100 * sp['checks'] / max(sp['reading'], 1e-9):.1f} %, fuse {100 * CF.ONLINE_SHARE:.0f} %); found {len(CF.FOUND)}; decision {d.what if d else None}")
    assert sp["positions"] > 0 and not CF.FOUND and d is not None and d.what == "Qb7"
    assert sp["checks"] <= CF.ONLINE_SHARE * sp["reading"] + 0.05               # (the last check may run past the share by one check)
    _fresh_process_state()
