"""First, don't lose (Kaveh 2026-10-12): when no move keeps everything, among equal nets the planner hands over the
least material -- a draw they may claim keeps our pieces (the carry2 run, seed 2097: Kd2 dropped the d5 rook rather than
allow Rb4's threefold repetition, both "net -5", and K+R v K+R then lost)."""
import chess

import catspace.incubator.analyst  # noqa: F401
import analyst_core as A

FEN = "8/8/2k5/3R4/2r5/8/8/2KR4 w - - 96 49"           # in check from c4; Kb1 / Kb2 allow Rb4+ a third time, Kd2 drops d5


def test_the_audit_sees_both_losses():
    hist = _history()
    nets = {u: (n, why) for u, n, why in A.audit_check(FEN, hist)}
    assert nets["c1b1"][0] == nets["c1b2"][0] == nets["c1d2"][0] == -5
    assert "threefold" in nets["c1b1"][1] and "d5" in nets["c1d2"][1]


def test_the_planner_keeps_the_rook():
    A.set_generic_frame(True)
    st = A.PlanState()
    for f in _history():
        st.record(f)
    tr, dec, *_ = st.observe(FEN)
    move = next((u for u in (dec or ())[:2] if u), None)        # (the driver plays the first of the two that is set)
    assert move in ("c1b1", "c1b2"), (dec, tr)


def _history():
    out = []
    for k in ("c1b1", "c1b2"):
        b = chess.Board(FEN); b.push_uci(k); b.push_san("Rb4+")
        out += [b.fen(), b.fen()]                       # each seen twice: Rb4+ would make the third
    return out
