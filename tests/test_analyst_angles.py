import chess

from catspace.incubator.analyst.angles import angles, summary, closed_families, events


def test_castled_king_behind_three_pawns():
    b = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PPQ2PPP/R4RK1 w - - 0 10")
    ang = {a.name: a for a in angles(b, chess.WHITE)}
    assert ang["north"].state == "closed" and ang["north"].shield == chess.G2
    assert ang["north-east"].state == "closed" and ang["north-east"].shield == chess.H2
    assert ang["north-west"].state == "closed" and ang["north-west"].shield == chess.F2
    assert "behind 3 pawns" in summary(b, chess.WHITE)
    fams = closed_families(b, chess.WHITE)
    assert any(chess.D8 in c for _, c in fams)                                # the black queen's attacks along closed angles are a refuted family


def test_an_angle_opens_when_the_shield_moves():
    b = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PPQ2PPP/R4RK1 w - - 0 10")
    b2 = b.copy(); b2.push_san("g4")
    ev = events(b, b2, chess.WHITE)
    assert any("north angle" in e and "closed ->" in e for e in ev)


def test_open_and_contested_angles_are_named():
    b = chess.Board("r3kb1r/ppp1qpp1/3pbn1p/4P3/2BQ1B2/2N5/PPP2PPP/R3K2R w KQkq - 0 1")
    ang = {a.name: a for a in angles(b, chess.BLACK)}
    assert ang["south-west"].state == "open" and chess.C4 in ang["south-west"].concerns    # the bishop on c4 eyes the a2-g8 diagonal
    ang_w = {a.name: a for a in angles(b, chess.WHITE)}
    assert ang_w["north"].state == "contested" and ang_w["north"].shield == chess.E5     # dxe5 would open the e-file (the pin motif)
