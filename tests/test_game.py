"""plan 71.2: the Game interface (layer 2, the axioms a domain supplies) and chess as its instance."""
import chess

from catspace.incubator.analyst.discovery import game as GM


def test_71_2_chess_is_an_instance_of_the_game_protocol_and_its_axioms_read_right():
    g = GM.CHESS
    assert isinstance(g, GM.Game)
    s = chess.Board("8/1Rb3kp/p2q2p1/2p1pp2/pQ6/1bP4P/1P3PPK/8 w - - 0 35")           # eSJqH
    assert g.turn(s) is chess.WHITE and g.turn(g.pass_(s)) is chess.BLACK and g.key(g.pass_(s)) != g.key(s)
    assert g.measure(s, chess.WHITE) == -g.measure(s, chess.BLACK) == -3                 # White is three down (a bishop and three pawns against ... the balance the status reads)
    assert g.terminal(s) is None and not g.in_check(s)
    m = s.parse_san("Qxb3")
    assert g.is_capture(s, m) and not g.gives_check(s, m) and g.name(s, m) == "Qxb3"
    t = g.apply(s, m)
    assert g.turn(t) is chess.BLACK and g.measure(t, chess.WHITE) == 0
    u = g.unit_at(s, chess.B4)
    assert u is not None and u.owner is chess.WHITE and u.worth == 9 and u.kind == chess.QUEEN
    assert len(g.units(s)) == len(s.piece_map()) and len(g.units(s, chess.BLACK)) == sum(1 for p in s.piece_map().values() if p.color == chess.BLACK)
    assert chess.B4 in g.attackers(s, chess.BLACK, chess.B4) or chess.B4 not in g.attackers(s, chess.BLACK, chess.B4)   # (a list of attacker squares)
    assert set(g.attackers(s, chess.BLACK, chess.B4)) == set(s.attackers(chess.BLACK, chess.B4))
    mate = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"); mate.push_san("Ra8#")
    assert g.terminal(mate) == "mate"
    assert g.loc_name(chess.E4) == "e4" and len(list(g.locations())) == 64
