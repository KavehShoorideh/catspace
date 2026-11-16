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


def test_71_4_the_lean_game_structure_states_the_axioms_and_restates_reach_and_the_race():
    """plan 71.4: `lean/ideas/Ideas/Game.lean` -- `structure Game` (moves, apply, pass, measure, inCheck), the derived step / terminal / mated / stalemated, `Game.Reach`
    (Forced.Reach with the game's step) and `Game.race_sound` (Race.race_sound restated with the game's hypotheses); `chess` the instance; `kothGoal` a variant by its goal."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "lean" / "ideas" / "Ideas" / "Game.lean").read_text() + (root / "lean" / "ideas" / "Ideas" / "Core" / "Game.lean").read_text()      # plan 78.56: the structure moved to the core
    for name in ("structure Game (State Move : Type) where", "def step", "def terminal", "def mated", "def stalemated", "theorem terminal_iff", "def Reach",
                 "theorem race_sound", "def chess : Game Position Chess.Move where", "theorem chess_measure_pass", "def kothGoal"):
        assert name in src, name
    assert "sorry" not in src
    assert "import Ideas.Game" in (root / "lean" / "ideas" / "Ideas.lean").read_text()
    idx = (root / "docs" / "LEAN_INDEX.md").read_text()
    assert "Ideas.Game.Game.race_sound" in idx and "Ideas.Game.Game.terminal_iff" in idx


def test_71_6_chess960_is_the_same_game_with_one_parameter_and_king_of_the_hill_is_a_goal():
    import chess
    from catspace.incubator.analyst.discovery import game as GM
    b = GM.CHESS960.from_key("bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w KQkq - 0 1")      # a Chess960 start: the castling rights parse by the 960 rule
    assert b.chess960 and len(GM.CHESS960.moves(b)) == 20 and GM.CHESS960.terminal(b) is None and GM.CHESS960.measure(b, chess.WHITE) == 0
    k = GM.KING_OF_THE_HILL
    assert k.terminal(chess.Board()) is None
    assert k.terminal(chess.Board("8/8/8/3k4/8/8/8/4K3 w - - 0 1")) == "loss"                   # their king stands on d5 with us to move: we have lost
    assert k.terminal(chess.Board("8/8/8/3K4/8/8/8/4k3 b - - 0 1")) == "loss" and k.terminal(chess.Board("8/8/8/3K4/8/8/8/4k3 w - - 0 1")) == "win"
    assert k.terminal(chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")) == "mate"               # the chess terminals still hold
