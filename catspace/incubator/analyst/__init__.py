"""catspace.incubator.analyst -- the layered position analyst (Kaveh 2026-09-04).

Layer stack; each layer imports ONLY the ones below it:
    L0 geometry   the board as a graph (squares, direction families, rays); size is a parameter
    L1 rules      piece set, movement, special moves, legality, terminal conditions, OBJECTIVE
                  (python-chess Board classes incl. variants). The objective is quarantined:
                  nothing in L2-L4 may read it.
    L2 facts      global, square-free state facts + the conventional exchange scale
    L3 relations  static relations over pieces, indexed by ARITY, defined only via L1 probes
    L4 dynamics   how relations change under moves: threats (null move), per-ply line effects,
                  tension, zugzwang
    L5 ideas      value enters here: Stockfish + the objective -> idea / killed ideas / alternatives
    L6 verbalize  names + the idea grammar: records -> text (deterministic)
The Stockfish adapter (engine.py) is a service to L4/L5, not a layer.
"""
