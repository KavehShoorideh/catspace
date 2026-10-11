"""catspace.incubator.analyst -- the layered position analyst (Kaveh 2026-09-04).

Layer stack; each layer imports ONLY the ones below it:
    L0 geometry   the board as a graph (squares, direction families, rays); size is a parameter
    L1 rules      piece set, movement, special moves, legality, terminal conditions, OBJECTIVE
                  (python-chess Board classes incl. variants), and the objective's EXCHANGE SCALE
                  (Scale: what a piece is worth in a trade -- a derivative of the objective, declared
                  with it, swappable per variant via rules.set_scale). The objective itself is
                  quarantined: nothing in L2-L4 may read it; the scale is the one thing they may
                  consume from that side, and only through value()/Scale so the dependence is visible.
    L2 facts      global, square-free state facts + the conventional exchange scale
    L3 relations  static relations over pieces, indexed by ARITY, defined only via L1 probes
    L3 operators  (defense.py) the composition rule (Kaveh 2026-09-05): an ATOM is defined by an L1 probe
                  (a legal move, an attack, a line through a piece), a COMPOUND by operators over atoms -- glue on
                  shared pieces, the removal test, EFFECTIVE DEFENDERS (a defender counts only if the rules let it
                  recapture and recapturing does not expose more than it saves, priced by SEE), the null move --
                  and nothing above the atoms is named by hand. Under a rule shift the atoms are recomputed by the
                  new rules and the compounds follow (test: the same placement under antichess changes who defends).
    L3.5 structure  positional facts with the same shape as L3 (structure.py): isolated / doubled / passed /
                  backward pawns, islands, rooks on open files, outposts, bishop pair, blocked (bad) bishops,
                  open files by a castled king, and each piece's SCOPE. Diffed per move like L3 (Effect.structure_
                  created / _ended); phrased as OBSERVATIONS ("gives up the bishop pair", "opens a line for the
                  bishop on c8"), never as causes -- the plan layer says why a change matters.
    L4 dynamics   how relations change under moves: threats (null move), per-ply line effects,
                  tension, zugzwang
    L5 ideas      value enters here: Stockfish + the objective -> idea / killed ideas / alternatives
    L6 verbalize  names + the idea grammar: records -> text (deterministic)
The Stockfish adapter (engine.py) is a service to L4/L5, not a layer.
"""

# THE FORCING SEARCH'S STOP POLICY (2026-10-11): analyst_core loads it at import from this variable; the trained file is
# the one in the registry. Unset it to run the full search.
import os as _os
try:
    from catspace.io import paths as _paths
    from pathlib import Path as _P
    _pol = _P(_paths.derived("forcing_policy.json"))                      # the latest trained (data/, DVC)
    if not _pol.exists():
        _pol = _P(__file__).with_name("forcing_policy.json")               # the shipped copy (rule 3: the engine loads it at play time)
    if _pol.exists():
        _os.environ.setdefault("CATSPACE_FORCING_POLICY", str(_pol))
except Exception:
    pass
