"""catspace/probe -- the modular end-to-end probe stack (Kaveh 2026-07-30:
"once we got an end to end system we can modularize the components so we can
try different algorithms").

Component boundaries (each swappable independently, registries below):
  Encoder    : boards -> phi                    (catspace.field.ReachabilityField)
  ReachModel : (phi, goals, ctx) -> P(hit), plies   (probe.reach.RegionReach)
  Atlas      : goal bank + region stats             (probe.atlas.RegionAtlas)
  Planner    : phi_now -> target chain + instruments (probe.planner.*, PLANNERS)
  Navigator  : (board, target) -> move               (probe.navigator.MCTSNavigator)
  OppModel   : board -> {move: prior} | None         (probe.navigator.make_maia2_policy)
  Harness    : games vs an engine + VERDICT lines    (probe.harness.run_games)

The registries are the iteration surface: add an implementation, register it,
select it from the CLI. Nothing here is load-bearing on being "the" algorithm.
"""
from catspace.probe.atlas import RegionAtlas                       # noqa: F401
from catspace.probe.planner import PLANNERS, ChutePlanner          # noqa: F401
from catspace.probe.navigator import MCTSNavigator, make_maia2_policy  # noqa: F401
from catspace.probe.reach import RegionReach                       # noqa: F401
from catspace.probe.harness import run_games                       # noqa: F401
