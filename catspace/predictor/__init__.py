"""Predictor component: everything that predicts futures, outcomes, and opponents —
reach/hazard fields, atlas statistics, committor value oracles, opponent move
models, endgame ground truth (tablebases, DTM, material)."""
from catspace.predictor.reach import ReachHead, RegionReach            # noqa: F401
from catspace.predictor.atlas import SubgoalRanker, RegionAtlas        # noqa: F401
from catspace.predictor.value import ClockField, CommittorGreedy       # noqa: F401
from catspace.predictor.opponent import make_maia2_policy              # noqa: F401
from catspace.predictor.endgame import mat_sig                         # noqa: F401
