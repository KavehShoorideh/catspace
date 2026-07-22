"""catspace/engine -- the LAYERED engine (Kaveh 2026-07-23: "layered, try different
models in each layer"). Each layer is a Protocol (interfaces.py); implementations are
constructor-injected into LayeredEngine, so any layer swaps independently:

    field      FieldModel (fields.py)          which learned map: cooperative / human / contrast
    value      ValueModel (values.py)          leaf evaluation: constant / DTM-CNN / field-goal-distance
                                               (TablebaseValue exists but is DIAGNOSTIC-ONLY)
    prior      MovePrior (priors.py)           uniform / alpha-mixture (subgoal-focused + global)
    subgoals   SubgoalSelector (interfaces)    region goals -- heuristic now, RL later (the seam)
    search     MCTSSearch (search.py)          the local searcher (policy_fn/value_fn sockets)
    engine     LayeredEngine (engine.py)       composition + phase logic (plan -> execute handoff)

Related, predating this package and slated to fold in: experiments/compute_layer.py
(uncertainty-carrying tool layer) + experiments/catspace_engine.py (coded policy on top).
"""
from catspace.engine.interfaces import MovePrior, Region, SearchOutcome, SubgoalSelector, ValueModel
from catspace.engine.fields import FieldModel
from catspace.engine.values import ConstantValue, DTMCNNValue, FieldGoalDistanceValue, TablebaseValue
from catspace.engine.priors import MixturePrior, UniformPrior
from catspace.engine.search import MCTSSearch
from catspace.engine.engine import LayeredEngine

__all__ = ["Region", "SearchOutcome", "ValueModel", "MovePrior", "SubgoalSelector",
           "FieldModel", "ConstantValue", "DTMCNNValue", "FieldGoalDistanceValue",
           "TablebaseValue", "UniformPrior", "MixturePrior", "MCTSSearch", "LayeredEngine"]
