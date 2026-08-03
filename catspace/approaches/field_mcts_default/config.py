"""field_mcts_default -- the incumbent wiring, expressed as an end-to-end config.

Reproduces what the engine did before the 2026-08-03 restructure: a trained
reachability field supplies the leaf value, subgoal regions shape the move prior
only (never the value), PUCT search runs the plan phase, and the finisher runs
pure (no value) because the field hurts near mate.
"""
from __future__ import annotations

from catspace.approaches.wiring import EndToEndConfig

CONFIG = EndToEndConfig(
    name="field_mcts_default",
    planner="planner:subgoal_cascade",
    searches=["search:puct_mcts"],
    encoders=["encoder:reachability_field"],
    memories=["memory:vector_store_retrieval"],
    params=dict(
        plan_nodes=400,
        execute_nodes=400,
        handoff_pieces=5,
        stall_patience=3,
        prior_alpha=0.7,
        # EXECUTE stays pure: the field value degrades play near mate.
        execute_value=None,
    ),
    notes="incumbent engine wiring carried over from the pre-restructure LayeredEngine",
)


def build(field_ckpt=None, device: str = "cpu", subgoals=None, **overrides):
    """Assemble the LayeredEngine for this config.

    field_ckpt=None gives a pure-search engine (no value), which is the honest
    default when no trained checkpoint is supplied.
    """
    from catspace.engine_core import LayeredEngine
    from catspace.research.components.search.approaches.puct_mcts.src.layer import MCTSSearch

    p = {**CONFIG.params, **overrides}
    value = None
    if field_ckpt is not None:
        from catspace.fields import FieldModel
        from catspace.values import FieldGoalDistanceValue
        value = FieldGoalDistanceValue(FieldModel(field_ckpt, device=device))

    prior = None
    if subgoals is not None:
        from catspace.priors import MixturePrior
        prior = MixturePrior(subgoals, alpha=p["prior_alpha"])

    return LayeredEngine(
        value=value,
        subgoals=subgoals,
        prior=prior,
        plan_search=MCTSSearch(nodes=p["plan_nodes"]),
        execute_search=MCTSSearch(nodes=p["execute_nodes"]),
        execute_value=p["execute_value"],
        handoff_pieces=p["handoff_pieces"],
        stall_patience=p["stall_patience"],
    )
