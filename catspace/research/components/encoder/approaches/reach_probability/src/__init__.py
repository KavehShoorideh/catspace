"""reach_probability -- can position a lead to position b, and if we cannot say yes, can we
bound the probability that it does?

See src/probability_less_than.py for the shipped predicate and src/reach_jepa.py for the model.
"""
from catspace.research.components.encoder.approaches.reach_probability.src.probability_less_than import (  # noqa: F401
    ReachVerdict,
    ReachPredicate,
)
from catspace.research.components.encoder.approaches.reach_probability.src.reach_jepa import (  # noqa: F401
    ReachJEPA,
)
