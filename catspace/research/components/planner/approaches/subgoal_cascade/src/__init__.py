"""Subgoal cascade: the planner owns WHERE to go, the navigator owns HOW.

Deliberately NOT re-exported here: `decompose`. The submodule and its main function share
a name, so `from ...src import decompose` would hand back the function and shadow the
module -- and audit.py wants the module (it reflects over several of its functions).
Import it as `...src.decompose` and take the function from there.
"""
from catspace.research.components.planner.approaches.subgoal_cascade.src.chute import ChutePlanner, PLANNERS  # noqa: F401,E402
