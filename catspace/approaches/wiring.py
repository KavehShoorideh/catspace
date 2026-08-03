"""Shared wiring vocabulary for end-to-end approaches."""
from __future__ import annotations

from dataclasses import dataclass, field

from catspace import registry


@dataclass
class EndToEndConfig:
    """Which approach fills each component slot, and with what arguments.

    Slots hold "component:approach" specs. `searches`, `encoders` and `memories`
    are lists because a config may compose several (e.g. a plan search plus a
    finisher search); `planner` is single by construction -- one agent, one intent.
    """
    name: str
    planner: str | None = None
    searches: list[str] = field(default_factory=list)
    encoders: list[str] = field(default_factory=list)
    memories: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    notes: str = ""

    def validate(self) -> None:
        for slot, specs in (("planner", [self.planner] if self.planner else []),
                            ("searches", self.searches),
                            ("encoders", self.encoders),
                            ("memories", self.memories)):
            for spec in specs:
                component, _, approach = spec.partition(":")
                if component not in registry.COMPONENTS:
                    raise ValueError(f"{self.name}.{slot}: unknown component in {spec!r}")
                available = registry.list_approaches(component)
                if approach not in available:
                    raise LookupError(
                        f"{self.name}.{slot}: no approach {spec!r}; available: {available}")

    def describe(self) -> str:
        lines = [f"end-to-end approach: {self.name}"]
        for slot in ("planner", "searches", "encoders", "memories"):
            v = getattr(self, slot)
            lines.append(f"  {slot:9s} {v if v else '-'}")
        if self.notes:
            lines.append(f"  notes     {self.notes}")
        return "\n".join(lines)
