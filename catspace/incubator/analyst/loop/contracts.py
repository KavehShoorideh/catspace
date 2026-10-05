"""THE RECORDS EVERY STAGE READS AND WRITES. JSON-serializable dataclasses, each with its provenance (round, the
ids of the inputs it came from, the code version). A research piece that replaces a stage must read and write these.

    Framed       a position where the player went wrong
    Guard        a conjunction of relation literals over frame squares, its claim and its evidence (lemma_gen)
    Candidate    an Idea in the handoff plan's schema: named hypotheses, footprint, protagonist, bounded claim
    Promoted     a candidate with a tier and the evidence for it
    UseRecord    how an idea is wired into the player (closure / target / order / sentence), or why it was refused
    Measurement  per set and arm: fire rate, paired nodes, solved, gained and lost, wall time
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass

TIERS = ("unconditional", "forcing", "conjecture")
USES = ("closure", "target", "order", "sentence")
CLAIM_KINDS = ("mate", "material", "promotion", "safety")
STANDARD_AXIOMS = ("propext", "Classical.choice", "Quot.sound")


@dataclass
class Provenance:
    round: int
    stage: str
    sources: list = field(default_factory=list)
    code: str = ""


@dataclass
class Framed:
    id: str                      # "puzzle:<PuzzleId>:<step>" | "game:<game_id>:<ply>"
    fen: str                     # the position before the move
    played: str                  # the move the player (or the human) chose, UCI
    better: str | None           # the move it should have chosen, when known
    loss: float | None           # what the choice cost, in pawns, when measured
    label: str                   # how the loss was labelled: "puzzle-solution" | "game-continuation"
    frame: list                  # the frame squares (names)
    source: str                  # "puzzle" | "game"
    detail: dict
    provenance: Provenance


@dataclass
class Guard:
    id: str                      # lemma_gen's lemma id
    text: str
    types: list
    literals: list               # [[relation, slots, polarity]]
    terms: list                  # the full guard: [[[relation, slots, polarity], ...], ...]
    slot: int
    claim: str                   # "acting loses" | "acting keeps"
    value: dict
    support: int
    holds: int
    purity: float
    held_out: dict               # lemma_gen's re-test: {"status": "PROVED" | "THEOREM" | "conjecture" | None, ...}
    library: str
    index: int
    provenance: Provenance


@dataclass
class Claim:
    kind: str                    # one of CLAIM_KINDS
    horizon: int                 # plies
    value: float                 # pawns (material, promotion); 0 for mate and safety
    text: str


@dataclass
class Candidate:
    id: str
    hypotheses: list             # [{"name": ..., "text": ..., "lean": None | "Lean definition"}]
    footprint: str               # how the footprint is computed for an instance
    protagonist: str
    claim: Claim
    guard: str | None            # the Guard it came from (None: registered by hand)
    lean: dict | None            # {"theorem": ..., "detector": ..., "tier": ...} when a Lean statement exists
    source: str                  # "mined" | "registered"
    sentence: str                # the explainer's template
    provenance: Provenance
    origin: dict | None = None   # where a mined idea lives: {"library": path, "index": i} (lemma_apply compiles it from there)


@dataclass
class Evidence:
    theorem: str | None = None
    axioms: list | None = None
    axioms_ok: bool | None = None
    bridge: dict | None = None   # {"checked": n, "wrong": k, "parts": {...}}
    held_out: dict | None = None


@dataclass
class Promoted:
    candidate: Candidate
    tier: str
    evidence: Evidence
    unsound: bool                # a Lean statement was claimed but its axioms or its bridge failed
    reason: str
    provenance: Provenance


@dataclass
class UseRecord:
    idea: str
    mode: str                    # one of USES
    switch: str                  # how it is wired
    refused: bool
    reason: str
    provenance: Provenance


@dataclass
class Measurement:
    set: str                     # "backrank" | "pawnendings" | "general" | ...
    arm: str                     # "baseline" | "round" | an idea id
    dump: str
    puzzles: int
    steps: int
    mean_nodes: float
    tied: int
    single: int
    wall_s: float | None
    paired: dict | None          # against the baseline, by position: before, after, ratio, fewer, more, equal
    gained: list
    lost: list
    provenance: Provenance


def to_dict(obj):
    return asdict(obj) if is_dataclass(obj) else obj


def _build(cls, d):
    if d is None or is_dataclass(d):
        return d
    kw = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
        sub = {"Provenance": Provenance, "Claim": Claim, "Candidate": Candidate, "Evidence": Evidence}.get(t.split(" ")[0])
        kw[f.name] = _build(sub, v) if sub is not None and isinstance(v, dict) else v
    return cls(**kw)


def from_dict(cls, d):
    """The inverse of to_dict for every record above (nested records included)."""
    return _build(cls, d)
