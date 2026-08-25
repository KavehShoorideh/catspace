"""The model, and the contract its heads must satisfy.

Kaveh 2026-08-25: "I want an abstract definition of the architecture so that it's clear
where the interfaces are and what's being fed into what" -- and then, on being shown
that the definition lived only in prose: "please do the refactor".

The point of this module is HEADS below. A head used to be whatever its author wired up,
and twice that quietly meant raw pre-quantisation trunk features: sf_av read h_mv until
--av-concepts, and move_logits read it until --mv-concepts. Both were found by hand,
months apart, and the second only because Kaveh asked. Nothing in the code objected.

Now each head DECLARES what it reads, from a closed set that contains no
pre-quantisation option, and check_head_widths() verifies the built module's actual
input width equals what its declaration implies. A head that quietly gains a raw feature
changes its width and fails the check. The startup audit line is generated from the same
table, so the log cannot drift from the code.

The contract in prose, with shapes, is docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import copy as _copy
from dataclasses import dataclass
from enum import Enum

import torch.nn as nn

from catspace.research.tools.training_infra.koopman_ops import (
    InteractionLift, KoopmanOps, BasisOps, TransformerLift)


class HeadInput(Enum):
    """What a head is permitted to read.

    There is deliberately NO member for pre-quantisation trunk features (h_mv, h_sq,
    pre_g). That absence is the enforcement: a head cannot declare something this enum
    cannot express, and check_head_widths() catches anything that reads it anyway.
    """
    CONCEPTS = "concepts"
    CONCEPTS_MOVE = "concepts | quantised move code"
    LIFTED = "concepts | phi"
    LIFTED_MOVE = "lifted | discrete move-descriptor embedding"


@dataclass(frozen=True)
class HeadSpec:
    name: str
    source: HeadInput
    what: str
    owner: str = "model"          # "field" for heads that live on the encoder


HEADS = (
    HeadSpec("move_head",  HeadInput.CONCEPTS_MOVE, "Maia policy", owner="field"),
    HeadSpec("sf_av",      HeadInput.CONCEPTS_MOVE, "per-move Stockfish value"),
    HeadSpec("fkind_head", HeadInput.CONCEPTS_MOVE, "which kind of future a move opens"),
    HeadSpec("iv",         HeadInput.CONCEPTS_MOVE, "interval bounds"),
    HeadSpec("wdl",        HeadInput.CONCEPTS,      "WDL, shared by root and rolled"),
    HeadSpec("sf_v_full",  HeadInput.LIFTED,        "WDL from the lifted state"),
    HeadSpec("av_roll",    HeadInput.LIFTED_MOVE,   "per-move value from rolled states"),
)


def _declared_width(spec: HeadSpec, field, lift, args) -> int | None:
    d_state = field.d_state
    mv = getattr(field, "mv_code_dim", 32)
    if spec.source is HeadInput.CONCEPTS:
        return d_state
    if spec.source is HeadInput.CONCEPTS_MOVE:
        return d_state + mv
    if lift is None:
        return None
    if spec.source is HeadInput.LIFTED:
        return lift.d_aug
    if spec.source is HeadInput.LIFTED_MOVE:
        return lift.d_aug + 64
    return None


def _first_linear(m):
    if isinstance(m, nn.Linear):
        return m
    for sub in getattr(m, "children", lambda: [])():
        got = _first_linear(sub)
        if got is not None:
            return got
    return None


def _is_alias(model, spec) -> bool:
    """True when this name is another name for a head already covered.

    With --linear-heads, sf_v_full and sf_v are bound to the SAME _ConceptWDL object as
    wdl. They are not separate heads and must not be checked against their historical
    source, or the contract reports a violation for an architecture that is correct.
    """
    if spec.name == "wdl":
        return False
    shared = getattr(model, "wdl", None)
    return shared is not None and getattr(model, spec.name, None) is shared


def check_head_widths(model, field, args) -> list[str]:
    """Every head's ACTUAL input width must equal what its declared source implies.

    Returns a list of human-readable violations; empty means the architecture matches
    its own declaration. This is the check that would have caught both historical
    breaches on the day they were written, instead of months later by inspection.
    """
    lift = getattr(model, "lift", None)
    out = []
    for spec in HEADS:
        owner = field if spec.owner == "field" else model
        head = getattr(owner, spec.name, None)
        if head is None or _is_alias(model, spec):
            continue                       # not built, or another name for `wdl`
        lin = _first_linear(head)
        want = _declared_width(spec, field, lift, args)
        if lin is None or want is None:
            continue
        got = lin.in_features
        if got != want:
            out.append(f"{spec.name}: declares {spec.source.value} ({want} dims) but "
                       f"its first Linear takes {got} -- {got - want:+d} unexplained "
                       f"dims, which is what reading a raw feature looks like")
    return out


def describe_heads(model, field, args) -> str:
    """The startup audit line, GENERATED from the table so it cannot drift."""
    lift = getattr(model, "lift", None)
    parts = []
    for spec in HEADS:
        owner = field if spec.owner == "field" else model
        head = getattr(owner, spec.name, None)
        if head is None or _is_alias(model, spec):
            continue
        lin = _first_linear(head)
        kind = "linear" if isinstance(head, nn.Linear) or (
            spec.name == "wdl") else "mlp"
        w = lin.in_features if lin is not None else "?"
        parts.append(f"{spec.name}[{spec.source.value}; {w}d; {kind}]")
    return "[gko0] heads -- " + " | ".join(parts)


def build_gko(field, args, A) -> nn.Module:
    """Assemble the model. Widths come from `field` and the lift, never from literals."""
    class GKO(nn.Module):
        def __init__(self):
            super().__init__()
            self.field = field
            # CONCEPTS-ONLY (Kaveh 2026-08-24: "I don't wanna read off of the raw
            # board or the encoded board, I wanna read off of the concepts"). The
            # sharp head was the ONE head still reading h_mv -- raw backbone output,
            # never quantised -- while the policy, value and rolled heads all read
            # quantised codes. --av-concepts feeds it the QUANTISED move code (q_mv,
            # 32-d) instead, so every head in the model reads concepts.
            # same literal-32 trap as move_head had: read the field's real width
            _av_mv = (getattr(field, "mv_code_dim", 32) if args.av_concepts
                      else A["d_model"])
            self.sf_av = (nn.Linear(field.d_state + _av_mv, args.k_bins)
                          if args.linear_heads else
                          nn.Sequential(nn.Linear(field.d_state + _av_mv, 512),
                                        nn.GELU(), nn.Linear(512, args.k_bins)))
            if args.linear_heads:
                # ONE linear map, concepts -> WDL, shared by the root state and by
                # operator-rolled states. Rolled concepts occupy the same first
                # d_state columns of the lifted vector, so the SAME head reads both:
                # "rolled-forward concepts must give the WDL Stockfish gives" becomes
                # true by construction instead of by two heads happening to agree.
                class _ConceptWDL(nn.Module):
                    def __init__(self, d_state, n_out=3):
                        super().__init__()
                        self.d_state = d_state
                        self.lin = nn.Linear(d_state, n_out)
                    def forward(self, x):
                        return self.lin(x[..., :self.d_state])
                self.wdl = _ConceptWDL(field.d_state)
            if args.w_fkind > 0:
                # WHICH KIND OF FUTURE does this move open. Reads exactly what sf_av
                # reads -- quantised state concepts plus the quantised move code --
                # per the concepts-only ruling: no head sees the unquantised encoding.
                # Its targets are frozen integers computed offline from a reference
                # field and Stockfish's own continuations, so unlike a
                # predict-your-own-future objective there is no collapse solution:
                # the encoder cannot make its targets easier.
                self.fkind_head = nn.Sequential(
                    nn.Linear(field.d_state + _av_mv, 512), nn.GELU(),
                    nn.Linear(512, args.fkind_K))
            if args.ops:
                if args.lift == "transformer":
                    # EVERY width read from the field. Passing literals here made the
                    # lift slice the state at the wrong offsets under --binary (board
                    # read as 256 dims when it is 96), so its interaction tokens were
                    # misaligned garbage -- and nothing crashed, because 2400 dims is
                    # enough to slice from. Tenth literal-width bug in two days.
                    self.lift = TransformerLift(
                        field.d_state, g_heads=A["g_heads"],
                        d_code=field.code_width,
                        d_sq=getattr(field, "sq_width", 16 if args.gen5 else 32),
                        d_phi=max(args.d_phi, 256),
                        n_layers=args.lift_layers,
                        n_piece=(32 if args.gen5 else 0),
                        d_piece=getattr(field, "d_piece", 4),
                        d_mv=getattr(field, "mv_code_dim", 32))
                else:
                    self.lift = InteractionLift(field.d_state, args.d_phi)
                if args.gen5:
                    self.ops = BasisOps(self.lift.d_aug, n_basis=args.op_bases,
                                        rank=args.op_rank,
                                        d_phi=(self.lift.d_aug - self.lift.d_state)
                                        if args.state_mix else 0)
                else:
                    self.ops = KoopmanOps(self.lift.d_aug, n_codes=A["move_codes"],
                                          rank=args.op_rank)
                # H1's baseline arm: same targets, DETACHED inputs (see docstring)
                self.dyn_z = nn.Sequential(
                    nn.Linear(field.d_state + 64, 512), nn.GELU(),
                    nn.Linear(512, field.d_state))
            if args.iv:
                self.iv = nn.Sequential(
                    nn.Linear(field.d_state + getattr(field, "mv_code_dim", 32),
                              256), nn.GELU(), nn.Linear(256, 3))
            if args.ops and args.ema_decay > 0:
                import copy as _copy
                self.ema_field = _copy.deepcopy(field).eval()
                for pr in self.ema_field.parameters():
                    pr.requires_grad_(False)
            if args.ops:
                # PHI-ONLY value head (the saturation falsification: heads reading
                # full z_aug route AROUND phi; value-at-rolled-horizons must be
                # impossible WITHOUT interaction features carrying it)
                v_out = 3 if args.gen5 else args.k_bins
                self.sf_v = nn.Sequential(
                    nn.Linear(self.lift.d_phi, 256), nn.GELU(),
                    nn.Linear(256, v_out))
                # FULL value head for SEARCH (2026-08-22 maia autopsy: phi-only sees
                # ~quarter of a removed queen -- material lives in the state part;
                # the phi-only head REMAINS the anti-collapse anchor, this one plays)
                self.sf_v_full = nn.Sequential(
                    nn.Linear(self.lift.d_aug, 256), nn.GELU(),
                    nn.Linear(256, 3 if args.gen5 else args.k_bins))
                # per-move future action-values from ROLLED states: [z_aug ; mv_emb]
                self.av_roll = (nn.Linear(self.lift.d_aug + 64, args.k_bins)
                                if args.linear_heads else
                                nn.Sequential(nn.Linear(self.lift.d_aug + 64, 512),
                                              nn.GELU(),
                                              nn.Linear(512, args.k_bins)))

            if args.linear_heads:
                # bound LAST: sf_v_full/sf_v are constructed further down in the ops
                # block, so binding earlier let them overwrite the shared head and the
                # value path silently reverted to the MLP
                self.sf_v_full = self.wdl
                self.sf_v = self.wdl

        def train(self, mode=True):
            # model.train() recurses into ALL submodules -- without this override the
            # EMA target encoder flips to train mode and its VQ codebooks self-update
            # from every child-target batch (the drift EMA exists to prevent)
            super(GKO, self).train(mode)
            if hasattr(self, "ema_field"):
                self.ema_field.eval()
            return self

    return GKO()
