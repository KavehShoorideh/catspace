"""bootstrap_mate -- the self-bootstrapping mate engine, expressed as an end-to-end config.

The engine starts with no mate bank: PUCT search over the reachability field discovers
checkmate leaves, harvests them into an online experience store, and the value becomes
distance-to-DISCOVERED-mates. Tablebase ground truth is a *logged fallback* only, never
consulted at play.

This is the wiring both deployment shells build from -- see build_engine().
"""
from __future__ import annotations

from pathlib import Path

from catspace.approaches.wiring import EndToEndConfig
from catspace.io import paths

CONFIG = EndToEndConfig(
    name="bootstrap_mate",
    planner="planner:endgame_groundtruth",
    searches=["search:puct_mcts"],
    encoders=["encoder:reachability_field"],
    memories=["memory:experience_store"],
    params=dict(
        max_nodes=64,          # per search chunk; the shells loop until their clock is spent
        pw_c=1.5,
        root_min_visits=10,
        batch_leaves=32,
        mate_stop=True,
        banks_prefix="assistant",
        device="cpu",
    ),
    notes="no external mate bank; own-experience only, tablebase is a logged fallback",
)


def _pointer_or_default(pointer: str, default: str) -> str:
    """Checkpoint pointers are text files holding the current path; fall back to the
    published default when a training run has not written one yet."""
    ptr = paths.sep_dir() / pointer
    if ptr.exists():
        return ptr.read_text().strip()
    return str(paths.sep_dir() / default)


def field_checkpoint() -> str:
    return _pointer_or_default("self_field_current.txt", "lichess_mc2.pt")


def opponent_energy_checkpoint() -> str:
    return _pointer_or_default("opponent_energy_current.txt", "opponent_energy_v1.pt")


def build_engine(field_ckpt: str | None = None, device: str | None = None,
                 banks_prefix: str | None = None, **overrides) -> dict:
    """Assemble the shared engine state: field model, the three online banks, the value
    and prior closures, and the planner.

    Returns a dict rather than an object because the shells thread the same pieces into
    different loops (a UCI `go`, an HTTP analysis request) and both need the banks by name.
    """
    from catspace.approaches.bootstrap_mate.src import (OnlineMateBank, make_batched_energy_prior,
                                                        make_boot_value, make_planner)
    from catspace.fields import FieldModel

    p = {**CONFIG.params, **overrides}
    fm = FieldModel(field_ckpt or field_checkpoint(), device=device or p["device"])

    prefix = paths.experiments_dir() / (banks_prefix or p["banks_prefix"])
    bank = OnlineMateBank(fm, Path(str(prefix) + "_bank.fens"))
    loss = OnlineMateBank(fm, Path(str(prefix) + "_lossbank.fens"))
    draw = OnlineMateBank(fm, Path(str(prefix) + "_drawbank.fens"))
    for bk in (bank, loss, draw):
        bk.sync()

    ctx = {"plan": "direct", "hist": {}}
    times: dict = {}
    vfn = make_boot_value(fm, bank, times, loss, draw_bank=draw, game_ctx=ctx)
    pfn, pfnb = make_batched_energy_prior(opponent_energy_checkpoint(), game_ctx=ctx)

    return dict(fm=fm, bank=bank, loss=loss, draw=draw, ctx=ctx, times=times,
                vfn=vfn, pfn=pfn, pfnb=pfnb, planner=make_planner(fm, bank), params=p)


# `catspace.approaches.load("bootstrap_mate").build()` -- the uniform entrypoint.
build = build_engine
