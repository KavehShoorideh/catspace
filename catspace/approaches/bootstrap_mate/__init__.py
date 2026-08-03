"""bootstrap_mate -- the self-bootstrapping mate engine, as an end-to-end approach.

Both deployment shells consume this: `catspace.deployment.server.uci_engine` (UCI, for
cutechess/fastchess) and `catspace.deployment.server.assistant_server` (the HTTP
co-analyst). Neither reaches into a research component directly.
"""
from catspace.approaches.bootstrap_mate.src import (  # noqa: F401
    MilestoneCache,
    OnlineMateBank,
    harvest,
    make_batched_energy_prior,
    make_boot_value,
    make_planner,
    mat_sig,
    tb_white_move,
)
