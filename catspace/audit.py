"""
audit.py — static + provenance leakage guard: certifies that a TorchFB
checkpoint (and hence anything built on it -- FBBoardPolicy, the M1.5
decomposer) was never trained or fine-tuned on Stockfish-derived signal
(shard eval_cp, sf_labels.npz, or any winprob_cp-based loss). This is the
one invariant experiments/experiment_report.py refuses to proceed without --
see nn/eval_head.py's --joint flag (fine-tunes F on the normative/Stockfish
loss; off by default, and even when used, train_eval_heads.py never writes
the fine-tuned F back to a checkpoint the planner could load) for the
existing design decision this module makes auditable instead of just implicit.

Two independent checks, because provenance metadata can be stale or absent
(older checkpoints) while the code itself is the ground truth:

  static_purity_check()          inspects the SOURCE of the exact functions
                                  that feed TorchFB.loss_fn / save_ckpt (the
                                  FB training path) and the planner's read
                                  path (FBBoardPolicy, planner.decompose) for
                                  any reference to Stockfish-derived
                                  identifiers.
  checkpoint_provenance_check()  reads the checkpoint's own `provenance`
                                  dict (stamped by train_lichess_fb.py at
                                  every save, see nn/fb.py::save_ckpt) and
                                  reports clean / unknown / contaminated.

audit_checkpoint() runs both and combines them into one clean: bool that
experiment_report.py treats as a hard gate, not a warning.
"""
from __future__ import annotations

import inspect
from typing import Callable

# case-insensitive substring match against each target function's OWN source
# text (not its callees) -- deliberately crude: a false positive (e.g. a
# comment mentioning "stockfish") just means a human re-reads that function,
# which is the correct failure mode for a safety gate.
FORBIDDEN = ("eval_cp", "winprob_cp", "sf_label", "stockfish", "wdl_w", "wdl_d", "wdl_l")


def _scan(fn: Callable) -> list[str]:
    src = inspect.getsource(fn).lower()
    return [tok for tok in FORBIDDEN if tok in src]


def static_purity_check(train_batch_fn: Callable | None = None,
                        train_main_fn: Callable | None = None) -> dict:
    """Re-inspects, at call time, the actual source of the FB-training path
    and the planner's read path -- so a future edit that starts reading
    eval_cp into the FB loss (or into FBBoardPolicy/decompose) fails this
    check automatically, without anyone remembering to update an audit.

    train_batch_fn/train_main_fn: pass these explicitly when calling from
    INSIDE experiments/train_lichess_fb.py itself (it already has its own
    `batch_tensors`/`main` in scope) -- importing `experiments.train_lichess_fb`
    by name from within that same script's own execution would re-import the
    file under a second module name (`__main__` vs `experiments.train_lichess_fb`)
    instead of reusing the running one. External callers (experiment_report.py,
    tests) omit these and the module is imported normally."""
    if train_batch_fn is None or train_main_fn is None:
        import experiments.train_lichess_fb as train_mod
        train_batch_fn = train_batch_fn or train_mod.batch_tensors
        train_main_fn = train_main_fn or train_mod.main

    from catspace.nn.fb import TorchFB
    from catspace.nn.policy_fb import FBBoardPolicy
    from catspace.planner import decompose as decompose_mod

    targets = [
        (train_batch_fn, "train_lichess_fb.batch_tensors (what the FB loss actually sees)"),
        (train_main_fn, "train_lichess_fb.main (the training loop)"),
        (TorchFB.loss_fn, "TorchFB.loss_fn"),
        (TorchFB.embed_F, "TorchFB.embed_F"),
        (TorchFB.embed_B, "TorchFB.embed_B"),
        (FBBoardPolicy.move_scored, "FBBoardPolicy.move_scored (the planner's read path)"),
        (decompose_mod.decompose, "planner.decompose.decompose"),
        (decompose_mod.waypoint_scores, "planner.decompose.waypoint_scores"),
    ]
    hits = {}
    for fn, label in targets:
        found = _scan(fn)
        if found:
            hits[label] = found
    return dict(clean=not hits, hits=hits, checked=[label for _, label in targets])


def checkpoint_provenance_check(payload: dict) -> dict:
    prov = payload.get("provenance")
    if not prov:
        return dict(status="unknown",
                   detail="checkpoint has no provenance stamp (pre-audit checkpoint); "
                          "relying on static_purity_check alone")
    if prov.get("stockfish_free") is True:
        return dict(status="clean", detail=prov)
    return dict(status="contaminated", detail=prov)


def audit_checkpoint(payload: dict) -> dict:
    """The combined gate: clean requires BOTH the static check (the code as
    it exists right now) and the checkpoint's own provenance (unknown is
    tolerated -- older checkpoints predate the provenance stamp -- but an
    explicit stockfish_free=False is not)."""
    static = static_purity_check()
    prov = checkpoint_provenance_check(payload)
    clean = bool(static["clean"] and prov["status"] in ("clean", "unknown"))
    return dict(clean=clean, static=static, provenance=prov)


def git_commit() -> str | None:
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5, cwd=__file__.rsplit("/", 2)[0])
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def build_provenance(script: str, args: dict, data_columns_used: list,
                     train_batch_fn: Callable | None = None,
                     train_main_fn: Callable | None = None) -> dict:
    """Called by train_lichess_fb.py at every save. The purity flag is the
    OUTPUT of static_purity_check() against the running code, not a literal
    True -- so it self-corrects if the training path ever changes."""
    static = static_purity_check(train_batch_fn, train_main_fn)
    return dict(script=script, args=args, data_columns_used=list(data_columns_used),
               stockfish_free=bool(static["clean"]), static_check=static, git_commit=git_commit())


def is_provenance_clean(provenance: dict) -> bool:
    """Split out from build_provenance/checkpoint_provenance_check on purpose:
    a caller like train_lichess_fb.py's own main() must not need to spell the
    forbidden word out in its own source to check this, or static_purity_check
    would flag that check as a hit against itself (it did, once)."""
    return bool(provenance.get("stockfish_free"))
