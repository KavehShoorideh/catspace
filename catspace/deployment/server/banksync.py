#!/usr/bin/env python
"""banksync -- push the online mate/loss/draw banks into Qdrant.

    python -m catspace.deployment.server.banksync

Runs as its own compose service so the engine container never blocks on a bulk embed.
Previously this was an inline `python -c` in docker-compose.yml with literal repo-relative
paths and a `sys.path.insert`; both are gone -- the checkpoint and bank prefix come from
the bootstrap_mate config and the path registry, so the container and a local run agree.
"""
from __future__ import annotations

import argparse

from catspace.approaches.bootstrap_mate import config as bootstrap_mate_config
from catspace.io import paths
from catspace.research.components.memory.approaches.vector_store_retrieval.src.vectordb import (
    sync_banks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default=bootstrap_mate_config.field_checkpoint(),
                    help="field checkpoint used to embed the bank exemplars")
    ap.add_argument("--banks-prefix", default=str(paths.experiments_dir() / "assistant"),
                    help="prefix of the *_bank.fens / *_lossbank.fens / *_drawbank.fens files")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    counts = sync_banks(args.field, args.banks_prefix, device=args.device, batch=args.batch)
    for name, n in sorted(counts.items()):
        print(f"{name}: {n}", flush=True)


if __name__ == "__main__":
    main()
