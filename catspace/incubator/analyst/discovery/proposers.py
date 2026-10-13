"""THE LEARNED PROPOSERS, KEPT AND UPDATED (Kaveh 2026-10-12: "is there some incremental training we could do?"; "keep both
the CPU and GPU busy at the same time and not waste resources"). Until now every hole retrained the graph network and the
transformer from scratch (25-35 s each), and only on the old labeled holes -- the loop's own verdicts never reached them.

  CHECKPOINTS  data/derived/proposers/<kind>.pt, and how far into the experience store they have trained
  FIRST        trained once on the labeled holes (the earlier experience), then kept
  UPDATE       after each hole, IN A BACKGROUND PROCESS (two threads, beside the verifier's and the games' workers): the
               store's new verdicts -- winners and losers, every defender -- fine-tune both models a few epochs; the next
               hole ranks with the newest checkpoint written (a hole never waits for training)
  DEVICE       the CPU: these models are tiny -- on the GPU (MPS) the graph network ran 20 ms a batch against 5.6 ms on the
               CPU (2026-10-12); the GPU earns its place when the proposers grow
"""
from __future__ import annotations

import json
import os

from catspace.io import paths

KINDS = ("gnn", "transformer")
FIRST_EPOCHS, UPDATE_EPOCHS = 40, 5


def _dir():
    d = paths.derived("proposers")
    os.makedirs(d, exist_ok=True)
    return d


def _ckpt(kind: str) -> str:
    from .models import CTX
    from .base_vocabulary import SYMBOLS
    # the context's width and the vocabulary's size in the name: a new feature or new words, a new model
    return os.path.join(_dir(), f"{kind}_ctx{CTX}_v{len(SYMBOLS)}.pt")


def _state_path() -> str:
    return os.path.join(_dir(), "state.json")


def _state() -> dict:
    p = _state_path()
    return json.load(open(p)) if os.path.exists(p) else {"offset": 0}


def _complete_lines(p: str) -> list:
    """The file's COMPLETE lines: a line the loop is still appending (no newline yet) is left for the next read -- the
    background update reads while the main process writes."""
    if not os.path.exists(p):
        return []
    return open(p).read().split("\n")[:-1]


def _store_lines() -> list:
    from .verifier import STORE
    return _complete_lines(paths.derived(STORE))


def get(kind: str):
    """The newest checkpoint; the first time, trained on the labeled holes (blocking, once)."""
    from . import models as M
    if os.path.exists(_ckpt(kind)):
        return M.load(kind, _ckpt(kind))
    from . import holes as H
    from .pipeline import train_on
    trained = [n for n in H.available() if os.path.exists(paths.derived(f"hole_{n}_labels.json"))]
    m = train_on(trained, kind, {}, FIRST_EPOCHS)
    M.save(m, _ckpt(kind))
    st = _state()
    st["offset"] = max(st.get("offset", 0), len(_store_lines()))     # the labels' verdicts are in the store already
    json.dump(st, open(_state_path(), "w"))
    print(f"OBS   the {kind} proposer trained once on {len(trained)} labeled holes -> {_ckpt(kind)}")
    return m


def update(epochs: int = UPDATE_EPOCHS) -> int:
    """The store's verdicts since the last update fine-tune both models; -> how many rows."""
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from . import models as M
    lines = _store_lines()
    st = _state()
    # THE REWARD (Kaveh 2026-10-12: "do we feed this back to the proposer?"): a real defender's candidates are learned by
    # their ADVANTAGE over the frame before it (verifier.advantage: the screening's boards, the confirmation's held-out
    # games) -- not their raw score, which mixes the hole's difficulty in and, on the boards that chose them, was mostly
    # noise (2026-10-12: 8 of 50 held-out boards held, like the frame's own 8); the 'self' defender's verdicts as before
    adv = _adv_lines()
    new = [json.loads(l) for l in lines[st.get("offset", 0):]]
    new = [r for r in new if r.get("defender", "self") == "self"]
    new += [dict(json.loads(l), score=json.loads(l)["adv"]) for l in adv[st.get("adv_offset", 0):]]
    rows, ctx, cache, holes = [], [], {}, {}
    for r in new:
        name = r["hole"]
        if name not in holes:
            p = paths.derived(f"hole_{name}.json")
            holes[name] = json.load(open(p)) if os.path.exists(p) else None
        h = holes[name]
        if h is None or not r["term"]:
            continue
        boards = h["stuck"][:100] if "adv" in r else h["stuck"][:r.get("n", 100)]   # (an advantage's n may count games)
        rows.append(dict(term=r["term"], schema=r["schema"], score=r["score"], hole=name))
        ctx.append(M.context_features(h, r["term"], r["schema"], boards, cache))
    if rows:
        for kind in KINDS:
            m = get(kind)
            M.save(M.train(kind, rows, np.stack(ctx), epochs=epochs, lr=5e-4, model=m), _ckpt(kind))
    st["offset"], st["adv_offset"] = len(lines), len(adv)
    json.dump(st, open(_state_path(), "w"))
    return len(rows)


def _adv_lines() -> list:
    from .verifier import ADVANTAGE
    return _complete_lines(paths.derived(ADVANTAGE))


_BG = None


def _bg_main():
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    import catspace.incubator.analyst  # noqa: F401
    n = update()
    print(f"OBS   (background) the proposers fine-tuned on {n} new verdicts", flush=True)


def update_in_background() -> None:
    """Start an update beside the CPU work; one at a time (a later call's verdicts wait for the next)."""
    global _BG
    import multiprocessing as mp
    if _BG is not None and _BG.is_alive():
        return
    _BG = mp.get_context("spawn").Process(target=_bg_main, daemon=False)
    _BG.start()


def wait() -> None:
    """The run's end: the background update joined, then the verdicts it skipped (one runs at a time) trained too."""
    if _BG is not None:
        _BG.join()
    st = _state()
    if st.get("offset", 0) < len(_store_lines()) or st.get("adv_offset", 0) < len(_adv_lines()):
        import catspace.incubator.analyst  # noqa: F401
        print(f"OBS   the proposers fine-tuned on {update()} verdicts left at the end")
