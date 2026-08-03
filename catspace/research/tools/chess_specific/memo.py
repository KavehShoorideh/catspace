"""catspace/search/memo.py -- bounded memoization for engine components.

Search and planning re-evaluate the same positions constantly (transpositions,
per-move re-roots, retrieval re-queries); the MCTS core already carries its own
eval caches (measured 20-34% of a game's evals are repeats). BoundedMemo is the
shared primitive for everything else: LRU, size-bounded, hit-rate instrumented
so the trace/verdicts can report cache effectiveness honestly.
"""
from __future__ import annotations

from collections import OrderedDict


class BoundedMemo:
    def __init__(self, maxsize: int = 200_000):
        self.d: OrderedDict = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0; self.misses = 0

    def get_or(self, key, fn):
        if key in self.d:
            self.hits += 1
            self.d.move_to_end(key)
            return self.d[key]
        self.misses += 1
        v = fn()
        self.d[key] = v
        if len(self.d) > self.maxsize:
            self.d.popitem(last=False)
        return v

    def batch_get_or(self, keys, batch_fn):
        """keys -> values; batch_fn(missing_keys) -> values for the misses
        (ONE batched call — keeps tensor ops batched through the cache)."""
        miss = [k for k in keys if k not in self.d]
        if miss:
            for k, v in zip(miss, batch_fn(miss)):
                self.d[k] = v
                if len(self.d) > self.maxsize:
                    self.d.popitem(last=False)
        self.hits += len(keys) - len(miss); self.misses += len(miss)
        for k in keys:
            self.d.move_to_end(k)
        return [self.d[k] for k in keys]

    @property
    def rate(self):
        n = self.hits + self.misses
        return self.hits / n if n else 0.0
