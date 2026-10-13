"""The curriculum's band: starts within a band of the tablebase's distance to zeroing (material_walk._start)."""
import random

import chess.syzygy

from catspace.incubator.analyst.material_walk import _start
from catspace.io import paths


def test_starts_fall_in_the_band():
    tb = chess.syzygy.open_tablebase(str(paths.syzygy_dir()))
    for seed in range(5):
        b = _start(random.Random(seed), "KBNvK", tb, (1, 12))
        assert tb.probe_wdl(b) == 2 and 1 <= abs(tb.probe_dtz(b)) <= 12
    b = _start(random.Random(0), "KRvK", tb)                 # no band: any won start, as before
    assert tb.probe_wdl(b) == 2
