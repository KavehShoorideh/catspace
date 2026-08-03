"""Atlas component: region-level statistics from the M3 composite table --
shrunk committor quality, chute fall rates, badness. Pure numpy aggregation."""
from __future__ import annotations

import numpy as np


class RegionAtlas:
    """Region-level view of a composite (region x cband x band) table."""

    def __init__(self, rk, elo_self: float, elo_oppo: float):
        b_self, b_opp = rk.band(elo_self), rk.band(elo_oppo)
        G, CB = len(rk.bank), rk.n_cband
        q = rk.quality[:, b_self].reshape(G, CB)
        c = rk.counts[:, b_self].reshape(G, CB).astype(float)
        n = c.sum(1)
        # quality with SHRINKAGE (a prior, not a cutoff): thin cells regress to
        # the population mean with pseudo-count = median support
        q_raw = (q * c).sum(1) / np.maximum(n, 1.0)
        q_bar = float((q * c).sum() / max(c.sum(), 1.0))
        n0 = float(np.median(n))
        self.q_region = (n * q_raw + n0 * q_bar) / (n + n0)
        self.badness = 1.0 - self.q_region                  # mated-ness of a destination
        # CHUTE fall rates: SF-refereed committor-crossing rates at THEIR band vs
        # OURS (M3 "their error zone" columns), count-weighted to region level
        fl_op = rk.flux[:, b_opp].reshape(G, CB)
        fl_us = rk.flux[:, b_self].reshape(G, CB)
        c_op = rk.counts[:, b_opp].reshape(G, CB).astype(float)
        self.fall_opp = (fl_op * c_op).sum(1) / np.maximum(c_op.sum(1), 1.0)
        self.fall_us = (fl_us * c).sum(1) / np.maximum(n, 1.0)
