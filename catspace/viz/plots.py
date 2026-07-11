"""viz/plots.py — shared matplotlib styling for the static atlas figures."""
from __future__ import annotations


def style(ax, title: str) -> None:
    ax.set_facecolor("#12181F")
    ax.set_title(title, color="#E8E4D9", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
    for spn in ax.spines.values():
        spn.set_color("#2A3542")


def kde_layer(ax, P, pts2d, color, levels: int = 5, alpha: float = 0.30, nsub: int = 2500, seed: int = 1):
    import numpy as np
    from scipy.stats import gaussian_kde

    if len(pts2d) > nsub:
        pts2d = pts2d[np.random.default_rng(seed).choice(len(pts2d), nsub, replace=False)]
    k = gaussian_kde(pts2d.T, bw_method=0.18)
    x0, x1 = P[:, 0].min(), P[:, 0].max(); y0, y1 = P[:, 1].min(), P[:, 1].max()
    gx, gy = np.meshgrid(np.linspace(x0, x1, 160), np.linspace(y0, y1, 160))
    z = k(np.stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, z, levels=np.linspace(z.max() * 0.12, z.max(), levels),
                colors=[color] * levels, alpha=alpha)
