"""Local indicators of spatial association (LISA) and their clustering, as in the R package lisaClust.

The neighbour counts, disc-window areas and distances run in the C++ core shared with lisaClust (``_core``).
This module prepares each image's inputs as lisaClust's ``inhomLocalK()`` does: the window, the density weights,
the cell-type densities and the edge corrections.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import _core
from ._input import format_cells


def _ring_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _dilate_convex(ring: np.ndarray, d: float, arc_step: float = np.pi / 16) -> np.ndarray:
    """An anticlockwise convex polygon grown by distance d, with its corners rounded by arcs."""
    out = []
    m = len(ring)
    for i in range(m):
        p, prev, nxt = ring[i], ring[i - 1], ring[(i + 1) % m]
        # outward normals of the edges before and after the vertex (right of the direction of travel)
        e0, e1 = p - prev, nxt - p
        n0 = np.array([e0[1], -e0[0]]) / np.hypot(*e0)
        n1 = np.array([e1[1], -e1[0]]) / np.hypot(*e1)
        a0, a1 = np.arctan2(n0[1], n0[0]), np.arctan2(n1[1], n1[0])
        if a1 < a0:
            a1 += 2 * np.pi
        steps = max(1, int(np.ceil((a1 - a0) / arc_step)))
        for a in np.linspace(a0, a1, steps + 1):
            out.append(p + d * np.array([np.cos(a), np.sin(a)]))
    return np.asarray(out)


def _window(x: np.ndarray, y: np.ndarray, window: str) -> tuple[list[np.ndarray], float]:
    """The image window grown by 0.01 (as ``expand.owin(W, distance = 0.01)``) and its area."""
    if window == "square":
        x0, x1, y0, y1 = x.min() - 0.01, x.max() + 0.01, y.min() - 0.01, y.max() + 0.01
        ring = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    elif window == "convex":
        ring = _dilate_convex(_core.convex_hull(x, y), 0.01)
    else:
        raise ValueError("window must be 'convex' or 'square'.")
    return [ring], _ring_area(ring)


def _density_weights(x, y, rings, sigma, min_lambda, n_pixels=128):
    """The inverse-density weight of each cell as a neighbour, mean(w) / w.

    Without ``sigma`` every weight is 1 (lisaClust's default bandwidth is so wide that its density is flat). With
    ``sigma``, the density is a Gaussian kernel smooth of the cells on a pixel grid over the window's frame, divided
    by the smooth of the window (edge correction), scaled to mean 1 over the window and floored at ``min_lambda``.
    It approximates spatstat's ``density.ppp()``, which lisaClust uses, to within the pixel resolution.
    """
    if sigma is None:
        return np.ones(len(x))
    from matplotlib.path import Path
    from scipy.ndimage import gaussian_filter

    ring = rings[0]
    xr, yr = (ring[:, 0].min(), ring[:, 0].max()), (ring[:, 1].min(), ring[:, 1].max())
    px, py = (xr[1] - xr[0]) / n_pixels, (yr[1] - yr[0]) / n_pixels
    cx = xr[0] + px * (np.arange(n_pixels) + 0.5)
    cy = yr[0] + py * (np.arange(n_pixels) + 0.5)
    gx, gy = np.meshgrid(cx, cy)
    inside = Path(ring).contains_points(np.column_stack([gx.ravel(), gy.ravel()])).reshape(n_pixels, n_pixels)
    ix = np.clip(((x - xr[0]) / px).astype(int), 0, n_pixels - 1)
    iy = np.clip(((y - yr[0]) / py).astype(int), 0, n_pixels - 1)
    counts = np.zeros((n_pixels, n_pixels))
    np.add.at(counts, (iy, ix), 1.0)
    s = (sigma / py, sigma / px)
    smooth = gaussian_filter(counts, s, mode="constant")
    mass = gaussian_filter(inside.astype(float), s, mode="constant")
    den = np.where(inside & (mass > 0), smooth / np.maximum(mass, 1e-300), np.nan)
    den = den / np.nanmean(den)
    den = np.maximum(den, min_lambda)
    w = den[iy, ix]
    if np.any(np.isnan(w)):
        # cells whose pixel centre is outside the window take the nearest valid pixel
        ok = ~np.isnan(den)
        vy, vx = np.nonzero(ok)
        for i in np.nonzero(np.isnan(w))[0]:
            j = np.argmin((vx - ix[i]) ** 2 + (vy - iy[i]) ** 2)
            w[i] = den[vy[j], vx[j]]
    return w.mean() / w


def _label(v: float) -> str:
    """A radius as R's as.character() writes it."""
    return format(v, ".15g")


def local_curves(
    x, y, cell_type, r=(20, 50, 100), sigma=None, window="convex", min_lambda=0.05, lisa_func="K", cell_ids=None
) -> pd.DataFrame:
    """The LISA of the cells of one image: one row per cell, one column per radius and neighbouring type.

    The equivalent of lisaClust's ``inhomLocalK()``. Columns are named ``<radius>_<type>`` for the radii and types
    that occur among the image's pairs of cells; cells with no neighbour within the largest radius are NaN.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ct = pd.Categorical(cell_type)
    n = len(x)
    rings, area = _window(x, y, window)
    frame = rings[0]
    max_r = min(np.ptp(frame[:, 0]), np.ptp(frame[:, 1])) / 2.01
    Rs = list(dict.fromkeys(np.minimum(np.r_[0.0, np.sort(np.asarray(r, float))], max_r).tolist()))
    labels = [_label(v) for v in Rs[1:]]
    wt = _density_weights(x, y, rings, sigma, min_lambda)
    lam = (np.bincount(ct.codes, minlength=len(ct.categories)) / area).tolist()
    dist = _core.distance_to_boundary(x, y, rings)
    edge = np.ones((n, len(Rs) - 1), order="F")
    for k, R in enumerate(Rs[1:]):
        inb = dist < R
        if inb.any():
            edge[inb, k] = _core.disc_window_area(x[inb], y[inb], R, 128, rings) / (np.pi * R**2)
    res = _core.local_curves(
        x,
        y,
        ct.codes.astype(np.int32),
        len(ct.categories),
        Rs,
        [float(v) for v in labels],
        wt,
        lam,
        edge,
        lisa_func == "L",
    )
    keep = np.nonzero(res["type"])[0]
    blocks, names = [], []
    for k in np.nonzero(res["bin"])[0]:
        blocks.append(res["value"][:, keep, k])
        names += [f"{labels[k]}_{ct.categories[j]}" for j in keep]
    values = np.hstack(blocks) if blocks else np.empty((n, 0))
    values[~res["cell"], :] = np.nan
    return pd.DataFrame(values, columns=names, index=cell_ids)


def lisa(
    cells,
    r=(20, 50, 100),
    image_id="imageID",
    cell_type="cellType",
    spatial_coords=("x", "y"),
    window="convex",
    sigma=None,
    lisa_func="K",
    min_lambda=0.05,
    n_jobs=1,
) -> pd.DataFrame:
    """Local indicators of spatial association for every cell.

    Parameters
    ----------
    cells
        A pandas DataFrame or an AnnData object (coordinates in ``obs`` columns or ``obsm["spatial"]``).
    r
        The radii, in the units of the coordinates.
    image_id, cell_type, spatial_coords
        The columns of the image, the cell type and the coordinates.
    window
        The image window: ``"convex"`` (the convex hull of the cells) or ``"square"`` (their bounding box).
    sigma
        The bandwidth of the density used to weight neighbours in inhomogeneous tissue; ``None`` for no weighting.
    lisa_func
        ``"K"``, (count - expected) / sqrt(expected), or ``"L"``, sqrt(count) - sqrt(expected).
    min_lambda
        The smallest relative density used when weighting by density.
    n_jobs
        Images computed in parallel.

    Returns
    -------
    A DataFrame with a row per cell, in the order of ``cells``, and a column per radius and cell type (``<radius>_<type>``).
    Combinations that do not occur in an image are 0.

    """
    df = format_cells(cells, image_id, cell_type, spatial_coords)
    types = df["cellType"].cat.categories

    def one(idx):
        d = df.iloc[idx]
        return local_curves(
            d["x"].to_numpy(),
            d["y"].to_numpy(),
            pd.Categorical(d["cellType"], categories=types),
            r,
            sigma,
            window,
            min_lambda,
            lisa_func,
            cell_ids=d.index,
        )

    groups = list(df.groupby("imageID", observed=True, sort=False).indices.values())
    if n_jobs == 1:
        parts = [one(g) for g in groups]
    else:
        with ThreadPoolExecutor(n_jobs) as ex:
            parts = list(ex.map(one, groups))
    out = pd.concat(parts, axis=0, sort=False).fillna(0.0)
    return out.loc[df.index]


def lisa_clust(cells, k=2, key_added="region", random_state=None, n_init=1, **kwargs):
    """Cluster the cells' LISA into ``k`` regions with k-means.

    Regions are named ``region_1`` to ``region_k``. For an AnnData object they are added to ``obs[key_added]``
    (in place, and the object is returned); for a DataFrame a copy with a ``key_added`` column is returned. Other
    arguments go to :func:`lisa`.
    """
    from sklearn.cluster import KMeans

    curves = lisa(cells, **kwargs)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(curves.to_numpy())
    regions = pd.Categorical([f"region_{c + 1}" for c in km.labels_], categories=[f"region_{c + 1}" for c in range(k)])
    if type(cells).__name__ == "AnnData":
        cells.obs[key_added] = regions
        return cells
    out = cells.copy()
    out[key_added] = np.asarray(regions)
    return out


def region_map(cells, region="region", cell_type="cellType", limit=(0.33, 3), plot=False, ax=None):
    """How much more often each cell type is in each region than if types were spread evenly over regions.

    Observed count over expected (row total x column total / total), as lisaClust's ``regionMap()``. With
    ``plot=True`` it also draws a bubble plot, the values capped at ``limit``.
    """
    obs = cells.obs if type(cells).__name__ == "AnnData" else cells
    tab = pd.crosstab(obs[cell_type], obs[region])
    expected = np.outer(tab.sum(axis=1), tab.sum(axis=0)) / tab.to_numpy().sum()
    enrichment = tab / expected
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        ax = ax or plt.subplots(figsize=(1 + 0.5 * tab.shape[1], 1 + 0.3 * tab.shape[0]))[1]
        v = enrichment.clip(*limit)
        xx, yy = np.meshgrid(np.arange(v.shape[1]), np.arange(v.shape[0]))
        sc = ax.scatter(
            xx.ravel(),
            yy.ravel(),
            s=40 * v.to_numpy().ravel(),
            c=v.to_numpy().ravel(),
            cmap="RdBu_r",
            norm=LogNorm(*limit),
        )
        ax.set_xticks(range(v.shape[1]), v.columns, rotation=90)
        ax.set_yticks(range(v.shape[0]), v.index)
        ax.figure.colorbar(sc, ax=ax, label="relative frequency")
    return enrichment
