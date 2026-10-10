"""Region outlines and hatching plots, as lisaClust's ``hatchingPlot()`` and ``geom_hatching()``.

Each region is outlined by the union of the Voronoi tiles of its cells, clipped to the image window, and drawn
with matplotlib's hatching, which clips the lines to the outline itself.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

from . import _core
from ._input import format_cells

# lisaClust's twelve hatchings: none, /, \, -, |, x, +, dots, circles, / with dots, \ with dots and - with dots
HATCHES = ["", "/", "\\", "-", "|", "x", "+", ".", "o", "/.", "\\.", "-."]

_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 20


def _window(x, y, window="concave", window_length=None):
    """The image window as a shapely polygon, as lisaClust's ``makeWindow()`` (without growing it)."""
    import shapely

    if window == "square":
        return shapely.box(x.min(), y.min(), x.max(), y.max())
    if window == "convex":
        return shapely.Polygon(_core.convex_hull(x, y))
    if window == "concave":
        # each cell and its 8 neighbours at distance (x range) / n, wrapped by concaveman's concave hull (the C++
        # port in the core, the same polygon as concaveman::concaveman() in R)
        span = x.max() - x.min()
        length = span / 20 if window_length is None else span / 20 * window_length
        d = span / len(x)
        ox = np.array([0, 1, 0, -1, -1, 0, 1, -1, 1]) * d
        oy = np.array([0, 1, 1, 1, -1, -1, -1, 0, 0]) * d
        ring = _core.concave_hull((x[:, None] + ox).ravel(), (y[:, None] + oy).ravel(), 1.0, length)
        return shapely.Polygon(ring[::-1])
    raise ValueError("window must be 'square', 'convex' or 'concave'.")


def _voronoi_tiles(x, y):
    """The Voronoi tile of each point as a shapely polygon, bounded by far-away frame points."""
    import shapely
    from scipy.spatial import Voronoi

    span = max(np.ptp(x), np.ptp(y), 1e-9)
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    far = 10 * span
    frame = np.array(
        [
            [cx - far, cy - far],
            [cx + far, cy - far],
            [cx + far, cy + far],
            [cx - far, cy + far],
            [cx, cy - far],
            [cx + far, cy],
            [cx, cy + far],
            [cx - far, cy],
        ]
    )
    vor = Voronoi(np.vstack([np.column_stack([x, y]), frame]))
    tiles = []
    for i in range(len(x)):
        region = vor.regions[vor.point_region[i]]
        tiles.append(shapely.Polygon(vor.vertices[region]))
    return tiles


def region_polygons(x, y, region, window="concave", window_length=None):
    """The outline of each region: the union of its cells' Voronoi tiles, clipped to the window.

    Parameters
    ----------
    x, y
        The cells' coordinates.
    region
        The cells' regions.
    window
        ``"concave"``, ``"convex"`` (the convex hull) or ``"square"`` (the bounding box).
    window_length
        For a concave window, a multiple of its default edge length (larger gives a smoother outline).

    Returns
    -------
    A dict from region to a shapely Polygon or MultiPolygon.

    """
    import shapely

    x, y = np.asarray(x, float), np.asarray(y, float)
    region = np.asarray(region)
    key = hashlib.sha256(
        b"".join(
            [
                x.tobytes(),
                y.tobytes(),
                pd.Series(region).astype(str).str.cat(sep="\x1f").encode(),
                repr((window, window_length)).encode(),
            ]
        )
    ).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    win = _window(x, y, window, window_length)
    keep = ~pd.DataFrame({"x": x, "y": y}).duplicated().to_numpy()
    x, y, region = x[keep], y[keep], region[keep]
    labels = sorted(pd.unique(region), key=str)
    if len(labels) == 1 or len(x) < 3:
        out = {labels[0]: win}
    else:
        tiles = np.array(_voronoi_tiles(x, y), dtype=object)
        out = {r: shapely.intersection(shapely.union_all(tiles[region == r]), win) for r in labels}
    _CACHE[key] = out
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return out


def _path(geom):
    """A matplotlib Path of a shapely (Multi)Polygon, holes included."""
    from matplotlib.path import Path

    polys = (
        [geom] if geom.geom_type == "Polygon" else [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    )
    paths = []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            if len(xy) >= 3:
                codes = np.full(len(xy), Path.LINETO)
                codes[0], codes[-1] = Path.MOVETO, Path.CLOSEPOLY
                paths.append(Path(xy, codes))
    return Path.make_compound_path(*paths) if paths else None


def hatching_plot(
    cells,
    use_images=None,
    region="region",
    image_id="imageID",
    cell_type="cellType",
    spatial_coords=("x", "y"),
    window="concave",
    window_length=None,
    density=1,
    hatch_colour="black",
    line_width=1.0,
    point_size=4,
    ncols=3,
    axes=None,
):
    r"""Plot cells coloured by type, with each region outlined and hatched, one panel per image.

    The equivalent of lisaClust's ``hatchingPlot()``. Regions get, in order, no hatching, ``/``, ``\``, ``-``,
    ``|``, ``x``, ``+``, dots, circles, ``/`` with dots, ``\`` with dots and ``-`` with dots; regions beyond the
    twelfth are drawn as the first.

    Parameters
    ----------
    cells
        A pandas DataFrame or an AnnData object with the regions (e.g. from :func:`lisa_clust`).
    use_images
        The images to plot; the first image by default.
    region, image_id, cell_type, spatial_coords
        The columns of the regions, images, cell types and coordinates.
    window, window_length
        The image window: see :func:`region_polygons`.
    density
        How many times the hatching is repeated per matplotlib hatch unit (larger is denser).
    hatch_colour, line_width
        The colour and width of the hatching and outlines.
    point_size
        The size of the cells' points.
    ncols
        Panels per row.
    axes
        Matplotlib axes to draw on, one per image; new ones by default.

    Returns
    -------
    The matplotlib axes.

    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, PathPatch

    df = format_cells(cells, image_id, cell_type, spatial_coords)
    obs = cells.obs if type(cells).__name__ == "AnnData" else cells
    if region not in obs.columns:
        raise ValueError(f"column '{region}' not found.")
    df["region"] = np.asarray(obs[region])
    images = list(pd.unique(df["imageID"])) if use_images is None else [str(i) for i in np.atleast_1d(use_images)]
    if use_images is None:
        images = images[:1]
    missing = set(images) - set(df["imageID"])
    if missing:
        raise ValueError(f"images not in the data: {sorted(missing)}")

    regions = sorted(pd.unique(df["region"]), key=str)
    if len(regions) > len(HATCHES):
        warnings.warn(
            f"Can not plot more than {len(HATCHES)} regions. Regions after the {len(HATCHES)}th are drawn as the first.",
            stacklevel=2,
        )
    hatch = {r: HATCHES[i] if i < len(HATCHES) else HATCHES[0] for i, r in enumerate(regions)}
    types = list(df["cellType"].cat.categories)
    shown = [t for t in types if t in set(df.loc[df["imageID"].isin(images), "cellType"])]
    type_cols = int(np.ceil(len(shown) / 12))
    palette = plt.get_cmap("tab20" if len(types) > 10 else "tab10")
    colour = {t: palette(i % palette.N) for i, t in enumerate(types)}

    if axes is None:
        n = len(images)
        nc = min(ncols, n)
        nr = int(np.ceil(n / nc))
        extra = 2.5 + 1.6 * (type_cols - 1)
        fig, axes = plt.subplots(nr, nc, figsize=(5 * nc + extra, max(5 * nr, 5.5)), squeeze=False)
        fig.subplots_adjust(right=5 * nc / (5 * nc + extra) - 0.02)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        axes = axes.ravel()[:n]
    axes = list(np.atleast_1d(axes))

    with plt.rc_context({"hatch.linewidth": line_width, "hatch.color": hatch_colour}):
        for ax, image in zip(axes, images, strict=True):
            d = df[df["imageID"] == image]
            ax.scatter(d["x"], d["y"], s=point_size, c=[colour[t] for t in d["cellType"]], linewidths=0, zorder=1)
            for r, geom in region_polygons(d["x"], d["y"], d["region"], window, window_length).items():
                path = _path(geom)
                if path is None:
                    continue
                ax.add_patch(
                    PathPatch(
                        path,
                        fill=False,
                        hatch=hatch[r] * density,
                        edgecolor=hatch_colour,
                        linewidth=line_width,
                        zorder=2,
                    )
                )
            ax.set_title(image)
            ax.set_aspect("equal")
            ax.set_xlabel(spatial_coords[0])
            ax.set_ylabel(spatial_coords[1])

        type_handles = [Line2D([], [], marker="o", linestyle="", markersize=5, color=colour[t], label=t) for t in shown]
        region_handles = [
            Patch(facecolor="none", edgecolor=hatch_colour, hatch=hatch[r] * density, label=str(r)) for r in regions
        ]
        # legends to the right of the panels
        fig = axes[0].figure
        right = max(ax.get_position().x1 for ax in axes)
        legend = fig.legend(
            handles=region_handles,
            title="region",
            loc="upper left",
            bbox_to_anchor=(right + 0.01, 0.95),
            handlelength=2.5,
            handleheight=2,
            frameon=False,
            fontsize="small",
            title_fontsize="small",
        )
        fig.canvas.draw()
        bottom = legend.get_window_extent().transformed(fig.transFigure.inverted()).y0
        fig.legend(
            handles=type_handles,
            title="cell type",
            loc="upper left",
            frameon=False,
            ncols=type_cols,
            fontsize="small",
            title_fontsize="small",
            columnspacing=0.8,
            bbox_to_anchor=(right + 0.01, bottom - 0.02),
        )
    return axes
