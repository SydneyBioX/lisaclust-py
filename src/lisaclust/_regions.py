"""Naming regions and comparing their shares between groups, as lisaClust's ``nameRegions()`` and ``regionBoxPlot()``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._lisa import region_map


def _obs(cells, columns):
    obs = cells.obs if type(cells).__name__ == "AnnData" else cells
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("`cells` must be a pandas DataFrame or an AnnData object.")
    missing = [c for c in columns if c not in obs.columns]
    if missing:
        raise ValueError(f"column(s) not found: {missing}")
    return obs


def name_regions(cells, markers, region="region", cell_type="cellType", key_added=None):
    """Name each region by the marker cell type it is most enriched for, as lisaClust's ``nameRegions()``.

    Enrichment is that of :func:`region_map`. Regions most enriched for the same marker get the same name, so
    several regions can be merged into one domain.

    Parameters
    ----------
    cells
        A pandas DataFrame or an AnnData object with a region column.
    markers
        A dict from region name to cell type, or a list of cell types (which are then the names).
    region, cell_type
        The columns of the regions and cell types.
    key_added
        The column for the named regions; ``region`` (replacing the regions) by default.

    Returns
    -------
    For an AnnData object, the object with the column added to ``obs`` in place; for a DataFrame, a copy.

    """
    if not isinstance(markers, dict):
        markers = {m: m for m in markers}
    obs = _obs(cells, [region, cell_type])
    enrichment = region_map(obs, region=region, cell_type=cell_type)
    absent = [m for m in markers.values() if m not in enrichment.index]
    if absent:
        raise ValueError(f"cell type(s) not found in '{cell_type}': {absent}")
    names = list(markers)
    best = enrichment.loc[list(markers.values())].to_numpy().argmax(axis=0)
    name_of = {str(r): names[i] for r, i in zip(enrichment.columns, best, strict=True)}
    named = obs[region].astype(str).map(name_of).to_numpy()
    key_added = key_added or region
    if type(cells).__name__ == "AnnData":
        cells.obs[key_added] = named
        return cells
    out = cells.copy()
    out[key_added] = named
    return out


def region_shares(cells, condition, region="region", image_id="imageID"):
    """The share of each region in each image (or patient), with its group: one row per unit and region.

    A unit with no cells in a region has a share of zero.
    """
    obs = _obs(cells, [image_id, condition, region])
    units = obs[[image_id, condition]].drop_duplicates()
    if units[image_id].duplicated().any():
        raise ValueError(f"each '{image_id}' must have a single '{condition}'")
    tab = pd.crosstab(obs[image_id].astype(str), obs[region].astype(str), normalize="index")
    shares = tab.stack().rename("share").reset_index()
    shares.columns = ["unit", "region", "share"]
    group = dict(zip(units[image_id].astype(str), units[condition], strict=True))
    shares["condition"] = shares["unit"].map(group)
    return shares.dropna(subset=["condition"]).reset_index(drop=True)


def region_box_plot(cells, condition, region="region", image_id="imageID", regions=None, axes=None):
    """Box plots of the share of each region in each image, by group, one panel per region.

    The equivalent of lisaClust's ``regionBoxPlot()``. With ``image_id`` set to a patient column, the images of
    each patient are pooled, so each patient gives one share of each region.

    Parameters
    ----------
    cells
        A pandas DataFrame or an AnnData object with a region column.
    condition
        The column of the group of each image (or patient).
    region, image_id
        The columns of the regions and of the units compared (images, or patients).
    regions
        The regions to show; all by default.
    axes
        Matplotlib axes to draw on, one per region; new ones by default.

    Returns
    -------
    The matplotlib axes.

    """
    import matplotlib.pyplot as plt

    shares = region_shares(cells, condition, region, image_id)
    shown = list(regions) if regions is not None else sorted(shares["region"].unique())
    groups = sorted(shares["condition"].unique(), key=str)
    if axes is None:
        _, axes = plt.subplots(1, len(shown), figsize=(1 + 1.6 * len(shown), 3.5), sharey=True, squeeze=False)
        axes = axes[0]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rng = np.random.default_rng(51773)
    for ax, reg in zip(axes, shown, strict=False):
        d = shares[shares["region"] == str(reg)]
        values = [d.loc[d["condition"] == g, "share"].to_numpy() for g in groups]
        ax.boxplot(values, showfliers=False, widths=0.6, medianprops={"color": "black"})
        for i, v in enumerate(values):
            ax.scatter(i + 1 + rng.uniform(-0.15, 0.15, len(v)), v, s=8, color=colours[i % len(colours)], zorder=3)
        ax.set_xticks(range(1, len(groups) + 1), [str(g) for g in groups], rotation=45, ha="right")
        ax.set_title(str(reg))
        ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:g}%")
    axes[0].set_ylabel("share of cells")
    return axes
