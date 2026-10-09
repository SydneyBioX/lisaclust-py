"""Inputs: a pandas DataFrame or an AnnData object, turned into one cell table (as lisaClust's ``.formatCells()``)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def format_cells(cells, image_id="imageID", cell_type="cellType", spatial_coords=("x", "y"), spatial_key="spatial"):
    """A DataFrame with columns imageID, cellType (categorical, in order of first appearance), x and y.

    The index is that of ``cells`` (the obs names of an AnnData object).
    """
    if type(cells).__name__ == "AnnData":
        obs = cells.obs.copy()
        if not (spatial_coords[0] in obs.columns and spatial_coords[1] in obs.columns):
            if spatial_key not in cells.obsm:
                raise ValueError(f"no coordinates: neither obs columns {spatial_coords} nor obsm['{spatial_key}'].")
            xy = np.asarray(cells.obsm[spatial_key])[:, :2].astype(float)
            obs[spatial_coords[0]], obs[spatial_coords[1]] = xy[:, 0], xy[:, 1]
    elif isinstance(cells, pd.DataFrame):
        obs = cells
    else:
        raise TypeError("`cells` must be a pandas DataFrame or an AnnData object.")
    for col in (image_id, cell_type, *spatial_coords):
        if col not in obs.columns:
            raise ValueError(f"column '{col}' not found; columns are: {list(obs.columns)}")
    if not obs.index.is_unique:
        raise ValueError("the cells' index (obs names) must be unique.")
    ct = obs[cell_type]
    categories = ct.cat.categories if isinstance(ct.dtype, pd.CategoricalDtype) else pd.unique(ct.astype(str))
    return pd.DataFrame(
        {
            "imageID": obs[image_id].astype(str).to_numpy(),
            "cellType": pd.Categorical(ct.astype(str), categories=[str(c) for c in categories]),
            "x": obs[spatial_coords[0]].to_numpy(float),
            "y": obs[spatial_coords[1]].to_numpy(float),
        },
        index=obs.index,
    )
