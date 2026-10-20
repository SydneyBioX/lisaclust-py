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


def format_weights(weights, n_cells, labels=None, cell_type=None):
    """Soft cell type membership as a CSR matrix (cells x labels, explicit zeros removed) and its labels.

    ``weights`` is a DataFrame (its columns are the labels), or a numpy array or scipy sparse matrix with ``labels``,
    with one row per cell and finite, non-negative weights; or a 1-D array (numpy array, list or pandas Series, taken
    by position) with one weight in [0, 1] per cell for its own cell type, which gives the cells x cell types matrix
    with each cell's weight in its type's column. The cell types are then those of ``cell_type`` (a categorical, one
    value per cell), in the order of its categories; a cell with a missing type has no weight. As lisaClust's
    ``.membership()``.
    """
    import scipy.sparse as sp

    if not isinstance(weights, pd.DataFrame) and not sp.issparse(weights) and np.ndim(weights) == 1:
        w = np.asarray(weights.to_numpy() if isinstance(weights, pd.Series) else weights)
        if not (np.issubdtype(w.dtype, np.number) or w.dtype == bool):
            raise ValueError("`cell_weights` must be numeric.")
        w = w.astype(float)
        if len(w) != n_cells:
            raise ValueError(
                f"`cell_weights` has {len(w)} values but there are {n_cells} cells (one per cell, in the order of "
                "`cells`)."
            )
        if not np.all(np.isfinite(w)) or np.any(w < 0) or np.any(w > 1):
            raise ValueError("`cell_weights` must be between 0 and 1, with no missing values.")
        cat = pd.Categorical(cell_type)
        j = np.asarray(cat.codes, dtype=np.int64)
        keep = (j >= 0) & (w != 0)
        m = sp.csr_matrix((w[keep], (np.flatnonzero(keep), j[keep])), shape=(n_cells, len(cat.categories)))
        m.sort_indices()
        return m, [str(v) for v in cat.categories]
    if isinstance(weights, pd.DataFrame):
        if labels is None:
            labels = list(weights.columns)
        weights = weights.to_numpy()
    if labels is None:
        raise ValueError("`cell_weights` needs labels: a DataFrame's columns, or `weight_labels`.")
    labels = [str(v) for v in labels]
    if len(set(labels)) != len(labels) or any(v == "" for v in labels):
        raise ValueError("`cell_weights` needs unique, non-empty labels.")
    if sp.issparse(weights):
        m = sp.csr_matrix(weights, dtype=float)
    else:
        a = np.asarray(weights)
        if a.ndim != 2 or not (np.issubdtype(a.dtype, np.number) or a.dtype == bool):
            raise ValueError("`cell_weights` must be a numeric 2-d array, DataFrame or scipy sparse matrix.")
        m = sp.csr_matrix(a.astype(float))
    if m.shape[1] != len(labels):
        raise ValueError(f"`cell_weights` has {m.shape[1]} columns but {len(labels)} labels.")
    if m.shape[0] != n_cells:
        raise ValueError(
            f"`cell_weights` has {m.shape[0]} rows but there are {n_cells} cells "
            "(one row per cell, in the order of `cells`)."
        )
    m.eliminate_zeros()
    m.sort_indices()
    if not np.all(np.isfinite(m.data)) or np.any(m.data < 0):
        raise ValueError("`cell_weights` must be finite and non-negative, with no missing values.")
    return m, labels
