"""Cell weights (lisa(cell_weights=)): a weight per cell, or soft cell type membership."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import lisaclust

HERE = Path(__file__).parent / "shared_cases"
CELLS = pd.read_csv(HERE / "cells.csv").set_index("cellID")
CELLS["cellType"] = pd.Categorical(CELLS["cellType"], categories=list(pd.unique(CELLS["cellType"])))
WEIGHTS = pd.read_csv(HERE / "weights.csv").set_index("cellID")


def one_hot(cells):
    return pd.get_dummies(cells["cellType"]).astype(float)


@pytest.mark.parametrize(("window", "lisa_func", "sigma"), [("square", "K", None), ("convex", "L", 50.0)])
def test_one_hot_weights_give_the_cell_type_curves(window, lisa_func, sigma):
    kw = {"r": (10, 25, 50), "window": window, "lisa_func": lisa_func, "sigma": sigma}
    hard = lisaclust.lisa(CELLS, **kw)
    soft = lisaclust.lisa(CELLS, cell_weights=one_hot(CELLS), **kw)
    assert list(soft.columns) == list(hard.columns)
    assert np.abs(soft.to_numpy() - hard.to_numpy()).max() <= 1e-12


def test_array_and_sparse_weights_match_a_data_frame():
    w = WEIGHTS.loc[CELLS.index]
    ref = lisaclust.lisa(CELLS, r=(10, 25), cell_weights=w)
    for m in (w.to_numpy(), sp.csr_matrix(w.to_numpy()), sp.coo_matrix(w.to_numpy())):
        got = lisaclust.lisa(CELLS, r=(10, 25), cell_weights=m, weight_labels=list(w.columns))
        pd.testing.assert_frame_equal(got, ref)


def test_weights_follow_the_rows_of_the_cells():
    w = WEIGHTS.loc[CELLS.index]
    ref = lisaclust.lisa(CELLS, r=(10, 25, 50), cell_weights=w)
    perm = np.random.default_rng(3).permutation(len(CELLS))
    got = lisaclust.lisa(CELLS.iloc[perm], r=(10, 25, 50), cell_weights=w.iloc[perm])
    assert list(got.index) == list(CELLS.index[perm])
    np.testing.assert_allclose(got.loc[ref.index].to_numpy(), ref.to_numpy(), rtol=0, atol=1e-12)


def test_soft_weights_match_r():
    expected = pd.read_csv(HERE / "soft_K.csv").set_index("cellID")
    got = lisaclust.lisa(CELLS, r=(10, 25, 50), window="square", lisa_func="K", cell_weights=WEIGHTS.loc[CELLS.index])
    assert list(got.columns) == list(expected.columns)
    # as the square cases of test_shared_cases.py: spatstat's density at lisaClust's default bandwidth is flat to 1e-9
    np.testing.assert_allclose(got.loc[expected.index].to_numpy(), expected.to_numpy(), rtol=1e-6, atol=1e-6)


def test_lisa_clust_passes_the_weights():
    out = lisaclust.lisa_clust(CELLS, k=3, r=(10, 25), random_state=0, cell_weights=one_hot(CELLS))
    ref = lisaclust.lisa_clust(CELLS, k=3, r=(10, 25), random_state=0)
    assert (out["region"] == ref["region"]).all()


@pytest.mark.parametrize(("window", "lisa_func", "sigma"), [("square", "K", None), ("convex", "L", 50.0)])
def test_a_vector_of_cell_weights_is_the_one_hot_matrix_scaled_by_it(window, lisa_func, sigma):
    kw = {"r": (10, 25, 50), "window": window, "lisa_func": lisa_func, "sigma": sigma}
    q = np.round(np.random.default_rng(7).uniform(0.2, 1, len(CELLS)), 3)
    q[::17] = 0.0
    ref = lisaclust.lisa(CELLS, cell_weights=one_hot(CELLS) * q[:, None], **kw)
    for v in (q, list(q), pd.Series(q, index=CELLS.index[::-1])):
        got = lisaclust.lisa(CELLS, cell_weights=v, **kw)
        pd.testing.assert_frame_equal(got, ref)
    hard = lisaclust.lisa(CELLS, **kw)
    ones = lisaclust.lisa(CELLS, cell_weights=np.ones(len(CELLS)), **kw)
    assert list(ones.columns) == list(hard.columns)
    assert np.abs(ones.to_numpy() - hard.to_numpy()).max() <= 1e-12


@pytest.mark.parametrize(
    ("weights", "labels", "message"),
    [
        (np.ones((5, 2)), ["a", "b"], "rows"),
        (np.ones((len(CELLS), 2)), None, "labels"),
        (np.ones((len(CELLS), 2)), ["a", "a"], "unique"),
        (-np.ones((len(CELLS), 2)), ["a", "b"], "non-negative"),
        (np.full((len(CELLS), 2), np.nan), ["a", "b"], "non-negative"),
        (np.ones(len(CELLS) - 1), None, "values but there are"),
        (np.full(len(CELLS), 1.5), None, "between 0 and 1"),
        (np.full(len(CELLS), np.nan), None, "between 0 and 1"),
    ],
)
def test_bad_weights_are_rejected(weights, labels, message):
    with pytest.raises(ValueError, match=message):
        lisaclust.lisa(CELLS, r=(10,), cell_weights=weights, weight_labels=labels)
