"""The Python package against lisaClust (R) on the shared cases (tests/shared_cases/make_cases.R)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lisaclust

HERE = Path(__file__).parent / "shared_cases"
CASES = json.loads((HERE / "cases.json").read_text())
CELLS = pd.read_csv(HERE / "cells.csv").set_index("cellID")
CELLS["cellType"] = pd.Categorical(CELLS["cellType"], categories=list(pd.unique(CELLS["cellType"])))


@pytest.mark.parametrize("name", sorted(CASES))
def test_case_matches_r(name):
    case = CASES[name]
    r = case["r"] if isinstance(case["r"], list) else [case["r"]]
    expected = pd.read_csv(HERE / f"{name}.csv").set_index("cellID")
    got = lisaclust.lisa(CELLS, r=r, window=case["window"], lisa_func=case["lisa_func"])
    assert list(got.columns) == list(expected.columns)
    got = got.loc[expected.index]
    # The square window is reproduced up to spatstat's density with lisaClust's default bandwidth, which is flat to
    # about 1e-9 (Python uses weights of exactly 1); the convex hull's rounded growth by 0.01 only approximately.
    tol = 1e-6 if case["window"] == "square" else 1e-4
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=tol, atol=tol)
