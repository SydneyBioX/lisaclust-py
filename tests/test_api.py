import numpy as np
import pandas as pd

import lisaclust


def toy(seed=1):
    rng = np.random.default_rng(seed)
    n = 800
    x = np.r_[rng.uniform(0, 100, n // 2), rng.uniform(100, 200, n // 2)]
    y = rng.uniform(0, 100, n)
    return pd.DataFrame(
        {"x": x, "y": y, "cellType": np.where(x < 100, "a", "b"), "imageID": rng.choice(["s1", "s2"], n)},
        index=[f"c{i}" for i in range(n)],
    )


def test_rows_follow_the_input_order():
    d = toy()
    cv = lisaclust.lisa(d, r=(10, 20))
    assert list(cv.index) == list(d.index)


def test_two_regions_separate_the_two_halves():
    d = toy()
    out = lisaclust.lisa_clust(d, k=2, r=(10, 20), random_state=0)
    tab = pd.crosstab(out["cellType"], out["region"])
    assert (tab.max(axis=1) / tab.sum(axis=1)).min() > 0.9


def test_region_map_is_observed_over_expected():
    d = toy()
    d["region"] = np.where(d["x"] < 100, "r1", "r2")
    e = lisaclust.region_map(d)
    assert np.isclose(e.loc["a", "r1"], 2.0)


def test_anndata_input():
    ad = __import__("anndata")
    d = toy()
    adata = ad.AnnData(obs=d[["cellType", "imageID"]].copy(), obsm={"spatial": d[["x", "y"]].to_numpy()})
    lisaclust.lisa_clust(adata, k=2, r=(10, 20), random_state=0)
    assert set(adata.obs["region"]) == {"region_1", "region_2"}
