"""name_regions() and region_box_plot(), on the toy data of lisaClust's tests/testthat/test-regions.R."""

import matplotlib
import pandas as pd
import pytest

import lisaclust

matplotlib.use("Agg")


def toy():
    return pd.DataFrame(
        {
            "imageID": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            "group": ["A"] * 8 + ["B"] * 4,
            "cellType": ["t", "t", "i", "i", "t", "t", "t", "i", "i", "i", "i", "t"],
            "region": ["r1", "r1", "r2", "r2", "r1", "r1", "r3", "r2", "r2", "r2", "r2", "r3"],
        }
    )


def test_name_regions_by_most_enriched_marker():
    out = lisaclust.name_regions(toy(), {"tumour": "t", "immune": "i"}, key_added="domain")
    # r1 and r3 hold only t cells, r2 only i cells
    assert list(out["domain"]) == list(toy()["region"].map({"r1": "tumour", "r2": "immune", "r3": "tumour"}))
    assert list(out["region"]) == list(toy()["region"])
    assert set(lisaclust.name_regions(toy(), ["i"])["region"]) == {"i"}
    with pytest.raises(ValueError, match="not found"):
        lisaclust.name_regions(toy(), ["b"])


def test_name_regions_on_anndata():
    ad = pytest.importorskip("anndata")
    adata = ad.AnnData(obs=toy().set_axis([f"c{i}" for i in range(12)]))
    lisaclust.name_regions(adata, {"tumour": "t", "immune": "i"})
    assert list(adata.obs["region"]) == list(lisaclust.name_regions(toy(), {"tumour": "t", "immune": "i"})["region"])


def test_region_shares_include_zeros():
    s = lisaclust.region_shares(toy(), "group")
    assert len(s) == 9
    assert s.query("unit == 'a' and region == 'r3'")["share"].item() == 0
    assert s.query("unit == 'c' and region == 'r2'")["share"].item() == 0.75
    assert set(s.loc[s["unit"] == "c", "condition"]) == {"B"}
    bad = toy()
    bad.loc[0, "group"] = "B"
    with pytest.raises(ValueError, match="single"):
        lisaclust.region_shares(bad, "group")


def test_region_box_plot_draws_one_panel_per_region():
    assert len(lisaclust.region_box_plot(toy(), "group")) == 3
    assert len(lisaclust.region_box_plot(toy(), "group", regions=["r1"])) == 1
