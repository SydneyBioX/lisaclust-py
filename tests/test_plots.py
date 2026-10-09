"""Region outlines against lisaClust (R) and the hatching plot."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

import lisaclust

matplotlib.use("Agg")
shapely = pytest.importorskip("shapely")

HERE = Path(__file__).parent / "shared_cases"


def r_outline(rings):
    """A shapely geometry from R's rings (outer anticlockwise, holes clockwise), by the even-odd rule."""
    geom = shapely.Polygon()
    for r in rings:
        geom = geom.symmetric_difference(shapely.Polygon(np.column_stack([r["x"], r["y"]])))
    return geom


@pytest.mark.parametrize("window", ["square", "convex"])
def test_outlines_match_r(window):
    cells = pd.read_csv(HERE / "outline_cells.csv")
    expected = json.loads((HERE / "outlines.json").read_text())[window]
    got = lisaclust.region_polygons(cells["x"], cells["y"], cells["region"], window=window)
    assert sorted(got) == sorted(expected)
    for r, rings in expected.items():
        ref = r_outline(rings)
        assert got[r].symmetric_difference(ref).area < 1e-6 * ref.area


def test_outlines_partition_the_window():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 100, 500), rng.uniform(0, 80, 500)
    region = np.where(x < 40, "a", np.where(y < 40, "b", "c"))
    for window in ["square", "convex", "concave"]:
        polys = lisaclust.region_polygons(x, y, region, window=window)
        total = shapely.union_all(list(polys.values()))
        assert abs(sum(p.area for p in polys.values()) - total.area) < 1e-6 * total.area
        for r, p in polys.items():
            inside = shapely.contains_xy(p.buffer(1e-9), x[region == r], y[region == r])
            assert inside.mean() > 0.98


def test_hatching_plot_draws():
    rng = np.random.default_rng(2)
    n = 600
    d = pd.DataFrame({"x": rng.uniform(0, 100, n), "y": rng.uniform(0, 100, n), "imageID": rng.choice(["i1", "i2"], n)})
    d["cellType"] = np.where(d["x"] < 50, "left", "right")
    d["region"] = np.where(d["x"] < 50, "region_1", "region_2")
    axes = lisaclust.hatching_plot(d, use_images=["i1", "i2"], window="convex")
    assert len(axes) == 2
    assert any(p.get_hatch() for p in axes[0].patches)
    axes[0].figure.savefig(Path(__file__).parent / "hatching_test.png")
    (Path(__file__).parent / "hatching_test.png").unlink()


def test_twelve_hatchings():
    rng = np.random.default_rng(3)
    n = 3000
    d = pd.DataFrame({"x": rng.uniform(0, 400, n), "y": rng.uniform(0, 300, n), "imageID": "a", "cellType": "c"})
    d["region"] = [f"region_{int(y // 100) * 4 + int(x // 100) + 1:02d}" for x, y in zip(d["x"], d["y"], strict=True)]
    axes = lisaclust.hatching_plot(d, window="square")
    hatches = sorted({p.get_hatch() or "" for p in axes[0].patches})
    assert hatches == sorted(lisaclust._plots.HATCHES)
    d.loc[d["x"] > 390, "region"] = "region_13"
    with pytest.warns(UserWarning, match="more than 12 regions"):
        lisaclust.hatching_plot(d, window="square")
