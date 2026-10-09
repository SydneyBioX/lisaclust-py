# lisaclust

Find tissue domains and cellular niches by clustering local indicators of spatial association (LISA). This is the
Python version of the Bioconductor package [lisaClust](https://bioconductor.org/packages/lisaClust), on the same
C++ code, for AnnData objects and pandas DataFrames.

For each cell, lisaclust counts the cells of every type within a few radii of it and compares each count with
what would be expected if that type were spread evenly over the image, correcting for cells near the edge of the
image. Clustering cells by these local indicators groups cells with similar surroundings into regions: with a few
regions they are tissue domains, with many they are cellular niches. It needs only the type and position of each
cell, so it applies to spatial proteomics (imaging mass cytometry, MIBI, CODEX) and to single-cell spatial
transcriptomics (Xenium, CosMx, MERSCOPE).

```python
import lisaclust

# adata.obs has "imageID" and "cellType"; coordinates are in adata.obsm["spatial"]
lisaclust.lisa_clust(adata, k=5, r=(20, 50, 100), random_state=0)  # adds adata.obs["region"]
lisaclust.region_map(adata, plot=True)  # how enriched each cell type is in each region

curves = lisaclust.lisa(adata, r=(20, 50, 100))  # the LISA, one row per cell, for your own clustering
```

| R (lisaClust) | Python (lisaclust) |
|---|---|
| `lisaClust(cells, k, r, imageID =, cellType =, spatialCoords =)` | `lisa_clust(cells, k, r=, image_id=, cell_type=, spatial_coords=)` |
| `lisa(cells, r)` | `lisa(cells, r=)` |
| `inhomLocalK(data, Rs)` | `local_curves(x, y, cell_type, r=)` |
| `regionMap(cells, type = "bubble")` | `region_map(cells, plot=True)` |

`lisa()` gives the same values as the R package for `window="square"`, and agrees to about 1e-5 for the default
convex-hull window (R grows the hull by 0.01 with spatstat's rounded dilation). k-means differs between R and
Python, so regions are numbered differently. Not yet in the Python package: concave windows, `hatchingPlot()`;
with `sigma`, the density weights approximate spatstat's `density.ppp()` on a pixel grid.

## Development

The C++ core in `src/core` is copied from lisaClust (`tools/sync_core.sh /path/to/lisaClust`) and must not be
edited here; `tests/test_core_sync.py` checks it. `tests/shared_cases/make_cases.R` computes the expected results
with the R package, and `tests/test_shared_cases.py` checks that Python reproduces them.

Authors: Ellis Patrick and Nicolas Canete. Licence: GPL (>= 2).
