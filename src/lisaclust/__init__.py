"""lisaclust: find tissue domains and cellular niches by clustering local indicators of spatial association.

The Python version of the Bioconductor package lisaClust, on the same C++ core.
"""

from ._lisa import lisa, lisa_clust, local_curves, region_map
from ._plots import hatching_plot, region_polygons
from ._regions import name_regions, region_box_plot, region_shares

__version__ = "1.21.5"
__all__ = [
    "hatching_plot",
    "lisa",
    "lisa_clust",
    "local_curves",
    "name_regions",
    "region_box_plot",
    "region_map",
    "region_polygons",
    "region_shares",
]
