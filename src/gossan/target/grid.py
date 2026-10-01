"""The master prediction grid.

One regular grid in the project CRS covers the AOI's bounding box, snapped to whole multiples
of the resolution so a re-run with the same AOI and resolution gives the same cells. A cell is
*in the study area* when its centre falls inside the AOI polygon. Everything downstream (features,
training, scoring) works on the in-AOI cells only, held as a flat "compact" vector in row-major
order; ``Grid.to_raster`` puts a compact vector back on the 2-D grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin


class GridError(Exception):
    """The grid cannot be built; the message says why."""


@dataclass
class Grid:
    transform: Affine
    width: int
    height: int
    crs: str
    resolution: float
    mask: np.ndarray  # (height, width) bool: cell centre inside the AOI
    #: (height, width) int: compact index of each in-AOI cell, -1 outside.
    index: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(self.mask.sum())

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def rows_cols(self) -> tuple[np.ndarray, np.ndarray]:
        """Row and column of every in-AOI cell, in compact order."""
        return np.nonzero(self.mask)

    @property
    def xy(self) -> np.ndarray:
        """(n_cells, 2) coordinates of the in-AOI cell centres."""
        rows, cols = self.rows_cols
        x = self.transform.c + (cols + 0.5) * self.transform.a
        y = self.transform.f + (rows + 0.5) * self.transform.e
        return np.column_stack([x, y])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        t = self.transform
        return t.c, t.f + t.e * self.height, t.c + t.a * self.width, t.f

    def to_raster(self, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """Spread a compact vector (one value per in-AOI cell) over the full 2-D grid."""
        out = np.full(self.shape, fill, dtype=np.float32)
        out[self.mask] = values
        return out

    def compact(self, array: np.ndarray) -> np.ndarray:
        """The in-AOI cells of a full 2-D array, as a compact vector."""
        return array[self.mask]

    def cell_of(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compact index of the cell holding each point, -1 where the point is outside the grid
        or in a cell outside the AOI."""
        cols = np.floor((np.asarray(x) - self.transform.c) / self.transform.a).astype(int)
        rows = np.floor((np.asarray(y) - self.transform.f) / self.transform.e).astype(int)
        inside = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        out = np.full(len(rows), -1, dtype=int)
        out[inside] = self.index[rows[inside], cols[inside]]
        return out

    def describe(self) -> dict[str, object]:
        minx, miny, maxx, maxy = self.bounds
        return {
            "crs": self.crs,
            "resolution_m": self.resolution,
            "width": self.width,
            "height": self.height,
            "n_cells_bbox": self.width * self.height,
            "n_cells_in_aoi": self.n_cells,
            "bounds": [minx, miny, maxx, maxy],
            "transform": list(self.transform)[:6],
            "area_km2": round(self.n_cells * self.resolution**2 / 1e6, 4),
        }


def build_grid(aoi: gpd.GeoDataFrame, crs: str, resolution: float, max_cells: int) -> Grid:
    """Grid over ``aoi`` (any CRS) in ``crs`` at ``resolution`` metres.

    Raises ``GridError`` if the AOI holds no cell centres, or the bounding box would exceed
    ``max_cells`` cells.
    """
    aoi = aoi.to_crs(crs)
    minx, miny, maxx, maxy = aoi.total_bounds
    x0 = math.floor(minx / resolution) * resolution
    y1 = math.ceil(maxy / resolution) * resolution
    width = max(1, math.ceil((maxx - x0) / resolution))
    height = max(1, math.ceil((y1 - miny) / resolution))
    if width * height > max_cells:
        raise GridError(
            f"a {resolution:g} m grid over this AOI would have {width * height:,} cells "
            f"(limit {max_cells:,}); use a coarser --resolution"
        )
    transform = from_origin(x0, y1, resolution, resolution)
    mask = (
        rasterize(
            [(geom, 1) for geom in aoi.geometry if not geom.is_empty],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=False,  # cell centres inside the polygon
            dtype="uint8",
        )
        > 0
    )
    if not mask.any():
        raise GridError(
            f"no {resolution:g} m cell centre falls inside the AOI; use a finer --resolution"
        )
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[mask] = np.arange(int(mask.sum()))
    return Grid(transform, width, height, crs, resolution, mask, index)
