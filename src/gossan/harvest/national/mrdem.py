"""Terrain: the NRCan Medium Resolution Digital Elevation Model (MRDEM-30), clipped to the AOI.

Access method: NRCan publishes MRDEM as one Canada-wide Cloud Optimized GeoTIFF (30 m, EPSG:3979,
float32 metres, nodata -32767) on public object storage. Only the window covering the AOI (plus a
buffer, so slope and drainage are right at the edges) is read, through GDAL's ``/vsicurl/`` HTTP
range requests: no account, no bulk download. The digital terrain model (bare earth) is used, not
the surface model.

Catalogue record: https://open.canada.ca/data/en/dataset/18752265-bda3-498c-a4ba-9dfe68cb98da
("Medium Resolution Digital Elevation Model (MRDEM) - CanElevation Series").

Terms of use: Open Government Licence - Canada
(https://open.canada.ca/en/open-government-licence-canada). Attribution: "Contains information
licensed under the Open Government Licence - Canada."

The window is saved under ``raw/dem/`` as fetched (EPSG:3979) and then clipped to the AOI plus
``BUFFER_M`` and reprojected to the project CRS as ``rasters/dem.tif``.

Manual fallback: download a tile from the record above (or any DEM), then
``gossan ingest <project> <dem.tif> --layer dem``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds

from gossan.core.raster import RASTERS_DIR, raster_path, write_cog
from gossan.harvest.base import Connector, SearchItem

MRDEM_URL = "https://canelevation-dem.s3.ca-central-1.amazonaws.com/mrdem-30/mrdem-30-dtm.tif"
MRDEM_CRS = "EPSG:3979"
#: Read and keep this far beyond the AOI, so terrain derivatives are right at its edge and stream
#: catchments that drain into the AOI from outside are delineated in full (``gossan analyze``).
BUFFER_M = 10_000.0
LICENSE = "Open Government Licence - Canada"


class MrdemConnector(Connector):
    """MRDEM-30 bare-earth elevation for the AOI, written to ``rasters/dem.tif``."""

    name = "dem"
    layer = "dem"
    source = "Natural Resources Canada, MRDEM-30 (CanElevation)"
    license = LICENSE
    jurisdiction = "bc"  # Canada-wide; BC is the only Canadian jurisdiction Gossan has
    output_description = "raster(s) written to rasters/"

    def search(self, aoi: gpd.GeoDataFrame) -> list[SearchItem]:
        minx, miny, maxx, maxy = aoi.to_crs(MRDEM_CRS).total_bounds
        bounds = (minx - BUFFER_M, miny - BUFFER_M, maxx + BUFFER_M, maxy + BUFFER_M)
        return [
            SearchItem(
                identifier="mrdem-30-dtm",
                url=MRDEM_URL,
                parameters={"bounds_epsg3979": [round(b, 1) for b in bounds], "buffer_m": BUFFER_M},
            )
        ]

    def fetch(self, items: list[SearchItem]) -> list[Path]:
        raw = self.project.raw_dir / self.name
        raw.mkdir(parents=True, exist_ok=True)
        files = []
        for item in items:
            retrieved_at = datetime.now(UTC)
            data = self._read_window(item.url, tuple(item.parameters["bounds_epsg3979"]))
            path = raw / f"mrdem30_{retrieved_at:%Y%m%dT%H%M%S%fZ}.tif"
            with path.open("xb") as f:  # "x": fail rather than overwrite an original
                f.write(data)
            self.project.provenance.log(
                dataset=self.layer,
                source=self.source,
                identifier=item.identifier,
                url=item.url,
                retrieved_at=retrieved_at,
                license=self.license,
                operation="fetch",
                parameters={**item.parameters, "bytes": len(data), "file": f"raw/{self.name}/{path.name}"},
            )
            files.append(path)
        return files

    def load(self, files: list[Path]) -> list[Path]:
        """Clip and reproject the fetched window to ``rasters/dem.tif``."""
        outputs = []
        for f in files:
            dst = raster_path(self.project, self.layer)
            facts = write_cog(self.project, f, dst, buffer_m=BUFFER_M)  # derived: replaced on re-harvest
            self.project.provenance.log(
                dataset=self.layer,
                source=self.source,
                identifier=f"{RASTERS_DIR}/{dst.name}",
                url=None,
                license=self.license,
                operation="write_raster",
                parameters={"input": f.name, **facts},
            )
            outputs.append(dst)
        return outputs

    def run(self) -> int:
        """Fetch the AOI window and write the DEM COG; returns the number of rasters written."""
        return len(self.load(self.fetch(self.search(self.project.read_aoi()))))

    def _read_window(self, url: str, bounds: tuple[float, ...]) -> bytes:
        """GeoTIFF bytes of the COG window covering ``bounds``; the single network call, replaced
        in tests."""
        with rasterio.open(f"/vsicurl/{url}") as src:
            window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            data = src.read(1, window=window)
            profile = {
                "driver": "GTiff",
                "width": data.shape[1],
                "height": data.shape[0],
                "count": 1,
                "dtype": data.dtype,
                "crs": src.crs,
                "transform": src.window_transform(window),
                "nodata": src.nodata,
                "compress": "deflate",
            }
        with MemoryFile() as mem:
            with mem.open(**profile) as dst:
                dst.write(data, 1)
            return mem.read()
