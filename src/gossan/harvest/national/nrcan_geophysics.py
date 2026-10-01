"""NRCan geophysical grids (magnetics, gravity) via a Web Coverage Service.

**Status: no usable NRCan WCS was found, so the normal result is the manual fallback.**
Checked 2026-10-01:

* The open.canada.ca records for "Canada - Aeromagnetic survey data compilation" and
  "Canada - Gravity data compilation" list only WMS (map images, not data values) and
  ZIP downloads from the Geoscience Data Repository (GDR, ``gdr.agg.nrcan.gc.ca``).
  No WCS endpoint or coverage identifiers are documented.
* ``wms.agg.nrcan.gc.ca`` and ``gdr.agg.nrcan.gc.ca`` resolved but did not accept
  connections from the development machine (timeouts on ports 80 and 443).

So the WCS path below is generic OGC WCS 2.0.1 and is only used if you configure a
service; it has **not** been verified against a live NRCan server (it is tested against
mocked responses). Configure with environment variables:

* ``GOSSAN_NRCAN_WCS_URL``: the WCS endpoint.
* ``GOSSAN_NRCAN_MAG_COVERAGE`` / ``GOSSAN_NRCAN_GRAV_COVERAGE``: coverage ids for the
  magnetic and gravity grids (each product is optional).

The request subsets with axis labels ``Long`` and ``Lat`` in EPSG:4326 and asks for
``image/tiff``. Anything else (endpoint unset, connection failure, an XML exception
instead of a GeoTIFF) raises ``ManualDownloadRequired`` with instructions.

Manual route: download the grid from the open.canada.ca record (links to the GDR ZIPs),
extract the GeoTIFF, then ``gossan ingest <project> <file.tif> --layer nrcan_magnetics``
(or ``nrcan_gravity``), which clips to the AOI and writes a COG to ``rasters/``.

Terms of use: Open Government Licence - Canada
(https://open.canada.ca/en/open-government-licence-canada), as stated on both
open.canada.ca records. Attribution: "Contains information licensed under the Open
Government Licence - Canada."
"""

from __future__ import annotations

import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd

from gossan.core.raster import RASTERS_DIR, raster_path, write_cog
from gossan.harvest.base import Connector, SearchItem

LICENSE = "Open Government Licence - Canada"
URL_VAR = "GOSSAN_NRCAN_WCS_URL"
# product name (also the COG name) -> (environment variable holding the coverage id, record)
PRODUCTS = {
    "nrcan_magnetics": (
        "GOSSAN_NRCAN_MAG_COVERAGE",
        "https://open.canada.ca/data/en/dataset/752fe3fc-d871-5ae1-9bd9-2b6f65880a8d",
    ),
    "nrcan_gravity": (
        "GOSSAN_NRCAN_GRAV_COVERAGE",
        "https://open.canada.ca/data/en/dataset/5a4e46fe-3e52-57ce-9335-832b5e79fecc",
    ),
}


class ManualDownloadRequired(Exception):
    """The WCS cannot be used; the message tells the user how to get the data by hand."""


def manual_instructions(reason: str) -> str:
    records = "\n".join(f"  - {p}: {url}" for p, (_, url) in PRODUCTS.items())
    return (
        f"NRCan geophysics could not be fetched automatically: {reason}\n"
        "Download the grids manually from these open.canada.ca records (links to the "
        "Geoscience Data Repository ZIPs), extract the GeoTIFF, and load each with "
        "'gossan ingest':\n"
        f"{records}\n"
        '  gossan ingest <project> <grid.tif> --layer nrcan_magnetics '
        '--source "NRCan Geoscience Data Repository" '
        f'--license "{LICENSE}"'
    )


class NrcanGeophysicsConnector(Connector):
    """Magnetic and gravity grids clipped to the AOI, written to ``rasters/`` as COGs."""

    name = "nrcan_geophysics"
    layer = "nrcan_geophysics"
    source = "Natural Resources Canada, Geoscience Data Repository (WCS)"
    license = LICENSE
    #: Not part of a plain ``gossan harvest``: it needs configuration (see module docs).
    default = False
    output_description = "raster(s) written to rasters/"

    def search(self, aoi: gpd.GeoDataFrame) -> list[SearchItem]:
        endpoint = os.environ.get(URL_VAR)
        if not endpoint:
            raise ManualDownloadRequired(manual_instructions(f"{URL_VAR} is not set"))
        minx, miny, maxx, maxy = aoi.to_crs("EPSG:4326").total_bounds
        items = []
        for product, (env_var, _) in PRODUCTS.items():
            coverage = os.environ.get(env_var)
            if not coverage:
                continue
            params = [
                ("service", "WCS"),
                ("version", "2.0.1"),
                ("request", "GetCoverage"),
                ("coverageId", coverage),
                ("format", "image/tiff"),
                ("subsettingCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
                ("subset", f"Long({minx:.6f},{maxx:.6f})"),
                ("subset", f"Lat({miny:.6f},{maxy:.6f})"),
            ]
            items.append(
                SearchItem(
                    identifier=coverage,
                    url=f"{endpoint}{'&' if '?' in endpoint else '?'}{urlencode(params)}",
                    parameters={"product": product, "coverage": coverage},
                )
            )
        if not items:
            raise ManualDownloadRequired(manual_instructions("no coverage ids are configured"))
        return items

    def fetch(self, items: list[SearchItem]) -> list[Path]:
        raw = self.project.raw_dir / self.name
        raw.mkdir(parents=True, exist_ok=True)
        files = []
        for item in items:
            retrieved_at = datetime.now(UTC)
            try:
                data = self._get(item.url)
            except OSError as e:
                raise ManualDownloadRequired(manual_instructions(f"{e}")) from e
            if data[:4] not in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:  # TIFF / BigTIFF
                snippet = data[:200].decode("utf-8", errors="replace").replace("\n", " ")
                reason = f"the service did not return a GeoTIFF ({snippet!r})"
                raise ManualDownloadRequired(manual_instructions(reason))
            product = item.parameters["product"]
            path = raw / f"{product}_{retrieved_at:%Y%m%dT%H%M%S%fZ}.tif"
            with path.open("xb") as f:  # "x": fail rather than overwrite an original
                f.write(data)
            self.project.provenance.log(
                dataset=product,
                source=self.source,
                identifier=item.identifier,
                url=item.url,
                retrieved_at=retrieved_at,
                license=self.license,
                operation="fetch",
                parameters={"bytes": len(data), "file": f"raw/{self.name}/{path.name}"},
            )
            files.append(path)
        return files

    def load(self, files: list[Path]) -> list[Path]:
        """Clip and reproject each fetched grid to a COG in ``rasters/``; returns the COGs."""
        outputs = []
        for f in files:
            product = f.name.rsplit("_", 1)[0]
            dst = raster_path(self.project, product)
            facts = write_cog(self.project, f, dst)  # derived output: replaced on re-harvest
            self.project.provenance.log(
                dataset=product,
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
        """Fetch, clip and write the configured grids; returns the number of COGs written."""
        aoi = self.project.read_aoi()
        return len(self.load(self.fetch(self.search(aoi))))

    def _get(self, url: str) -> bytes:
        """HTTP GET; the single network call, replaced in tests."""
        request = urllib.request.Request(url, headers={"User-Agent": "gossan"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()
