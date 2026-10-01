"""ASTER L1T precision terrain corrected at-sensor radiance (Terra).

Access method: NASA's CMR-STAC catalogue for LP DAAC,
``https://cmr.earthdata.nasa.gov/stac/LPCLOUD`` (``POST /search``, no credentials needed),
collection ``AST_L1T_004`` (catalogue record
https://cmr.earthdata.nasa.gov/search/concepts/C3306888411-LPCLOUD.html). Change it with
``satellite.aster_collection`` in ``project.yaml``.

Downloads need a NASA Earthdata Login bearer token in ``EARTHDATA_TOKEN`` (see ``emit.py``).
The download path has not been exercised against the live service from this code.

What is downloaded per scene: one GeoTIFF per VNIR and SWIR band (``VNIR_B01``, ``B02``,
``B03N``, ``SWIR_B04`` to ``B09``). The thermal bands and the stacked files are skipped.
Whether these GeoTIFFs are Cloud Optimized was not checked, so they are kept in
``raw/satellite/aster/`` as delivered, not placed in ``rasters/``.

**ASTER's SWIR bands are unusable for scenes acquired after April 2008** (the SWIR detectors
failed). Unless ``satellite.datetime`` is set, this connector therefore only searches up to
2008-04-30. Set ``satellite.datetime`` explicitly (for example ``"2000-01-01/.."``) if you
want later scenes for their VNIR or thermal data.

Terms of use: NASA EOSDIS Data Use Policy; the collection is listed as CC0-1.0. Free, no
restrictions; cite the dataset (https://earthdata.nasa.gov/earth-observation-data/data-use-policy).
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from gossan.harvest.satellite.emit import EARTHDATA_LICENSE, earthdata_headers
from gossan.harvest.satellite.stac import StacSatelliteConnector

SWIR_FAILURE_CUTOFF = "2008-04-30T23:59:59Z"
_BAND_FILE = re.compile(r"_(VNIR_B0[123]N?|SWIR_B0[4-9])\.tif$")


class AsterConnector(StacSatelliteConnector):
    """ASTER L1T scenes intersecting the AOI (VNIR and SWIR bands)."""

    name = "aster"
    source = "NASA LP DAAC, ASTER L1T (CMR-STAC)"
    license = EARTHDATA_LICENSE
    stac_url = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
    collection_setting = "aster_collection"

    def default_datetime(self) -> str | None:
        return f"../{SWIR_FAILURE_CUTOFF}"

    def select_assets(self, item: dict) -> list[tuple[str, str]]:
        out = []
        for key, asset in item.get("assets", {}).items():
            href = asset.get("href", "")
            if href.startswith("https://") and _BAND_FILE.search(urlparse(href).path):
                out.append((key, href))
        return out

    def auth_headers(self, url: str) -> dict[str, str]:
        return earthdata_headers(self)
