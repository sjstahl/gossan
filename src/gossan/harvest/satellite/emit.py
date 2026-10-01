"""EMIT (Earth Surface Mineral Dust Source Investigation) L2A surface reflectance.

Access method: NASA's CMR-STAC catalogue for LP DAAC,
``https://cmr.earthdata.nasa.gov/stac/LPCLOUD`` (``POST /search``, no credentials needed),
collection ``EMITL2ARFL_001`` ("EMIT L2A Estimated Surface Reflectance and Uncertainty and
Masks 60 m V001"; catalogue record
https://cmr.earthdata.nasa.gov/search/concepts/C2408750690-LPCLOUD.html). Change the
collection with ``satellite.emit_collection`` in ``project.yaml`` (for example
``EMITL1BRAD_001`` for radiance).

Downloads come from ``data.lpdaac.earthdatacloud.nasa.gov`` and need a NASA Earthdata Login
bearer token in the ``EARTHDATA_TOKEN`` environment variable (free account at
https://urs.earthdata.nasa.gov, token under "Generate Token"). The download path has not been
exercised against the live service from this code (the tests mock it); a HEAD request to a scene
answered with a redirect to a signed CloudFront URL.

What is downloaded per scene: the reflectance NetCDF (``..._RFL_...nc``) and the mask
(``..._MASK_...nc``). The uncertainty file is skipped. Expect roughly a gigabyte or more per
scene. EMIT flies on the ISS, so coverage of a given AOI is patchy and mostly at latitudes
below about 52 degrees north.

Terms of use: NASA EOSDIS Data Use Policy; the collection is listed as CC0-1.0. Free, no
restrictions on use or redistribution; NASA asks that the data be cited
(https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance).
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from gossan.harvest.satellite.stac import SatelliteAuthError, StacSatelliteConnector

EARTHDATA_TOKEN = "EARTHDATA_TOKEN"
EARTHDATA_LICENSE = "CC0-1.0 (NASA EOSDIS Data Use Policy): open and free; cite the dataset"


def earthdata_headers(connector: StacSatelliteConnector) -> dict[str, str]:
    token = os.environ.get(EARTHDATA_TOKEN)
    if not token:
        raise SatelliteAuthError(
            f"{EARTHDATA_TOKEN} is not set. Create a free NASA Earthdata Login account, generate a "
            "token at https://urs.earthdata.nasa.gov, and set the variable. " + connector.manual_help()
        )
    return {"Authorization": f"Bearer {token}"}


class EmitConnector(StacSatelliteConnector):
    """EMIT L2A reflectance scenes intersecting the AOI."""

    name = "emit"
    source = "NASA LP DAAC, EMIT L2A reflectance (CMR-STAC)"
    license = EARTHDATA_LICENSE
    stac_url = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
    collection_setting = "emit_collection"

    def select_assets(self, item: dict) -> list[tuple[str, str]]:
        out = []
        for key, asset in item.get("assets", {}).items():
            href = asset.get("href", "")
            fname = urlparse(href).path.rsplit("/", 1)[-1]
            if href.startswith("https://") and fname.endswith(".nc") and ("_RFL_" in fname or "_MASK_" in fname):
                out.append((key, href))
        return out

    def auth_headers(self, url: str) -> dict[str, str]:
        return earthdata_headers(self)
