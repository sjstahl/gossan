"""EnMAP (Environmental Mapping and Analysis Program) L2A hyperspectral scenes.

Access method: the DLR Earth Observation Center STAC API,
``https://geoservice.dlr.de/eoc/ogc/stac/v1`` (``POST /search``, no credentials needed for
searching), collection ``ENMAP_HSI_L2A`` (CEOS-ARD surface reflectance, 224 bands). Change it
with ``satellite.enmap_collection`` in ``project.yaml``. EnMAP is a tasked satellite: many AOIs
have no scenes.

Downloads come from ``download.geoservice.dlr.de``, which redirects anonymous requests to a DLR
single-sign-on page. This connector sends HTTP Basic credentials from ``EOC_USERNAME`` and
``EOC_PASSWORD`` (an account registered at the EnMAP Instrument Planning Portal,
https://planning.enmap.org). **Whether that authentication method is accepted has not been
verified**: no account was available, so the download path is only tested against mocks. If the
service answers with the SSO login page the connector stops with a message and the manual
route: download the scenes from the EOC Geoservice by hand into
``raw/satellite/enmap/<scene id>/`` (the ``enmap_scenes`` layer lists them).

What is downloaded per scene: the metadata XML, the spectral image (a Cloud Optimized GeoTIFF,
``SPECTRAL_IMAGE_COG.TIF``, typically over a gigabyte) and the pixel quality classes. The files
are in the scene's UTM zone and unclipped, so they stay in ``raw/satellite/enmap/``, not
``rasters/``.

Terms of use: EnMAP Data License v1.1
(https://geoservice.dlr.de/resources/licenses/enmap/EnMAP-Data_License_v1_1.pdf), accepted by
registering at the planning portal. Summary of that document: free of charge, commercial and
non-commercial use; you may copy, process and combine the data; you may share the data with
affiliated users (for example a client in a joint project) who are not on sanctions lists; you
may NOT sell, lease or sub-license the data or pass it to anyone else without DLR's
authorisation. Products that do not keep the original pixel values ("value-added products") are
not covered by these limits. Copyright notices in the data must be kept. Read the licence before
putting raw EnMAP data in a client deliverable.
"""

from __future__ import annotations

import base64
import os

from gossan.harvest.satellite.stac import SatelliteAuthError, StacSatelliteConnector

EOC_LICENSE = (
    "EnMAP Data License v1.1 (DLR): free commercial/non-commercial use after registration; "
    "no sale or redistribution; raw data may be shared only with affiliated users"
)
_WANTED_KEYS = ("metadata", "image", "quality_classes")


class EnmapConnector(StacSatelliteConnector):
    """EnMAP L2A scenes intersecting the AOI."""

    name = "enmap"
    source = "DLR EOC Geoservice, EnMAP L2A (STAC)"
    license = EOC_LICENSE
    stac_url = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
    collection_setting = "enmap_collection"

    def select_assets(self, item: dict) -> list[tuple[str, str]]:
        assets = item.get("assets", {})
        return [(k, assets[k]["href"]) for k in _WANTED_KEYS if k in assets and assets[k].get("href")]

    def auth_headers(self, url: str) -> dict[str, str]:
        user, password = os.environ.get("EOC_USERNAME"), os.environ.get("EOC_PASSWORD")
        if not user or not password:
            raise SatelliteAuthError(
                "EOC_USERNAME and EOC_PASSWORD are not set. Register at https://planning.enmap.org "
                "(this accepts the EnMAP Data License), then set both. " + self.manual_help()
            )
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        return {"Authorization": f"Basic {token}"}
