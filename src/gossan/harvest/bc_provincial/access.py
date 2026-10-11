"""Access and infrastructure: roads and transmission lines.

Access method: the DataBC public WFS (see ``gossan.harvest.bc_provincial.wfs``).

``roads``: ``pub:WHSE_BASEMAPPING.DRA_DGTL_ROAD_ATLAS_MPAR_SP`` ("Digital Road Atlas (DRA) - Master
Partially-Attributed Roads"), lines (``GEOMETRY``) with ``ROAD_CLASS`` (highway, arterial, local,
resource, trail, ...) and ``ROAD_SURFACE``. Resource roads are incomplete and some are deactivated;
use it for relative access, not route planning. Large: an AOI of a few thousand km2 can hold tens
of thousands of segments. Terms: **Access Only**
(https://catalogue.data.gov.bc.ca/dataset/bb060417-b6e6-4548-b837-f9060d94743e).

``transmission_lines``: ``pub:WHSE_BASEMAPPING.GBA_TRANSMISSION_LINES_SP`` ("BC Transmission Lines"),
lines (``SHAPE``) with voltage and owner. Terms: Open Government Licence - British Columbia
(https://catalogue.data.gov.bc.ca/dataset/384d551b-dee1-4df8-8148-b3fcf865096a).
"""

from __future__ import annotations

from gossan.harvest.bc_provincial.wfs import ACCESS_ONLY_BC, OGL_BC, WfsConnector


class RoadsConnector(WfsConnector):
    """Digital Road Atlas segments touching the AOI, written to ``roads``."""

    name = "roads"
    source = "BC Digital Road Atlas (DataBC WFS)"
    license = ACCESS_ONLY_BC
    type_name = "pub:WHSE_BASEMAPPING.DRA_DGTL_ROAD_ATLAS_MPAR_SP"
    geometry_column = "GEOMETRY"


class TransmissionLinesConnector(WfsConnector):
    """Transmission lines touching the AOI, written to ``transmission_lines``."""

    name = "transmission_lines"
    source = "BC Transmission Lines (DataBC WFS)"
    license = OGL_BC
    type_name = "pub:WHSE_BASEMAPPING.GBA_TRANSMISSION_LINES_SP"
    geometry_column = "SHAPE"
