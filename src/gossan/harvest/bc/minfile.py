"""BC MINFILE mineral occurrences.

Access method: the BC Geographic Warehouse (DataBC) public WFS, layer
``pub:WHSE_MINERAL_TENURE.MINFIL_MINERAL_FILE`` ("MINFILE Mineral Occurrence
Database"); see ``gossan.harvest.bc.wfs`` for the request details.

Catalogue record: https://catalogue.data.gov.bc.ca/dataset/92206d94-bc64-4111-a295-cd14eb5a501c

Terms of use: Open Government Licence - British Columbia
(https://www2.gov.bc.ca/gov/content/data/policy-standards/open-data/open-government-licence-bc).
Reuse, including commercially, is permitted with attribution: "Contains
information licensed under the Open Government Licence - British Columbia."
The source is the BC Geological Survey, Ministry of Mining and Critical Minerals.
"""

from __future__ import annotations

from gossan.harvest.bc.wfs import OGL_BC, WfsConnector


class MinfileConnector(WfsConnector):
    """MINFILE occurrences intersecting the AOI, written to the ``minfile`` layer."""

    name = "minfile"
    source = "BC Geological Survey (DataBC WFS)"
    license = OGL_BC
    type_name = "pub:WHSE_MINERAL_TENURE.MINFIL_MINERAL_FILE"
    geometry_column = "SHAPE"
