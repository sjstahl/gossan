"""BC Mineral Titles Online (MTO) tenure polygons.

Access method: the DataBC public WFS, layer
``pub:WHSE_MINERAL_TENURE.MTA_ACQUIRED_TENURE_SVW`` ("MTA - Mineral, Placer and
Coal Tenure Spatial View"); see ``gossan.harvest.bc_provincial.wfs`` for the request details.
The geometry property of this layer is ``GEOMETRY`` (not ``SHAPE``). The layer holds
mineral, placer and coal tenure; ``TENURE_TYPE_CODE`` distinguishes them. Whole
tenure polygons that intersect the AOI are kept.

This is a snapshot of *current* tenure. The layer does not carry change history;
detecting changes (Phase 3 leads) means comparing snapshots taken over time.

Catalogue record: https://catalogue.data.gov.bc.ca/dataset/923c5330-c798-4276-82c1-705000c5caac

Terms of use: Open Government Licence - British Columbia
(https://www2.gov.bc.ca/gov/content/data/policy-standards/open-data/open-government-licence-bc),
as stated on the catalogue record (publisher: Mineral Titles). Reuse is permitted with
attribution: "Contains information licensed under the Open Government Licence -
British Columbia." Owner names are published in this layer; treat them as public
record but do not use them for purposes beyond the licence.
"""

from __future__ import annotations

from gossan.harvest.bc_provincial.wfs import OGL_BC, WfsConnector


class MtoTenureConnector(WfsConnector):
    """Tenure polygons intersecting the AOI, written to the ``mto_tenure`` layer."""

    name = "mto_tenure"
    source = "BC Mineral Titles Online (DataBC WFS)"
    license = OGL_BC
    type_name = "pub:WHSE_MINERAL_TENURE.MTA_ACQUIRED_TENURE_SVW"
    geometry_column = "GEOMETRY"
