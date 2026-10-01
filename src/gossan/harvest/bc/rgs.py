"""BC Regional Geochemical Survey (RGS) stream-sediment, moss-mat and water data.

Access method: the DataBC public WFS, layer
``pub:WHSE_MINERAL_TENURE.GEOL_REG_GEOCHMCL_SURV_PNT_SP`` ("Regional Geochemical
Survey Data"); see ``gossan.harvest.bc.wfs`` for the request details. One point per
sample, with field observations and one column per element/method pair (for example
``COPPER_AAS_PPM``, ``GOLD_INAA_PPB``). Several analytical methods can report the same
element, and many columns are null for any one sample; values below detection limits
are as published, so check the survey documentation before treating them as numbers.
The geometry property of this layer is ``GEOMETRY``.

Catalogue record: https://catalogue.data.gov.bc.ca/dataset/49a73c36-4a25-4be5-b6d1-abd020fb031a

Terms of use: Open Government Licence - British Columbia
(https://www2.gov.bc.ca/gov/content/data/policy-standards/open-data/open-government-licence-bc),
as stated on the catalogue record (publisher: BC Geological Survey). Reuse is permitted
with attribution: "Contains information licensed under the Open Government Licence -
British Columbia."
"""

from __future__ import annotations

from gossan.harvest.bc.wfs import OGL_BC, WfsConnector


class RgsConnector(WfsConnector):
    """RGS samples intersecting the AOI, written to the ``rgs`` layer."""

    name = "rgs"
    source = "BC Geological Survey, RGS (DataBC WFS)"
    license = OGL_BC
    type_name = "pub:WHSE_MINERAL_TENURE.GEOL_REG_GEOCHMCL_SURV_PNT_SP"
    geometry_column = "GEOMETRY"
