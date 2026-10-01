"""BC Assessment Report Indexing System (ARIS): the spatial index only.

Access method: the DataBC public WFS, layer
``pub:WHSE_MINERAL_TENURE.ARIS_MINERAL_REPORTS`` ("Assessment Report Index System
(ARIS) Database"); see ``gossan.harvest.bc.wfs`` for the request details. Each feature
is one report, as a point at the report's property location (not a footprint), with
year, operator, work types, claim names and ``ARIS_REPORT_URL``. This connector does
not download report PDFs.

Note: the layer named in the original plan, ``WHSE_GEOSCIENCE.GSC_ASSESSMENT_REPORTS_SP``,
does not exist on this WFS; the ARIS index is published under WHSE_MINERAL_TENURE.

Catalogue record: https://catalogue.data.gov.bc.ca/dataset/af68d1d1-b85e-4ea9-b5f3-e09fc83f3f4f

Terms of use: the catalogue lists this dataset as **Access Only**, not under the Open
Government Licence. Per the Province's copyright policy
(https://www2.gov.bc.ca/gov/content?id=1AAACC9C65754E4D89A118B875E0FBDA), access-only
materials may not be reproduced or redistributed without written permission from the
Province. Use the index for internal analysis and lead generation; get written
permission (or confirm the terms with the BC Geological Survey) before reproducing it
in client deliverables.
"""

from __future__ import annotations

from gossan.harvest.bc.wfs import WfsConnector


class ArisIndexConnector(WfsConnector):
    """ARIS report locations intersecting the AOI, written to the ``aris_index`` layer."""

    name = "aris_index"
    source = "BC Geological Survey, ARIS (DataBC WFS)"
    license = (
        "Access Only (BC Data Catalogue): no reproduction or redistribution "
        "without written permission of the Province of BC"
    )
    type_name = "pub:WHSE_MINERAL_TENURE.ARIS_MINERAL_REPORTS"
    geometry_column = "SHAPE"
