"""Past and present mining activity: historic Notices of Work, permitted mine areas, crown grants.

Access method: the DataBC public WFS (see ``gossan.harvest.bc_provincial.wfs``).

``historic_notices``: ``pub:WHSE_MINERAL_TENURE.MMS_NOTICE_OF_WORK`` ("Historic (MMS) - Notice of
Work (NOW) Spatial Locations"), points (``GEOMETRY``) for Notices of Work from the legacy Mine
Management System, with application type and date, project and property name and work dates.
The modern equivalent (with drilling only) is ``drilling_notices``. Terms: **Access Only**
(https://catalogue.data.gov.bc.ca/dataset/636822f1-8eaa-4a88-be69-9b9c12f80a20).

``permitted_mine_areas``: ``pub:WHSE_MINERAL_TENURE.HSP_NOTICE_OF_WORK_PA_POLY_SV`` ("Notice of Work
(NoW) - Permitted Mine Areas - Regional Mine - Public"), polygons (``SHAPE``) of permitted areas for
regional (non-major) mines and exploration, with the same fields as the NoW point layer. Terms:
Open Government Licence - British Columbia
(https://catalogue.data.gov.bc.ca/dataset/c728435d-410e-42f9-81d5-95978c90e44a).

``major_mine_areas``: ``pub:WHSE_MINERAL_TENURE.HSP_MJR_MINES_PERMTTD_AREAS_SP`` ("Permitted Mine
Areas - Major Mine"), polygons (``SHAPE``) with mine name, permit, permittee and operating status.
Terms: **Access Only** (https://catalogue.data.gov.bc.ca/dataset/01e8a35e-35e3-4b48-93a7-b0d7c9705b62).

``crown_grants``: ``pub:WHSE_MINERAL_TENURE.MTA_CROWN_GRANT_MIN_CLAIM_SVW`` ("MTA - Crown Granted
Mineral Claims"), polygons (``SHAPE``) of surveyed crown-granted claims with claim name and lot.
Crown grants mark historic workings (mostly pre-1950) and are private mineral title, so they are
not open for staking. Terms: **Access Only**
(https://catalogue.data.gov.bc.ca/dataset/247d2589-8a6b-46c2-a8b1-068769ecee17).

Layers include permittee and owner names; treat them as public record within the licence.
"""

from __future__ import annotations

from gossan.harvest.bc_provincial.wfs import ACCESS_ONLY_BC, OGL_BC, WfsConnector


class HistoricNoticesConnector(WfsConnector):
    """Historic (MMS) Notice of Work points in the AOI, written to ``historic_notices``."""

    name = "historic_notices"
    source = "BC Mines Act Notice of Work, historic MMS (DataBC WFS)"
    license = ACCESS_ONLY_BC
    type_name = "pub:WHSE_MINERAL_TENURE.MMS_NOTICE_OF_WORK"
    geometry_column = "GEOMETRY"


class PermittedMineAreasConnector(WfsConnector):
    """Regional mine permitted-area polygons touching the AOI, written to ``permitted_mine_areas``."""

    name = "permitted_mine_areas"
    source = "BC Mines Act Notice of Work, permitted mine areas (DataBC WFS)"
    license = OGL_BC
    type_name = "pub:WHSE_MINERAL_TENURE.HSP_NOTICE_OF_WORK_PA_POLY_SV"
    geometry_column = "SHAPE"


class MajorMineAreasConnector(WfsConnector):
    """Major mine permitted-area polygons touching the AOI, written to ``major_mine_areas``."""

    name = "major_mine_areas"
    source = "BC Mines Act, major mine permitted areas (DataBC WFS)"
    license = ACCESS_ONLY_BC
    type_name = "pub:WHSE_MINERAL_TENURE.HSP_MJR_MINES_PERMTTD_AREAS_SP"
    geometry_column = "SHAPE"


class CrownGrantsConnector(WfsConnector):
    """Crown-granted mineral claim polygons touching the AOI, written to ``crown_grants``."""

    name = "crown_grants"
    source = "BC Mineral Titles Online, crown granted claims (DataBC WFS)"
    license = ACCESS_ONLY_BC
    type_name = "pub:WHSE_MINERAL_TENURE.MTA_CROWN_GRANT_MIN_CLAIM_SVW"
    geometry_column = "SHAPE"
