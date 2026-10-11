"""Land status: where mineral exploration is excluded, restricted or needs extra engagement.

These layers do not say where mineralization is; they say whether a target is worth pursuing.
Mineral tenure cannot be acquired in parks, ecological reserves, protected areas or conservancies,
nor on Indian reserves or treaty settlement lands (Mineral Tenure Act); reserve sites restrict
some or all tenure types. Always confirm land status on Mineral Titles Online before acting.

Access method: the DataBC public WFS (see ``gossan.harvest.bc_provincial.wfs``).

``parks_protected``: ``pub:WHSE_TANTALIS.TA_PARK_ECORES_PA_SVW`` ("BC Parks, Ecological Reserves, and
Protected Areas"), geometry ``SHAPE``. Terms: Open Government Licence - British Columbia
(https://catalogue.data.gov.bc.ca/dataset/1130248f-f1a3-4956-8b2e-38d29d3e4af7).

``conservancies``: ``pub:WHSE_TANTALIS.TA_CONSERVANCY_AREAS_SVW`` ("TANTALIS - Conservancy Areas"),
geometry ``SHAPE``. Terms: Open Government Licence - British Columbia
(https://catalogue.data.gov.bc.ca/dataset/550b3133-2004-468f-ba1f-b95d0e281e78).

``national_parks``: ``pub:WHSE_ADMIN_BOUNDARIES.CLAB_NATIONAL_PARKS`` ("National Parks of Canada
within British Columbia"), geometry ``GEOMETRY``. Terms: **Access Only** on the BC catalogue
(https://catalogue.data.gov.bc.ca/dataset/88e61a14-19a0-46ab-bdae-f68401d3d0fb).

``indian_reserves``: ``pub:WHSE_ADMIN_BOUNDARIES.CLAB_INDIAN_RESERVES`` ("Indian Reserves -
Administrative Boundaries", from the federal Canada Lands Administrative Boundaries), geometry
``GEOMETRY``. Terms: Open Government Licence - Canada
(https://catalogue.data.gov.bc.ca/dataset/8efe9193-80d2-4fdf-a18c-d531a94196ad).

``treaty_lands``: ``pub:WHSE_LEGAL_ADMIN_BOUNDARIES.FNT_TREATY_LAND_SP`` ("First Nations Treaty
Lands"), geometry ``GEOMETRY``. Terms: **Access Only**
(https://catalogue.data.gov.bc.ca/dataset/fd34808d-4be6-45ea-bc23-65109147933a).

``reserve_sites``: ``pub:WHSE_MINERAL_TENURE.MTA_SITE_SP`` ("MTA - Mineral Reserve Sites Business
View"), geometry ``GEOMETRY``: Mineral Titles reserves (no-registration and conditional reserves,
designated placer areas, ...), with ``MTA_SITE_ORDER_RESTR_DESC`` and which tenure types
(``SITE_MINERAL_IND``, ``SITE_PLACER_IND``, ``SITE_COAL_IND``) they restrict. Terms: Open Government
Licence - British Columbia (https://catalogue.data.gov.bc.ca/dataset/fef4b3ac-2cfc-4d30-afa7-2f29d9d25a10).

Absence from these layers is not proof of open ground: other designations exist (for example
critical habitat, community watersheds, Agricultural Land Reserve) and boundaries change.
"""

from __future__ import annotations

from gossan.harvest.bc_provincial.wfs import (
    ACCESS_ONLY_BC,
    OGL_BC,
    OGL_CANADA,
    WfsConnector,
)


class ParksProtectedConnector(WfsConnector):
    """BC parks, ecological reserves and protected areas touching the AOI."""

    name = "parks_protected"
    source = "BC Parks / TANTALIS (DataBC WFS)"
    license = OGL_BC
    type_name = "pub:WHSE_TANTALIS.TA_PARK_ECORES_PA_SVW"
    geometry_column = "SHAPE"


class ConservanciesConnector(WfsConnector):
    """Conservancy areas touching the AOI."""

    name = "conservancies"
    source = "TANTALIS (DataBC WFS)"
    license = OGL_BC
    type_name = "pub:WHSE_TANTALIS.TA_CONSERVANCY_AREAS_SVW"
    geometry_column = "SHAPE"


class NationalParksConnector(WfsConnector):
    """National parks touching the AOI."""

    name = "national_parks"
    source = "Canada Lands Administrative Boundaries (DataBC WFS)"
    license = ACCESS_ONLY_BC
    type_name = "pub:WHSE_ADMIN_BOUNDARIES.CLAB_NATIONAL_PARKS"
    geometry_column = "GEOMETRY"


class IndianReservesConnector(WfsConnector):
    """Indian reserves touching the AOI."""

    name = "indian_reserves"
    source = "Canada Lands Administrative Boundaries (DataBC WFS)"
    license = OGL_CANADA
    type_name = "pub:WHSE_ADMIN_BOUNDARIES.CLAB_INDIAN_RESERVES"
    geometry_column = "GEOMETRY"


class TreatyLandsConnector(WfsConnector):
    """First Nations treaty lands touching the AOI."""

    name = "treaty_lands"
    source = "BC Ministry of Indigenous Relations and Reconciliation (DataBC WFS)"
    license = ACCESS_ONLY_BC
    type_name = "pub:WHSE_LEGAL_ADMIN_BOUNDARIES.FNT_TREATY_LAND_SP"
    geometry_column = "GEOMETRY"


class ReserveSitesConnector(WfsConnector):
    """Mineral Titles reserve sites touching the AOI."""

    name = "reserve_sites"
    source = "BC Mineral Titles Online, reserve sites (DataBC WFS)"
    license = OGL_BC
    type_name = "pub:WHSE_MINERAL_TENURE.MTA_SITE_SP"
    geometry_column = "GEOMETRY"
