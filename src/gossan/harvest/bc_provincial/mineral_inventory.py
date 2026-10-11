"""MINFILE production and inventory (resource/reserve) records.

Access method: the DataBC public WFS (see ``gossan.harvest.bc_provincial.wfs``). Both layers are
points at the MINFILE occurrence location, geometry property ``SHAPE``, keyed by
``MINFILE_NUMBER`` so they join to the ``minfile`` layer.

``minfile_production``: ``pub:WHSE_MINERAL_TENURE.MINFIL_PRODUCT`` ("MINFILE Production Database"):
one record per producing occurrence with mined and milled tonnes, first and last year mined, and
up to eight commodities with quantities (``COMMODITY_QUANTITYn``; units per the MINFILE coding
manual: grams for precious metals, kilograms for base metals, tonnes for industrial minerals).
Catalogue: https://catalogue.data.gov.bc.ca/dataset/5ea4ccb3-7174-498d-8679-f51dbf714d29

``minfile_inventory``: ``pub:WHSE_MINERAL_TENURE.MINFIL_RESERVE`` ("MINFILE Inventory Database"):
one record per ore zone and estimate, with year, category (``INVENTORY_CATEGORY_DESCRIPTION``,
which includes historical, non-NI 43-101 categories and single assays), tonnage and grades.
Treat it as historical information, not a current resource.
Catalogue: https://catalogue.data.gov.bc.ca/dataset/925bb403-439c-4a35-8cfc-02ae1a645f6d

Terms of use: both are listed as **Access Only** on the BC Data Catalogue: internal use; no
reproduction in deliverables without the Province's written permission.
"""

from __future__ import annotations

from gossan.harvest.bc_provincial.wfs import ACCESS_ONLY_BC, WfsConnector

BCGS = "BC Geological Survey, MINFILE (DataBC WFS)"


class MinfileProductionConnector(WfsConnector):
    """MINFILE production records in the AOI, written to ``minfile_production``."""

    name = "minfile_production"
    source = BCGS
    license = ACCESS_ONLY_BC
    type_name = "pub:WHSE_MINERAL_TENURE.MINFIL_PRODUCT"
    geometry_column = "SHAPE"


class MinfileInventoryConnector(WfsConnector):
    """MINFILE inventory (historical resource/reserve) records in the AOI, written to
    ``minfile_inventory``."""

    name = "minfile_inventory"
    source = BCGS
    license = ACCESS_ONLY_BC
    type_name = "pub:WHSE_MINERAL_TENURE.MINFIL_RESERVE"
    geometry_column = "SHAPE"
