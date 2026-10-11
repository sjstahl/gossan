"""BC Geological Survey digital geology: bedrock units, faults, Quaternary cover, mineral potential.

Access method: the DataBC public WFS (see ``gossan.harvest.bc_provincial.wfs``). All four layers
use the geometry property ``GEOMETRY``.

``bedrock_geology``: ``pub:WHSE_MINERAL_TENURE.GEOL_BEDROCK_UNIT_POLY_SVW`` ("Bedrock Geology"),
the provincial 1:50 000 to 1:250 000 compilation. One polygon per map unit, with stratigraphic
name, group/formation, rock type and class, era/period, minimum and maximum age, terrane and
morphotectonic belt. Whole polygons that touch the AOI are kept, so some extend well past it.
Catalogue: https://catalogue.data.gov.bc.ca/dataset/bedrock-geology
Terms: Open Government Licence - British Columbia.

``faults``: ``pub:WHSE_MINERAL_TENURE.GEOL_FAULT_LINE`` ("Geology Faults"), the faults of the same
compilation, with ``FAULT_TYPE`` (thrust, normal, strike-slip, ...).
Catalogue: https://catalogue.data.gov.bc.ca/dataset/c94e0c13-5385-49c1-9922-822e10135fc6
Terms: Open Government Licence - British Columbia.

``quaternary_cover``: ``pub:WHSE_MINERAL_TENURE.GEOL_QUATERNARY_POLY`` ("Geology Quaternary
Alluvium and Cover"): areas where bedrock is hidden by young cover. Useful to explain gaps in
geochemistry and outcrop. Terms: **Access Only** (catalogue
https://catalogue.data.gov.bc.ca/dataset/b3f58ed8-376f-4962-9657-36297a5f41cf): internal use.

``mineral_potential``: ``pub:WHSE_MINERAL_TENURE.MINPOT_MINERAL_POTENTIAL`` ("Mineral Potential"),
the BCGS Mineral Potential Project tracts (1990s), with metallic and industrial mineral ranks,
occurrence counts, past production and exploration expenditure per tract. Coarse but an
independent expert assessment. Terms: **Access Only** (catalogue
https://catalogue.data.gov.bc.ca/dataset/df9b5a9a-9a4b-4164-81f7-c3bafbc22d71): internal use.
"""

from __future__ import annotations

from gossan.harvest.bc_provincial.wfs import ACCESS_ONLY_BC, OGL_BC, WfsConnector

BCGS = "BC Geological Survey (DataBC WFS)"


class BedrockGeologyConnector(WfsConnector):
    """Bedrock geology polygons touching the AOI, written to ``bedrock_geology``."""

    name = "bedrock_geology"
    source = BCGS
    license = OGL_BC
    type_name = "pub:WHSE_MINERAL_TENURE.GEOL_BEDROCK_UNIT_POLY_SVW"
    geometry_column = "GEOMETRY"


class FaultsConnector(WfsConnector):
    """Fault lines touching the AOI, written to ``faults``."""

    name = "faults"
    source = BCGS
    license = OGL_BC
    type_name = "pub:WHSE_MINERAL_TENURE.GEOL_FAULT_LINE"
    geometry_column = "GEOMETRY"


class QuaternaryCoverConnector(WfsConnector):
    """Quaternary cover polygons touching the AOI, written to ``quaternary_cover``."""

    name = "quaternary_cover"
    source = BCGS
    license = ACCESS_ONLY_BC
    type_name = "pub:WHSE_MINERAL_TENURE.GEOL_QUATERNARY_POLY"
    geometry_column = "GEOMETRY"


class MineralPotentialConnector(WfsConnector):
    """Mineral Potential Project tracts touching the AOI, written to ``mineral_potential``."""

    name = "mineral_potential"
    source = BCGS
    license = ACCESS_ONLY_BC
    type_name = "pub:WHSE_MINERAL_TENURE.MINPOT_MINERAL_POTENTIAL"
    geometry_column = "GEOMETRY"
