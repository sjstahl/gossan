"""Public Land Survey System (PLSS) sections as polygons, from the BLM national PLSS service.

State notices give a location as sections, township and range ("Sections 8, 9, and 16, Township 20
North, Range 66 East: Mount Diablo Baseline and Meridian"). This module turns that into the section
polygons, using the BLM National PLSS CadNSDI service (layer 2, "PLSS Section"):

    https://gis.blm.gov/arcgis/rest/services/Cadastral/BLM_Natl_PLSS_CadNSDI/MapServer/2

A section is identified by ``PLSSID`` (the township: state, meridian, township, range) and
``FRSTDIVNO`` (the section number). Example: Nevada (``NV``), Mount Diablo meridian (``21``),
T20N R66E is ``NV210200N0660E0``.

Public domain (U.S. Government work, BLM). Section outlines are the PLSS as recorded by BLM and
are approximate where surveys are incomplete; a section is about one square mile.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlencode

PLSS_SECTION_URL = "https://gis.blm.gov/arcgis/rest/services/Cadastral/BLM_Natl_PLSS_CadNSDI/MapServer/2/query"
#: State code and BLM meridian code of the principal meridians Gossan resolves.
MERIDIANS = {"mount diablo": ("NV", "21")}
LICENSE = "Public domain (U.S. Government work, BLM)"


@dataclass(frozen=True)
class Township:
    township: int
    ns: str  # "N" or "S"
    range: int
    ew: str  # "E" or "W"
    sections: tuple[int, ...]
    meridian: str = "mount diablo"

    def plss_id(self) -> str:
        """The ``PLSSID`` of this township, for example ``NV210200N0660E0``."""
        state, code = MERIDIANS[self.meridian]
        return f"{state}{code}{self.township:03d}0{self.ns}{self.range:03d}0{self.ew}0"


def section_query_url(t: Township) -> str:
    numbers = ",".join(f"'{s:02d}'" for s in t.sections)
    params = {
        "where": f"PLSSID='{t.plss_id()}' AND FRSTDIVNO IN ({numbers})",
        "outFields": "PLSSID,FRSTDIVNO",
        "outSR": "4326",
        "f": "geojson",
    }
    return f"{PLSS_SECTION_URL}?{urlencode(params)}"


def parse_sections(content: bytes) -> list[dict]:
    """GeoJSON geometries (EPSG:4326) from a section query response; [] when nothing matched."""
    data = json.loads(content)
    if "error" in data:
        raise OSError(f"PLSS service error: {data['error']}")
    return [f["geometry"] for f in data.get("features", []) if f.get("geometry")]
