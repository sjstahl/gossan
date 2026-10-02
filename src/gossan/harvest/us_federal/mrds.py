"""USGS Mineral Resources Data System (MRDS): mineral occurrences, mapped to the ``minfile`` layer.

Access method: the USGS Mineral Resources Data Systems OGC WFS,
``https://mrdata.usgs.gov/services/mrds``, feature type ``mrds-high`` (no credentials). It is
a MapServer WFS that offers only GML output (no GeoJSON), so the raw copy is GML. It is queried
as WFS 1.0.0 with a bounding box in longitude, latitude order and a very large ``maxFeatures``
(the service has no paging in that version; a bounding box over all of Nevada returned about
17,000 records in a single response). Results are then selected against the AOI polygon locally.
Page: https://mrdata.usgs.gov/mrds/.

Fields in this layer: ``dep_id`` (record id), ``site_name``, ``dev_stat`` (development status),
``code_list`` (commodity codes, e.g. "AU AG CU"), ``url`` (the record page), and FIPS, HUC and
quadrangle codes. Mapping to the canonical ``minfile`` columns:

* MINFILE_NUMBER = dep_id; MINFILE_NAME1 = site_name; STATUS_DESCRIPTION = dev_stat
* COMMODITY_DESCRIPTION1 to 3 = the first three codes of ``code_list`` as names ("AU" becomes
  "Gold"); codes this module does not know stay as the code. All codes are kept in COMMODITY_CODES.
* DEPOSIT_TYPE_DESCRIPTION1 is empty: this layer carries no deposit model.
* LATITUDE_DECIMAL, LONGITUDE_DECIMAL come from the geometry; MINFILE_SUMMARY_URL = url.

Caveats: USGS stopped systematic updates of MRDS in 2011 (the page says it is working on a
successor for the conterminous US), so it misses newer discoveries and its coordinates come from
many sources of varying quality. Treat it as a historical inventory, not a current one.

Terms of use: MRDS is a USGS publication. The data page states no licence; U.S. Government
works are generally in the public domain (17 U.S.C. 105), but no explicit statement was found,
so confirm before relying on that for redistribution. The page asks users to cite Mason and
Arndt (1996, USGS Data Series 20) and Schweitzer (2019).
"""

from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd

from gossan.core.schemas import MINFILE_LAYER, conform
from gossan.harvest.base import SearchItem
from gossan.harvest.vector_service import Download, VectorServiceConnector

WFS_URL = "https://mrdata.usgs.gov/services/mrds"
TYPE_NAME = "mrds-high"
MAX_FEATURES = 200_000
LICENSE = "Public domain (U.S. Government work, USGS); no explicit licence stated; cite Mason and Arndt 1996 / Schweitzer 2019"

#: MRDS commodity codes (mostly element symbols). Unknown codes are kept as they are.
COMMODITY_NAMES = {
    "AG": "Silver", "AL": "Aluminum", "AS": "Arsenic", "AU": "Gold", "B": "Boron", "BA": "Barium",
    "BE": "Beryllium", "BI": "Bismuth", "CA": "Calcium", "CD": "Cadmium", "CE": "Cerium",
    "CO": "Cobalt", "CR": "Chromium", "CS": "Cesium", "CU": "Copper", "F": "Fluorine",
    "FE": "Iron", "GA": "Gallium", "GE": "Germanium", "HG": "Mercury", "IN": "Indium",
    "K": "Potassium", "LI": "Lithium", "MG": "Magnesium", "MN": "Manganese", "MO": "Molybdenum",
    "NA": "Sodium", "NB": "Niobium", "NI": "Nickel", "P": "Phosphorus", "PB": "Lead",
    "PD": "Palladium", "PT": "Platinum", "RB": "Rubidium", "RE": "Rhenium", "S": "Sulfur",
    "SB": "Antimony", "SC": "Scandium", "SE": "Selenium", "SI": "Silicon", "SN": "Tin",
    "SR": "Strontium", "TA": "Tantalum", "TE": "Tellurium", "TH": "Thorium", "TI": "Titanium",
    "U": "Uranium", "V": "Vanadium", "W": "Tungsten", "Y": "Yttrium", "ZN": "Zinc", "ZR": "Zirconium",
    "REE": "Rare earths",
}  # fmt: skip


def commodity_names(code_list: object) -> list[str]:
    """['AU', 'AG', 'XX'] from ' AU AG XX' -> ['Gold', 'Silver', 'XX']; empty for missing values."""
    if not isinstance(code_list, str):
        return []
    return [COMMODITY_NAMES.get(code.upper(), code) for code in code_list.split()]


class MrdsConnector(VectorServiceConnector):
    """MRDS occurrences intersecting the AOI, written to the ``minfile`` layer."""

    name = "mrds"
    canonical_layer = MINFILE_LAYER
    jurisdiction = "us"
    source = "USGS Mineral Resources Data System (mrdata WFS)"
    license = LICENSE

    def search(self, aoi: gpd.GeoDataFrame) -> list[SearchItem]:
        minx, miny, maxx, maxy = aoi.to_crs("EPSG:4326").total_bounds
        params = {
            "service": "WFS",
            "version": "1.0.0",  # 1.0.0 takes the bounding box as lon,lat; 1.1.0 would need lat,lon
            "request": "GetFeature",
            "typename": TYPE_NAME,
            "maxFeatures": MAX_FEATURES,
            "bbox": f"{minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}",
        }
        return [SearchItem(identifier=TYPE_NAME, url=f"{WFS_URL}?{urlencode(params)}", parameters=params)]

    def download(self, item: SearchItem) -> Download:
        content = self._get(item.url)
        head = content[:4000].lower()
        if b"exceptionreport" in head or b"serviceexception" in head:
            raise OSError("MRDS WFS returned an error: " + content[:400].decode("utf-8", errors="replace").replace("\n", " "))
        n = content.count(b"<gml:featureMember>")
        if n >= MAX_FEATURES:
            raise OSError(f"MRDS returned {n} features, the most it was asked for: the result may be truncated")
        return Download(content, "gml", n, {"bbox": item.parameters["bbox"], "typename": TYPE_NAME})

    def read_raw(self, path: Path) -> gpd.GeoDataFrame | None:
        content = path.read_bytes()
        if b"<gml:featureMember>" not in content:
            return None
        # Read from memory: opening the GML file itself would make GDAL write a .gfs sidecar next
        # to the original in raw/, and originals are immutable.
        return gpd.read_file(io.BytesIO(content), engine="pyogrio")

    def normalize(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        out = gdf.copy()
        names = out["code_list"].map(commodity_names) if "code_list" in out else [[] for _ in range(len(out))]
        wgs = out.geometry.to_crs("EPSG:4326")
        out["MINFILE_NUMBER"] = out["dep_id"]
        out["MINFILE_NAME1"] = out["site_name"]
        out["STATUS_DESCRIPTION"] = out["dev_stat"]
        for i in range(3):
            out[f"COMMODITY_DESCRIPTION{i + 1}"] = [n[i] if len(n) > i else None for n in names]
        out["COMMODITY_CODES"] = out["code_list"].map(lambda v: " ".join(v.split()) if isinstance(v, str) else None)
        out["LATITUDE_DECIMAL"] = wgs.y
        out["LONGITUDE_DECIMAL"] = wgs.x
        out["MINFILE_SUMMARY_URL"] = out["url"]
        return conform(out, MINFILE_LAYER)
