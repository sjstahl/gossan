"""Shared logic for satellite scene connectors that speak STAC.

Each connector (EMIT, EnMAP, ASTER) does the same three things:

1. **Search** a STAC API (``POST <stac_url>/search``) for scenes whose footprint intersects
   the AOI, in one collection, optionally within a date range. Cloud cover is filtered on
   our side from ``eo:cloud_cover`` (scenes without it are kept). Result pages are followed
   through the ``next`` links, whether the server uses GET links (NASA CMR) or POST links
   with a body to merge (DLR).
2. **Catalogue** the scenes as footprints in a ``<name>_scenes`` layer of ``data.gpkg``
   (scene id, date, cloud cover, how much of the AOI the scene covers, STAC link), best
   candidates first, and log the search to provenance. This is cheap and always done.
3. **Download** the chosen assets into ``raw/satellite/<name>/<scene id>/``, only for the
   first ``limit_scenes`` scenes (default 0: none). A scene is hundreds of MB to several GB.
   Files are written complete-or-not-at-all, never overwritten, checked for the right file
   signature (an auth failure often returns an HTML login page with status 200), and each
   one is logged to provenance with the scene id, retrieval time, size and SHA-256.

Scenes are stored as delivered: not clipped, reprojected or converted. Spectral processing
is outside Gossan; ``raw/satellite/`` is its input.

Searching needs no credentials. Downloading does; each connector names its environment
variables. If they are missing or rejected, ``SatelliteAuthError`` says how to get the scenes
by hand (the catalogue layer lists the scene ids).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import geopandas as gpd
from shapely.geometry import shape

from gossan.harvest.base import Connector, SearchItem

MAX_RESULTS = 1000  # scenes listed per search, across pages
PAGE_SIZE = 25  # ASTER items carry ~50 assets each; 100 per page makes the gateway return 502
RETRIES = 2
RETRY_PAUSE_SECONDS = 2.0
USER_AGENT = "gossan"


class SatelliteAuthError(Exception):
    """Scenes cannot be downloaded (no credentials, or they were refused); says how to proceed."""


class _SameHostRedirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects but never send an Authorization header to a different host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlparse(newurl).netloc != urlparse(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


@dataclass
class Scene:
    scene_id: str
    datetime: str
    cloud_cover: float | None
    coverage_pct: float
    geometry: Any  # shapely, project CRS
    stac_url: str
    assets: list[tuple[str, str]]  # (key, href) of the assets this connector downloads


@dataclass
class SearchReport:
    n_found: int = 0
    n_after_filters: int = 0
    downloaded: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)

    def text(self, name: str) -> str:
        line = (
            f"{self.n_found} scene(s) found, {self.n_after_filters} after the cloud/date filters; "
            f"footprints in layer {name}_scenes"
        )
        if self.downloaded or self.skipped_existing:
            line += (
                f"; {len(self.downloaded)} scene(s) downloaded to raw/satellite/{name}/, "
                f"{len(self.skipped_existing)} already there"
            )
        return line


class StacSatelliteConnector(Connector):
    """Base class; subclasses fill in the class attributes and ``select_assets``/``auth_headers``."""

    stac_url: str  # STAC API root, without /search
    #: Which ``satellite`` setting in project.yaml names the collection (e.g. "emit_collection").
    collection_setting: str
    default = False  # needs credentials and disk space: ask for it with --source
    output_description = "scene(s) catalogued"

    def __init__(self, project, limit_scenes: int = 0) -> None:
        super().__init__(project)
        self.limit_scenes = limit_scenes
        self.report = SearchReport()

    # --- to be defined by subclasses ----------------------------------------------------

    def select_assets(self, item: dict) -> list[tuple[str, str]]:
        """(key, href) of the STAC assets to download for ``item``."""
        raise NotImplementedError

    def auth_headers(self, url: str) -> dict[str, str]:
        """Headers that authorise a download from ``url``; raises ``SatelliteAuthError``
        if the credentials are not configured."""
        raise NotImplementedError

    def default_datetime(self) -> str | None:
        """Date range used when ``satellite.datetime`` is not set (STAC interval syntax)."""
        return None

    # --- config -------------------------------------------------------------------------

    @property
    def layer(self) -> str:  # type: ignore[override]
        return f"{self.name}_scenes"

    @property
    def collection(self) -> str:
        return getattr(self.project.config.satellite, self.collection_setting)

    @property
    def datetime_range(self) -> str | None:
        return self.project.config.satellite.datetime or self.default_datetime()

    # --- search -------------------------------------------------------------------------

    def search(self, aoi: gpd.GeoDataFrame) -> list[SearchItem]:
        """Scenes intersecting the AOI, best candidates first, as ``SearchItem`` objects whose
        ``parameters`` hold the ``Scene``."""
        aoi_wgs = aoi.to_crs("EPSG:4326")
        footprint = aoi_wgs.union_all().simplify(0.0005)  # keep the request small
        body: dict[str, Any] = {
            "collections": [self.collection],
            "intersects": json.loads(gpd.GeoSeries([footprint]).to_json())["features"][0]["geometry"],
            "limit": PAGE_SIZE,
        }
        if self.datetime_range:
            body["datetime"] = self.datetime_range

        features = self._search_all(body)
        self.report.n_found = len(features)
        max_cloud = self.project.config.satellite.max_cloud_percent
        aoi_proj = aoi.to_crs(self.project.config.crs)
        aoi_union, aoi_area = aoi_proj.union_all(), aoi_proj.union_all().area
        scenes: list[Scene] = []
        for f in features:
            cloud = f["properties"].get("eo:cloud_cover")
            if cloud is not None and cloud > max_cloud:
                continue
            geom = gpd.GeoSeries([shape(f["geometry"])], crs="EPSG:4326").to_crs(self.project.config.crs).iloc[0]
            coverage = 100 * geom.intersection(aoi_union).area / aoi_area if aoi_area else 0.0
            self_link = next((link["href"] for link in f.get("links", []) if link.get("rel") == "self"), "")
            scenes.append(
                Scene(
                    scene_id=f["id"],
                    datetime=f["properties"].get("datetime") or f["properties"].get("start_datetime", ""),
                    cloud_cover=cloud,
                    coverage_pct=round(coverage, 1),
                    geometry=geom,
                    stac_url=self_link,
                    assets=self.select_assets(f),
                )
            )
        # Best first: most of the AOI covered, then least cloud, then newest.
        scenes.sort(key=lambda s: s.datetime, reverse=True)
        scenes.sort(key=lambda s: (-s.coverage_pct, s.cloud_cover if s.cloud_cover is not None else 101))
        self.report.n_after_filters = len(scenes)
        return [
            SearchItem(identifier=s.scene_id, url=s.stac_url, parameters={"scene": s, "body": body})
            for s in scenes
        ]

    def _search_all(self, body: dict) -> list[dict]:
        url = f"{self.stac_url}/search"
        page = self._request_json("POST", url, body)
        features = list(page.get("features", []))
        while len(features) < MAX_RESULTS:
            nxt = next((link for link in page.get("links", []) if link.get("rel") == "next"), None)
            if nxt is None or not page.get("features"):
                break
            if nxt.get("method", "GET").upper() == "POST":
                merged = {**body, **nxt["body"]} if nxt.get("merge") else nxt.get("body", body)
                page = self._request_json("POST", nxt["href"], merged)
            else:
                page = self._request_json("GET", nxt["href"], None)
            features.extend(page.get("features", []))
        return features[:MAX_RESULTS]

    # --- catalogue + download -----------------------------------------------------------

    def fetch(self, items: list[SearchItem]) -> list[Path]:
        """Download the assets of the first ``limit_scenes`` scenes in ``items``."""
        files: list[Path] = []
        for item in items[: self.limit_scenes]:
            scene: Scene = item.parameters["scene"]
            if not scene.assets:
                raise OSError(f"scene {scene.scene_id} has no downloadable assets of the kinds this connector uses")
            folder = self.project.raw_dir / "satellite" / self.name / scene.scene_id
            fetched_any = False
            for key, href in scene.assets:
                dest = folder / Path(urlparse(href).path).name
                if dest.exists():  # never re-download or replace an original
                    files.append(dest)
                    continue
                files.append(self._fetch_asset(scene, key, href, dest))
                fetched_any = True
            (self.report.downloaded if fetched_any else self.report.skipped_existing).append(scene.scene_id)
        return files

    def _fetch_asset(self, scene: Scene, key: str, href: str, dest: Path) -> Path:
        headers = self.auth_headers(href)  # raises before any request if credentials are missing
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        retrieved_at = datetime.now(UTC)
        try:
            self._download(href, part, headers)
            problem = _signature_problem(part)
            if problem:
                raise SatelliteAuthError(
                    f"{dest.name}: {problem}; the credentials were probably rejected. "
                    + self.manual_help(scene)
                )
            sha = _sha256(part)
            size = part.stat().st_size
            part.rename(dest)  # dest was checked absent above; never replaces an original
        finally:
            part.unlink(missing_ok=True)
        self.project.provenance.log(
            dataset=self.name,
            source=self.source,
            identifier=scene.scene_id,
            url=href,
            retrieved_at=retrieved_at,
            license=self.license,
            operation="fetch",
            parameters={
                "collection": self.collection,
                "asset": key,
                "file": dest.relative_to(self.project.root).as_posix(),
                "bytes": size,
                "sha256": sha,
                "scene_datetime": scene.datetime,
                "cloud_cover": scene.cloud_cover,
                "aoi_coverage_pct": scene.coverage_pct,
            },
        )
        return dest

    def manual_help(self, scene: Scene | None = None) -> str:
        return (
            f"Manual fallback: download the scene(s) listed in layer {self.layer} (scene id, STAC "
            f"link {scene.stac_url if scene else 'in the layer'}) from the provider's portal and "
            f"put the files in raw/satellite/{self.name}/<scene id>/."
        )

    def load(self, files: list[Path]) -> list[Path]:
        """Scenes stay as delivered in ``raw/satellite``; nothing is converted here."""
        return files

    def run(self) -> int:
        """Search, write the scene footprints, then download up to ``limit_scenes`` scenes.

        Returns the number of scenes found after filtering.
        """
        aoi = self.project.read_aoi()
        items = self.search(aoi)
        search_url = f"{self.stac_url}/search"
        body = items[0].parameters["body"] if items else None
        self.project.provenance.log(
            dataset=self.layer,
            source=self.source,
            identifier=self.collection,
            url=search_url,
            license=self.license,
            operation="search",
            parameters={
                "request": body,
                "n_found": self.report.n_found,
                "n_after_filters": self.report.n_after_filters,
                "max_cloud_percent": self.project.config.satellite.max_cloud_percent,
                "datetime": self.datetime_range,
            },
        )
        if items:
            scenes = [i.parameters["scene"] for i in items]
            gdf = gpd.GeoDataFrame(
                {
                    "scene_id": [s.scene_id for s in scenes],
                    "collection": self.collection,
                    "datetime": [s.datetime for s in scenes],
                    "cloud_cover": [s.cloud_cover for s in scenes],
                    "aoi_coverage_pct": [s.coverage_pct for s in scenes],
                    "stac_url": [s.stac_url for s in scenes],
                },
                geometry=[s.geometry for s in scenes],
                crs=self.project.config.crs,
            )
            gdf.to_file(self.project.data_gpkg, layer=self.layer, driver="GPKG", engine="pyogrio")
            self.project.provenance.log(
                dataset=self.layer,
                source=self.source,
                identifier=self.layer,
                url=None,
                license=self.license,
                operation="write_layer",
                parameters={
                    "gpkg": self.project.data_gpkg.name,
                    "crs": self.project.config.crs,
                    "n_features": len(gdf),
                },
            )
        if self.limit_scenes and items:
            self.fetch(items)
        return len(items)

    # --- network (replaced in tests) ----------------------------------------------------

    def _request_json(self, method: str, url: str, body: dict | None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Accept": "application/geo+json, application/json", "User-Agent": USER_AGENT}
        if data:
            headers["Content-Type"] = "application/json"
        for attempt in range(RETRIES + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.load(response)
            except urllib.error.HTTPError as e:
                if e.code not in (502, 503, 504) or attempt == RETRIES:  # gateways time out now and then
                    raise
            time.sleep(RETRY_PAUSE_SECONDS * (attempt + 1))
        raise AssertionError("unreachable")

    def _download(self, url: str, part: Path, headers: dict[str, str]) -> None:
        """Stream ``url`` into ``part``; the single download call, replaced in tests."""
        opener = urllib.request.build_opener(_SameHostRedirects)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
        try:
            with opener.open(request, timeout=300) as response, part.open("wb") as f:
                final_host = urlparse(response.geturl()).netloc
                if re.match(r"^(sso|login|urs)\.", final_host):
                    raise SatelliteAuthError(
                        f"redirected to a login page ({final_host}); credentials missing or rejected. "
                        + self.manual_help()
                    )
                shutil.copyfileobj(response, f)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise SatelliteAuthError(f"HTTP {e.code} from {urlparse(url).netloc}; credentials rejected. " + self.manual_help()) from e
            raise OSError(f"HTTP {e.code} for {url}") from e


# --- file checks ------------------------------------------------------------------------

_TIFF = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
_NETCDF = (b"\x89HDF", b"CDF")


def _signature_problem(path: Path) -> str | None:
    """Why ``path`` does not look like what its extension says, or None if it does."""
    with path.open("rb") as f:
        head = f.read(512)
    if not head:
        return "the download is empty"
    suffix = path.name.removesuffix(".part").rsplit(".", 1)[-1].lower()
    if head.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        return "the server returned an HTML page instead of the file"
    if suffix in {"tif", "tiff"} and not head.startswith(_TIFF):
        return "the file is not a TIFF"
    if suffix in {"nc", "nc4"} and not head.startswith(_NETCDF):
        return "the file is not NetCDF/HDF5"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
