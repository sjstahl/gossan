"""Download ARIS assessment report PDFs for reports in the ``aris_index`` layer.

Access method: the public ARIS web application. Each index feature carries
``ARIS_REPORT_URL`` (``https://apps.nrs.gov.bc.ca/pub/aris/detail/<n>``); the report
PDF is served at ``https://apps.nrs.gov.bc.ca/pub/aris/Report/<n, 5 digits>.pdf/``
(checked against reports 724, 8038 and 42141; a report with no scanned PDF returns 404).
Reports can be tens of MB, so downloads are capped by ``limit`` and are polite
(one request at a time, with a pause between them).

Terms of use: as for the index, ARIS is **Access Only** (see ``aris.py``): no
reproduction or redistribution without written permission of the Province. The reports
are also copyright of their authors. Use them internally; do not put them in client
deliverables without permission.

Files go to ``raw/aris_pdfs/ARIS_<n>.pdf``. Existing files are never replaced or
re-downloaded.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pyogrio

from gossan.core.project import Project
from gossan.harvest.bc.aris import ArisIndexConnector

ARIS_BASE = "https://apps.nrs.gov.bc.ca/pub/aris"
DETAIL_URL = re.compile(r"^(?P<base>https?://[^?#]*?)/detail/(?P<number>\d+)/?$")
RAW_DIR = "aris_pdfs"
PAUSE_SECONDS = 2.0


@dataclass
class PdfResult:
    downloaded: list[str]
    skipped_existing: int
    failed: list[tuple[str, str]]  # (ARIS number, reason)
    remaining: int  # reports in the index still not downloaded


def pdf_url(report_url: str) -> str:
    """Report PDF URL for an ``ARIS_REPORT_URL`` (detail page) value."""
    m = DETAIL_URL.match(report_url.strip())
    if not m:
        raise ValueError(f"unrecognised ARIS report URL: {report_url!r}")
    return f"{m['base']}/Report/{int(m['number']):05d}.pdf/"


class ArisPdfDownloader:
    """Downloads up to ``limit`` not-yet-downloaded PDFs, newest reports first."""

    def __init__(self, project: Project, pause: float | None = None) -> None:
        self.project = project
        self.pause = PAUSE_SECONDS if pause is None else pause
        self.raw = project.raw_dir / RAW_DIR

    def run(self, limit: int) -> PdfResult:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        reports = self._reports()
        self.raw.mkdir(parents=True, exist_ok=True)
        result = PdfResult([], 0, [], 0)
        todo = []
        for number, url in reports:
            if self._dest(number).exists():
                result.skipped_existing += 1
            else:
                todo.append((number, url))
        result.remaining = max(len(todo) - limit, 0)

        for i, (number, url) in enumerate(todo[:limit]):
            if i and self.pause:
                time.sleep(self.pause)
            try:
                self._fetch_one(number, url)
            except (OSError, ValueError) as e:  # HTTP/network errors, or not a PDF
                result.failed.append((number, str(e)))
            else:
                result.downloaded.append(number)
        return result

    def _reports(self) -> list[tuple[str, str]]:
        """(ARIS number, PDF URL) from the index, newest report year first."""
        gpkg = self.project.data_gpkg
        layers = pyogrio.list_layers(gpkg)[:, 0] if gpkg.exists() else []
        if ArisIndexConnector.name not in layers:
            raise FileNotFoundError(
                "no aris_index layer in data.gpkg; run 'gossan harvest --source aris_index' first"
            )
        gdf = gpd.read_file(gpkg, layer=ArisIndexConnector.name, engine="pyogrio")
        gdf = gdf.dropna(subset=["ARIS_REPORT_URL", "ARIS_NUMBER"])
        gdf = gdf.sort_values(["REPORT_YEAR", "ARIS_NUMBER"], ascending=False)
        out = []
        for url, number in zip(gdf["ARIS_REPORT_URL"], gdf["ARIS_NUMBER"], strict=True):
            out.append((str(int(number)), pdf_url(url)))
        return out

    def _dest(self, number: str) -> Path:
        return self.raw / f"ARIS_{int(number):05d}.pdf"

    def _fetch_one(self, number: str, url: str) -> None:
        dest = self._dest(number)
        part = dest.with_name(dest.name + ".part")  # only complete, checked files get the final name
        retrieved_at = datetime.now(UTC)
        try:
            self._download(url, part)
            with part.open("rb") as f:
                if f.read(5) != b"%PDF-":
                    raise ValueError(f"ARIS {number}: response is not a PDF")
            sha256 = hashlib.sha256(part.read_bytes()).hexdigest()
            part.rename(dest)  # dest was checked absent above; never replaces an original
        finally:
            part.unlink(missing_ok=True)
        self.project.provenance.log(
            dataset="aris_pdfs",
            source=ArisIndexConnector.source,
            identifier=f"ARIS {number}",
            url=url,
            retrieved_at=retrieved_at,
            license=ArisIndexConnector.license,
            operation="fetch_pdf",
            parameters={
                "file": f"raw/{RAW_DIR}/{dest.name}",
                "bytes": dest.stat().st_size,
                "sha256": sha256,
            },
        )

    def _download(self, url: str, part: Path) -> None:
        """Stream ``url`` into ``part``; the single network call, replaced in tests."""
        request = urllib.request.Request(url, headers={"User-Agent": "gossan"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as f:
                shutil.copyfileobj(response, f)
        except urllib.error.HTTPError as e:
            raise OSError(f"HTTP {e.code} for {url}") from e
