"""Nevada NDEP Bureau of Mining Regulation and Reclamation (BMRR) notices: a drilling indicator for Nevada.

A Reclamation Permit for an exploration project (a "drilling project") is a strong sign that drilling is
planned or authorised on the ground it covers. NDEP publishes each decision and each proposed action as
a notice with a PDF. This connector reads them into the ``drilling_notices`` layer, the same layer the
BC Notice of Work connector writes, so lead scoring reads one thing.

Where the notices come from, and why this is best effort:

* The public notice list, ``https://ndep.nv.gov/posts/category/land`` (HTML, five items a page), is a
  short rolling list that mixes water, air and mining notices. Items titled "BMRR" that concern a
  reclamation permit are followed to their post page, whose PDF links are downloaded and read.
  Nothing on the site lists *all* exploration permits, and old notices drop off, so coverage depends
  on how often you harvest: each run keeps what it saw, and the layer accumulates across runs.
* NDEP offers an email list ("Get Notices" at the bottom of ndep.nv.gov/posts, option "Mining Regulation
  and Reclamation Public Notice"). Save those PDFs and load them with ``gossan ingest-notices``.
* The public records search (ndep-onbase.nv.gov) is interactive and is not crawled.

Each notice gives sections, township and range (PLSS). The geometry stored is the union of those
sections (about one square mile each, from the BLM national PLSS service), not the surveyed
disturbance, so overlap with a claim means "the permit's project area touches this claim".
Only exploration projects drive the lead flag; mine permits are kept in the layer for context.

Access: plain HTTP GET with the user agent ``gossan`` and a pause between requests. ndep.nv.gov serves no
robots.txt (404). No licence is stated on the site; these are State of Nevada public notices.
Confirm the terms with NDEP before redistributing them in client deliverables.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
import pymupdf
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from gossan.harvest.base import SearchItem
from gossan.harvest.us_state.ndep_parse import NdepNotice, parse_notice
from gossan.harvest.us_state.plss import LICENSE as PLSS_LICENSE
from gossan.harvest.us_state.plss import Township, parse_sections, section_query_url
from gossan.harvest.vector_service import Download, VectorServiceConnector

BASE_URL = "https://ndep.nv.gov"
LISTING_URL = f"{BASE_URL}/posts/category/land"
PAGE_STEP = 5  # the list is paged as /P5, /P10 ...
MAX_PAGES = 20
PAUSE_SECONDS = 1.0
MAX_PDFS_PER_POST = 4
LICENSE = "Nevada public notices (NDEP BMRR); no licence stated by the publisher, confirm terms before redistribution"
ARTICLE = re.compile(r'<article class="row postList-post">(.*?)</article>', re.DOTALL)


@dataclass(frozen=True)
class ListingItem:
    title: str
    description: str
    path: str


def _text(fragment: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def parse_listing(page: str) -> list[ListingItem]:
    """The posts on one list page."""
    items = []
    for article in ARTICLE.findall(page):
        title = re.search(r'summary-title">(.*?)</h3>', article, re.DOTALL)
        desc = re.search(r'summary-desc">(.*?)</p>', article, re.DOTALL)
        link = re.search(r'href="(/posts/[^"]+)"', article)
        if title and link:
            items.append(ListingItem(_text(title.group(1)), _text(desc.group(1)) if desc else "", link.group(1)))
    return items


def wanted(item: ListingItem) -> bool:
    """A BMRR notice about a reclamation permit (not a water pollution control or air permit)."""
    text = f"{item.title} {item.description}".lower()
    if "bmrr" not in text:
        return False
    if "water pollution control" in text and "reclamation" not in text:
        return False
    return bool(re.search(r"final decision|proposed action|nopa|notice of decision", text))


def pdf_links(page: str) -> list[str]:
    seen: list[str] = []
    for href in re.findall(r'href="([^"]+\.pdf)"', page, re.IGNORECASE):
        url = urljoin(BASE_URL, html.unescape(href))
        if url not in seen:
            seen.append(url)
    return seen


def pdf_text(content: bytes) -> str:
    with pymupdf.open(stream=content, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


class NdepNoticesConnector(VectorServiceConnector):
    """NDEP reclamation permit notices, written to the ``drilling_notices`` layer (Nevada)."""

    name = "ndep_notices"
    canonical_layer = "drilling_notices"
    source = "Nevada Division of Environmental Protection, Bureau of Mining Regulation and Reclamation (public notices)"
    license = LICENSE
    jurisdiction = "us"
    default = False  # Nevada only, and best effort: ask for it with --source ndep_notices
    output_description = "NDEP notice area(s) written to data.gpkg"

    # --- finding the notices ------------------------------------------------------------

    def search(self, aoi: gpd.GeoDataFrame) -> list[SearchItem]:
        return [SearchItem(identifier="ndep_bmrr_public_notices", url=LISTING_URL)]

    def download(self, item: SearchItem) -> Download:
        features: list[dict] = []
        skipped: list[str] = []
        pdfs: list[str] = []
        seen_posts: set[str] = set()
        for page_no in range(MAX_PAGES):
            url = LISTING_URL if page_no == 0 else f"{LISTING_URL}/P{page_no * PAGE_STEP}"
            items = parse_listing(self._polite_get(url).decode("utf-8", errors="replace"))
            if not items:
                break
            for post in filter(wanted, items):
                if post.path in seen_posts:
                    continue
                seen_posts.add(post.path)
                page = self._polite_get(urljoin(BASE_URL, post.path)).decode("utf-8", errors="replace")
                for pdf_url in pdf_links(page)[:MAX_PDFS_PER_POST]:
                    feature, why = self._notice_from_url(pdf_url)
                    if feature is not None:
                        features.append(feature)
                        pdfs.append(pdf_url)
                    elif why:
                        skipped.append(f"{pdf_url}: {why}")
        return self._download_result(features, pdfs, skipped, {"list_url": LISTING_URL, "posts_followed": len(seen_posts)})

    def _polite_get(self, url: str) -> bytes:
        time.sleep(PAUSE_SECONDS)
        return self._get(url)

    def _notice_from_url(self, url: str) -> tuple[dict | None, str]:
        content = self._polite_get(url)
        if not content.startswith(b"%PDF"):
            return None, "not a PDF"
        saved = self._save_pdf(url, content)
        return self._notice_from_pdf(content, saved, url)

    # --- one notice -> one feature ------------------------------------------------------

    def _notice_from_pdf(self, content: bytes, saved: str, url: str | None) -> tuple[dict | None, str]:
        notice = parse_notice(pdf_text(content))
        if notice is None:
            return None, "not a reclamation permit notice"
        if not notice.townships:
            return None, f"permit {notice.permit_no}: no section, township and range found"
        geoms = []
        for township in notice.townships:
            geoms += [shape(g) for g in parse_sections(self._polite_get(section_query_url(township)))]
        if not geoms:
            return None, f"permit {notice.permit_no}: the PLSS service found no matching sections"
        return {
            "type": "Feature",
            "geometry": mapping(unary_union(geoms)),
            "properties": _properties(notice, url, saved, hashlib.sha256(content).hexdigest(), township_labels(notice.townships)),
        }, ""

    def _save_pdf(self, name_or_url: str, content: bytes) -> str:
        """Keep the original PDF under raw/ (never overwriting) and log its retrieval."""
        folder = self.project.raw_dir / self.name / "pdfs"
        folder.mkdir(parents=True, exist_ok=True)
        stem = Path(name_or_url.split("?")[0]).name or "notice.pdf"
        path = folder / stem
        digest = hashlib.sha256(content).hexdigest()
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() == digest:
                return f"raw/{self.name}/pdfs/{path.name}"
            path = folder / f"{path.stem}_{datetime.now(UTC):%Y%m%dT%H%M%S%f}{path.suffix}"
        with path.open("xb") as f:
            f.write(content)
        self.project.provenance.log(
            dataset="ndep_notices_pdf",
            source=self.source,
            identifier=stem,
            url=name_or_url if name_or_url.startswith("http") else None,
            license=self.license,
            operation="fetch" if name_or_url.startswith("http") else "manual_ingest",
            parameters={"file": f"raw/{self.name}/pdfs/{path.name}", "sha256": digest},
        )
        return f"raw/{self.name}/pdfs/{path.name}"

    def _download_result(self, features: list[dict], pdfs: list[str], skipped: list[str], extra: dict) -> Download:
        content = json.dumps({"type": "FeatureCollection", "features": features}).encode("utf-8")
        params = {**extra, "n_pdfs_read": len(pdfs), "pdfs": pdfs, "skipped": skipped[:20], "n_skipped": len(skipped),
                  "plss_source": "BLM National PLSS CadNSDI, PLSS Section", "plss_license": PLSS_LICENSE}  # fmt: skip
        return Download(content, "geojson", len(features), params)

    # --- manual fallback ----------------------------------------------------------------

    def ingest_pdfs(self, paths: list[Path]) -> tuple[int, list[str]]:
        """Read notice PDFs you saved yourself (for example from NDEP's email list) into the layer.

        Returns (features written to the layer, messages about files that were skipped).
        """
        features, pdfs, skipped = [], [], []
        for path in paths:
            content = path.read_bytes()
            if not content.startswith(b"%PDF"):
                skipped.append(f"{path.name}: not a PDF")
                continue
            feature, why = self._notice_from_pdf(content, self._save_pdf(path.name, content), None)
            if feature is None:
                skipped.append(f"{path.name}: {why}")
            else:
                features.append(feature)
                pdfs.append(path.name)
        if features:
            raw = self.project.raw_dir / self.name
            raw.mkdir(parents=True, exist_ok=True)
            result = self._download_result(features, pdfs, skipped, {"manual": True})
            retrieved_at = datetime.now(UTC)
            target = raw / f"{self.name}_{retrieved_at:%Y%m%dT%H%M%S%fZ}.geojson"
            with target.open("xb") as f:
                f.write(result.content)
            self.project.provenance.log(
                dataset=self.layer, source=self.source, identifier="manual notice PDFs", url=None,
                retrieved_at=retrieved_at, license=self.license, operation="manual_ingest",
                parameters={**result.parameters, "n_features": len(features), "file": f"raw/{self.name}/{target.name}"},
            )  # fmt: skip
        return self._write_layer(), skipped

    def _write_layer(self) -> int:
        files = sorted((self.project.raw_dir / self.name).glob(f"{self.name}_*.geojson"))
        gdf = self.load(files)
        if not gdf.empty:
            gdf.to_file(self.project.data_gpkg, layer=self.layer, driver="GPKG", engine="pyogrio")
        self.project.provenance.log(
            dataset=self.layer, source=self.source, identifier=self.layer, url=None, license=self.license,
            operation="write_layer",
            parameters={"gpkg": self.project.data_gpkg.name, "crs": self.project.config.crs, "n_features": len(gdf), "inputs": [f.name for f in files]},
        )  # fmt: skip
        return len(gdf)

    # --- raw -> layer -------------------------------------------------------------------

    def read_raw(self, path: Path) -> gpd.GeoDataFrame | None:
        if not json.loads(path.read_text(encoding="utf-8"))["features"]:
            return None  # an empty collection has no schema for the reader to infer
        return gpd.read_file(path, engine="pyogrio")

    def load(self, files: list[Path]) -> gpd.GeoDataFrame:
        """Notices accumulate: read every raw file of this source, not only the one just fetched."""
        everything = sorted(set(files) | set((self.project.raw_dir / self.name).glob(f"{self.name}_*.geojson")))
        return super().load(everything)

    def normalize(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """One row per permit: a decision replaces its proposed action; the latest posting wins."""
        gdf = gdf.assign(_decided=(gdf["NOTICE_KIND"] == "decision"), _posted=gdf["NOW_APPLICATION_SUBMITTED_DATE"])
        gdf = gdf.sort_values(["_decided", "_posted"], ascending=False).drop_duplicates("PERMIT_NO")
        return gdf.drop(columns=["_decided", "_posted"]).sort_values("PERMIT_NO").reset_index(drop=True)


def township_labels(townships: list[Township]) -> str:
    return "; ".join(
        f"T{t.township}{t.ns} R{t.range}{t.ew} sec {','.join(str(s) for s in t.sections)}" for t in townships
    )


def _properties(notice: NdepNotice, url: str | None, saved: str, sha256: str, plss: str) -> dict:
    """The notice in the column names lead scoring reads (the BC Notice of Work names), plus NDEP extras."""
    posted = notice.posted.isoformat() if isinstance(notice.posted, date) else None
    decided = notice.kind == "decision"
    return {
        "NOW_NUMBER": f"NDEP {notice.permit_no}",
        "NOW_APPLICATION_STATUS_DESC": "Approved" if decided else "Received",
        "PERMIT_STATUS_CODE_DESC": None,
        "AMENDMENT_AUTH_END_DATE": None,
        "NOW_APPLICATION_SUBMITTED_DATE": posted,
        "TENURE_NUMBER": None,
        "PERMITTEE_NAME": notice.permittee or None,
        "MINE_NAME": notice.project or None,
        "NOTICE_SOURCE": "NDEP BMRR",
        "NOTICE_KIND": notice.kind,
        "DECISION_DATE": posted if decided else None,
        "PERMIT_NO": notice.permit_no,
        "PROJECT_TYPE": notice.project_type,
        "COUNTY": notice.county,
        "ACRES": notice.acres,
        "PLSS": plss,
        "PDF_URL": url,
        "PDF_FILE": saved,
        "PDF_SHA256": sha256,
    }
